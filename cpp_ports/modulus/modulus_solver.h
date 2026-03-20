// modulus_solver.h - Full solver with training loop, validation, checkpointing
// Port of Modulus solver/training infrastructure
//
// C++17, no external dependencies.

#ifndef MODULUS_SOLVER_H
#define MODULUS_SOLVER_H

#include "modulus.h"
#include "modulus_loss.h"
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace modulus {
namespace solver {

// ============================================================
// Learning Rate Schedulers
// ============================================================
class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual double get_lr(int step, double base_lr) const = 0;
};

class ConstantLR : public LRScheduler {
public:
    double get_lr(int /*step*/, double base_lr) const override { return base_lr; }
};

class StepLR : public LRScheduler {
    int step_size;
    double gamma;
public:
    StepLR(int step_sz = 100, double g = 0.5) : step_size(step_sz), gamma(g) {}
    double get_lr(int step, double base_lr) const override {
        return base_lr * std::pow(gamma, step / step_size);
    }
};

class ExponentialLR : public LRScheduler {
    double gamma;
public:
    ExponentialLR(double g = 0.999) : gamma(g) {}
    double get_lr(int step, double base_lr) const override {
        return base_lr * std::pow(gamma, step);
    }
};

class CosineAnnealingLR : public LRScheduler {
    int T_max;
    double eta_min;
public:
    CosineAnnealingLR(int t_max = 1000, double min_lr = 1e-6)
        : T_max(t_max), eta_min(min_lr) {}
    double get_lr(int step, double base_lr) const override {
        return eta_min + 0.5 * (base_lr - eta_min) * (1.0 + std::cos(M_PI * step / T_max));
    }
};

class WarmupCosineAnnealingLR : public LRScheduler {
    int warmup_steps, total_steps;
    double min_lr;
public:
    WarmupCosineAnnealingLR(int warmup = 100, int total = 10000, double min_lr_ = 1e-6)
        : warmup_steps(warmup), total_steps(total), min_lr(min_lr_) {}
    double get_lr(int step, double base_lr) const override {
        if (step < warmup_steps)
            return base_lr * step / warmup_steps;
        int cosine_step = step - warmup_steps;
        int cosine_total = total_steps - warmup_steps;
        return min_lr + 0.5 * (base_lr - min_lr) *
               (1.0 + std::cos(M_PI * cosine_step / std::max(cosine_total, 1)));
    }
};

// ============================================================
// Gradient Clipping
// ============================================================
inline void clip_grad_norm(std::vector<Parameter*>& params, double max_norm) {
    double total_norm = 0;
    for (auto* p : params) total_norm += p->grad.norm_sq();
    total_norm = std::sqrt(total_norm);

    if (total_norm > max_norm) {
        double scale = max_norm / total_norm;
        for (auto* p : params)
            for (auto& g : p->grad.data) g *= scale;
    }
}

inline void clip_grad_value(std::vector<Parameter*>& params, double clip_value) {
    for (auto* p : params)
        for (auto& g : p->grad.data)
            g = std::clamp(g, -clip_value, clip_value);
}

// ============================================================
// Checkpoint: save/load model parameters
// ============================================================
namespace checkpoint {

// Save parameters to binary file
inline void save(const std::string& path, const std::vector<Parameter*>& params) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open checkpoint file: " + path);

    // Write number of parameters
    uint32_t n_params = (uint32_t)params.size();
    f.write(reinterpret_cast<const char*>(&n_params), 4);

    for (auto* p : params) {
        // Write name length and name
        uint32_t name_len = (uint32_t)p->name.size();
        f.write(reinterpret_cast<const char*>(&name_len), 4);
        f.write(p->name.data(), name_len);

        // Write shape
        uint32_t ndim = (uint32_t)p->value.shape.size();
        f.write(reinterpret_cast<const char*>(&ndim), 4);
        for (auto& s : p->value.shape) {
            int32_t dim = s;
            f.write(reinterpret_cast<const char*>(&dim), 4);
        }

        // Write data
        uint32_t n = (uint32_t)p->value.data.size();
        f.write(reinterpret_cast<const char*>(&n), 4);
        f.write(reinterpret_cast<const char*>(p->value.data.data()), n * sizeof(double));
    }
}

// Load parameters from binary file
inline void load(const std::string& path, std::vector<Parameter*>& params) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open checkpoint file: " + path);

    uint32_t n_params;
    f.read(reinterpret_cast<char*>(&n_params), 4);

    // Build name -> parameter map
    std::unordered_map<std::string, Parameter*> param_map;
    for (auto* p : params) param_map[p->name] = p;

    for (uint32_t i = 0; i < n_params; ++i) {
        uint32_t name_len;
        f.read(reinterpret_cast<char*>(&name_len), 4);
        std::string name(name_len, '\0');
        f.read(&name[0], name_len);

        uint32_t ndim;
        f.read(reinterpret_cast<char*>(&ndim), 4);
        std::vector<int> shape(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            int32_t dim;
            f.read(reinterpret_cast<char*>(&dim), 4);
            shape[d] = dim;
        }

        uint32_t n;
        f.read(reinterpret_cast<char*>(&n), 4);
        std::vector<double> data(n);
        f.read(reinterpret_cast<char*>(data.data()), n * sizeof(double));

        // Load into matching parameter
        auto it = param_map.find(name);
        if (it != param_map.end()) {
            if (it->second->value.numel() == (int)n) {
                it->second->value.data = data;
            }
        }
    }
}

} // namespace checkpoint

// ============================================================
// Training metrics tracker
// ============================================================
class MetricsTracker {
public:
    struct Metrics {
        double train_loss;
        double val_loss;
        double learning_rate;
        double epoch_time;
    };

    std::vector<Metrics> history;
    double best_val_loss;
    int best_epoch;

    MetricsTracker() : best_val_loss(1e30), best_epoch(0) {}

    void record(int epoch, double train_loss, double val_loss, double lr, double time) {
        history.push_back({train_loss, val_loss, lr, time});
        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            best_epoch = epoch;
        }
    }

    void print_summary() const {
        std::cout << "\n  Training Summary:" << std::endl;
        std::cout << "    Total epochs: " << history.size() << std::endl;
        std::cout << "    Best val loss: " << std::scientific << best_val_loss
                  << " (epoch " << best_epoch << ")" << std::endl;
        if (!history.empty()) {
            double total_time = 0;
            for (auto& m : history) total_time += m.epoch_time;
            std::cout << "    Total time: " << std::fixed << std::setprecision(1)
                      << total_time << "s" << std::endl;
            std::cout << "    Avg epoch time: " << total_time / history.size() << "s" << std::endl;
        }
    }

    // Export to CSV
    void export_csv(const std::string& path) const {
        std::ofstream f(path);
        f << "epoch,train_loss,val_loss,lr,time\n";
        for (int i = 0; i < (int)history.size(); ++i) {
            auto& m = history[i];
            f << i << "," << std::scientific << m.train_loss << ","
              << m.val_loss << "," << m.learning_rate << ","
              << std::fixed << m.epoch_time << "\n";
        }
    }
};

// ============================================================
// Solver - Full training pipeline
// ============================================================
class Solver {
public:
    // Configuration
    struct Config {
        double learning_rate = 1e-3;
        int epochs = 100;
        int print_every = 10;
        int checkpoint_every = 0;     // 0 = no checkpointing
        std::string checkpoint_dir = ".";
        double grad_clip_norm = 0.0;  // 0 = no clipping
        double grad_clip_value = 0.0;
        bool early_stopping = false;
        int patience = 20;
        double min_delta = 1e-7;
    };

    Config config;
    Adam optimizer;
    std::unique_ptr<LRScheduler> scheduler;
    MetricsTracker metrics;

    Solver() {
        optimizer = Adam(config.learning_rate);
        scheduler = std::make_unique<ConstantLR>();
    }

    explicit Solver(const Config& cfg) : config(cfg) {
        optimizer = Adam(cfg.learning_rate);
        scheduler = std::make_unique<ConstantLR>();
    }

    void set_scheduler(std::unique_ptr<LRScheduler> sched) {
        scheduler = std::move(sched);
    }

    // Generic training loop
    using ForwardFn = std::function<Tensor(const Tensor&)>;
    using BackwardFn = std::function<void(const Tensor&)>;
    using ZeroGradFn = std::function<void()>;

    void train(std::vector<Parameter*>& params,
               const std::vector<Tensor>& train_inputs,
               const std::vector<Tensor>& train_targets,
               const std::vector<Tensor>& val_inputs,
               const std::vector<Tensor>& val_targets,
               ForwardFn forward_fn,
               BackwardFn backward_fn,
               ZeroGradFn zero_grad_fn,
               const std::string& loss_type = "mse") {

        int n_train = (int)train_inputs.size();
        int n_val = (int)val_inputs.size();
        int patience_counter = 0;

        auto get_loss_fn = [&](const std::string& type) -> std::function<double(const Tensor&, const Tensor&)> {
            if (type == "mse") return loss::mse_loss;
            if (type == "l1") return loss::l1_loss;
            if (type == "huber") return [](const Tensor& a, const Tensor& b) { return loss::huber_loss(a, b); };
            if (type == "relative_l2") return loss::relative_l2_loss;
            return loss::mse_loss;
        };

        auto loss_fn = get_loss_fn(loss_type);

        for (int epoch = 0; epoch < config.epochs; ++epoch) {
            auto t_start = std::chrono::high_resolution_clock::now();

            // Update learning rate
            double lr = scheduler->get_lr(epoch, config.learning_rate);
            optimizer.lr = lr;

            // Training
            double train_loss = 0;
            for (int s = 0; s < n_train; ++s) {
                zero_grad_fn();
                Tensor pred = forward_fn(train_inputs[s]);
                double sample_loss = loss_fn(pred, train_targets[s]);
                train_loss += sample_loss;

                Tensor grad = loss::mse_grad(pred, train_targets[s]);
                backward_fn(grad);

                if (config.grad_clip_norm > 0)
                    clip_grad_norm(params, config.grad_clip_norm);
                if (config.grad_clip_value > 0)
                    clip_grad_value(params, config.grad_clip_value);

                optimizer.step(params);
            }
            train_loss /= std::max(n_train, 1);

            // Validation
            double val_loss = 0;
            for (int s = 0; s < n_val; ++s) {
                Tensor pred = forward_fn(val_inputs[s]);
                val_loss += loss_fn(pred, val_targets[s]);
            }
            val_loss /= std::max(n_val, 1);

            auto t_end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_end - t_start).count();

            metrics.record(epoch, train_loss, val_loss, lr, elapsed);

            // Print progress
            if ((epoch + 1) % config.print_every == 0 || epoch == 0) {
                std::cout << "  Epoch " << std::setw(4) << epoch + 1
                          << "/" << config.epochs
                          << "  train=" << std::scientific << std::setprecision(4) << train_loss
                          << "  val=" << val_loss
                          << "  lr=" << std::setprecision(2) << lr
                          << "  time=" << std::fixed << std::setprecision(2) << elapsed << "s"
                          << std::endl;
            }

            // Checkpointing
            if (config.checkpoint_every > 0 && (epoch + 1) % config.checkpoint_every == 0) {
                std::string path = config.checkpoint_dir + "/ckpt_epoch_" +
                                   std::to_string(epoch + 1) + ".bin";
                checkpoint::save(path, params);
            }

            // Early stopping
            if (config.early_stopping) {
                if (val_loss < metrics.best_val_loss - config.min_delta) {
                    patience_counter = 0;
                } else {
                    patience_counter++;
                    if (patience_counter >= config.patience) {
                        std::cout << "  Early stopping at epoch " << epoch + 1 << std::endl;
                        break;
                    }
                }
            }
        }
    }
};

// ============================================================
// Inferencer - Run trained model for inference
// ============================================================
class Inferencer {
public:
    using ForwardFn = std::function<Tensor(const Tensor&)>;

    // Run inference on a batch of inputs
    static std::vector<Tensor> run(ForwardFn forward_fn,
                                    const std::vector<Tensor>& inputs) {
        std::vector<Tensor> outputs;
        outputs.reserve(inputs.size());
        for (auto& inp : inputs)
            outputs.push_back(forward_fn(inp));
        return outputs;
    }

    // Compute error statistics
    struct ErrorStats {
        double mean_mse, max_mse, min_mse;
        double mean_relative_l2;
    };

    static ErrorStats compute_errors(ForwardFn forward_fn,
                                     const std::vector<Tensor>& inputs,
                                     const std::vector<Tensor>& targets) {
        ErrorStats stats = {0, 0, 1e30, 0};
        int n = (int)inputs.size();
        for (int i = 0; i < n; ++i) {
            Tensor pred = forward_fn(inputs[i]);
            double mse = loss::mse_loss(pred, targets[i]);
            double rel = loss::relative_l2_loss(pred, targets[i]);
            stats.mean_mse += mse;
            stats.max_mse = std::max(stats.max_mse, mse);
            stats.min_mse = std::min(stats.min_mse, mse);
            stats.mean_relative_l2 += rel;
        }
        stats.mean_mse /= n;
        stats.mean_relative_l2 /= n;
        return stats;
    }
};

// ============================================================
// Validator - Monitor model quality during/after training
// ============================================================
class Validator {
public:
    struct ValidationResult {
        double loss;
        double relative_error;
        int num_samples;
    };

    using ForwardFn = std::function<Tensor(const Tensor&)>;

    static ValidationResult validate(ForwardFn forward_fn,
                                     const std::vector<Tensor>& inputs,
                                     const std::vector<Tensor>& targets) {
        ValidationResult result = {0, 0, (int)inputs.size()};
        for (int i = 0; i < result.num_samples; ++i) {
            Tensor pred = forward_fn(inputs[i]);
            result.loss += loss::mse_loss(pred, targets[i]);
            result.relative_error += loss::relative_l2_loss(pred, targets[i]);
        }
        result.loss /= result.num_samples;
        result.relative_error /= result.num_samples;
        return result;
    }
};

// ============================================================
// Monitor - Track quantities during training
// ============================================================
class Monitor {
public:
    struct MonitorEntry {
        std::string name;
        std::function<double()> compute_fn;
        std::vector<double> values;
    };

    std::vector<MonitorEntry> monitors;

    void add(const std::string& name, std::function<double()> fn) {
        monitors.push_back({name, std::move(fn), {}});
    }

    void record() {
        for (auto& m : monitors) {
            m.values.push_back(m.compute_fn());
        }
    }

    void print_latest() const {
        for (auto& m : monitors) {
            if (!m.values.empty())
                std::cout << "  " << m.name << ": " << std::scientific
                          << m.values.back() << std::endl;
        }
    }

    void export_csv(const std::string& path) const {
        std::ofstream f(path);
        // Header
        for (int i = 0; i < (int)monitors.size(); ++i) {
            if (i) f << ",";
            f << monitors[i].name;
        }
        f << "\n";
        // Data
        int n_steps = 0;
        for (auto& m : monitors) n_steps = std::max(n_steps, (int)m.values.size());
        for (int s = 0; s < n_steps; ++s) {
            for (int i = 0; i < (int)monitors.size(); ++i) {
                if (i) f << ",";
                if (s < (int)monitors[i].values.size())
                    f << std::scientific << monitors[i].values[s];
            }
            f << "\n";
        }
    }
};

} // namespace solver
} // namespace modulus

#endif // MODULUS_SOLVER_H
