// deepxde_callbacks.h - Callback system for DeepXDE C++ port
// Ports: Callback, CallbackList, EarlyStopping, ModelCheckpoint, MovieDumper,
//        DropoutUncertainty, PDEPointResampler, OperatorPredictor, Timer, VariableValue
#ifndef DEEPXDE_CALLBACKS_H
#define DEEPXDE_CALLBACKS_H

#include "deepxde.h"
#include <chrono>
#include <fstream>
#include <limits>

namespace deepxde {

// Forward declaration
class Model;

// ============================================================================
// Callback base class
// ============================================================================
struct Callback {
    Model* model = nullptr;
    virtual ~Callback() = default;
    virtual void set_model(Model* m) { model = m; init(); }
    virtual void init() {}
    virtual void on_epoch_begin() {}
    virtual void on_epoch_end() {}
    virtual void on_batch_begin() {}
    virtual void on_batch_end() {}
    virtual void on_train_begin() {}
    virtual void on_train_end() {}
    virtual void on_predict_begin() {}
    virtual void on_predict_end() {}
};

// ============================================================================
// CallbackList - Container for callbacks
// ============================================================================
struct CallbackList : Callback {
    std::vector<std::shared_ptr<Callback>> callbacks;

    void set_model(Model* m) override {
        model = m;
        for (auto& cb : callbacks) cb->set_model(m);
    }
    void on_epoch_begin() override { for (auto& cb : callbacks) cb->on_epoch_begin(); }
    void on_epoch_end() override { for (auto& cb : callbacks) cb->on_epoch_end(); }
    void on_batch_begin() override { for (auto& cb : callbacks) cb->on_batch_begin(); }
    void on_batch_end() override { for (auto& cb : callbacks) cb->on_batch_end(); }
    void on_train_begin() override { for (auto& cb : callbacks) cb->on_train_begin(); }
    void on_train_end() override { for (auto& cb : callbacks) cb->on_train_end(); }
    void on_predict_begin() override { for (auto& cb : callbacks) cb->on_predict_begin(); }
    void on_predict_end() override { for (auto& cb : callbacks) cb->on_predict_end(); }

    void add(std::shared_ptr<Callback> cb) { callbacks.push_back(cb); }
};

// ============================================================================
// EarlyStopping
// ============================================================================
struct EarlyStopping : Callback {
    double min_delta;
    int patience;
    double baseline;
    std::string monitor;
    int start_from_epoch;

    int wait = 0;
    int stopped_epoch = 0;
    double best;
    bool use_baseline;
    // Reference to model's stop_training flag
    bool* stop_training_ptr = nullptr;

    // Track losses externally
    std::function<double()> get_loss;

    EarlyStopping(double md = 0, int pat = 0, double bl = std::numeric_limits<double>::infinity(),
                  const std::string& mon = "train_loss", int start = 0)
        : min_delta(md), patience(pat), baseline(bl), monitor(mon),
          start_from_epoch(start), best(bl), use_baseline(bl != std::numeric_limits<double>::infinity()) {}

    void set_loss_fn(std::function<double()> fn) { get_loss = std::move(fn); }
    void set_stop_ptr(bool* p) { stop_training_ptr = p; }

    void on_train_begin() override {
        wait = 0;
        stopped_epoch = 0;
        if (!use_baseline) best = std::numeric_limits<double>::infinity();
    }

    void on_epoch_end() override {
        if (!get_loss) return;
        double current = get_loss();
        if (current - min_delta < best) {
            best = current;
            wait = 0;
        } else {
            wait++;
            if (wait >= patience && stop_training_ptr) {
                *stop_training_ptr = true;
                stopped_epoch = wait;
            }
        }
    }

    void on_train_end() override {
        if (stopped_epoch > 0)
            std::cout << "Early stopping triggered at epoch " << stopped_epoch << "\n";
    }
};

// ============================================================================
// ModelCheckpoint
// ============================================================================
struct ModelCheckpoint : Callback {
    std::string filepath;
    int verbose;
    bool save_better_only;
    int period;
    std::string monitor;
    double best = std::numeric_limits<double>::infinity();
    int epochs_since_last = 0;

    std::function<double()> get_loss;
    std::function<void(const std::string&)> save_fn;

    ModelCheckpoint(const std::string& fp, int verb = 0, bool sbo = false,
                    int per = 1, const std::string& mon = "train_loss")
        : filepath(fp), verbose(verb), save_better_only(sbo), period(per), monitor(mon) {}

    void on_epoch_end() override {
        epochs_since_last++;
        if (epochs_since_last < period) return;
        epochs_since_last = 0;
        if (save_better_only && get_loss) {
            double current = get_loss();
            if (current < best) {
                best = current;
                if (save_fn) save_fn(filepath);
                if (verbose > 0)
                    std::cout << "Checkpoint: loss improved to " << current
                              << ", saving to " << filepath << "\n";
            }
        } else if (save_fn) {
            save_fn(filepath);
        }
    }
};

// ============================================================================
// Timer - Stop training after time limit
// ============================================================================
struct TimerCallback : Callback {
    double available_seconds;
    std::chrono::steady_clock::time_point start_time;
    bool started = false;
    bool* stop_training_ptr = nullptr;

    TimerCallback(double minutes) : available_seconds(minutes * 60.0) {}

    void set_stop_ptr(bool* p) { stop_training_ptr = p; }

    void on_train_begin() override {
        if (!started) { start_time = std::chrono::steady_clock::now(); started = true; }
    }

    void on_epoch_end() override {
        if (!started) return;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        if (elapsed > available_seconds && stop_training_ptr) {
            *stop_training_ptr = true;
            std::cout << "Timer: stopping after " << elapsed / 60.0 << " minutes\n";
        }
    }
};

// ============================================================================
// PDEPointResampler
// ============================================================================
struct PDEPointResampler : Callback {
    int period;
    bool pde_points;
    bool bc_points;
    int epochs_since_last = 0;

    std::function<void()> resample_fn;

    PDEPointResampler(int per = 100, bool pde = true, bool bc = false)
        : period(per), pde_points(pde), bc_points(bc) {}

    void on_epoch_end() override {
        epochs_since_last++;
        if (epochs_since_last >= period) {
            epochs_since_last = 0;
            if (resample_fn) resample_fn();
        }
    }
};

// ============================================================================
// DropoutUncertainty - MC Dropout uncertainty estimation
// ============================================================================
struct DropoutUncertainty : Callback {
    int period;
    int n_samples;
    int epochs_since_last = 0;
    std::vector<double> y_std;

    std::function<Matrix(const Matrix&)> predict_fn;
    Matrix test_x;

    DropoutUncertainty(int per = 1000, int ns = 1000)
        : period(per), n_samples(ns) {}

    void on_epoch_end() override {
        epochs_since_last++;
        if (epochs_since_last >= period && predict_fn && test_x.rows > 0) {
            epochs_since_last = 0;
            // Collect predictions
            std::vector<Matrix> preds;
            for (int s = 0; s < n_samples; ++s)
                preds.push_back(predict_fn(test_x));
            // Compute std dev
            y_std.assign(test_x.rows, 0.0);
            for (int i = 0; i < test_x.rows; ++i) {
                double mean = 0, var = 0;
                for (int s = 0; s < n_samples; ++s) mean += preds[s](i, 0);
                mean /= n_samples;
                for (int s = 0; s < n_samples; ++s) {
                    double d = preds[s](i, 0) - mean;
                    var += d * d;
                }
                y_std[i] = std::sqrt(var / n_samples);
            }
        }
    }
};

// ============================================================================
// MovieDumper - Save training progress for animation
// ============================================================================
struct MovieDumper : Callback {
    std::string filename;
    Matrix x_line;
    int period;
    int component;
    std::vector<std::vector<double>> y_history;
    int epochs_since_last = 0;

    std::function<std::vector<double>(const Matrix&, int)> predict_component;

    MovieDumper(const std::string& fn, const std::vector<double>& x1,
                const std::vector<double>& x2, int num_points = 100,
                int per = 1, int comp = 0)
        : filename(fn), period(per), component(comp)
    {
        x_line = Matrix(num_points, static_cast<int>(x1.size()));
        for (int i = 0; i < num_points; ++i)
            for (size_t d = 0; d < x1.size(); ++d)
                x_line(i, static_cast<int>(d)) = x1[d] + (x2[d] - x1[d]) * i / (num_points - 1);
    }

    void on_epoch_end() override {
        epochs_since_last++;
        if (epochs_since_last >= period && predict_component) {
            epochs_since_last = 0;
            y_history.push_back(predict_component(x_line, component));
        }
    }

    void on_train_end() override {
        // Save to files
        std::ofstream fx(filename + "_x.txt");
        for (int i = 0; i < x_line.rows; ++i) {
            for (int d = 0; d < x_line.cols; ++d) {
                if (d > 0) fx << " ";
                fx << x_line(i, d);
            }
            fx << "\n";
        }
        std::ofstream fy(filename + "_y.txt");
        for (auto& y : y_history) {
            for (size_t i = 0; i < y.size(); ++i) {
                if (i > 0) fy << " ";
                fy << y[i];
            }
            fy << "\n";
        }
    }
};

// ============================================================================
// OperatorPredictor - Evaluate an operator during training
// ============================================================================
struct OperatorPredictor : Callback {
    int period;
    int epochs_since_last = 0;
    std::vector<double> value;

    std::function<std::vector<double>()> eval_fn;

    OperatorPredictor(int per = 1) : period(per) {}

    void on_epoch_end() override {
        epochs_since_last++;
        if (epochs_since_last >= period && eval_fn) {
            epochs_since_last = 0;
            value = eval_fn();
        }
    }
};

// ============================================================================
// VariableValue - Track trainable variable values
// ============================================================================
struct VariableValue : Callback {
    std::vector<double*> var_ptrs;
    int period;
    std::vector<double> values;
    int epochs_since_last = 0;
    std::ofstream file;
    bool use_file;

    VariableValue(const std::vector<double*>& vars, int per = 1,
                  const std::string& fname = "")
        : var_ptrs(vars), period(per), use_file(!fname.empty())
    {
        if (use_file) file.open(fname);
    }

    void on_epoch_end() override {
        epochs_since_last++;
        if (epochs_since_last >= period) {
            epochs_since_last = 0;
            values.clear();
            for (auto* p : var_ptrs) values.push_back(*p);
            if (use_file && file.is_open()) {
                for (size_t i = 0; i < values.size(); ++i) {
                    if (i > 0) file << " ";
                    file << values[i];
                }
                file << "\n";
                file.flush();
            }
        }
    }
};

} // namespace deepxde
#endif // DEEPXDE_CALLBACKS_H
