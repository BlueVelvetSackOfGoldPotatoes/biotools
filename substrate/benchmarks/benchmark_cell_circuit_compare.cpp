#include "core/data/dataloader.h"
#include "core/io/run_logger.h"
#include "core/metrics/metrics.h"
#include "models/cells/src/cell_circuit.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

int env_int(const char* key, int fallback) {
    const char* v = std::getenv(key);
    if (!v || !*v) return fallback;
    try { return std::stoi(v); } catch (...) { return fallback; }
}

double env_double(const char* key, double fallback) {
    const char* v = std::getenv(key);
    if (!v || !*v) return fallback;
    try { return std::stod(v); } catch (...) { return fallback; }
}

std::vector<double> row_to_vec(const Tensor& t, std::size_t row) {
    std::vector<double> out(t.cols, 0.0);
    for (std::size_t j = 0; j < t.cols; ++j) out[j] = t(row, j);
    return out;
}

double macro_precision_recall_f1(const Tensor& cm, double& p_out, double& r_out) {
    auto per = Metrics::per_class_metrics(cm);
    double p = 0.0, r = 0.0, f = 0.0;
    for (const auto& c : per) {
        p += c.precision;
        r += c.recall;
        f += c.f1;
    }
    const double d = per.empty() ? 1.0 : static_cast<double>(per.size());
    p_out = p / d;
    r_out = r / d;
    return f / d;
}

class FrozenLinearBaseline {
public:
    FrozenLinearBaseline(std::size_t input_dim,
                         std::size_t classes,
                         double lr,
                         double weight_decay,
                         unsigned int seed)
        : input_dim_(input_dim), classes_(classes), lr_(lr), wd_(weight_decay), rng_(seed) {
        std::normal_distribution<double> nd(0.0, 0.01);
        w_.resize(input_dim_ * classes_);
        b_.assign(classes_, 0.0);
        for (double& v : w_) v = nd(rng_);
    }

    struct StepInfo {
        int pred = 0;
        double loss = 0.0;
        double confidence = 0.0;
    };

    StepInfo train_one(const std::vector<double>& x, int label) {
        const std::vector<double> p = predict_proba(x);
        StepInfo s = make_step_info(p, label);

        for (std::size_t j = 0; j < classes_; ++j) {
            const double t = (static_cast<int>(j) == label) ? 1.0 : 0.0;
            const double delta = t - p[j];
            for (std::size_t i = 0; i < input_dim_; ++i) {
                const std::size_t idx = i * classes_ + j;
                w_[idx] += lr_ * delta * x[i] - wd_ * w_[idx];
            }
            b_[j] += lr_ * delta;
        }
        return s;
    }

    StepInfo evaluate_one(const std::vector<double>& x, int label) const {
        const std::vector<double> p = predict_proba(x);
        return make_step_info(p, label);
    }

private:
    std::vector<double> predict_logits(const std::vector<double>& x) const {
        std::vector<double> z(classes_, 0.0);
        for (std::size_t j = 0; j < classes_; ++j) z[j] = b_[j];
        for (std::size_t i = 0; i < input_dim_; ++i) {
            const double xi = (i < x.size()) ? x[i] : 0.0;
            const std::size_t row = i * classes_;
            for (std::size_t j = 0; j < classes_; ++j) {
                z[j] += xi * w_[row + j];
            }
        }
        return z;
    }

    static std::vector<double> softmax(const std::vector<double>& z) {
        if (z.empty()) return {};
        const double m = *std::max_element(z.begin(), z.end());
        std::vector<double> e(z.size(), 0.0);
        double s = 0.0;
        for (std::size_t i = 0; i < z.size(); ++i) {
            e[i] = std::exp(z[i] - m);
            s += e[i];
        }
        if (s <= 0.0) return std::vector<double>(z.size(), 1.0 / static_cast<double>(z.size()));
        for (double& v : e) v /= s;
        return e;
    }

    std::vector<double> predict_proba(const std::vector<double>& x) const {
        return softmax(predict_logits(x));
    }

    static StepInfo make_step_info(const std::vector<double>& p, int label) {
        StepInfo s;
        if (p.empty()) return s;
        s.pred = 0;
        s.confidence = p[0];
        for (std::size_t j = 1; j < p.size(); ++j) {
            if (p[j] > s.confidence) {
                s.confidence = p[j];
                s.pred = static_cast<int>(j);
            }
        }
        const int y = std::max(0, std::min(static_cast<int>(p.size() - 1), label));
        s.loss = -std::log(std::max(1e-12, p[static_cast<std::size_t>(y)]));
        return s;
    }

    std::size_t input_dim_;
    std::size_t classes_;
    double lr_;
    double wd_;
    std::mt19937 rng_;
    std::vector<double> w_;
    std::vector<double> b_;
};

} // namespace

int main() {
    std::cout << "=== Cell Circuit Fair Compare (No-Shortcut vs Frozen Linear) ===" << std::endl;

    const int epochs = env_int("CELL_COMPARE_EPOCHS", 12);
    const int train_limit = env_int("CELL_COMPARE_TRAIN_SAMPLES", 20000);
    const int test_limit = env_int("CELL_COMPARE_TEST_SAMPLES", 5000);
    const int seed = env_int("CELL_COMPARE_SEED", 42);
    const double linear_lr = env_double("CELL_COMPARE_LINEAR_LR", 0.05);
    const double linear_wd = env_double("CELL_COMPARE_LINEAR_WD", 1e-6);

    MNISTData train = MNISTLoader::load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    MNISTData test = MNISTLoader::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    const int n_train = std::min(train_limit, static_cast<int>(train.images.rows));
    const int n_test = std::min(test_limit, static_cast<int>(test.images.rows));

    cells::CellCircuitConfig cfg;
    cfg.input_dim = 784;
    cfg.retina_dim = static_cast<std::size_t>(env_int("CELL_COMPARE_RETINA", 196));
    cfg.v1_dim = static_cast<std::size_t>(env_int("CELL_COMPARE_V1", 256));
    cfg.assoc_dim = static_cast<std::size_t>(env_int("CELL_COMPARE_ASSOC", 128));
    cfg.decision_dim = 10;
    cfg.dt = env_double("CELL_COMPARE_DT", 0.02);
    cfg.lr = env_double("CELL_COMPARE_LR", 0.004);
    cfg.eligibility_decay = env_double("CELL_COMPARE_ELIGIBILITY_DECAY", 0.95);
    cfg.weight_decay = env_double("CELL_COMPARE_WEIGHT_DECAY", 1e-4);
    cfg.homeostasis_hmax = env_double("CELL_COMPARE_HMAX", 2.0);
    cfg.seed = static_cast<unsigned int>(std::max(0, seed));

    cells::CellCircuit circuit(cfg);
    FrozenLinearBaseline linear(cfg.input_dim, cfg.decision_dim, linear_lr, linear_wd, static_cast<unsigned int>(seed + 991));

    std::vector<std::vector<int>> orders(static_cast<std::size_t>(epochs), std::vector<int>(n_train));
    std::vector<int> base_indices(n_train);
    std::iota(base_indices.begin(), base_indices.end(), 0);
    std::mt19937 order_rng(static_cast<unsigned int>(seed));
    for (int ep = 0; ep < epochs; ++ep) {
        orders[static_cast<std::size_t>(ep)] = base_indices;
        std::shuffle(orders[static_cast<std::size_t>(ep)].begin(), orders[static_cast<std::size_t>(ep)].end(), order_rng);
    }

    std::ostringstream circuit_params_json;
    circuit_params_json << "{"
                        << "\"mode\":\"no_shortcut\","
                        << "\"epochs\":" << epochs << ","
                        << "\"train_limit\":" << n_train << ","
                        << "\"test_limit\":" << n_test << ","
                        << "\"retina\":" << cfg.retina_dim << ","
                        << "\"v1\":" << cfg.v1_dim << ","
                        << "\"assoc\":" << cfg.assoc_dim << ","
                        << "\"lr\":" << cfg.lr << ","
                        << "\"dt\":" << cfg.dt << ","
                        << "\"eligibility_decay\":" << cfg.eligibility_decay
                        << "}";

    std::ostringstream linear_params_json;
    linear_params_json << "{"
                       << "\"mode\":\"frozen_linear\","
                       << "\"epochs\":" << epochs << ","
                       << "\"train_limit\":" << n_train << ","
                       << "\"test_limit\":" << n_test << ","
                       << "\"linear_lr\":" << linear_lr << ","
                       << "\"linear_wd\":" << linear_wd
                       << "}";

    RunLogger circuit_logger("cells", "cell_circuit_no_shortcut_mnist", seed, "mnist-idx-v1",
                             circuit_params_json.str(), "mnist", "MNIST", "classification");
    RunLogger linear_logger("cells", "frozen_linear_mnist", seed, "mnist-idx-v1",
                            linear_params_json.str(), "mnist", "MNIST", "classification");

    std::filesystem::create_directories("output/cells");
    const std::string compare_csv =
        "output/cells/cell_compare_seed" + std::to_string(seed) + "_e" + std::to_string(epochs) + ".csv";
    std::ofstream cmp(compare_csv, std::ios::trunc);
    cmp << "epoch,circuit_train_acc,circuit_test_acc,linear_train_acc,linear_test_acc,test_acc_gap_linear_minus_circuit\n";

    std::cout << "[compare] no-shortcut params=" << circuit.parameter_count()
              << " linear params=" << (cfg.input_dim * cfg.decision_dim + cfg.decision_dim)
              << " train=" << n_train
              << " test=" << n_test
              << std::endl;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto t0 = std::chrono::high_resolution_clock::now();

        const std::vector<int>& order = orders[static_cast<std::size_t>(epoch - 1)];

        // Train no-shortcut cell circuit.
        circuit.reset_state();
        double circuit_train_loss = 0.0;
        int circuit_train_correct = 0;
        for (int i = 0; i < n_train; ++i) {
            const int idx = order[static_cast<std::size_t>(i)];
            std::vector<double> x = row_to_vec(train.images, static_cast<std::size_t>(idx));
            const int y = static_cast<int>(train.labels[static_cast<std::size_t>(idx)]);
            const auto step = circuit.train_one(x, y);
            circuit_train_loss += step.loss;
            if (step.pred == y) circuit_train_correct++;
        }
        const double circuit_train_acc = static_cast<double>(circuit_train_correct) / std::max(1, n_train);
        circuit_train_loss /= std::max(1, n_train);

        // Train frozen linear baseline with same order.
        double linear_train_loss = 0.0;
        int linear_train_correct = 0;
        for (int i = 0; i < n_train; ++i) {
            const int idx = order[static_cast<std::size_t>(i)];
            std::vector<double> x = row_to_vec(train.images, static_cast<std::size_t>(idx));
            const int y = static_cast<int>(train.labels[static_cast<std::size_t>(idx)]);
            const auto step = linear.train_one(x, y);
            linear_train_loss += step.loss;
            if (step.pred == y) linear_train_correct++;
        }
        const double linear_train_acc = static_cast<double>(linear_train_correct) / std::max(1, n_train);
        linear_train_loss /= std::max(1, n_train);

        // Evaluate both models.
        circuit.reset_state();
        std::vector<int> circuit_preds;
        std::vector<int> linear_preds;
        std::vector<int> truths;
        circuit_preds.reserve(n_test);
        linear_preds.reserve(n_test);
        truths.reserve(n_test);
        double circuit_test_loss = 0.0;
        double linear_test_loss = 0.0;

        for (int i = 0; i < n_test; ++i) {
            std::vector<double> x = row_to_vec(test.images, static_cast<std::size_t>(i));
            const int y = static_cast<int>(test.labels[static_cast<std::size_t>(i)]);

            const auto cs = circuit.evaluate_one(x, y);
            const auto ls = linear.evaluate_one(x, y);
            circuit_test_loss += cs.loss;
            linear_test_loss += ls.loss;
            circuit_preds.push_back(cs.pred);
            linear_preds.push_back(ls.pred);
            truths.push_back(y);
        }

        circuit_test_loss /= std::max(1, n_test);
        linear_test_loss /= std::max(1, n_test);
        const double circuit_test_acc = Metrics::accuracy(circuit_preds, truths);
        const double linear_test_acc = Metrics::accuracy(linear_preds, truths);

        const Tensor circuit_cm = Metrics::confusion_matrix(circuit_preds, truths, static_cast<int>(cfg.decision_dim));
        const Tensor linear_cm = Metrics::confusion_matrix(linear_preds, truths, static_cast<int>(cfg.decision_dim));
        double cp = 0.0, cr = 0.0, lp = 0.0, lr = 0.0;
        const double circuit_f1 = macro_precision_recall_f1(circuit_cm, cp, cr);
        const double linear_f1 = macro_precision_recall_f1(linear_cm, lp, lr);

        auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double sps = (1000.0 * (2.0 * n_train + 2.0 * n_test)) / std::max(1e-6, ms);

        circuit_logger.log_epoch_metric(epoch, "train", circuit_train_loss, circuit_train_acc,
                                        std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN(),
                                        cfg.lr,
                                        std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN(),
                                        ms, sps);
        circuit_logger.log_epoch_metric(epoch, "test", circuit_test_loss, circuit_test_acc,
                                        cp, cr, circuit_f1, cfg.lr,
                                        std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN(),
                                        std::numeric_limits<double>::quiet_NaN(),
                                        ms, sps);

        linear_logger.log_epoch_metric(epoch, "train", linear_train_loss, linear_train_acc,
                                       std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::quiet_NaN(),
                                       linear_lr,
                                       std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::quiet_NaN(),
                                       ms, sps);
        linear_logger.log_epoch_metric(epoch, "test", linear_test_loss, linear_test_acc,
                                       lp, lr, linear_f1, linear_lr,
                                       std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::quiet_NaN(),
                                       std::numeric_limits<double>::quiet_NaN(),
                                       ms, sps);

        cmp << epoch << ","
            << std::fixed << std::setprecision(6)
            << circuit_train_acc << ","
            << circuit_test_acc << ","
            << linear_train_acc << ","
            << linear_test_acc << ","
            << (linear_test_acc - circuit_test_acc)
            << "\n";

        std::cout << "[compare] epoch=" << epoch
                  << " circuit_test_acc=" << std::fixed << std::setprecision(4) << circuit_test_acc
                  << " linear_test_acc=" << linear_test_acc
                  << " gap=" << (linear_test_acc - circuit_test_acc)
                  << std::endl;
    }

    cmp.close();
    circuit_logger.write_manifest_end();
    linear_logger.write_manifest_end();

    std::cout << "[compare] wrote " << compare_csv << std::endl;
    std::cout << "[compare] run_ids: circuit=" << circuit_logger.run_id()
              << " linear=" << linear_logger.run_id() << std::endl;
    return 0;
}
