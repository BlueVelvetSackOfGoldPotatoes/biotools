#include "core/data/dataloader.h"
#include "core/io/run_logger.h"
#include "core/metrics/metrics.h"
#include "models/cells/src/cell_circuit.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
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

} // namespace

int main() {
    std::cout << "=== Cell Circuit Benchmark (MNIST, local three-factor) ===" << std::endl;

    const int epochs = env_int("CELL_CIRCUIT_EPOCHS", 6);
    const int train_limit = env_int("CELL_CIRCUIT_TRAIN_SAMPLES", 12000);
    const int test_limit = env_int("CELL_CIRCUIT_TEST_SAMPLES", 3000);
    const int seed = env_int("CELL_CIRCUIT_SEED", 42);

    cells::CellCircuitConfig cfg;
    cfg.input_dim = 784;
    cfg.retina_dim = static_cast<std::size_t>(env_int("CELL_CIRCUIT_RETINA", 196));
    cfg.v1_dim = static_cast<std::size_t>(env_int("CELL_CIRCUIT_V1", 256));
    cfg.assoc_dim = static_cast<std::size_t>(env_int("CELL_CIRCUIT_ASSOC", 128));
    cfg.decision_dim = 10;
    cfg.dt = env_double("CELL_CIRCUIT_DT", 0.02);
    cfg.lr = env_double("CELL_CIRCUIT_LR", 0.004);
    cfg.eligibility_decay = env_double("CELL_CIRCUIT_ELIGIBILITY_DECAY", 0.95);
    cfg.weight_decay = env_double("CELL_CIRCUIT_WEIGHT_DECAY", 1e-4);
    cfg.homeostasis_hmax = env_double("CELL_CIRCUIT_HMAX", 2.0);
    cfg.seed = static_cast<unsigned int>(std::max(0, seed));

    MNISTData train = MNISTLoader::load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    MNISTData test = MNISTLoader::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    const int n_train = std::min(train_limit, static_cast<int>(train.images.rows));
    const int n_test = std::min(test_limit, static_cast<int>(test.images.rows));

    cells::CellCircuit circuit(cfg);

    std::ostringstream params_json;
    params_json << "{"
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

    RunLogger logger("cells", "cell_circuit_mnist", seed, "mnist-idx-v1", params_json.str(), "mnist", "MNIST", "classification");

    std::vector<int> base_indices(n_train);
    std::iota(base_indices.begin(), base_indices.end(), 0);
    std::mt19937 rng(static_cast<unsigned int>(seed));

    std::cout << "[cell-circuit] params=" << circuit.parameter_count()
              << " train=" << n_train << " test=" << n_test << std::endl;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        auto t0 = std::chrono::high_resolution_clock::now();

        std::vector<int> order = base_indices;
        std::shuffle(order.begin(), order.end(), rng);

        circuit.reset_state();
        double train_loss = 0.0;
        double train_reward = 0.0;
        int train_correct = 0;
        double train_energy = 0.0;
        double train_homeo = 0.0;

        for (int i = 0; i < n_train; ++i) {
            const int idx = order[static_cast<std::size_t>(i)];
            std::vector<double> x = row_to_vec(train.images, static_cast<std::size_t>(idx));
            const int y = static_cast<int>(train.labels[static_cast<std::size_t>(idx)]);

            const auto step = circuit.train_one(x, y);
            train_loss += step.loss;
            train_reward += step.reward;
            train_energy += step.mean_energy;
            train_homeo += step.mean_homeostasis;
            if (step.pred == y) train_correct++;
        }

        const double train_acc = static_cast<double>(train_correct) / std::max(1, n_train);
        train_loss /= std::max(1, n_train);
        train_reward /= std::max(1, n_train);
        train_energy /= std::max(1, n_train);
        train_homeo /= std::max(1, n_train);

        circuit.reset_state();
        std::vector<int> preds;
        preds.reserve(n_test);
        std::vector<int> truths;
        truths.reserve(n_test);

        double test_loss = 0.0;
        double test_energy = 0.0;
        double test_homeo = 0.0;

        for (int i = 0; i < n_test; ++i) {
            std::vector<double> x = row_to_vec(test.images, static_cast<std::size_t>(i));
            const int y = static_cast<int>(test.labels[static_cast<std::size_t>(i)]);
            const auto step = circuit.evaluate_one(x, y);
            preds.push_back(step.pred);
            truths.push_back(y);
            test_loss += step.loss;
            test_energy += step.mean_energy;
            test_homeo += step.mean_homeostasis;
        }

        test_loss /= std::max(1, n_test);
        test_energy /= std::max(1, n_test);
        test_homeo /= std::max(1, n_test);

        const double test_acc = Metrics::accuracy(preds, truths);
        const Tensor cm = Metrics::confusion_matrix(preds, truths, 10);
        double p_macro = 0.0;
        double r_macro = 0.0;
        const double f1_macro = macro_precision_recall_f1(cm, p_macro, r_macro);

        auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double sps = (1000.0 * (n_train + n_test)) / std::max(1e-6, ms);

        logger.log_epoch_metric(epoch, "train", train_loss, train_acc,
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                cfg.lr,
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                ms, sps);

        logger.log_epoch_metric(epoch, "test", test_loss, test_acc,
                                p_macro, r_macro, f1_macro,
                                cfg.lr,
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                ms, sps);

        logger.append_csv_row(
            "model_specific/cells/circuit_dynamics.csv",
            {"run_id","epoch","train_reward_mean","train_energy_mean","train_homeostasis_mean",
             "test_energy_mean","test_homeostasis_mean","parameter_count"},
            {logger.run_id(), std::to_string(epoch), std::to_string(train_reward),
             std::to_string(train_energy), std::to_string(train_homeo),
             std::to_string(test_energy), std::to_string(test_homeo),
             std::to_string(circuit.parameter_count())});

        std::cout << "[cell-circuit] epoch=" << epoch
                  << " train_loss=" << std::fixed << std::setprecision(4) << train_loss
                  << " train_acc=" << train_acc
                  << " test_acc=" << test_acc
                  << " test_f1=" << f1_macro
                  << " reward=" << train_reward
                  << std::endl;
    }

    logger.write_manifest_end();
    return 0;
}
