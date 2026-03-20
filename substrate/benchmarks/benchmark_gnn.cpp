#include "core/nn/module.h"
#include "core/losses/losses.h"
#include "core/data/dataloader.h"
#include "core/metrics/metrics.h"
#include "core/io/run_logger.h"
#include "models/gnn/src/gnn_models.h"
#include "benchmarks/logging_utils.h"
#include "benchmarks/hpo_utils.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

static Tensor create_node_features_7x7(const Tensor& images) {
    size_t n = images.rows;
    Tensor out(n, 49 * 16);
    for (size_t img = 0; img < n; ++img) {
        for (size_t r = 0; r < 7; ++r) {
            for (size_t c = 0; c < 7; ++c) {
                size_t node = r * 7 + c;
                for (size_t dr = 0; dr < 4; ++dr) {
                    for (size_t dc = 0; dc < 4; ++dc) {
                        size_t orig_r = r * 4 + dr;
                        size_t orig_c = c * 4 + dc;
                        if (orig_r < 28 && orig_c < 28) {
                            out(img, node * 16 + dr * 4 + dc) = images(img, orig_r * 28 + orig_c);
                        } else {
                            out(img, node * 16 + dr * 4 + dc) = 0.0;
                        }
                    }
                }
            }
        }
    }
    return out;
}

int main() {
    std::cout << "=== GNN Benchmark (MNIST) ===" << std::endl;
    const int seed = hpo::env_int("GNN_SEED", 42);
    std::mt19937 rng(static_cast<unsigned int>(seed));
    const int EPOCHS = hpo::env_int("GNN_EPOCHS", 30);
    const double LR = hpo::env_double("GNN_LR", 0.003);
    const size_t hidden_size = hpo::env_size("GNN_HIDDEN", 128);
    const size_t num_layers = hpo::env_size("GNN_LAYERS", 4);
    const size_t train_n_cfg = hpo::env_size("GNN_TRAIN_N", 20000);
    const size_t mid_test_n_cfg = hpo::env_size("GNN_MID_TEST_N", 2000);

    auto train = MNISTLoader::load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    auto test = MNISTLoader::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
    std::ostringstream params_json;
    params_json << "{\"epochs\":" << EPOCHS
                << ",\"lr\":" << LR
                << ",\"hidden\":" << hidden_size
                << ",\"layers\":" << num_layers
                << ",\"train_n\":" << train_n_cfg
                << ",\"mid_test_n\":" << mid_test_n_cfg
                << "}";
    RunLogger logger(
        "gnn",
        "gcn_grid7x7",
        seed,
        "mnist-idx-v1",
        params_json.str()
    );

    std::vector<int> train_labels(train.labels.begin(), train.labels.end());
    std::vector<int> test_labels(test.labels.begin(), test.labels.end());

    std::cout << "Creating node features..." << std::endl;
    Tensor train_nf = create_node_features_7x7(train.images);
    Tensor test_nf = create_node_features_7x7(test.images);

    GNNClassifier model(16, hidden_size, 10, num_layers, rng, 7, 7);
    benchlog::BenchBioHarness bio("gnn", "gcn_grid7x7", seed);
    bio.attach(model.parameters(), model.gradients(), "gnn_param");

    size_t train_n = std::min(train_n_cfg, train_nf.rows);
    // Mid-training eval uses a subset for speed
    size_t mid_test_n = std::min(mid_test_n_cfg, test_nf.rows);

    auto start = std::chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        bio.begin_epoch(epoch + 1);
        auto epoch_start = std::chrono::high_resolution_clock::now();
        std::vector<size_t> indices(train_n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        Tensor train_x = train_nf.slice_rows(0, train_n);
        std::vector<int> train_y(train_labels.begin(), train_labels.begin() + static_cast<long>(train_n));

        size_t feat_per_img = 49 * 16;
        Tensor shuffled_x(train_n, feat_per_img);
        std::vector<int> shuffled_y(train_n);
        for (size_t i = 0; i < train_n; ++i) {
            shuffled_y[i] = train_y[indices[i]];
            for (size_t j = 0; j < feat_per_img; ++j) shuffled_x(i, j) = train_x(indices[i], j);
        }

        bio.before_forward();
        model.train_step(shuffled_x, shuffled_y, LR);
        bio.after_backward(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(), LR);
        bio.after_optimizer_step();
        auto epoch_end = std::chrono::high_resolution_clock::now();
        const double epoch_ms =
            std::chrono::duration<double, std::milli>(epoch_end - epoch_start).count();

        Tensor test_x = test_nf.slice_rows(0, mid_test_n);
        std::vector<int> test_y(test_labels.begin(), test_labels.begin() + static_cast<long>(mid_test_n));
        auto eval_start = std::chrono::high_resolution_clock::now();
        auto preds = model.predict(test_x);
        const double acc = Metrics::accuracy(preds, test_y);
        auto eval_end = std::chrono::high_resolution_clock::now();
        const double eval_ms =
            std::chrono::duration<double, std::milli>(eval_end - eval_start).count();

        auto [precision_macro, recall_macro, f1_macro] =
            benchlog::macro_prf_from_preds(preds, test_y, 10);
        logger.log_epoch_metric(
            epoch + 1,
            "train",
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            LR,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            epoch_ms,
            train_n / (epoch_ms / 1000.0 + 1e-12)
        );
        logger.log_epoch_metric(
            epoch + 1,
            "test",
            std::numeric_limits<double>::quiet_NaN(),
            acc,
            precision_macro,
            recall_macro,
            f1_macro,
            LR,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            eval_ms,
            mid_test_n / (eval_ms / 1000.0 + 1e-12)
        );
        benchlog::log_confusion_and_class_metrics(logger, epoch + 1, "test", preds, test_y, 10);
        bio.end_epoch(logger, epoch + 1);

        auto params = model.parameters();
        for (size_t i = 0; i < params.size(); ++i) {
            logger.append_csv_row(
                "model_specific/gnn/message_norms.csv",
                {"run_id", "epoch", "layer", "msg_norm_mean"},
                {logger.run_id(), std::to_string(epoch + 1), "layer_" + std::to_string(i), std::to_string(params[i]->norm())}
            );
        }

        // Layer-wise parameter statistics
        double total_energy = 0.0;
        for (size_t i = 0; i < params.size(); ++i) {
            double energy = 0.0;
            size_t near_zero = 0;
            for (double v : params[i]->data) {
                energy += v * v;
                if (std::abs(v) < 0.001) near_zero++;
            }
            total_energy += energy;
            double sparsity = params[i]->data.empty() ? 0.0 : static_cast<double>(near_zero) / params[i]->data.size();
            logger.append_csv_row(
                "model_specific/gnn/layer_stats.csv",
                {"run_id", "epoch", "layer", "energy", "sparsity", "param_count"},
                {logger.run_id(), std::to_string(epoch + 1), "layer_" + std::to_string(i),
                 std::to_string(energy), std::to_string(sparsity),
                 std::to_string(params[i]->data.size())}
            );
        }

        std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS
                  << " | Test Acc: " << std::fixed << std::setprecision(2) << acc * 100 << "%" << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();

    Tensor final_test_x = test_nf;
    std::vector<int> final_test_y = test_labels;
    auto infer_start = std::chrono::high_resolution_clock::now();
    auto preds = model.predict(final_test_x);
    auto infer_end = std::chrono::high_resolution_clock::now();
    const double infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    const double per_sample_ms = infer_ms / static_cast<double>(final_test_x.rows);
    const double acc = Metrics::accuracy(preds, final_test_y);

    benchlog::log_inference_from_preds(logger, "test", preds, final_test_y, per_sample_ms, static_cast<int>(final_test_x.rows));
    benchlog::log_calibration_from_preds(logger, "test", preds, final_test_y, 10);
    logger.log_system_metric(
        RunLogger::utc_now_iso8601(),
        final_test_x.rows / (infer_ms / 1000.0 + 1e-12),
        per_sample_ms,
        per_sample_ms,
        per_sample_ms,
        0.0,
        0.0
    );

    const size_t emb_n = std::min(static_cast<size_t>(500), final_test_x.rows);
    for (size_t i = 0; i < emb_n; ++i) {
        logger.append_csv_row(
            "model_specific/gnn/embedding_2d.csv",
            {"run_id", "x", "y", "class"},
            {logger.run_id(), std::to_string(final_test_x(i, 0)), std::to_string(final_test_x(i, 1)), std::to_string(final_test_y[i])}
        );
    }

    std::cout << "\nGNN Final Acc: " << std::fixed << std::setprecision(2) << acc * 100 << "%"
              << " (" << std::setprecision(1) << elapsed << "s)"
              << " | " << (acc >= 0.80 ? "PASS" : "FAIL") << std::endl;
    logger.write_manifest_end();

    std::cout << "\n=== GNN Benchmark Complete ===" << std::endl;
    return (acc >= 0.80) ? 0 : 1;
}
