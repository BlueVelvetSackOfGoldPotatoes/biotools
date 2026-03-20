#include "core/nn/module.h"
#include "core/losses/losses.h"
#include "core/data/dataloader.h"
#include "core/metrics/metrics.h"
#include "core/io/run_logger.h"
#include "models/forward_forward/src/ff_models.h"
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

int main() {
    std::cout << "=== Forward-Forward Benchmark (MNIST) ===" << std::endl;
    const int seed = hpo::env_int("FF_SEED", 42);
    std::mt19937 rng(static_cast<unsigned int>(seed));
    const int EPOCHS = hpo::env_int("FF_EPOCHS", 60);
    const size_t BATCH_SIZE = hpo::env_size("FF_BATCH_SIZE", 64);
    const double LR = hpo::env_double("FF_LR", 0.03);
    const double threshold = hpo::env_double("FF_THRESHOLD", 2.0);
    const size_t hidden1 = hpo::env_size("FF_HIDDEN1", 500);
    const size_t hidden2 = hpo::env_size("FF_HIDDEN2", 500);
    const size_t TRAIN_N = hpo::env_size("FF_TRAIN_N", 60000);
    std::ostringstream params_json;
    params_json << "{\"epochs\":" << EPOCHS
                << ",\"batch_size\":" << BATCH_SIZE
                << ",\"lr\":" << LR
                << ",\"threshold\":" << threshold
                << ",\"layers\":[784," << hidden1 << "," << hidden2 << "]"
                << "}";
    const std::string variant =
        "ff_784_" + std::to_string(hidden1) + "_" + std::to_string(hidden2);

    auto train = MNISTLoader::load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    auto test = MNISTLoader::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    std::vector<int> test_labels(test.labels.begin(), test.labels.end());
    ForwardForwardNetwork model({784, hidden1, hidden2}, 10, threshold, rng);
    RunLogger logger(
        "forward_forward",
        variant,
        seed,
        "mnist-idx-v1",
        params_json.str()
    );
    benchlog::BenchBioHarness bio("forward_forward", variant, seed);
    bio.attach(model.parameters(), model.all_gradients(), "ff_param");

    // Mid-training eval uses a subset for speed
    const size_t MID_TEST_N = std::min(static_cast<size_t>(2000), test.images.rows);

    auto start = std::chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        bio.begin_epoch(epoch + 1);
        auto epoch_start = std::chrono::high_resolution_clock::now();
        std::vector<size_t> indices(train.images.rows);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        double epoch_loss = 0.0;
        size_t n_batches = 0;
        for (size_t b = 0; b + BATCH_SIZE <= TRAIN_N; b += BATCH_SIZE) {
            bio.before_forward();
            Tensor bx = extract_batch(train.images, indices, b, BATCH_SIZE);
            Tensor by(BATCH_SIZE, 1);
            for (size_t i = 0; i < BATCH_SIZE; ++i) {
                by(i, 0) = static_cast<double>(train.labels[indices[b + i]]);
            }
            const double loss = model.train_step(bx, by, LR);
            epoch_loss += loss;
            bio.after_backward(loss, std::numeric_limits<double>::quiet_NaN(), LR);
            bio.after_optimizer_step();
            n_batches++;
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        const double epoch_ms =
            std::chrono::duration<double, std::milli>(epoch_end - epoch_start).count();
        const double avg_loss = epoch_loss / static_cast<double>(n_batches);

        Tensor test_x = test.images.slice_rows(0, MID_TEST_N);
        std::vector<int> test_y(test_labels.begin(), test_labels.begin() + static_cast<long>(MID_TEST_N));
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
            avg_loss,
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
            TRAIN_N / (epoch_ms / 1000.0 + 1e-12)
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
            MID_TEST_N / (eval_ms / 1000.0 + 1e-12)
        );
        benchlog::log_confusion_and_class_metrics(logger, epoch + 1, "test", preds, test_y, 10);
        bio.end_epoch(logger, epoch + 1);

        Tensor probe_x = train.images.slice_rows(0, 64);
        Tensor probe_labels(64, 1);
        for (size_t i = 0; i < 64; ++i) probe_labels(i, 0) = static_cast<double>(train.labels[i]);
        Tensor pos_in = ForwardForwardNetwork::embed_label(probe_x, probe_labels);
        Tensor neg_in = ForwardForwardNetwork::embed_wrong_label(probe_x, probe_labels, rng);
        for (size_t li = 0; li < model.layers.size(); ++li) {
            Tensor pos_g = model.layers[li].goodness(pos_in);
            Tensor neg_g = model.layers[li].goodness(neg_in);
            const double pos_mean = pos_g.mean();
            const double neg_mean = neg_g.mean();
            const double margin = pos_mean - neg_mean;
            logger.append_csv_row(
                "model_specific/forward_forward/goodness_stats.csv",
                {"run_id", "epoch", "layer", "margin", "pos_mean", "neg_mean", "threshold"},
                {logger.run_id(),
                 std::to_string(epoch + 1),
                 "layer_" + std::to_string(li),
                 std::to_string(margin),
                 std::to_string(pos_mean),
                 std::to_string(neg_mean),
                 std::to_string(model.layers[li].threshold)}
            );
            pos_in = model.layers[li].forward(pos_in);
            neg_in = model.layers[li].forward(neg_in);
        }

        if ((epoch + 1) % 5 == 0 || epoch == EPOCHS - 1) {
            std::cout << "Epoch " << std::setw(2) << epoch + 1 << "/" << EPOCHS
                      << " | Test Acc: " << std::fixed << std::setprecision(2) << acc * 100 << "%"
                      << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();

    Tensor final_test_x = test.images;
    std::vector<int> final_test_y = test_labels;
    auto infer_start = std::chrono::high_resolution_clock::now();
    auto preds = model.predict(final_test_x);
    auto infer_end = std::chrono::high_resolution_clock::now();
    const double infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    const double per_sample_ms = infer_ms / static_cast<double>(test.images.rows);
    const double acc = Metrics::accuracy(preds, final_test_y);

    benchlog::log_inference_from_preds(logger, "test", preds, final_test_y, per_sample_ms, static_cast<int>(test.images.rows));
    benchlog::log_calibration_from_preds(logger, "test", preds, final_test_y, 10);
    logger.log_system_metric(
        RunLogger::utc_now_iso8601(),
        test.images.rows / (infer_ms / 1000.0 + 1e-12),
        per_sample_ms,
        per_sample_ms,
        per_sample_ms,
        0.0,
        0.0
    );

    std::cout << "\nForward-Forward Final Acc: " << std::fixed << std::setprecision(2) << acc * 100 << "%"
              << " (" << std::setprecision(1) << elapsed << "s)"
              << " | " << (acc >= 0.70 ? "PASS" : "FAIL") << std::endl;

    logger.write_manifest_end();
    std::cout << "\n=== Forward-Forward Benchmark Complete ===" << std::endl;
    return (acc >= 0.70) ? 0 : 1;
}
