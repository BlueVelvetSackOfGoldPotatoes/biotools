#include "core/nn/module.h"
#include "core/losses/losses.h"
#include "core/data/dataloader.h"
#include "core/metrics/metrics.h"
#include "core/io/run_logger.h"
#include "models/hebbian/src/hebbian_models.h"
#include "benchmarks/logging_utils.h"
#include "benchmarks/hpo_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>

static double tensor_l2(const Tensor& t) {
    double s = 0.0;
    for (double v : t.data) s += v * v;
    return std::sqrt(s);
}

int main() {
    std::cout << "=== Hebbian Benchmark (MNIST) ===" << std::endl;
    const int seed = hpo::env_int("HEBBIAN_SEED", 42);
    std::mt19937 rng(static_cast<unsigned int>(seed));
    const int hebb_passes = hpo::env_int("HEBBIAN_PASSES", 10);
    const int probe_epochs = hpo::env_int("HEBBIAN_PROBE_EPOCHS", 100);
    const size_t feature_width = hpo::env_size("HEBBIAN_FEATURE_WIDTH", 512);
    const double hebbian_lr = hpo::env_double("HEBBIAN_LR", 0.01);
    const double probe_lr = hpo::env_double("HEBBIAN_PROBE_LR", 0.001);
    const std::string hebb_rule = hpo::to_lower(hpo::env_string("HEBBIAN_RULE", "sanger"));
    const auto probe_optimizer_type =
        hpo::parse_optimizer(hpo::env_string("HEBBIAN_PROBE_OPTIMIZER", "adam"));
    const auto probe_loss_type =
        hpo::parse_loss(hpo::env_string("HEBBIAN_PROBE_LOSS", "cross_entropy"));

    auto train = MNISTLoader::load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    auto test = MNISTLoader::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");
    const size_t eval_batch_size = benchlog::resolve_eval_batch_size(64);

    // Mid-training eval uses a subset for speed; final eval uses the full test set.
    size_t mid_test_n = std::min(static_cast<size_t>(2000), test.images.rows);
    Tensor mid_test_x = test.images.slice_rows(0, mid_test_n);
    Tensor mid_test_y = test.one_hot.slice_rows(0, mid_test_n);

    std::cout << "\n--- HebbianClassifier (Oja + LinearProbe) ---" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::ostringstream params_json;
    params_json << "{\"feature_layers\":[784," << feature_width << "],\"rule\":\"" << hebb_rule
                << "\",\"hebbian_lr\":" << hebbian_lr
                << ",\"probe_epochs\":" << probe_epochs
                << ",\"probe_lr\":" << probe_lr
                << ",\"probe_optimizer\":\"" << hpo::optimizer_name(probe_optimizer_type) << "\""
                << ",\"probe_loss\":\"" << hpo::loss_name(probe_loss_type) << "\"}";
    RunLogger logger("hebbian", "hebbian_oja_linear_probe", seed, "mnist-idx-v1", params_json.str());

    HebbianClassifier model({784, feature_width}, 10, hebbian_lr, hebb_rule, rng);
    benchlog::BenchBioHarness bio("hebbian", "hebbian_oja_linear_probe", seed);
    bio.attach(model.parameters(), model.gradients(), "hebbian_param");

    std::cout << "Phase 1: Unsupervised Hebbian training (" << hebb_passes << " passes)..." << std::endl;
    for (int pass = 0; pass < hebb_passes; ++pass) {
        bio.begin_epoch(pass + 1);
        auto weights = model.feature_extractor.get_weights();
        std::vector<double> before;
        before.reserve(weights.size());
        for (const auto* w : weights) before.push_back(tensor_l2(*w));

        std::vector<size_t> indices(train.images.rows);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        for (size_t i = 0; i + 32 <= train.images.rows; i += 32) {
            bio.before_forward();
            Tensor batch = extract_batch(train.images, indices, i, 32);
            model.train_features(batch);
            bio.after_optimizer_step();
        }

        for (size_t li = 0; li < weights.size(); ++li) {
            const double after = tensor_l2(*weights[li]);
            const double delta = std::abs(after - before[li]);
            logger.append_csv_row(
                "model_specific/hebbian/hebb_update.csv",
                {"run_id", "step", "layer", "delta_w_l2"},
                {logger.run_id(), std::to_string(pass + 1), "layer_" + std::to_string(li), std::to_string(delta)}
            );
        }
        // Weight feature norms (learned features as pixel importance)
        if (pass == 0 || pass == 4 || pass == 9) {
            auto weights = model.feature_extractor.get_weights();
            if (!weights.empty()) {
                const Tensor& w0 = *weights[0];
                for (size_t p = 0; p < std::min(w0.rows, static_cast<size_t>(784)); ++p) {
                    double sq = 0.0;
                    for (size_t n = 0; n < w0.cols; ++n) sq += w0(p, n) * w0(p, n);
                    logger.append_csv_row(
                        "model_specific/hebbian/weight_feature_norms.csv",
                        {"run_id", "pass", "pixel_idx", "weight_norm", "row", "col"},
                        {logger.run_id(), std::to_string(pass + 1), std::to_string(p),
                         std::to_string(std::sqrt(sq)),
                         std::to_string(p / 28), std::to_string(p % 28)}
                    );
                }
            }
        }
        bio.end_epoch(logger, pass + 1);
        std::cout << "  Pass " << pass + 1 << " complete" << std::endl;
    }

    std::cout << "Phase 2: Training linear probe (Adam)..." << std::endl;
    auto loss_fn = hpo::make_loss(probe_loss_type);
    hpo::OptimizerConfig opt_cfg;
    opt_cfg.type = probe_optimizer_type;
    opt_cfg.lr = probe_lr;
    opt_cfg.momentum = hpo::env_double("HEBBIAN_PROBE_MOMENTUM", 0.9);
    opt_cfg.weight_decay = hpo::env_double("HEBBIAN_PROBE_WEIGHT_DECAY", 0.0);
    opt_cfg.beta1 = hpo::env_double("HEBBIAN_PROBE_BETA1", 0.9);
    opt_cfg.beta2 = hpo::env_double("HEBBIAN_PROBE_BETA2", 0.999);
    opt_cfg.alpha = hpo::env_double("HEBBIAN_PROBE_RMSPROP_ALPHA", 0.99);
    opt_cfg.eps = hpo::env_double("HEBBIAN_PROBE_EPS", 1e-8);
    opt_cfg.grad_clip_norm = hpo::env_double("HEBBIAN_PROBE_GRAD_CLIP", 1.0);
    hpo::TensorParamOptimizer optimizer(opt_cfg);
    const size_t probe_batch_size = 64;
    const size_t batch_log_every = std::max<std::size_t>(1, hpo::env_size("BATCH_LOG_EVERY", 10));
    const size_t total_batches = train.images.rows / probe_batch_size;
    size_t global_step = 0;

    for (int epoch = 0; epoch < probe_epochs; ++epoch) {
        const int bio_epoch = epoch + hebb_passes + 1;
        bio.begin_epoch(bio_epoch);
        auto epoch_start = std::chrono::high_resolution_clock::now();
        std::vector<size_t> indices(train.images.rows);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        double epoch_loss = 0.0;
        double epoch_acc = 0.0;
        double grad_norm_mean_sum = 0.0;
        double param_norm_mean_sum = 0.0;
        double grad_norm_max_epoch = 0.0;
        double param_norm_max_epoch = 0.0;
        size_t n_batches = 0;

        for (size_t b = 0; b + probe_batch_size <= train.images.rows; b += probe_batch_size) {
            bio.before_forward();
            Tensor bx = extract_batch(train.images, indices, b, probe_batch_size);
            Tensor by = extract_batch(train.one_hot, indices, b, probe_batch_size);

            model.zero_grad();
            Tensor logits = model.forward(bx);
            Tensor probs = LossFunctions::softmax(logits);
            const double loss = loss_fn->forward(probs, by);
            const double batch_acc = Metrics::accuracy_from_tensor(probs, by);
            epoch_loss += loss;
            epoch_acc += batch_acc;

            Tensor grad = loss_fn->backward(probs, by);
            if (probe_loss_type != hpo::LossType::CrossEntropy) {
                grad = LossFunctions::softmax_backward(probs, grad);
            }
            model.backward(grad);
            bio.after_backward(loss, std::numeric_limits<double>::quiet_NaN(), probe_lr);

            auto params = model.parameters();
            auto grads = model.gradients();
            auto [gmean, gmax] = benchlog::norm_stats(grads);
            auto [pmean, pmax] = benchlog::norm_stats(params);
            grad_norm_mean_sum += gmean;
            param_norm_mean_sum += pmean;
            grad_norm_max_epoch = std::max(grad_norm_max_epoch, gmax);
            param_norm_max_epoch = std::max(param_norm_max_epoch, pmax);

            optimizer.step(params, grads);
            hpo::TensorParamOptimizer::zero_grad(grads);
            bio.after_optimizer_step();
            n_batches++;
            global_step++;

            const size_t batch_idx = n_batches;
            if (batch_idx % batch_log_every == 0 || batch_idx == total_batches) {
                logger.append_csv_row(
                    "learning/batch_metrics.csv",
                    {"run_id", "model_family", "model_variant", "epoch", "batch", "total_batches",
                     "global_step", "split", "loss", "accuracy", "lr"},
                    {logger.run_id(), "hebbian", "hebbian_oja_linear_probe", std::to_string(epoch + 1),
                     std::to_string(batch_idx), std::to_string(total_batches), std::to_string(global_step),
                     "train_batch", std::to_string(loss), std::to_string(batch_acc), std::to_string(probe_lr)}
                );
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        const double epoch_ms =
            std::chrono::duration<double, std::milli>(epoch_end - epoch_start).count();
        const double train_loss = epoch_loss / static_cast<double>(n_batches);
        const double train_acc = epoch_acc / static_cast<double>(n_batches);
        const double grad_norm_mean = grad_norm_mean_sum / static_cast<double>(n_batches);
        const double param_norm_mean = param_norm_mean_sum / static_cast<double>(n_batches);
        const double samples_per_sec =
            (n_batches * static_cast<double>(probe_batch_size)) / (epoch_ms / 1000.0 + 1e-12);

        logger.log_epoch_metric(
            epoch + 1,
            "train",
            train_loss,
            train_acc,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            probe_lr,
            grad_norm_mean,
            grad_norm_max_epoch,
            param_norm_mean,
            param_norm_max_epoch,
            epoch_ms,
            samples_per_sec
        );

        auto eval = benchlog::evaluate_classification_batched(
            model, mid_test_x, mid_test_y, *loss_fn, eval_batch_size
        );
        const double test_acc = eval.accuracy;
        const double test_loss = eval.loss;

        logger.log_epoch_metric(
            epoch + 1,
            "test",
            test_loss,
            test_acc,
            eval.precision_macro,
            eval.recall_macro,
            eval.f1_macro,
            probe_lr,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            eval.elapsed_ms,
            eval.samples_per_sec
        );
        benchlog::log_confusion_and_class_metrics(logger, epoch + 1, "test", eval.preds, eval.truths, 10);
        bio.end_epoch(logger, bio_epoch);

        if ((epoch + 1) % 10 == 0 || epoch == probe_epochs - 1) {
            std::cout << "  Probe epoch " << epoch + 1
                      << " | Loss: " << std::fixed << std::setprecision(4) << train_loss
                      << " | Test Acc: " << std::setprecision(2) << test_acc * 100 << "%" << std::endl;
        }
    }

    auto final_eval = benchlog::evaluate_classification_batched(
        model, test.images, test.one_hot, *loss_fn, eval_batch_size, &logger, "test", true, true, 10
    );
    logger.log_system_metric(
        RunLogger::utc_now_iso8601(),
        final_eval.samples_per_sec,
        final_eval.per_sample_ms,
        final_eval.per_sample_ms,
        final_eval.per_sample_ms,
        0.0,
        0.0
    );

    const double acc = final_eval.accuracy;
    auto end = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "HebbianClassifier Final Acc: " << std::fixed << std::setprecision(2) << acc * 100
              << "% (" << std::setprecision(1) << elapsed << "s)"
              << " | " << (acc >= 0.70 ? "PASS" : "FAIL") << std::endl;

    std::cout << "\n--- Hopfield Network (pattern recall) ---" << std::endl;
    HopfieldNetwork hopfield(784);
    const int n_store = 3;
    std::vector<Tensor> patterns;
    for (int i = 0; i < n_store; ++i) {
        Tensor pattern = train.images.row(i * 2000).apply([](double x) { return x > 0.3 ? 1.0 : -1.0; });
        hopfield.store(pattern);
        patterns.push_back(pattern);
    }

    int recalled = 0;
    int energy_step = 1;
    for (int i = 0; i < n_store; ++i) {
        Tensor noisy = patterns[i];
        for (size_t j = 0; j < noisy.cols; ++j) {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            if (dist(rng) < 0.1) noisy(0, j) *= -1.0;
        }

        const double e_before = hopfield.energy(noisy);
        Tensor retrieved = hopfield.recall(noisy, 200);
        const double e_after = hopfield.energy(retrieved);
        logger.append_csv_row(
            "model_specific/hebbian/energy_curve.csv",
            {"run_id", "step", "energy"},
            {logger.run_id(), std::to_string(energy_step++), std::to_string(e_before)}
        );
        logger.append_csv_row(
            "model_specific/hebbian/energy_curve.csv",
            {"run_id", "step", "energy"},
            {logger.run_id(), std::to_string(energy_step++), std::to_string(e_after)}
        );

        double match = 0.0;
        for (size_t j = 0; j < patterns[i].cols; ++j) {
            if (retrieved(0, j) == patterns[i](0, j)) match += 1.0;
        }
        const double ratio = match / patterns[i].cols;
        std::cout << "  Pattern " << i << ": " << std::fixed << std::setprecision(1) << ratio * 100
                  << "% match" << std::endl;
        if (ratio > 0.90) recalled++;
    }

    std::cout << "Recalled " << recalled << "/" << n_store << " patterns (>90% match)"
              << " | " << (recalled >= n_store - 1 ? "PASS" : "FAIL") << std::endl;

    logger.write_manifest_end();
    std::cout << "\n=== Hebbian Benchmark Complete ===" << std::endl;
    return (acc >= 0.70) ? 0 : 1;
}
