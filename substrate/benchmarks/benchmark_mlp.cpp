#include "core/nn/module.h"
#include "core/losses/losses.h"
#include "core/optim/optimizer.h"
#include "core/data/dataloader.h"
#include "core/metrics/metrics.h"
#include "core/io/run_logger.h"
#include "benchmarks/logging_utils.h"
#include "benchmarks/hpo_utils.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <sstream>

namespace {

std::pair<double, double> norm_stats(const std::vector<Tensor*>& ts) {
    if (ts.empty()) {
        double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan};
    }
    double sum = 0.0;
    double mx = 0.0;
    for (const auto* t : ts) {
        double n = t->norm();
        sum += n;
        mx = std::max(mx, n);
    }
    return {sum / ts.size(), mx};
}

void log_mlp_model_specific(RunLogger& logger, Sequential& model, int epoch) {
    auto* first_linear = dynamic_cast<Linear*>(&model.get(0));
    if (!first_linear) return;

    for (size_t neuron = 0; neuron < first_linear->out_features; ++neuron) {
        double sq_sum = 0.0;
        double abs_sum = 0.0;
        for (size_t i = 0; i < first_linear->in_features; ++i) {
            double w = first_linear->weights(i, neuron);
            sq_sum += w * w;
            abs_sum += std::abs(w);
        }
        double weight_l2 = std::sqrt(sq_sum);
        double weight_abs_mean = abs_sum / static_cast<double>(first_linear->in_features);
        logger.append_csv_row(
            "model_specific/mlp/neuron_stats.csv",
            {"run_id","epoch","layer","neuron","mean_act","std_act","sparsity","weight_l2","weight_abs_mean"},
            {logger.run_id(),
             std::to_string(epoch),
             "fc1",
             std::to_string(neuron),
             "nan",
             "nan",
             "nan",
             std::to_string(weight_l2),
             std::to_string(weight_abs_mean)}
        );
    }

    // --- layer_activation_stats.csv ---
    // Layer indices in the Sequential: 0=fc1, 2=fc2, 4=fc3 (ReLU layers at 1, 3).
    const std::pair<int, std::string> layer_map[] = {{0, "fc1"}, {2, "fc2"}, {4, "fc3"}};
    for (const auto& [idx, name] : layer_map) {
        auto* lin = dynamic_cast<Linear*>(&model.get(idx));
        if (!lin) continue;

        size_t n_neurons = lin->out_features;
        size_t n_inputs  = lin->in_features;

        size_t dead_count    = 0;
        size_t sparse_count  = 0;
        size_t total_weights = n_neurons * n_inputs;
        double l2_sum        = 0.0;
        double l2_max        = 0.0;

        for (size_t neuron = 0; neuron < n_neurons; ++neuron) {
            double sq_sum = 0.0;
            for (size_t i = 0; i < n_inputs; ++i) {
                double w = lin->weights(i, neuron);
                sq_sum += w * w;
                if (std::abs(w) < 0.01) ++sparse_count;
            }
            double l2 = std::sqrt(sq_sum);
            if (l2 < 0.001) ++dead_count;
            l2_sum += l2;
            l2_max = std::max(l2_max, l2);
        }

        double dead_fraction  = static_cast<double>(dead_count)   / static_cast<double>(n_neurons);
        double mean_weight_l2 = l2_sum / static_cast<double>(n_neurons);
        double weight_sparsity = static_cast<double>(sparse_count) / static_cast<double>(total_weights);

        logger.append_csv_row(
            "model_specific/mlp/layer_activation_stats.csv",
            {"run_id", "epoch", "layer", "dead_fraction", "mean_weight_l2", "max_weight_l2", "weight_sparsity"},
            {logger.run_id(),
             std::to_string(epoch),
             name,
             std::to_string(dead_fraction),
             std::to_string(mean_weight_l2),
             std::to_string(l2_max),
             std::to_string(weight_sparsity)}
        );
    }

    // --- pixel_importance.csv --- logged only at epochs 1, 10, and 20.
    if (epoch == 1 || epoch == 10 || epoch == 20) {
        // fc1 has shape (in_features=784, out_features=256).
        // For each input pixel p, compute L2 norm over all outgoing weights.
        size_t n_pixels  = first_linear->in_features;   // 784
        size_t n_neurons = first_linear->out_features;  // 256

        for (size_t pixel = 0; pixel < n_pixels; ++pixel) {
            double sq_sum = 0.0;
            for (size_t neuron = 0; neuron < n_neurons; ++neuron) {
                double w = first_linear->weights(pixel, neuron);
                sq_sum += w * w;
            }
            double weight_norm = std::sqrt(sq_sum);
            size_t row = pixel / 28;
            size_t col = pixel % 28;

            logger.append_csv_row(
                "model_specific/mlp/pixel_importance.csv",
                {"run_id", "epoch", "pixel_idx", "weight_norm", "row", "col"},
                {logger.run_id(),
                 std::to_string(epoch),
                 std::to_string(pixel),
                 std::to_string(weight_norm),
                 std::to_string(row),
                 std::to_string(col)}
            );
        }
    }
}

} // namespace

int main() {
    std::cout << "=== MLP Benchmark (MNIST) ===" << std::endl;
    const size_t EPOCHS = hpo::env_size("MLP_EPOCHS", 20);
    const size_t BATCH_SIZE = hpo::env_size("MLP_BATCH_SIZE", 64);
    const size_t EVAL_BATCH_SIZE =
        benchlog::resolve_eval_batch_size(hpo::env_size("MLP_EVAL_BATCH_SIZE", BATCH_SIZE));
    const double LR = hpo::env_double("MLP_LR", 0.01);
    const size_t H1 = hpo::env_size("MLP_H1", 256);
    const size_t H2 = hpo::env_size("MLP_H2", 128);
    const auto OPTIM = hpo::parse_optimizer(hpo::env_string("MLP_OPTIMIZER", "adam"));
    const auto LOSS = hpo::parse_loss(hpo::env_string("MLP_LOSS", "cross_entropy"));
    const int seed = hpo::env_int("MLP_SEED", 42);
    std::mt19937 rng(static_cast<unsigned int>(seed));

    auto train = MNISTLoader::load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    auto test = MNISTLoader::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    Sequential model;
    model.add(std::make_unique<Linear>(784, H1, "fc1", rng));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(H1, H2, "fc2", rng));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(H2, 10, "fc3", rng));

    hpo::OptimizerConfig opt_cfg;
    opt_cfg.type = OPTIM;
    opt_cfg.lr = LR;
    opt_cfg.momentum = hpo::env_double("MLP_MOMENTUM", 0.9);
    opt_cfg.weight_decay = hpo::env_double("MLP_WEIGHT_DECAY", 0.0);
    opt_cfg.beta1 = hpo::env_double("MLP_BETA1", 0.9);
    opt_cfg.beta2 = hpo::env_double("MLP_BETA2", 0.999);
    opt_cfg.alpha = hpo::env_double("MLP_RMSPROP_ALPHA", 0.99);
    opt_cfg.eps = hpo::env_double("MLP_EPS", 1e-8);
    opt_cfg.grad_clip_norm = hpo::env_double("MLP_GRAD_CLIP", 0.0);
    hpo::TensorParamOptimizer optimizer(opt_cfg);
    auto loss_fn = hpo::make_loss(LOSS);

    std::ostringstream params_json;
    params_json << "{\"epochs\":" << EPOCHS
                << ",\"batch_size\":" << BATCH_SIZE
                << ",\"lr\":" << LR
                << ",\"optimizer\":\"" << hpo::optimizer_name(OPTIM) << "\""
                << ",\"loss\":\"" << hpo::loss_name(LOSS) << "\""
                << ",\"h1\":" << H1
                << ",\"h2\":" << H2
                << "}";
    RunLogger logger("mlp", "baseline_784_256_128_10", seed, "mnist-idx-v1", params_json.str());
    benchlog::BenchBioHarness bio("mlp", "baseline_784_256_128_10", seed);
    bio.attach(model.parameters(), model.gradients(), "mlp_param");
    const size_t batch_log_every = std::max<std::size_t>(1, hpo::env_size("BATCH_LOG_EVERY", 10));
    const size_t total_batches = train.images.rows / BATCH_SIZE;
    size_t global_step = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        bio.begin_epoch(static_cast<int>(epoch + 1));
        auto epoch_start = std::chrono::high_resolution_clock::now();
        double epoch_loss = 0.0;
        double epoch_acc = 0.0;
        double grad_norm_mean_sum = 0.0;
        double param_norm_mean_sum = 0.0;
        double grad_norm_max_epoch = 0.0;
        double param_norm_max_epoch = 0.0;
        size_t n_batches = 0;

        std::vector<size_t> indices(train.images.rows);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        model.train();
        for (size_t b = 0; b + BATCH_SIZE <= train.images.rows; b += BATCH_SIZE) {
            bio.before_forward();
            Tensor batch_x = extract_batch(train.images, indices, b, BATCH_SIZE);
            Tensor batch_y = extract_batch(train.one_hot, indices, b, BATCH_SIZE);

            Tensor logits = model.forward(batch_x);
            Tensor probs = LossFunctions::softmax(logits);
            double loss = loss_fn->forward(probs, batch_y);
            const double batch_acc = Metrics::accuracy_from_tensor(probs, batch_y);
            epoch_loss += loss;
            epoch_acc += batch_acc;

            Tensor grad = loss_fn->backward(probs, batch_y);
            if (LOSS != hpo::LossType::CrossEntropy) {
                grad = LossFunctions::softmax_backward(probs, grad);
            }
            model.backward(grad);
            bio.after_backward(loss, std::numeric_limits<double>::quiet_NaN(), LR);

            auto grads = model.gradients();
            auto params = model.parameters();
            auto [gmean, gmax] = norm_stats(grads);
            auto [pmean, pmax] = norm_stats(params);
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
                    {logger.run_id(), "mlp", "baseline_784_256_128_10", std::to_string(epoch + 1),
                     std::to_string(batch_idx), std::to_string(total_batches), std::to_string(global_step),
                     "train_batch", std::to_string(loss), std::to_string(batch_acc), std::to_string(LR)}
                );
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_ms = std::chrono::duration<double, std::milli>(epoch_end - epoch_start).count();
        double train_loss = epoch_loss / static_cast<double>(n_batches);
        double train_acc = epoch_acc / static_cast<double>(n_batches);
        double grad_norm_mean = grad_norm_mean_sum / static_cast<double>(n_batches);
        double param_norm_mean = param_norm_mean_sum / static_cast<double>(n_batches);
        double samples_per_sec = (n_batches * BATCH_SIZE) / (epoch_ms / 1000.0 + 1e-12);

        // Eval
        model.eval();
        auto eval = benchlog::evaluate_classification_batched(
            model, test.images, test.one_hot, *loss_fn, EVAL_BATCH_SIZE
        );
        double test_acc = eval.accuracy;
        double test_loss = eval.loss;

        logger.log_epoch_metric(static_cast<int>(epoch + 1), "train",
                                train_loss, train_acc,
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                LR, grad_norm_mean, grad_norm_max_epoch,
                                param_norm_mean, param_norm_max_epoch,
                                epoch_ms, samples_per_sec);
        logger.log_epoch_metric(static_cast<int>(epoch + 1), "test",
                                test_loss, test_acc,
                                eval.precision_macro, eval.recall_macro, eval.f1_macro,
                                LR,
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::quiet_NaN(),
                                eval.elapsed_ms,
                                eval.samples_per_sec);
        benchlog::log_confusion_and_class_metrics(
            logger, static_cast<int>(epoch + 1), "test", eval.preds, eval.truths, 10
        );
        bio.end_epoch(logger, static_cast<int>(epoch + 1));

        std::cout << "Epoch " << std::setw(2) << epoch+1 << "/" << EPOCHS
                  << " | Loss: " << std::fixed << std::setprecision(4) << train_loss
                  << " | Test Acc: " << std::setprecision(2) << test_acc*100 << "%" << std::endl;

        log_mlp_model_specific(logger, model, static_cast<int>(epoch + 1));
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    model.eval();
    auto final_eval = benchlog::evaluate_classification_batched(
        model, test.images, test.one_hot, *loss_fn, EVAL_BATCH_SIZE, &logger, "test", true, true, 10
    );
    logger.log_system_metric(RunLogger::utc_now_iso8601(),
                             final_eval.samples_per_sec,
                             final_eval.per_sample_ms,
                             final_eval.per_sample_ms,
                             final_eval.per_sample_ms,
                             0.0, 0.0);

    double final_acc = final_eval.accuracy;
    std::cout << "\nFinal Test Accuracy: " << std::fixed << std::setprecision(2) << final_acc*100 << "%"
              << " (Time: " << std::setprecision(1) << elapsed << "s)"
              << " | " << (final_acc >= 0.90 ? "PASS" : "FAIL") << std::endl;
    logger.write_manifest_end();
    return final_acc >= 0.90 ? 0 : 1;
}
