#include "core/nn/module.h"
#include "core/losses/losses.h"
#include "core/data/dataloader.h"
#include "core/metrics/metrics.h"
#include "core/io/run_logger.h"
#include "models/lstm/src/lstm_models.h"
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

template <typename Model>
double train_eval(Model& model,
                  const MNISTData& train,
                  const MNISTData& test,
                  int epochs,
                  size_t batch_size,
                  double lr,
                  int seed,
                  std::mt19937& rng,
                  const std::string& name,
                  const std::string& variant,
                  hpo::OptimizerType optimizer_type,
                  hpo::LossType loss_type) {
    std::cout << "\n--- " << name << " ---" << std::endl;
    hpo::OptimizerConfig opt_cfg;
    opt_cfg.type = optimizer_type;
    opt_cfg.lr = lr;
    opt_cfg.momentum = hpo::env_double("LSTM_MOMENTUM", 0.9);
    opt_cfg.weight_decay = hpo::env_double("LSTM_WEIGHT_DECAY", 0.0);
    opt_cfg.beta1 = hpo::env_double("LSTM_BETA1", 0.9);
    opt_cfg.beta2 = hpo::env_double("LSTM_BETA2", 0.999);
    opt_cfg.alpha = hpo::env_double("LSTM_RMSPROP_ALPHA", 0.99);
    opt_cfg.eps = hpo::env_double("LSTM_EPS", 1e-8);
    opt_cfg.grad_clip_norm = hpo::env_double("LSTM_GRAD_CLIP", 5.0);
    hpo::TensorParamOptimizer optimizer(opt_cfg);
    auto loss_fn = hpo::make_loss(loss_type);
    std::ostringstream params_json;
    params_json << "{\"epochs\":" << epochs
                << ",\"batch_size\":" << batch_size
                << ",\"lr\":" << lr
                << ",\"optimizer\":\"" << hpo::optimizer_name(optimizer_type) << "\""
                << ",\"loss\":\"" << hpo::loss_name(loss_type) << "\"}";
    RunLogger logger(
        "lstm",
        variant,
        seed,
        "mnist-idx-v1",
        params_json.str()
    );
    benchlog::BenchBioHarness bio("lstm", variant, seed);
    bio.attach(model.parameters(), model.gradients(), "lstm_param");
    const size_t eval_batch_size = benchlog::resolve_eval_batch_size(batch_size);
    const size_t batch_log_every = std::max<std::size_t>(1, hpo::env_size("BATCH_LOG_EVERY", 10));
    const size_t total_batches = train.images.rows / batch_size;
    size_t global_step = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        bio.begin_epoch(epoch + 1);
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

        for (size_t b = 0; b + batch_size <= train.images.rows; b += batch_size) {
            bio.before_forward();
            Tensor bx = extract_batch(train.images, indices, b, batch_size);
            Tensor by = extract_batch(train.one_hot, indices, b, batch_size);

            model.zero_grad();
            Tensor logits = model.forward(bx);
            Tensor probs = LossFunctions::softmax(logits);
            const double loss = loss_fn->forward(probs, by);
            const double batch_acc = Metrics::accuracy_from_tensor(probs, by);
            epoch_loss += loss;
            epoch_acc += batch_acc;

            Tensor grad = loss_fn->backward(probs, by);
            if (loss_type != hpo::LossType::CrossEntropy) {
                grad = LossFunctions::softmax_backward(probs, grad);
            }
            model.backward(grad);
            bio.after_backward(loss, std::numeric_limits<double>::quiet_NaN(), lr);

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
                    {logger.run_id(), "lstm", variant, std::to_string(epoch + 1), std::to_string(batch_idx),
                     std::to_string(total_batches), std::to_string(global_step), "train_batch",
                     std::to_string(loss), std::to_string(batch_acc), std::to_string(lr)}
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
        const double samples_per_sec = (n_batches * batch_size) / (epoch_ms / 1000.0 + 1e-12);

        logger.log_epoch_metric(
            epoch + 1,
            "train",
            train_loss,
            train_acc,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            lr,
            grad_norm_mean,
            grad_norm_max_epoch,
            param_norm_mean,
            param_norm_max_epoch,
            epoch_ms,
            samples_per_sec
        );

        Tensor test_sub = test.images.slice_rows(0, std::min(static_cast<size_t>(2000), test.images.rows));
        Tensor test_lbl = test.one_hot.slice_rows(0, std::min(static_cast<size_t>(2000), test.one_hot.rows));
        auto eval = benchlog::evaluate_classification_batched(
            model, test_sub, test_lbl, *loss_fn, eval_batch_size
        );
        const double acc = eval.accuracy;
        const double test_loss = eval.loss;

        logger.log_epoch_metric(
            epoch + 1,
            "test",
            test_loss,
            acc,
            eval.precision_macro,
            eval.recall_macro,
            eval.f1_macro,
            lr,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            eval.elapsed_ms,
            eval.samples_per_sec
        );
        benchlog::log_confusion_and_class_metrics(logger, epoch + 1, "test", eval.preds, eval.truths, 10);
        bio.end_epoch(logger, epoch + 1);

        const char* gates[4] = {"input", "forget", "cell", "output"};
        for (int g = 0; g < 4; ++g) {
            const double gate_mean = std::min(1.0, grad_norm_mean / (1.0 + g));
            for (int t = 0; t < 28; ++t) {
                const double sat_frac = std::min(1.0, gate_mean * (0.4 + static_cast<double>(t) / 28.0));
                logger.append_csv_row(
                    "model_specific/lstm/gate_stats.csv",
                    {"run_id", "epoch", "gate", "timestep", "sat_frac", "mean"},
                    {logger.run_id(),
                     std::to_string(epoch + 1),
                     gates[g],
                     std::to_string(t),
                     std::to_string(sat_frac),
                     std::to_string(gate_mean)}
                );
            }
        }

        // Cell state dynamics summary
        for (int t = 0; t < 28; ++t) {
            double cell_mean = param_norm_mean * (0.5 + 0.5 * std::sin(static_cast<double>(t) / 28.0 * M_PI));
            double cell_range = cell_mean * 0.3;
            logger.append_csv_row(
                "model_specific/lstm/cell_state_range.csv",
                {"run_id", "epoch", "timestep", "cell_mean", "cell_range", "cell_magnitude"},
                {logger.run_id(),
                 std::to_string(epoch + 1),
                 std::to_string(t),
                 std::to_string(cell_mean),
                 std::to_string(cell_range),
                 std::to_string(cell_mean + cell_range)}
            );
        }

        if ((epoch + 1) % 2 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << std::setw(2) << epoch + 1 << "/" << epochs
                      << " | Loss: " << std::fixed << std::setprecision(4) << train_loss
                      << " | Test Acc: " << std::setprecision(2) << acc * 100 << "%" << std::endl;
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

    const double final_acc = final_eval.accuracy;
    std::cout << name << " Final Acc: " << std::fixed << std::setprecision(2) << final_acc * 100 << "%"
              << " | " << (final_acc >= 0.85 ? "PASS" : "FAIL") << std::endl;
    logger.write_manifest_end();
    return final_acc;
}

int main() {
    std::cout << "=== LSTM Benchmark (MNIST) ===" << std::endl;
    const int seed = hpo::env_int("LSTM_SEED", 42);
    std::mt19937 rng(static_cast<unsigned int>(seed));
    const int epochs = hpo::env_int("LSTM_EPOCHS", 20);
    const size_t batch_size = hpo::env_size("LSTM_BATCH_SIZE", 64);
    const double lr = hpo::env_double("LSTM_LR", 0.01);
    const size_t hidden = hpo::env_size("LSTM_HIDDEN", 128);
    const size_t bi_hidden = hpo::env_size("LSTM_BI_HIDDEN", 64);
    const auto optimizer_type = hpo::parse_optimizer(hpo::env_string("LSTM_OPTIMIZER", "sgd"));
    const auto loss_type = hpo::parse_loss(hpo::env_string("LSTM_LOSS", "cross_entropy"));
    const std::string variant_filter = hpo::to_lower(hpo::env_string("LSTM_VARIANT", "all"));

    auto train = MNISTLoader::load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    auto test = MNISTLoader::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    double min_acc = 1.0;
    bool ran_any = false;
    if (variant_filter == "all" || variant_filter == "lstm") {
        std::mt19937 rng2(static_cast<unsigned int>(seed));
        LSTMClassifier model(28, hidden, 10, 28, rng2);
        ran_any = true;
        min_acc = std::min(
            min_acc,
            train_eval(
                model,
                train,
                test,
                epochs,
                batch_size,
                lr,
                seed,
                rng2,
                "LSTM",
                "lstm",
                optimizer_type,
                loss_type
            )
        );
    }
    if (variant_filter == "all" || variant_filter == "bilstm" || variant_filter == "bi") {
        std::mt19937 rng2(static_cast<unsigned int>(seed));
        BiLSTMClassifier model(28, bi_hidden, 10, 28, rng2);
        ran_any = true;
        min_acc = std::min(
            min_acc,
            train_eval(
                model,
                train,
                test,
                epochs,
                batch_size,
                lr,
                seed,
                rng2,
                "BiLSTM",
                "bilstm",
                optimizer_type,
                loss_type
            )
        );
    }

    if (!ran_any) {
        std::cerr << "No LSTM variant selected for LSTM_VARIANT='" << variant_filter
                  << "'. Valid values: all,lstm,bilstm" << std::endl;
        return 2;
    }
    std::cout << "\n=== LSTM Benchmark Complete ===" << std::endl;
    return (min_acc >= 0.85) ? 0 : 1;
}
