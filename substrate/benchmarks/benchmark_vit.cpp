#include "core/nn/module.h"
#include "core/losses/losses.h"
#include "core/data/dataloader.h"
#include "core/metrics/metrics.h"
#include "core/io/run_logger.h"
#include "models/vit/src/vit_models.h"
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

int main() {
    std::cout << "=== ViT Benchmark (MNIST) ===" << std::endl;
    const int seed = hpo::env_int("VIT_SEED", 42);
    std::mt19937 rng(static_cast<unsigned int>(seed));
    const int EPOCHS = hpo::env_int("VIT_EPOCHS", 20);
    const size_t BATCH_SIZE = hpo::env_size("VIT_BATCH_SIZE", 64);
    const size_t EVAL_BATCH_SIZE =
        benchlog::resolve_eval_batch_size(hpo::env_size("VIT_EVAL_BATCH_SIZE", BATCH_SIZE));
    const double LR = hpo::env_double("VIT_LR", 0.001);
    const size_t patch_size = hpo::env_size("VIT_PATCH_SIZE", 7);
    const size_t num_heads = std::max<std::size_t>(1, hpo::env_size("VIT_NUM_HEADS", 4));
    size_t embed_dim = hpo::env_size("VIT_EMBED_DIM", 64);
    if (embed_dim % num_heads != 0) {
        embed_dim = ((embed_dim + num_heads - 1) / num_heads) * num_heads;
    }
    const size_t d_ff = hpo::env_size("VIT_D_FF", 128);
    const size_t num_blocks = hpo::env_size("VIT_NUM_BLOCKS", 2);
    const double dropout = hpo::env_double("VIT_DROPOUT", 0.1);
    const auto optimizer_type = hpo::parse_optimizer(hpo::env_string("VIT_OPTIMIZER", "adam"));
    const auto loss_type = hpo::parse_loss(hpo::env_string("VIT_LOSS", "cross_entropy"));

    auto train = MNISTLoader::load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    auto test = MNISTLoader::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    ViTClassifier model(28, patch_size, 1, 10, embed_dim, num_heads, d_ff, num_blocks, dropout, rng);
    auto loss_fn = hpo::make_loss(loss_type);
    hpo::OptimizerConfig opt_cfg;
    opt_cfg.type = optimizer_type;
    opt_cfg.lr = LR;
    opt_cfg.momentum = hpo::env_double("VIT_MOMENTUM", 0.9);
    opt_cfg.weight_decay = hpo::env_double("VIT_WEIGHT_DECAY", 0.0);
    opt_cfg.beta1 = hpo::env_double("VIT_BETA1", 0.9);
    opt_cfg.beta2 = hpo::env_double("VIT_BETA2", 0.999);
    opt_cfg.alpha = hpo::env_double("VIT_RMSPROP_ALPHA", 0.99);
    opt_cfg.eps = hpo::env_double("VIT_EPS", 1e-8);
    opt_cfg.grad_clip_norm = hpo::env_double("VIT_GRAD_CLIP", 1.0);
    hpo::TensorParamOptimizer optimizer(opt_cfg);
    std::ostringstream params_json;
    params_json << "{\"epochs\":" << EPOCHS
                << ",\"batch_size\":" << BATCH_SIZE
                << ",\"lr\":" << LR
                << ",\"optimizer\":\"" << hpo::optimizer_name(optimizer_type) << "\""
                << ",\"loss\":\"" << hpo::loss_name(loss_type) << "\""
                << ",\"patch_size\":" << patch_size
                << ",\"embed_dim\":" << embed_dim
                << ",\"num_heads\":" << num_heads
                << ",\"d_ff\":" << d_ff
                << ",\"num_blocks\":" << num_blocks
                << ",\"dropout\":" << dropout
                << "}";
    RunLogger logger(
        "vit",
        "vit_classifier",
        seed,
        "mnist-idx-v1",
        params_json.str()
    );
    benchlog::BenchBioHarness bio("vit", "vit_classifier", seed);
    bio.attach(model.parameters(), model.gradients(), "vit_param");
    const size_t batch_log_every = std::max<std::size_t>(1, hpo::env_size("BATCH_LOG_EVERY", 10));
    const size_t total_batches = train.images.rows / BATCH_SIZE;
    size_t global_step = 0;

    auto start = std::chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        bio.begin_epoch(epoch + 1);
        auto epoch_start = std::chrono::high_resolution_clock::now();
        double epoch_loss = 0.0;
        double epoch_acc = 0.0;
        double grad_norm_mean_sum = 0.0;
        double param_norm_mean_sum = 0.0;
        double grad_norm_max_epoch = 0.0;
        double param_norm_max_epoch = 0.0;
        size_t n_batches = 0;

        model.train();
        std::vector<size_t> indices(train.images.rows);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (size_t b = 0; b + BATCH_SIZE <= train.images.rows; b += BATCH_SIZE) {
            bio.before_forward();
            Tensor bx = extract_batch(train.images, indices, b, BATCH_SIZE);
            Tensor by = extract_batch(train.one_hot, indices, b, BATCH_SIZE);

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
            bio.after_backward(loss, std::numeric_limits<double>::quiet_NaN(), LR);

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
                    {logger.run_id(), "vit", "vit_classifier", std::to_string(epoch + 1),
                     std::to_string(batch_idx), std::to_string(total_batches), std::to_string(global_step),
                     "train_batch", std::to_string(loss), std::to_string(batch_acc), std::to_string(LR)}
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
        const double samples_per_sec = (n_batches * BATCH_SIZE) / (epoch_ms / 1000.0 + 1e-12);

        logger.log_epoch_metric(
            epoch + 1,
            "train",
            train_loss,
            train_acc,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            LR,
            grad_norm_mean,
            grad_norm_max_epoch,
            param_norm_mean,
            param_norm_max_epoch,
            epoch_ms,
            samples_per_sec
        );

        model.eval();
        Tensor test_sub = test.images.slice_rows(0, std::min(static_cast<size_t>(2000), test.images.rows));
        Tensor test_lbl = test.one_hot.slice_rows(0, std::min(static_cast<size_t>(2000), test.one_hot.rows));
        auto eval = benchlog::evaluate_classification_batched(
            model, test_sub, test_lbl, *loss_fn, EVAL_BATCH_SIZE
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
            LR,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            eval.elapsed_ms,
            eval.samples_per_sec
        );
        benchlog::log_confusion_and_class_metrics(logger, epoch + 1, "test", eval.preds, eval.truths, 10);
        bio.end_epoch(logger, epoch + 1);

        auto params = model.parameters();
        const double cls_norm = params.empty() ? 0.0 : params[0]->norm();
        logger.append_csv_row(
            "model_specific/vit/cls_embedding_stats.csv",
            {"run_id", "epoch", "mean_norm"},
            {logger.run_id(), std::to_string(epoch + 1), std::to_string(cls_norm)}
        );
        for (size_t block_idx = 0; block_idx < model.blocks.size(); ++block_idx) {
            const auto& block = model.blocks[block_idx];
            const Tensor& attn = block.mha.cached_attn_weights;
            const size_t batch = block.mha.cached_batch;
            const size_t heads = block.mha.num_heads;
            const size_t sl = block.mha.cached_seq_len;
            if (attn.rows == 0 || attn.cols == 0 || batch == 0 || heads == 0 || sl == 0) continue;

            for (size_t head = 0; head < heads; ++head) {
                for (size_t patch = 0; patch < sl; ++patch) {
                    double focus_sum = 0.0;
                    for (size_t b_ix = 0; b_ix < batch; ++b_ix) {
                        const size_t row = (b_ix * heads + head) * sl;
                        focus_sum += attn(row, patch);
                    }
                    const double attn_mean = focus_sum / static_cast<double>(batch);
                    logger.append_csv_row(
                        "model_specific/vit/patch_attention.csv",
                        {"run_id", "epoch", "block", "head", "patch_idx", "attn"},
                        {logger.run_id(),
                         std::to_string(epoch + 1),
                         std::to_string(block_idx),
                         std::to_string(head),
                         std::to_string(patch),
                         std::to_string(attn_mean)}
                    );
                }
            }
        }

        // Spatial attention summary: mean attention to each patch across all heads/blocks
        {
            const size_t num_patches = model.blocks.empty() ? 0 : model.blocks[0].mha.cached_seq_len;
            if (num_patches > 0) {
                std::vector<double> spatial_attn(num_patches, 0.0);
                int count = 0;
                for (size_t bi = 0; bi < model.blocks.size(); ++bi) {
                    const auto& block = model.blocks[bi];
                    const Tensor& attn = block.mha.cached_attn_weights;
                    const size_t batch = block.mha.cached_batch;
                    const size_t heads = block.mha.num_heads;
                    const size_t sl = block.mha.cached_seq_len;
                    if (attn.rows == 0 || batch == 0 || sl == 0) continue;
                    for (size_t h = 0; h < heads; ++h) {
                        for (size_t b_ix = 0; b_ix < batch; ++b_ix) {
                            const size_t row = (b_ix * heads + h) * sl;
                            for (size_t p = 0; p < sl; ++p) {
                                spatial_attn[p] += attn(row, p);
                            }
                            count++;
                        }
                    }
                }
                for (size_t p = 0; p < num_patches && count > 0; ++p) {
                    spatial_attn[p] /= static_cast<double>(count);
                    int row = (p > 0) ? (static_cast<int>(p) - 1) / 4 : -1;
                    int col = (p > 0) ? (static_cast<int>(p) - 1) % 4 : -1;
                    logger.append_csv_row(
                        "model_specific/vit/spatial_attention_map.csv",
                        {"run_id", "epoch", "patch_idx", "attn_weight", "grid_row", "grid_col"},
                        {logger.run_id(), std::to_string(epoch + 1), std::to_string(p),
                         std::to_string(spatial_attn[p]),
                         std::to_string(row), std::to_string(col)}
                    );
                }
            }
        }

        if ((epoch + 1) % 3 == 0 || epoch == EPOCHS - 1) {
            std::cout << "Epoch " << std::setw(2) << epoch + 1 << "/" << EPOCHS
                      << " | Loss: " << std::fixed << std::setprecision(4) << train_loss
                      << " | Test Acc: " << std::setprecision(2) << acc * 100 << "%" << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();

    model.eval();
    auto final_eval = benchlog::evaluate_classification_batched(
        model, test.images, test.one_hot, *loss_fn, EVAL_BATCH_SIZE, &logger, "test", true, true, 10
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
    std::cout << "\nViT Final Acc: " << std::fixed << std::setprecision(2) << final_acc * 100 << "%"
              << " (" << std::setprecision(1) << elapsed << "s)"
              << " | " << (final_acc >= 0.85 ? "PASS" : "FAIL") << std::endl;
    logger.write_manifest_end();
    return (final_acc >= 0.85) ? 0 : 1;
}
