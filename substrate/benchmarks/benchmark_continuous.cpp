#include "benchmarks/logging_utils.h"
#include "benchmarks/hpo_utils.h"
#include "core/data/dataloader.h"
#include "core/io/run_logger.h"
#include "core/losses/losses.h"
#include "core/metrics/metrics.h"
#include "core/online/continuous_runtime.h"
#include "models/hybrid/src/hybrid_models.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct EvalResult {
    double loss = std::numeric_limits<double>::quiet_NaN();
    double accuracy = std::numeric_limits<double>::quiet_NaN();
    double precision_macro = std::numeric_limits<double>::quiet_NaN();
    double recall_macro = std::numeric_limits<double>::quiet_NaN();
    double f1_macro = std::numeric_limits<double>::quiet_NaN();
    double eval_ms = 0.0;
    double samples_per_sec = 0.0;
    Tensor probs;
    Tensor labels;
    std::vector<int> preds;
    std::vector<int> truths;
};

EvalResult evaluate_model(TrainableModel& model,
                          const MNISTData& data,
                          std::size_t subset_size,
                          CrossEntropyLoss& loss_fn) {
    EvalResult out;

    const std::size_t n = std::min(subset_size, data.images.rows);
    if (n == 0) return out;

    Tensor x = data.images.slice_rows(0, n);
    Tensor y = data.one_hot.slice_rows(0, n);

    model.eval();
    auto eval_start = std::chrono::high_resolution_clock::now();
    Tensor logits = model.forward(x);
    out.probs = LossFunctions::softmax(logits);
    out.loss = loss_fn.forward(out.probs, y);
    out.accuracy = Metrics::accuracy_from_tensor(out.probs, y);
    auto eval_end = std::chrono::high_resolution_clock::now();

    out.eval_ms = std::chrono::duration<double, std::milli>(eval_end - eval_start).count();
    out.samples_per_sec = n / (out.eval_ms / 1000.0 + 1e-12);
    out.labels = y;

    benchlog::argmax_rows_to_vec(out.probs, out.preds);
    benchlog::argmax_rows_to_vec(y, out.truths);
    std::tie(out.precision_macro, out.recall_macro, out.f1_macro) =
        benchlog::macro_prf_from_preds(out.preds, out.truths, 10);
    return out;
}

std::string json_config(const ContinuousConfig& cfg,
                        double lr,
                        const std::string& mode,
                        const std::string& spec,
                        bool target_on_full_test) {
    std::ostringstream oss;
    oss << "{"
        << "\"mode\":\"" << mode << "\","
        << "\"spec\":\"" << spec << "\","
        << "\"cycles\":" << cfg.cycles << ","
        << "\"active_steps\":" << cfg.active_steps_per_cycle << ","
        << "\"sleep_steps\":" << cfg.sleep_steps_per_cycle << ","
        << "\"active_batch\":" << cfg.active_batch_size << ","
        << "\"sleep_batch\":" << cfg.sleep_batch_size << ","
        << "\"sleep_lr_scale\":" << cfg.sleep_lr_scale << ","
        << "\"sleep_replay_ratio\":" << cfg.sleep_replay_ratio << ","
        << "\"sleep_noise_std\":" << cfg.sleep_noise_std << ","
        << "\"replay_capacity\":" << cfg.replay_capacity << ","
        << "\"replay_seed_samples\":" << cfg.replay_seed_samples << ","
        << "\"eval_subset\":" << cfg.eval_subset << ","
        << "\"grad_clip_norm\":" << cfg.grad_clip_norm << ","
        << "\"target_accuracy\":" << cfg.target_accuracy << ","
        << "\"stop_on_target\":" << (cfg.stop_on_target ? "true" : "false") << ","
        << "\"target_on_full_test\":" << (target_on_full_test ? "true" : "false") << ","
        << "\"lr\":" << lr
        << "}";
    return oss.str();
}

double run_continuous_experiment(const std::string& family,
                                 const std::string& variant,
                                 const std::string& mode,
                                 const std::string& spec,
                                 std::unique_ptr<TrainableModel> model,
                                 const MNISTData& train,
                                 const MNISTData& test,
                                 ContinuousConfig cfg,
                                 double lr,
                                 int seed,
                                 double pass_threshold,
                                 bool target_on_full_test) {
    std::cout << "\n--- " << variant << " ---" << std::endl;
    std::cout << "System: " << model->id() << std::endl;

    RunLogger logger(
        family,
        variant,
        seed,
        "mnist-idx-v1",
        json_config(cfg, lr, mode, spec, target_on_full_test)
    );
    benchlog::BenchBioHarness bio(family, variant, seed);
    bio.attach(model->parameters(), model->gradients(), "continuous_param");

    Adam optimizer(lr);
    CrossEntropyLoss loss_fn;
    ContinuousRuntime runtime(cfg, seed);
    runtime.seed_replay(train.images.rows);

    std::size_t samples_to_target = 0;
    int target_cycle = -1;
    const auto wall_start = std::chrono::high_resolution_clock::now();

    double best_acc = 0.0;
    int completed_cycles = 0;

    const auto hooks = ContinuousHooks{
        nullptr,
        [&bio]() { bio.before_forward(); },
        [&bio](double loss, double effective_lr) {
            bio.after_backward(loss, std::numeric_limits<double>::quiet_NaN(), effective_lr);
        },
        [&bio]() { bio.after_optimizer_step(); },
        nullptr
    };

    for (int cycle = 1; cycle <= cfg.cycles; ++cycle) {
        bio.begin_epoch(cycle);
        CycleMetrics cm = runtime.run_cycle(*model, optimizer, loss_fn, train, cycle, hooks);
        EvalResult eval = evaluate_model(*model, test, cfg.eval_subset, loss_fn);

        completed_cycles = cycle;
        best_acc = std::max(best_acc, eval.accuracy);
        double target_metric_accuracy = eval.accuracy;
        if (target_on_full_test) {
            EvalResult eval_target = evaluate_model(*model, test, test.images.rows, loss_fn);
            target_metric_accuracy = eval_target.accuracy;
        }
        const bool reached_target = target_metric_accuracy >= cfg.target_accuracy;
        if (reached_target && target_cycle < 0) {
            target_cycle = cycle;
            samples_to_target = cm.cumulative_samples;
        }

        logger.log_epoch_metric(
            cycle,
            "active",
            cm.active.loss,
            cm.active.accuracy,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            lr,
            cm.active.grad_norm_mean,
            cm.active.grad_norm_max,
            cm.active.param_norm_mean,
            cm.active.param_norm_max,
            cm.active.phase_time_ms,
            cm.active.samples / (cm.active.phase_time_ms / 1000.0 + 1e-12)
        );
        logger.log_epoch_metric(
            cycle,
            "sleep",
            cm.sleep.loss,
            cm.sleep.accuracy,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            lr * cfg.sleep_lr_scale,
            cm.sleep.grad_norm_mean,
            cm.sleep.grad_norm_max,
            cm.sleep.param_norm_mean,
            cm.sleep.param_norm_max,
            cm.sleep.phase_time_ms,
            cm.sleep.samples / (cm.sleep.phase_time_ms / 1000.0 + 1e-12)
        );
        logger.log_epoch_metric(
            cycle,
            "test",
            eval.loss,
            eval.accuracy,
            eval.precision_macro,
            eval.recall_macro,
            eval.f1_macro,
            lr,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            eval.eval_ms,
            eval.samples_per_sec
        );
        benchlog::log_confusion_and_class_metrics(logger, cycle, "test", eval.preds, eval.truths, 10);
        bio.end_epoch(logger, cycle);

        logger.append_csv_row(
            "model_specific/" + family + "/continuous_phase_metrics.csv",
            {"run_id",
             "cycle",
             "phase",
             "steps",
             "samples",
             "loss",
             "accuracy",
             "lr_scale",
             "replay_fraction",
             "phase_time_ms",
             "replay_size",
             "cumulative_samples",
             "grad_norm_mean",
             "grad_norm_max",
             "param_norm_mean",
             "param_norm_max"},
            {logger.run_id(),
             std::to_string(cycle),
             "active",
             std::to_string(cm.active.steps),
             std::to_string(cm.active.samples),
             std::to_string(cm.active.loss),
             std::to_string(cm.active.accuracy),
             std::to_string(cm.active.lr_scale),
             std::to_string(cm.active.replay_fraction),
             std::to_string(cm.active.phase_time_ms),
             std::to_string(cm.replay_size),
             std::to_string(cm.cumulative_samples),
             std::to_string(cm.active.grad_norm_mean),
             std::to_string(cm.active.grad_norm_max),
             std::to_string(cm.active.param_norm_mean),
             std::to_string(cm.active.param_norm_max)}
        );
        logger.append_csv_row(
            "model_specific/" + family + "/continuous_phase_metrics.csv",
            {"run_id",
             "cycle",
             "phase",
             "steps",
             "samples",
             "loss",
             "accuracy",
             "lr_scale",
             "replay_fraction",
             "phase_time_ms",
             "replay_size",
             "cumulative_samples",
             "grad_norm_mean",
             "grad_norm_max",
             "param_norm_mean",
             "param_norm_max"},
            {logger.run_id(),
             std::to_string(cycle),
             "sleep",
             std::to_string(cm.sleep.steps),
             std::to_string(cm.sleep.samples),
             std::to_string(cm.sleep.loss),
             std::to_string(cm.sleep.accuracy),
             std::to_string(cm.sleep.lr_scale),
             std::to_string(cm.sleep.replay_fraction),
             std::to_string(cm.sleep.phase_time_ms),
             std::to_string(cm.replay_size),
             std::to_string(cm.cumulative_samples),
             std::to_string(cm.sleep.grad_norm_mean),
             std::to_string(cm.sleep.grad_norm_max),
             std::to_string(cm.sleep.param_norm_mean),
             std::to_string(cm.sleep.param_norm_max)}
        );

        const double samples_per_acc_point = cm.cumulative_samples / std::max(1e-6, best_acc);
        logger.append_csv_row(
            "model_specific/" + family + "/continuous_efficiency.csv",
            {"run_id",
             "cycle",
             "test_loss",
             "test_accuracy",
             "best_accuracy",
             "target_accuracy",
             "target_metric_accuracy",
             "reached_target",
             "cycle_time_ms",
             "active_loss",
             "sleep_loss",
             "active_accuracy",
             "sleep_accuracy",
             "replay_size",
             "cumulative_samples",
             "samples_per_accuracy_point"},
            {logger.run_id(),
             std::to_string(cycle),
             std::to_string(eval.loss),
             std::to_string(eval.accuracy),
             std::to_string(best_acc),
             std::to_string(cfg.target_accuracy),
             std::to_string(target_metric_accuracy),
             reached_target ? "1" : "0",
             std::to_string(cm.cycle_time_ms),
             std::to_string(cm.active.loss),
             std::to_string(cm.sleep.loss),
             std::to_string(cm.active.accuracy),
             std::to_string(cm.sleep.accuracy),
             std::to_string(cm.replay_size),
             std::to_string(cm.cumulative_samples),
             std::to_string(samples_per_acc_point)}
        );

        if (auto* hybrid_model = dynamic_cast<hybrid::HybridSystem*>(model.get())) {
            const auto names = hybrid_model->expert_names();
            const auto strengths = hybrid_model->expert_fusion_strength();
            const std::size_t n_eval = std::min(cfg.eval_subset, test.images.rows);
            Tensor hx = test.images.slice_rows(0, n_eval);
            Tensor hy = test.one_hot.slice_rows(0, n_eval);
            const auto expert_acc = hybrid_model->expert_accuracy(hx, hy);

            const std::size_t n = std::min(names.size(), std::min(strengths.size(), expert_acc.size()));
            for (std::size_t i = 0; i < n; ++i) {
                logger.append_csv_row(
                    "model_specific/" + family + "/hybrid_expert_metrics.csv",
                    {"run_id", "cycle", "expert", "fusion_strength", "expert_accuracy", "test_accuracy", "best_accuracy"},
                    {logger.run_id(),
                     std::to_string(cycle),
                     names[i],
                     std::to_string(strengths[i]),
                     std::to_string(expert_acc[i]),
                     std::to_string(eval.accuracy),
                     std::to_string(best_acc)}
                );
            }
        }

        model->train();

        std::cout << "Cycle " << std::setw(2) << cycle << "/" << cfg.cycles
                  << " | active acc: " << std::fixed << std::setprecision(3) << cm.active.accuracy
                  << " | sleep acc: " << cm.sleep.accuracy
                  << " | test acc: " << eval.accuracy
                  << " | replay: " << cm.replay_size
                  << std::endl;

        if (cfg.stop_on_target &&
            cycle >= cfg.min_cycles_before_stop &&
            reached_target) {
            break;
        }
    }

    // Final full-test evaluation for deployment diagnostics.
    EvalResult final_eval = evaluate_model(*model, test, test.images.rows, loss_fn);
    const double per_sample_ms = final_eval.eval_ms / static_cast<double>(test.images.rows);
    benchlog::log_inference_from_probs(
        logger,
        "test",
        final_eval.probs,
        final_eval.preds,
        final_eval.truths,
        per_sample_ms,
        static_cast<int>(test.images.rows)
    );
    benchlog::log_calibration_from_probs(
        logger,
        "test",
        final_eval.probs,
        final_eval.preds,
        final_eval.truths,
        10
    );
    logger.log_system_metric(
        RunLogger::utc_now_iso8601(),
        test.images.rows / (final_eval.eval_ms / 1000.0 + 1e-12),
        per_sample_ms,
        per_sample_ms,
        per_sample_ms,
        0.0,
        0.0
    );

    const auto wall_end = std::chrono::high_resolution_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
    logger.append_csv_row(
        "model_specific/" + family + "/continuous_summary.csv",
        {"run_id",
         "variant",
         "cycles_completed",
         "final_accuracy",
         "best_accuracy",
         "target_accuracy",
         "target_cycle",
         "samples_to_target",
         "cumulative_samples",
         "wall_time_ms"},
        {logger.run_id(),
         variant,
         std::to_string(completed_cycles),
         std::to_string(final_eval.accuracy),
         std::to_string(best_acc),
         std::to_string(cfg.target_accuracy),
         std::to_string(target_cycle),
         std::to_string(samples_to_target),
         std::to_string(runtime.cumulative_samples()),
         std::to_string(wall_ms)}
    );
    logger.write_manifest_end();

    std::cout << "Final full-test accuracy: " << std::fixed << std::setprecision(4) << final_eval.accuracy
              << " | best cycle accuracy: " << best_acc
              << " | samples seen: " << runtime.cumulative_samples()
              << " | " << (final_eval.accuracy >= pass_threshold ? "PASS" : "FAIL")
              << std::endl;
    return final_eval.accuracy;
}

} // namespace

int main() {
    std::cout << "=== Continuous + Hybrid Benchmark (MNIST) ===" << std::endl;
    const int seed = hpo::env_int("ONLINE_SEED", 42);
    std::mt19937 rng(seed);

    auto train = MNISTLoader::load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    auto test = MNISTLoader::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    ContinuousConfig cfg;
    cfg.cycles = hpo::env_int("ONLINE_CYCLES", cfg.cycles);
    cfg.active_steps_per_cycle = hpo::env_int("ONLINE_ACTIVE_STEPS", cfg.active_steps_per_cycle);
    cfg.sleep_steps_per_cycle = hpo::env_int("ONLINE_SLEEP_STEPS", cfg.sleep_steps_per_cycle);
    cfg.active_batch_size = hpo::env_size("ONLINE_ACTIVE_BATCH", cfg.active_batch_size);
    cfg.sleep_batch_size = hpo::env_size("ONLINE_SLEEP_BATCH", cfg.sleep_batch_size);
    cfg.sleep_lr_scale = hpo::env_double("ONLINE_SLEEP_LR_SCALE", cfg.sleep_lr_scale);
    cfg.sleep_replay_ratio = hpo::env_double("ONLINE_SLEEP_REPLAY_RATIO", cfg.sleep_replay_ratio);
    cfg.sleep_noise_std = hpo::env_double("ONLINE_SLEEP_NOISE_STD", cfg.sleep_noise_std);
    cfg.replay_capacity = hpo::env_size("ONLINE_REPLAY_CAPACITY", cfg.replay_capacity);
    cfg.replay_seed_samples = hpo::env_size("ONLINE_REPLAY_SEED", cfg.replay_seed_samples);
    cfg.eval_subset = hpo::env_size("ONLINE_EVAL_SUBSET", cfg.eval_subset);
    cfg.grad_clip_norm = hpo::env_double("ONLINE_GRAD_CLIP_NORM", cfg.grad_clip_norm);
    cfg.target_accuracy = hpo::env_double("ONLINE_TARGET_ACC", cfg.target_accuracy);
    cfg.stop_on_target = hpo::env_bool("ONLINE_STOP_ON_TARGET", cfg.stop_on_target);
    cfg.min_cycles_before_stop = hpo::env_int("ONLINE_MIN_CYCLES_BEFORE_STOP", cfg.min_cycles_before_stop);

    const double lr = hpo::env_double("ONLINE_LR", 1e-3);
    const double pass_threshold = hpo::env_double("ONLINE_PASS_ACC", 0.75);
    const bool target_on_full_test = hpo::env_bool("ONLINE_TARGET_ON_FULL_TEST", false);
    const std::string mode = hpo::env_string("ONLINE_MODE", "single"); // single|hybrid|suite
    const std::string single_model_name = hpo::env_string("ONLINE_MODEL", "cnn");
    const std::string hybrid_spec = hpo::env_string("ONLINE_HYBRID_SPEC", "cnn,transformer,mlp");
    const std::string suite_models = hpo::env_string("ONLINE_SUITE_MODELS", "mlp,cnn,gru");

    std::cout << "Mode: " << mode
              << " | cycles: " << cfg.cycles
              << " | active/sleep steps: " << cfg.active_steps_per_cycle << "/" << cfg.sleep_steps_per_cycle
              << " | target eval: " << (target_on_full_test ? "full-test" : "subset")
              << " | lr: " << lr
              << std::endl;
    std::cout << "Config: batch=" << cfg.active_batch_size << "/" << cfg.sleep_batch_size
              << " sleep_lr_scale=" << cfg.sleep_lr_scale
              << " replay=" << cfg.replay_capacity << " seed_samples=" << cfg.replay_seed_samples
              << " noise=" << cfg.sleep_noise_std
              << " grad_clip=" << cfg.grad_clip_norm
              << " pass_acc=" << pass_threshold
              << " model=" << single_model_name
              << " hybrid=" << hybrid_spec
              << std::endl;

    if (mode != "single" && mode != "hybrid" && mode != "suite") {
        std::cerr << "Invalid ONLINE_MODE='" << mode << "'. Valid values: single,hybrid,suite" << std::endl;
        return 2;
    }

    double min_acc = 1.0;
    bool ran_any = false;

    if (mode == "single" || mode == "suite") {
        std::vector<std::string> single_specs;
        if (mode == "single") {
            single_specs = {single_model_name};
        } else {
            single_specs = hybrid::parse_model_spec(suite_models);
            if (single_specs.empty()) single_specs = {"cnn"};
        }

        for (const auto& name : single_specs) {
            auto model = hybrid::build_model_by_name(name, rng);
            const std::string variant = "continuous_" + name;
            ran_any = true;
            min_acc = std::min(
                min_acc,
                run_continuous_experiment(
                    "continuous",
                    variant,
                    mode,
                    name,
                    std::move(model),
                    train,
                    test,
                    cfg,
                    lr,
                    seed,
                    pass_threshold,
                    target_on_full_test
                )
            );
        }
    }

    if (mode == "hybrid" || mode == "suite") {
        std::vector<std::string> names = hybrid::parse_model_spec(hybrid_spec);
        if (names.empty()) {
            throw std::runtime_error("ONLINE_HYBRID_SPEC produced no models");
        }

        auto named_models = hybrid::build_named_models(names, rng);
        auto hybrid_model = std::make_unique<hybrid::HybridSystem>(std::move(named_models), rng);

        std::string variant = "hybrid_";
        for (std::size_t i = 0; i < names.size(); ++i) {
            if (i) variant += "_";
            variant += names[i];
        }

        ran_any = true;
        min_acc = std::min(
            min_acc,
            run_continuous_experiment(
                "hybrid",
                variant,
                mode,
                hybrid_spec,
                std::move(hybrid_model),
                train,
                test,
                cfg,
                lr,
                seed,
                pass_threshold,
                target_on_full_test
            )
        );
    }

    std::cout << "\n=== Continuous + Hybrid Benchmark Complete ===" << std::endl;
    if (!ran_any) {
        std::cerr << "No continuous experiment executed." << std::endl;
        return 2;
    }
    return (min_acc >= pass_threshold) ? 0 : 1;
}
