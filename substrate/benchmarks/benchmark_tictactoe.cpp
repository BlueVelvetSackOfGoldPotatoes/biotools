#include "core/benchmark/bit_bridge_model.h"
#include "core/benchmark/discrete_q_runtime.h"
#include "core/benchmark/tictactoe_env.h"
#include "core/io/run_logger.h"
#include "models/hybrid/src/hybrid_models.h"
#include "benchmarks/hpo_utils.h"
#include "benchmarks/logging_utils.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string trim(std::string s) {
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!s.empty() && is_space(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && is_space(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

std::string sanitize_variant_token(std::string s) {
    for (char& c : s) {
        const bool ok = (c >= 'a' && c <= 'z') ||
                        (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9');
        if (!ok) c = '_';
    }
    return s;
}

std::string encode_active_bit_indices(const std::vector<uint8_t>& bits) {
    std::ostringstream oss;
    bool first = true;
    for (std::size_t i = 0; i < bits.size(); ++i) {
        if (bits[i] == 0) continue;
        if (!first) oss << ";";
        oss << i;
        first = false;
    }
    return oss.str();
}

double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    if (q <= 0.0) return *std::min_element(values.begin(), values.end());
    if (q >= 1.0) return *std::max_element(values.begin(), values.end());

    std::sort(values.begin(), values.end());
    const double rank = q * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(rank));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(rank));
    if (lo == hi) return values[lo];
    const double t = rank - static_cast<double>(lo);
    return values[lo] * (1.0 - t) + values[hi] * t;
}

void log_calibration_from_inference(RunLogger& logger,
                                    const std::vector<benchmark::DiscreteInferenceSample>& rows,
                                    int bins = 10) {
    if (bins <= 0) return;

    std::vector<int> counts(static_cast<std::size_t>(bins), 0);
    std::vector<double> sum_conf(static_cast<std::size_t>(bins), 0.0);
    std::vector<double> sum_acc(static_cast<std::size_t>(bins), 0.0);

    for (const auto& row : rows) {
        const double conf = std::clamp(row.confidence, 0.0, 1.0);
        const int bin = std::min(bins - 1, static_cast<int>(conf * bins));
        counts[static_cast<std::size_t>(bin)]++;
        sum_conf[static_cast<std::size_t>(bin)] += conf;
        sum_acc[static_cast<std::size_t>(bin)] += static_cast<double>(row.is_correct);
    }

    const double n = std::max(1.0, static_cast<double>(rows.size()));
    for (int b = 0; b < bins; ++b) {
        const double low = static_cast<double>(b) / static_cast<double>(bins);
        const double high = static_cast<double>(b + 1) / static_cast<double>(bins);
        const int cnt = counts[static_cast<std::size_t>(b)];
        const double avg_conf = cnt ? sum_conf[static_cast<std::size_t>(b)] / static_cast<double>(cnt) : 0.0;
        const double empirical_acc = cnt ? sum_acc[static_cast<std::size_t>(b)] / static_cast<double>(cnt) : 0.0;
        const double ece_contrib = (static_cast<double>(cnt) / n) * std::abs(avg_conf - empirical_acc);
        logger.log_calibration_bin("test", b, low, high, cnt, avg_conf, empirical_acc, ece_contrib);
    }
}

std::unique_ptr<TrainableModel> build_backbone_model(const std::string& model_spec, std::mt19937& rng) {
    const std::string spec = to_lower(trim(model_spec));
    if (starts_with(spec, "hybrid:")) {
        const std::string sub = spec.substr(7);
        const std::vector<std::string> names = hybrid::parse_model_spec(sub);
        if (names.empty()) {
            throw std::runtime_error("TICTACTOE_MODEL=hybrid:<spec> requires at least one model name");
        }
        auto experts = hybrid::build_named_models(names, rng);
        return std::make_unique<hybrid::HybridSystem>(std::move(experts), rng);
    }

    return hybrid::build_model_by_name(spec, rng);
}

std::string model_family_from_spec(const std::string& model_spec) {
    const std::string spec = to_lower(trim(model_spec));
    if (starts_with(spec, "hybrid:")) return "hybrid";
    return spec;
}

std::string params_json(const std::string& model_spec,
                        const benchmark::BitBridgeConfig& bridge_cfg,
                        const benchmark::DiscreteQConfig& cfg,
                        double lr,
                        double opp_noise_train,
                        double opp_noise_eval,
                        int trace_every,
                        std::size_t trace_max_rows) {
    std::ostringstream ss;
    ss << "{"
       << "\"algorithm\":\"deep_q\""
       << ",\"adapter\":\"bit_bridge\""
       << ",\"benchmark\":\"tictactoe\""
       << ",\"model\":\"" << model_spec << "\""
       << ",\"episodes\":" << cfg.episodes
       << ",\"eval_every\":" << cfg.eval_every
       << ",\"eval_episodes\":" << cfg.eval_episodes
       << ",\"max_steps\":" << cfg.max_steps_per_episode
       << ",\"gamma\":" << cfg.gamma
       << ",\"epsilon_start\":" << cfg.epsilon_start
       << ",\"epsilon_min\":" << cfg.epsilon_min
       << ",\"epsilon_decay\":" << cfg.epsilon_decay
       << ",\"lr\":" << lr
       << ",\"bridge_input_bits\":" << bridge_cfg.input_bits
       << ",\"bridge_width\":" << bridge_cfg.bridge_width
       << ",\"bridge_backbone_input\":" << bridge_cfg.backbone_input_dim
       << ",\"bridge_actions\":" << bridge_cfg.action_dim
       << ",\"opp_noise_train\":" << opp_noise_train
       << ",\"opp_noise_eval\":" << opp_noise_eval
       << ",\"trace_every\":" << trace_every
       << ",\"trace_max_rows\":" << trace_max_rows
       << "}";
    return ss.str();
}

std::string model_variant_from_spec(const std::string& model_spec) {
    // Model variant must describe model/runtime setup only, not benchmark identity.
    return "bit_bridge_" + sanitize_variant_token(to_lower(model_spec)) + "_deep_q";
}

} // namespace

int main() {
    std::cout << "=== TicTacToe Benchmark (Model-Agnostic, Bit-Bridge) ===" << std::endl;

    const int seed = hpo::env_int("TICTACTOE_SEED", 42);
    std::mt19937 rng(static_cast<unsigned int>(seed));

    const std::string model_spec = hpo::env_string("TICTACTOE_MODEL", "mlp");
    const std::string legacy_algo = to_lower(hpo::env_string("TICTACTOE_ALGORITHM", "deep_q"));
    if (legacy_algo != "deep_q" && legacy_algo != "q_learning" && legacy_algo != "sarsa") {
        std::cerr << "Unsupported TICTACTOE_ALGORITHM='" << legacy_algo
                  << "'. Supported: deep_q, q_learning, sarsa (q_learning/sarsa map to deep_q runtime)."
                  << std::endl;
        return 1;
    }

    benchmark::DiscreteQConfig runtime_cfg;
    runtime_cfg.episodes = std::max(100, hpo::env_int("TICTACTOE_EPISODES", runtime_cfg.episodes));
    runtime_cfg.eval_every = std::max(10, hpo::env_int("TICTACTOE_EVAL_EVERY", runtime_cfg.eval_every));
    runtime_cfg.eval_episodes = std::max(10, hpo::env_int("TICTACTOE_EVAL_EPISODES", runtime_cfg.eval_episodes));
    runtime_cfg.max_steps_per_episode = std::max(4, hpo::env_int("TICTACTOE_MAX_STEPS", runtime_cfg.max_steps_per_episode));
    runtime_cfg.gamma = hpo::env_double("TICTACTOE_GAMMA", runtime_cfg.gamma);
    runtime_cfg.epsilon_start = hpo::env_double("TICTACTOE_EPS_START", runtime_cfg.epsilon_start);
    runtime_cfg.epsilon_min = hpo::env_double("TICTACTOE_EPS_MIN", runtime_cfg.epsilon_min);
    runtime_cfg.epsilon_decay = hpo::env_double("TICTACTOE_EPS_DECAY", runtime_cfg.epsilon_decay);
    runtime_cfg.grad_clip_norm = hpo::env_double("TICTACTOE_GRAD_CLIP", runtime_cfg.grad_clip_norm);
    runtime_cfg.batch_log_every = std::max<std::size_t>(1, hpo::env_size("BATCH_LOG_EVERY", runtime_cfg.batch_log_every));

    const double lr = hpo::env_double("TICTACTOE_LR", 0.0015);
    const double opp_noise_train = hpo::env_double("TICTACTOE_OPP_NOISE_TRAIN", 0.15);
    const double opp_noise_eval = hpo::env_double("TICTACTOE_OPP_NOISE_EVAL", 0.0);
    const int trace_every = std::max(1, hpo::env_int("TICTACTOE_TRACE_EVERY", 1));
    const std::size_t trace_max_rows = hpo::env_size("TICTACTOE_TRACE_MAX_ROWS", 250000);

    benchmark::TicTacToeConfig env_cfg_train;
    env_cfg_train.opponent_noise = opp_noise_train;
    benchmark::TicTacToeConfig env_cfg_eval;
    env_cfg_eval.opponent_noise = opp_noise_eval;

    benchmark::TicTacToeEnv env_train(env_cfg_train);
    benchmark::TicTacToeEnv env_eval(env_cfg_eval);

    std::unique_ptr<TrainableModel> backbone;
    try {
        backbone = build_backbone_model(model_spec, rng);
    } catch (const std::exception& e) {
        std::cerr << "Failed to build model for TICTACTOE_MODEL='" << model_spec << "': " << e.what() << std::endl;
        return 1;
    }

    benchmark::BitBridgeConfig bridge_cfg;
    bridge_cfg.input_bits = env_train.observation_bits();
    bridge_cfg.bridge_width = std::max<std::size_t>(64, hpo::env_size("TICTACTOE_BRIDGE_WIDTH", 784));
    bridge_cfg.backbone_input_dim = std::max<std::size_t>(128, hpo::env_size("TICTACTOE_BACKBONE_INPUT", 784));
    bridge_cfg.backbone_output_dim = std::max<std::size_t>(8, hpo::env_size("TICTACTOE_BACKBONE_LOGITS", 10));
    bridge_cfg.action_dim = static_cast<std::size_t>(env_train.num_actions());

    benchmark::BitBridgeModel model(std::move(backbone), bridge_cfg, rng);
    Adam optimizer(lr, 0.9, 0.999, 1e-8, 0.0);
    benchmark::DiscreteQRuntime runtime(runtime_cfg, seed);

    const std::string family = model_family_from_spec(model_spec);
    const std::string variant = model_variant_from_spec(model_spec);

    RunLogger logger(
        family,
        variant,
        seed,
        "tictactoe-v2",
        params_json(model_spec, bridge_cfg, runtime_cfg, lr, opp_noise_train, opp_noise_eval, trace_every, trace_max_rows),
        env_train.benchmark_id(),
        env_train.benchmark_name(),
        env_train.task_type()
    );

    benchlog::BenchBioHarness bio(family, variant, seed);
    bio.attach(model.parameters(), model.gradients(), family + "_param");
    int bio_epoch_open = 0;

    const std::string model_specific_policy = "model_specific/" + family + "/tictactoe_policy_metrics.csv";
    const std::string model_specific_deploy = "model_specific/" + family + "/tictactoe_deployment_summary.csv";
    const std::string model_specific_training = "model_specific/" + family + "/tictactoe_training_tactics.csv";
    const std::string model_specific_game_trace = "model_specific/" + family + "/tictactoe_game_trace.csv";
    const std::string bridge_diag = "model_specific/" + family + "/benchmark_bridge_diagnostics.csv";
    std::size_t trace_rows_written = 0;

    logger.append_csv_row(
        bridge_diag,
        {"run_id", "benchmark_id", "model_spec", "bridge_input_bits", "bridge_width", "backbone_input", "backbone_logits", "action_dim"},
        {logger.run_id(), env_train.benchmark_id(), model_spec,
         std::to_string(bridge_cfg.input_bits), std::to_string(bridge_cfg.bridge_width),
         std::to_string(bridge_cfg.backbone_input_dim), std::to_string(bridge_cfg.backbone_output_dim),
         std::to_string(bridge_cfg.action_dim)}
    );

    benchmark::DiscreteBatchMetrics latest_batch;
    bool have_latest_batch = false;

    auto on_batch = [&](const benchmark::DiscreteBatchMetrics& bm) {
        have_latest_batch = true;
        latest_batch = bm;

        double proxy_acc = bm.action_match_ema;
        if (!std::isfinite(proxy_acc)) {
            proxy_acc = std::clamp((bm.reward_ema + 1.0) * 0.5, 0.0, 1.0);
        }

        logger.append_csv_row(
            "learning/batch_metrics.csv",
            {"run_id", "model_family", "model_variant", "epoch", "batch", "total_batches", "global_step", "split", "loss", "accuracy", "lr"},
            {logger.run_id(), family, variant,
             std::to_string(bm.episode), std::to_string(bm.episode), std::to_string(runtime_cfg.episodes),
             std::to_string(bm.global_step), "train_batch", std::to_string(bm.q_loss_ema),
             std::to_string(proxy_acc), std::to_string(lr)}
        );
        logger.append_csv_row(
            model_specific_training,
            {"run_id",
             "episode",
             "global_step",
             "epsilon",
             "reward_ema",
             "q_loss_ema",
             "td_error_ema",
             "action_match_ema",
             "missed_win_rate_ema",
             "missed_block_rate_ema"},
            {logger.run_id(),
             std::to_string(bm.episode),
             std::to_string(bm.global_step),
             std::to_string(bm.epsilon),
             std::to_string(bm.reward_ema),
             std::to_string(bm.q_loss_ema),
             std::to_string(bm.td_error_ema),
             std::to_string(bm.action_match_ema),
             std::to_string(bm.missed_win_rate_ema),
             std::to_string(bm.missed_block_rate_ema)}
        );
    };

    auto on_eval = [&](const benchmark::DiscreteEvalMetrics& em) {
        const auto [precision_macro, recall_macro, f1_macro] =
            benchlog::macro_prf_from_preds(em.preds, em.truths, env_train.num_actions());

        double train_loss = std::numeric_limits<double>::quiet_NaN();
        double train_acc = std::numeric_limits<double>::quiet_NaN();
        if (have_latest_batch) {
            train_loss = latest_batch.q_loss_ema;
            train_acc = latest_batch.action_match_ema;
            if (!std::isfinite(train_acc)) {
                train_acc = std::clamp((latest_batch.reward_ema + 1.0) * 0.5, 0.0, 1.0);
            }
        }

        logger.log_epoch_metric(
            em.episode,
            "train",
            train_loss,
            train_acc,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            lr,
            latest_batch.td_error_ema,
            latest_batch.td_error_ema,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            0.0,
            0.0
        );

        logger.log_epoch_metric(
            em.episode,
            "test",
            std::max(0.0, 1.0 - em.non_loss_rate),
            em.non_loss_rate,
            precision_macro,
            recall_macro,
            f1_macro,
            lr,
            latest_batch.td_error_ema,
            latest_batch.td_error_ema,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            em.eval_time_ms,
            em.moves / (em.eval_time_ms / 1000.0 + 1e-12)
        );

        logger.append_csv_row(
            model_specific_policy,
            {"run_id",
             "episode",
             "wins",
             "draws",
             "losses",
             "win_rate",
             "draw_rate",
             "loss_rate",
             "non_loss_rate",
             "invalid_rate",
             "action_match_rate",
             "win_opportunities",
             "missed_wins",
             "win_capture_rate",
             "missed_win_rate",
             "block_opportunities",
             "missed_blocks",
             "block_capture_rate",
             "missed_block_rate",
             "avg_plies",
             "avg_latency_ms"},
            {logger.run_id(), std::to_string(em.episode),
             std::to_string(em.wins), std::to_string(em.draws), std::to_string(em.losses),
             std::to_string(em.win_rate), std::to_string(em.draw_rate), std::to_string(em.loss_rate),
             std::to_string(em.non_loss_rate), std::to_string(em.invalid_rate),
             std::to_string(em.action_match_rate),
             std::to_string(em.win_opportunities), std::to_string(em.missed_wins),
             std::to_string(em.win_capture_rate), std::to_string(em.missed_win_rate),
             std::to_string(em.block_opportunities), std::to_string(em.missed_blocks),
             std::to_string(em.block_capture_rate), std::to_string(em.missed_block_rate),
             std::to_string(em.avg_plies), std::to_string(em.avg_latency_ms)}
        );

        std::cout << "Episode " << em.episode << "/" << runtime_cfg.episodes
                  << " | non-loss=" << std::fixed << std::setprecision(2) << (em.non_loss_rate * 100.0) << "%"
                  << " win=" << (em.win_rate * 100.0) << "%"
                  << " draw=" << (em.draw_rate * 100.0) << "%"
                  << " loss=" << (em.loss_rate * 100.0) << "%"
                  << " missed-win=" << (std::isfinite(em.missed_win_rate) ? em.missed_win_rate * 100.0 : 0.0) << "%"
                  << " missed-block=" << (std::isfinite(em.missed_block_rate) ? em.missed_block_rate * 100.0 : 0.0) << "%"
                  << std::endl;
    };

    benchmark::DiscreteQRuntime::TrainHooks hooks;
    hooks.on_episode_begin = [&](int episode) {
        if (bio_epoch_open == 0) {
            bio.begin_epoch(episode);
            bio_epoch_open = episode;
        }
    };
    hooks.before_forward = [&]() { bio.before_forward(); };
    hooks.after_backward = [&](double loss, double reward) { bio.after_backward(loss, reward, lr); };
    hooks.after_optimizer_step = [&]() { bio.after_optimizer_step(); };
    hooks.on_transition = [&](const benchmark::DiscreteTransitionTrace& tr) {
        if (trace_every > 1 && (tr.global_step % trace_every != 0)) return;
        if (trace_max_rows > 0 && trace_rows_written >= trace_max_rows) return;
        logger.append_csv_row(
            model_specific_game_trace,
            {"run_id",
             "benchmark_id",
             "episode",
             "step",
             "global_step",
             "elapsed_ms",
             "epsilon",
             "action",
             "teacher_action",
             "immediate_win_action",
             "immediate_block_action",
             "reward",
             "done",
             "outcome",
             "obs_bits",
             "obs_pre_active",
             "obs_post_active"},
            {logger.run_id(),
             env_train.benchmark_id(),
             std::to_string(tr.episode),
             std::to_string(tr.step),
             std::to_string(tr.global_step),
             std::to_string(tr.elapsed_ms),
             std::to_string(tr.epsilon),
             std::to_string(tr.action),
             std::to_string(tr.teacher_action),
             std::to_string(tr.immediate_win_action),
             std::to_string(tr.immediate_block_action),
             std::to_string(tr.reward),
             tr.done ? "1" : "0",
             std::to_string(tr.outcome),
             std::to_string(tr.obs_after.size()),
             encode_active_bit_indices(tr.obs_before),
             encode_active_bit_indices(tr.obs_after)}
        );
        trace_rows_written++;
    };
    hooks.on_episode_end = [&](int episode) {
        const bool boundary = (episode % runtime_cfg.eval_every == 0) || (episode == runtime_cfg.episodes);
        if (!boundary) return;
        bio.end_epoch(logger, episode);
        if (episode < runtime_cfg.episodes) {
            bio.begin_epoch(episode + 1);
            bio_epoch_open = episode + 1;
        } else {
            bio_epoch_open = 0;
        }
    };

    const benchmark::DiscreteEvalMetrics final_eval =
        runtime.train(env_train, env_eval, model, optimizer, on_batch, on_eval, hooks);

    benchlog::log_confusion_and_class_metrics(
        logger,
        final_eval.episode,
        "test",
        final_eval.preds,
        final_eval.truths,
        env_train.num_actions()
    );

    for (const auto& row : final_eval.inference) {
        logger.log_inference_metric(
            "test",
            row.sample_id,
            row.true_action,
            row.pred_action,
            row.confidence,
            row.entropy,
            row.top2_margin,
            row.latency_ms,
            1,
            row.is_correct
        );
    }
    log_calibration_from_inference(logger, final_eval.inference, 10);

    std::vector<double> latencies;
    latencies.reserve(final_eval.inference.size());
    for (const auto& row : final_eval.inference) latencies.push_back(row.latency_ms);
    const double p50 = percentile(latencies, 0.50);
    const double p95 = percentile(latencies, 0.95);
    const double p99 = percentile(latencies, 0.99);

    logger.log_system_metric(
        RunLogger::utc_now_iso8601(),
        final_eval.moves / (final_eval.eval_time_ms / 1000.0 + 1e-12),
        p50,
        p95,
        p99,
        0.0,
        0.0
    );

    logger.append_csv_row(
        model_specific_deploy,
        {"run_id",
         "episode",
         "non_loss_rate",
         "win_rate",
         "draw_rate",
         "loss_rate",
         "action_match_rate",
         "invalid_rate",
         "win_opportunities",
         "missed_wins",
         "win_capture_rate",
         "missed_win_rate",
         "block_opportunities",
         "missed_blocks",
         "block_capture_rate",
         "missed_block_rate",
         "moves",
         "p50_ms",
         "p95_ms",
         "p99_ms"},
        {logger.run_id(), std::to_string(final_eval.episode),
         std::to_string(final_eval.non_loss_rate), std::to_string(final_eval.win_rate),
         std::to_string(final_eval.draw_rate), std::to_string(final_eval.loss_rate),
         std::to_string(final_eval.action_match_rate), std::to_string(final_eval.invalid_rate),
         std::to_string(final_eval.win_opportunities), std::to_string(final_eval.missed_wins),
         std::to_string(final_eval.win_capture_rate), std::to_string(final_eval.missed_win_rate),
         std::to_string(final_eval.block_opportunities), std::to_string(final_eval.missed_blocks),
         std::to_string(final_eval.block_capture_rate), std::to_string(final_eval.missed_block_rate),
         std::to_string(final_eval.moves), std::to_string(p50), std::to_string(p95), std::to_string(p99)}
    );

    logger.write_manifest_end();

    std::cout << "Final non-loss rate: " << std::fixed << std::setprecision(2)
              << (final_eval.non_loss_rate * 100.0) << "%"
              << " | win=" << (final_eval.win_rate * 100.0) << "%"
              << " | draw=" << (final_eval.draw_rate * 100.0) << "%"
              << " | missed-win=" << (std::isfinite(final_eval.missed_win_rate) ? final_eval.missed_win_rate * 100.0 : 0.0) << "%"
              << " | missed-block=" << (std::isfinite(final_eval.missed_block_rate) ? final_eval.missed_block_rate * 100.0 : 0.0) << "%"
              << std::endl;

    return 0;
}
