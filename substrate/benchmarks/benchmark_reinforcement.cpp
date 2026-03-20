#include "core/data/dataloader.h"
#include "core/io/run_logger.h"
#include "core/metrics/metrics.h"
#include "models/reinforcement/src/rl_models.h"
#include "benchmarks/logging_utils.h"
#include "benchmarks/hpo_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kNumActions = 10;
constexpr int kTabularStates = 256; // 4 quantized quadrant means -> 4^4

// Intentionally lossy compression of a 28x28 image into one of 256 discrete states.
// Each 14x14 quadrant's mean intensity is quantized into 4 bins (25% threshold each).
// This produces 4^4 = 256 states — coarse enough for tabular Q-learning and SARSA
// to converge in reasonable time, yet discriminative enough for >25% accuracy.
int bucket_image_row(const Tensor& row) {
    auto qmean = [&](int r0, int r1, int c0, int c1) {
        double s = 0.0;
        int n = 0;
        for (int r = r0; r < r1; ++r) {
            for (int c = c0; c < c1; ++c) {
                s += row(0, static_cast<size_t>(r * 28 + c));
                n++;
            }
        }
        return s / static_cast<double>(n);
    };
    auto qbin = [](double v) {
        int b = static_cast<int>(v * 4.0);
        if (b < 0) b = 0;
        if (b > 3) b = 3;
        return b;
    };

    const int b0 = qbin(qmean(0, 14, 0, 14));
    const int b1 = qbin(qmean(0, 14, 14, 28));
    const int b2 = qbin(qmean(14, 28, 0, 14));
    const int b3 = qbin(qmean(14, 28, 14, 28));
    return b0 + 4 * b1 + 16 * b2 + 64 * b3;
}

std::vector<int> precompute_buckets(const Tensor& images) {
    std::vector<int> out(images.rows, 0);
    for (size_t i = 0; i < images.rows; ++i) {
        out[i] = bucket_image_row(images.row(i));
    }
    return out;
}

double accuracy_from_preds(const std::vector<int>& preds, const std::vector<int>& labels) {
    return Metrics::accuracy(preds, labels);
}

void log_final_common(RunLogger& logger,
                      int epoch,
                      const std::vector<int>& preds,
                      const std::vector<int>& truths,
                      double infer_ms) {
    auto [precision_macro, recall_macro, f1_macro] = benchlog::macro_prf_from_preds(preds, truths, kNumActions);
    const double acc = accuracy_from_preds(preds, truths);
    logger.log_epoch_metric(
        epoch,
        "test",
        std::numeric_limits<double>::quiet_NaN(),
        acc,
        precision_macro,
        recall_macro,
        f1_macro,
        0.0,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        infer_ms,
        truths.size() / (infer_ms / 1000.0 + 1e-12)
    );
    benchlog::log_confusion_and_class_metrics(logger, epoch, "test", preds, truths, kNumActions);

    const double per_sample_ms = infer_ms / static_cast<double>(truths.size());
    benchlog::log_inference_from_preds(logger, "test", preds, truths, per_sample_ms, static_cast<int>(truths.size()));
    benchlog::log_calibration_from_preds(logger, "test", preds, truths, 10);
    logger.log_system_metric(
        RunLogger::utc_now_iso8601(),
        truths.size() / (infer_ms / 1000.0 + 1e-12),
        per_sample_ms,
        per_sample_ms,
        per_sample_ms,
        0.0,
        0.0
    );
}

std::vector<int> eval_tabular(const std::vector<int>& buckets,
                              std::vector<int> const& labels,
                              const std::function<int(int)>& act_fn) {
    std::vector<int> preds(labels.size(), 0);
    for (size_t i = 0; i < labels.size(); ++i) {
        preds[i] = act_fn(buckets[i]);
    }
    return preds;
}

double run_q_learning(const std::vector<int>& train_buckets,
                      const std::vector<int>& train_labels,
                      const std::vector<int>& test_buckets,
                      const std::vector<int>& test_labels,
                      int seed,
                      std::mt19937& rng) {
    std::cout << "\n--- Tabular Q-Learning ---" << std::endl;
    const int STEPS = hpo::env_int("RL_Q_STEPS", 12000);
    const int EVAL_EVERY = hpo::env_int("RL_Q_EVAL_EVERY", 1000);
    const double ALPHA = hpo::env_double("RL_Q_ALPHA", 0.20);
    const double GAMMA = hpo::env_double("RL_Q_GAMMA", 0.0);
    const double epsilon_decay = hpo::env_double("RL_Q_EPS_DECAY", 0.9995);
    const double epsilon_min = hpo::env_double("RL_Q_EPS_MIN", 0.05);
    const double epsilon_start = hpo::env_double("RL_Q_EPS_START", 1.0);
    std::ostringstream params_json;
    params_json << "{\"algorithm\":\"q_learning\""
                << ",\"states\":" << kTabularStates
                << ",\"actions\":" << kNumActions
                << ",\"steps\":" << STEPS
                << ",\"eval_every\":" << EVAL_EVERY
                << ",\"alpha\":" << ALPHA
                << ",\"gamma\":" << GAMMA
                << ",\"epsilon_start\":" << epsilon_start
                << ",\"epsilon_min\":" << epsilon_min
                << ",\"epsilon_decay\":" << epsilon_decay
                << "}";
    RunLogger logger(
        "reinforcement",
        "q_learning_tabular",
        seed,
        "mnist-idx-v1",
        params_json.str()
    );

    TabularQLearningAgent agent(kTabularStates, kNumActions);
    double epsilon = epsilon_start;
    double reward_ema = 0.0;
    std::uniform_int_distribution<size_t> dist(0, train_labels.size() - 1);

    for (int step = 1; step <= STEPS; ++step) {
        const size_t idx = dist(rng);
        const int s = train_buckets[idx];
        const int a = agent.select_action(s, epsilon, rng);
        const double r = (a == train_labels[idx]) ? 1.0 : 0.0;
        const int s_next = train_buckets[dist(rng)];
        agent.update(s, a, r, s_next, ALPHA, GAMMA);

        reward_ema = 0.99 * reward_ema + 0.01 * r;
        epsilon = std::max(epsilon_min, epsilon * epsilon_decay);

        if (step % EVAL_EVERY == 0) {
            auto preds = eval_tabular(
                test_buckets, test_labels, [&](int b) { return agent.greedy_action(b); }
            );
            const double acc = accuracy_from_preds(preds, test_labels);
            auto [precision_macro, recall_macro, f1_macro] =
                benchlog::macro_prf_from_preds(preds, test_labels, kNumActions);

            logger.log_epoch_metric(
                step,
                "train",
                std::numeric_limits<double>::quiet_NaN(),
                reward_ema,
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                ALPHA,
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                0.0,
                0.0
            );
            logger.log_epoch_metric(
                step,
                "test",
                std::numeric_limits<double>::quiet_NaN(),
                acc,
                precision_macro,
                recall_macro,
                f1_macro,
                ALPHA,
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                0.0,
                0.0
            );
            logger.append_csv_row(
                "model_specific/reinforcement/rl_algorithms.csv",
                {"run_id", "algorithm", "step", "reward_mean", "loss", "epsilon", "test_acc", "plan_acc"},
                {logger.run_id(),
                 "q_learning",
                 std::to_string(step),
                 std::to_string(reward_ema),
                 "nan",
                 std::to_string(epsilon),
                 std::to_string(acc),
                 "nan"}
            );
            std::cout << "Step " << step << "/" << STEPS << " | Test Acc: " << std::fixed
                      << std::setprecision(2) << acc * 100 << "%" << std::endl;
        }
    }

    auto infer_start = std::chrono::high_resolution_clock::now();
    auto final_preds = eval_tabular(test_buckets, test_labels, [&](int b) { return agent.greedy_action(b); });
    auto infer_end = std::chrono::high_resolution_clock::now();
    const double infer_ms =
        std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    const double final_acc = accuracy_from_preds(final_preds, test_labels);

    log_final_common(logger, STEPS, final_preds, test_labels, infer_ms);
    logger.write_manifest_end();
    return final_acc;
}

double run_sarsa(const std::vector<int>& train_buckets,
                 const std::vector<int>& train_labels,
                 const std::vector<int>& test_buckets,
                 const std::vector<int>& test_labels,
                 int seed,
                 std::mt19937& rng) {
    std::cout << "\n--- Tabular SARSA ---" << std::endl;
    const int STEPS = hpo::env_int("RL_SARSA_STEPS", 12000);
    const int EVAL_EVERY = hpo::env_int("RL_SARSA_EVAL_EVERY", 1000);
    const double ALPHA = hpo::env_double("RL_SARSA_ALPHA", 0.20);
    const double GAMMA = hpo::env_double("RL_SARSA_GAMMA", 0.0);
    const double epsilon_decay = hpo::env_double("RL_SARSA_EPS_DECAY", 0.9995);
    const double epsilon_min = hpo::env_double("RL_SARSA_EPS_MIN", 0.05);
    const double epsilon_start = hpo::env_double("RL_SARSA_EPS_START", 1.0);
    std::ostringstream params_json;
    params_json << "{\"algorithm\":\"sarsa\""
                << ",\"states\":" << kTabularStates
                << ",\"actions\":" << kNumActions
                << ",\"steps\":" << STEPS
                << ",\"eval_every\":" << EVAL_EVERY
                << ",\"alpha\":" << ALPHA
                << ",\"gamma\":" << GAMMA
                << ",\"epsilon_start\":" << epsilon_start
                << ",\"epsilon_min\":" << epsilon_min
                << ",\"epsilon_decay\":" << epsilon_decay
                << "}";
    RunLogger logger(
        "reinforcement",
        "sarsa_tabular",
        seed,
        "mnist-idx-v1",
        params_json.str()
    );

    TabularSARSAAgent agent(kTabularStates, kNumActions);
    double epsilon = epsilon_start;
    double reward_ema = 0.0;
    std::uniform_int_distribution<size_t> dist(0, train_labels.size() - 1);

    for (int step = 1; step <= STEPS; ++step) {
        const size_t idx = dist(rng);
        const int s = train_buckets[idx];
        const int a = agent.select_action(s, epsilon, rng);
        const double r = (a == train_labels[idx]) ? 1.0 : 0.0;
        const int s_next = train_buckets[dist(rng)];
        const int a_next = agent.select_action(s_next, epsilon, rng);
        agent.update(s, a, r, s_next, a_next, ALPHA, GAMMA);

        reward_ema = 0.99 * reward_ema + 0.01 * r;
        epsilon = std::max(epsilon_min, epsilon * epsilon_decay);

        if (step % EVAL_EVERY == 0) {
            auto preds = eval_tabular(
                test_buckets, test_labels, [&](int b) { return agent.greedy_action(b); }
            );
            const double acc = accuracy_from_preds(preds, test_labels);
            auto [precision_macro, recall_macro, f1_macro] =
                benchlog::macro_prf_from_preds(preds, test_labels, kNumActions);

            logger.log_epoch_metric(
                step,
                "train",
                std::numeric_limits<double>::quiet_NaN(),
                reward_ema,
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                ALPHA,
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                0.0,
                0.0
            );
            logger.log_epoch_metric(
                step,
                "test",
                std::numeric_limits<double>::quiet_NaN(),
                acc,
                precision_macro,
                recall_macro,
                f1_macro,
                ALPHA,
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                0.0,
                0.0
            );
            logger.append_csv_row(
                "model_specific/reinforcement/rl_algorithms.csv",
                {"run_id", "algorithm", "step", "reward_mean", "loss", "epsilon", "test_acc", "plan_acc"},
                {logger.run_id(),
                 "sarsa",
                 std::to_string(step),
                 std::to_string(reward_ema),
                 "nan",
                 std::to_string(epsilon),
                 std::to_string(acc),
                 "nan"}
            );
            std::cout << "Step " << step << "/" << STEPS << " | Test Acc: " << std::fixed
                      << std::setprecision(2) << acc * 100 << "%" << std::endl;
        }
    }

    auto infer_start = std::chrono::high_resolution_clock::now();
    auto final_preds = eval_tabular(test_buckets, test_labels, [&](int b) { return agent.greedy_action(b); });
    auto infer_end = std::chrono::high_resolution_clock::now();
    const double infer_ms =
        std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    const double final_acc = accuracy_from_preds(final_preds, test_labels);

    log_final_common(logger, STEPS, final_preds, test_labels, infer_ms);
    logger.write_manifest_end();
    return final_acc;
}

double run_dqn_like(const Tensor& train_x,
                    const std::vector<int>& train_y,
                    const Tensor& test_x,
                    const std::vector<int>& test_y,
                    bool double_q,
                    int seed,
                    std::mt19937& rng) {
    const std::string algo = double_q ? "double_dqn" : "dqn";
    std::cout << "\n--- " << (double_q ? "Double-DQN" : "DQN") << " ---" << std::endl;
    const int hidden = hpo::env_int("RL_DQN_HIDDEN", 256);
    const double dqn_lr = hpo::env_double("RL_DQN_LR", 5e-4);
    const double dqn_gamma = hpo::env_double("RL_DQN_GAMMA", 0.0);
    const size_t replay_capacity = hpo::env_size("RL_DQN_REPLAY", 50000);
    const size_t dqn_batch = hpo::env_size("RL_DQN_BATCH", 64);
    const int target_update = hpo::env_int("RL_DQN_TARGET_UPDATE", 250);
    const int STEPS = hpo::env_int("RL_DQN_STEPS", 8000);
    const int EVAL_EVERY = hpo::env_int("RL_DQN_EVAL_EVERY", 1000);
    std::ostringstream params_json;
    params_json << "{\"algorithm\":\"" << algo << "\""
                << ",\"steps\":" << STEPS
                << ",\"eval_every\":" << EVAL_EVERY
                << ",\"batch_size\":" << dqn_batch
                << ",\"replay_capacity\":" << replay_capacity
                << ",\"target_update\":" << target_update
                << ",\"hidden\":" << hidden
                << ",\"lr\":" << dqn_lr
                << ",\"gamma\":" << dqn_gamma
                << ",\"double_q\":" << (double_q ? "true" : "false")
                << "}";
    RunLogger logger(
        "reinforcement",
        algo,
        seed,
        "mnist-idx-v1",
        params_json.str()
    );

    DQNAgent agent(
        784, hidden, kNumActions, dqn_lr, dqn_gamma, replay_capacity, dqn_batch, target_update, double_q, rng
    );
    benchlog::BenchBioHarness bio("reinforcement", algo, seed);
    bio.attach(agent.parameters(), agent.gradients(), algo + "_param");
    std::uniform_int_distribution<size_t> dist(0, train_y.size() - 1);
    double reward_ema = 0.0;
    double loss_ema = 0.0;
    bio.begin_epoch(1);

    for (int step = 1; step <= STEPS; ++step) {
        const size_t idx = dist(rng);
        Tensor s = train_x.row(idx);
        const int a = agent.select_action(s, rng);
        const double r = (a == train_y[idx]) ? 1.0 : 0.0;
        agent.observe({s, a, r, s, 1});
        bio.before_forward();
        const double loss = agent.train_step(rng);
        bio.after_backward(loss, r, dqn_lr);
        bio.after_optimizer_step();
        agent.decay_epsilon();

        reward_ema = 0.99 * reward_ema + 0.01 * r;
        if (!std::isnan(loss)) {
            loss_ema = 0.99 * loss_ema + 0.01 * loss;
        }

        if (step % EVAL_EVERY == 0) {
            bio.end_epoch(logger, step);
            bio.begin_epoch(step + 1);
            auto preds = agent.predict_actions(test_x);
            const double acc = accuracy_from_preds(preds, test_y);
            auto [precision_macro, recall_macro, f1_macro] =
                benchlog::macro_prf_from_preds(preds, test_y, kNumActions);

            logger.log_epoch_metric(
                step,
                "train",
                loss_ema,
                reward_ema,
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                dqn_lr,
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                0.0,
                0.0
            );
            logger.log_epoch_metric(
                step,
                "test",
                std::numeric_limits<double>::quiet_NaN(),
                acc,
                precision_macro,
                recall_macro,
                f1_macro,
                dqn_lr,
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                0.0,
                0.0
            );
            logger.append_csv_row(
                "model_specific/reinforcement/rl_algorithms.csv",
                {"run_id", "algorithm", "step", "reward_mean", "loss", "epsilon", "test_acc", "plan_acc"},
                {logger.run_id(),
                 algo,
                 std::to_string(step),
                 std::to_string(reward_ema),
                 std::to_string(loss_ema),
                 std::to_string(agent.epsilon()),
                 std::to_string(acc),
                 "nan"}
            );

            std::cout << "Step " << step << "/" << STEPS << " | Test Acc: " << std::fixed
                      << std::setprecision(2) << acc * 100 << "%" << std::endl;
        }
    }

    auto infer_start = std::chrono::high_resolution_clock::now();
    auto final_preds = agent.predict_actions(test_x);
    auto infer_end = std::chrono::high_resolution_clock::now();
    const double infer_ms =
        std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    const double final_acc = accuracy_from_preds(final_preds, test_y);

    log_final_common(logger, STEPS, final_preds, test_y, infer_ms);
    logger.write_manifest_end();
    return final_acc;
}

double run_muzero_lite(const Tensor& train_x,
                       const std::vector<int>& train_y,
                       const Tensor& test_x,
                       const std::vector<int>& test_y,
                       int seed,
                       std::mt19937& rng) {
    std::cout << "\n--- MuZero-Lite ---" << std::endl;
    const int latent = hpo::env_int("RL_MUZERO_LATENT", 64);
    const double muzero_lr = hpo::env_double("RL_MUZERO_LR", 1e-3);
    const double muzero_gamma = hpo::env_double("RL_MUZERO_GAMMA", 0.0);
    const int simulations = hpo::env_int("RL_MUZERO_SIMULATIONS", 15);
    const int STEPS = hpo::env_int("RL_MUZERO_STEPS", 4000);
    const int EVAL_EVERY = hpo::env_int("RL_MUZERO_EVAL_EVERY", 500);
    const size_t BATCH = hpo::env_size("RL_MUZERO_BATCH", 64);
    std::ostringstream params_json;
    params_json << "{\"algorithm\":\"muzero_lite\""
                << ",\"steps\":" << STEPS
                << ",\"eval_every\":" << EVAL_EVERY
                << ",\"batch_size\":" << BATCH
                << ",\"simulations\":" << simulations
                << ",\"latent\":" << latent
                << ",\"lr\":" << muzero_lr
                << ",\"gamma\":" << muzero_gamma
                << "}";
    RunLogger logger(
        "reinforcement",
        "muzero_lite",
        seed,
        "mnist-idx-v1",
        params_json.str()
    );

    MuZeroLiteAgent agent(784, latent, kNumActions, muzero_lr, muzero_gamma, simulations, rng);
    benchlog::BenchBioHarness bio("reinforcement", "muzero_lite", seed);
    bio.attach(agent.parameters(), agent.gradients(), "muzero_param");
    std::uniform_int_distribution<size_t> dist(0, train_y.size() - 1);
    double loss_ema = 0.0;
    bio.begin_epoch(1);

    for (int step = 1; step <= STEPS; ++step) {
        Tensor bx(BATCH, train_x.cols);
        std::vector<int> by(BATCH, 0);
        for (size_t i = 0; i < BATCH; ++i) {
            const size_t idx = dist(rng);
            by[i] = train_y[idx];
            for (size_t j = 0; j < train_x.cols; ++j) {
                bx(i, j) = train_x(idx, j);
            }
        }

        bio.before_forward();
        auto st = agent.train_step(bx, by, rng);
        bio.after_backward(st.loss_total, std::numeric_limits<double>::quiet_NaN(), muzero_lr);
        bio.after_optimizer_step();
        loss_ema = 0.99 * loss_ema + 0.01 * st.loss_total;

        if (step % EVAL_EVERY == 0) {
            bio.end_epoch(logger, step);
            bio.begin_epoch(step + 1);
            auto preds = agent.predict_policy_actions(test_x);
            const double acc = accuracy_from_preds(preds, test_y);
            auto [precision_macro, recall_macro, f1_macro] =
                benchlog::macro_prf_from_preds(preds, test_y, kNumActions);

            const size_t plan_n = std::min(static_cast<size_t>(64), test_x.rows);
            int plan_ok = 0;
            for (size_t i = 0; i < plan_n; ++i) {
                if (agent.plan_action(test_x.row(i)) == test_y[i]) plan_ok++;
            }
            const double plan_acc = plan_ok / static_cast<double>(plan_n);

            const size_t probe_idx = static_cast<size_t>(step / EVAL_EVERY) % test_x.rows;
            auto search = agent.plan_action_with_stats(test_x.row(probe_idx));
            logger.append_csv_row(
                "model_specific/reinforcement/muzero_search.csv",
                {"run_id", "step", "simulations", "root_entropy", "visit_max", "chosen_action", "true_label"},
                {logger.run_id(),
                 std::to_string(step),
                 std::to_string(agent.simulations()),
                 std::to_string(search.root_entropy),
                 std::to_string(search.visit_max),
                 std::to_string(search.action),
                 std::to_string(test_y[probe_idx])}
            );

            logger.log_epoch_metric(
                step,
                "train",
                loss_ema,
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                muzero_lr,
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                0.0,
                0.0
            );
            logger.log_epoch_metric(
                step,
                "test",
                std::numeric_limits<double>::quiet_NaN(),
                acc,
                precision_macro,
                recall_macro,
                f1_macro,
                muzero_lr,
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN(),
                0.0,
                0.0
            );
            logger.append_csv_row(
                "model_specific/reinforcement/rl_algorithms.csv",
                {"run_id", "algorithm", "step", "reward_mean", "loss", "epsilon", "test_acc", "plan_acc"},
                {logger.run_id(),
                 "muzero_lite",
                 std::to_string(step),
                 "nan",
                 std::to_string(loss_ema),
                 "nan",
                 std::to_string(acc),
                 std::to_string(plan_acc)}
            );

            std::cout << "Step " << step << "/" << STEPS
                      << " | Policy Acc: " << std::fixed << std::setprecision(2) << acc * 100
                      << "% | Plan@" << plan_n << ": " << plan_acc * 100 << "%" << std::endl;
        }
    }

    auto infer_start = std::chrono::high_resolution_clock::now();
    auto final_preds = agent.predict_policy_actions(test_x);
    auto infer_end = std::chrono::high_resolution_clock::now();
    const double infer_ms =
        std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    const double final_acc = accuracy_from_preds(final_preds, test_y);

    log_final_common(logger, STEPS, final_preds, test_y, infer_ms);
    logger.write_manifest_end();
    return final_acc;
}

} // namespace

int main() {
    std::cout << "=== Reinforcement Benchmark (MNIST as Contextual Bandit) ===" << std::endl;
    const int seed = hpo::env_int("RL_SEED", 42);
    std::mt19937 rng(static_cast<unsigned int>(seed));
    const size_t train_n_cfg = hpo::env_size("RL_TRAIN_N", 20000);
    const size_t test_n_cfg = hpo::env_size("RL_TEST_N", 2000);
    const std::string algo_filter = hpo::to_lower(hpo::env_string("RL_ALGORITHMS", "all"));
    const auto algo_tokens = hpo::split_csv(algo_filter);
    auto wants = [&](const std::string& key) {
        if (algo_filter == "all") return true;
        for (const auto& t : algo_tokens) {
            if (t == key) return true;
        }
        return false;
    };

    auto train = MNISTLoader::load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    auto test = MNISTLoader::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    const size_t train_n = std::min(train_n_cfg, train.images.rows);
    const size_t test_n = std::min(test_n_cfg, test.images.rows);
    Tensor train_x = train.images.slice_rows(0, train_n);
    Tensor test_x = test.images.slice_rows(0, test_n);
    std::vector<int> train_y(train.labels.begin(), train.labels.begin() + static_cast<long>(train_n));
    std::vector<int> test_y(test.labels.begin(), test.labels.begin() + static_cast<long>(test_n));

    std::cout << "Precomputing tabular state buckets..." << std::endl;
    auto train_buckets = precompute_buckets(train_x);
    auto test_buckets = precompute_buckets(test_x);

    double acc_q = std::numeric_limits<double>::quiet_NaN();
    double acc_sarsa = std::numeric_limits<double>::quiet_NaN();
    double acc_dqn = std::numeric_limits<double>::quiet_NaN();
    double acc_ddqn = std::numeric_limits<double>::quiet_NaN();
    double acc_muzero = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> executed_accs;
    bool ran_any = false;
    if (wants("q") || wants("q_learning")) {
        acc_q = run_q_learning(train_buckets, train_y, test_buckets, test_y, seed, rng);
        executed_accs.push_back(acc_q);
        ran_any = true;
    }
    if (wants("sarsa")) {
        acc_sarsa = run_sarsa(train_buckets, train_y, test_buckets, test_y, seed, rng);
        executed_accs.push_back(acc_sarsa);
        ran_any = true;
    }
    if (wants("dqn")) {
        acc_dqn = run_dqn_like(train_x, train_y, test_x, test_y, false, seed, rng);
        executed_accs.push_back(acc_dqn);
        ran_any = true;
    }
    if (wants("ddqn") || wants("double_dqn")) {
        acc_ddqn = run_dqn_like(train_x, train_y, test_x, test_y, true, seed, rng);
        executed_accs.push_back(acc_ddqn);
        ran_any = true;
    }
    if (wants("muzero") || wants("muzero_lite")) {
        acc_muzero = run_muzero_lite(train_x, train_y, test_x, test_y, seed, rng);
        executed_accs.push_back(acc_muzero);
        ran_any = true;
    }

    if (!ran_any) {
        std::cerr << "No RL algorithm selected for RL_ALGORITHMS='" << algo_filter
                  << "'. Valid values include: all,q,sarsa,dqn,ddqn,muzero" << std::endl;
        return 2;
    }

    std::cout << "\n=== Reinforcement Benchmark Complete ===" << std::endl;
    auto print_acc = [](const std::string& label, double acc) {
        std::cout << label;
        if (std::isnan(acc)) {
            std::cout << "N/A";
        } else {
            std::cout << std::fixed << std::setprecision(2) << acc * 100 << "%";
        }
        std::cout << std::endl;
    };
    print_acc("Q-Learning Acc:  ", acc_q);
    print_acc("SARSA Acc:       ", acc_sarsa);
    print_acc("DQN Acc:         ", acc_dqn);
    print_acc("Double-DQN Acc:  ", acc_ddqn);
    print_acc("MuZero-Lite Acc: ", acc_muzero);

    const double min_acc = *std::min_element(executed_accs.begin(), executed_accs.end());
    return (min_acc >= 0.25) ? 0 : 1;
}
