#include "core/nn/module.h"
#include "core/losses/losses.h"
#include "core/data/dataloader.h"
#include "core/metrics/metrics.h"
#include "core/io/run_logger.h"
#include "models/actor_critic/src/ac_models.h"
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
#include <vector>

static double empirical_entropy(const std::vector<int>& actions, int n_actions = 10) {
    std::vector<int> counts(n_actions, 0);
    for (int a : actions) {
        if (a >= 0 && a < n_actions) counts[static_cast<size_t>(a)]++;
    }
    double ent = 0.0;
    const double n = static_cast<double>(actions.size());
    for (int c : counts) {
        if (c == 0) continue;
        const double p = c / n;
        ent -= p * std::log(p + 1e-12);
    }
    return ent;
}

int main() {
    std::cout << "=== Actor-Critic Benchmark (MNIST as Bandit) ===" << std::endl;
    const int seed = hpo::env_int("ACTOR_CRITIC_SEED", 42);
    std::mt19937 rng(static_cast<unsigned int>(seed));
    const int hidden = hpo::env_int("ACTOR_CRITIC_HIDDEN", 256);
    const int steps = hpo::env_int("ACTOR_CRITIC_STEPS", 50000);
    const size_t batch_size = hpo::env_size("ACTOR_CRITIC_BATCH", 128);
    const double gamma = hpo::env_double("ACTOR_CRITIC_GAMMA", 0.99);
    const int eval_every = hpo::env_int("ACTOR_CRITIC_EVAL_EVERY", 10000);
    const double a2c_lr_policy = hpo::env_double("A2C_LR_POLICY", 0.001);
    const double a2c_lr_value = hpo::env_double("A2C_LR_VALUE", 0.001);
    const double ppo_lr = hpo::env_double("PPO_LR", 0.0003);
    const double ppo_clip_eps = hpo::env_double("PPO_CLIP_EPS", 0.2);

    auto train = MNISTLoader::load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    auto test = MNISTLoader::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    std::vector<int> train_labels(train.labels.begin(), train.labels.end());
    std::vector<int> test_labels(test.labels.begin(), test.labels.end());

    Tensor test_x = test.images;
    std::vector<int> test_y = test_labels;
    double final_a2c_acc = 0.0;
    double final_ppo_acc = 0.0;

    {
        std::cout << "\n--- A2C ---" << std::endl;
        std::ostringstream a2c_params;
        a2c_params << "{\"steps\":" << steps
                   << ",\"batch_size\":" << batch_size
                   << ",\"gamma\":" << gamma
                   << ",\"hidden\":" << hidden
                   << ",\"lr_policy\":" << a2c_lr_policy
                   << ",\"lr_value\":" << a2c_lr_value
                   << "}";
        RunLogger logger(
            "actor_critic",
            "a2c",
            seed,
            "mnist-idx-v1",
            a2c_params.str()
        );

        auto start = std::chrono::high_resolution_clock::now();
        A2C agent(784, hidden, 10, gamma, a2c_lr_policy, a2c_lr_value, rng);
        benchlog::BenchBioHarness bio("actor_critic", "a2c", seed);
        bio.attach(agent.parameters(), agent.gradients(), "actor_critic_a2c_param");
        const size_t BATCH = batch_size;
        const int STEPS = steps;
        std::uniform_int_distribution<size_t> dist(0, train.images.rows - 1);
        bio.begin_epoch(1);

        for (int step = 0; step < STEPS; ++step) {
            std::vector<size_t> idx(BATCH);
            for (auto& i : idx) i = dist(rng);

            Tensor states(BATCH, 784);
            for (size_t i = 0; i < BATCH; ++i) {
                for (size_t j = 0; j < 784; ++j) {
                    states(i, j) = train.images(idx[i], j);
                }
            }

            auto actions = agent.select_actions(states, rng);
            std::vector<double> rewards(BATCH);
            for (size_t i = 0; i < BATCH; ++i) {
                rewards[i] = (actions[i] == train_labels[idx[i]]) ? 1.0 : 0.0;
            }
            const double reward_mean =
                std::accumulate(rewards.begin(), rewards.end(), 0.0) / static_cast<double>(BATCH);
            const double entropy = empirical_entropy(actions, 10);
            A2C::Experience exp{states, actions, rewards};
            bio.before_forward();
            agent.train_step(exp);
            bio.after_backward(
                std::numeric_limits<double>::quiet_NaN(),
                reward_mean,
                a2c_lr_policy
            );
            bio.after_optimizer_step();

            const bool bio_epoch_boundary =
                (((step + 1) % eval_every) == 0) || (step + 1 == STEPS);
            if (bio_epoch_boundary) {
                bio.end_epoch(logger, step + 1);
                if (step + 1 < STEPS) {
                    bio.begin_epoch(step + 2);
                }
            }

            if ((step + 1) % 500 == 0) {
                logger.append_csv_row(
                    "model_specific/actor_critic/rl_train.csv",
                    {"run_id", "step", "reward_mean", "entropy"},
                    {logger.run_id(), std::to_string(step + 1), std::to_string(reward_mean), std::to_string(entropy)}
                );
                logger.append_csv_row(
                    "model_specific/actor_critic/value_stats.csv",
                    {"run_id", "algorithm", "step", "reward_mean", "entropy", "reward_entropy_ratio"},
                    {logger.run_id(), "a2c", std::to_string(step + 1),
                     std::to_string(reward_mean), std::to_string(entropy),
                     std::to_string(entropy > 1e-12 ? reward_mean / entropy : 0.0)}
                );
                logger.log_epoch_metric(
                    step + 1,
                    "train",
                    std::numeric_limits<double>::quiet_NaN(),
                    reward_mean,
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    a2c_lr_policy,
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    0.0,
                    0.0
                );
            }

            if ((step + 1) % eval_every == 0) {
                auto eval_start = std::chrono::high_resolution_clock::now();
                auto preds = agent.predict(test_x);
                const double acc = Metrics::accuracy(preds, test_y);
                auto eval_end = std::chrono::high_resolution_clock::now();
                const double eval_ms =
                    std::chrono::duration<double, std::milli>(eval_end - eval_start).count();
                auto [precision_macro, recall_macro, f1_macro] =
                    benchlog::macro_prf_from_preds(preds, test_y, 10);

                logger.log_epoch_metric(
                    step + 1,
                    "test",
                    std::numeric_limits<double>::quiet_NaN(),
                    acc,
                    precision_macro,
                    recall_macro,
                    f1_macro,
                    a2c_lr_policy,
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    eval_ms,
                    test_x.rows / (eval_ms / 1000.0 + 1e-12)
                );

                std::cout << "Step " << step + 1 << "/" << STEPS
                          << " | Test Acc: " << std::fixed << std::setprecision(2) << acc * 100 << "%"
                          << std::endl;
            }
        }

        auto infer_start = std::chrono::high_resolution_clock::now();
        auto preds = agent.predict(test_x);
        auto infer_end = std::chrono::high_resolution_clock::now();
        const double infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
        const double per_sample_ms = infer_ms / static_cast<double>(test_x.rows);
        const double acc = Metrics::accuracy(preds, test_y);
        final_a2c_acc = acc;

        benchlog::log_confusion_and_class_metrics(logger, STEPS, "test", preds, test_y, 10);
        benchlog::log_inference_from_preds(
            logger, "test", preds, test_y, per_sample_ms, static_cast<int>(test_x.rows)
        );
        benchlog::log_calibration_from_preds(logger, "test", preds, test_y, 10);
        logger.log_system_metric(
            RunLogger::utc_now_iso8601(),
            test_x.rows / (infer_ms / 1000.0 + 1e-12),
            per_sample_ms,
            per_sample_ms,
            per_sample_ms,
            0.0,
            0.0
        );

        auto end = std::chrono::high_resolution_clock::now();
        const double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "A2C Final Acc: " << std::fixed << std::setprecision(2) << acc * 100 << "%"
                  << " (" << std::setprecision(1) << elapsed << "s)"
                  << " | " << (acc >= 0.90 ? "PASS" : "FAIL") << std::endl;
        logger.write_manifest_end();
    }

    {
        std::cout << "\n--- PPO ---" << std::endl;
        std::ostringstream ppo_params;
        ppo_params << "{\"steps\":" << steps
                   << ",\"batch_size\":" << batch_size
                   << ",\"gamma\":" << gamma
                   << ",\"hidden\":" << hidden
                   << ",\"clip_eps\":" << ppo_clip_eps
                   << ",\"lr\":" << ppo_lr
                   << "}";
        RunLogger logger(
            "actor_critic",
            "ppo",
            seed,
            "mnist-idx-v1",
            ppo_params.str()
        );

        auto start = std::chrono::high_resolution_clock::now();
        PPO agent(784, hidden, 10, gamma, ppo_clip_eps, ppo_lr, rng);
        benchlog::BenchBioHarness bio("actor_critic", "ppo", seed);
        bio.attach(agent.parameters(), agent.gradients(), "actor_critic_ppo_param");
        const size_t BATCH = batch_size;
        const int STEPS = steps;
        std::uniform_int_distribution<size_t> dist(0, train.images.rows - 1);
        bio.begin_epoch(1);

        for (int step = 0; step < STEPS; ++step) {
            std::vector<size_t> idx(BATCH);
            for (auto& i : idx) i = dist(rng);

            Tensor states(BATCH, 784);
            for (size_t i = 0; i < BATCH; ++i) {
                for (size_t j = 0; j < 784; ++j) {
                    states(i, j) = train.images(idx[i], j);
                }
            }

            auto actions = agent.select_actions(states, rng);
            std::vector<double> rewards(BATCH);
            for (size_t i = 0; i < BATCH; ++i) {
                rewards[i] = (actions[i] == train_labels[idx[i]]) ? 1.0 : 0.0;
            }
            const double reward_mean =
                std::accumulate(rewards.begin(), rewards.end(), 0.0) / static_cast<double>(BATCH);
            const double entropy = empirical_entropy(actions, 10);
            A2C::Experience exp{states, actions, rewards};
            bio.before_forward();
            agent.train_step(exp);
            bio.after_backward(
                std::numeric_limits<double>::quiet_NaN(),
                reward_mean,
                ppo_lr
            );
            bio.after_optimizer_step();

            agent.update_old_policy();

            const bool bio_epoch_boundary =
                (((step + 1) % eval_every) == 0) || (step + 1 == STEPS);
            if (bio_epoch_boundary) {
                bio.end_epoch(logger, step + 1);
                if (step + 1 < STEPS) {
                    bio.begin_epoch(step + 2);
                }
            }

            if ((step + 1) % 500 == 0) {
                logger.append_csv_row(
                    "model_specific/actor_critic/rl_train.csv",
                    {"run_id", "step", "reward_mean", "entropy"},
                    {logger.run_id(), std::to_string(step + 1), std::to_string(reward_mean), std::to_string(entropy)}
                );
                logger.append_csv_row(
                    "model_specific/actor_critic/value_stats.csv",
                    {"run_id", "algorithm", "step", "reward_mean", "entropy", "reward_entropy_ratio"},
                    {logger.run_id(), "ppo", std::to_string(step + 1),
                     std::to_string(reward_mean), std::to_string(entropy),
                     std::to_string(entropy > 1e-12 ? reward_mean / entropy : 0.0)}
                );
                logger.log_epoch_metric(
                    step + 1,
                    "train",
                    std::numeric_limits<double>::quiet_NaN(),
                    reward_mean,
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    ppo_lr,
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    0.0,
                    0.0
                );
            }

            if ((step + 1) % eval_every == 0) {
                auto eval_start = std::chrono::high_resolution_clock::now();
                auto preds = agent.predict(test_x);
                const double acc = Metrics::accuracy(preds, test_y);
                auto eval_end = std::chrono::high_resolution_clock::now();
                const double eval_ms =
                    std::chrono::duration<double, std::milli>(eval_end - eval_start).count();
                auto [precision_macro, recall_macro, f1_macro] =
                    benchlog::macro_prf_from_preds(preds, test_y, 10);

                logger.log_epoch_metric(
                    step + 1,
                    "test",
                    std::numeric_limits<double>::quiet_NaN(),
                    acc,
                    precision_macro,
                    recall_macro,
                    f1_macro,
                    ppo_lr,
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN(),
                    eval_ms,
                    test_x.rows / (eval_ms / 1000.0 + 1e-12)
                );

                std::cout << "Step " << step + 1 << "/" << STEPS
                          << " | Test Acc: " << std::fixed << std::setprecision(2) << acc * 100 << "%"
                          << std::endl;
            }
        }

        auto infer_start = std::chrono::high_resolution_clock::now();
        auto preds = agent.predict(test_x);
        auto infer_end = std::chrono::high_resolution_clock::now();
        const double infer_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
        const double per_sample_ms = infer_ms / static_cast<double>(test_x.rows);
        const double acc = Metrics::accuracy(preds, test_y);
        final_ppo_acc = acc;

        benchlog::log_confusion_and_class_metrics(logger, STEPS, "test", preds, test_y, 10);
        benchlog::log_inference_from_preds(
            logger, "test", preds, test_y, per_sample_ms, static_cast<int>(test_x.rows)
        );
        benchlog::log_calibration_from_preds(logger, "test", preds, test_y, 10);
        logger.log_system_metric(
            RunLogger::utc_now_iso8601(),
            test_x.rows / (infer_ms / 1000.0 + 1e-12),
            per_sample_ms,
            per_sample_ms,
            per_sample_ms,
            0.0,
            0.0
        );

        auto end = std::chrono::high_resolution_clock::now();
        const double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "PPO Final Acc: " << std::fixed << std::setprecision(2) << acc * 100 << "%"
                  << " (" << std::setprecision(1) << elapsed << "s)"
                  << " | " << (acc >= 0.90 ? "PASS" : "FAIL") << std::endl;
        logger.write_manifest_end();
    }

    std::cout << "\n=== Actor-Critic Benchmark Complete ===" << std::endl;
    const double min_acc = std::min(final_a2c_acc, final_ppo_acc);
    return (min_acc >= 0.90) ? 0 : 1;
}
