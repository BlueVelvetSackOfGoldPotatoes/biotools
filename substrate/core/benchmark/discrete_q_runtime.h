#pragma once

#include "core/benchmark/discrete_env.h"
#include "core/online/trainable_model.h"
#include "core/optim/optimizer.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <vector>

namespace benchmark {

struct DiscreteQConfig {
    int episodes = 30000;
    int eval_every = 1500;
    int eval_episodes = 300;
    int max_steps_per_episode = 16;

    double gamma = 0.96;
    double epsilon_start = 1.0;
    double epsilon_min = 0.03;
    double epsilon_decay = 0.9997;
    double grad_clip_norm = 5.0;

    std::size_t batch_log_every = 50;
};

struct DiscreteInferenceSample {
    int sample_id = 0;
    int true_action = 0;
    int pred_action = 0;
    double confidence = 0.0;
    double entropy = 0.0;
    double top2_margin = 0.0;
    double latency_ms = 0.0;
    int is_correct = 0;
};

struct DiscreteBatchMetrics {
    int episode = 0;
    int global_step = 0;
    double epsilon = 0.0;
    double reward_ema = 0.0;
    double td_error_ema = 0.0;
    double q_loss_ema = 0.0;
    double action_match_ema = std::numeric_limits<double>::quiet_NaN();
    double missed_win_rate_ema = std::numeric_limits<double>::quiet_NaN();
    double missed_block_rate_ema = std::numeric_limits<double>::quiet_NaN();
};

struct DiscreteTransitionTrace {
    int episode = 0;
    int step = 0;
    int global_step = 0;

    int action = -1;
    int teacher_action = -1;
    int immediate_win_action = -1;
    int immediate_block_action = -1;

    double reward = 0.0;
    bool done = false;
    int outcome = 0;
    double epsilon = 0.0;
    double elapsed_ms = 0.0;

    std::vector<uint8_t> obs_before;
    std::vector<uint8_t> obs_after;
};

struct DiscreteEvalMetrics {
    int episode = 0;
    bool final_pass = false;

    int episodes = 0;
    int wins = 0;
    int draws = 0;
    int losses = 0;
    int invalid_actions = 0;
    int moves = 0;
    int total_plies = 0;

    double win_rate = 0.0;
    double draw_rate = 0.0;
    double loss_rate = 0.0;
    double non_loss_rate = 0.0;
    double invalid_rate = 0.0;
    double action_match_rate = std::numeric_limits<double>::quiet_NaN();
    int win_opportunities = 0;
    int missed_wins = 0;
    int block_opportunities = 0;
    int missed_blocks = 0;
    double win_capture_rate = std::numeric_limits<double>::quiet_NaN();
    double block_capture_rate = std::numeric_limits<double>::quiet_NaN();
    double missed_win_rate = std::numeric_limits<double>::quiet_NaN();
    double missed_block_rate = std::numeric_limits<double>::quiet_NaN();
    double avg_plies = 0.0;
    double avg_latency_ms = 0.0;
    double eval_time_ms = 0.0;

    std::vector<int> preds;
    std::vector<int> truths;
    std::vector<DiscreteInferenceSample> inference;
};

class DiscreteQRuntime {
public:
    using BatchCallback = std::function<void(const DiscreteBatchMetrics&)>;
    using EvalCallback = std::function<void(const DiscreteEvalMetrics&)>;
    struct TrainHooks {
        std::function<void(int episode)> on_episode_begin;
        std::function<void()> before_forward;
        std::function<void(double loss, double reward)> after_backward;
        std::function<void()> after_optimizer_step;
        std::function<void(const DiscreteTransitionTrace&)> on_transition;
        std::function<void(int episode)> on_episode_end;
    };

    explicit DiscreteQRuntime(DiscreteQConfig cfg = {}, int seed = 42);

    DiscreteEvalMetrics train(DiscreteEnv& env_train,
                              DiscreteEnv& env_eval,
                              TrainableModel& model,
                              Adam& optimizer,
                              const BatchCallback& on_batch = {},
                              const EvalCallback& on_eval = {},
                              const TrainHooks& hooks = {});

    DiscreteEvalMetrics evaluate(DiscreteEnv& env,
                                 TrainableModel& model,
                                 int episodes,
                                 bool collect_inference = true);

private:
    DiscreteQConfig cfg_;
    std::mt19937 rng_;
    int global_step_ = 0;

    static int select_action_epsilon_greedy(const Tensor& q,
                                            const std::vector<int>& legal_actions,
                                            double epsilon,
                                            std::mt19937& rng);

    static int select_action_greedy(const Tensor& q, const std::vector<int>& legal_actions);
    static double max_legal_q(const Tensor& q, const std::vector<int>& legal_actions);
};

} // namespace benchmark
