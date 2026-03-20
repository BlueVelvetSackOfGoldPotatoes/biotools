#pragma once

#include "../../../core/nn/module.h"
#include "../../../core/losses/losses.h"
#include "../../../core/optim/optimizer.h"

#include <cstddef>
#include <random>
#include <string>
#include <vector>

// ============================================================================
// Shared transition + replay buffer
// ============================================================================
struct RLTransition {
    Tensor state;      // 1 x state_dim
    int action = 0;
    double reward = 0.0;
    Tensor next_state; // 1 x state_dim
    int done = 0;
};

class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t capacity);

    void push(const RLTransition& t);
    std::vector<RLTransition> sample(size_t n, std::mt19937& rng) const;
    size_t size() const;

private:
    std::vector<RLTransition> buffer_;
    size_t capacity_;
    size_t pos_ = 0;
    bool full_ = false;
};

// ============================================================================
// Tabular RL baselines
// ============================================================================
class TabularQLearningAgent {
public:
    TabularQLearningAgent(size_t num_states, size_t num_actions);

    int greedy_action(int state_id) const;
    int greedy_action_masked(int state_id, const std::vector<int>& legal_actions) const;
    int select_action(int state_id, double epsilon, std::mt19937& rng) const;
    void update(int state_id, int action, double reward, int next_state_id, double alpha, double gamma);
    double q_value(int state_id, int action) const;

private:
    size_t num_states_, num_actions_;
    std::vector<double> q_; // flattened [state * num_actions + action]
};

class TabularSARSAAgent {
public:
    TabularSARSAAgent(size_t num_states, size_t num_actions);

    int greedy_action(int state_id) const;
    int greedy_action_masked(int state_id, const std::vector<int>& legal_actions) const;
    int select_action(int state_id, double epsilon, std::mt19937& rng) const;
    void update(int state_id,
                int action,
                double reward,
                int next_state_id,
                int next_action,
                double alpha,
                double gamma);
    double q_value(int state_id, int action) const;

private:
    size_t num_states_, num_actions_;
    std::vector<double> q_; // flattened [state * num_actions + action]
};

// ============================================================================
// Deep Q-learning (DQN / Double-DQN)
// ============================================================================
class DQNAgent {
public:
    DQNAgent(size_t state_dim,
             size_t hidden_dim,
             size_t num_actions,
             double lr,
             double gamma,
             size_t replay_capacity,
             size_t batch_size,
             int target_update_interval,
             bool double_q,
             std::mt19937& rng,
             double eps_start = 1.0,
             double eps_end = 0.05,
             double eps_decay = 0.9995);

    int select_action(const Tensor& state, std::mt19937& rng);
    void observe(const RLTransition& t);
    double train_step(std::mt19937& rng); // NaN if insufficient replay
    std::vector<int> predict_actions(const Tensor& states);
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();

    void decay_epsilon();
    double epsilon() const { return epsilon_; }
    void set_epsilon(double eps) { epsilon_ = eps; }

private:
    size_t state_dim_, num_actions_, batch_size_;
    double gamma_;
    int target_update_interval_;
    int train_steps_ = 0;
    bool double_q_;

    double epsilon_, epsilon_end_, epsilon_decay_;

    Sequential online_net_;
    Sequential target_net_;
    Adam optimizer_;
    ReplayBuffer replay_;

    static void copy_params(Sequential& dst, const Sequential& src);
    int greedy_action_single(const Tensor& state);
};

// ============================================================================
// MuZero-lite for contextual bandit planning
// ----------------------------------------------------------------------------
// This is a compact MuZero-style agent:
// - representation h(s) -> z
// - dynamics g(z, a) -> (z', r)
// - prediction f(z) -> (policy, value)
// plus a shallow PUCT search over actions.
// ============================================================================
class MuZeroLiteAgent {
public:
    struct TrainStats {
        double loss_total = 0.0;
        double loss_policy = 0.0;
        double loss_value = 0.0;
        double loss_reward = 0.0;
    };

    struct SearchStats {
        int action = 0;
        double root_entropy = 0.0;
        double visit_max = 0.0;
        double root_value = 0.0;
    };

    MuZeroLiteAgent(size_t state_dim,
                    size_t latent_dim,
                    size_t num_actions,
                    double lr,
                    double gamma,
                    int simulations,
                    std::mt19937& rng);

    TrainStats train_step(const Tensor& states, const std::vector<int>& labels, std::mt19937& rng);
    std::vector<int> predict_policy_actions(const Tensor& states);
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();

    SearchStats plan_action_with_stats(const Tensor& state);
    int plan_action(const Tensor& state) { return plan_action_with_stats(state).action; }

    int simulations() const { return simulations_; }

private:
    size_t state_dim_, latent_dim_, num_actions_;
    double gamma_;
    int simulations_;
    std::mt19937* rng_;

    Sequential repr_net_;
    Sequential dyn_net_;    // input latent+action_onehot -> latent + reward
    Sequential policy_net_; // latent -> action logits
    Sequential value_net_;  // latent -> scalar value

    Adam opt_repr_, opt_dyn_, opt_policy_, opt_value_;

    static Tensor one_hot_actions(const std::vector<int>& actions, size_t num_actions);
    static std::vector<int> argmax_rows(const Tensor& logits);
    static double entropy_from_probs(const Tensor& probs_row);
};
