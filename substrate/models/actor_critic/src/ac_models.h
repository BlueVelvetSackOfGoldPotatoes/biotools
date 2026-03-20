#pragma once

#include "../../../core/nn/module.h"
#include "../../../core/optim/optimizer.h"
#include "../../../core/losses/losses.h"
#include <random>
#include <vector>
#include <string>

// ============================================================
// PolicyNetwork
// Maps state (flattened MNIST image) to action logits (digit 0-9).
// Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear (logits)
// ============================================================
class PolicyNetwork {
    Sequential net;
public:
    PolicyNetwork() = default;
    PolicyNetwork(size_t input_size, size_t hidden_size, size_t num_actions, std::mt19937& rng);

    // Returns raw logits: batch x num_actions
    Tensor forward(const Tensor& state);

    // Returns softmax probabilities: batch x num_actions
    Tensor action_probs(const Tensor& state);

    // Backward through the network given gradient on logits
    Tensor backward(const Tensor& grad);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();

    // Deep copy parameters from another PolicyNetwork
    void copy_params_from(const PolicyNetwork& other);
};

// ============================================================
// ValueNetwork
// Maps state to a scalar value estimate.
// Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear(1)
// ============================================================
class ValueNetwork {
    Sequential net;
public:
    ValueNetwork() = default;
    ValueNetwork(size_t input_size, size_t hidden_size, std::mt19937& rng);

    // Returns value estimate: batch x 1
    Tensor forward(const Tensor& state);

    // Backward through the network given gradient on value
    Tensor backward(const Tensor& grad);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();
};

// ============================================================
// A2C (Advantage Actor-Critic)
// Treats MNIST classification as a contextual bandit:
//   - State: image features (batch x 784)
//   - Action: predicted digit (0-9)
//   - Reward: 1.0 if correct, 0.0 if wrong
// Single-step episodes (no discount needed, but gamma kept for generality).
// ============================================================
class A2C {
    PolicyNetwork policy;
    ValueNetwork value;
    double gamma;
    Adam policy_optimizer_;
    Adam value_optimizer_;

public:
    struct Experience {
        Tensor state;                // batch x features
        std::vector<int> actions;    // action taken per sample
        std::vector<double> rewards; // reward per sample
    };

    A2C(size_t state_size, size_t hidden_size, size_t num_actions,
        double gamma, double lr_policy, double lr_value, std::mt19937& rng);

    // Perform one training step on a batch of experiences.
    // Updates both policy and value networks.
    void train_step(const Experience& exp);

    // Sample actions from the policy for a batch of states.
    std::vector<int> select_actions(const Tensor& states, std::mt19937& rng);

    // Greedy action selection (argmax) for evaluation.
    std::vector<int> predict(const Tensor& states);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
};

// ============================================================
// PPO (Proximal Policy Optimization)
// Extends A2C with clipped surrogate objective and an old policy
// for importance-weighted updates.
// ============================================================
class PPO {
    PolicyNetwork policy, old_policy;
    ValueNetwork value;
    double gamma, clip_eps;
    Adam policy_optimizer_;
    Adam value_optimizer_;

public:
    PPO(size_t state_size, size_t hidden_size, size_t num_actions,
        double gamma, double clip_eps, double lr, std::mt19937& rng);

    // Train on a batch of experiences using clipped PPO objective.
    void train_step(const A2C::Experience& exp);

    // Copy current policy parameters into old_policy.
    void update_old_policy();

    // Sample actions from the current policy.
    std::vector<int> select_actions(const Tensor& states, std::mt19937& rng);

    // Greedy action selection (argmax) for evaluation.
    std::vector<int> predict(const Tensor& states);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
};
