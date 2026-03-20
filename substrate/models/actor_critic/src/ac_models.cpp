#include "ac_models.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#define AC_OMP_PAR_FOR _Pragma("omp parallel for schedule(static)")
#define AC_OMP_PAR_FOR_COLLAPSE2 _Pragma("omp parallel for collapse(2) schedule(static)")
#define AC_OMP_PAR_FOR_REDUCE_SUM _Pragma("omp parallel for reduction(+ : adv_mean) schedule(static)")
#define AC_OMP_PAR_FOR_REDUCE_VAR _Pragma("omp parallel for reduction(+ : adv_var) schedule(static)")
#else
#define AC_OMP_PAR_FOR
#define AC_OMP_PAR_FOR_COLLAPSE2
#define AC_OMP_PAR_FOR_REDUCE_SUM
#define AC_OMP_PAR_FOR_REDUCE_VAR
#endif

// ============================================================
// Helper: row-wise softmax (same as LossFunctions::softmax)
// ============================================================
static Tensor row_softmax(const Tensor& logits) {
    Tensor result(logits.rows, logits.cols);
    AC_OMP_PAR_FOR
    for (size_t i = 0; i < logits.rows; ++i) {
        double max_val = logits(i, 0);
        for (size_t j = 1; j < logits.cols; ++j)
            max_val = std::max(max_val, logits(i, j));
        double sum_exp = 0.0;
        for (size_t j = 0; j < logits.cols; ++j) {
            result(i, j) = std::exp(logits(i, j) - max_val);
            sum_exp += result(i, j);
        }
        for (size_t j = 0; j < logits.cols; ++j)
            result(i, j) /= sum_exp;
    }
    return result;
}

// ============================================================
// Helper: row-wise log-softmax
// ============================================================
[[maybe_unused]] static Tensor row_log_softmax(const Tensor& logits) {
    Tensor result(logits.rows, logits.cols);
    AC_OMP_PAR_FOR
    for (size_t i = 0; i < logits.rows; ++i) {
        double max_val = logits(i, 0);
        for (size_t j = 1; j < logits.cols; ++j)
            max_val = std::max(max_val, logits(i, j));
        double sum_exp = 0.0;
        for (size_t j = 0; j < logits.cols; ++j)
            sum_exp += std::exp(logits(i, j) - max_val);
        double log_sum = max_val + std::log(sum_exp);
        for (size_t j = 0; j < logits.cols; ++j)
            result(i, j) = logits(i, j) - log_sum;
    }
    return result;
}

// ============================================================
// PolicyNetwork
// ============================================================

PolicyNetwork::PolicyNetwork(size_t input_size, size_t hidden_size,
                             size_t num_actions, std::mt19937& rng) {
    net.add(std::make_unique<Linear>(input_size, hidden_size, "policy_fc1", rng));
    net.add(std::make_unique<ReLU>());
    net.add(std::make_unique<Linear>(hidden_size, hidden_size, "policy_fc2", rng));
    net.add(std::make_unique<ReLU>());
    net.add(std::make_unique<Linear>(hidden_size, num_actions, "policy_fc3", rng));
}

Tensor PolicyNetwork::forward(const Tensor& state) {
    return net.forward(state);
}

Tensor PolicyNetwork::action_probs(const Tensor& state) {
    Tensor logits = forward(state);
    return row_softmax(logits);
}

Tensor PolicyNetwork::backward(const Tensor& grad) {
    return net.backward(grad);
}

std::vector<Tensor*> PolicyNetwork::parameters() {
    return net.parameters();
}

std::vector<Tensor*> PolicyNetwork::gradients() {
    return net.gradients();
}

void PolicyNetwork::zero_grad() {
    auto grads = net.gradients();
    for (auto* g : grads)
        g->fill_zeros();
}

void PolicyNetwork::copy_params_from(const PolicyNetwork& other) {
    if (net.size() != other.net.size()) {
        throw std::runtime_error("PolicyNetwork::copy_params_from architecture mismatch");
    }
    for (size_t i = 0; i < other.net.size(); ++i) {
        auto src_params = other.net.get(i).parameters();  // const overload
        auto dst_layer_params = net.get(i).parameters();
        if (src_params.size() != dst_layer_params.size()) {
            throw std::runtime_error("PolicyNetwork::copy_params_from layer parameter count mismatch");
        }
        for (size_t p = 0; p < src_params.size(); ++p) {
            dst_layer_params[p]->data = src_params[p]->data;
            dst_layer_params[p]->rows = src_params[p]->rows;
            dst_layer_params[p]->cols = src_params[p]->cols;
        }
    }
}

// ============================================================
// ValueNetwork
// ============================================================

ValueNetwork::ValueNetwork(size_t input_size, size_t hidden_size,
                           std::mt19937& rng) {
    net.add(std::make_unique<Linear>(input_size, hidden_size, "value_fc1", rng));
    net.add(std::make_unique<ReLU>());
    net.add(std::make_unique<Linear>(hidden_size, hidden_size, "value_fc2", rng));
    net.add(std::make_unique<ReLU>());
    net.add(std::make_unique<Linear>(hidden_size, 1, "value_fc3", rng));
}

Tensor ValueNetwork::forward(const Tensor& state) {
    return net.forward(state);
}

Tensor ValueNetwork::backward(const Tensor& grad) {
    return net.backward(grad);
}

std::vector<Tensor*> ValueNetwork::parameters() {
    return net.parameters();
}

std::vector<Tensor*> ValueNetwork::gradients() {
    return net.gradients();
}

void ValueNetwork::zero_grad() {
    auto grads = net.gradients();
    for (auto* g : grads)
        g->fill_zeros();
}

// ============================================================
// A2C
// ============================================================

A2C::A2C(size_t state_size, size_t hidden_size, size_t num_actions,
         double gamma, double lr_policy, double lr_value, std::mt19937& rng)
    : policy(state_size, hidden_size, num_actions, rng),
      value(state_size, hidden_size, rng),
      gamma(gamma),
      policy_optimizer_(lr_policy),
      value_optimizer_(lr_value) {}

std::vector<int> A2C::select_actions(const Tensor& states, std::mt19937& rng) {
    Tensor logits = policy.forward(states);
    Tensor probs = row_softmax(logits);
    size_t batch = states.rows;
    size_t num_actions = probs.cols;
    std::vector<int> actions(batch);

    for (size_t i = 0; i < batch; ++i) {
        // Sample from the categorical distribution defined by probs[i, :]
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double u = dist(rng);
        double cumulative = 0.0;
        int chosen = (int)num_actions - 1; // fallback to last action
        for (size_t a = 0; a < num_actions; ++a) {
            cumulative += probs(i, a);
            if (u <= cumulative) {
                chosen = (int)a;
                break;
            }
        }
        actions[i] = chosen;
    }
    return actions;
}

void A2C::train_step(const Experience& exp) {
    size_t batch = exp.state.rows;
    size_t num_actions = policy.forward(exp.state).cols;

    // --- Value network update ---
    value.zero_grad();
    Tensor values = value.forward(exp.state);

    Tensor value_targets(batch, 1);
    AC_OMP_PAR_FOR
    for (size_t i = 0; i < batch; ++i)
        value_targets(i, 0) = exp.rewards[i];

    Tensor value_grad(batch, 1);
    AC_OMP_PAR_FOR
    for (size_t i = 0; i < batch; ++i)
        value_grad(i, 0) = 2.0 * (values(i, 0) - value_targets(i, 0)) / (double)batch;

    value.backward(value_grad);

    auto v_params = value.parameters();
    auto v_grads = value.gradients();
    value_optimizer_.step(v_params, v_grads);

    // --- Policy network update (REINFORCE with baseline) ---
    policy.zero_grad();
    Tensor logits = policy.forward(exp.state);

    std::vector<double> advantages(batch);
    AC_OMP_PAR_FOR
    for (size_t i = 0; i < batch; ++i)
        advantages[i] = exp.rewards[i] - values(i, 0);

    // Policy gradient: Loss = -E[A * log pi(a|s)]
    // dLoss/d_logits = A * (pi - 1_{a})
    Tensor probs = row_softmax(logits);
    Tensor policy_grad(batch, num_actions, 0.0);
    AC_OMP_PAR_FOR
    for (size_t i = 0; i < batch; ++i) {
        double adv = advantages[i];
        for (size_t a = 0; a < num_actions; ++a) {
            double indicator = (a == (size_t)exp.actions[i]) ? 1.0 : 0.0;
            policy_grad(i, a) = adv * (probs(i, a) - indicator) / (double)batch;
        }
    }

    policy.backward(policy_grad);

    auto p_params = policy.parameters();
    auto p_grads = policy.gradients();
    policy_optimizer_.step(p_params, p_grads);
}

std::vector<int> A2C::predict(const Tensor& states) {
    Tensor logits = policy.forward(states);
    size_t batch = states.rows;
    size_t num_actions = logits.cols;
    std::vector<int> actions(batch);
    AC_OMP_PAR_FOR
    for (size_t i = 0; i < batch; ++i) {
        int best = 0;
        double best_val = logits(i, 0);
        for (size_t a = 1; a < num_actions; ++a) {
            if (logits(i, a) > best_val) {
                best_val = logits(i, a);
                best = (int)a;
            }
        }
        actions[i] = best;
    }
    return actions;
}

std::vector<Tensor*> A2C::parameters() {
    std::vector<Tensor*> params;
    for (auto* p : policy.parameters()) params.push_back(p);
    for (auto* p : value.parameters()) params.push_back(p);
    return params;
}

std::vector<Tensor*> A2C::gradients() {
    std::vector<Tensor*> grads;
    for (auto* g : policy.gradients()) grads.push_back(g);
    for (auto* g : value.gradients()) grads.push_back(g);
    return grads;
}

// ============================================================
// PPO
// ============================================================

PPO::PPO(size_t state_size, size_t hidden_size, size_t num_actions,
         double gamma, double clip_eps, double lr, std::mt19937& rng)
    : policy(state_size, hidden_size, num_actions, rng),
      old_policy(state_size, hidden_size, num_actions, rng),
      value(state_size, hidden_size, rng),
      gamma(gamma), clip_eps(clip_eps),
      policy_optimizer_(lr),
      value_optimizer_(lr) {
    old_policy.copy_params_from(policy);
}

void PPO::update_old_policy() {
    old_policy.copy_params_from(policy);
}

std::vector<int> PPO::select_actions(const Tensor& states, std::mt19937& rng) {
    Tensor logits = policy.forward(states);
    Tensor probs = row_softmax(logits);
    size_t batch = states.rows;
    size_t num_actions = probs.cols;
    std::vector<int> actions(batch);

    for (size_t i = 0; i < batch; ++i) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double u = dist(rng);
        double cumulative = 0.0;
        int chosen = (int)num_actions - 1;
        for (size_t a = 0; a < num_actions; ++a) {
            cumulative += probs(i, a);
            if (u <= cumulative) {
                chosen = (int)a;
                break;
            }
        }
        actions[i] = chosen;
    }
    return actions;
}

void PPO::train_step(const A2C::Experience& exp) {
    size_t batch = exp.state.rows;

    Tensor old_logits = old_policy.forward(exp.state);
    Tensor old_probs = row_softmax(old_logits);

    // --- Value update ---
    value.zero_grad();
    Tensor values = value.forward(exp.state);

    Tensor value_targets(batch, 1);
    AC_OMP_PAR_FOR
    for (size_t i = 0; i < batch; ++i)
        value_targets(i, 0) = exp.rewards[i];

    Tensor value_grad(batch, 1);
    AC_OMP_PAR_FOR
    for (size_t i = 0; i < batch; ++i)
        value_grad(i, 0) = 2.0 * (values(i, 0) - value_targets(i, 0)) / (double)batch;

    value.backward(value_grad);

    auto v_params = value.parameters();
    auto v_grads = value.gradients();
    value_optimizer_.step(v_params, v_grads);

    // --- PPO clipped policy update ---
    policy.zero_grad();
    Tensor logits = policy.forward(exp.state);
    Tensor probs = row_softmax(logits);
    size_t num_actions = probs.cols;

    std::vector<double> advantages(batch);
    AC_OMP_PAR_FOR
    for (size_t i = 0; i < batch; ++i)
        advantages[i] = exp.rewards[i] - values(i, 0);

    // Normalize advantages (critical for PPO stability)
    if (batch > 1) {
        double adv_mean = 0.0;
        AC_OMP_PAR_FOR_REDUCE_SUM
        for (size_t i = 0; i < batch; ++i) adv_mean += advantages[i];
        adv_mean /= (double)batch;
        double adv_var = 0.0;
        AC_OMP_PAR_FOR_REDUCE_VAR
        for (size_t i = 0; i < batch; ++i)
            adv_var += (advantages[i] - adv_mean) * (advantages[i] - adv_mean);
        adv_var /= (double)batch;
        double adv_std = std::sqrt(adv_var + 1e-8);
        AC_OMP_PAR_FOR
        for (size_t i = 0; i < batch; ++i)
            advantages[i] = (advantages[i] - adv_mean) / adv_std;
    }

    Tensor policy_grad(batch, num_actions, 0.0);
    AC_OMP_PAR_FOR
    for (size_t i = 0; i < batch; ++i) {
        int ai = exp.actions[i];
        double pi_new = std::max(probs(i, (size_t)ai), 1e-12);
        double pi_old = std::max(old_probs(i, (size_t)ai), 1e-12);
        double ratio = pi_new / pi_old;
        double adv = advantages[i];

        double unclipped = ratio * adv;
        double clipped_ratio = std::clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps);
        double clipped = clipped_ratio * adv;
        double use_unclipped = (unclipped <= clipped) ? 1.0 : 0.0;

        for (size_t a = 0; a < num_actions; ++a) {
            double delta_ai = (a == (size_t)ai) ? 1.0 : 0.0;
            double d_ratio_d_logit = ratio * (delta_ai - probs(i, a));
            policy_grad(i, a) = -adv * use_unclipped * d_ratio_d_logit / (double)batch;
        }
    }

    policy.backward(policy_grad);

    auto p_params = policy.parameters();
    auto p_grads = policy.gradients();
    policy_optimizer_.step(p_params, p_grads);
}

std::vector<int> PPO::predict(const Tensor& states) {
    Tensor logits = policy.forward(states);
    size_t batch = states.rows;
    size_t num_actions = logits.cols;
    std::vector<int> actions(batch);
    AC_OMP_PAR_FOR
    for (size_t i = 0; i < batch; ++i) {
        int best = 0;
        double best_val = logits(i, 0);
        for (size_t a = 1; a < num_actions; ++a) {
            if (logits(i, a) > best_val) {
                best_val = logits(i, a);
                best = (int)a;
            }
        }
        actions[i] = best;
    }
    return actions;
}

std::vector<Tensor*> PPO::parameters() {
    std::vector<Tensor*> params;
    for (auto* p : policy.parameters()) params.push_back(p);
    for (auto* p : value.parameters()) params.push_back(p);
    return params;
}

std::vector<Tensor*> PPO::gradients() {
    std::vector<Tensor*> grads;
    for (auto* g : policy.gradients()) grads.push_back(g);
    for (auto* g : value.gradients()) grads.push_back(g);
    return grads;
}
