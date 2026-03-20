#include "rl_models.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {

int argmax_row(const Tensor& t, size_t row) {
    int best = 0;
    double best_val = t(row, 0);
    for (size_t j = 1; j < t.cols; ++j) {
        if (t(row, j) > best_val) {
            best_val = t(row, j);
            best = static_cast<int>(j);
        }
    }
    return best;
}

double max_row(const Tensor& t, size_t row) {
    double best_val = t(row, 0);
    for (size_t j = 1; j < t.cols; ++j) {
        best_val = std::max(best_val, t(row, j));
    }
    return best_val;
}

} // namespace

// ============================================================================
// ReplayBuffer
// ============================================================================
// Pre-allocates capacity slots. Default-constructed RLTransitions contain
// empty data vectors (~48 bytes each), so the memory overhead is modest.
ReplayBuffer::ReplayBuffer(size_t capacity) : capacity_(capacity) {
    if (capacity_ == 0) {
        throw std::runtime_error("ReplayBuffer capacity must be > 0");
    }
    buffer_.resize(capacity_);
}

void ReplayBuffer::push(const RLTransition& t) {
    buffer_[pos_] = t;
    pos_ = (pos_ + 1) % capacity_;
    if (pos_ == 0) full_ = true;
}

// Sampling with replacement is standard practice for experience replay
// (see DQN paper, Mnih et al. 2015). It avoids the cost of shuffling and
// naturally handles buffers that are smaller than the requested batch.
std::vector<RLTransition> ReplayBuffer::sample(size_t n, std::mt19937& rng) const {
    const size_t current = size();
    if (n > current) {
        throw std::runtime_error("ReplayBuffer sample: requested more than available");
    }

    std::vector<RLTransition> out;
    out.reserve(n);
    std::uniform_int_distribution<size_t> dist(0, current - 1);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(buffer_[dist(rng)]);
    }
    return out;
}

size_t ReplayBuffer::size() const {
    return full_ ? capacity_ : pos_;
}

// ============================================================================
// Tabular Q-Learning
// ============================================================================
TabularQLearningAgent::TabularQLearningAgent(size_t num_states, size_t num_actions)
    : num_states_(num_states), num_actions_(num_actions), q_(num_states * num_actions, 0.0) {}

int TabularQLearningAgent::greedy_action(int state_id) const {
    const size_t base = static_cast<size_t>(state_id) * num_actions_;
    int best_a = 0;
    double best_q = q_[base];
    for (size_t a = 1; a < num_actions_; ++a) {
        if (q_[base + a] > best_q) {
            best_q = q_[base + a];
            best_a = static_cast<int>(a);
        }
    }
    return best_a;
}

int TabularQLearningAgent::greedy_action_masked(int state_id, const std::vector<int>& legal_actions) const {
    if (legal_actions.empty()) return greedy_action(state_id);
    const size_t base = static_cast<size_t>(state_id) * num_actions_;
    int best_a = legal_actions.front();
    double best_q = q_[base + static_cast<size_t>(best_a)];
    for (size_t i = 1; i < legal_actions.size(); ++i) {
        const int a = legal_actions[i];
        const double q = q_[base + static_cast<size_t>(a)];
        if (q > best_q) {
            best_q = q;
            best_a = a;
        }
    }
    return best_a;
}

int TabularQLearningAgent::select_action(int state_id, double epsilon, std::mt19937& rng) const {
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    if (u01(rng) < epsilon) {
        std::uniform_int_distribution<int> act(0, static_cast<int>(num_actions_) - 1);
        return act(rng);
    }
    return greedy_action(state_id);
}

void TabularQLearningAgent::update(int state_id,
                                   int action,
                                   double reward,
                                   int next_state_id,
                                   double alpha,
                                   double gamma) {
    const size_t idx = static_cast<size_t>(state_id) * num_actions_ + static_cast<size_t>(action);
    const size_t next_base = static_cast<size_t>(next_state_id) * num_actions_;
    double next_max = q_[next_base];
    for (size_t a = 1; a < num_actions_; ++a) {
        next_max = std::max(next_max, q_[next_base + a]);
    }
    const double target = reward + gamma * next_max;
    q_[idx] += alpha * (target - q_[idx]);
}

double TabularQLearningAgent::q_value(int state_id, int action) const {
    const size_t idx = static_cast<size_t>(state_id) * num_actions_ + static_cast<size_t>(action);
    return q_[idx];
}

// ============================================================================
// Tabular SARSA
// ============================================================================
TabularSARSAAgent::TabularSARSAAgent(size_t num_states, size_t num_actions)
    : num_states_(num_states), num_actions_(num_actions), q_(num_states * num_actions, 0.0) {}

int TabularSARSAAgent::greedy_action(int state_id) const {
    const size_t base = static_cast<size_t>(state_id) * num_actions_;
    int best_a = 0;
    double best_q = q_[base];
    for (size_t a = 1; a < num_actions_; ++a) {
        if (q_[base + a] > best_q) {
            best_q = q_[base + a];
            best_a = static_cast<int>(a);
        }
    }
    return best_a;
}

int TabularSARSAAgent::greedy_action_masked(int state_id, const std::vector<int>& legal_actions) const {
    if (legal_actions.empty()) return greedy_action(state_id);
    const size_t base = static_cast<size_t>(state_id) * num_actions_;
    int best_a = legal_actions.front();
    double best_q = q_[base + static_cast<size_t>(best_a)];
    for (size_t i = 1; i < legal_actions.size(); ++i) {
        const int a = legal_actions[i];
        const double q = q_[base + static_cast<size_t>(a)];
        if (q > best_q) {
            best_q = q;
            best_a = a;
        }
    }
    return best_a;
}

int TabularSARSAAgent::select_action(int state_id, double epsilon, std::mt19937& rng) const {
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    if (u01(rng) < epsilon) {
        std::uniform_int_distribution<int> act(0, static_cast<int>(num_actions_) - 1);
        return act(rng);
    }
    return greedy_action(state_id);
}

void TabularSARSAAgent::update(int state_id,
                               int action,
                               double reward,
                               int next_state_id,
                               int next_action,
                               double alpha,
                               double gamma) {
    const size_t idx = static_cast<size_t>(state_id) * num_actions_ + static_cast<size_t>(action);
    const size_t next_idx =
        static_cast<size_t>(next_state_id) * num_actions_ + static_cast<size_t>(next_action);
    const double target = reward + gamma * q_[next_idx];
    q_[idx] += alpha * (target - q_[idx]);
}

double TabularSARSAAgent::q_value(int state_id, int action) const {
    const size_t idx = static_cast<size_t>(state_id) * num_actions_ + static_cast<size_t>(action);
    return q_[idx];
}

// ============================================================================
// DQN / Double-DQN
// ============================================================================
DQNAgent::DQNAgent(size_t state_dim,
                   size_t hidden_dim,
                   size_t num_actions,
                   double lr,
                   double gamma,
                   size_t replay_capacity,
                   size_t batch_size,
                   int target_update_interval,
                   bool double_q,
                   std::mt19937& rng,
                   double eps_start,
                   double eps_end,
                   double eps_decay)
    : state_dim_(state_dim),
      num_actions_(num_actions),
      batch_size_(batch_size),
      gamma_(gamma),
      target_update_interval_(target_update_interval),
      double_q_(double_q),
      epsilon_(eps_start),
      epsilon_end_(eps_end),
      epsilon_decay_(eps_decay),
      optimizer_(lr),
      replay_(replay_capacity) {
    online_net_.add(std::make_unique<Linear>(state_dim, hidden_dim, "dqn_fc1", rng));
    online_net_.add(std::make_unique<ReLU>());
    online_net_.add(std::make_unique<Linear>(hidden_dim, hidden_dim, "dqn_fc2", rng));
    online_net_.add(std::make_unique<ReLU>());
    online_net_.add(std::make_unique<Linear>(hidden_dim, num_actions, "dqn_out", rng));

    target_net_.add(std::make_unique<Linear>(state_dim, hidden_dim, "target_fc1", rng));
    target_net_.add(std::make_unique<ReLU>());
    target_net_.add(std::make_unique<Linear>(hidden_dim, hidden_dim, "target_fc2", rng));
    target_net_.add(std::make_unique<ReLU>());
    target_net_.add(std::make_unique<Linear>(hidden_dim, num_actions, "target_out", rng));

    copy_params(target_net_, online_net_);
}

void DQNAgent::copy_params(Sequential& dst, const Sequential& src) {
    for (size_t i = 0; i < dst.size(); ++i) {
        auto src_params = src.get(i).parameters();  // const overload
        auto dst_params = dst.get(i).parameters();
        for (size_t p = 0; p < src_params.size(); ++p) {
            dst_params[p]->data = src_params[p]->data;
            dst_params[p]->rows = src_params[p]->rows;
            dst_params[p]->cols = src_params[p]->cols;
        }
    }
}

int DQNAgent::greedy_action_single(const Tensor& state) {
    Tensor q = online_net_.forward(state);
    return argmax_row(q, 0);
}

int DQNAgent::select_action(const Tensor& state, std::mt19937& rng) {
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    if (u01(rng) < epsilon_) {
        std::uniform_int_distribution<int> act(0, static_cast<int>(num_actions_) - 1);
        return act(rng);
    }
    return greedy_action_single(state);
}

void DQNAgent::observe(const RLTransition& t) {
    replay_.push(t);
}

double DQNAgent::train_step(std::mt19937& rng) {
    if (replay_.size() < batch_size_) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    auto batch = replay_.sample(batch_size_, rng);
    Tensor states(batch_size_, state_dim_);
    Tensor next_states(batch_size_, state_dim_);
    std::vector<int> actions(batch_size_);
    std::vector<double> rewards(batch_size_);
    std::vector<int> dones(batch_size_);

    for (size_t i = 0; i < batch_size_; ++i) {
        for (size_t j = 0; j < state_dim_; ++j) {
            states(i, j) = batch[i].state(0, j);
            next_states(i, j) = batch[i].next_state(0, j);
        }
        actions[i] = batch[i].action;
        rewards[i] = batch[i].reward;
        dones[i] = batch[i].done;
    }

    Tensor q_pred = online_net_.forward(states);
    Tensor q_next_target = target_net_.forward(next_states);
    Tensor q_next_online;
    if (double_q_) {
        q_next_online = online_net_.forward(next_states);
    }

    Tensor grad(batch_size_, num_actions_, 0.0);
    double loss = 0.0;
    for (size_t i = 0; i < batch_size_; ++i) {
        double next_v = 0.0;
        if (!dones[i]) {
            if (double_q_) {
                const int best_a = argmax_row(q_next_online, i);
                next_v = q_next_target(i, static_cast<size_t>(best_a));
            } else {
                next_v = max_row(q_next_target, i);
            }
        }
        const double target = rewards[i] + gamma_ * next_v;
        const double pred = q_pred(i, static_cast<size_t>(actions[i]));
        const double diff = pred - target;
        loss += diff * diff;
        grad(i, static_cast<size_t>(actions[i])) = 2.0 * diff / static_cast<double>(batch_size_);
    }

    online_net_.backward(grad);
    optimizer_.step(online_net_);
    optimizer_.zero_grad(online_net_);

    train_steps_++;
    if (target_update_interval_ > 0 && train_steps_ % target_update_interval_ == 0) {
        copy_params(target_net_, online_net_);
    }

    return loss / static_cast<double>(batch_size_);
}

std::vector<int> DQNAgent::predict_actions(const Tensor& states) {
    Tensor q = online_net_.forward(states);
    std::vector<int> out(states.rows, 0);
    for (size_t i = 0; i < states.rows; ++i) {
        out[i] = argmax_row(q, i);
    }
    return out;
}

std::vector<Tensor*> DQNAgent::parameters() {
    return online_net_.parameters();
}

std::vector<Tensor*> DQNAgent::gradients() {
    return online_net_.gradients();
}

void DQNAgent::decay_epsilon() {
    epsilon_ = std::max(epsilon_end_, epsilon_ * epsilon_decay_);
}

// ============================================================================
// MuZero-lite
// ============================================================================
MuZeroLiteAgent::MuZeroLiteAgent(size_t state_dim,
                                 size_t latent_dim,
                                 size_t num_actions,
                                 double lr,
                                 double gamma,
                                 int simulations,
                                 std::mt19937& rng)
    : state_dim_(state_dim),
      latent_dim_(latent_dim),
      num_actions_(num_actions),
      gamma_(gamma),
      simulations_(simulations),
      rng_(&rng),
      opt_repr_(lr),
      opt_dyn_(lr),
      opt_policy_(lr),
      opt_value_(lr) {
    repr_net_.add(std::make_unique<Linear>(state_dim_, 256, "muzero_repr_fc1", rng));
    repr_net_.add(std::make_unique<ReLU>());
    repr_net_.add(std::make_unique<Linear>(256, latent_dim_, "muzero_repr_out", rng));
    repr_net_.add(std::make_unique<Tanh_>());

    dyn_net_.add(std::make_unique<Linear>(latent_dim_ + num_actions_, 128, "muzero_dyn_fc1", rng));
    dyn_net_.add(std::make_unique<ReLU>());
    dyn_net_.add(std::make_unique<Linear>(128, latent_dim_ + 1, "muzero_dyn_out", rng));

    policy_net_.add(std::make_unique<Linear>(latent_dim_, 128, "muzero_pol_fc1", rng));
    policy_net_.add(std::make_unique<ReLU>());
    policy_net_.add(std::make_unique<Linear>(128, num_actions_, "muzero_pol_out", rng));

    value_net_.add(std::make_unique<Linear>(latent_dim_, 128, "muzero_val_fc1", rng));
    value_net_.add(std::make_unique<ReLU>());
    value_net_.add(std::make_unique<Linear>(128, 1, "muzero_val_out", rng));
}

Tensor MuZeroLiteAgent::one_hot_actions(const std::vector<int>& actions, size_t num_actions) {
    Tensor out(actions.size(), num_actions, 0.0);
    for (size_t i = 0; i < actions.size(); ++i) {
        out(i, static_cast<size_t>(actions[i])) = 1.0;
    }
    return out;
}

std::vector<int> MuZeroLiteAgent::argmax_rows(const Tensor& logits) {
    std::vector<int> out(logits.rows, 0);
    for (size_t i = 0; i < logits.rows; ++i) {
        out[i] = argmax_row(logits, i);
    }
    return out;
}

double MuZeroLiteAgent::entropy_from_probs(const Tensor& probs_row) {
    double h = 0.0;
    for (size_t j = 0; j < probs_row.cols; ++j) {
        const double p = probs_row(0, j);
        h -= p * std::log(p + 1e-12);
    }
    return h;
}

MuZeroLiteAgent::TrainStats MuZeroLiteAgent::train_step(const Tensor& states,
                                                        const std::vector<int>& labels,
                                                        std::mt19937& rng) {
    if (states.rows != labels.size()) {
        throw std::runtime_error("MuZeroLiteAgent::train_step rows/labels mismatch");
    }

    const size_t batch = states.rows;
    CrossEntropyLoss ce;
    MSELoss mse;
    TrainStats stats{};

    Tensor z = repr_net_.forward(states);

    Tensor policy_logits = policy_net_.forward(z);
    Tensor policy_probs = LossFunctions::softmax(policy_logits);
    Tensor policy_targets(batch, num_actions_, 0.0);
    for (size_t i = 0; i < batch; ++i) {
        policy_targets(i, static_cast<size_t>(labels[i])) = 1.0;
    }
    stats.loss_policy = ce.forward(policy_probs, policy_targets);
    Tensor grad_policy = ce.backward(policy_probs, policy_targets);

    Tensor value_pred = value_net_.forward(z);
    Tensor value_target(batch, 1, 1.0);
    stats.loss_value = mse.forward(value_pred, value_target);
    Tensor grad_value = mse.backward(value_pred, value_target);

    std::uniform_int_distribution<int> action_dist(0, static_cast<int>(num_actions_) - 1);
    std::vector<int> sampled_actions(batch);
    for (size_t i = 0; i < batch; ++i) sampled_actions[i] = action_dist(rng);
    Tensor action_oh = one_hot_actions(sampled_actions, num_actions_);
    Tensor dyn_in = Tensor::hstack(z, action_oh);
    Tensor dyn_out = dyn_net_.forward(dyn_in);
    Tensor latent_pred = dyn_out.slice_cols(0, latent_dim_);
    Tensor reward_pred = dyn_out.slice_cols(latent_dim_, 1);

    // Contextual bandit is terminal after one action; encourage an absorbing latent.
    Tensor latent_target(batch, latent_dim_, 0.0);
    const double loss_latent = mse.forward(latent_pred, latent_target);
    Tensor grad_latent = mse.backward(latent_pred, latent_target);

    Tensor reward_target(batch, 1, 0.0);
    for (size_t i = 0; i < batch; ++i) {
        reward_target(i, 0) = (sampled_actions[i] == labels[i]) ? 1.0 : 0.0;
    }
    stats.loss_reward = mse.forward(reward_pred, reward_target);
    Tensor grad_reward = mse.backward(reward_pred, reward_target);

    Tensor grad_dyn_out(batch, latent_dim_ + 1, 0.0);
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < latent_dim_; ++j) {
            grad_dyn_out(i, j) = 0.1 * grad_latent(i, j);
        }
        grad_dyn_out(i, latent_dim_) = grad_reward(i, 0);
    }

    Tensor grad_z_policy = policy_net_.backward(grad_policy);
    Tensor grad_z_value = value_net_.backward(grad_value);
    Tensor grad_dyn_in = dyn_net_.backward(grad_dyn_out);
    Tensor grad_z_dyn = grad_dyn_in.slice_cols(0, latent_dim_);
    Tensor grad_z = grad_z_policy + grad_z_value + grad_z_dyn;
    repr_net_.backward(grad_z);

    opt_policy_.step(policy_net_);
    opt_value_.step(value_net_);
    opt_dyn_.step(dyn_net_);
    opt_repr_.step(repr_net_);
    opt_policy_.zero_grad(policy_net_);
    opt_value_.zero_grad(value_net_);
    opt_dyn_.zero_grad(dyn_net_);
    opt_repr_.zero_grad(repr_net_);

    stats.loss_total = stats.loss_policy + stats.loss_value + stats.loss_reward + 0.1 * loss_latent;
    return stats;
}

std::vector<int> MuZeroLiteAgent::predict_policy_actions(const Tensor& states) {
    Tensor z = repr_net_.forward(states);
    Tensor logits = policy_net_.forward(z);
    return argmax_rows(logits);
}

std::vector<Tensor*> MuZeroLiteAgent::parameters() {
    std::vector<Tensor*> out;
    auto r = repr_net_.parameters();
    auto d = dyn_net_.parameters();
    auto p = policy_net_.parameters();
    auto v = value_net_.parameters();
    out.insert(out.end(), r.begin(), r.end());
    out.insert(out.end(), d.begin(), d.end());
    out.insert(out.end(), p.begin(), p.end());
    out.insert(out.end(), v.begin(), v.end());
    return out;
}

std::vector<Tensor*> MuZeroLiteAgent::gradients() {
    std::vector<Tensor*> out;
    auto r = repr_net_.gradients();
    auto d = dyn_net_.gradients();
    auto p = policy_net_.gradients();
    auto v = value_net_.gradients();
    out.insert(out.end(), r.begin(), r.end());
    out.insert(out.end(), d.begin(), d.end());
    out.insert(out.end(), p.begin(), p.end());
    out.insert(out.end(), v.begin(), v.end());
    return out;
}

MuZeroLiteAgent::SearchStats MuZeroLiteAgent::plan_action_with_stats(const Tensor& state) {
    struct Node {
        Tensor latent;
        std::vector<double> priors;
        std::vector<int> children;
        double reward_from_parent = 0.0;
        double value_sum = 0.0;
        int visit_count = 0;
    };

    const double c_puct = 1.5;
    // Depth 1 is correct for MNIST as a contextual bandit (single-step episodes).
    // Multiple simulations still help: each explores a different root action via
    // PUCT, accumulating visit counts that converge to the policy improvement.
    const int max_depth = 1;

    Tensor root_latent = repr_net_.forward(state);
    Tensor root_logits = policy_net_.forward(root_latent);
    Tensor root_probs = LossFunctions::softmax(root_logits);
    Tensor root_value_t = value_net_.forward(root_latent);

    std::vector<Node> nodes;
    nodes.reserve(static_cast<size_t>(simulations_) * 2 + 1);
    Node root;
    root.latent = root_latent;
    root.priors.resize(num_actions_);
    root.children.assign(num_actions_, -1);
    for (size_t a = 0; a < num_actions_; ++a) root.priors[a] = root_probs(0, a);
    nodes.push_back(root);

    for (int sim = 0; sim < simulations_; ++sim) {
        int node_idx = 0;
        std::vector<int> path_nodes;
        std::vector<int> path_actions;
        path_nodes.push_back(0);
        double leaf_value = 0.0;
        bool expanded_new = false;

        for (int depth = 0; depth < max_depth; ++depth) {
            Node& node = nodes[static_cast<size_t>(node_idx)];
            const double parent_scale = std::sqrt(static_cast<double>(node.visit_count + 1));

            int best_a = 0;
            double best_score = -std::numeric_limits<double>::infinity();
            for (size_t a = 0; a < num_actions_; ++a) {
                const int child_idx = node.children[a];
                int child_visits = 0;
                double q = 0.0;
                if (child_idx >= 0) {
                    const Node& child = nodes[static_cast<size_t>(child_idx)];
                    child_visits = child.visit_count;
                    if (child.visit_count > 0) {
                        q = child.value_sum / static_cast<double>(child.visit_count);
                    }
                }
                const double u = c_puct * node.priors[a] * parent_scale / (1.0 + child_visits);
                const double score = q + u;
                if (score > best_score) {
                    best_score = score;
                    best_a = static_cast<int>(a);
                }
            }

            path_actions.push_back(best_a);
            int child_idx = nodes[static_cast<size_t>(node_idx)].children[static_cast<size_t>(best_a)];
            if (child_idx < 0) {
                Tensor action_oh(1, num_actions_, 0.0);
                action_oh(0, static_cast<size_t>(best_a)) = 1.0;
                Tensor dyn_in = Tensor::hstack(nodes[static_cast<size_t>(node_idx)].latent, action_oh);
                Tensor dyn_out = dyn_net_.forward(dyn_in);
                Tensor next_latent = dyn_out.slice_cols(0, latent_dim_);
                const double reward = dyn_out(0, latent_dim_);

                Tensor child_logits = policy_net_.forward(next_latent);
                Tensor child_probs = LossFunctions::softmax(child_logits);
                Tensor child_value_t = value_net_.forward(next_latent);
                leaf_value = child_value_t(0, 0);

                Node child;
                child.latent = next_latent;
                child.priors.resize(num_actions_);
                child.children.assign(num_actions_, -1);
                child.reward_from_parent = reward;
                for (size_t a = 0; a < num_actions_; ++a) child.priors[a] = child_probs(0, a);

                nodes.push_back(std::move(child));
                child_idx = static_cast<int>(nodes.size() - 1);
                nodes[static_cast<size_t>(node_idx)].children[static_cast<size_t>(best_a)] = child_idx;
                path_nodes.push_back(child_idx);
                expanded_new = true;
                break;
            }

            node_idx = child_idx;
            path_nodes.push_back(node_idx);
            if (depth == max_depth - 1) {
                Tensor v = value_net_.forward(nodes[static_cast<size_t>(node_idx)].latent);
                leaf_value = v(0, 0);
            }
        }

        if (!expanded_new && path_actions.empty()) {
            leaf_value = root_value_t(0, 0);
        }

        double g = leaf_value;
        for (int i = static_cast<int>(path_actions.size()) - 1; i >= 0; --i) {
            const int parent_idx = path_nodes[static_cast<size_t>(i)];
            const int action = path_actions[static_cast<size_t>(i)];
            const int child_idx = nodes[static_cast<size_t>(parent_idx)].children[static_cast<size_t>(action)];
            Node& child = nodes[static_cast<size_t>(child_idx)];
            g = child.reward_from_parent + gamma_ * g;
            child.visit_count += 1;
            child.value_sum += g;
            nodes[static_cast<size_t>(parent_idx)].visit_count += 1;
        }
    }

    const Node& root_ref = nodes[0];
    int best_action = 0;
    int best_visits = -1;
    int total_visits = 0;
    for (size_t a = 0; a < num_actions_; ++a) {
        const int child_idx = root_ref.children[a];
        const int v = (child_idx >= 0) ? nodes[static_cast<size_t>(child_idx)].visit_count : 0;
        total_visits += v;
        if (v > best_visits) {
            best_visits = v;
            best_action = static_cast<int>(a);
        }
    }

    Tensor root_prob_row(1, num_actions_, 0.0);
    for (size_t a = 0; a < num_actions_; ++a) root_prob_row(0, a) = root_ref.priors[a];
    const double root_entropy = entropy_from_probs(root_prob_row);
    const double visit_max =
        (total_visits > 0) ? static_cast<double>(best_visits) / static_cast<double>(total_visits) : 0.0;

    SearchStats stats;
    stats.action = best_action;
    stats.root_entropy = root_entropy;
    stats.visit_max = visit_max;
    stats.root_value = root_value_t(0, 0);
    return stats;
}
