#include "core/benchmark/piecewise_bit_bridge_model.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace benchmark {

PiecewiseBitBridgeModel::PiecewiseBitBridgeModel(std::vector<std::unique_ptr<TrainableModel>> specialists,
                                                 BitBridgeConfig cfg,
                                                 std::mt19937& rng,
                                                 std::vector<std::string> owner_group_names)
    : owner_group_names_(std::move(owner_group_names)),
      action_dim_(cfg.action_dim) {
    if (specialists.empty()) {
        throw std::runtime_error("PiecewiseBitBridgeModel requires at least one specialist");
    }
    if (cfg.action_dim == 0) {
        throw std::runtime_error("PiecewiseBitBridgeModel requires action_dim > 0");
    }

    specialists_.reserve(specialists.size());
    for (auto& backbone : specialists) {
        if (!backbone) {
            throw std::runtime_error("PiecewiseBitBridgeModel received null specialist backbone");
        }
        specialists_.push_back(std::make_unique<BitBridgeModel>(std::move(backbone), cfg, rng));
    }

    owner_by_action_.assign(action_dim_, 0);
    num_owner_groups_ = static_cast<int>(std::max<std::size_t>(1, specialists_.size()));

    std::ostringstream oss;
    oss << "piecewise_bit_bridge[";
    for (std::size_t i = 0; i < specialists_.size(); ++i) {
        if (i) oss << "+";
        if (i < owner_group_names_.size() && !owner_group_names_[i].empty()) {
            oss << owner_group_names_[i];
        } else {
            oss << "group_" << i;
        }
    }
    oss << "]";
    id_ = oss.str();
}

std::size_t PiecewiseBitBridgeModel::specialist_index_for_action(std::size_t action) const {
    if (specialists_.empty()) return 0;
    if (action >= owner_by_action_.size()) return 0;
    const int owner = owner_by_action_[action];
    if (owner < 0) return 0;
    return static_cast<std::size_t>(owner) % specialists_.size();
}

Tensor PiecewiseBitBridgeModel::forward(const Tensor& input) {
    cached_logits_.clear();
    cached_logits_.reserve(specialists_.size());
    for (auto& specialist : specialists_) {
        cached_logits_.push_back(specialist->forward(input));
    }

    Tensor out(input.rows, action_dim_, 0.0);
    for (std::size_t r = 0; r < out.rows; ++r) {
        for (std::size_t a = 0; a < out.cols; ++a) {
            const std::size_t idx = specialist_index_for_action(a);
            out(r, a) = cached_logits_[idx](r, a);
        }
    }
    return out;
}

Tensor PiecewiseBitBridgeModel::backward(const Tensor& grad_output) {
    if (grad_output.cols != action_dim_) {
        throw std::runtime_error("PiecewiseBitBridgeModel::backward grad width mismatch");
    }

    Tensor grad_input;
    for (std::size_t i = 0; i < specialists_.size(); ++i) {
        Tensor grad_i(grad_output.rows, grad_output.cols, 0.0);
        for (std::size_t r = 0; r < grad_output.rows; ++r) {
            for (std::size_t a = 0; a < grad_output.cols; ++a) {
                if (specialist_index_for_action(a) == i) {
                    grad_i(r, a) = grad_output(r, a);
                }
            }
        }

        Tensor in_grad = specialists_[i]->backward(grad_i);
        if (in_grad.data.empty()) continue;
        if (grad_input.data.empty()) {
            grad_input = std::move(in_grad);
        } else if (grad_input.rows == in_grad.rows && grad_input.cols == in_grad.cols) {
            grad_input += in_grad;
        }
    }

    return grad_input;
}

std::vector<Tensor*> PiecewiseBitBridgeModel::parameters() {
    std::vector<Tensor*> out;
    for (auto& specialist : specialists_) {
        auto p = specialist->parameters();
        out.insert(out.end(), p.begin(), p.end());
    }
    return out;
}

std::vector<Tensor*> PiecewiseBitBridgeModel::gradients() {
    std::vector<Tensor*> out;
    for (auto& specialist : specialists_) {
        auto g = specialist->gradients();
        out.insert(out.end(), g.begin(), g.end());
    }
    return out;
}

void PiecewiseBitBridgeModel::zero_grad() {
    for (auto& specialist : specialists_) specialist->zero_grad();
}

void PiecewiseBitBridgeModel::train() {
    for (auto& specialist : specialists_) specialist->train();
}

void PiecewiseBitBridgeModel::eval() {
    for (auto& specialist : specialists_) specialist->eval();
}

void PiecewiseBitBridgeModel::set_action_owner_map(const std::vector<int>& owner_by_action,
                                                   int num_owner_groups) {
    num_owner_groups_ = std::max(1, num_owner_groups);
    owner_by_action_ = owner_by_action;
    if (owner_by_action_.size() < action_dim_) {
        owner_by_action_.resize(action_dim_, 0);
    } else if (owner_by_action_.size() > action_dim_) {
        owner_by_action_.resize(action_dim_);
    }

    for (int& owner : owner_by_action_) {
        if (owner < 0) continue;
        owner = owner % std::max(1, num_owner_groups_);
    }
}

} // namespace benchmark

