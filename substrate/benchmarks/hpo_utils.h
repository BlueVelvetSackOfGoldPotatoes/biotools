#pragma once

#include "core/losses/losses.h"
#include "core/tensor/tensor.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace hpo {

inline std::string to_lower(std::string s) {
    std::transform(
        s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
    );
    return s;
}

inline int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (!v) return fallback;
    try {
        return std::stoi(v);
    } catch (...) {
        return fallback;
    }
}

inline std::size_t env_size(const char* name, std::size_t fallback) {
    const char* v = std::getenv(name);
    if (!v) return fallback;
    try {
        return static_cast<std::size_t>(std::stoull(v));
    } catch (...) {
        return fallback;
    }
}

inline double env_double(const char* name, double fallback) {
    const char* v = std::getenv(name);
    if (!v) return fallback;
    try {
        return std::stod(v);
    } catch (...) {
        return fallback;
    }
}

inline std::string env_string(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    if (!v) return fallback;
    return std::string(v);
}

inline bool env_bool(const char* name, bool fallback) {
    const char* v = std::getenv(name);
    if (!v) return fallback;
    const std::string s = to_lower(std::string(v));
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

enum class OptimizerType {
    Adam,
    SGD,
    RMSProp
};

inline std::string optimizer_name(OptimizerType t) {
    switch (t) {
    case OptimizerType::Adam:
        return "adam";
    case OptimizerType::SGD:
        return "sgd";
    case OptimizerType::RMSProp:
        return "rmsprop";
    default:
        return "adam";
    }
}

inline OptimizerType parse_optimizer(const std::string& s, OptimizerType fallback = OptimizerType::Adam) {
    const std::string k = to_lower(s);
    if (k == "adam") return OptimizerType::Adam;
    if (k == "sgd") return OptimizerType::SGD;
    if (k == "rmsprop") return OptimizerType::RMSProp;
    return fallback;
}

enum class LossType {
    CrossEntropy,
    MSE,
    Huber,
    KLDiv
};

inline std::string loss_name(LossType t) {
    switch (t) {
    case LossType::CrossEntropy:
        return "cross_entropy";
    case LossType::MSE:
        return "mse";
    case LossType::Huber:
        return "huber";
    case LossType::KLDiv:
        return "kldiv";
    default:
        return "cross_entropy";
    }
}

inline LossType parse_loss(const std::string& s, LossType fallback = LossType::CrossEntropy) {
    const std::string k = to_lower(s);
    if (k == "cross_entropy" || k == "ce") return LossType::CrossEntropy;
    if (k == "mse") return LossType::MSE;
    if (k == "huber" || k == "smooth_l1") return LossType::Huber;
    if (k == "kldiv" || k == "kl") return LossType::KLDiv;
    return fallback;
}

inline std::unique_ptr<Loss> make_loss(LossType t) {
    switch (t) {
    case LossType::CrossEntropy:
        return std::make_unique<CrossEntropyLoss>();
    case LossType::MSE:
        return std::make_unique<MSELoss>();
    case LossType::Huber:
        return std::make_unique<HuberLoss>(1.0);
    case LossType::KLDiv:
        return std::make_unique<KLDivLoss>();
    default:
        return std::make_unique<CrossEntropyLoss>();
    }
}

struct OptimizerConfig {
    OptimizerType type = OptimizerType::Adam;
    double lr = 1e-3;
    double weight_decay = 0.0;

    // SGD
    double momentum = 0.0;

    // Adam
    double beta1 = 0.9;
    double beta2 = 0.999;

    // RMSProp
    double alpha = 0.99;

    double eps = 1e-8;
    double grad_clip_norm = 0.0;
};

class TensorParamOptimizer {
public:
    explicit TensorParamOptimizer(const OptimizerConfig& cfg) : cfg_(cfg) {}

    void step(const std::vector<Tensor*>& params, const std::vector<Tensor*>& grads) {
        if (params.size() != grads.size()) {
            throw std::runtime_error("TensorParamOptimizer: params/grads size mismatch");
        }
        if (params.empty()) return;

        ensure_state(params, grads);
        const double grad_scale = compute_grad_scale(grads);
        t_++;

        for (std::size_t i = 0; i < params.size(); ++i) {
            Tensor& p = *params[i];
            const Tensor& g = *grads[i];

            switch (cfg_.type) {
            case OptimizerType::SGD:
                step_sgd(i, p, g, grad_scale);
                break;
            case OptimizerType::Adam:
                step_adam(i, p, g, grad_scale);
                break;
            case OptimizerType::RMSProp:
                step_rmsprop(i, p, g, grad_scale);
                break;
            }
        }
    }

    static void zero_grad(const std::vector<Tensor*>& grads) {
        for (auto* g : grads) {
            if (g) g->fill_zeros();
        }
    }

private:
    OptimizerConfig cfg_;
    bool initialized_ = false;
    int t_ = 0;
    std::vector<Tensor> m_;
    std::vector<Tensor> v_;
    std::vector<Tensor> velocity_;
    std::vector<Tensor> sq_avg_;

    void ensure_state(const std::vector<Tensor*>& params, const std::vector<Tensor*>& grads) {
        if (initialized_) return;
        m_.reserve(params.size());
        v_.reserve(params.size());
        velocity_.reserve(params.size());
        sq_avg_.reserve(params.size());

        for (std::size_t i = 0; i < params.size(); ++i) {
            if (!params[i] || !grads[i] || !params[i]->same_shape(*grads[i])) {
                throw std::runtime_error("TensorParamOptimizer: invalid tensor pointers/shapes");
            }
            m_.push_back(Tensor::zeros(params[i]->rows, params[i]->cols));
            v_.push_back(Tensor::zeros(params[i]->rows, params[i]->cols));
            velocity_.push_back(Tensor::zeros(params[i]->rows, params[i]->cols));
            sq_avg_.push_back(Tensor::zeros(params[i]->rows, params[i]->cols));
        }
        initialized_ = true;
    }

    double compute_grad_scale(const std::vector<Tensor*>& grads) const {
        if (cfg_.grad_clip_norm <= 0.0) return 1.0;
        double sq = 0.0;
        for (const auto* g : grads) {
            if (!g) continue;
            const double n = g->norm();
            sq += n * n;
        }
        const double total = std::sqrt(sq);
        if (total <= cfg_.grad_clip_norm) return 1.0;
        return cfg_.grad_clip_norm / (total + 1e-12);
    }

    inline double weight_decay_term(double w) const {
        return cfg_.weight_decay > 0.0 ? cfg_.weight_decay * w : 0.0;
    }

    void step_sgd(std::size_t idx, Tensor& p, const Tensor& g, double grad_scale) {
        Tensor& vel = velocity_[idx];
        for (std::size_t k = 0; k < p.data.size(); ++k) {
            double grad = g.data[k] * grad_scale + weight_decay_term(p.data[k]);
            if (!std::isfinite(grad)) grad = 0.0;
            if (cfg_.momentum > 0.0) {
                vel.data[k] = cfg_.momentum * vel.data[k] + grad;
                p.data[k] -= cfg_.lr * vel.data[k];
            } else {
                p.data[k] -= cfg_.lr * grad;
            }
        }
    }

    void step_adam(std::size_t idx, Tensor& p, const Tensor& g, double grad_scale) {
        Tensor& m = m_[idx];
        Tensor& v = v_[idx];
        const double bc1 = 1.0 - std::pow(cfg_.beta1, t_);
        const double bc2 = 1.0 - std::pow(cfg_.beta2, t_);
        for (std::size_t k = 0; k < p.data.size(); ++k) {
            double grad = g.data[k] * grad_scale + weight_decay_term(p.data[k]);
            if (!std::isfinite(grad)) grad = 0.0;
            m.data[k] = cfg_.beta1 * m.data[k] + (1.0 - cfg_.beta1) * grad;
            v.data[k] = cfg_.beta2 * v.data[k] + (1.0 - cfg_.beta2) * grad * grad;
            const double m_hat = m.data[k] / bc1;
            const double v_hat = v.data[k] / bc2;
            p.data[k] -= cfg_.lr * m_hat / (std::sqrt(v_hat) + cfg_.eps);
        }
    }

    void step_rmsprop(std::size_t idx, Tensor& p, const Tensor& g, double grad_scale) {
        Tensor& sq = sq_avg_[idx];
        for (std::size_t k = 0; k < p.data.size(); ++k) {
            double grad = g.data[k] * grad_scale + weight_decay_term(p.data[k]);
            if (!std::isfinite(grad)) grad = 0.0;
            sq.data[k] = cfg_.alpha * sq.data[k] + (1.0 - cfg_.alpha) * grad * grad;
            p.data[k] -= cfg_.lr * grad / (std::sqrt(sq.data[k]) + cfg_.eps);
        }
    }
};

inline std::vector<std::string> split_csv(const std::string& csv) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : csv) {
        if (ch == ',') {
            if (!cur.empty()) out.push_back(to_lower(cur));
            cur.clear();
        } else if (!std::isspace(static_cast<unsigned char>(ch))) {
            cur.push_back(ch);
        }
    }
    if (!cur.empty()) out.push_back(to_lower(cur));
    return out;
}

} // namespace hpo
