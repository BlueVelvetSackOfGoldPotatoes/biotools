#include "optimizer.h"
#include <stdexcept>

#ifdef _OPENMP
#define ADAM_OMP_PAR_FOR _Pragma("omp parallel for schedule(static)")
#else
#define ADAM_OMP_PAR_FOR
#endif

void Optimizer::zero_grad(Sequential& model) {
    for (auto* g : model.gradients()) g->fill_zeros();
}

// ============================================================
// SGD with momentum and weight decay
// ============================================================
SGDOptimizer::SGDOptimizer(double lr, double momentum, double weight_decay)
    : momentum_(momentum), weight_decay_(weight_decay) { lr_ = lr; }

void SGDOptimizer::step(Sequential& model) {
    auto params = model.parameters();
    auto grads = model.gradients();

    if (!initialized_ && momentum_ > 0) {
        velocity_params.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i)
            velocity_params[i] = Tensor::zeros(params[i]->rows, params[i]->cols);
        initialized_ = true;
    }

    for (size_t i = 0; i < params.size(); ++i) {
        Tensor& p = *params[i];
        Tensor g = *grads[i]; // copy to avoid mutating model gradients

        if (weight_decay_ > 0) {
            g += p * weight_decay_;
        }

        if (momentum_ > 0) {
            velocity_params[i] = velocity_params[i] * momentum_ + g;
            p -= velocity_params[i] * lr_;
        } else {
            p -= g * lr_;
        }
    }
}

// ============================================================
// Adam
// ============================================================
Adam::Adam(double lr, double beta1, double beta2, double eps, double weight_decay)
    : beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay) { lr_ = lr; }

void Adam::step(Sequential& model) {
    auto params = model.parameters();
    auto grads = model.gradients();
    step(params, grads);
}

void Adam::step(const std::vector<Tensor*>& params,
                const std::vector<Tensor*>& grads,
                double lr_scale,
                double clip_global_grad_norm) {
    if (params.size() != grads.size()) {
        throw std::runtime_error("Adam::step params/grads size mismatch");
    }

    if (!initialized_) {
        param_ptrs_.clear();
        m_params.reserve(params.size());
        v_params.reserve(params.size());
        for (std::size_t i = 0; i < params.size(); ++i) {
            if (!params[i] || !grads[i]) {
                throw std::runtime_error("Adam::step null tensor pointer");
            }
            if (!params[i]->same_shape(*grads[i])) {
                throw std::runtime_error("Adam::step shape mismatch");
            }
            param_ptrs_.push_back(params[i]);
            m_params.push_back(Tensor::zeros(params[i]->rows, params[i]->cols));
            v_params.push_back(Tensor::zeros(params[i]->rows, params[i]->cols));
        }
        initialized_ = true;
    } else {
        if (params.size() != m_params.size() || params.size() != param_ptrs_.size()) {
            throw std::runtime_error("Adam::step parameter list changed after initialization");
        }
        for (std::size_t i = 0; i < params.size(); ++i) {
            if (params[i] != param_ptrs_[i]) {
                throw std::runtime_error("Adam::step parameter pointer order changed");
            }
        }
    }

    double grad_scale = 1.0;
    if (clip_global_grad_norm > 0.0) {
        double total_sq = 0.0;
        for (const auto* g : grads) {
            const double n = g->norm();
            total_sq += n * n;
        }
        const double total = std::sqrt(total_sq);
        if (total > clip_global_grad_norm) {
            grad_scale = clip_global_grad_norm / (total + 1e-12);
        }
    }

    t_ += 1;
    const double bc1 = 1.0 - std::pow(beta1_, t_);
    const double bc2 = 1.0 - std::pow(beta2_, t_);
    const double effective_lr = lr_ * lr_scale;

    for (std::size_t i = 0; i < params.size(); ++i) {
        Tensor& p = *params[i];
        const Tensor& g = *grads[i];
        Tensor& m = m_params[i];
        Tensor& v = v_params[i];

        ADAM_OMP_PAR_FOR
        for (std::size_t k = 0; k < p.data.size(); ++k) {
            double grad = g.data[k] * grad_scale;
            if (!std::isfinite(grad)) grad = 0.0;
            if (weight_decay_ > 0.0) {
                grad += weight_decay_ * p.data[k];
            }

            m.data[k] = beta1_ * m.data[k] + (1.0 - beta1_) * grad;
            v.data[k] = beta2_ * v.data[k] + (1.0 - beta2_) * grad * grad;

            const double m_hat = m.data[k] / bc1;
            const double v_hat = v.data[k] / bc2;
            const double update = effective_lr * m_hat / (std::sqrt(v_hat) + eps_);
            if (std::isfinite(update)) {
                p.data[k] -= update;
            }
        }
    }
}

void Adam::zero_grad(const std::vector<Tensor*>& grads) {
    for (auto* g : grads) {
        if (g) g->fill_zeros();
    }
}

// ============================================================
// RMSProp
// ============================================================
RMSProp::RMSProp(double lr, double alpha, double eps, double weight_decay)
    : alpha_(alpha), eps_(eps), weight_decay_(weight_decay) { lr_ = lr; }

void RMSProp::step(Sequential& model) {
    auto params = model.parameters();
    auto grads = model.gradients();

    if (!initialized_) {
        sq_avg.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i)
            sq_avg[i] = Tensor::zeros(params[i]->rows, params[i]->cols);
        initialized_ = true;
    }

    for (size_t i = 0; i < params.size(); ++i) {
        Tensor& p = *params[i];
        Tensor g = *grads[i];

        if (weight_decay_ > 0)
            g += p * weight_decay_;

        sq_avg[i] = sq_avg[i] * alpha_ + g.square() * (1.0 - alpha_);
        p -= (g / (sq_avg[i].sqrt_t() + eps_)) * lr_;
    }
}
