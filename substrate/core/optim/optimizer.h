#pragma once
#include "../nn/module.h"
#include <cmath>

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(Sequential& model) = 0;
    virtual void zero_grad(Sequential& model);
    double lr_;
    void set_lr(double lr) { lr_ = lr; }
    double get_lr() const { return lr_; }
};

// SGD with optional momentum and weight decay
class SGDOptimizer : public Optimizer {
    double momentum_;
    double weight_decay_;
    std::vector<Tensor> velocity_params;
    bool initialized_ = false;
public:
    SGDOptimizer(double lr, double momentum = 0.0, double weight_decay = 0.0);
    void step(Sequential& model) override;
};

// Adam optimizer
class Adam : public Optimizer {
    double beta1_, beta2_, eps_, weight_decay_;
    std::vector<Tensor> m_params, v_params;
    std::vector<const Tensor*> param_ptrs_; // for validation across calls
    int t_ = 0;
    bool initialized_ = false;
public:
    Adam(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999,
         double eps = 1e-8, double weight_decay = 0.0);
    void step(Sequential& model) override;

    // Vector-based interface for TrainableModel and other non-Sequential models.
    // Supports per-step lr_scale and global gradient norm clipping.
    void step(const std::vector<Tensor*>& params,
              const std::vector<Tensor*>& grads,
              double lr_scale = 1.0,
              double clip_global_grad_norm = 0.0);

    double lr() const { return lr_; }
    using Optimizer::zero_grad;
    static void zero_grad(const std::vector<Tensor*>& grads);
};

// RMSProp
class RMSProp : public Optimizer {
    double alpha_, eps_, weight_decay_;
    std::vector<Tensor> sq_avg;
    bool initialized_ = false;
public:
    RMSProp(double lr = 0.01, double alpha = 0.99, double eps = 1e-8, double weight_decay = 0.0);
    void step(Sequential& model) override;
};

// Learning rate schedulers
class LRScheduler {
public:
    virtual ~LRScheduler() = default;
    virtual double get_lr(int epoch) = 0;
};

class StepLR : public LRScheduler {
    double base_lr_, gamma_;
    int step_size_;
public:
    StepLR(double base_lr, int step_size, double gamma = 0.1)
        : base_lr_(base_lr), gamma_(gamma), step_size_(step_size) {}
    double get_lr(int epoch) override {
        return base_lr_ * std::pow(gamma_, epoch / step_size_);
    }
};

class CosineAnnealingLR : public LRScheduler {
    double base_lr_, min_lr_;
    int T_max_;
public:
    CosineAnnealingLR(double base_lr, int T_max, double min_lr = 0.0)
        : base_lr_(base_lr), min_lr_(min_lr), T_max_(T_max) {}
    double get_lr(int epoch) override {
        return min_lr_ + 0.5 * (base_lr_ - min_lr_) * (1 + std::cos(M_PI * epoch / T_max_));
    }
};

class WarmupCosineAnnealing : public LRScheduler {
    double base_lr_, min_lr_;
    int warmup_epochs_, total_epochs_;
public:
    WarmupCosineAnnealing(double base_lr, int warmup_epochs, int total_epochs, double min_lr = 0.0)
        : base_lr_(base_lr), min_lr_(min_lr), warmup_epochs_(warmup_epochs), total_epochs_(total_epochs) {}
    double get_lr(int epoch) override {
        if (epoch < warmup_epochs_)
            return base_lr_ * (epoch + 1) / warmup_epochs_;
        int cos_epoch = epoch - warmup_epochs_;
        int cos_total = total_epochs_ - warmup_epochs_;
        return min_lr_ + 0.5 * (base_lr_ - min_lr_) * (1 + std::cos(M_PI * cos_epoch / cos_total));
    }
};
