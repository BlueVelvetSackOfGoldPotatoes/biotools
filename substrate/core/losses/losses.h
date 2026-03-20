#pragma once
#include "../tensor/tensor.h"
#include <cmath>

class Loss {
public:
    virtual ~Loss() = default;
    virtual double forward(const Tensor& predictions, const Tensor& targets) = 0;
    virtual Tensor backward(const Tensor& predictions, const Tensor& targets) = 0;
};

// Cross-entropy loss (expects softmax output + one-hot targets).
// IMPORTANT: backward() returns the fused softmax+CE gradient (pred - targets)/N.
// Predictions MUST be softmax outputs for correct gradients.
class CrossEntropyLoss : public Loss {
    double eps_ = 1e-12;
public:
    double forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor backward(const Tensor& predictions, const Tensor& targets) override;
};

// MSE loss
class MSELoss : public Loss {
public:
    double forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor backward(const Tensor& predictions, const Tensor& targets) override;
};

// Binary cross-entropy (expects sigmoid output)
class BCELoss : public Loss {
    double eps_ = 1e-12;
public:
    double forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor backward(const Tensor& predictions, const Tensor& targets) override;
};

// Negative log likelihood (expects log-softmax output + class indices in targets)
class NLLLoss : public Loss {
public:
    double forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor backward(const Tensor& predictions, const Tensor& targets) override;
};

// KL divergence
class KLDivLoss : public Loss {
    double eps_ = 1e-12;
public:
    double forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor backward(const Tensor& predictions, const Tensor& targets) override;
};

// Huber loss (smooth L1)
class HuberLoss : public Loss {
    double delta_;
public:
    HuberLoss(double delta = 1.0) : delta_(delta) {}
    double forward(const Tensor& predictions, const Tensor& targets) override;
    Tensor backward(const Tensor& predictions, const Tensor& targets) override;
};

// Softmax + CrossEntropy combined (takes raw logits + one-hot)
namespace LossFunctions {
    Tensor softmax(const Tensor& logits);
    Tensor softmax_backward(const Tensor& softmax_output, const Tensor& grad_output);
    double cross_entropy_with_softmax(const Tensor& logits, const Tensor& targets);
    Tensor cross_entropy_with_softmax_backward(const Tensor& softmax_output, const Tensor& targets);
}
