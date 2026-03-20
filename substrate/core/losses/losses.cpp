#include "losses.h"
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#define LOSSES_OMP_PAR_FOR _Pragma("omp parallel for schedule(static)")
#define LOSSES_OMP_PAR_FOR_COLLAPSE2 _Pragma("omp parallel for collapse(2) schedule(static)")
#define LOSSES_OMP_PAR_FOR_REDUCE_SUM _Pragma("omp parallel for reduction(+ : loss) schedule(static)")
#define LOSSES_OMP_PAR_FOR_COLLAPSE2_REDUCE_SUM _Pragma("omp parallel for collapse(2) reduction(+ : loss) schedule(static)")
#define LOSSES_OMP_PAR_FOR_REDUCE_SUMEXP _Pragma("omp parallel for reduction(+ : sum_exp) schedule(static)")
#else
#define LOSSES_OMP_PAR_FOR
#define LOSSES_OMP_PAR_FOR_COLLAPSE2
#define LOSSES_OMP_PAR_FOR_REDUCE_SUM
#define LOSSES_OMP_PAR_FOR_COLLAPSE2_REDUCE_SUM
#define LOSSES_OMP_PAR_FOR_REDUCE_SUMEXP
#endif

// ============================================================
// Cross-Entropy Loss
// ============================================================
double CrossEntropyLoss::forward(const Tensor& pred, const Tensor& targets) {
    TENSOR_CHECK(pred.rows == targets.rows && pred.cols == targets.cols,
                 "CrossEntropyLoss: shape mismatch");
    double loss = 0;
    LOSSES_OMP_PAR_FOR_COLLAPSE2_REDUCE_SUM
    for (size_t i = 0; i < pred.rows; ++i)
        for (size_t j = 0; j < pred.cols; ++j)
            loss -= targets(i, j) * std::log(pred(i, j) + eps_);
    return loss / pred.rows;
}

// Returns fused softmax+CE gradient. Predictions MUST be softmax outputs.
Tensor CrossEntropyLoss::backward(const Tensor& pred, const Tensor& targets) {
    TENSOR_CHECK(pred.rows == targets.rows && pred.cols == targets.cols,
                 "CrossEntropyLoss: shape mismatch");
    return (pred - targets) / (double)pred.rows;
}

// ============================================================
// MSE Loss
// ============================================================
double MSELoss::forward(const Tensor& pred, const Tensor& targets) {
    TENSOR_CHECK(pred.rows == targets.rows && pred.cols == targets.cols,
                 "MSELoss: shape mismatch");
    TENSOR_CHECK(!pred.data.empty(), "MSELoss: empty tensor");
    double loss = 0;
    LOSSES_OMP_PAR_FOR_REDUCE_SUM
    for (size_t i = 0; i < pred.data.size(); ++i) {
        double d = pred.data[i] - targets.data[i];
        loss += d * d;
    }
    return loss / pred.data.size();
}

Tensor MSELoss::backward(const Tensor& pred, const Tensor& targets) {
    TENSOR_CHECK(pred.rows == targets.rows && pred.cols == targets.cols,
                 "MSELoss: shape mismatch");
    TENSOR_CHECK(!pred.data.empty(), "MSELoss: empty tensor");
    return (pred - targets) * (2.0 / pred.data.size());
}

// ============================================================
// BCE Loss
// ============================================================
double BCELoss::forward(const Tensor& pred, const Tensor& targets) {
    TENSOR_CHECK(pred.rows == targets.rows && pred.cols == targets.cols,
                 "BCELoss: shape mismatch");
    TENSOR_CHECK(!pred.data.empty(), "BCELoss: empty tensor");
    double loss = 0;
    LOSSES_OMP_PAR_FOR_REDUCE_SUM
    for (size_t i = 0; i < pred.data.size(); ++i) {
        double p = std::clamp(pred.data[i], eps_, 1.0 - eps_);
        loss -= targets.data[i] * std::log(p) + (1.0 - targets.data[i]) * std::log(1.0 - p);
    }
    return loss / pred.data.size();
}

Tensor BCELoss::backward(const Tensor& pred, const Tensor& targets) {
    TENSOR_CHECK(pred.rows == targets.rows && pred.cols == targets.cols,
                 "BCELoss: shape mismatch");
    TENSOR_CHECK(!pred.data.empty(), "BCELoss: empty tensor");
    Tensor grad(pred.rows, pred.cols);
    LOSSES_OMP_PAR_FOR
    for (size_t i = 0; i < pred.data.size(); ++i) {
        double p = std::clamp(pred.data[i], eps_, 1.0 - eps_);
        grad.data[i] = (-targets.data[i] / p + (1.0 - targets.data[i]) / (1.0 - p)) / pred.data.size();
    }
    return grad;
}

// ============================================================
// NLL Loss (expects log-probs and class indices)
// ============================================================
double NLLLoss::forward(const Tensor& pred, const Tensor& targets) {
    TENSOR_CHECK(pred.rows == targets.rows && targets.cols >= 1,
                 "NLLLoss: shape mismatch");
    TENSOR_CHECK(pred.cols > 0, "NLLLoss: zero classes");
    TENSOR_CHECK(pred.rows > 0, "NLLLoss: empty batch");
    double loss = 0;
    LOSSES_OMP_PAR_FOR_REDUCE_SUM
    for (size_t i = 0; i < pred.rows; ++i) {
        const double raw = targets(i, 0);
        TENSOR_CHECK(std::isfinite(raw), "NLLLoss: target class must be finite");
        const long long cls_ll = static_cast<long long>(std::llround(raw));
        TENSOR_CHECK(std::abs(raw - static_cast<double>(cls_ll)) < 1e-9,
                     "NLLLoss: target class index must be integer-valued");
        TENSOR_CHECK(cls_ll >= 0 && cls_ll < static_cast<long long>(pred.cols),
                     "NLLLoss: target class out of range");
        size_t cls = static_cast<size_t>(cls_ll);
        loss -= pred(i, cls);
    }
    return loss / pred.rows;
}

Tensor NLLLoss::backward(const Tensor& pred, const Tensor& targets) {
    TENSOR_CHECK(pred.rows == targets.rows && targets.cols >= 1,
                 "NLLLoss: shape mismatch");
    TENSOR_CHECK(pred.cols > 0, "NLLLoss: zero classes");
    TENSOR_CHECK(pred.rows > 0, "NLLLoss: empty batch");
    Tensor grad(pred.rows, pred.cols);
    LOSSES_OMP_PAR_FOR
    for (size_t i = 0; i < pred.rows; ++i) {
        const double raw = targets(i, 0);
        TENSOR_CHECK(std::isfinite(raw), "NLLLoss: target class must be finite");
        const long long cls_ll = static_cast<long long>(std::llround(raw));
        TENSOR_CHECK(std::abs(raw - static_cast<double>(cls_ll)) < 1e-9,
                     "NLLLoss: target class index must be integer-valued");
        TENSOR_CHECK(cls_ll >= 0 && cls_ll < static_cast<long long>(pred.cols),
                     "NLLLoss: target class out of range");
        size_t cls = static_cast<size_t>(cls_ll);
        grad(i, cls) = -1.0 / pred.rows;
    }
    return grad;
}

// ============================================================
// KL Divergence
// ============================================================
double KLDivLoss::forward(const Tensor& pred, const Tensor& targets) {
    TENSOR_CHECK(pred.rows == targets.rows && pred.cols == targets.cols,
                 "KLDivLoss: shape mismatch");
    TENSOR_CHECK(pred.rows > 0, "KLDivLoss: empty batch");
    double loss = 0;
    LOSSES_OMP_PAR_FOR_REDUCE_SUM
    for (size_t i = 0; i < pred.data.size(); ++i) {
        if (targets.data[i] > eps_)
            loss += targets.data[i] * (std::log(targets.data[i] + eps_) - std::log(pred.data[i] + eps_));
    }
    return loss / pred.rows;
}

Tensor KLDivLoss::backward(const Tensor& pred, const Tensor& targets) {
    TENSOR_CHECK(pred.rows == targets.rows && pred.cols == targets.cols,
                 "KLDivLoss: shape mismatch");
    TENSOR_CHECK(pred.rows > 0, "KLDivLoss: empty batch");
    Tensor grad(pred.rows, pred.cols);
    LOSSES_OMP_PAR_FOR
    for (size_t i = 0; i < pred.data.size(); ++i)
        grad.data[i] = -targets.data[i] / (pred.data[i] + eps_) / pred.rows;
    return grad;
}

// ============================================================
// Huber Loss
// ============================================================
double HuberLoss::forward(const Tensor& pred, const Tensor& targets) {
    TENSOR_CHECK(pred.rows == targets.rows && pred.cols == targets.cols,
                 "HuberLoss: shape mismatch");
    TENSOR_CHECK(!pred.data.empty(), "HuberLoss: empty tensor");
    double loss = 0;
    LOSSES_OMP_PAR_FOR_REDUCE_SUM
    for (size_t i = 0; i < pred.data.size(); ++i) {
        double d = std::abs(pred.data[i] - targets.data[i]);
        if (d <= delta_) loss += 0.5 * d * d;
        else loss += delta_ * (d - 0.5 * delta_);
    }
    return loss / pred.data.size();
}

Tensor HuberLoss::backward(const Tensor& pred, const Tensor& targets) {
    TENSOR_CHECK(pred.rows == targets.rows && pred.cols == targets.cols,
                 "HuberLoss: shape mismatch");
    TENSOR_CHECK(!pred.data.empty(), "HuberLoss: empty tensor");
    Tensor grad(pred.rows, pred.cols);
    double n = pred.data.size();
    LOSSES_OMP_PAR_FOR
    for (size_t i = 0; i < pred.data.size(); ++i) {
        double d = pred.data[i] - targets.data[i];
        if (std::abs(d) <= delta_) grad.data[i] = d / n;
        else grad.data[i] = delta_ * (d > 0 ? 1.0 : -1.0) / n;
    }
    return grad;
}

// ============================================================
// Softmax + CrossEntropy combined
// ============================================================
namespace LossFunctions {

Tensor softmax(const Tensor& logits) {
    Tensor result(logits.rows, logits.cols);
    LOSSES_OMP_PAR_FOR
    for (size_t i = 0; i < logits.rows; ++i) {
        double max_val = logits(i, 0);
        for (size_t j = 1; j < logits.cols; ++j)
            max_val = std::max(max_val, logits(i, j));
        double sum_exp = 0;
        for (size_t j = 0; j < logits.cols; ++j) {
            result(i, j) = std::exp(logits(i, j) - max_val);
            sum_exp += result(i, j);
        }
        for (size_t j = 0; j < logits.cols; ++j)
            result(i, j) /= sum_exp;
    }
    return result;
}

Tensor softmax_backward(const Tensor& softmax_output, const Tensor& grad_output) {
    TENSOR_CHECK(
        softmax_output.rows == grad_output.rows && softmax_output.cols == grad_output.cols,
        "softmax_backward: shape mismatch"
    );
    Tensor grad_logits(softmax_output.rows, softmax_output.cols);
    LOSSES_OMP_PAR_FOR
    for (size_t i = 0; i < softmax_output.rows; ++i) {
        double dot = 0.0;
        for (size_t j = 0; j < softmax_output.cols; ++j) {
            dot += grad_output(i, j) * softmax_output(i, j);
        }
        for (size_t j = 0; j < softmax_output.cols; ++j) {
            grad_logits(i, j) = softmax_output(i, j) * (grad_output(i, j) - dot);
        }
    }
    return grad_logits;
}

double cross_entropy_with_softmax(const Tensor& logits, const Tensor& targets) {
    Tensor probs = softmax(logits);
    double loss = 0;
    LOSSES_OMP_PAR_FOR_COLLAPSE2_REDUCE_SUM
    for (size_t i = 0; i < probs.rows; ++i)
        for (size_t j = 0; j < probs.cols; ++j)
            loss -= targets(i, j) * std::log(probs(i, j) + 1e-12);
    return loss / probs.rows;
}

Tensor cross_entropy_with_softmax_backward(const Tensor& softmax_output, const Tensor& targets) {
    return (softmax_output - targets) / (double)softmax_output.rows;
}

} // namespace LossFunctions
