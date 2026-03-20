// modulus_activations.h - Extended activation functions
// Port of physicsnemo/nn/module/activations.py
// All activation types from Modulus ACT2FN dictionary
//
// C++17, no external dependencies (uses modulus.h Tensor).

#ifndef MODULUS_ACTIVATIONS_H
#define MODULUS_ACTIVATIONS_H

#include "modulus.h"
#include <cmath>
#include <stdexcept>
#include <string>
#include <functional>

namespace modulus {
namespace act {

// ============================================================
// Element-wise activation functions and their gradients
// ============================================================

inline double sigmoid_fn(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double sigmoid_grad(double x) {
    double s = sigmoid_fn(x);
    return s * (1.0 - s);
}

inline double tanh_fn(double x) { return std::tanh(x); }
inline double tanh_grad(double x) {
    double t = std::tanh(x);
    return 1.0 - t * t;
}

inline double relu6_fn(double x) { return std::min(std::max(x, 0.0), 6.0); }
inline double relu6_grad(double x) { return (x > 0.0 && x < 6.0) ? 1.0 : 0.0; }

inline double elu_fn(double x, double alpha = 1.0) {
    return x >= 0 ? x : alpha * (std::exp(x) - 1.0);
}
inline double elu_grad(double x, double alpha = 1.0) {
    return x >= 0 ? 1.0 : alpha * std::exp(x);
}

inline double celu_fn(double x, double alpha = 1.0) {
    return std::max(x, 0.0) + std::min(0.0, alpha * (std::exp(x / alpha) - 1.0));
}
inline double celu_grad(double x, double /*alpha*/ = 1.0) {
    if (x >= 0) return 1.0;
    return std::exp(x);
}

inline double selu_fn(double x) {
    constexpr double alpha = 1.6732632423543772;
    constexpr double scale = 1.0507009873554805;
    return x > 0 ? scale * x : scale * alpha * (std::exp(x) - 1.0);
}
inline double selu_grad(double x) {
    constexpr double alpha = 1.6732632423543772;
    constexpr double scale = 1.0507009873554805;
    return x > 0 ? scale : scale * alpha * std::exp(x);
}

inline double leaky_relu_fn(double x, double neg_slope = 0.1) {
    return x >= 0 ? x : neg_slope * x;
}
inline double leaky_relu_grad(double x, double neg_slope = 0.1) {
    return x >= 0 ? 1.0 : neg_slope;
}

inline double prelu_fn(double x, double alpha = 0.25) {
    return x >= 0 ? x : alpha * x;
}
inline double prelu_grad(double x, double alpha = 0.25) {
    return x >= 0 ? 1.0 : alpha;
}

inline double log_sigmoid_fn(double x) { return -std::log(1.0 + std::exp(-x)); }
inline double log_sigmoid_grad(double x) { return 1.0 / (1.0 + std::exp(x)); }

inline double softplus_fn(double x, double beta = 1.0) {
    if (beta * x > 20.0) return x;
    return std::log(1.0 + std::exp(beta * x)) / beta;
}
inline double softplus_grad(double x, double beta = 1.0) {
    return sigmoid_fn(beta * x);
}

inline double softshrink_fn(double x, double lam = 0.5) {
    if (x > lam) return x - lam;
    if (x < -lam) return x + lam;
    return 0.0;
}
inline double softshrink_grad(double x, double lam = 0.5) {
    return (x > lam || x < -lam) ? 1.0 : 0.0;
}

inline double softsign_fn(double x) { return x / (1.0 + std::abs(x)); }
inline double softsign_grad(double x) {
    double d = 1.0 + std::abs(x);
    return 1.0 / (d * d);
}

inline double tanhshrink_fn(double x) { return x - std::tanh(x); }
inline double tanhshrink_grad(double x) {
    double t = std::tanh(x);
    return t * t;
}

inline double hardtanh_fn(double x, double lo = -1.0, double hi = 1.0) {
    return std::min(std::max(x, lo), hi);
}
inline double hardtanh_grad(double x, double lo = -1.0, double hi = 1.0) {
    return (x >= lo && x <= hi) ? 1.0 : 0.0;
}

inline double mish_fn(double x) {
    return x * std::tanh(softplus_fn(x));
}
inline double mish_grad(double x) {
    double sp = softplus_fn(x);
    double tsp = std::tanh(sp);
    double sig = sigmoid_fn(x);
    return tsp + x * sig * (1.0 - tsp * tsp);
}

inline double squareplus_fn(double x) {
    return 0.5 * (x + std::sqrt(x * x + 4.0));
}
inline double squareplus_grad(double x) {
    return 0.5 * (1.0 + x / std::sqrt(x * x + 4.0));
}

inline double capped_leaky_relu_fn(double x, double cap = 1.0, double neg_slope = 0.01) {
    double v = x >= 0 ? x : neg_slope * x;
    return std::min(v, cap);
}
inline double capped_leaky_relu_grad(double x, double cap = 1.0, double neg_slope = 0.01) {
    double v = x >= 0 ? x : neg_slope * x;
    if (v >= cap) return 0.0;
    return x >= 0 ? 1.0 : neg_slope;
}

inline double capped_gelu_fn(double x, double cap = 1.0) {
    double g = activation::gelu(x);
    return std::min(g, cap);
}
inline double capped_gelu_grad(double x, double cap = 1.0) {
    double g = activation::gelu(x);
    if (g >= cap) return 0.0;
    return activation::gelu_grad(x);
}

inline double sin_fn(double x) { return std::sin(x); }
inline double sin_grad(double x) { return std::cos(x); }

// Stan: Self-scalable Tanh: tanh(x) * (1 + beta * x)
inline double stan_fn(double x, double beta = 1.0) {
    return std::tanh(x) * (1.0 + beta * x);
}
inline double stan_grad(double x, double beta = 1.0) {
    double t = std::tanh(x);
    double sech2 = 1.0 - t * t;
    return sech2 * (1.0 + beta * x) + t * beta;
}

// ============================================================
// Comprehensive activation dispatcher
// Returns (forward_fn, gradient_fn) pair
// ============================================================
using ActFn = double (*)(double);

inline std::pair<ActFn, ActFn> get_activation_extended(const std::string& name) {
    if (name == "gelu") return {activation::gelu, activation::gelu_grad};
    if (name == "relu") return {activation::relu, activation::relu_grad};
    if (name == "silu" || name == "swish") return {activation::silu, activation::silu_grad};
    if (name == "none" || name == "identity" || name == "linear")
        return {activation::identity, activation::identity_grad};
    if (name == "sigmoid") return {sigmoid_fn, sigmoid_grad};
    if (name == "tanh") return {tanh_fn, tanh_grad};
    if (name == "relu6") return {relu6_fn, relu6_grad};
    if (name == "selu") return {selu_fn, selu_grad};
    if (name == "log_sigmoid" || name == "logsigmoid") return {log_sigmoid_fn, log_sigmoid_grad};
    if (name == "softsign") return {softsign_fn, softsign_grad};
    if (name == "mish") return {mish_fn, mish_grad};
    if (name == "squareplus") return {squareplus_fn, squareplus_grad};
    if (name == "sin" || name == "sine") return {sin_fn, sin_grad};

    // Activations with default parameters (use lambdas via static)
    if (name == "leaky_relu") {
        static auto fwd = [](double x) -> double { return leaky_relu_fn(x, 0.1); };
        static auto grd = [](double x) -> double { return leaky_relu_grad(x, 0.1); };
        return {fwd, grd};
    }
    if (name == "prelu") {
        static auto fwd = [](double x) -> double { return prelu_fn(x, 0.25); };
        static auto grd = [](double x) -> double { return prelu_grad(x, 0.25); };
        return {fwd, grd};
    }
    if (name == "elu") {
        static auto fwd = [](double x) -> double { return elu_fn(x, 1.0); };
        static auto grd = [](double x) -> double { return elu_grad(x, 1.0); };
        return {fwd, grd};
    }
    if (name == "celu") {
        static auto fwd = [](double x) -> double { return celu_fn(x, 1.0); };
        static auto grd = [](double x) -> double { return celu_grad(x, 1.0); };
        return {fwd, grd};
    }
    if (name == "softplus") {
        static auto fwd = [](double x) -> double { return softplus_fn(x); };
        static auto grd = [](double x) -> double { return softplus_grad(x); };
        return {fwd, grd};
    }
    if (name == "softshrink") {
        static auto fwd = [](double x) -> double { return softshrink_fn(x); };
        static auto grd = [](double x) -> double { return softshrink_grad(x); };
        return {fwd, grd};
    }
    if (name == "tanhshrink") return {tanhshrink_fn, tanhshrink_grad};
    if (name == "hardtanh") {
        static auto fwd = [](double x) -> double { return hardtanh_fn(x); };
        static auto grd = [](double x) -> double { return hardtanh_grad(x); };
        return {fwd, grd};
    }
    if (name == "stan") {
        static auto fwd = [](double x) -> double { return stan_fn(x); };
        static auto grd = [](double x) -> double { return stan_grad(x); };
        return {fwd, grd};
    }
    if (name == "capped_leaky_relu") {
        static auto fwd = [](double x) -> double { return capped_leaky_relu_fn(x); };
        static auto grd = [](double x) -> double { return capped_leaky_relu_grad(x); };
        return {fwd, grd};
    }
    if (name == "capped_gelu") {
        static auto fwd = [](double x) -> double { return capped_gelu_fn(x); };
        static auto grd = [](double x) -> double { return capped_gelu_grad(x); };
        return {fwd, grd};
    }

    throw std::runtime_error("Unknown activation: " + name);
}

// Apply activation to tensor
inline Tensor apply_activation(const Tensor& x, const std::string& act_name) {
    auto [fn, gfn] = get_activation_extended(act_name);
    (void)gfn;
    return x.apply(fn);
}

} // namespace act
} // namespace modulus

#endif // MODULUS_ACTIVATIONS_H
