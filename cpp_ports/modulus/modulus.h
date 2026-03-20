// modulus.h - Header-only C++ port of NVIDIA Modulus/PhysicsNeMo
// Physics-informed machine learning framework
//
// Ported architectures:
//   - Dense Tensor with FFT (Cooley-Tukey)
//   - MLP (multi-layer perceptron) with backpropagation
//   - FNO (Fourier Neural Operator) with spectral convolution
//   - DeepONet (Deep Operator Network)
//   - Physics constraint (PDE residual loss)
//   - Adam optimizer
//   - Training loop
//
// C++17, no external dependencies.

#ifndef MODULUS_H
#define MODULUS_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace modulus {

// ============================================================
// Global random engine
// ============================================================
inline std::mt19937& global_rng() {
    static std::mt19937 rng(42);
    return rng;
}

inline void seed_rng(unsigned s) { global_rng().seed(s); }

// ============================================================
// Complex number alias
// ============================================================
using Complex = std::complex<double>;

// ============================================================
// Tensor - Dense row-major multi-dimensional array
// ============================================================
class Tensor {
public:
    std::vector<int> shape;
    std::vector<double> data;

    Tensor() = default;

    explicit Tensor(const std::vector<int>& shape, double val = 0.0)
        : shape(shape), data(numel(shape), val) {}

    Tensor(const std::vector<int>& shape, const std::vector<double>& data)
        : shape(shape), data(data) {
        assert((int)data.size() == numel(shape));
    }

    static int numel(const std::vector<int>& s) {
        if (s.empty()) return 0;
        int n = 1;
        for (int d : s) n *= d;
        return n;
    }

    int numel() const { return numel(shape); }
    int ndim() const { return (int)shape.size(); }
    bool empty() const { return data.empty(); }

    double& operator[](int i) { return data[i]; }
    double operator[](int i) const { return data[i]; }

    // Flat index from multi-dim indices
    int flat_index(const std::vector<int>& idx) const {
        int fi = 0;
        int stride = 1;
        for (int d = ndim() - 1; d >= 0; --d) {
            fi += idx[d] * stride;
            stride *= shape[d];
        }
        return fi;
    }

    // 2D access (row, col)
    double& at2(int r, int c) { return data[r * shape[1] + c]; }
    double at2(int r, int c) const { return data[r * shape[1] + c]; }

    // 3D access (i, j, k)
    double& at3(int i, int j, int k) {
        return data[(i * shape[1] + j) * shape[2] + k];
    }
    double at3(int i, int j, int k) const {
        return data[(i * shape[1] + j) * shape[2] + k];
    }

    // Reshape (must preserve total elements)
    Tensor reshape(const std::vector<int>& new_shape) const {
        assert(numel(new_shape) == numel());
        Tensor out;
        out.shape = new_shape;
        out.data = data;
        return out;
    }

    // Zeros, ones, random
    static Tensor zeros(const std::vector<int>& s) { return Tensor(s, 0.0); }
    static Tensor ones(const std::vector<int>& s) { return Tensor(s, 1.0); }

    static Tensor rand(const std::vector<int>& s) {
        Tensor t(s);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (auto& v : t.data) v = dist(global_rng());
        return t;
    }

    static Tensor randn(const std::vector<int>& s) {
        Tensor t(s);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (auto& v : t.data) v = dist(global_rng());
        return t;
    }

    // Xavier uniform initialization
    static Tensor xavier_uniform(int rows, int cols) {
        double limit = std::sqrt(6.0 / (rows + cols));
        Tensor t({rows, cols});
        std::uniform_real_distribution<double> dist(-limit, limit);
        for (auto& v : t.data) v = dist(global_rng());
        return t;
    }

    // Element-wise operations
    Tensor operator+(const Tensor& o) const {
        assert(numel() == o.numel());
        Tensor r(shape);
        for (int i = 0; i < numel(); ++i) r.data[i] = data[i] + o.data[i];
        return r;
    }

    Tensor operator-(const Tensor& o) const {
        assert(numel() == o.numel());
        Tensor r(shape);
        for (int i = 0; i < numel(); ++i) r.data[i] = data[i] - o.data[i];
        return r;
    }

    Tensor operator*(const Tensor& o) const {
        assert(numel() == o.numel());
        Tensor r(shape);
        for (int i = 0; i < numel(); ++i) r.data[i] = data[i] * o.data[i];
        return r;
    }

    Tensor operator*(double s) const {
        Tensor r(shape);
        for (int i = 0; i < numel(); ++i) r.data[i] = data[i] * s;
        return r;
    }

    Tensor& operator+=(const Tensor& o) {
        for (int i = 0; i < numel(); ++i) data[i] += o.data[i];
        return *this;
    }

    Tensor& operator-=(const Tensor& o) {
        for (int i = 0; i < numel(); ++i) data[i] -= o.data[i];
        return *this;
    }

    // Sum all elements
    double sum() const {
        double s = 0;
        for (auto v : data) s += v;
        return s;
    }

    // Mean
    double mean() const { return sum() / numel(); }

    // Squared L2 norm
    double norm_sq() const {
        double s = 0;
        for (auto v : data) s += v * v;
        return s;
    }

    // Apply function element-wise
    Tensor apply(std::function<double(double)> fn) const {
        Tensor r(shape);
        for (int i = 0; i < numel(); ++i) r.data[i] = fn(data[i]);
        return r;
    }

    // Matrix multiply: (M x K) @ (K x N) -> (M x N)
    static Tensor matmul(const Tensor& A, const Tensor& B) {
        assert(A.ndim() == 2 && B.ndim() == 2);
        assert(A.shape[1] == B.shape[0]);
        int M = A.shape[0], K = A.shape[1], N = B.shape[1];
        Tensor C({M, N}, 0.0);
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k) {
                double a = A.at2(i, k);
                for (int j = 0; j < N; ++j)
                    C.at2(i, j) += a * B.at2(k, j);
            }
        return C;
    }

    // Transpose 2D
    Tensor T() const {
        assert(ndim() == 2);
        Tensor r({shape[1], shape[0]});
        for (int i = 0; i < shape[0]; ++i)
            for (int j = 0; j < shape[1]; ++j)
                r.at2(j, i) = at2(i, j);
        return r;
    }

    // Add bias (broadcast row vector across rows)
    Tensor add_bias(const Tensor& bias) const {
        assert(ndim() == 2 && bias.numel() == shape[1]);
        Tensor r = *this;
        for (int i = 0; i < shape[0]; ++i)
            for (int j = 0; j < shape[1]; ++j)
                r.at2(i, j) += bias.data[j];
        return r;
    }

    // Sum along columns: (M x N) -> (1 x N), summing each column
    Tensor sum_rows() const {
        assert(ndim() == 2);
        Tensor r({1, shape[1]}, 0.0);
        for (int i = 0; i < shape[0]; ++i)
            for (int j = 0; j < shape[1]; ++j)
                r.data[j] += at2(i, j);
        return r;
    }

    // Concatenate along last axis
    static Tensor cat_last(const Tensor& a, const Tensor& b) {
        assert(a.ndim() == 2 && b.ndim() == 2 && a.shape[0] == b.shape[0]);
        int rows = a.shape[0];
        int ca = a.shape[1], cb = b.shape[1];
        Tensor r({rows, ca + cb});
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < ca; ++j) r.at2(i, j) = a.at2(i, j);
            for (int j = 0; j < cb; ++j) r.at2(i, ca + j) = b.at2(i, j);
        }
        return r;
    }

    // Print (debug)
    void print(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << " ";
        std::cout << "shape=[";
        for (int i = 0; i < ndim(); ++i) {
            if (i) std::cout << ",";
            std::cout << shape[i];
        }
        std::cout << "] data=[";
        int n = std::min(numel(), 8);
        for (int i = 0; i < n; ++i) {
            if (i) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << data[i];
        }
        if (numel() > 8) std::cout << ", ...";
        std::cout << "]" << std::endl;
    }
};

// ============================================================
// FFT - Cooley-Tukey radix-2 DIT implementation
// ============================================================
namespace fft {

// Bit-reversal permutation
inline void bit_reverse(std::vector<Complex>& a) {
    int n = (int)a.size();
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
}

// In-place FFT (iterative Cooley-Tukey)
// inverse=false: DFT, inverse=true: IDFT
inline void fft_inplace(std::vector<Complex>& a, bool inverse) {
    int n = (int)a.size();
    if (n == 1) return;

    bit_reverse(a);

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / len * (inverse ? -1 : 1);
        Complex wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            Complex w(1, 0);
            for (int j = 0; j < len / 2; ++j) {
                Complex u = a[i + j];
                Complex v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        for (auto& x : a) x /= (double)n;
    }
}

// Pad/truncate to next power of 2
inline int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// 1D FFT: real input -> complex output (positive frequencies)
inline std::vector<Complex> rfft(const std::vector<double>& x) {
    int n = next_pow2((int)x.size());
    std::vector<Complex> a(n);
    for (int i = 0; i < (int)x.size(); ++i) a[i] = Complex(x[i], 0);
    fft_inplace(a, false);
    // Return first n/2 + 1 elements (positive frequencies)
    int out_len = n / 2 + 1;
    return std::vector<Complex>(a.begin(), a.begin() + out_len);
}

// 1D IFFT: complex (rfft output) -> real output
inline std::vector<double> irfft(const std::vector<Complex>& X, int output_len) {
    int n = next_pow2(output_len);
    // Reconstruct full spectrum from rfft output (Hermitian symmetry)
    std::vector<Complex> a(n);
    int rfft_len = n / 2 + 1;
    for (int i = 0; i < std::min((int)X.size(), rfft_len); ++i) a[i] = X[i];
    // Mirror for negative frequencies
    for (int i = 1; i < n / 2; ++i) a[n - i] = std::conj(a[i]);
    fft_inplace(a, true);
    std::vector<double> out(output_len);
    for (int i = 0; i < output_len; ++i) out[i] = a[i].real();
    return out;
}

} // namespace fft

// ============================================================
// Parameter: a named tensor that participates in optimization
// ============================================================
struct Parameter {
    std::string name;
    Tensor value;
    Tensor grad;

    Parameter() = default;
    Parameter(const std::string& n, const Tensor& v)
        : name(n), value(v), grad(Tensor::zeros(v.shape)) {}
};

// ============================================================
// Activation functions
// ============================================================
namespace activation {

inline double gelu(double x) {
    return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
}

inline double gelu_grad(double x) {
    double c = std::sqrt(2.0 / M_PI);
    double inner = c * (x + 0.044715 * x * x * x);
    double tanh_val = std::tanh(inner);
    double sech2 = 1.0 - tanh_val * tanh_val;
    double inner_grad = c * (1.0 + 3.0 * 0.044715 * x * x);
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * inner_grad;
}

inline double relu(double x) { return x > 0 ? x : 0; }
inline double relu_grad(double x) { return x > 0 ? 1.0 : 0.0; }

inline double silu(double x) {
    double sig = 1.0 / (1.0 + std::exp(-x));
    return x * sig;
}

inline double silu_grad(double x) {
    double sig = 1.0 / (1.0 + std::exp(-x));
    return sig * (1.0 + x * (1.0 - sig));
}

inline double identity(double x) { return x; }
inline double identity_grad(double /*x*/) { return 1.0; }

using ActivationFn = double (*)(double);

inline std::pair<ActivationFn, ActivationFn> get_activation(const std::string& name) {
    if (name == "gelu") return {gelu, gelu_grad};
    if (name == "relu") return {relu, relu_grad};
    if (name == "silu" || name == "swish") return {silu, silu_grad};
    if (name == "none" || name == "identity" || name == "linear")
        return {identity, identity_grad};
    throw std::runtime_error("Unknown activation: " + name);
}

} // namespace activation

// ============================================================
// Linear layer (fully connected)
// ============================================================
class Linear {
public:
    Parameter weight; // (out_features, in_features)
    Parameter bias;   // (1, out_features)
    int in_features, out_features;

    // Cache for backprop
    Tensor input_cache;

    Linear() : in_features(0), out_features(0) {}

    Linear(const std::string& prefix, int in_f, int out_f)
        : in_features(in_f), out_features(out_f) {
        weight = Parameter(prefix + ".weight",
                           Tensor::xavier_uniform(out_f, in_f));
        bias = Parameter(prefix + ".bias", Tensor::zeros({1, out_f}));
    }

    // Forward: x (batch x in) -> (batch x out)
    Tensor forward(const Tensor& x) {
        input_cache = x;
        // out = x @ W^T + b
        Tensor wt = weight.value.T(); // (in, out)
        Tensor out = Tensor::matmul(x, wt);
        out = out.add_bias(bias.value);
        return out;
    }

    // Backward: grad_output (batch x out) -> grad_input (batch x in)
    Tensor backward(const Tensor& grad_output) {
        // grad_weight += grad_output^T @ input  -> (out, in)
        Tensor go_t = grad_output.T();
        Tensor dw = Tensor::matmul(go_t, input_cache);
        for (int i = 0; i < dw.numel(); ++i)
            weight.grad.data[i] += dw.data[i];

        // grad_bias += sum of grad_output over batch -> (1, out)
        Tensor db = grad_output.sum_rows();
        for (int i = 0; i < db.numel(); ++i)
            bias.grad.data[i] += db.data[i];

        // grad_input = grad_output @ W -> (batch, in)
        Tensor grad_input = Tensor::matmul(grad_output, weight.value);
        return grad_input;
    }

    void zero_grad() {
        std::fill(weight.grad.data.begin(), weight.grad.data.end(), 0.0);
        std::fill(bias.grad.data.begin(), bias.grad.data.end(), 0.0);
    }

    std::vector<Parameter*> parameters() { return {&weight, &bias}; }
};

// ============================================================
// MLP (Multi-Layer Perceptron) with backpropagation
// Mirrors Modulus FullyConnected
// ============================================================
class MLP {
public:
    std::vector<Linear> layers;
    std::string act_name;
    activation::ActivationFn act_fn;
    activation::ActivationFn act_grad_fn;
    int num_layers;
    bool skip_connections;

    // Caches for backprop
    std::vector<Tensor> pre_act_cache;   // outputs before activation
    std::vector<Tensor> post_act_cache;  // outputs after activation

    MLP() : act_fn(nullptr), act_grad_fn(nullptr), num_layers(0), skip_connections(false) {}

    MLP(const std::string& prefix,
        int in_features, int layer_size, int out_features,
        int num_hidden_layers,
        const std::string& activation = "gelu",
        bool skip_conn = false)
        : act_name(activation), num_layers(num_hidden_layers), skip_connections(skip_conn) {

        auto [fn, gfn] = activation::get_activation(activation);
        act_fn = fn;
        act_grad_fn = gfn;

        // Hidden layers
        int in_f = in_features;
        for (int i = 0; i < num_hidden_layers; ++i) {
            layers.emplace_back(prefix + ".layer" + std::to_string(i), in_f, layer_size);
            in_f = layer_size;
        }
        // Final layer (no activation)
        layers.emplace_back(prefix + ".final", in_f, out_features);
    }

    Tensor forward(const Tensor& x) {
        pre_act_cache.clear();
        post_act_cache.clear();

        Tensor h = x;
        Tensor skip;

        for (int i = 0; i < (int)layers.size(); ++i) {
            h = layers[i].forward(h);
            pre_act_cache.push_back(h);

            if (i < (int)layers.size() - 1) {
                // Apply activation to hidden layers
                h = h.apply(act_fn);
                post_act_cache.push_back(h);

                // Skip connections every 2 layers
                if (skip_connections && i % 2 == 0) {
                    if (!skip.empty() && skip.numel() == h.numel()) {
                        h = h + skip;
                    }
                    skip = h;
                }
            } else {
                post_act_cache.push_back(h); // no activation on final
            }
        }
        return h;
    }

    Tensor backward(const Tensor& grad_output) {
        Tensor dh = grad_output;

        for (int i = (int)layers.size() - 1; i >= 0; --i) {
            if (i < (int)layers.size() - 1) {
                // Multiply by activation gradient
                Tensor& pre = pre_act_cache[i];
                Tensor act_g(pre.shape);
                for (int j = 0; j < pre.numel(); ++j)
                    act_g.data[j] = act_grad_fn(pre.data[j]);
                dh = dh * act_g;
            }
            dh = layers[i].backward(dh);
        }
        return dh;
    }

    void zero_grad() {
        for (auto& l : layers) l.zero_grad();
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        for (auto& l : layers) {
            auto lp = l.parameters();
            params.insert(params.end(), lp.begin(), lp.end());
        }
        return params;
    }
};

// ============================================================
// SpectralConv1d - 1D Fourier spectral convolution layer
// Mirrors Modulus SpectralConv1d: FFT -> complex multiply -> IFFT
// ============================================================
class SpectralConv1d {
public:
    int in_channels, out_channels, modes;
    // Spectral weights stored as separate real/imag parameter tensors
    // shape: (in_ch, out_ch, modes) each
    Parameter weights_real;
    Parameter weights_imag;

    // Caches for backprop
    int cached_batch = 0;
    int cached_spatial = 0;
    int cached_fft_len = 0;
    std::vector<Complex> cached_input_fft;
    Tensor cached_input;

    SpectralConv1d() : in_channels(0), out_channels(0), modes(0) {}

    SpectralConv1d(const std::string& prefix, int in_ch, int out_ch, int num_modes)
        : in_channels(in_ch), out_channels(out_ch), modes(num_modes) {

        double scale = 1.0 / (in_ch * out_ch);
        Tensor wr({in_ch, out_ch, modes});
        Tensor wi({in_ch, out_ch, modes});
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < wr.numel(); ++i) {
            wr.data[i] = scale * dist(global_rng());
            wi.data[i] = scale * dist(global_rng());
        }
        weights_real = Parameter(prefix + ".wr", wr);
        weights_imag = Parameter(prefix + ".wi", wi);
    }

    // Forward: input shape (batch, in_channels, spatial_len)
    // Output shape (batch, out_channels, spatial_len)
    Tensor forward(const Tensor& x) {
        int batch = x.shape[0];
        int spatial = x.shape[2];
        int fft_len = fft::next_pow2(spatial);
        int rfft_len = fft_len / 2 + 1;

        cached_batch = batch;
        cached_spatial = spatial;
        cached_fft_len = fft_len;
        cached_input = x;

        cached_input_fft.resize(batch * in_channels * rfft_len);

        Tensor output({batch, out_channels, spatial}, 0.0);

        for (int b = 0; b < batch; ++b) {
            // FFT of each input channel
            std::vector<std::vector<Complex>> x_ft(in_channels);
            for (int ic = 0; ic < in_channels; ++ic) {
                std::vector<double> channel_data(spatial);
                for (int s = 0; s < spatial; ++s)
                    channel_data[s] = x.at3(b, ic, s);
                x_ft[ic] = fft::rfft(channel_data);
                for (int f = 0; f < rfft_len; ++f)
                    cached_input_fft[(b * in_channels + ic) * rfft_len + f] = x_ft[ic][f];
            }

            // Spectral convolution: out_ft[oc][f] = sum_ic x_ft[ic][f] * W[ic][oc][f]
            for (int oc = 0; oc < out_channels; ++oc) {
                std::vector<Complex> out_ft(rfft_len, {0, 0});
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int f = 0; f < std::min(modes, rfft_len); ++f) {
                        int widx = (ic * out_channels + oc) * modes + f;
                        Complex w(weights_real.value.data[widx],
                                  weights_imag.value.data[widx]);
                        out_ft[f] += x_ft[ic][f] * w;
                    }
                }
                std::vector<double> y = fft::irfft(out_ft, spatial);
                for (int s = 0; s < spatial; ++s)
                    output.at3(b, oc, s) = y[s];
            }
        }
        return output;
    }

    // Backward: grad_output shape (batch, out_channels, spatial_len)
    Tensor backward(const Tensor& grad_output) {
        int batch = cached_batch;
        int spatial = cached_spatial;
        int fft_len = cached_fft_len;
        int rfft_len = fft_len / 2 + 1;

        Tensor grad_input({batch, in_channels, spatial}, 0.0);

        for (int b = 0; b < batch; ++b) {
            std::vector<std::vector<Complex>> go_ft(out_channels);
            for (int oc = 0; oc < out_channels; ++oc) {
                std::vector<double> go_data(spatial);
                for (int s = 0; s < spatial; ++s)
                    go_data[s] = grad_output.at3(b, oc, s);
                go_ft[oc] = fft::rfft(go_data);
            }

            for (int ic = 0; ic < in_channels; ++ic) {
                std::vector<Complex> dx_ft(rfft_len, {0, 0});
                for (int oc = 0; oc < out_channels; ++oc) {
                    for (int f = 0; f < std::min(modes, rfft_len); ++f) {
                        int widx = (ic * out_channels + oc) * modes + f;
                        Complex w(weights_real.value.data[widx],
                                  weights_imag.value.data[widx]);
                        Complex xf = cached_input_fft[(b * in_channels + ic) * rfft_len + f];

                        // Weight gradient
                        Complex dw = xf * std::conj(go_ft[oc][f]);
                        weights_real.grad.data[widx] += dw.real();
                        weights_imag.grad.data[widx] += dw.imag();

                        // Input gradient
                        dx_ft[f] += go_ft[oc][f] * std::conj(w);
                    }
                }
                std::vector<double> dx = fft::irfft(dx_ft, spatial);
                for (int s = 0; s < spatial; ++s)
                    grad_input.at3(b, ic, s) += dx[s];
            }
        }
        return grad_input;
    }

    void zero_grad() {
        std::fill(weights_real.grad.data.begin(), weights_real.grad.data.end(), 0.0);
        std::fill(weights_imag.grad.data.begin(), weights_imag.grad.data.end(), 0.0);
    }

    std::vector<Parameter*> parameters() {
        return {&weights_real, &weights_imag};
    }
};

// ============================================================
// Conv1x1 - Channel mixing (mirrors nn.Conv1d with kernel=1)
// ============================================================
class Conv1x1 {
public:
    Parameter weight; // (out_ch, in_ch)
    Parameter bias;   // (out_ch)
    int in_channels, out_channels;
    Tensor input_cache;

    Conv1x1() : in_channels(0), out_channels(0) {}

    Conv1x1(const std::string& prefix, int in_ch, int out_ch)
        : in_channels(in_ch), out_channels(out_ch) {
        weight = Parameter(prefix + ".weight", Tensor::xavier_uniform(out_ch, in_ch));
        bias = Parameter(prefix + ".bias", Tensor::zeros({1, out_ch}));
    }

    // Forward: x (batch, in_ch, spatial) -> (batch, out_ch, spatial)
    Tensor forward(const Tensor& x) {
        input_cache = x;
        int batch = x.shape[0], spatial = x.shape[2];
        Tensor out({batch, out_channels, spatial}, 0.0);

        for (int b = 0; b < batch; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int s = 0; s < spatial; ++s) {
                    double val = bias.value.data[oc];
                    for (int ic = 0; ic < in_channels; ++ic)
                        val += weight.value.at2(oc, ic) * x.at3(b, ic, s);
                    out.at3(b, oc, s) = val;
                }
            }
        }
        return out;
    }

    Tensor backward(const Tensor& grad_output) {
        int batch = grad_output.shape[0], spatial = grad_output.shape[2];
        Tensor grad_input({batch, in_channels, spatial}, 0.0);

        for (int b = 0; b < batch; ++b) {
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int s = 0; s < spatial; ++s) {
                    double go = grad_output.at3(b, oc, s);
                    bias.grad.data[oc] += go;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        weight.grad.at2(oc, ic) += go * input_cache.at3(b, ic, s);
                        grad_input.at3(b, ic, s) += go * weight.value.at2(oc, ic);
                    }
                }
            }
        }
        return grad_input;
    }

    void zero_grad() {
        std::fill(weight.grad.data.begin(), weight.grad.data.end(), 0.0);
        std::fill(bias.grad.data.begin(), bias.grad.data.end(), 0.0);
    }

    std::vector<Parameter*> parameters() { return {&weight, &bias}; }
};

// ============================================================
// FNO1D - Fourier Neural Operator (1D)
// Architecture mirrors Modulus FNO:
//   lift -> [SpectralConv + Conv1x1 skip + activation] x N -> decode
// ============================================================
class FNO1D {
public:
    int in_channels, out_channels, latent_channels;
    int num_fno_layers, num_fno_modes;
    bool coord_features;

    // Lift network: 2 Conv1x1 layers with activation
    Conv1x1 lift1, lift2;

    // FNO layers
    std::vector<SpectralConv1d> spconv_layers;
    std::vector<Conv1x1> conv_layers;

    // Decoder MLP
    MLP decoder;

    // Caches for backprop
    std::vector<Tensor> fno_pre_act;
    std::vector<Tensor> fno_post_act;
    Tensor lift1_out, lift1_act;
    Tensor lifted;

    FNO1D() : in_channels(0), out_channels(0), latent_channels(0),
              num_fno_layers(0), num_fno_modes(0), coord_features(true) {}

    FNO1D(int in_ch, int out_ch,
          int latent_ch = 32,
          int n_fno_layers = 4,
          int n_fno_modes = 16,
          int decoder_layers = 1,
          int decoder_layer_size = 32,
          const std::string& act_name = "gelu",
          bool coord_feat = true)
        : in_channels(in_ch), out_channels(out_ch), latent_channels(latent_ch),
          num_fno_layers(n_fno_layers), num_fno_modes(n_fno_modes),
          coord_features(coord_feat) {

        int lift_in = in_ch + (coord_feat ? 1 : 0);

        lift1 = Conv1x1("fno.lift1", lift_in, latent_ch / 2);
        lift2 = Conv1x1("fno.lift2", latent_ch / 2, latent_ch);

        for (int i = 0; i < n_fno_layers; ++i) {
            spconv_layers.emplace_back("fno.spconv" + std::to_string(i),
                                       latent_ch, latent_ch, n_fno_modes);
            conv_layers.emplace_back("fno.conv" + std::to_string(i),
                                     latent_ch, latent_ch);
        }

        decoder = MLP("fno.decoder", latent_ch, decoder_layer_size, out_ch,
                       decoder_layers, act_name);
    }

    // Forward: x (batch, in_channels, spatial_len)
    // Output: (batch, out_channels, spatial_len)
    Tensor forward(const Tensor& x) {
        int batch = x.shape[0];
        int spatial = x.shape[2];

        Tensor h = x;
        if (coord_features) {
            Tensor combined({batch, in_channels + 1, spatial});
            for (int b = 0; b < batch; ++b) {
                for (int c = 0; c < in_channels; ++c)
                    for (int s = 0; s < spatial; ++s)
                        combined.at3(b, c, s) = x.at3(b, c, s);
                for (int s = 0; s < spatial; ++s)
                    combined.at3(b, in_channels, s) = (double)s / std::max(spatial - 1, 1);
            }
            h = combined;
        }

        // Lift
        lift1_out = lift1.forward(h);
        lift1_act = lift1_out.apply(activation::gelu);
        lifted = lift2.forward(lift1_act);

        // FNO layers
        fno_pre_act.clear();
        fno_post_act.clear();
        h = lifted;

        for (int k = 0; k < num_fno_layers; ++k) {
            Tensor sp = spconv_layers[k].forward(h);
            Tensor cv = conv_layers[k].forward(h);
            Tensor combined = sp + cv;
            fno_pre_act.push_back(combined);

            if (k < num_fno_layers - 1) {
                combined = combined.apply(activation::gelu);
            }
            fno_post_act.push_back(combined);
            h = combined;
        }

        // Decode: reshape to pointwise (batch*spatial, latent) -> MLP -> reshape
        Tensor points({batch * spatial, latent_channels});
        for (int b = 0; b < batch; ++b)
            for (int s = 0; s < spatial; ++s)
                for (int c = 0; c < latent_channels; ++c)
                    points.at2(b * spatial + s, c) = h.at3(b, c, s);

        Tensor decoded = decoder.forward(points);

        Tensor output({batch, out_channels, spatial});
        for (int b = 0; b < batch; ++b)
            for (int s = 0; s < spatial; ++s)
                for (int c = 0; c < out_channels; ++c)
                    output.at3(b, c, s) = decoded.at2(b * spatial + s, c);

        return output;
    }

    Tensor backward(const Tensor& grad_output) {
        int batch = grad_output.shape[0];
        int spatial = grad_output.shape[2];

        // Reshape grad to pointwise
        Tensor grad_points({batch * spatial, out_channels});
        for (int b = 0; b < batch; ++b)
            for (int s = 0; s < spatial; ++s)
                for (int c = 0; c < out_channels; ++c)
                    grad_points.at2(b * spatial + s, c) = grad_output.at3(b, c, s);

        Tensor grad_latent_flat = decoder.backward(grad_points);

        // Reshape to (batch, latent, spatial)
        Tensor dh({batch, latent_channels, spatial});
        for (int b = 0; b < batch; ++b)
            for (int s = 0; s < spatial; ++s)
                for (int c = 0; c < latent_channels; ++c)
                    dh.at3(b, c, s) = grad_latent_flat.at2(b * spatial + s, c);

        // Backward through FNO layers (reverse order)
        for (int k = num_fno_layers - 1; k >= 0; --k) {
            if (k < num_fno_layers - 1) {
                Tensor& pre = fno_pre_act[k];
                for (int i = 0; i < dh.numel(); ++i)
                    dh.data[i] *= activation::gelu_grad(pre.data[i]);
            }
            Tensor dsp = spconv_layers[k].backward(dh);
            Tensor dcv = conv_layers[k].backward(dh);
            dh = dsp + dcv;
        }

        // Backward through lift
        Tensor dlift2 = lift2.backward(dh);
        for (int i = 0; i < dlift2.numel(); ++i)
            dlift2.data[i] *= activation::gelu_grad(lift1_out.data[i]);
        Tensor dlift1 = lift1.backward(dlift2);

        return dlift1;
    }

    void zero_grad() {
        lift1.zero_grad();
        lift2.zero_grad();
        for (auto& sp : spconv_layers) sp.zero_grad();
        for (auto& cv : conv_layers) cv.zero_grad();
        decoder.zero_grad();
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        auto add = [&](std::vector<Parameter*> ps) {
            params.insert(params.end(), ps.begin(), ps.end());
        };
        add(lift1.parameters());
        add(lift2.parameters());
        for (auto& sp : spconv_layers) add(sp.parameters());
        for (auto& cv : conv_layers) add(cv.parameters());
        add(decoder.parameters());
        return params;
    }
};

// ============================================================
// DeepONet - Deep Operator Network
// Branch network: encodes input function samples
// Trunk network: encodes query coordinates
// Output: dot product of branch and trunk outputs
// ============================================================
class DeepONet {
public:
    MLP branch_net;
    MLP trunk_net;
    Parameter final_bias;
    int p_dim;

    DeepONet() : p_dim(0) {}

    DeepONet(int branch_input_dim, int trunk_input_dim, int p,
             int branch_layers = 4, int branch_width = 64,
             int trunk_layers = 4, int trunk_width = 64,
             const std::string& activation = "gelu")
        : p_dim(p) {
        branch_net = MLP("deeponet.branch", branch_input_dim, branch_width, p,
                         branch_layers, activation);
        trunk_net = MLP("deeponet.trunk", trunk_input_dim, trunk_width, p,
                        trunk_layers, activation);
        final_bias = Parameter("deeponet.bias", Tensor::zeros({1, 1}));
    }

    // Forward:
    //   branch_input: (batch, branch_input_dim)
    //   trunk_input:  (num_points, trunk_input_dim)
    // Output: (batch, num_points)
    Tensor forward(const Tensor& branch_input, const Tensor& trunk_input) {
        Tensor b_out = branch_net.forward(branch_input);
        Tensor t_out = trunk_net.forward(trunk_input);

        int batch = b_out.shape[0];
        int num_pts = t_out.shape[0];

        Tensor t_out_T = t_out.T();
        Tensor dot = Tensor::matmul(b_out, t_out_T);

        double bias = final_bias.value.data[0];
        Tensor output({batch, num_pts});
        for (int i = 0; i < dot.numel(); ++i)
            output.data[i] = dot.data[i] + bias;

        return output;
    }

    void backward(const Tensor& grad_output) {
        Tensor& b_out = branch_net.post_act_cache.back();
        Tensor& t_out = trunk_net.post_act_cache.back();

        for (int i = 0; i < grad_output.numel(); ++i)
            final_bias.grad.data[0] += grad_output.data[i];

        Tensor grad_b = Tensor::matmul(grad_output, t_out);
        Tensor go_t = grad_output.T();
        Tensor grad_t = Tensor::matmul(go_t, b_out);

        branch_net.backward(grad_b);
        trunk_net.backward(grad_t);
    }

    void zero_grad() {
        branch_net.zero_grad();
        trunk_net.zero_grad();
        std::fill(final_bias.grad.data.begin(), final_bias.grad.data.end(), 0.0);
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        auto bp = branch_net.parameters();
        auto tp = trunk_net.parameters();
        params.insert(params.end(), bp.begin(), bp.end());
        params.insert(params.end(), tp.begin(), tp.end());
        params.push_back(&final_bias);
        return params;
    }
};

// ============================================================
// Physics Constraint - PDE residual loss
// ============================================================
class PhysicsConstraint {
public:
    using ResidualFn = std::function<Tensor(const Tensor& x, const Tensor& u,
                                            const std::vector<double>& params)>;

    ResidualFn residual_fn;
    double lambda;
    std::vector<double> pde_params;

    PhysicsConstraint() : lambda(1.0) {}

    PhysicsConstraint(ResidualFn fn, double weight = 1.0,
                      const std::vector<double>& params = {})
        : residual_fn(fn), lambda(weight), pde_params(params) {}

    double compute_loss(const Tensor& x_coords, const Tensor& u_pred) {
        Tensor residual = residual_fn(x_coords, u_pred, pde_params);
        return lambda * residual.norm_sq() / residual.numel();
    }

    // Finite difference du/dx for 1D (batch, ch, spatial)
    static Tensor finite_diff_dx(const Tensor& u, double dx) {
        int batch = u.shape[0], ch = u.shape[1], spatial = u.shape[2];
        Tensor dudx({batch, ch, spatial}, 0.0);
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < ch; ++c) {
                for (int s = 1; s < spatial - 1; ++s)
                    dudx.at3(b, c, s) = (u.at3(b, c, s+1) - u.at3(b, c, s-1)) / (2.0*dx);
                dudx.at3(b, c, 0) = (u.at3(b, c, 1) - u.at3(b, c, 0)) / dx;
                dudx.at3(b, c, spatial-1) = (u.at3(b, c, spatial-1) - u.at3(b, c, spatial-2)) / dx;
            }
        return dudx;
    }

    // Second derivative d2u/dx2
    static Tensor finite_diff_d2x(const Tensor& u, double dx) {
        int batch = u.shape[0], ch = u.shape[1], spatial = u.shape[2];
        Tensor d2udx2({batch, ch, spatial}, 0.0);
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < ch; ++c)
                for (int s = 1; s < spatial - 1; ++s)
                    d2udx2.at3(b, c, s) = (u.at3(b, c, s+1) - 2.0*u.at3(b, c, s) + u.at3(b, c, s-1)) / (dx*dx);
        return d2udx2;
    }
};

// ============================================================
// Adam Optimizer
// ============================================================
class Adam {
public:
    double lr, beta1, beta2, eps;
    int t;

    struct State {
        Tensor m, v;
    };
    std::unordered_map<std::string, State> states;

    Adam(double learning_rate = 1e-3, double b1 = 0.9, double b2 = 0.999,
         double epsilon = 1e-8)
        : lr(learning_rate), beta1(b1), beta2(b2), eps(epsilon), t(0) {}

    void step(std::vector<Parameter*>& params) {
        t++;
        for (auto* p : params) {
            if (states.find(p->name) == states.end()) {
                states[p->name] = {Tensor::zeros(p->value.shape),
                                   Tensor::zeros(p->value.shape)};
            }
            auto& st = states[p->name];
            int n = p->value.numel();
            for (int i = 0; i < n; ++i) {
                double g = p->grad.data[i];
                st.m.data[i] = beta1 * st.m.data[i] + (1.0 - beta1) * g;
                st.v.data[i] = beta2 * st.v.data[i] + (1.0 - beta2) * g * g;
                double m_hat = st.m.data[i] / (1.0 - std::pow(beta1, t));
                double v_hat = st.v.data[i] / (1.0 - std::pow(beta2, t));
                p->value.data[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
    }
};

// ============================================================
// Training Loop
// ============================================================
class Trainer {
public:
    // Train FNO with data loss (and optional physics loss)
    static void train_fno(
        FNO1D& model,
        const std::vector<Tensor>& inputs,
        const std::vector<Tensor>& targets,
        int epochs,
        double learning_rate = 1e-3,
        int print_every = 10,
        PhysicsConstraint* physics = nullptr,
        const Tensor* x_coords = nullptr) {

        Adam optimizer(learning_rate);
        auto params = model.parameters();
        int n_samples = (int)inputs.size();

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;

            for (int s = 0; s < n_samples; ++s) {
                model.zero_grad();
                Tensor pred = model.forward(inputs[s]);

                Tensor diff = pred - targets[s];
                double data_loss = diff.norm_sq() / diff.numel();

                double phys_loss = 0.0;
                if (physics && x_coords)
                    phys_loss = physics->compute_loss(*x_coords, pred);

                total_loss += data_loss + phys_loss;

                // dL/dpred = 2*(pred - target) / N
                Tensor grad_out(pred.shape);
                for (int i = 0; i < pred.numel(); ++i)
                    grad_out.data[i] = 2.0 * diff.data[i] / diff.numel();

                model.backward(grad_out);
                optimizer.step(params);
            }

            if ((epoch + 1) % print_every == 0 || epoch == 0) {
                std::cout << "Epoch " << std::setw(4) << epoch + 1
                          << "/" << epochs
                          << "  Loss: " << std::scientific << std::setprecision(6)
                          << total_loss / n_samples << std::endl;
            }
        }
    }

    // Train DeepONet
    static void train_deeponet(
        DeepONet& model,
        const std::vector<Tensor>& branch_inputs,
        const std::vector<Tensor>& trunk_inputs,
        const std::vector<Tensor>& targets,
        int epochs,
        double learning_rate = 1e-3,
        int print_every = 10) {

        Adam optimizer(learning_rate);
        auto params = model.parameters();
        int n_samples = (int)branch_inputs.size();

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;

            for (int s = 0; s < n_samples; ++s) {
                model.zero_grad();
                Tensor pred = model.forward(branch_inputs[s], trunk_inputs[s]);
                Tensor diff = pred - targets[s];
                double loss = diff.norm_sq() / diff.numel();
                total_loss += loss;

                Tensor grad_out(pred.shape);
                for (int i = 0; i < pred.numel(); ++i)
                    grad_out.data[i] = 2.0 * diff.data[i] / diff.numel();

                model.backward(grad_out);
                optimizer.step(params);
            }

            if ((epoch + 1) % print_every == 0 || epoch == 0) {
                std::cout << "Epoch " << std::setw(4) << epoch + 1
                          << "/" << epochs
                          << "  Loss: " << std::scientific << std::setprecision(6)
                          << total_loss / n_samples << std::endl;
            }
        }
    }
};

// ============================================================
// Burgers equation utilities
// u_t + u * u_x = nu * u_xx
// ============================================================
namespace burgers {

// Generate initial condition: sum of sine waves
inline Tensor generate_initial_condition(int spatial_len, int batch = 1) {
    Tensor u({batch, 1, spatial_len});
    std::uniform_real_distribution<double> amp_dist(0.5, 2.0);
    std::uniform_real_distribution<double> freq_dist(1.0, 4.0);
    std::uniform_real_distribution<double> phase_dist(0.0, 2.0 * M_PI);

    for (int b = 0; b < batch; ++b) {
        int n_modes = 2 + (int)(global_rng()() % 3);
        for (int s = 0; s < spatial_len; ++s) {
            double x = 2.0 * M_PI * s / spatial_len;
            double val = 0;
            for (int m = 0; m < n_modes; ++m) {
                double amp = amp_dist(global_rng());
                double freq = (double)(1 + (int)(freq_dist(global_rng())));
                double phase = phase_dist(global_rng());
                val += amp * std::sin(freq * x + phase);
            }
            u.at3(b, 0, s) = val;
        }
    }
    return u;
}

// Forward Euler solver for Burgers equation on [0, 2*pi] periodic
inline Tensor solve(const Tensor& u0, double nu, double dt, int n_steps) {
    int batch = u0.shape[0];
    int spatial = u0.shape[2];
    double dx = 2.0 * M_PI / spatial;

    Tensor u = u0;
    for (int step = 0; step < n_steps; ++step) {
        Tensor u_new({batch, 1, spatial});
        for (int b = 0; b < batch; ++b) {
            for (int s = 0; s < spatial; ++s) {
                int sp = (s + 1) % spatial;
                int sm = (s - 1 + spatial) % spatial;
                double ux = (u.at3(b, 0, sp) - u.at3(b, 0, sm)) / (2.0 * dx);
                double uxx = (u.at3(b, 0, sp) - 2.0 * u.at3(b, 0, s) + u.at3(b, 0, sm)) / (dx * dx);
                u_new.at3(b, 0, s) = u.at3(b, 0, s) + dt * (-u.at3(b, 0, s) * ux + nu * uxx);
            }
        }
        u = u_new;
    }
    return u;
}

// Create Burgers PDE physics constraint
inline PhysicsConstraint make_constraint(double nu, double dx, double lambda = 0.1) {
    return PhysicsConstraint(
        [nu, dx](const Tensor& /*x_coords*/, const Tensor& u,
                 const std::vector<double>& /*params*/) -> Tensor {
            Tensor ux = PhysicsConstraint::finite_diff_dx(u, dx);
            Tensor uxx = PhysicsConstraint::finite_diff_d2x(u, dx);
            Tensor residual(u.shape, 0.0);
            for (int i = 0; i < u.numel(); ++i)
                residual.data[i] = u.data[i] * ux.data[i] - nu * uxx.data[i];
            return residual;
        },
        lambda);
}

} // namespace burgers

} // namespace modulus

#endif // MODULUS_H
