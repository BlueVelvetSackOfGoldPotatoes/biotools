// modulus_models.h - Extended model architectures
// Port of physicsnemo/models/
// Includes: FNO2D, FNO3D, AFNO, MeshGraphNet, GraphCastNet,
//           SRResNet, Pix2Pix, One2ManyRNN, SwinRNN, DLWP,
//           FullyConnected (extended), SIREN, UNet
//
// C++17, no external dependencies.

#ifndef MODULUS_MODELS_H
#define MODULUS_MODELS_H

#include "modulus.h"
#include "modulus_activations.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace modulus {
namespace models {

// ============================================================
// Weight initialization methods
// Port of physicsnemo/nn/module/utils/weight_init.py
// ============================================================
namespace init {

inline Tensor xavier_uniform(int rows, int cols) {
    return Tensor::xavier_uniform(rows, cols);
}

inline Tensor xavier_normal(int rows, int cols) {
    double std_val = std::sqrt(2.0 / (rows + cols));
    Tensor t({rows, cols});
    std::normal_distribution<double> dist(0.0, std_val);
    for (auto& v : t.data) v = dist(global_rng());
    return t;
}

inline Tensor kaiming_uniform(int rows, int cols) {
    double limit = std::sqrt(6.0 / rows);
    Tensor t({rows, cols});
    std::uniform_real_distribution<double> dist(-limit, limit);
    for (auto& v : t.data) v = dist(global_rng());
    return t;
}

inline Tensor kaiming_normal(int rows, int cols) {
    double std_val = std::sqrt(2.0 / rows);
    Tensor t({rows, cols});
    std::normal_distribution<double> dist(0.0, std_val);
    for (auto& v : t.data) v = dist(global_rng());
    return t;
}

inline Tensor truncated_normal(int rows, int cols, double std_val = 0.02) {
    Tensor t({rows, cols});
    std::normal_distribution<double> dist(0.0, std_val);
    for (auto& v : t.data) {
        do { v = dist(global_rng()); } while (std::abs(v) > 2.0 * std_val);
    }
    return t;
}

inline Tensor uniform(const std::vector<int>& shape, double lo, double hi) {
    Tensor t(shape);
    std::uniform_real_distribution<double> dist(lo, hi);
    for (auto& v : t.data) v = dist(global_rng());
    return t;
}

} // namespace init

// ============================================================
// Layer Normalization
// ============================================================
class LayerNorm {
public:
    int features;
    Parameter gamma, beta;
    double eps;

    LayerNorm() : features(0), eps(1e-6) {}

    LayerNorm(const std::string& prefix, int feat, double epsilon = 1e-6)
        : features(feat), eps(epsilon) {
        gamma = Parameter(prefix + ".gamma", Tensor::ones({1, feat}));
        beta = Parameter(prefix + ".beta", Tensor::zeros({1, feat}));
    }

    // Forward: x (batch, features) -> normalized (batch, features)
    Tensor forward(const Tensor& x) {
        int batch = x.shape[0];
        Tensor out(x.shape);
        for (int b = 0; b < batch; ++b) {
            double mean = 0, var = 0;
            for (int f = 0; f < features; ++f)
                mean += x.at2(b, f);
            mean /= features;
            for (int f = 0; f < features; ++f) {
                double d = x.at2(b, f) - mean;
                var += d * d;
            }
            var /= features;
            double inv_std = 1.0 / std::sqrt(var + eps);
            for (int f = 0; f < features; ++f) {
                out.at2(b, f) = gamma.value.data[f] * (x.at2(b, f) - mean) * inv_std
                              + beta.value.data[f];
            }
        }
        return out;
    }

    std::vector<Parameter*> parameters() { return {&gamma, &beta}; }
    void zero_grad() {
        std::fill(gamma.grad.data.begin(), gamma.grad.data.end(), 0.0);
        std::fill(beta.grad.data.begin(), beta.grad.data.end(), 0.0);
    }
};

// ============================================================
// BatchNorm1D
// ============================================================
class BatchNorm1D {
public:
    int features;
    Parameter gamma, beta;
    Tensor running_mean, running_var;
    double eps, momentum;
    bool training;

    BatchNorm1D() : features(0), eps(1e-5), momentum(0.1), training(true) {}

    BatchNorm1D(const std::string& prefix, int feat, double epsilon = 1e-5)
        : features(feat), eps(epsilon), momentum(0.1), training(true) {
        gamma = Parameter(prefix + ".gamma", Tensor::ones({1, feat}));
        beta = Parameter(prefix + ".beta", Tensor::zeros({1, feat}));
        running_mean = Tensor::zeros({1, feat});
        running_var = Tensor::ones({1, feat});
    }

    Tensor forward(const Tensor& x) {
        int batch = x.shape[0];
        Tensor out(x.shape);

        for (int f = 0; f < features; ++f) {
            double mean = 0, var = 0;
            if (training) {
                for (int b = 0; b < batch; ++b) mean += x.at2(b, f);
                mean /= batch;
                for (int b = 0; b < batch; ++b) {
                    double d = x.at2(b, f) - mean;
                    var += d * d;
                }
                var /= batch;
                running_mean.data[f] = (1 - momentum) * running_mean.data[f] + momentum * mean;
                running_var.data[f] = (1 - momentum) * running_var.data[f] + momentum * var;
            } else {
                mean = running_mean.data[f];
                var = running_var.data[f];
            }
            double inv_std = 1.0 / std::sqrt(var + eps);
            for (int b = 0; b < batch; ++b) {
                out.at2(b, f) = gamma.value.data[f] * (x.at2(b, f) - mean) * inv_std
                              + beta.value.data[f];
            }
        }
        return out;
    }

    std::vector<Parameter*> parameters() { return {&gamma, &beta}; }
    void zero_grad() {
        std::fill(gamma.grad.data.begin(), gamma.grad.data.end(), 0.0);
        std::fill(beta.grad.data.begin(), beta.grad.data.end(), 0.0);
    }
};

// ============================================================
// Dropout
// ============================================================
class Dropout {
public:
    double rate;
    bool training;

    Dropout(double r = 0.0) : rate(r), training(true) {}

    Tensor forward(const Tensor& x) {
        if (!training || rate <= 0.0) return x;
        Tensor out(x.shape);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double scale = 1.0 / (1.0 - rate);
        for (int i = 0; i < x.numel(); ++i) {
            out.data[i] = (dist(global_rng()) > rate) ? x.data[i] * scale : 0.0;
        }
        return out;
    }
};

// ============================================================
// SpectralConv2d - 2D Fourier spectral convolution
// Port of physicsnemo/nn/module/spectral_layers.py SpectralConv2d
// ============================================================
class SpectralConv2d {
public:
    int in_channels, out_channels, modes1, modes2;
    // Two sets of weights for positive and negative freq halves
    Parameter w1_real, w1_imag; // (in_ch, out_ch, modes1, modes2)
    Parameter w2_real, w2_imag; // (in_ch, out_ch, modes1, modes2)

    SpectralConv2d() : in_channels(0), out_channels(0), modes1(0), modes2(0) {}

    SpectralConv2d(const std::string& prefix, int in_ch, int out_ch, int m1, int m2)
        : in_channels(in_ch), out_channels(out_ch), modes1(m1), modes2(m2) {
        double scale = 1.0 / (in_ch * out_ch);
        auto make_w = [&](const std::string& n) {
            Tensor w({in_ch, out_ch, m1, m2});
            std::uniform_real_distribution<double> dist(0, 1);
            for (auto& v : w.data) v = scale * dist(global_rng());
            return Parameter(prefix + "." + n, w);
        };
        w1_real = make_w("w1r"); w1_imag = make_w("w1i");
        w2_real = make_w("w2r"); w2_imag = make_w("w2i");
    }

    // Forward: x (batch, in_ch, h, w) -> (batch, out_ch, h, w)
    Tensor forward(const Tensor& x) {
        int batch = x.shape[0];
        int h = x.shape[2], w = x.shape[3];

        // 2D FFT per channel -> spectral multiply -> IFFT
        // Simplified: operate on each spatial location pair
        Tensor output({batch, out_channels, h, w}, 0.0);

        for (int b = 0; b < batch; ++b) {
            // FFT each input channel row-by-row, then column-by-column
            // For efficiency, compute 1D FFTs along each dimension

            // Step 1: row-wise FFT for each channel
            int rfft_w = fft::next_pow2(w) / 2 + 1;
            int fft_h = fft::next_pow2(h);
            int rfft_h = fft_h / 2 + 1;

            // Full 2D FFT: first along w (rfft), then along h (fft)
            // x_ft[ic][fh][fw] = complex
            std::vector<std::vector<std::vector<Complex>>> x_ft(
                in_channels, std::vector<std::vector<Complex>>(fft_h, std::vector<Complex>(rfft_w, {0, 0})));

            for (int ic = 0; ic < in_channels; ++ic) {
                // Row-wise FFT
                std::vector<std::vector<Complex>> row_ft(h);
                for (int ih = 0; ih < h; ++ih) {
                    std::vector<double> row(w);
                    for (int iw = 0; iw < w; ++iw) {
                        int idx = ((b * in_channels + ic) * h + ih) * w + iw;
                        row[iw] = x.data[idx];
                    }
                    row_ft[ih] = fft::rfft(row);
                }

                // Column-wise FFT on the row-FFT results
                for (int fw = 0; fw < rfft_w && fw < (int)row_ft[0].size(); ++fw) {
                    std::vector<Complex> col(fft_h, {0, 0});
                    for (int ih = 0; ih < h; ++ih) {
                        if (fw < (int)row_ft[ih].size())
                            col[ih] = row_ft[ih][fw];
                    }
                    fft::fft_inplace(col, false);
                    for (int fh = 0; fh < fft_h; ++fh)
                        x_ft[ic][fh][fw] = col[fh];
                }
            }

            // Spectral multiply: two mode groups
            std::vector<std::vector<std::vector<Complex>>> out_ft(
                out_channels, std::vector<std::vector<Complex>>(fft_h, std::vector<Complex>(rfft_w, {0, 0})));

            for (int oc = 0; oc < out_channels; ++oc) {
                for (int ic = 0; ic < in_channels; ++ic) {
                    // Positive freq modes (top-left block)
                    for (int f1 = 0; f1 < std::min(modes1, rfft_h); ++f1) {
                        for (int f2 = 0; f2 < std::min(modes2, rfft_w); ++f2) {
                            int widx = ((ic * out_channels + oc) * modes1 + f1) * modes2 + f2;
                            Complex w1(w1_real.value.data[widx], w1_imag.value.data[widx]);
                            out_ft[oc][f1][f2] += x_ft[ic][f1][f2] * w1;
                        }
                    }
                    // Negative freq modes (bottom-left block)
                    for (int f1 = 0; f1 < std::min(modes1, rfft_h); ++f1) {
                        for (int f2 = 0; f2 < std::min(modes2, rfft_w); ++f2) {
                            int widx = ((ic * out_channels + oc) * modes1 + f1) * modes2 + f2;
                            Complex w2(w2_real.value.data[widx], w2_imag.value.data[widx]);
                            int neg_f1 = fft_h - modes1 + f1;
                            if (neg_f1 >= 0 && neg_f1 < fft_h)
                                out_ft[oc][neg_f1][f2] += x_ft[ic][neg_f1][f2] * w2;
                        }
                    }
                }
            }

            // Inverse 2D FFT: column-wise IFFT then row-wise IRFFT
            for (int oc = 0; oc < out_channels; ++oc) {
                // Column-wise IFFT
                for (int fw = 0; fw < rfft_w; ++fw) {
                    std::vector<Complex> col(fft_h);
                    for (int fh = 0; fh < fft_h; ++fh)
                        col[fh] = out_ft[oc][fh][fw];
                    fft::fft_inplace(col, true);
                    for (int fh = 0; fh < fft_h; ++fh)
                        out_ft[oc][fh][fw] = col[fh];
                }

                // Row-wise IRFFT
                for (int ih = 0; ih < h; ++ih) {
                    std::vector<Complex> row(rfft_w);
                    for (int fw = 0; fw < rfft_w; ++fw)
                        row[fw] = out_ft[oc][ih][fw];
                    auto real_row = fft::irfft(row, w);
                    for (int iw = 0; iw < w; ++iw) {
                        int idx = ((b * out_channels + oc) * h + ih) * w + iw;
                        output.data[idx] = real_row[iw];
                    }
                }
            }
        }
        return output;
    }

    void zero_grad() {
        for (auto* p : parameters())
            std::fill(p->grad.data.begin(), p->grad.data.end(), 0.0);
    }

    std::vector<Parameter*> parameters() {
        return {&w1_real, &w1_imag, &w2_real, &w2_imag};
    }
};

// ============================================================
// SpectralConv3d - 3D Fourier spectral convolution
// Port of physicsnemo/nn/module/spectral_layers.py SpectralConv3d
// ============================================================
class SpectralConv3d {
public:
    int in_channels, out_channels, modes1, modes2, modes3;
    // Four weight sets for the 4 octants
    std::vector<Parameter> w_real, w_imag; // 4 each

    SpectralConv3d() : in_channels(0), out_channels(0), modes1(0), modes2(0), modes3(0) {}

    SpectralConv3d(const std::string& prefix, int in_ch, int out_ch, int m1, int m2, int m3)
        : in_channels(in_ch), out_channels(out_ch), modes1(m1), modes2(m2), modes3(m3) {
        double scale = 1.0 / (in_ch * out_ch);
        w_real.resize(4);
        w_imag.resize(4);
        for (int q = 0; q < 4; ++q) {
            Tensor wr({in_ch, out_ch, m1, m2, m3});
            Tensor wi({in_ch, out_ch, m1, m2, m3});
            std::uniform_real_distribution<double> dist(0, 1);
            for (auto& v : wr.data) v = scale * dist(global_rng());
            for (auto& v : wi.data) v = scale * dist(global_rng());
            w_real[q] = Parameter(prefix + ".w" + std::to_string(q) + "r", wr);
            w_imag[q] = Parameter(prefix + ".w" + std::to_string(q) + "i", wi);
        }
    }

    // Forward: x (batch, in_ch, d1, d2, d3) -> (batch, out_ch, d1, d2, d3)
    // Simplified 3D spectral conv using 1D FFTs along each dimension
    Tensor forward(const Tensor& x) {
        int batch = x.shape[0];
        int d1 = x.shape[2], d2 = x.shape[3], d3 = x.shape[4];

        Tensor output({batch, out_channels, d1, d2, d3}, 0.0);

        // For each batch element, perform 3D spectral convolution
        // This is computationally intensive; we use a simplified approach
        // that processes the modes directly

        for (int b = 0; b < batch; ++b) {
            // 1D FFTs along last dimension for each channel and spatial location
            int rfft3 = fft::next_pow2(d3) / 2 + 1;

            // For simplicity, we just process the 1D spectral domain along d3
            // and handle modes1/modes2 via spatial domain processing
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int i1 = 0; i1 < d1; ++i1) {
                    for (int i2 = 0; i2 < d2; ++i2) {
                        std::vector<Complex> out_spec(rfft3, {0, 0});
                        for (int ic = 0; ic < in_channels; ++ic) {
                            // Get 1D slice along d3
                            std::vector<double> slice(d3);
                            for (int i3 = 0; i3 < d3; ++i3) {
                                int idx = (((b * in_channels + ic) * d1 + i1) * d2 + i2) * d3 + i3;
                                slice[i3] = x.data[idx];
                            }
                            auto spec = fft::rfft(slice);

                            // Spectral multiply for first modes
                            for (int f = 0; f < std::min(modes3, rfft3) && f < (int)spec.size(); ++f) {
                                // Use first weight set (simplified)
                                int m1_idx = std::min(i1, modes1 - 1);
                                int m2_idx = std::min(i2, modes2 - 1);
                                if (m1_idx < 0 || m2_idx < 0) continue;
                                int widx = (((ic * out_channels + oc) * modes1 + m1_idx) * modes2 + m2_idx) * modes3 + f;
                                if (widx >= (int)w_real[0].value.data.size()) continue;
                                Complex w(w_real[0].value.data[widx], w_imag[0].value.data[widx]);
                                out_spec[f] += spec[f] * w;
                            }
                        }
                        auto real_out = fft::irfft(out_spec, d3);
                        for (int i3 = 0; i3 < d3; ++i3) {
                            int idx = (((b * out_channels + oc) * d1 + i1) * d2 + i2) * d3 + i3;
                            output.data[idx] = real_out[i3];
                        }
                    }
                }
            }
        }
        return output;
    }

    void zero_grad() {
        for (auto* p : parameters())
            std::fill(p->grad.data.begin(), p->grad.data.end(), 0.0);
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        for (int q = 0; q < 4; ++q) {
            params.push_back(&w_real[q]);
            params.push_back(&w_imag[q]);
        }
        return params;
    }
};

// ============================================================
// Conv2D layer (simple, for models that need it)
// ============================================================
class Conv2D {
public:
    int in_ch, out_ch, kh, kw, stride, padding;
    Parameter weight; // (out_ch, in_ch, kh, kw)
    Parameter bias;   // (out_ch)
    Tensor input_cache;

    Conv2D() : in_ch(0), out_ch(0), kh(0), kw(0), stride(1), padding(0) {}

    Conv2D(const std::string& prefix, int in_c, int out_c, int kernel = 3,
           int str = 1, int pad = 1)
        : in_ch(in_c), out_ch(out_c), kh(kernel), kw(kernel), stride(str), padding(pad) {
        weight = Parameter(prefix + ".w",
                           init::kaiming_uniform(out_c * in_c * kh * kw, 1).reshape({out_c, in_c * kh * kw}));
        bias = Parameter(prefix + ".b", Tensor::zeros({1, out_c}));
    }

    Tensor forward(const Tensor& x) {
        input_cache = x;
        int batch = x.shape[0];
        int h_in = x.shape[2], w_in = x.shape[3];
        int h_out = (h_in + 2 * padding - kh) / stride + 1;
        int w_out = (w_in + 2 * padding - kw) / stride + 1;

        Tensor out({batch, out_ch, h_out, w_out}, 0.0);

        for (int b = 0; b < batch; ++b) {
            for (int oc = 0; oc < out_ch; ++oc) {
                for (int oh = 0; oh < h_out; ++oh) {
                    for (int ow = 0; ow < w_out; ++ow) {
                        double val = bias.value.data[oc];
                        for (int ic = 0; ic < in_ch; ++ic) {
                            for (int fh = 0; fh < kh; ++fh) {
                                for (int fw = 0; fw < kw; ++fw) {
                                    int ih = oh * stride + fh - padding;
                                    int iw = ow * stride + fw - padding;
                                    if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                                        int x_idx = ((b * in_ch + ic) * h_in + ih) * w_in + iw;
                                        int w_idx = ((oc * in_ch + ic) * kh + fh) * kw + fw;
                                        val += x.data[x_idx] * weight.value.data[w_idx];
                                    }
                                }
                            }
                        }
                        int o_idx = ((b * out_ch + oc) * h_out + oh) * w_out + ow;
                        out.data[o_idx] = val;
                    }
                }
            }
        }
        return out;
    }

    void zero_grad() {
        std::fill(weight.grad.data.begin(), weight.grad.data.end(), 0.0);
        std::fill(bias.grad.data.begin(), bias.grad.data.end(), 0.0);
    }

    std::vector<Parameter*> parameters() { return {&weight, &bias}; }
};

// ============================================================
// TransposeConv2D (for upsampling)
// ============================================================
class TransposeConv2D {
public:
    int in_ch, out_ch, kh, kw, stride, padding;
    Parameter weight, bias;

    TransposeConv2D() : in_ch(0), out_ch(0), kh(0), kw(0), stride(2), padding(0) {}

    TransposeConv2D(const std::string& prefix, int in_c, int out_c, int kernel = 4,
                    int str = 2, int pad = 1)
        : in_ch(in_c), out_ch(out_c), kh(kernel), kw(kernel), stride(str), padding(pad) {
        weight = Parameter(prefix + ".w",
                           init::kaiming_uniform(in_c * out_c * kh * kw, 1).reshape({in_c, out_c * kh * kw}));
        bias = Parameter(prefix + ".b", Tensor::zeros({1, out_c}));
    }

    Tensor forward(const Tensor& x) {
        int batch = x.shape[0];
        int h_in = x.shape[2], w_in = x.shape[3];
        int h_out = (h_in - 1) * stride - 2 * padding + kh;
        int w_out = (w_in - 1) * stride - 2 * padding + kw;

        Tensor out({batch, out_ch, h_out, w_out}, 0.0);

        for (int b = 0; b < batch; ++b) {
            for (int oc = 0; oc < out_ch; ++oc) {
                // Add bias
                for (int oh = 0; oh < h_out; ++oh)
                    for (int ow = 0; ow < w_out; ++ow) {
                        int o_idx = ((b * out_ch + oc) * h_out + oh) * w_out + ow;
                        out.data[o_idx] = bias.value.data[oc];
                    }

                for (int ic = 0; ic < in_ch; ++ic) {
                    for (int ih = 0; ih < h_in; ++ih) {
                        for (int iw = 0; iw < w_in; ++iw) {
                            double x_val = x.data[((b * in_ch + ic) * h_in + ih) * w_in + iw];
                            for (int fh = 0; fh < kh; ++fh) {
                                for (int fw = 0; fw < kw; ++fw) {
                                    int oh = ih * stride + fh - padding;
                                    int ow = iw * stride + fw - padding;
                                    if (oh >= 0 && oh < h_out && ow >= 0 && ow < w_out) {
                                        int w_idx = ((ic * out_ch + oc) * kh + fh) * kw + fw;
                                        int o_idx = ((b * out_ch + oc) * h_out + oh) * w_out + ow;
                                        out.data[o_idx] += x_val * weight.value.data[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return out;
    }

    void zero_grad() {
        std::fill(weight.grad.data.begin(), weight.grad.data.end(), 0.0);
        std::fill(bias.grad.data.begin(), bias.grad.data.end(), 0.0);
    }

    std::vector<Parameter*> parameters() { return {&weight, &bias}; }
};

// ============================================================
// Conv1x1_2D - 2D Channel mixing (1x1 conv for 4D tensors)
// ============================================================
class Conv1x1_2D {
public:
    Parameter weight, bias;
    int in_ch, out_ch;

    Conv1x1_2D() : in_ch(0), out_ch(0) {}

    Conv1x1_2D(const std::string& prefix, int in_c, int out_c)
        : in_ch(in_c), out_ch(out_c) {
        weight = Parameter(prefix + ".w", Tensor::xavier_uniform(out_c, in_c));
        bias = Parameter(prefix + ".b", Tensor::zeros({1, out_c}));
    }

    // x: (batch, in_ch, h, w) -> (batch, out_ch, h, w)
    Tensor forward(const Tensor& x) {
        int batch = x.shape[0], h = x.shape[2], w = x.shape[3];
        Tensor out({batch, out_ch, h, w}, 0.0);
        for (int b = 0; b < batch; ++b)
            for (int oc = 0; oc < out_ch; ++oc)
                for (int ih = 0; ih < h; ++ih)
                    for (int iw = 0; iw < w; ++iw) {
                        double val = bias.value.data[oc];
                        for (int ic = 0; ic < in_ch; ++ic)
                            val += weight.value.at2(oc, ic) *
                                   x.data[((b * in_ch + ic) * h + ih) * w + iw];
                        out.data[((b * out_ch + oc) * h + ih) * w + iw] = val;
                    }
        return out;
    }

    void zero_grad() {
        std::fill(weight.grad.data.begin(), weight.grad.data.end(), 0.0);
        std::fill(bias.grad.data.begin(), bias.grad.data.end(), 0.0);
    }

    std::vector<Parameter*> parameters() { return {&weight, &bias}; }
};

// ============================================================
// FNO2D - 2D Fourier Neural Operator
// Port of physicsnemo/models/fno/fno.py with dimension=2
// ============================================================
class FNO2D {
public:
    int in_channels, out_channels, latent_channels;
    int num_fno_layers, modes1, modes2;
    bool coord_features;
    std::string act_name;

    Conv1x1_2D lift1, lift2;
    std::vector<SpectralConv2d> spconv_layers;
    std::vector<Conv1x1_2D> conv_layers;
    MLP decoder;

    FNO2D() : in_channels(0), out_channels(0), latent_channels(0),
              num_fno_layers(0), modes1(0), modes2(0), coord_features(true) {}

    FNO2D(int in_ch, int out_ch, int latent_ch = 32,
          int n_layers = 4, int m1 = 16, int m2 = 16,
          int dec_layers = 1, int dec_size = 32,
          const std::string& act = "gelu", bool coord = true)
        : in_channels(in_ch), out_channels(out_ch), latent_channels(latent_ch),
          num_fno_layers(n_layers), modes1(m1), modes2(m2),
          coord_features(coord), act_name(act) {

        int lift_in = in_ch + (coord ? 2 : 0);
        lift1 = Conv1x1_2D("fno2d.lift1", lift_in, latent_ch / 2);
        lift2 = Conv1x1_2D("fno2d.lift2", latent_ch / 2, latent_ch);

        for (int i = 0; i < n_layers; ++i) {
            spconv_layers.emplace_back("fno2d.sp" + std::to_string(i),
                                       latent_ch, latent_ch, m1, m2);
            conv_layers.emplace_back("fno2d.cv" + std::to_string(i),
                                     latent_ch, latent_ch);
        }
        decoder = MLP("fno2d.dec", latent_ch, dec_size, out_ch, dec_layers, act);
    }

    // Forward: x (batch, in_ch, h, w) -> (batch, out_ch, h, w)
    Tensor forward(const Tensor& x) {
        int batch = x.shape[0], h = x.shape[2], w = x.shape[3];
        auto [act_fn, act_gfn] = act::get_activation_extended(act_name);
        (void)act_gfn;

        Tensor inp = x;
        if (coord_features) {
            // Add 2D coordinate features
            Tensor combined({batch, in_channels + 2, h, w});
            for (int b = 0; b < batch; ++b) {
                for (int c = 0; c < in_channels; ++c)
                    for (int ih = 0; ih < h; ++ih)
                        for (int iw = 0; iw < w; ++iw)
                            combined.data[((b * (in_channels + 2) + c) * h + ih) * w + iw] =
                                x.data[((b * in_channels + c) * h + ih) * w + iw];
                for (int ih = 0; ih < h; ++ih)
                    for (int iw = 0; iw < w; ++iw) {
                        combined.data[((b * (in_channels + 2) + in_channels) * h + ih) * w + iw] =
                            (double)ih / std::max(h - 1, 1);
                        combined.data[((b * (in_channels + 2) + in_channels + 1) * h + ih) * w + iw] =
                            (double)iw / std::max(w - 1, 1);
                    }
            }
            inp = combined;
        }

        // Lift
        Tensor lifted = lift2.forward(lift1.forward(inp).apply(act_fn));

        // FNO layers
        Tensor feat = lifted;
        for (int k = 0; k < num_fno_layers; ++k) {
            Tensor sp = spconv_layers[k].forward(feat);
            Tensor cv = conv_layers[k].forward(feat);
            feat = sp + cv;
            if (k < num_fno_layers - 1) feat = feat.apply(act_fn);
        }

        // Decode pointwise
        Tensor points({batch * h * w, latent_channels});
        for (int b = 0; b < batch; ++b)
            for (int ih = 0; ih < h; ++ih)
                for (int iw = 0; iw < w; ++iw)
                    for (int c = 0; c < latent_channels; ++c)
                        points.at2(b * h * w + ih * w + iw, c) =
                            feat.data[((b * latent_channels + c) * h + ih) * w + iw];

        Tensor decoded = decoder.forward(points);

        Tensor output({batch, out_channels, h, w});
        for (int b = 0; b < batch; ++b)
            for (int ih = 0; ih < h; ++ih)
                for (int iw = 0; iw < w; ++iw)
                    for (int c = 0; c < out_channels; ++c)
                        output.data[((b * out_channels + c) * h + ih) * w + iw] =
                            decoded.at2(b * h * w + ih * w + iw, c);
        return output;
    }

    void zero_grad() {
        lift1.zero_grad(); lift2.zero_grad();
        for (auto& sp : spconv_layers) sp.zero_grad();
        for (auto& cv : conv_layers) cv.zero_grad();
        decoder.zero_grad();
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        auto add = [&](std::vector<Parameter*> ps) {
            params.insert(params.end(), ps.begin(), ps.end());
        };
        add(lift1.parameters()); add(lift2.parameters());
        for (auto& sp : spconv_layers) add(sp.parameters());
        for (auto& cv : conv_layers) add(cv.parameters());
        add(decoder.parameters());
        return params;
    }
};

// ============================================================
// FNO3D - 3D Fourier Neural Operator
// ============================================================
class FNO3D {
public:
    int in_channels, out_channels, latent_channels;
    int num_fno_layers, modes1, modes2, modes3;
    MLP decoder;
    std::vector<SpectralConv3d> spconv_layers;
    // Using Linear layers for 1x1 conv in 3D
    std::vector<Linear> conv_layers;
    Linear lift;
    std::string act_name;

    FNO3D() : in_channels(0), out_channels(0), latent_channels(0), num_fno_layers(0),
              modes1(0), modes2(0), modes3(0) {}

    FNO3D(int in_ch, int out_ch, int latent_ch = 32,
          int n_layers = 4, int m1 = 8, int m2 = 8, int m3 = 8,
          int dec_layers = 1, int dec_size = 32,
          const std::string& act = "gelu")
        : in_channels(in_ch), out_channels(out_ch), latent_channels(latent_ch),
          num_fno_layers(n_layers), modes1(m1), modes2(m2), modes3(m3), act_name(act) {

        lift = Linear("fno3d.lift", in_ch + 3, latent_ch); // +3 for coord features

        for (int i = 0; i < n_layers; ++i) {
            spconv_layers.emplace_back("fno3d.sp" + std::to_string(i),
                                       latent_ch, latent_ch, m1, m2, m3);
            conv_layers.emplace_back("fno3d.cv" + std::to_string(i),
                                     latent_ch, latent_ch);
        }
        decoder = MLP("fno3d.dec", latent_ch, dec_size, out_ch, dec_layers, act);
    }

    // Forward: x (batch, in_ch, d1, d2, d3) -> (batch, out_ch, d1, d2, d3)
    Tensor forward(const Tensor& x) {
        int batch = x.shape[0];
        int d1 = x.shape[2], d2 = x.shape[3], d3 = x.shape[4];
        int total_pts = d1 * d2 * d3;
        auto [act_fn, act_gfn] = act::get_activation_extended(act_name);
        (void)act_gfn;

        // Reshape to pointwise for lifting: (batch*d1*d2*d3, in_ch+3)
        Tensor pts({batch * total_pts, in_channels + 3});
        for (int b = 0; b < batch; ++b)
            for (int i1 = 0; i1 < d1; ++i1)
                for (int i2 = 0; i2 < d2; ++i2)
                    for (int i3 = 0; i3 < d3; ++i3) {
                        int pt_idx = b * total_pts + (i1 * d2 + i2) * d3 + i3;
                        for (int c = 0; c < in_channels; ++c)
                            pts.at2(pt_idx, c) = x.data[(((b * in_channels + c) * d1 + i1) * d2 + i2) * d3 + i3];
                        pts.at2(pt_idx, in_channels) = (double)i1 / std::max(d1 - 1, 1);
                        pts.at2(pt_idx, in_channels + 1) = (double)i2 / std::max(d2 - 1, 1);
                        pts.at2(pt_idx, in_channels + 2) = (double)i3 / std::max(d3 - 1, 1);
                    }

        // Lift
        Tensor lifted_flat = lift.forward(pts).apply(act_fn);

        // Reshape to (batch, latent_ch, d1, d2, d3)
        Tensor feat({batch, latent_channels, d1, d2, d3});
        for (int b = 0; b < batch; ++b)
            for (int i1 = 0; i1 < d1; ++i1)
                for (int i2 = 0; i2 < d2; ++i2)
                    for (int i3 = 0; i3 < d3; ++i3) {
                        int pt_idx = b * total_pts + (i1 * d2 + i2) * d3 + i3;
                        for (int c = 0; c < latent_channels; ++c)
                            feat.data[(((b * latent_channels + c) * d1 + i1) * d2 + i2) * d3 + i3] =
                                lifted_flat.at2(pt_idx, c);
                    }

        // FNO layers
        for (int k = 0; k < num_fno_layers; ++k) {
            Tensor sp = spconv_layers[k].forward(feat);
            // 1x1 conv via pointwise linear
            Tensor feat_pts({batch * total_pts, latent_channels});
            for (int b = 0; b < batch; ++b)
                for (int pt = 0; pt < total_pts; ++pt)
                    for (int c = 0; c < latent_channels; ++c) {
                        int i1 = pt / (d2 * d3), i2 = (pt / d3) % d2, i3 = pt % d3;
                        feat_pts.at2(b * total_pts + pt, c) =
                            feat.data[(((b * latent_channels + c) * d1 + i1) * d2 + i2) * d3 + i3];
                    }
            Tensor cv_flat = conv_layers[k].forward(feat_pts);
            // Reshape back and add
            Tensor cv({batch, latent_channels, d1, d2, d3});
            for (int b = 0; b < batch; ++b)
                for (int pt = 0; pt < total_pts; ++pt)
                    for (int c = 0; c < latent_channels; ++c) {
                        int i1 = pt / (d2 * d3), i2 = (pt / d3) % d2, i3 = pt % d3;
                        cv.data[(((b * latent_channels + c) * d1 + i1) * d2 + i2) * d3 + i3] =
                            cv_flat.at2(b * total_pts + pt, c);
                    }

            feat = sp + cv;
            if (k < num_fno_layers - 1) feat = feat.apply(act_fn);
        }

        // Decode pointwise
        Tensor dec_pts({batch * total_pts, latent_channels});
        for (int b = 0; b < batch; ++b)
            for (int pt = 0; pt < total_pts; ++pt)
                for (int c = 0; c < latent_channels; ++c) {
                    int i1 = pt / (d2 * d3), i2 = (pt / d3) % d2, i3 = pt % d3;
                    dec_pts.at2(b * total_pts + pt, c) =
                        feat.data[(((b * latent_channels + c) * d1 + i1) * d2 + i2) * d3 + i3];
                }

        Tensor decoded = decoder.forward(dec_pts);

        Tensor output({batch, out_channels, d1, d2, d3});
        for (int b = 0; b < batch; ++b)
            for (int pt = 0; pt < total_pts; ++pt)
                for (int c = 0; c < out_channels; ++c) {
                    int i1 = pt / (d2 * d3), i2 = (pt / d3) % d2, i3 = pt % d3;
                    output.data[(((b * out_channels + c) * d1 + i1) * d2 + i2) * d3 + i3] =
                        decoded.at2(b * total_pts + pt, c);
                }
        return output;
    }

    void zero_grad() {
        lift.zero_grad();
        for (auto& sp : spconv_layers) sp.zero_grad();
        for (auto& cv : conv_layers) cv.zero_grad();
        decoder.zero_grad();
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        auto add = [&](std::vector<Parameter*> ps) {
            params.insert(params.end(), ps.begin(), ps.end());
        };
        add(lift.parameters());
        for (auto& sp : spconv_layers) add(sp.parameters());
        for (auto& cv : conv_layers) add(cv.parameters());
        add(decoder.parameters());
        return params;
    }
};

// ============================================================
// AFNO - Adaptive Fourier Neural Operator
// Port of physicsnemo/models/afno/afno.py
// ============================================================
class AFNO {
public:
    std::vector<int> inp_shape;   // [H, W]
    int in_channels, out_channels;
    std::vector<int> patch_size;  // [pH, pW]
    int embed_dim, depth, num_blocks;
    double sparsity_threshold, hard_threshold_frac;

    // Patch embedding: linear projection
    Linear patch_proj;
    Parameter pos_embed;

    // AFNO blocks: each has norm1, filter (spectral), norm2, mlp
    struct AFNOBlock {
        LayerNorm norm1, norm2;
        // Spectral filter weights (block diagonal)
        Parameter filter_wr, filter_wi; // (num_blocks, block_size, block_size)
        // MLP
        Linear mlp1, mlp2;
        int embed_dim, num_blocks;
    };
    std::vector<AFNOBlock> blocks;

    // Output head
    Linear head;

    int h_patches, w_patches;

    AFNO() : in_channels(0), out_channels(0), embed_dim(0), depth(0), num_blocks(0),
             sparsity_threshold(0.01), hard_threshold_frac(1.0), h_patches(0), w_patches(0) {}

    AFNO(const std::vector<int>& inp_sh, int in_ch, int out_ch,
         const std::vector<int>& p_size = {16, 16},
         int emb_dim = 256, int dep = 4, double mlp_ratio = 4.0,
         int n_blocks = 16, double sparsity = 0.01, double hard_frac = 1.0)
        : inp_shape(inp_sh), in_channels(in_ch), out_channels(out_ch),
          patch_size(p_size), embed_dim(emb_dim), depth(dep), num_blocks(n_blocks),
          sparsity_threshold(sparsity), hard_threshold_frac(hard_frac) {

        h_patches = inp_shape[0] / patch_size[0];
        w_patches = inp_shape[1] / patch_size[1];
        int n_patches = h_patches * w_patches;

        // Patch embedding
        int patch_dim = in_ch * patch_size[0] * patch_size[1];
        patch_proj = Linear("afno.patch", patch_dim, embed_dim);

        // Positional embedding
        pos_embed = Parameter("afno.pos", init::truncated_normal(1, n_patches * embed_dim).reshape({n_patches, embed_dim}));

        // AFNO blocks
        int block_size = embed_dim / num_blocks;
        int mlp_hidden = (int)(embed_dim * mlp_ratio);
        blocks.resize(depth);
        for (int d = 0; d < depth; ++d) {
            auto& blk = blocks[d];
            std::string pfx = "afno.blk" + std::to_string(d);
            blk.embed_dim = embed_dim;
            blk.num_blocks = num_blocks;
            blk.norm1 = LayerNorm(pfx + ".n1", embed_dim);
            blk.norm2 = LayerNorm(pfx + ".n2", embed_dim);

            // Block-diagonal spectral weights
            double scale = 1.0 / block_size;
            Tensor wr({num_blocks, block_size, block_size});
            Tensor wi({num_blocks, block_size, block_size});
            std::uniform_real_distribution<double> dist(0, 1);
            for (auto& v : wr.data) v = scale * dist(global_rng());
            for (auto& v : wi.data) v = scale * dist(global_rng());
            blk.filter_wr = Parameter(pfx + ".fwr", wr);
            blk.filter_wi = Parameter(pfx + ".fwi", wi);

            blk.mlp1 = Linear(pfx + ".mlp1", embed_dim, mlp_hidden);
            blk.mlp2 = Linear(pfx + ".mlp2", mlp_hidden, embed_dim);
        }

        // Output head
        int out_dim = out_ch * patch_size[0] * patch_size[1];
        head = Linear("afno.head", embed_dim, out_dim);
    }

    // Forward: x (batch, in_ch, H, W) -> (batch, out_ch, H, W)
    Tensor forward(const Tensor& x) {
        int batch = x.shape[0];
        int H = x.shape[2], W = x.shape[3];
        int n_patches = h_patches * w_patches;

        // Extract patches and embed
        int patch_dim = in_channels * patch_size[0] * patch_size[1];
        Tensor patches({batch * n_patches, patch_dim});

        for (int b = 0; b < batch; ++b) {
            for (int ph = 0; ph < h_patches; ++ph) {
                for (int pw = 0; pw < w_patches; ++pw) {
                    int patch_idx = b * n_patches + ph * w_patches + pw;
                    int feat_idx = 0;
                    for (int c = 0; c < in_channels; ++c) {
                        for (int dh = 0; dh < patch_size[0]; ++dh) {
                            for (int dw = 0; dw < patch_size[1]; ++dw) {
                                int ih = ph * patch_size[0] + dh;
                                int iw = pw * patch_size[1] + dw;
                                patches.at2(patch_idx, feat_idx++) =
                                    x.data[((b * in_channels + c) * H + ih) * W + iw];
                            }
                        }
                    }
                }
            }
        }

        // Project patches
        Tensor embedded = patch_proj.forward(patches); // (batch*n_patches, embed_dim)

        // Add positional embedding
        for (int b = 0; b < batch; ++b)
            for (int p = 0; p < n_patches; ++p)
                for (int d = 0; d < embed_dim; ++d)
                    embedded.at2(b * n_patches + p, d) += pos_embed.value.at2(p, d);

        // Apply AFNO blocks
        // Reshape to (batch, n_patches, embed_dim) conceptually
        for (int d = 0; d < depth; ++d) {
            auto& blk = blocks[d];
            Tensor residual = embedded;

            // LayerNorm
            embedded = blk.norm1.forward(embedded);

            // Spectral filter (simplified: just apply block-diagonal in spatial domain)
            // In full AFNO this is FFT -> block-diagonal multiply -> IFFT
            // Here we do a simplified version: direct linear per block
            Tensor filtered(embedded.shape, 0.0);
            int block_size = embed_dim / num_blocks;
            for (int b = 0; b < batch * n_patches; ++b) {
                for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
                    int offset = blk_idx * block_size;
                    for (int i = 0; i < block_size; ++i) {
                        double val = 0;
                        for (int j = 0; j < block_size; ++j) {
                            int w_idx = (blk_idx * block_size + i) * block_size + j;
                            val += blk.filter_wr.value.data[w_idx] * embedded.at2(b, offset + j);
                        }
                        // Soft-shrink (sparsity)
                        if (std::abs(val) > sparsity_threshold) {
                            val = val > 0 ? val - sparsity_threshold : val + sparsity_threshold;
                        } else {
                            val = 0;
                        }
                        filtered.at2(b, offset + i) = val;
                    }
                }
            }

            // Double skip
            embedded = filtered + residual;
            residual = embedded;

            // LayerNorm + MLP
            embedded = blk.norm2.forward(embedded);
            Tensor h1 = blk.mlp1.forward(embedded).apply(activation::gelu);
            embedded = blk.mlp2.forward(h1);
            embedded = embedded + residual;
        }

        // Head projection
        Tensor proj = head.forward(embedded); // (batch*n_patches, out_ch*pH*pW)

        // Reassemble patches to image
        Tensor output({batch, out_channels, H, W});
        for (int b = 0; b < batch; ++b) {
            for (int ph = 0; ph < h_patches; ++ph) {
                for (int pw = 0; pw < w_patches; ++pw) {
                    int patch_idx = b * n_patches + ph * w_patches + pw;
                    int feat_idx = 0;
                    for (int c = 0; c < out_channels; ++c) {
                        for (int dh = 0; dh < patch_size[0]; ++dh) {
                            for (int dw = 0; dw < patch_size[1]; ++dw) {
                                int oh = ph * patch_size[0] + dh;
                                int ow = pw * patch_size[1] + dw;
                                output.data[((b * out_channels + c) * H + oh) * W + ow] =
                                    proj.at2(patch_idx, feat_idx++);
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

    void zero_grad() {
        patch_proj.zero_grad();
        std::fill(pos_embed.grad.data.begin(), pos_embed.grad.data.end(), 0.0);
        head.zero_grad();
        for (auto& blk : blocks) {
            blk.norm1.zero_grad(); blk.norm2.zero_grad();
            std::fill(blk.filter_wr.grad.data.begin(), blk.filter_wr.grad.data.end(), 0.0);
            std::fill(blk.filter_wi.grad.data.begin(), blk.filter_wi.grad.data.end(), 0.0);
            blk.mlp1.zero_grad(); blk.mlp2.zero_grad();
        }
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        auto add = [&](std::vector<Parameter*> ps) {
            params.insert(params.end(), ps.begin(), ps.end());
        };
        add(patch_proj.parameters());
        params.push_back(&pos_embed);
        for (auto& blk : blocks) {
            add(blk.norm1.parameters()); add(blk.norm2.parameters());
            params.push_back(&blk.filter_wr);
            params.push_back(&blk.filter_wi);
            add(blk.mlp1.parameters()); add(blk.mlp2.parameters());
        }
        add(head.parameters());
        return params;
    }
};

// ============================================================
// FullyConnected (Extended) - matches Modulus FullyConnected
// Port of physicsnemo/models/mlp/fully_connected.py
// With adaptive activations and weight norm options
// ============================================================
class FullyConnected {
public:
    MLP net;
    bool adaptive_activations;
    Parameter act_scale; // learnable scale for adaptive activations

    FullyConnected() : adaptive_activations(false) {}

    FullyConnected(int in_features, int out_features, int layer_size = 512,
                   int num_layers = 6, const std::string& activation = "silu",
                   bool skip_connections = false, bool adaptive_act = false)
        : adaptive_activations(adaptive_act) {

        net = MLP("fc", in_features, layer_size, out_features,
                  num_layers, activation, skip_connections);

        if (adaptive_act) {
            act_scale = Parameter("fc.act_scale", Tensor::ones({1, 1}));
        }
    }

    Tensor forward(const Tensor& x) {
        if (adaptive_activations) {
            // Scale input before each activation (simplified)
            return net.forward(x * act_scale.value.data[0]);
        }
        return net.forward(x);
    }

    void zero_grad() {
        net.zero_grad();
        if (adaptive_activations)
            std::fill(act_scale.grad.data.begin(), act_scale.grad.data.end(), 0.0);
    }

    std::vector<Parameter*> parameters() {
        auto params = net.parameters();
        if (adaptive_activations) params.push_back(&act_scale);
        return params;
    }
};

// ============================================================
// SIREN - Sinusoidal Representation Network
// Port of physicsnemo/nn/module/siren_layers.py
// ============================================================
class SirenLayer {
public:
    Linear linear;
    double omega_0;
    bool apply_sin;

    SirenLayer() : omega_0(30.0), apply_sin(true) {}

    SirenLayer(const std::string& prefix, int in_feat, int out_feat,
               bool is_first = false, bool is_last = false, double omega = 30.0)
        : omega_0(omega), apply_sin(!is_last) {

        linear = Linear(prefix, in_feat, out_feat);

        // Custom initialization per SIREN paper
        double range;
        if (is_first) {
            range = 1.0 / in_feat;
        } else if (is_last) {
            range = std::sqrt(6.0 / in_feat);
        } else {
            range = std::sqrt(6.0 / in_feat) / omega_0;
        }
        std::uniform_real_distribution<double> dist(-range, range);
        for (auto& v : linear.weight.value.data) v = dist(global_rng());
    }

    Tensor forward(const Tensor& x) {
        Tensor h = linear.forward(x);
        if (apply_sin) {
            h = h.apply([this](double v) { return std::sin(omega_0 * v); });
        }
        return h;
    }

    void zero_grad() { linear.zero_grad(); }
    std::vector<Parameter*> parameters() { return linear.parameters(); }
};

class SIREN {
public:
    std::vector<SirenLayer> layers;
    double omega_0;

    SIREN() : omega_0(30.0) {}

    SIREN(int in_features, int out_features, int hidden_size = 256,
          int num_layers = 5, double omega = 30.0)
        : omega_0(omega) {

        layers.emplace_back("siren.0", in_features, hidden_size, true, false, omega);
        for (int i = 1; i < num_layers; ++i)
            layers.emplace_back("siren." + std::to_string(i), hidden_size, hidden_size,
                                false, false, omega);
        layers.emplace_back("siren.final", hidden_size, out_features, false, true, omega);
    }

    Tensor forward(const Tensor& x) {
        Tensor h = x;
        for (auto& l : layers) h = l.forward(h);
        return h;
    }

    void zero_grad() { for (auto& l : layers) l.zero_grad(); }

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
// Graph structures for GNN models
// ============================================================
struct Graph {
    int num_nodes;
    int num_edges;
    std::vector<int> edge_src;  // source node indices
    std::vector<int> edge_dst;  // destination node indices
    Tensor node_features;       // (num_nodes, node_dim)
    Tensor edge_features;       // (num_edges, edge_dim)

    Graph() : num_nodes(0), num_edges(0) {}

    // Build a graph from adjacency pairs
    static Graph from_edges(int n_nodes,
                           const std::vector<std::pair<int,int>>& edges,
                           int node_dim = 0, int edge_dim = 0) {
        Graph g;
        g.num_nodes = n_nodes;
        g.num_edges = (int)edges.size();
        g.edge_src.resize(g.num_edges);
        g.edge_dst.resize(g.num_edges);
        for (int i = 0; i < g.num_edges; ++i) {
            g.edge_src[i] = edges[i].first;
            g.edge_dst[i] = edges[i].second;
        }
        if (node_dim > 0)
            g.node_features = Tensor::zeros({n_nodes, node_dim});
        if (edge_dim > 0)
            g.edge_features = Tensor::zeros({g.num_edges, edge_dim});
        return g;
    }

    // Build K-nearest neighbor graph from point cloud
    static Graph build_knn(const Tensor& points, int k) {
        int n = points.shape[0];
        int dim = points.shape[1];
        Graph g;
        g.num_nodes = n;

        for (int i = 0; i < n; ++i) {
            // Compute distances to all other points
            std::vector<std::pair<double, int>> dists;
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                double d = 0;
                for (int d_idx = 0; d_idx < dim; ++d_idx) {
                    double diff = points.at2(i, d_idx) - points.at2(j, d_idx);
                    d += diff * diff;
                }
                dists.push_back({d, j});
            }
            std::sort(dists.begin(), dists.end());

            int actual_k = std::min(k, (int)dists.size());
            for (int ki = 0; ki < actual_k; ++ki) {
                g.edge_src.push_back(i);
                g.edge_dst.push_back(dists[ki].second);
            }
        }
        g.num_edges = (int)g.edge_src.size();
        return g;
    }

    // Build radius graph
    static Graph build_radius(const Tensor& points, double radius) {
        int n = points.shape[0];
        int dim = points.shape[1];
        Graph g;
        g.num_nodes = n;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == j) continue;
                double d = 0;
                for (int d_idx = 0; d_idx < dim; ++d_idx) {
                    double diff = points.at2(i, d_idx) - points.at2(j, d_idx);
                    d += diff * diff;
                }
                if (d <= radius * radius) {
                    g.edge_src.push_back(i);
                    g.edge_dst.push_back(j);
                }
            }
        }
        g.num_edges = (int)g.edge_src.size();
        return g;
    }

    // Compute edge features from node positions (relative positions + distance)
    void compute_edge_features(const Tensor& positions) {
        int dim = positions.shape[1];
        edge_features = Tensor({num_edges, dim + 1});
        for (int e = 0; e < num_edges; ++e) {
            double dist = 0;
            for (int d = 0; d < dim; ++d) {
                double diff = positions.at2(edge_dst[e], d) - positions.at2(edge_src[e], d);
                edge_features.at2(e, d) = diff;
                dist += diff * diff;
            }
            edge_features.at2(e, dim) = std::sqrt(dist);
        }
    }
};

// ============================================================
// MeshGraphMLP - MLP with LayerNorm (used in GNN models)
// Port of physicsnemo/nn/module/gnn_layers/mesh_graph_mlp.py
// ============================================================
class MeshGraphMLP {
public:
    std::vector<Linear> layers;
    LayerNorm norm;
    std::string act_name;
    bool has_norm;

    MeshGraphMLP() : has_norm(false) {}

    MeshGraphMLP(const std::string& prefix, int in_dim, int out_dim,
                 int hidden_dim = 128, int hidden_layers = 1,
                 const std::string& act = "silu", bool use_norm = true)
        : act_name(act), has_norm(use_norm) {

        layers.emplace_back(prefix + ".l0", in_dim, hidden_dim);
        for (int i = 1; i < hidden_layers; ++i)
            layers.emplace_back(prefix + ".l" + std::to_string(i), hidden_dim, hidden_dim);
        layers.emplace_back(prefix + ".out", hidden_dim, out_dim);

        if (use_norm)
            norm = LayerNorm(prefix + ".norm", out_dim);
    }

    Tensor forward(const Tensor& x) {
        auto [act_fn, act_gfn] = act::get_activation_extended(act_name);
        (void)act_gfn;
        Tensor h = x;
        for (int i = 0; i < (int)layers.size() - 1; ++i)
            h = layers[i].forward(h).apply(act_fn);
        h = layers.back().forward(h);
        if (has_norm) h = norm.forward(h);
        return h;
    }

    void zero_grad() {
        for (auto& l : layers) l.zero_grad();
        if (has_norm) norm.zero_grad();
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        for (auto& l : layers) {
            auto lp = l.parameters();
            params.insert(params.end(), lp.begin(), lp.end());
        }
        if (has_norm) {
            auto np = norm.parameters();
            params.insert(params.end(), np.begin(), np.end());
        }
        return params;
    }
};

// ============================================================
// MeshGraphNet - Graph Neural Network for mesh-based simulation
// Port of physicsnemo/models/meshgraphnet/meshgraphnet.py
// ============================================================
class MeshGraphNet {
public:
    int input_dim_nodes, input_dim_edges, output_dim;
    int processor_size, hidden_dim;
    std::string aggregation;

    MeshGraphMLP node_encoder, edge_encoder, node_decoder;

    // Processor: alternating edge and node update blocks
    struct ProcessorBlock {
        MeshGraphMLP edge_mlp;
        MeshGraphMLP node_mlp;
    };
    std::vector<ProcessorBlock> processor_blocks;

    MeshGraphNet() : input_dim_nodes(0), input_dim_edges(0), output_dim(0),
                     processor_size(0), hidden_dim(0) {}

    MeshGraphNet(int node_dim, int edge_dim, int out_dim,
                 int proc_size = 15, int hid_dim = 128,
                 const std::string& act = "relu",
                 const std::string& agg = "sum")
        : input_dim_nodes(node_dim), input_dim_edges(edge_dim), output_dim(out_dim),
          processor_size(proc_size), hidden_dim(hid_dim), aggregation(agg) {

        node_encoder = MeshGraphMLP("mgn.ne", node_dim, hid_dim, hid_dim, 2, act);
        edge_encoder = MeshGraphMLP("mgn.ee", edge_dim, hid_dim, hid_dim, 2, act);
        node_decoder = MeshGraphMLP("mgn.nd", hid_dim, out_dim, hid_dim, 2, act, false);

        processor_blocks.resize(proc_size);
        for (int i = 0; i < proc_size; ++i) {
            std::string pfx = "mgn.p" + std::to_string(i);
            // Edge block input: sender_node + receiver_node + edge = 3*hidden
            processor_blocks[i].edge_mlp = MeshGraphMLP(pfx + ".e", 3 * hid_dim, hid_dim, hid_dim, 2, act);
            // Node block input: node + aggregated_edges = 2*hidden
            processor_blocks[i].node_mlp = MeshGraphMLP(pfx + ".n", 2 * hid_dim, hid_dim, hid_dim, 2, act);
        }
    }

    // Forward: node_features (n_nodes, node_dim), edge_features (n_edges, edge_dim), graph
    Tensor forward(const Tensor& node_feat, const Tensor& edge_feat, const Graph& graph) {
        // Encode
        Tensor nf = node_encoder.forward(node_feat);
        Tensor ef = edge_encoder.forward(edge_feat);

        // Processor - message passing
        for (int p = 0; p < processor_size; ++p) {
            // Edge update: concat(sender, receiver, edge)
            Tensor edge_input({graph.num_edges, 3 * hidden_dim});
            for (int e = 0; e < graph.num_edges; ++e) {
                int src = graph.edge_src[e];
                int dst = graph.edge_dst[e];
                for (int d = 0; d < hidden_dim; ++d) {
                    edge_input.at2(e, d) = nf.at2(src, d);
                    edge_input.at2(e, hidden_dim + d) = nf.at2(dst, d);
                    edge_input.at2(e, 2 * hidden_dim + d) = ef.at2(e, d);
                }
            }
            Tensor new_ef = processor_blocks[p].edge_mlp.forward(edge_input);
            ef = ef + new_ef; // Residual

            // Node update: aggregate incoming edges, concat with node
            Tensor agg({graph.num_nodes, hidden_dim}, 0.0);
            std::vector<int> edge_count(graph.num_nodes, 0);
            for (int e = 0; e < graph.num_edges; ++e) {
                int dst = graph.edge_dst[e];
                for (int d = 0; d < hidden_dim; ++d)
                    agg.at2(dst, d) += ef.at2(e, d);
                edge_count[dst]++;
            }
            if (aggregation == "mean") {
                for (int n = 0; n < graph.num_nodes; ++n)
                    if (edge_count[n] > 0)
                        for (int d = 0; d < hidden_dim; ++d)
                            agg.at2(n, d) /= edge_count[n];
            }

            Tensor node_input({graph.num_nodes, 2 * hidden_dim});
            for (int n = 0; n < graph.num_nodes; ++n) {
                for (int d = 0; d < hidden_dim; ++d) {
                    node_input.at2(n, d) = nf.at2(n, d);
                    node_input.at2(n, hidden_dim + d) = agg.at2(n, d);
                }
            }
            Tensor new_nf = processor_blocks[p].node_mlp.forward(node_input);
            nf = nf + new_nf; // Residual
        }

        // Decode
        return node_decoder.forward(nf);
    }

    void zero_grad() {
        node_encoder.zero_grad();
        edge_encoder.zero_grad();
        node_decoder.zero_grad();
        for (auto& pb : processor_blocks) {
            pb.edge_mlp.zero_grad();
            pb.node_mlp.zero_grad();
        }
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        auto add = [&](std::vector<Parameter*> ps) {
            params.insert(params.end(), ps.begin(), ps.end());
        };
        add(node_encoder.parameters());
        add(edge_encoder.parameters());
        add(node_decoder.parameters());
        for (auto& pb : processor_blocks) {
            add(pb.edge_mlp.parameters());
            add(pb.node_mlp.parameters());
        }
        return params;
    }
};

// ============================================================
// GraphCastNet - Graph-based weather forecasting
// Port of physicsnemo/models/graphcast/graph_cast_net.py
// Encoder-Processor-Decoder on mesh graph
// ============================================================
class GraphCastNet {
public:
    int input_dim_grid, output_dim_grid, hidden_dim;
    int processor_layers;

    // Encoder: grid2mesh
    MeshGraphMLP grid_node_encoder;
    MeshGraphMLP mesh_node_encoder;
    MeshGraphMLP grid2mesh_edge_encoder;
    MeshGraphMLP mesh_edge_encoder;

    // Processor blocks on mesh
    struct ProcBlock {
        MeshGraphMLP edge_mlp;
        MeshGraphMLP node_mlp;
    };
    std::vector<ProcBlock> proc_blocks;

    // Decoder: mesh2grid
    MeshGraphMLP mesh2grid_edge_encoder;
    MeshGraphMLP grid_node_decoder;

    GraphCastNet() : input_dim_grid(0), output_dim_grid(0), hidden_dim(0), processor_layers(0) {}

    GraphCastNet(int in_grid = 474, int out_grid = 227, int hid = 512,
                 int proc_layers = 16, const std::string& act = "silu")
        : input_dim_grid(in_grid), output_dim_grid(out_grid),
          hidden_dim(hid), processor_layers(proc_layers) {

        grid_node_encoder = MeshGraphMLP("gc.gne", in_grid, hid, hid, 1, act);
        mesh_node_encoder = MeshGraphMLP("gc.mne", 3, hid, hid, 1, act);
        grid2mesh_edge_encoder = MeshGraphMLP("gc.g2me", 4, hid, hid, 1, act);
        mesh_edge_encoder = MeshGraphMLP("gc.mee", 4, hid, hid, 1, act);

        proc_blocks.resize(proc_layers);
        for (int i = 0; i < proc_layers; ++i) {
            std::string pfx = "gc.p" + std::to_string(i);
            proc_blocks[i].edge_mlp = MeshGraphMLP(pfx + ".e", 3 * hid, hid, hid, 1, act);
            proc_blocks[i].node_mlp = MeshGraphMLP(pfx + ".n", 2 * hid, hid, hid, 1, act);
        }

        mesh2grid_edge_encoder = MeshGraphMLP("gc.m2ge", 4, hid, hid, 1, act);
        grid_node_decoder = MeshGraphMLP("gc.gnd", hid, out_grid, hid, 1, act, false);
    }

    // Forward pass (simplified: operates on pre-built graph structures)
    Tensor forward(const Tensor& grid_node_feat,  // (n_grid, in_dim)
                   const Tensor& mesh_node_feat,  // (n_mesh, 3)
                   const Graph& grid2mesh_graph,
                   const Graph& mesh_graph,
                   const Graph& mesh2grid_graph) {

        // Encode
        Tensor gf = grid_node_encoder.forward(grid_node_feat);
        Tensor mf = mesh_node_encoder.forward(mesh_node_feat);

        // Grid-to-mesh transfer
        if (grid2mesh_graph.num_edges > 0 && !grid2mesh_graph.edge_features.empty()) {
            Tensor g2m_ef = grid2mesh_edge_encoder.forward(grid2mesh_graph.edge_features);
            // Aggregate grid features to mesh nodes
            for (int e = 0; e < grid2mesh_graph.num_edges; ++e) {
                int dst = grid2mesh_graph.edge_dst[e];
                int src = grid2mesh_graph.edge_src[e];
                if (dst < mf.shape[0] && src < gf.shape[0]) {
                    for (int d = 0; d < hidden_dim; ++d)
                        mf.at2(dst, d) += gf.at2(src, d) * 0.1; // simple aggregation
                }
            }
        }

        // Mesh processor
        Tensor ef;
        if (!mesh_graph.edge_features.empty())
            ef = mesh_edge_encoder.forward(mesh_graph.edge_features);
        else
            ef = Tensor::zeros({mesh_graph.num_edges, hidden_dim});

        for (int p = 0; p < processor_layers; ++p) {
            // Edge update
            Tensor edge_in({mesh_graph.num_edges, 3 * hidden_dim});
            for (int e = 0; e < mesh_graph.num_edges; ++e) {
                int src = mesh_graph.edge_src[e], dst = mesh_graph.edge_dst[e];
                for (int d = 0; d < hidden_dim; ++d) {
                    edge_in.at2(e, d) = mf.at2(src, d);
                    edge_in.at2(e, hidden_dim + d) = mf.at2(dst, d);
                    edge_in.at2(e, 2 * hidden_dim + d) = ef.at2(e, d);
                }
            }
            Tensor new_ef = proc_blocks[p].edge_mlp.forward(edge_in);
            ef = ef + new_ef;

            // Node update
            Tensor agg({mesh_graph.num_nodes, hidden_dim}, 0.0);
            for (int e = 0; e < mesh_graph.num_edges; ++e) {
                int dst = mesh_graph.edge_dst[e];
                for (int d = 0; d < hidden_dim; ++d)
                    agg.at2(dst, d) += ef.at2(e, d);
            }
            Tensor node_in({mesh_graph.num_nodes, 2 * hidden_dim});
            for (int n = 0; n < mesh_graph.num_nodes; ++n) {
                for (int d = 0; d < hidden_dim; ++d) {
                    node_in.at2(n, d) = mf.at2(n, d);
                    node_in.at2(n, hidden_dim + d) = agg.at2(n, d);
                }
            }
            Tensor new_mf = proc_blocks[p].node_mlp.forward(node_in);
            mf = mf + new_mf;
        }

        // Mesh-to-grid transfer (simplified)
        Tensor grid_out = gf; // Start from encoded grid features
        if (mesh2grid_graph.num_edges > 0) {
            for (int e = 0; e < mesh2grid_graph.num_edges; ++e) {
                int src = mesh2grid_graph.edge_src[e]; // mesh node
                int dst = mesh2grid_graph.edge_dst[e]; // grid node
                if (src < mf.shape[0] && dst < grid_out.shape[0]) {
                    for (int d = 0; d < hidden_dim; ++d)
                        grid_out.at2(dst, d) += mf.at2(src, d) * 0.1;
                }
            }
        }

        // Decode
        return grid_node_decoder.forward(grid_out);
    }

    void zero_grad() {
        grid_node_encoder.zero_grad();
        mesh_node_encoder.zero_grad();
        grid2mesh_edge_encoder.zero_grad();
        mesh_edge_encoder.zero_grad();
        mesh2grid_edge_encoder.zero_grad();
        grid_node_decoder.zero_grad();
        for (auto& pb : proc_blocks) {
            pb.edge_mlp.zero_grad();
            pb.node_mlp.zero_grad();
        }
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        auto add = [&](std::vector<Parameter*> ps) {
            params.insert(params.end(), ps.begin(), ps.end());
        };
        add(grid_node_encoder.parameters());
        add(mesh_node_encoder.parameters());
        add(grid2mesh_edge_encoder.parameters());
        add(mesh_edge_encoder.parameters());
        for (auto& pb : proc_blocks) {
            add(pb.edge_mlp.parameters());
            add(pb.node_mlp.parameters());
        }
        add(mesh2grid_edge_encoder.parameters());
        add(grid_node_decoder.parameters());
        return params;
    }
};

// ============================================================
// Pix2Pix - Encoder-Decoder for image-to-image translation
// Port of physicsnemo/models/pix2pix/pix2pix.py
// ============================================================
class Pix2Pix {
public:
    int in_channels, out_channels, conv_layer_size;
    int n_downsampling, n_upsampling, n_blocks;
    std::string act_name;

    // Encoder (downsampling path)
    Conv2D initial_conv;
    std::vector<Conv2D> down_convs;
    // Residual blocks
    std::vector<std::pair<Conv2D, Conv2D>> resid_blocks;
    // Decoder (upsampling path)
    std::vector<TransposeConv2D> up_convs;
    Conv2D final_conv;

    Pix2Pix() : in_channels(0), out_channels(0), conv_layer_size(0),
                n_downsampling(0), n_upsampling(0), n_blocks(0) {}

    Pix2Pix(int in_ch, int out_ch, int conv_size = 64,
            int n_down = 3, int n_up = 3, int n_resid = 3,
            const std::string& act = "relu")
        : in_channels(in_ch), out_channels(out_ch), conv_layer_size(conv_size),
          n_downsampling(n_down), n_upsampling(n_up), n_blocks(n_resid), act_name(act) {

        initial_conv = Conv2D("p2p.init", in_ch, conv_size, 7, 1, 3);

        // Downsampling
        int ch = conv_size;
        for (int i = 0; i < n_down; ++i) {
            int mult = 1 << i;
            down_convs.emplace_back("p2p.d" + std::to_string(i),
                                    conv_size * mult, conv_size * mult * 2, 3, 2, 1);
            ch = conv_size * mult * 2;
        }

        // Residual blocks at bottleneck
        for (int i = 0; i < n_resid; ++i) {
            resid_blocks.push_back({
                Conv2D("p2p.r" + std::to_string(i) + "a", ch, ch, 3, 1, 1),
                Conv2D("p2p.r" + std::to_string(i) + "b", ch, ch, 3, 1, 1)
            });
        }

        // Upsampling
        for (int i = 0; i < n_up; ++i) {
            int mult = 1 << (n_down - i);
            int in_c = conv_size * mult;
            int out_c = conv_size * mult / 2;
            up_convs.emplace_back("p2p.u" + std::to_string(i), in_c, out_c, 4, 2, 1);
        }

        int final_in = conv_size;
        if (n_up < n_down) final_in = conv_size * (1 << (n_down - n_up));
        final_conv = Conv2D("p2p.final", final_in, out_ch, 7, 1, 3);
    }

    Tensor forward(const Tensor& x) {
        auto [act_fn, _] = act::get_activation_extended(act_name);

        // Initial conv
        Tensor h = initial_conv.forward(x).apply(act_fn);

        // Downsample
        for (auto& dc : down_convs)
            h = dc.forward(h).apply(act_fn);

        // Residual blocks
        for (auto& [r1, r2] : resid_blocks) {
            Tensor residual = h;
            h = r1.forward(h).apply(act_fn);
            h = r2.forward(h);
            h = h + residual;
        }

        // Upsample
        for (auto& uc : up_convs)
            h = uc.forward(h).apply(act_fn);

        // Final conv with tanh
        h = final_conv.forward(h).apply(act::tanh_fn);
        return h;
    }

    void zero_grad() {
        initial_conv.zero_grad();
        for (auto& dc : down_convs) dc.zero_grad();
        for (auto& [r1, r2] : resid_blocks) { r1.zero_grad(); r2.zero_grad(); }
        for (auto& uc : up_convs) uc.zero_grad();
        final_conv.zero_grad();
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        auto add = [&](std::vector<Parameter*> ps) {
            params.insert(params.end(), ps.begin(), ps.end());
        };
        add(initial_conv.parameters());
        for (auto& dc : down_convs) add(dc.parameters());
        for (auto& [r1, r2] : resid_blocks) { add(r1.parameters()); add(r2.parameters()); }
        for (auto& uc : up_convs) add(uc.parameters());
        add(final_conv.parameters());
        return params;
    }
};

// ============================================================
// SRResNet - Super Resolution Residual Network
// Port of physicsnemo/models/srrn/super_res_net.py
// Simplified to 2D for practical use
// ============================================================
class SRResNet {
public:
    int in_channels, out_channels, conv_layer_size;
    int n_resid_blocks, scaling_factor;

    Conv2D conv1;
    std::vector<std::pair<Conv2D, Conv2D>> resid_blocks;
    Conv2D conv2;
    std::vector<Conv2D> upscale_convs;
    Conv2D final_conv;

    SRResNet() : in_channels(0), out_channels(0), conv_layer_size(0),
                 n_resid_blocks(0), scaling_factor(1) {}

    SRResNet(int in_ch, int out_ch, int conv_size = 32,
             int n_resid = 8, int scale = 4, const std::string& act = "relu")
        : in_channels(in_ch), out_channels(out_ch), conv_layer_size(conv_size),
          n_resid_blocks(n_resid), scaling_factor(scale) {

        conv1 = Conv2D("sr.c1", in_ch, conv_size, 7, 1, 3);

        for (int i = 0; i < n_resid; ++i) {
            resid_blocks.push_back({
                Conv2D("sr.r" + std::to_string(i) + "a", conv_size, conv_size, 3, 1, 1),
                Conv2D("sr.r" + std::to_string(i) + "b", conv_size, conv_size, 3, 1, 1)
            });
        }

        conv2 = Conv2D("sr.c2", conv_size, conv_size, 3, 1, 1);

        // Upscaling via transposed convolutions
        int n_up = 0;
        int s = scale;
        while (s > 1) { n_up++; s /= 2; }
        for (int i = 0; i < n_up; ++i)
            upscale_convs.emplace_back("sr.up" + std::to_string(i), conv_size, conv_size, 3, 1, 1);

        final_conv = Conv2D("sr.final", conv_size, out_ch, 7, 1, 3);
    }

    Tensor forward(const Tensor& x) {
        Tensor h = conv1.forward(x).apply(activation::relu);
        Tensor skip = h;

        for (auto& [r1, r2] : resid_blocks) {
            Tensor res = h;
            h = r1.forward(h).apply(activation::relu);
            h = r2.forward(h);
            h = h + res;
        }

        h = conv2.forward(h);
        h = h + skip;

        // Upscale (nearest-neighbor interpolation + conv)
        for (int up_idx = 0; up_idx < (int)upscale_convs.size(); ++up_idx) {
            // 2x nearest-neighbor upsample
            int batch = h.shape[0], ch = h.shape[1], ih = h.shape[2], iw = h.shape[3];
            Tensor upsampled({batch, ch, ih * 2, iw * 2});
            for (int b = 0; b < batch; ++b)
                for (int c = 0; c < ch; ++c)
                    for (int y = 0; y < ih * 2; ++y)
                        for (int xi = 0; xi < iw * 2; ++xi)
                            upsampled.data[((b * ch + c) * ih * 2 + y) * iw * 2 + xi] =
                                h.data[((b * ch + c) * ih + y / 2) * iw + xi / 2];
            h = upscale_convs[up_idx].forward(upsampled).apply(activation::relu);
        }

        return final_conv.forward(h);
    }

    void zero_grad() {
        conv1.zero_grad(); conv2.zero_grad(); final_conv.zero_grad();
        for (auto& [r1, r2] : resid_blocks) { r1.zero_grad(); r2.zero_grad(); }
        for (auto& uc : upscale_convs) uc.zero_grad();
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        auto add = [&](std::vector<Parameter*> ps) {
            params.insert(params.end(), ps.begin(), ps.end());
        };
        add(conv1.parameters());
        for (auto& [r1, r2] : resid_blocks) { add(r1.parameters()); add(r2.parameters()); }
        add(conv2.parameters());
        for (auto& uc : upscale_convs) add(uc.parameters());
        add(final_conv.parameters());
        return params;
    }
};

// ============================================================
// GRU Cell (for RNN models)
// ============================================================
class GRUCell {
public:
    int input_dim, hidden_dim;
    Linear Wz, Wr, Wh;  // gates: update, reset, candidate
    Linear Uz, Ur, Uh;

    GRUCell() : input_dim(0), hidden_dim(0) {}

    GRUCell(const std::string& prefix, int in_dim, int hid_dim)
        : input_dim(in_dim), hidden_dim(hid_dim) {
        Wz = Linear(prefix + ".Wz", in_dim, hid_dim);
        Wr = Linear(prefix + ".Wr", in_dim, hid_dim);
        Wh = Linear(prefix + ".Wh", in_dim, hid_dim);
        Uz = Linear(prefix + ".Uz", hid_dim, hid_dim);
        Ur = Linear(prefix + ".Ur", hid_dim, hid_dim);
        Uh = Linear(prefix + ".Uh", hid_dim, hid_dim);
    }

    // Forward: x (batch, input_dim), h (batch, hidden_dim) -> h_new (batch, hidden_dim)
    Tensor forward(const Tensor& x, const Tensor& h) {
        Tensor z = (Wz.forward(x) + Uz.forward(h)).apply(act::sigmoid_fn);
        Tensor r = (Wr.forward(x) + Ur.forward(h)).apply(act::sigmoid_fn);
        Tensor rh = r * h;
        Tensor h_tilde = (Wh.forward(x) + Uh.forward(rh)).apply(act::tanh_fn);

        // h_new = (1 - z) * h + z * h_tilde
        Tensor h_new(h.shape);
        for (int i = 0; i < h.numel(); ++i)
            h_new.data[i] = (1.0 - z.data[i]) * h.data[i] + z.data[i] * h_tilde.data[i];
        return h_new;
    }

    void zero_grad() {
        Wz.zero_grad(); Wr.zero_grad(); Wh.zero_grad();
        Uz.zero_grad(); Ur.zero_grad(); Uh.zero_grad();
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        auto add = [&](std::vector<Parameter*> ps) {
            params.insert(params.end(), ps.begin(), ps.end());
        };
        add(Wz.parameters()); add(Wr.parameters()); add(Wh.parameters());
        add(Uz.parameters()); add(Ur.parameters()); add(Uh.parameters());
        return params;
    }
};

// ============================================================
// LSTM Cell
// ============================================================
class LSTMCell {
public:
    int input_dim, hidden_dim;
    Linear Wi, Wf, Wg, Wo; // input, forget, cell, output gates
    Linear Ui, Uf, Ug, Uo;

    LSTMCell() : input_dim(0), hidden_dim(0) {}

    LSTMCell(const std::string& prefix, int in_dim, int hid_dim)
        : input_dim(in_dim), hidden_dim(hid_dim) {
        Wi = Linear(prefix + ".Wi", in_dim, hid_dim);
        Wf = Linear(prefix + ".Wf", in_dim, hid_dim);
        Wg = Linear(prefix + ".Wg", in_dim, hid_dim);
        Wo = Linear(prefix + ".Wo", in_dim, hid_dim);
        Ui = Linear(prefix + ".Ui", hid_dim, hid_dim);
        Uf = Linear(prefix + ".Uf", hid_dim, hid_dim);
        Ug = Linear(prefix + ".Ug", hid_dim, hid_dim);
        Uo = Linear(prefix + ".Uo", hid_dim, hid_dim);
    }

    struct State { Tensor h, c; };

    State forward(const Tensor& x, const State& prev) {
        Tensor i = (Wi.forward(x) + Ui.forward(prev.h)).apply(act::sigmoid_fn);
        Tensor f = (Wf.forward(x) + Uf.forward(prev.h)).apply(act::sigmoid_fn);
        Tensor g = (Wg.forward(x) + Ug.forward(prev.h)).apply(act::tanh_fn);
        Tensor o = (Wo.forward(x) + Uo.forward(prev.h)).apply(act::sigmoid_fn);

        Tensor c_new = f * prev.c + i * g;
        Tensor h_new = o * c_new.apply(act::tanh_fn);
        return {h_new, c_new};
    }

    void zero_grad() {
        Wi.zero_grad(); Wf.zero_grad(); Wg.zero_grad(); Wo.zero_grad();
        Ui.zero_grad(); Uf.zero_grad(); Ug.zero_grad(); Uo.zero_grad();
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        auto add = [&](std::vector<Parameter*> ps) {
            params.insert(params.end(), ps.begin(), ps.end());
        };
        add(Wi.parameters()); add(Wf.parameters()); add(Wg.parameters()); add(Wo.parameters());
        add(Ui.parameters()); add(Uf.parameters()); add(Ug.parameters()); add(Uo.parameters());
        return params;
    }
};

// ============================================================
// One2ManyRNN - RNN for temporal prediction from single IC
// Port of physicsnemo/models/rnn/rnn_one2many.py
// ============================================================
class One2ManyRNN {
public:
    int input_channels, nr_tsteps, hidden_dim;
    // Encoder
    std::vector<Linear> encoder_layers;
    // RNN
    GRUCell rnn;
    // Decoder
    std::vector<Linear> decoder_layers;
    Linear final_proj;

    One2ManyRNN() : input_channels(0), nr_tsteps(0), hidden_dim(0) {}

    One2ManyRNN(int in_ch, int hid_dim = 128, int n_tsteps = 32,
                int n_encoder = 2, int n_decoder = 2, const std::string& act = "relu")
        : input_channels(in_ch), nr_tsteps(n_tsteps), hidden_dim(hid_dim) {

        // Encoder
        encoder_layers.emplace_back("rnn.enc0", in_ch, hid_dim);
        for (int i = 1; i < n_encoder; ++i)
            encoder_layers.emplace_back("rnn.enc" + std::to_string(i), hid_dim, hid_dim);

        // RNN cell
        rnn = GRUCell("rnn.gru", hid_dim, hid_dim);

        // Decoder
        for (int i = 0; i < n_decoder; ++i)
            decoder_layers.emplace_back("rnn.dec" + std::to_string(i), hid_dim, hid_dim);
        final_proj = Linear("rnn.proj", hid_dim, in_ch);
    }

    // Forward: x (batch, in_ch) -> (batch, in_ch, nr_tsteps)
    Tensor forward(const Tensor& x) {
        int batch = x.shape[0];

        // Encode
        Tensor h = x;
        for (auto& l : encoder_layers)
            h = l.forward(h).apply(activation::relu);

        // RNN rollout
        Tensor output({batch, input_channels, nr_tsteps});
        Tensor state = h; // initial hidden state

        for (int t = 0; t < nr_tsteps; ++t) {
            state = rnn.forward(h, state);

            // Decode
            Tensor dec = state;
            for (auto& l : decoder_layers)
                dec = l.forward(dec).apply(activation::relu);
            Tensor pred = final_proj.forward(dec);

            for (int b = 0; b < batch; ++b)
                for (int c = 0; c < input_channels; ++c)
                    output.at3(b, c, t) = pred.at2(b, c);
        }
        return output;
    }

    void zero_grad() {
        for (auto& l : encoder_layers) l.zero_grad();
        rnn.zero_grad();
        for (auto& l : decoder_layers) l.zero_grad();
        final_proj.zero_grad();
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        auto add = [&](std::vector<Parameter*> ps) {
            params.insert(params.end(), ps.begin(), ps.end());
        };
        for (auto& l : encoder_layers) add(l.parameters());
        add(rnn.parameters());
        for (auto& l : decoder_layers) add(l.parameters());
        add(final_proj.parameters());
        return params;
    }
};

// ============================================================
// UNet - Encoder-Decoder with skip connections
// ============================================================
class UNet {
public:
    int in_channels, out_channels;
    int n_levels, base_channels;

    std::vector<Conv2D> encoder_convs;
    std::vector<Conv2D> encoder_convs2;
    Conv2D bottleneck1, bottleneck2;
    std::vector<TransposeConv2D> upconvs;
    std::vector<Conv2D> decoder_convs;
    std::vector<Conv2D> decoder_convs2;
    Conv2D final_conv;

    UNet() : in_channels(0), out_channels(0), n_levels(0), base_channels(0) {}

    UNet(int in_ch, int out_ch, int levels = 4, int base_ch = 32)
        : in_channels(in_ch), out_channels(out_ch), n_levels(levels), base_channels(base_ch) {

        // Encoder path
        int ch_in = in_ch;
        for (int l = 0; l < levels; ++l) {
            int ch_out = base_ch * (1 << l);
            encoder_convs.emplace_back("unet.e" + std::to_string(l) + "a", ch_in, ch_out, 3, 1, 1);
            encoder_convs2.emplace_back("unet.e" + std::to_string(l) + "b", ch_out, ch_out, 3, 1, 1);
            ch_in = ch_out;
        }

        // Bottleneck
        int bot_ch = base_ch * (1 << levels);
        bottleneck1 = Conv2D("unet.bot1", ch_in, bot_ch, 3, 1, 1);
        bottleneck2 = Conv2D("unet.bot2", bot_ch, bot_ch, 3, 1, 1);

        // Decoder path
        int ch = bot_ch;
        for (int l = levels - 1; l >= 0; --l) {
            int skip_ch = base_ch * (1 << l);
            upconvs.emplace_back("unet.up" + std::to_string(l), ch, skip_ch, 4, 2, 1);
            decoder_convs.emplace_back("unet.d" + std::to_string(l) + "a", skip_ch * 2, skip_ch, 3, 1, 1);
            decoder_convs2.emplace_back("unet.d" + std::to_string(l) + "b", skip_ch, skip_ch, 3, 1, 1);
            ch = skip_ch;
        }

        final_conv = Conv2D("unet.final", base_ch, out_ch, 1, 1, 0);
    }

    Tensor forward(const Tensor& x) {
        // Encoder with skip connections
        std::vector<Tensor> skips;
        Tensor h = x;

        for (int l = 0; l < n_levels; ++l) {
            h = encoder_convs[l].forward(h).apply(activation::relu);
            h = encoder_convs2[l].forward(h).apply(activation::relu);
            skips.push_back(h);
            // Max-pool 2x2 (using stride-2 slicing)
            int batch = h.shape[0], ch = h.shape[1], ih = h.shape[2], iw = h.shape[3];
            Tensor pooled({batch, ch, ih / 2, iw / 2});
            for (int b = 0; b < batch; ++b)
                for (int c = 0; c < ch; ++c)
                    for (int y = 0; y < ih / 2; ++y)
                        for (int xi = 0; xi < iw / 2; ++xi) {
                            double maxv = -1e30;
                            for (int dy = 0; dy < 2; ++dy)
                                for (int dx = 0; dx < 2; ++dx)
                                    maxv = std::max(maxv, h.data[((b * ch + c) * ih + y * 2 + dy) * iw + xi * 2 + dx]);
                            pooled.data[((b * ch + c) * (ih / 2) + y) * (iw / 2) + xi] = maxv;
                        }
            h = pooled;
        }

        // Bottleneck
        h = bottleneck1.forward(h).apply(activation::relu);
        h = bottleneck2.forward(h).apply(activation::relu);

        // Decoder
        for (int l = 0; l < n_levels; ++l) {
            h = upconvs[l].forward(h).apply(activation::relu);

            // Concatenate with skip connection
            Tensor& skip = skips[n_levels - 1 - l];
            int batch = h.shape[0], ch_h = h.shape[1], ch_s = skip.shape[1];
            int oh = std::min(h.shape[2], skip.shape[2]);
            int ow = std::min(h.shape[3], skip.shape[3]);
            Tensor cat({batch, ch_h + ch_s, oh, ow});
            for (int b = 0; b < batch; ++b) {
                for (int y = 0; y < oh; ++y)
                    for (int xi = 0; xi < ow; ++xi) {
                        for (int c = 0; c < ch_h; ++c)
                            cat.data[((b * (ch_h + ch_s) + c) * oh + y) * ow + xi] =
                                h.data[((b * ch_h + c) * h.shape[2] + y) * h.shape[3] + xi];
                        for (int c = 0; c < ch_s; ++c)
                            cat.data[((b * (ch_h + ch_s) + ch_h + c) * oh + y) * ow + xi] =
                                skip.data[((b * ch_s + c) * skip.shape[2] + y) * skip.shape[3] + xi];
                    }
            }
            h = decoder_convs[l].forward(cat).apply(activation::relu);
            h = decoder_convs2[l].forward(h).apply(activation::relu);
        }

        return final_conv.forward(h);
    }

    void zero_grad() {
        for (auto& c : encoder_convs) c.zero_grad();
        for (auto& c : encoder_convs2) c.zero_grad();
        bottleneck1.zero_grad(); bottleneck2.zero_grad();
        for (auto& u : upconvs) u.zero_grad();
        for (auto& c : decoder_convs) c.zero_grad();
        for (auto& c : decoder_convs2) c.zero_grad();
        final_conv.zero_grad();
    }

    std::vector<Parameter*> parameters() {
        std::vector<Parameter*> params;
        auto add = [&](std::vector<Parameter*> ps) {
            params.insert(params.end(), ps.begin(), ps.end());
        };
        for (auto& c : encoder_convs) add(c.parameters());
        for (auto& c : encoder_convs2) add(c.parameters());
        add(bottleneck1.parameters()); add(bottleneck2.parameters());
        for (auto& u : upconvs) add(u.parameters());
        for (auto& c : decoder_convs) add(c.parameters());
        for (auto& c : decoder_convs2) add(c.parameters());
        add(final_conv.parameters());
        return params;
    }
};

} // namespace models
} // namespace modulus

#endif // MODULUS_MODELS_H
