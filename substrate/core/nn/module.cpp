#include "module.h"
#include "core/bio/bio_runtime.h"
#include <cmath>
#include <limits>

#ifdef _OPENMP
#define NN_OMP_PAR_FOR _Pragma("omp parallel for schedule(static)")
#define NN_OMP_PAR_FOR_COLLAPSE2 _Pragma("omp parallel for collapse(2) schedule(static)")
#define NN_OMP_PAR_FOR_COLLAPSE3 _Pragma("omp parallel for collapse(3) schedule(static)")
#else
#define NN_OMP_PAR_FOR
#define NN_OMP_PAR_FOR_COLLAPSE2
#define NN_OMP_PAR_FOR_COLLAPSE3
#endif

// ============================================================
// Linear
// ============================================================
Linear::Linear(size_t in_f, size_t out_f, const std::string& n, std::mt19937& rng)
    : name_(n), in_features(in_f), out_features(out_f) {
    weights = Tensor(in_f, out_f);
    weights.fill_he(in_f, rng);
    biases = Tensor(1, out_f);
    grad_weights = Tensor(in_f, out_f);
    grad_biases = Tensor(1, out_f);
}

Tensor Linear::forward(const Tensor& input) {
    Tensor effective_input = input;
    if (auto* rt = bio::active_runtime()) {
        rt->apply_linear_input_transport(&weights, input, effective_input);
    }
    cached_input = effective_input;
    Tensor out = effective_input.matmul(weights).add_row_broadcast(biases);
    if (auto* rt = bio::active_runtime()) {
        rt->record_linear_activity(&weights, effective_input, out);
    }
    return out;
}

Tensor Linear::backward(const Tensor& grad_output) {
    grad_weights = cached_input.transpose().matmul(grad_output);
    grad_biases = grad_output.sum_rows();
    return grad_output.matmul(weights.transpose());
}

// ============================================================
// ReLU
// ============================================================
Tensor ReLU::forward(const Tensor& input) {
    cached_input = input;
    return input.apply([](double x) { return x > 0 ? x : 0.0; });
}

Tensor ReLU::backward(const Tensor& grad_output) {
    Tensor mask = cached_input.apply([](double x) { return x > 0 ? 1.0 : 0.0; });
    return grad_output * mask;
}

// ============================================================
// Sigmoid
// ============================================================
Tensor Sigmoid::forward(const Tensor& input) {
    cached_output = input.apply([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
    return cached_output;
}

Tensor Sigmoid::backward(const Tensor& grad_output) {
    // sig * (1 - sig)
    Tensor one_minus = cached_output.apply([](double x) { return 1.0 - x; });
    return grad_output * cached_output * one_minus;
}

// ============================================================
// Tanh
// ============================================================
Tensor Tanh_::forward(const Tensor& input) {
    cached_output = input.apply([](double x) { return std::tanh(x); });
    return cached_output;
}

Tensor Tanh_::backward(const Tensor& grad_output) {
    Tensor deriv = cached_output.apply([](double x) { return 1.0 - x * x; });
    return grad_output * deriv;
}

// ============================================================
// GELU
// ============================================================
static constexpr double SQRT_2_PI = 0.7978845608028654; // sqrt(2/pi)

Tensor GELU::forward(const Tensor& input) {
    cached_input = input;
    return input.apply([](double x) {
        return 0.5 * x * (1.0 + std::tanh(SQRT_2_PI * (x + 0.044715 * x * x * x)));
    });
}

Tensor GELU::backward(const Tensor& grad_output) {
    Tensor grad(cached_input.rows, cached_input.cols);
    NN_OMP_PAR_FOR
    for (size_t i = 0; i < cached_input.data.size(); ++i) {
        double x = cached_input.data[i];
        double inner = SQRT_2_PI * (x + 0.044715 * x * x * x);
        double tanh_val = std::tanh(inner);
        double sech2 = 1.0 - tanh_val * tanh_val;
        double inner_deriv = SQRT_2_PI * (1.0 + 3.0 * 0.044715 * x * x);
        double gelu_deriv = 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * inner_deriv;
        grad.data[i] = grad_output.data[i] * gelu_deriv;
    }
    return grad;
}

// ============================================================
// LeakyReLU
// ============================================================
Tensor LeakyReLU::forward(const Tensor& input) {
    cached_input = input;
    double a = alpha_;
    return input.apply([a](double x) { return x > 0 ? x : a * x; });
}

Tensor LeakyReLU::backward(const Tensor& grad_output) {
    double a = alpha_;
    Tensor mask = cached_input.apply([a](double x) { return x > 0 ? 1.0 : a; });
    return grad_output * mask;
}

// ============================================================
// Softmax (row-wise)
// ============================================================
Tensor Softmax::forward(const Tensor& input) {
    Tensor result(input.rows, input.cols);
    NN_OMP_PAR_FOR
    for (size_t i = 0; i < input.rows; ++i) {
        double max_val = input(i, 0);
        for (size_t j = 1; j < input.cols; ++j)
            max_val = std::max(max_val, input(i, j));
        double sum_exp = 0;
        for (size_t j = 0; j < input.cols; ++j) {
            result(i, j) = std::exp(input(i, j) - max_val);
            sum_exp += result(i, j);
        }
        for (size_t j = 0; j < input.cols; ++j)
            result(i, j) /= sum_exp;
    }
    cached_output = result;
    return result;
}

Tensor Softmax::backward(const Tensor& grad_output) {
    // For softmax + CE combined, this is typically not called alone
    // But implement full Jacobian version for completeness
    Tensor grad_input(grad_output.rows, grad_output.cols);
    NN_OMP_PAR_FOR
    for (size_t i = 0; i < grad_output.rows; ++i) {
        for (size_t j = 0; j < grad_output.cols; ++j) {
            double sum = 0;
            for (size_t k = 0; k < grad_output.cols; ++k) {
                double delta = (j == k) ? 1.0 : 0.0;
                sum += grad_output(i, k) * cached_output(i, k) * (delta - cached_output(i, j));
            }
            grad_input(i, j) = sum;
        }
    }
    return grad_input;
}

// ============================================================
// BatchNorm1d
// ============================================================
BatchNorm1d::BatchNorm1d(size_t nf, double momentum, double eps)
    : num_features(nf), momentum_(momentum), eps_(eps) {
    gamma = Tensor::ones(1, nf);
    beta = Tensor::zeros(1, nf);
    grad_gamma = Tensor::zeros(1, nf);
    grad_beta = Tensor::zeros(1, nf);
    running_mean = Tensor::zeros(1, nf);
    running_var = Tensor::ones(1, nf);
}

Tensor BatchNorm1d::forward(const Tensor& input) {
    size_t N = input.rows;

    if (training_) {
        // Compute batch mean and variance
        Tensor mean = input.sum_rows() / (double)N;
        Tensor diff(input.rows, input.cols);
        NN_OMP_PAR_FOR_COLLAPSE2
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                diff(i, j) = input(i, j) - mean(0, j);

        Tensor var = (diff * diff).sum_rows() / (double)N;
        cached_std = (var + eps_).sqrt_t();

        // Normalize
        cached_input_norm = Tensor(N, input.cols);
        NN_OMP_PAR_FOR_COLLAPSE2
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                cached_input_norm(i, j) = diff(i, j) / cached_std(0, j);

        // Update running stats
        NN_OMP_PAR_FOR
        for (size_t j = 0; j < num_features; ++j) {
            running_mean(0, j) = (1.0 - momentum_) * running_mean(0, j) + momentum_ * mean(0, j);
            // Bessel's correction: use unbiased variance for running stats
            double unbiased_var = var(0, j);
            if (N > 1) {
                unbiased_var = var(0, j) * (double)N / (double)(N - 1);
            }
            running_var(0, j) = (1.0 - momentum_) * running_var(0, j) + momentum_ * unbiased_var;
        }
    } else {
        cached_std = (running_var + eps_).sqrt_t();
        cached_input_norm = Tensor(N, input.cols);
        NN_OMP_PAR_FOR_COLLAPSE2
        for (size_t i = 0; i < N; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                cached_input_norm(i, j) = (input(i, j) - running_mean(0, j)) / cached_std(0, j);
    }

    // Scale and shift
    Tensor output(N, input.cols);
    NN_OMP_PAR_FOR_COLLAPSE2
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            output(i, j) = gamma(0, j) * cached_input_norm(i, j) + beta(0, j);

    return output;
}

Tensor BatchNorm1d::backward(const Tensor& grad_output) {
    size_t N = grad_output.rows;
    size_t D = grad_output.cols;

    // grad_gamma = sum(grad_output * x_norm)
    grad_gamma = (grad_output * cached_input_norm).sum_rows();
    // grad_beta = sum(grad_output)
    grad_beta = grad_output.sum_rows();

    // grad_input
    Tensor dx_norm(N, D);
    NN_OMP_PAR_FOR_COLLAPSE2
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < D; ++j)
            dx_norm(i, j) = grad_output(i, j) * gamma(0, j);

    // Full BN backward
    Tensor grad_input(N, D);
    NN_OMP_PAR_FOR
    for (size_t j = 0; j < D; ++j) {
        double inv_std = 1.0 / cached_std(0, j);
        double sum_dx = 0, sum_dx_xhat = 0;
        for (size_t i = 0; i < N; ++i) {
            sum_dx += dx_norm(i, j);
            sum_dx_xhat += dx_norm(i, j) * cached_input_norm(i, j);
        }
        for (size_t i = 0; i < N; ++i) {
            grad_input(i, j) = inv_std / (double)N *
                (N * dx_norm(i, j) - sum_dx - cached_input_norm(i, j) * sum_dx_xhat);
        }
    }
    return grad_input;
}

// ============================================================
// BatchNorm2d — per-channel normalization for conv layers
// Input:  (N, C*H*W),  Output: (N, C*H*W)
// Statistics computed per channel across batch*H*W positions.
// ============================================================
BatchNorm2d::BatchNorm2d(size_t c, size_t h, size_t w, double momentum, double eps)
    : channels_(c), h_(h), w_(w), momentum_(momentum), eps_(eps) {
    gamma = Tensor::ones(1, c);
    beta = Tensor::zeros(1, c);
    grad_gamma = Tensor::zeros(1, c);
    grad_beta = Tensor::zeros(1, c);
    running_mean = Tensor::zeros(1, c);
    running_var = Tensor::ones(1, c);
}

Tensor BatchNorm2d::forward(const Tensor& input) {
    size_t N = input.rows;
    size_t spatial = h_ * w_;
    size_t C = channels_;
    size_t M = N * spatial; // total elements per channel

    cached_input_norm = Tensor(N, C * spatial);

    if (training_) {
        Tensor mean(1, C);
        Tensor var(1, C);

        // Compute per-channel mean
        NN_OMP_PAR_FOR
        for (size_t c = 0; c < C; ++c) {
            double sum = 0;
            for (size_t n = 0; n < N; ++n) {
                size_t offset = c * spatial;
                for (size_t s = 0; s < spatial; ++s)
                    sum += input(n, offset + s);
            }
            mean(0, c) = sum / (double)M;
        }

        // Compute per-channel variance
        NN_OMP_PAR_FOR
        for (size_t c = 0; c < C; ++c) {
            double sum = 0;
            for (size_t n = 0; n < N; ++n) {
                size_t offset = c * spatial;
                for (size_t s = 0; s < spatial; ++s) {
                    double d = input(n, offset + s) - mean(0, c);
                    sum += d * d;
                }
            }
            var(0, c) = sum / (double)M;
        }

        cached_std = Tensor(1, C);
        NN_OMP_PAR_FOR
        for (size_t c = 0; c < C; ++c)
            cached_std(0, c) = std::sqrt(var(0, c) + eps_);

        // Normalize
        NN_OMP_PAR_FOR_COLLAPSE2
        for (size_t n = 0; n < N; ++n) {
            for (size_t c = 0; c < C; ++c) {
                size_t offset = c * spatial;
                for (size_t s = 0; s < spatial; ++s)
                    cached_input_norm(n, offset + s) =
                        (input(n, offset + s) - mean(0, c)) / cached_std(0, c);
            }
        }

        // Update running stats with Bessel's correction
        NN_OMP_PAR_FOR
        for (size_t c = 0; c < C; ++c) {
            running_mean(0, c) = (1.0 - momentum_) * running_mean(0, c) + momentum_ * mean(0, c);
            double unbiased_var = var(0, c);
            if (M > 1) {
                unbiased_var = var(0, c) * (double)M / (double)(M - 1);
            }
            running_var(0, c) = (1.0 - momentum_) * running_var(0, c) + momentum_ * unbiased_var;
        }
    } else {
        cached_std = Tensor(1, C);
        NN_OMP_PAR_FOR
        for (size_t c = 0; c < C; ++c)
            cached_std(0, c) = std::sqrt(running_var(0, c) + eps_);

        NN_OMP_PAR_FOR_COLLAPSE2
        for (size_t n = 0; n < N; ++n) {
            for (size_t c = 0; c < C; ++c) {
                size_t offset = c * spatial;
                for (size_t s = 0; s < spatial; ++s)
                    cached_input_norm(n, offset + s) =
                        (input(n, offset + s) - running_mean(0, c)) / cached_std(0, c);
            }
        }
    }

    // Scale and shift per channel
    Tensor output(N, C * spatial);
    NN_OMP_PAR_FOR_COLLAPSE2
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            size_t offset = c * spatial;
            for (size_t s = 0; s < spatial; ++s)
                output(n, offset + s) = gamma(0, c) * cached_input_norm(n, offset + s) + beta(0, c);
        }
    }
    return output;
}

Tensor BatchNorm2d::backward(const Tensor& grad_output) {
    size_t N = grad_output.rows;
    size_t spatial = h_ * w_;
    size_t C = channels_;
    size_t M = N * spatial;

    // grad_gamma[c] = sum over n,s of grad_output[n, c*S+s] * x_norm[n, c*S+s]
    // grad_beta[c]  = sum over n,s of grad_output[n, c*S+s]
    grad_gamma = Tensor::zeros(1, C);
    grad_beta = Tensor::zeros(1, C);
    NN_OMP_PAR_FOR
    for (size_t c = 0; c < C; ++c) {
        for (size_t n = 0; n < N; ++n) {
            size_t offset = c * spatial;
            for (size_t s = 0; s < spatial; ++s) {
                grad_gamma(0, c) += grad_output(n, offset + s) * cached_input_norm(n, offset + s);
                grad_beta(0, c) += grad_output(n, offset + s);
            }
        }
    }

    // Backward through normalization (full BN backward per channel)
    Tensor grad_input(N, C * spatial);
    NN_OMP_PAR_FOR
    for (size_t c = 0; c < C; ++c) {
        double inv_std = 1.0 / cached_std(0, c);
        double sum_dx = 0, sum_dx_xhat = 0;
        for (size_t n = 0; n < N; ++n) {
            size_t offset = c * spatial;
            for (size_t s = 0; s < spatial; ++s) {
                double dx = grad_output(n, offset + s) * gamma(0, c);
                sum_dx += dx;
                sum_dx_xhat += dx * cached_input_norm(n, offset + s);
            }
        }
        for (size_t n = 0; n < N; ++n) {
            size_t offset = c * spatial;
            for (size_t s = 0; s < spatial; ++s) {
                double dx = grad_output(n, offset + s) * gamma(0, c);
                grad_input(n, offset + s) = inv_std / (double)M *
                    (M * dx - sum_dx - cached_input_norm(n, offset + s) * sum_dx_xhat);
            }
        }
    }
    return grad_input;
}

// ============================================================
// LayerNorm
// ============================================================
LayerNorm::LayerNorm(size_t ns, double eps) : normalized_size(ns), eps_(eps) {
    gamma = Tensor::ones(1, ns);
    beta = Tensor::zeros(1, ns);
    grad_gamma = Tensor::zeros(1, ns);
    grad_beta = Tensor::zeros(1, ns);
}

Tensor LayerNorm::forward(const Tensor& input) {
    cached_input = input;
    size_t N = input.rows;
    size_t D = input.cols;

    cached_input_norm = Tensor(N, D);
    cached_std = Tensor(N, 1);

    NN_OMP_PAR_FOR
    for (size_t i = 0; i < N; ++i) {
        double mean = 0;
        for (size_t j = 0; j < D; ++j) mean += input(i, j);
        mean /= D;

        double var = 0;
        for (size_t j = 0; j < D; ++j) {
            double d = input(i, j) - mean;
            var += d * d;
        }
        var /= D;
        cached_std(i, 0) = std::sqrt(var + eps_);

        for (size_t j = 0; j < D; ++j)
            cached_input_norm(i, j) = (input(i, j) - mean) / cached_std(i, 0);
    }

    Tensor output(N, D);
    NN_OMP_PAR_FOR_COLLAPSE2
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < D; ++j)
            output(i, j) = gamma(0, j) * cached_input_norm(i, j) + beta(0, j);
    return output;
}

Tensor LayerNorm::backward(const Tensor& grad_output) {
    size_t N = grad_output.rows;
    size_t D = grad_output.cols;

    grad_gamma = (grad_output * cached_input_norm).sum_rows();
    grad_beta = grad_output.sum_rows();

    Tensor dx_norm(N, D);
    NN_OMP_PAR_FOR_COLLAPSE2
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < D; ++j)
            dx_norm(i, j) = grad_output(i, j) * gamma(0, j);

    Tensor grad_input(N, D);
    NN_OMP_PAR_FOR
    for (size_t i = 0; i < N; ++i) {
        double inv_std = 1.0 / cached_std(i, 0);
        double sum_dx = 0, sum_dx_xhat = 0;
        for (size_t j = 0; j < D; ++j) {
            sum_dx += dx_norm(i, j);
            sum_dx_xhat += dx_norm(i, j) * cached_input_norm(i, j);
        }
        for (size_t j = 0; j < D; ++j) {
            grad_input(i, j) = inv_std / (double)D *
                (D * dx_norm(i, j) - sum_dx - cached_input_norm(i, j) * sum_dx_xhat);
        }
    }
    return grad_input;
}

// ============================================================
// Dropout
// ============================================================
Tensor Dropout::forward(const Tensor& input) {
    if (!training_ || rate_ == 0) {
        applied_mask_ = false;
        return input;
    }
    applied_mask_ = true;
    mask_ = Tensor(input.rows, input.cols);
    std::uniform_real_distribution<double> dist(0, 1);
    double scale = 1.0 / (1.0 - rate_);
    for (size_t i = 0; i < mask_.data.size(); ++i)
        mask_.data[i] = (dist(*rng_) > rate_) ? scale : 0.0;
    return input * mask_;
}

Tensor Dropout::backward(const Tensor& grad_output) {
    // Use the mask state from forward, not current training_ flag
    if (!applied_mask_) return grad_output;
    return grad_output * mask_;
}

// ============================================================
// Embedding
// ============================================================
Embedding::Embedding(size_t vs, size_t ed, const std::string& n, std::mt19937& rng)
    : vocab_size(vs), embed_dim(ed), name_(n) {
    weights = Tensor(vs, ed);
    weights.fill_random_normal(0, 0.1, rng);
    grad_weights = Tensor::zeros(vs, ed);
}

Tensor Embedding::forward(const Tensor& input) {
    // input: batch x seq_len with token indices as doubles
    size_t total = input.rows * input.cols;
    cached_indices.resize(total);
    Tensor output(total, embed_dim);
    NN_OMP_PAR_FOR
    for (size_t i = 0; i < total; ++i) {
        size_t idx = (size_t)input.data[i];
        TENSOR_CHECK(idx < vocab_size, "Embedding index out of range");
        cached_indices[i] = idx;
        for (size_t j = 0; j < embed_dim; ++j)
            output(i, j) = weights(idx, j);
    }
    return output;
}

Tensor Embedding::backward(const Tensor& grad_output) {
    grad_weights.fill_zeros();
    for (size_t i = 0; i < cached_indices.size(); ++i) {
        size_t idx = cached_indices[i];
        for (size_t j = 0; j < embed_dim; ++j)
            grad_weights(idx, j) += grad_output(i, j);
    }
    return Tensor(); // no gradient to propagate further
}

// ============================================================
// Conv2D
// ============================================================
Conv2D::Conv2D(size_t ic, size_t oc, size_t ks, size_t ih, size_t iw,
               size_t s, size_t p, const std::string& n, std::mt19937& rng)
    : in_channels(ic), out_channels(oc), kernel_size(ks),
      stride(s), padding(p), in_h(ih), in_w(iw), name_(n) {
    size_t fan_in = ic * ks * ks;
    weights = Tensor(oc, fan_in);
    weights.fill_he(fan_in, rng);
    biases = Tensor(1, oc);
    grad_weights = Tensor::zeros(oc, fan_in);
    grad_biases = Tensor::zeros(1, oc);
}

Tensor Conv2D::im2col(const Tensor& input, size_t batch_size) const {
    size_t oh = out_h(), ow = out_w();
    size_t col_rows = batch_size * oh * ow;
    size_t col_cols = in_channels * kernel_size * kernel_size;
    Tensor col(col_rows, col_cols);

    NN_OMP_PAR_FOR_COLLAPSE3
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < oh; ++i) {
            for (size_t j = 0; j < ow; ++j) {
                size_t row_idx = b * oh * ow + i * ow + j;
                size_t col_idx = 0;
                for (size_t c = 0; c < in_channels; ++c) {
                    for (size_t ki = 0; ki < kernel_size; ++ki) {
                        for (size_t kj = 0; kj < kernel_size; ++kj) {
                            int hi = (int)(i * stride + ki) - (int)padding;
                            int wj = (int)(j * stride + kj) - (int)padding;
                            if (hi >= 0 && hi < (int)in_h && wj >= 0 && wj < (int)in_w) {
                                col(row_idx, col_idx) = input(b, c * in_h * in_w + hi * in_w + wj);
                            }
                            col_idx++;
                        }
                    }
                }
            }
        }
    }
    return col;
}

Tensor Conv2D::col2im(const Tensor& col, size_t batch_size) const {
    size_t oh = out_h(), ow = out_w();
    Tensor img(batch_size, in_channels * in_h * in_w);

    NN_OMP_PAR_FOR
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < oh; ++i) {
            for (size_t j = 0; j < ow; ++j) {
                size_t row_idx = b * oh * ow + i * ow + j;
                size_t col_idx = 0;
                for (size_t c = 0; c < in_channels; ++c) {
                    for (size_t ki = 0; ki < kernel_size; ++ki) {
                        for (size_t kj = 0; kj < kernel_size; ++kj) {
                            int hi = (int)(i * stride + ki) - (int)padding;
                            int wj = (int)(j * stride + kj) - (int)padding;
                            if (hi >= 0 && hi < (int)in_h && wj >= 0 && wj < (int)in_w) {
                                img(b, c * in_h * in_w + hi * in_w + wj) += col(row_idx, col_idx);
                            }
                            col_idx++;
                        }
                    }
                }
            }
        }
    }
    return img;
}

Tensor Conv2D::forward(const Tensor& input) {
    cached_batch = input.rows;
    cached_col = im2col(input, cached_batch);
    Tensor effective_col = cached_col;
    if (auto* rt = bio::active_runtime()) {
        rt->apply_linear_input_transport(&weights, cached_col, effective_col);
    }
    cached_col = effective_col;
    // col: (B*OH*OW) x (C*K*K), weights: OC x (C*K*K)
    Tensor output = effective_col.matmul(weights.transpose()); // (B*OH*OW) x OC
    if (auto* rt = bio::active_runtime()) {
        rt->record_linear_activity(&weights, effective_col, output);
    }
    // Add bias
    NN_OMP_PAR_FOR_COLLAPSE2
    for (size_t i = 0; i < output.rows; ++i)
        for (size_t j = 0; j < output.cols; ++j)
            output(i, j) += biases(0, j);
    // Reshape to batch x (OC*OH*OW)
    size_t oh = out_h(), ow = out_w();
    Tensor result(cached_batch, out_channels * oh * ow);
    NN_OMP_PAR_FOR_COLLAPSE3
    for (size_t b = 0; b < cached_batch; ++b)
        for (size_t i = 0; i < oh * ow; ++i)
            for (size_t c = 0; c < out_channels; ++c)
                result(b, c * oh * ow + i) = output(b * oh * ow + i, c);
    return result;
}

Tensor Conv2D::backward(const Tensor& grad_output) {
    size_t oh = out_h(), ow = out_w();
    // Reshape grad from batch x (OC*OH*OW) to (B*OH*OW) x OC
    Tensor grad_col(cached_batch * oh * ow, out_channels);
    NN_OMP_PAR_FOR_COLLAPSE3
    for (size_t b = 0; b < cached_batch; ++b)
        for (size_t i = 0; i < oh * ow; ++i)
            for (size_t c = 0; c < out_channels; ++c)
                grad_col(b * oh * ow + i, c) = grad_output(b, c * oh * ow + i);

    // grad_weights = grad_col^T * cached_col
    grad_weights = grad_col.transpose().matmul(cached_col);
    // grad_biases
    grad_biases = grad_col.sum_rows();

    // grad_input via col2im
    Tensor d_col = grad_col.matmul(weights); // (B*OH*OW) x (C*K*K)
    return col2im(d_col, cached_batch);
}

// ============================================================
// MaxPool2D
// ============================================================
MaxPool2D::MaxPool2D(size_t ps, size_t s, size_t c, size_t ih, size_t iw)
    : pool_size(ps), stride(s), channels(c), in_h(ih), in_w(iw) {}

Tensor MaxPool2D::forward(const Tensor& input) {
    cached_input = input;
    size_t batch = input.rows;
    size_t oh = out_h(), ow = out_w();
    Tensor output(batch, channels * oh * ow);
    cached_max_indices = Tensor(batch, channels * oh * ow);

    NN_OMP_PAR_FOR_COLLAPSE2
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t i = 0; i < oh; ++i) {
                for (size_t j = 0; j < ow; ++j) {
                    double max_val = -std::numeric_limits<double>::infinity();
                    size_t max_idx = 0;
                    for (size_t pi = 0; pi < pool_size; ++pi) {
                        for (size_t pj = 0; pj < pool_size; ++pj) {
                            size_t hi = i * stride + pi;
                            size_t wj = j * stride + pj;
                            size_t idx = c * in_h * in_w + hi * in_w + wj;
                            double val = input(b, idx);
                            if (val > max_val) {
                                max_val = val;
                                max_idx = idx;
                            }
                        }
                    }
                    size_t out_idx = c * oh * ow + i * ow + j;
                    output(b, out_idx) = max_val;
                    cached_max_indices(b, out_idx) = (double)max_idx;
                }
            }
        }
    }
    return output;
}

Tensor MaxPool2D::backward(const Tensor& grad_output) {
    size_t batch = grad_output.rows;
    Tensor grad_input(batch, channels * in_h * in_w);
    size_t oh = out_h(), ow = out_w();

    NN_OMP_PAR_FOR
    for (size_t b = 0; b < batch; ++b) {
        for (size_t idx = 0; idx < channels * oh * ow; ++idx) {
            size_t max_idx = (size_t)cached_max_indices(b, idx);
            grad_input(b, max_idx) += grad_output(b, idx);
        }
    }
    return grad_input;
}

// ============================================================
// AvgPool2D
// ============================================================
AvgPool2D::AvgPool2D(size_t ps, size_t s, size_t c, size_t ih, size_t iw)
    : pool_size(ps), stride(s), channels(c), in_h(ih), in_w(iw) {}

Tensor AvgPool2D::forward(const Tensor& input) {
    size_t batch = input.rows;
    size_t oh = out_h(), ow = out_w();
    Tensor output(batch, channels * oh * ow);
    double pool_area = (double)(pool_size * pool_size);

    NN_OMP_PAR_FOR_COLLAPSE2
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t i = 0; i < oh; ++i) {
                for (size_t j = 0; j < ow; ++j) {
                    double sum = 0;
                    for (size_t pi = 0; pi < pool_size; ++pi)
                        for (size_t pj = 0; pj < pool_size; ++pj)
                            sum += input(b, c * in_h * in_w + (i * stride + pi) * in_w + (j * stride + pj));
                    output(b, c * oh * ow + i * ow + j) = sum / pool_area;
                }
            }
        }
    }
    return output;
}

Tensor AvgPool2D::backward(const Tensor& grad_output) {
    size_t batch = grad_output.rows;
    size_t oh = out_h(), ow = out_w();
    Tensor grad_input(batch, channels * in_h * in_w);
    double pool_area = (double)(pool_size * pool_size);

    NN_OMP_PAR_FOR
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t i = 0; i < oh; ++i) {
                for (size_t j = 0; j < ow; ++j) {
                    double g = grad_output(b, c * oh * ow + i * ow + j) / pool_area;
                    for (size_t pi = 0; pi < pool_size; ++pi)
                        for (size_t pj = 0; pj < pool_size; ++pj)
                            grad_input(b, c * in_h * in_w + (i * stride + pi) * in_w + (j * stride + pj)) += g;
                }
            }
        }
    }
    return grad_input;
}

// ============================================================
// Flatten
// ============================================================
Tensor Flatten::forward(const Tensor& input) {
    cached_rows = input.rows;
    cached_cols = input.cols;
    return input; // already batch x features in our representation
}

Tensor Flatten::backward(const Tensor& grad_output) {
    return grad_output; // already correct shape
}

// ============================================================
// Sequential
// ============================================================
void Sequential::add(std::unique_ptr<Module> layer) {
    layers.push_back(std::move(layer));
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor x = input;
    for (auto& layer : layers) x = layer->forward(x);
    return x;
}

Tensor Sequential::backward(const Tensor& grad_output) {
    Tensor grad = grad_output;
    for (int i = (int)layers.size() - 1; i >= 0; --i)
        grad = layers[i]->backward(grad);
    return grad;
}

std::vector<Tensor*> Sequential::parameters() {
    std::vector<Tensor*> params;
    for (auto& layer : layers)
        for (auto* p : layer->parameters()) params.push_back(p);
    return params;
}

std::vector<Tensor*> Sequential::gradients() {
    std::vector<Tensor*> grads;
    for (auto& layer : layers)
        for (auto* g : layer->gradients()) grads.push_back(g);
    return grads;
}

std::vector<const Tensor*> Sequential::parameters() const {
    std::vector<const Tensor*> params;
    for (const auto& layer : layers)
        for (const auto* p : layer->parameters()) params.push_back(p);
    return params;
}

std::vector<const Tensor*> Sequential::gradients() const {
    std::vector<const Tensor*> grads;
    for (const auto& layer : layers)
        for (const auto* g : layer->gradients()) grads.push_back(g);
    return grads;
}

bool Sequential::has_parameters() const {
    for (auto& layer : layers)
        if (layer->has_parameters()) return true;
    return false;
}

void Sequential::train() {
    training_ = true;
    for (auto& l : layers) l->train();
}

void Sequential::eval() {
    training_ = false;
    for (auto& l : layers) l->eval();
}
