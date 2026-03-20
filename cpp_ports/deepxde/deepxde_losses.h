// deepxde_losses.h - Loss functions, metrics, and optimizers for DeepXDE C++ port
// Ports: losses.py, metrics.py, optimizers/, gradients/
#ifndef DEEPXDE_LOSSES_H
#define DEEPXDE_LOSSES_H

#include "deepxde.h"
#include <cmath>
#include <numeric>
#include <limits>

namespace deepxde {

// ============================================================================
// Loss functions
// Ports deepxde.losses
// ============================================================================
namespace losses {

inline double mean_squared_error(const std::vector<double>& y_true,
                                  const std::vector<double>& y_pred) {
    double mse = 0;
    int n = static_cast<int>(y_true.size());
    for (int i = 0; i < n; ++i) {
        double d = y_true[i] - y_pred[i];
        mse += d * d;
    }
    return mse / std::max(n, 1);
}

inline double mean_absolute_error(const std::vector<double>& y_true,
                                   const std::vector<double>& y_pred) {
    double mae = 0;
    int n = static_cast<int>(y_true.size());
    for (int i = 0; i < n; ++i) mae += std::fabs(y_true[i] - y_pred[i]);
    return mae / std::max(n, 1);
}

inline double mean_l2_relative_error(const std::vector<double>& y_true,
                                      const std::vector<double>& y_pred) {
    double err_norm = 0, true_norm = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double d = y_true[i] - y_pred[i];
        err_norm += d * d;
        true_norm += y_true[i] * y_true[i];
    }
    if (true_norm < 1e-30) return 0;
    return std::sqrt(err_norm) / std::sqrt(true_norm);
}

inline double mean_absolute_percentage_error(const std::vector<double>& y_true,
                                              const std::vector<double>& y_pred) {
    double mape = 0;
    int n = static_cast<int>(y_true.size());
    for (int i = 0; i < n; ++i) {
        double denom = std::max(std::fabs(y_true[i]), 1e-15);
        mape += std::fabs(y_true[i] - y_pred[i]) / denom;
    }
    return 100.0 * mape / std::max(n, 1);
}

inline double softmax_cross_entropy(const std::vector<double>& y_true,
                                     const std::vector<double>& y_pred) {
    // logits -> log softmax -> cross entropy
    int n = static_cast<int>(y_pred.size());
    double max_val = *std::max_element(y_pred.begin(), y_pred.end());
    double sum_exp = 0;
    for (int i = 0; i < n; ++i) sum_exp += std::exp(y_pred[i] - max_val);
    double log_sum = max_val + std::log(sum_exp);
    double ce = 0;
    for (int i = 0; i < n; ++i) ce -= y_true[i] * (y_pred[i] - log_sum);
    return ce;
}

inline double zero_loss(const std::vector<double>&, const std::vector<double>&) {
    return 0.0;
}

// MSE on AD variables (returns VarPtr for gradient computation)
inline VarPtr mse_ad(const std::vector<VarPtr>& errors) {
    VarPtr loss = make_var(0.0);
    for (auto& e : errors) loss = loss + ad::square(e);
    return loss / static_cast<double>(errors.size());
}

inline VarPtr mae_ad(const std::vector<VarPtr>& errors) {
    VarPtr loss = make_var(0.0);
    for (auto& e : errors) loss = loss + ad::abs_val(e);
    return loss / static_cast<double>(errors.size());
}

// Loss function type enum for convenience
enum class LossType { MSE, MAE, MAPE, L2Relative, SoftmaxCE, Zero };

using LossFn = std::function<double(const std::vector<double>&, const std::vector<double>&)>;

inline LossFn get_loss(const std::string& name) {
    if (name == "MSE" || name == "mse" || name == "mean squared error")
        return mean_squared_error;
    if (name == "MAE" || name == "mae" || name == "mean absolute error")
        return mean_absolute_error;
    if (name == "MAPE" || name == "mape")
        return mean_absolute_percentage_error;
    if (name == "mean l2 relative error")
        return mean_l2_relative_error;
    if (name == "softmax cross entropy")
        return softmax_cross_entropy;
    if (name == "zero")
        return zero_loss;
    throw std::invalid_argument("Unknown loss: " + name);
}

} // namespace losses

// ============================================================================
// Metrics
// Ports deepxde.metrics
// ============================================================================
namespace metrics {

inline double l2_relative_error(const Matrix& y_true, const Matrix& y_pred) {
    double err = 0, norm = 0;
    for (int i = 0; i < y_true.rows; ++i)
        for (int j = 0; j < y_true.cols; ++j) {
            double d = y_true(i, j) - y_pred(i, j);
            err += d * d;
            norm += y_true(i, j) * y_true(i, j);
        }
    return std::sqrt(err) / std::max(std::sqrt(norm), 1e-30);
}

inline double mean_squared_error(const Matrix& y_true, const Matrix& y_pred) {
    double mse = 0;
    int n = y_true.rows * y_true.cols;
    for (int i = 0; i < y_true.rows; ++i)
        for (int j = 0; j < y_true.cols; ++j) {
            double d = y_true(i, j) - y_pred(i, j);
            mse += d * d;
        }
    return mse / std::max(n, 1);
}

inline double mean_absolute_percentage_error(const Matrix& y_true, const Matrix& y_pred) {
    double mape = 0;
    int n = y_true.rows * y_true.cols;
    for (int i = 0; i < y_true.rows; ++i)
        for (int j = 0; j < y_true.cols; ++j) {
            double denom = std::max(std::fabs(y_true(i, j)), 1e-15);
            mape += std::fabs(y_true(i, j) - y_pred(i, j)) / denom;
        }
    return 100.0 * mape / std::max(n, 1);
}

inline double accuracy(const Matrix& y_true, const Matrix& y_pred) {
    int correct = 0;
    for (int i = 0; i < y_true.rows; ++i) {
        int true_class = 0, pred_class = 0;
        for (int j = 1; j < y_true.cols; ++j) {
            if (y_true(i, j) > y_true(i, true_class)) true_class = j;
            if (y_pred(i, j) > y_pred(i, pred_class)) pred_class = j;
        }
        if (true_class == pred_class) correct++;
    }
    return static_cast<double>(correct) / y_true.rows;
}

} // namespace metrics

// ============================================================================
// Optimizers
// Ports deepxde.optimizers (SGD, RMSProp, L-BFGS, learning rate schedulers)
// ============================================================================

// SGD with momentum
class SGD {
public:
    double lr;
    double momentum;
    std::vector<double> velocity;

    SGD(double learning_rate = 0.01, double mom = 0.0)
        : lr(learning_rate), momentum(mom) {}

    void init(int n_params) { velocity.assign(n_params, 0.0); }

    void step(std::vector<double*>& params, const std::vector<double>& grads) {
        if (velocity.empty()) init(static_cast<int>(params.size()));
        for (size_t i = 0; i < params.size(); ++i) {
            velocity[i] = momentum * velocity[i] - lr * grads[i];
            *params[i] += velocity[i];
        }
    }
};

// RMSProp
class RMSProp {
public:
    double lr, alpha, eps;
    std::vector<double> sq_avg;

    RMSProp(double learning_rate = 0.001, double a = 0.99, double e = 1e-8)
        : lr(learning_rate), alpha(a), eps(e) {}

    void init(int n_params) { sq_avg.assign(n_params, 0.0); }

    void step(std::vector<double*>& params, const std::vector<double>& grads) {
        if (sq_avg.empty()) init(static_cast<int>(params.size()));
        for (size_t i = 0; i < params.size(); ++i) {
            sq_avg[i] = alpha * sq_avg[i] + (1.0 - alpha) * grads[i] * grads[i];
            *params[i] -= lr * grads[i] / (std::sqrt(sq_avg[i]) + eps);
        }
    }
};

// L-BFGS optimizer (two-loop recursion)
class LBFGS {
public:
    int m;  // History size
    double lr;
    int max_iter;
    double ftol, gtol;

    struct State {
        std::vector<std::vector<double>> s_history, y_history;
        std::vector<double> rho_history;
        std::vector<double> prev_params, prev_grad;
        bool initialized = false;
    };
    State state;

    LBFGS(int history_size = 10, double learning_rate = 1.0,
          int max_iterations = 20, double ft = 1e-10, double gt = 1e-8)
        : m(history_size), lr(learning_rate), max_iter(max_iterations),
          ftol(ft), gtol(gt) {}

    void step(std::vector<double*>& params, const std::vector<double>& grads,
              std::function<double()> closure = nullptr) {
        int n = static_cast<int>(params.size());
        std::vector<double> current_params(n), current_grad = grads;
        for (int i = 0; i < n; ++i) current_params[i] = *params[i];

        if (state.initialized) {
            std::vector<double> s(n), y(n);
            for (int i = 0; i < n; ++i) {
                s[i] = current_params[i] - state.prev_params[i];
                y[i] = current_grad[i] - state.prev_grad[i];
            }
            double ys = 0;
            for (int i = 0; i < n; ++i) ys += y[i] * s[i];
            if (std::fabs(ys) > 1e-30) {
                if (static_cast<int>(state.s_history.size()) >= m) {
                    state.s_history.erase(state.s_history.begin());
                    state.y_history.erase(state.y_history.begin());
                    state.rho_history.erase(state.rho_history.begin());
                }
                state.s_history.push_back(s);
                state.y_history.push_back(y);
                state.rho_history.push_back(1.0 / ys);
            }
        }

        // Two-loop recursion
        std::vector<double> q = current_grad;
        int k = static_cast<int>(state.s_history.size());
        std::vector<double> alpha_vals(k);
        for (int i = k - 1; i >= 0; --i) {
            double a = 0;
            for (int j = 0; j < n; ++j) a += state.s_history[i][j] * q[j];
            a *= state.rho_history[i];
            alpha_vals[i] = a;
            for (int j = 0; j < n; ++j) q[j] -= a * state.y_history[i][j];
        }

        // Scale by gamma
        double gamma = 1.0;
        if (k > 0) {
            double yy = 0, sy = 0;
            for (int j = 0; j < n; ++j) {
                yy += state.y_history[k-1][j] * state.y_history[k-1][j];
                sy += state.s_history[k-1][j] * state.y_history[k-1][j];
            }
            if (yy > 0) gamma = sy / yy;
        }
        for (int j = 0; j < n; ++j) q[j] *= gamma;

        for (int i = 0; i < k; ++i) {
            double b = 0;
            for (int j = 0; j < n; ++j) b += state.y_history[i][j] * q[j];
            b *= state.rho_history[i];
            for (int j = 0; j < n; ++j)
                q[j] += (alpha_vals[i] - b) * state.s_history[i][j];
        }

        // Update parameters
        for (int i = 0; i < n; ++i) *params[i] -= lr * q[i];

        state.prev_params = current_params;
        state.prev_grad = current_grad;
        state.initialized = true;
    }
};

// ============================================================================
// Learning rate schedulers
// ============================================================================
namespace lr_schedule {

// Step decay: lr = lr0 * gamma^(epoch / step_size)
struct StepLR {
    double lr0;
    int step_size;
    double gamma;

    StepLR(double lr = 1e-3, int ss = 1000, double g = 0.1)
        : lr0(lr), step_size(ss), gamma(g) {}

    double operator()(int epoch) const {
        return lr0 * std::pow(gamma, epoch / step_size);
    }
};

// Cosine annealing
struct CosineAnnealingLR {
    double lr0, eta_min;
    int T_max;

    CosineAnnealingLR(double lr = 1e-3, int tmax = 1000, double emin = 0.0)
        : lr0(lr), eta_min(emin), T_max(tmax) {}

    double operator()(int epoch) const {
        return eta_min + (lr0 - eta_min) * (1.0 + std::cos(M_PI * epoch / T_max)) / 2.0;
    }
};

// Exponential decay
struct ExponentialLR {
    double lr0, gamma;

    ExponentialLR(double lr = 1e-3, double g = 0.999)
        : lr0(lr), gamma(g) {}

    double operator()(int epoch) const {
        return lr0 * std::pow(gamma, epoch);
    }
};

// Inverse time decay: lr = lr0 / (1 + decay_rate * step / step_size)
struct InverseTimeLR {
    double lr0, decay_rate;
    int step_size;

    InverseTimeLR(double lr = 1e-3, int ss = 1000, double dr = 0.5)
        : lr0(lr), decay_rate(dr), step_size(ss) {}

    double operator()(int epoch) const {
        return lr0 / (1.0 + decay_rate * static_cast<double>(epoch) / step_size);
    }
};

} // namespace lr_schedule

// ============================================================================
// Gradient computation helpers (Jacobian, Hessian)
// Ports deepxde.gradients
// ============================================================================
namespace gradients {

// Jacobian: dy_i/dx_j computed via AD
// Returns a Matrix(dim_y, dim_x) for a single input point
inline Matrix jacobian_ad(const FNN& net, const std::vector<double>& x) {
    int dim_x = net.input_dim();
    int dim_y = net.output_dim();
    Matrix J(dim_y, dim_x);

    for (int i = 0; i < dim_y; ++i) {
        std::vector<VarPtr> x_ad(dim_x);
        for (int d = 0; d < dim_x; ++d) x_ad[d] = make_var(x[d]);
        auto u = net.forward_ad(x_ad);
        u[i]->backward();
        for (int j = 0; j < dim_x; ++j)
            J(i, j) = x_ad[j]->grad;
    }
    return J;
}

// Hessian: d^2 y_component / (dx_i dx_j) via finite differences on AD gradient
inline Matrix hessian_ad(const FNN& net, const std::vector<double>& x,
                          int component = 0, double h = 1e-5) {
    int dim_x = net.input_dim();
    Matrix H(dim_x, dim_x);

    for (int j = 0; j < dim_x; ++j) {
        // Compute dy/dx at x + h*e_j and x - h*e_j
        auto xp = x; xp[j] += h;
        auto xm = x; xm[j] -= h;

        auto Jp = jacobian_ad(net, xp);
        auto Jm = jacobian_ad(net, xm);

        for (int i = 0; i < dim_x; ++i)
            H(i, j) = (Jp(component, i) - Jm(component, i)) / (2.0 * h);
    }
    return H;
}

// Laplacian: sum of d^2y/dx_i^2
inline double laplacian_ad(const FNN& net, const std::vector<double>& x,
                            int component = 0, double h = 1e-5) {
    double lap = 0;
    int dim_x = net.input_dim();
    for (int i = 0; i < dim_x; ++i)
        lap += compute_d2u_dx2(net, x, component, i, h);
    return lap;
}

} // namespace gradients

// ============================================================================
// Fractional calculus helpers
// Ports deepxde.data.fpde (Grunwald-Letnikov weights)
// ============================================================================
namespace fractional {

// Grunwald-Letnikov weights for fractional derivative of order alpha
inline std::vector<double> gl_weights(double alpha, int n) {
    std::vector<double> w(n + 1);
    w[0] = 1.0;
    for (int j = 1; j <= n; ++j)
        w[j] = w[j-1] * (j - 1 - alpha) / j;
    return w;
}

// 1D fractional Laplacian constant C(alpha, D=1) = 1 / (2 * cos(alpha * pi / 2))
inline double frac_laplacian_constant_1d(double alpha) {
    return 1.0 / (2.0 * std::cos(alpha * M_PI / 2.0));
}

// N-D fractional Laplacian constant C(alpha, D)
// C = Gamma((1-alpha)/2) * Gamma((D+alpha)/2) / (2 * pi^((D+1)/2))
inline double frac_laplacian_constant(double alpha, int D) {
    // Use simplified Stirling approx for Gamma
    // For exact computation, use std::tgamma
    double num = std::tgamma((1.0 - alpha) / 2.0) * std::tgamma((D + alpha) / 2.0);
    double den = 2.0 * std::pow(M_PI, (D + 1.0) / 2.0);
    return num / den;
}

} // namespace fractional

// ============================================================================
// Feature transforms
// Ports deepxde.maps (Fourier features, input/output transforms)
// ============================================================================
namespace transforms {

// Random Fourier features transform
// Maps x (dim) -> [cos(Bx), sin(Bx)] (2 * num_features)
struct FourierFeatures {
    Matrix B;  // (dim, num_features)
    int num_features;

    FourierFeatures(int dim, int nf, double sigma = 1.0) : num_features(nf), B(dim, nf) {
        std::normal_distribution<double> norm(0.0, sigma);
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < nf; ++j)
                B(i, j) = norm(global_rng());
    }

    std::vector<double> transform(const std::vector<double>& x) const {
        std::vector<double> out(2 * num_features);
        for (int j = 0; j < num_features; ++j) {
            double dot = 0;
            for (int i = 0; i < B.rows; ++i) dot += x[i] * B(i, j);
            out[j] = std::cos(dot);
            out[j + num_features] = std::sin(dot);
        }
        return out;
    }

    Matrix transform_batch(const Matrix& X) const {
        Matrix out(X.rows, 2 * num_features);
        for (int i = 0; i < X.rows; ++i) {
            auto t = transform(X.row(i));
            out.set_row(i, t);
        }
        return out;
    }
};

// Standardization (zero mean, unit variance)
struct Standardizer {
    std::vector<double> mean, std_dev;
    int dim;
    bool fitted = false;

    void fit(const Matrix& X) {
        dim = X.cols;
        mean.assign(dim, 0.0);
        std_dev.assign(dim, 0.0);
        for (int i = 0; i < X.rows; ++i)
            for (int j = 0; j < dim; ++j) mean[j] += X(i, j);
        for (int j = 0; j < dim; ++j) mean[j] /= X.rows;
        for (int i = 0; i < X.rows; ++i)
            for (int j = 0; j < dim; ++j) {
                double d = X(i, j) - mean[j];
                std_dev[j] += d * d;
            }
        for (int j = 0; j < dim; ++j) std_dev[j] = std::sqrt(std_dev[j] / X.rows);
        fitted = true;
    }

    Matrix transform(const Matrix& X) const {
        Matrix out(X.rows, X.cols);
        for (int i = 0; i < X.rows; ++i)
            for (int j = 0; j < X.cols; ++j)
                out(i, j) = (std_dev[j] > 1e-15) ? (X(i, j) - mean[j]) / std_dev[j] : X(i, j);
        return out;
    }

    Matrix inverse_transform(const Matrix& X) const {
        Matrix out(X.rows, X.cols);
        for (int i = 0; i < X.rows; ++i)
            for (int j = 0; j < X.cols; ++j)
                out(i, j) = X(i, j) * std_dev[j] + mean[j];
        return out;
    }
};

} // namespace transforms

// ============================================================================
// Full Model class (extended)
// Ports deepxde.model.Model (compile, train, predict, save, restore)
// ============================================================================
struct TrainState {
    int iteration = 0;
    double loss_train = 0;
    double loss_test = 0;
    std::vector<double> loss_train_history;
    std::vector<double> loss_test_history;
    Matrix X_test;
    Matrix y_pred_test;
    Matrix y_std_test;
};

class FullModel {
public:
    FNN net;
    std::shared_ptr<Geometry> geom;
    std::shared_ptr<GeometryXTime> geom_time;
    PDEResidualAD pde_residual;
    std::vector<std::shared_ptr<BCBase>> bcs;
    std::vector<std::shared_ptr<IC>> ics;

    Adam optimizer;
    double learning_rate_ = 1e-3;
    std::string loss_name = "MSE";
    std::vector<double> loss_weights;

    CallbackList callbacks;
    TrainState train_state;
    bool stop_training = false;

    int num_domain = 100, num_boundary = 50, num_initial = 50;
    double weight_pde = 1.0, weight_bc = 1.0, weight_ic = 1.0;
    std::string train_distribution = "pseudo";

    Matrix train_pde_points, train_bc_points, train_ic_points;
    std::vector<Matrix> bc_collocation, ic_collocation;

    // Input/output transforms
    std::function<std::vector<double>(const std::vector<double>&)> input_transform;
    std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&)> output_transform;

    void set_network(const FNN& network) { net = network; }
    void set_pde(PDEResidualAD residual) { pde_residual = std::move(residual); }
    void add_bc(std::shared_ptr<BCBase> bc) { bcs.push_back(bc); }
    void add_ic(std::shared_ptr<IC> ic) { ics.push_back(ic); }
    void set_geometry(std::shared_ptr<Geometry> g) { geom = g; }
    void set_geometry_time(std::shared_ptr<GeometryXTime> gt) { geom_time = gt; geom = gt; }

    void compile(const std::string& opt_name = "adam", double lr = 1e-3,
                 const std::string& loss = "MSE",
                 const std::vector<double>& lw = {}) {
        learning_rate_ = lr;
        loss_name = loss;
        loss_weights = lw;
        optimizer = Adam(lr);
        auto params = net.param_vals();
        optimizer.init(static_cast<int>(params.size()));
        sample_train_points();
    }

    void sample_train_points() {
        if (geom_time) {
            train_pde_points = geom_time->random_points(num_domain);
            train_bc_points = geom_time->random_boundary_points(num_boundary);
            train_ic_points = geom_time->random_initial_points(num_initial);
        } else if (geom) {
            train_pde_points = geom->random_points(num_domain);
            train_bc_points = geom->random_boundary_points(num_boundary);
            train_ic_points = Matrix(0, geom->dim);
        }
        bc_collocation.clear();
        for (auto& bc : bcs) {
            Matrix all(train_pde_points.rows + train_bc_points.rows, train_pde_points.cols);
            for (int i = 0; i < train_bc_points.rows; ++i) all.set_row(i, train_bc_points.row(i));
            for (int i = 0; i < train_pde_points.rows; ++i) all.set_row(train_bc_points.rows + i, train_pde_points.row(i));
            bc_collocation.push_back(bc->collocation_points(all));
        }
        ic_collocation.clear();
        for (auto& ic : ics) {
            Matrix all(train_ic_points.rows + train_pde_points.rows, train_pde_points.cols);
            for (int i = 0; i < train_ic_points.rows; ++i) all.set_row(i, train_ic_points.row(i));
            for (int i = 0; i < train_pde_points.rows; ++i) all.set_row(train_ic_points.rows + i, train_pde_points.row(i));
            ic_collocation.push_back(ic->collocation_points(all));
        }
    }

    std::vector<TrainResult> train(int epochs, int display_every = 100,
                                   bool resample = false, int resample_every = 100) {
        callbacks.set_model(nullptr);
        callbacks.on_train_begin();
        std::vector<TrainResult> history;
        for (int ep = 0; ep < epochs && !stop_training; ++ep) {
            callbacks.on_epoch_begin();
            if (resample && ep > 0 && ep % resample_every == 0)
                sample_train_points();
            double loss = train_step_internal();
            train_state.iteration = ep;
            train_state.loss_train = loss;
            callbacks.on_epoch_end();
            if (ep % display_every == 0 || ep == epochs - 1) {
                std::cout << "Epoch " << std::setw(6) << ep
                          << "  loss=" << std::scientific << std::setprecision(4) << loss
                          << std::defaultfloat << "\n";
                train_state.loss_train_history.push_back(loss);
                history.push_back({ep, loss, 0, 0, 0});
            }
        }
        callbacks.on_train_end();
        return history;
    }

    Matrix predict(const Matrix& points) const { return net.forward_batch(points); }
    std::vector<double> predict(const std::vector<double>& point) const { return net.forward(point); }

    void save(const std::string& filepath) const {
        std::ofstream f(filepath, std::ios::binary);
        // Save layer sizes
        int nl = net.num_layers();
        f.write(reinterpret_cast<const char*>(&nl), sizeof(nl));
        for (auto sz : net.layer_sizes)
            f.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        // Save weights and biases
        for (int l = 0; l < nl; ++l) {
            for (auto& w : net.w_val[l]) f.write(reinterpret_cast<const char*>(&w), sizeof(w));
            for (auto& b : net.b_val[l]) f.write(reinterpret_cast<const char*>(&b), sizeof(b));
        }
    }

    void restore(const std::string& filepath) {
        std::ifstream f(filepath, std::ios::binary);
        int nl;
        f.read(reinterpret_cast<char*>(&nl), sizeof(nl));
        std::vector<int> sizes(nl + 1);
        for (auto& sz : sizes) f.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        net = FNN(sizes, net.hidden_activation, net.output_activation);
        for (int l = 0; l < nl; ++l) {
            for (auto& w : net.w_val[l]) f.read(reinterpret_cast<char*>(&w), sizeof(w));
            for (auto& b : net.b_val[l]) f.read(reinterpret_cast<char*>(&b), sizeof(b));
        }
        net.sync_var_from_val();
    }

private:
    double train_step_internal() {
        net.sync_var_from_val();
        VarPtr total_loss = make_var(0.0);
        // PDE loss
        if (pde_residual && train_pde_points.rows > 0) {
            VarPtr pde_loss = make_var(0.0);
            int n = train_pde_points.rows;
            for (int i = 0; i < n; ++i) {
                auto row = train_pde_points.row(i);
                std::vector<VarPtr> x_ad(row.size());
                for (size_t d = 0; d < row.size(); ++d) x_ad[d] = make_var(row[d]);
                auto u_ad = net.forward_ad(x_ad);
                VarPtr res = pde_residual(x_ad, u_ad);
                pde_loss = pde_loss + ad::square(res);
            }
            total_loss = total_loss + pde_loss / static_cast<double>(n) * weight_pde;
        }
        // BC loss
        for (size_t b = 0; b < bcs.size(); ++b) {
            const Matrix& pts = bc_collocation[b];
            if (pts.rows == 0) continue;
            VarPtr bc_loss = make_var(0.0);
            for (int i = 0; i < pts.rows; ++i) {
                auto row = pts.row(i);
                std::vector<VarPtr> x_ad(row.size());
                for (size_t d = 0; d < row.size(); ++d) x_ad[d] = make_var(row[d]);
                auto u_ad = net.forward_ad(x_ad);
                std::vector<std::vector<double>> out = {net.forward(row)};
                auto errs = bcs[b]->error([&](){Matrix m(1,pts.cols);m.set_row(0,row);return m;}(), out);
                double target = u_ad[0]->val - errs[0];
                VarPtr e = u_ad[0] - make_var(target);
                bc_loss = bc_loss + ad::square(e);
            }
            total_loss = total_loss + bc_loss / static_cast<double>(pts.rows) * weight_bc;
        }
        // IC loss
        for (size_t c = 0; c < ics.size(); ++c) {
            const Matrix& pts = ic_collocation[c];
            if (pts.rows == 0) continue;
            VarPtr ic_loss = make_var(0.0);
            for (int i = 0; i < pts.rows; ++i) {
                auto row = pts.row(i);
                std::vector<VarPtr> x_ad(row.size());
                for (size_t d = 0; d < row.size(); ++d) x_ad[d] = make_var(row[d]);
                auto u_ad = net.forward_ad(x_ad);
                double tv = ics[c]->func(row);
                VarPtr e = u_ad[ics[c]->component] - make_var(tv);
                ic_loss = ic_loss + ad::square(e);
            }
            total_loss = total_loss + ic_loss / static_cast<double>(pts.rows) * weight_ic;
        }
        // Backward
        total_loss->backward();
        auto param_ptrs = net.param_vals();
        std::vector<double> grads(param_ptrs.size());
        int idx = 0;
        for (int l = 0; l < net.num_layers(); ++l) {
            for (size_t i = 0; i < net.weights[l].size(); ++i) grads[idx++] = net.weights[l][i]->grad;
            for (size_t i = 0; i < net.biases[l].size(); ++i) grads[idx++] = net.biases[l][i]->grad;
        }
        optimizer.step(param_ptrs, grads);
        return total_loss->val;
    }
};

} // namespace deepxde
#endif // DEEPXDE_LOSSES_H
