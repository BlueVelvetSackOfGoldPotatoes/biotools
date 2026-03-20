// deepxde.h - Header-only C++17 port of DeepXDE
// Physics-Informed Neural Networks (PINNs) for solving PDEs
//
// Port of: https://github.com/lululxvi/deepxde
//
// Key components ported:
//   - Tape-based automatic differentiation (reverse mode)
//   - Feedforward neural network (MLP) with backpropagation
//   - PDE residual computation via AD
//   - Domain/geometry classes (Interval, Rectangle, GeometryXTime)
//   - Boundary/initial condition types (Dirichlet, Neumann, IC)
//   - PINN solver with Adam optimizer training loop
//   - Collocation point sampling (uniform, pseudo-random)

#ifndef DEEPXDE_H
#define DEEPXDE_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace deepxde {

// ============================================================================
// Forward declarations
// ============================================================================
class Var;
using VarPtr = std::shared_ptr<Var>;

// ============================================================================
// Computational-graph automatic differentiation (reverse mode, tape-based)
//
// Mirrors the role of PyTorch autograd / TF GradientTape used by DeepXDE.
// Each Var node stores its value, gradient, and pointers to parent nodes
// plus a local-gradient lambda used during backward().
// ============================================================================

class Var : public std::enable_shared_from_this<Var> {
public:
    double val = 0.0;
    double grad = 0.0;
    // Parents and local gradient values (d_this/d_parent)
    std::vector<std::pair<VarPtr, double>> parents;
    bool requires_grad = false;

    explicit Var(double v, bool rg = false) : val(v), requires_grad(rg) {}

    // Topological sort for backward pass
    void backward() {
        // Build topological order
        std::vector<Var*> topo;
        std::unordered_set<Var*> visited;
        std::function<void(Var*)> build = [&](Var* node) {
            if (visited.count(node)) return;
            visited.insert(node);
            for (auto& p : node->parents) {
                build(p.first.get());
            }
            topo.push_back(node);
        };
        build(this);

        // Zero all gradients
        for (auto* n : topo) n->grad = 0.0;
        this->grad = 1.0;

        // Reverse iterate
        for (int i = static_cast<int>(topo.size()) - 1; i >= 0; --i) {
            Var* node = topo[i];
            for (auto& p : node->parents) {
                p.first->grad += node->grad * p.second;
            }
        }
    }
};

// Convenience constructors
inline VarPtr make_var(double v, bool rg = false) {
    return std::make_shared<Var>(v, rg);
}

// ============================================================================
// AD operations  (operator overloads on VarPtr)
// ============================================================================
inline VarPtr operator+(const VarPtr& a, const VarPtr& b) {
    auto out = make_var(a->val + b->val);
    out->parents.push_back({a, 1.0});
    out->parents.push_back({b, 1.0});
    return out;
}
inline VarPtr operator-(const VarPtr& a, const VarPtr& b) {
    auto out = make_var(a->val - b->val);
    out->parents.push_back({a, 1.0});
    out->parents.push_back({b, -1.0});
    return out;
}
inline VarPtr operator*(const VarPtr& a, const VarPtr& b) {
    auto out = make_var(a->val * b->val);
    out->parents.push_back({a, b->val});
    out->parents.push_back({b, a->val});
    return out;
}
inline VarPtr operator/(const VarPtr& a, const VarPtr& b) {
    auto out = make_var(a->val / b->val);
    out->parents.push_back({a, 1.0 / b->val});
    out->parents.push_back({b, -a->val / (b->val * b->val)});
    return out;
}
inline VarPtr operator-(const VarPtr& a) {
    auto out = make_var(-a->val);
    out->parents.push_back({a, -1.0});
    return out;
}

// Scalar-Var mixed ops
inline VarPtr operator+(double s, const VarPtr& a) { return make_var(s) + a; }
inline VarPtr operator+(const VarPtr& a, double s) { return a + make_var(s); }
inline VarPtr operator-(double s, const VarPtr& a) { return make_var(s) - a; }
inline VarPtr operator-(const VarPtr& a, double s) { return a - make_var(s); }
inline VarPtr operator*(double s, const VarPtr& a) { return make_var(s) * a; }
inline VarPtr operator*(const VarPtr& a, double s) { return a * make_var(s); }
inline VarPtr operator/(double s, const VarPtr& a) { return make_var(s) / a; }
inline VarPtr operator/(const VarPtr& a, double s) { return a / make_var(s); }

// Transcendental functions
namespace ad {

inline VarPtr sin(const VarPtr& a) {
    auto out = make_var(std::sin(a->val));
    out->parents.push_back({a, std::cos(a->val)});
    return out;
}
inline VarPtr cos(const VarPtr& a) {
    auto out = make_var(std::cos(a->val));
    out->parents.push_back({a, -std::sin(a->val)});
    return out;
}
inline VarPtr exp(const VarPtr& a) {
    double ev = std::exp(a->val);
    auto out = make_var(ev);
    out->parents.push_back({a, ev});
    return out;
}
inline VarPtr log(const VarPtr& a) {
    auto out = make_var(std::log(a->val));
    out->parents.push_back({a, 1.0 / a->val});
    return out;
}
inline VarPtr tanh(const VarPtr& a) {
    double tv = std::tanh(a->val);
    auto out = make_var(tv);
    out->parents.push_back({a, 1.0 - tv * tv});
    return out;
}
inline VarPtr sigmoid(const VarPtr& a) {
    double sv = 1.0 / (1.0 + std::exp(-a->val));
    auto out = make_var(sv);
    out->parents.push_back({a, sv * (1.0 - sv)});
    return out;
}
inline VarPtr relu(const VarPtr& a) {
    double rv = a->val > 0.0 ? a->val : 0.0;
    auto out = make_var(rv);
    out->parents.push_back({a, a->val > 0.0 ? 1.0 : 0.0});
    return out;
}
inline VarPtr square(const VarPtr& a) {
    auto out = make_var(a->val * a->val);
    out->parents.push_back({a, 2.0 * a->val});
    return out;
}
inline VarPtr abs_val(const VarPtr& a) {
    auto out = make_var(std::fabs(a->val));
    out->parents.push_back({a, a->val >= 0.0 ? 1.0 : -1.0});
    return out;
}
inline VarPtr pow(const VarPtr& a, double p) {
    auto out = make_var(std::pow(a->val, p));
    out->parents.push_back({a, p * std::pow(a->val, p - 1.0)});
    return out;
}
inline VarPtr sqrt(const VarPtr& a) {
    double sv = std::sqrt(a->val);
    auto out = make_var(sv);
    out->parents.push_back({a, 0.5 / sv});
    return out;
}

} // namespace ad

// ============================================================================
// Simple Matrix / Vector types (row-major, double)
// ============================================================================
struct Matrix {
    int rows = 0, cols = 0;
    std::vector<double> data;

    Matrix() = default;
    Matrix(int r, int c, double fill = 0.0) : rows(r), cols(c), data(r * c, fill) {}

    double& operator()(int i, int j) { return data[i * cols + j]; }
    double  operator()(int i, int j) const { return data[i * cols + j]; }

    std::vector<double> row(int i) const {
        return std::vector<double>(data.begin() + i * cols, data.begin() + (i + 1) * cols);
    }
    void set_row(int i, const std::vector<double>& v) {
        for (int j = 0; j < cols; ++j) data[i * cols + j] = v[j];
    }
};

// ============================================================================
// Random number generation
// ============================================================================
inline std::mt19937& global_rng() {
    static std::mt19937 rng(42);
    return rng;
}

inline void set_seed(unsigned seed) {
    global_rng().seed(seed);
}

// ============================================================================
// Geometry classes (ports of deepxde.geometry)
// ============================================================================

class Geometry {
public:
    int dim;
    std::vector<double> bbox_lo, bbox_hi;
    double diam;
    virtual ~Geometry() = default;

    Geometry(int d, const std::vector<double>& lo, const std::vector<double>& hi, double diameter)
        : dim(d), bbox_lo(lo), bbox_hi(hi), diam(diameter) {}

    virtual bool inside(const std::vector<double>& x) const = 0;
    virtual bool on_boundary(const std::vector<double>& x) const = 0;
    virtual Matrix random_points(int n) const = 0;
    virtual Matrix uniform_points(int n, bool boundary = true) const = 0;
    virtual Matrix random_boundary_points(int n) const = 0;
    virtual Matrix uniform_boundary_points(int n) const = 0;

    virtual std::vector<double> boundary_normal(const std::vector<double>& /*x*/) const {
        throw std::runtime_error("boundary_normal not implemented");
    }
};

// --- Interval [l, r] (1D) ---
class Interval : public Geometry {
public:
    double l, r;

    Interval(double l_, double r_)
        : Geometry(1, {l_}, {r_}, r_ - l_), l(l_), r(r_) {}

    bool inside(const std::vector<double>& x) const override {
        return x[0] >= l && x[0] <= r;
    }
    bool on_boundary(const std::vector<double>& x) const override {
        return std::fabs(x[0] - l) < 1e-12 || std::fabs(x[0] - r) < 1e-12;
    }
    std::vector<double> boundary_normal(const std::vector<double>& x) const override {
        if (std::fabs(x[0] - l) < 1e-12) return {-1.0};
        if (std::fabs(x[0] - r) < 1e-12) return {1.0};
        return {0.0};
    }
    Matrix random_points(int n) const override {
        Matrix m(n, 1);
        std::uniform_real_distribution<double> dist(l, r);
        for (int i = 0; i < n; ++i) m(i, 0) = dist(global_rng());
        return m;
    }
    Matrix uniform_points(int n, bool boundary = true) const override {
        Matrix m(n, 1);
        if (boundary) {
            for (int i = 0; i < n; ++i)
                m(i, 0) = l + (r - l) * i / std::max(n - 1, 1);
        } else {
            for (int i = 0; i < n; ++i)
                m(i, 0) = l + (r - l) * (i + 1.0) / (n + 1.0);
        }
        return m;
    }
    Matrix random_boundary_points(int n) const override {
        Matrix m(n, 1);
        for (int i = 0; i < n; ++i)
            m(i, 0) = (global_rng()() % 2 == 0) ? l : r;
        return m;
    }
    Matrix uniform_boundary_points(int n) const override {
        Matrix m(n, 1);
        int half = n / 2;
        for (int i = 0; i < half; ++i) m(i, 0) = l;
        for (int i = half; i < n; ++i) m(i, 0) = r;
        return m;
    }
};

// --- Rectangle [xmin, xmax] x [ymin, ymax] (2D) ---
class Rectangle : public Geometry {
public:
    double xmin_v[2], xmax_v[2];

    Rectangle(const std::vector<double>& lo, const std::vector<double>& hi)
        : Geometry(2, lo, hi, std::hypot(hi[0] - lo[0], hi[1] - lo[1])) {
        xmin_v[0] = lo[0]; xmin_v[1] = lo[1];
        xmax_v[0] = hi[0]; xmax_v[1] = hi[1];
    }

    bool inside(const std::vector<double>& x) const override {
        return x[0] >= xmin_v[0] && x[0] <= xmax_v[0] &&
               x[1] >= xmin_v[1] && x[1] <= xmax_v[1];
    }
    bool on_boundary(const std::vector<double>& x) const override {
        bool on_x = std::fabs(x[0] - xmin_v[0]) < 1e-12 || std::fabs(x[0] - xmax_v[0]) < 1e-12;
        bool on_y = std::fabs(x[1] - xmin_v[1]) < 1e-12 || std::fabs(x[1] - xmax_v[1]) < 1e-12;
        return (on_x && x[1] >= xmin_v[1] - 1e-12 && x[1] <= xmax_v[1] + 1e-12) ||
               (on_y && x[0] >= xmin_v[0] - 1e-12 && x[0] <= xmax_v[0] + 1e-12);
    }
    std::vector<double> boundary_normal(const std::vector<double>& x) const override {
        std::vector<double> n = {0, 0};
        if (std::fabs(x[0] - xmin_v[0]) < 1e-12) n[0] = -1;
        else if (std::fabs(x[0] - xmax_v[0]) < 1e-12) n[0] = 1;
        if (std::fabs(x[1] - xmin_v[1]) < 1e-12) n[1] = -1;
        else if (std::fabs(x[1] - xmax_v[1]) < 1e-12) n[1] = 1;
        double len = std::hypot(n[0], n[1]);
        if (len > 0) { n[0] /= len; n[1] /= len; }
        return n;
    }
    Matrix random_points(int n) const override {
        Matrix m(n, 2);
        std::uniform_real_distribution<double> dx(xmin_v[0], xmax_v[0]);
        std::uniform_real_distribution<double> dy(xmin_v[1], xmax_v[1]);
        for (int i = 0; i < n; ++i) {
            m(i, 0) = dx(global_rng());
            m(i, 1) = dy(global_rng());
        }
        return m;
    }
    Matrix uniform_points(int n, bool boundary = true) const override {
        double lx = xmax_v[0] - xmin_v[0], ly = xmax_v[1] - xmin_v[1];
        int nx = std::max(1, static_cast<int>(std::ceil(std::sqrt(n * lx / ly))));
        int ny = std::max(1, static_cast<int>(std::ceil(static_cast<double>(n) / nx)));
        Matrix m(nx * ny, 2);
        int idx = 0;
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j) {
                m(idx, 0) = boundary ? xmin_v[0] + lx * i / std::max(nx - 1, 1)
                                     : xmin_v[0] + lx * (i + 1.0) / (nx + 1.0);
                m(idx, 1) = boundary ? xmin_v[1] + ly * j / std::max(ny - 1, 1)
                                     : xmin_v[1] + ly * (j + 1.0) / (ny + 1.0);
                idx++;
            }
        return m;
    }
    Matrix random_boundary_points(int n) const override {
        Matrix m(n, 2);
        double lx = xmax_v[0] - xmin_v[0], ly = xmax_v[1] - xmin_v[1];
        double perim = 2 * (lx + ly);
        std::uniform_real_distribution<double> dist(0, perim);
        for (int i = 0; i < n; ++i) {
            double u = dist(global_rng());
            if (u < lx) { m(i, 0) = xmin_v[0] + u; m(i, 1) = xmin_v[1]; }
            else if (u < lx + ly) { m(i, 0) = xmax_v[0]; m(i, 1) = xmin_v[1] + (u - lx); }
            else if (u < 2 * lx + ly) { m(i, 0) = xmax_v[0] - (u - lx - ly); m(i, 1) = xmax_v[1]; }
            else { m(i, 0) = xmin_v[0]; m(i, 1) = xmax_v[1] - (u - 2 * lx - ly); }
        }
        return m;
    }
    Matrix uniform_boundary_points(int n) const override {
        return random_boundary_points(n);
    }
};

// --- TimeDomain (1D interval representing time) ---
class TimeDomain : public Interval {
public:
    double t0, t1;
    TimeDomain(double t0_, double t1_) : Interval(t0_, t1_), t0(t0_), t1(t1_) {}

    bool on_initial(double t) const {
        return std::fabs(t - t0) < 1e-12;
    }
};

// --- GeometryXTime (spatial geometry x time domain) ---
class GeometryXTime : public Geometry {
public:
    std::shared_ptr<Geometry> geometry;
    std::shared_ptr<TimeDomain> timedomain;

    GeometryXTime(std::shared_ptr<Geometry> geom, std::shared_ptr<TimeDomain> td)
        : Geometry(geom->dim + 1,
                   [&]() {
                       auto lo = geom->bbox_lo;
                       lo.push_back(td->t0);
                       return lo;
                   }(),
                   [&]() {
                       auto hi = geom->bbox_hi;
                       hi.push_back(td->t1);
                       return hi;
                   }(),
                   std::max(geom->diam, td->diam)),
          geometry(geom), timedomain(td) {}

    bool inside(const std::vector<double>& x) const override {
        std::vector<double> xs(x.begin(), x.begin() + geometry->dim);
        return geometry->inside(xs) && x.back() >= timedomain->t0 && x.back() <= timedomain->t1;
    }
    bool on_boundary(const std::vector<double>& x) const override {
        std::vector<double> xs(x.begin(), x.begin() + geometry->dim);
        return geometry->on_boundary(xs);
    }
    bool on_initial(const std::vector<double>& x) const {
        return timedomain->on_initial(x.back());
    }
    std::vector<double> boundary_normal(const std::vector<double>& x) const override {
        std::vector<double> xs(x.begin(), x.begin() + geometry->dim);
        auto n = geometry->boundary_normal(xs);
        n.push_back(0.0);
        return n;
    }

    Matrix random_points(int n) const override {
        Matrix m(n, dim);
        auto spatial = geometry->random_points(n);
        std::uniform_real_distribution<double> dt(timedomain->t0, timedomain->t1);
        std::vector<double> times(n);
        for (int i = 0; i < n; ++i) times[i] = dt(global_rng());
        std::shuffle(times.begin(), times.end(), global_rng());
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < geometry->dim; ++j)
                m(i, j) = spatial(i, j);
            m(i, dim - 1) = times[i];
        }
        return m;
    }

    Matrix uniform_points(int n, bool boundary = true) const override {
        int nx = std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(n)))));
        int nt = std::max(1, static_cast<int>(std::ceil(static_cast<double>(n) / nx)));
        auto x_pts = geometry->uniform_points(nx, boundary);
        nx = x_pts.rows;
        std::vector<double> t_vals(nt);
        for (int i = 0; i < nt; ++i) {
            t_vals[i] = boundary ? timedomain->t0 + (timedomain->t1 - timedomain->t0) * i / std::max(nt - 1, 1)
                                 : timedomain->t0 + (timedomain->t1 - timedomain->t0) * (i + 1.0) / (nt + 1.0);
        }
        Matrix m(nx * nt, dim);
        int idx = 0;
        for (int ti = 0; ti < nt; ++ti) {
            for (int xi = 0; xi < nx; ++xi) {
                for (int d = 0; d < geometry->dim; ++d)
                    m(idx, d) = x_pts(xi, d);
                m(idx, dim - 1) = t_vals[ti];
                idx++;
            }
        }
        return m;
    }

    Matrix random_boundary_points(int n) const override {
        Matrix m(n, dim);
        auto sp = geometry->random_boundary_points(n);
        std::uniform_real_distribution<double> dt(timedomain->t0, timedomain->t1);
        std::vector<double> times(n);
        for (int i = 0; i < n; ++i) times[i] = dt(global_rng());
        std::shuffle(times.begin(), times.end(), global_rng());
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < geometry->dim; ++j)
                m(i, j) = sp(i, j);
            m(i, dim - 1) = times[i];
        }
        return m;
    }

    Matrix uniform_boundary_points(int n) const override {
        return random_boundary_points(n);
    }

    Matrix random_initial_points(int n) const {
        Matrix m(n, dim);
        auto sp = geometry->random_points(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < geometry->dim; ++j)
                m(i, j) = sp(i, j);
            m(i, dim - 1) = timedomain->t0;
        }
        return m;
    }

    Matrix uniform_initial_points(int n) const {
        auto sp = geometry->uniform_points(n, true);
        int actual_n = sp.rows;
        Matrix out(actual_n, dim);
        for (int i = 0; i < actual_n; ++i) {
            for (int j = 0; j < geometry->dim; ++j)
                out(i, j) = sp(i, j);
            out(i, dim - 1) = timedomain->t0;
        }
        return out;
    }
};

// ============================================================================
// Neural Network: Feedforward MLP with AD support
// Mirrors deepxde.nn.pytorch.FNN
// ============================================================================

enum class Activation { Tanh, Sigmoid, ReLU, Sin, Linear };

inline VarPtr apply_activation(const VarPtr& x, Activation act) {
    switch (act) {
        case Activation::Tanh:    return ad::tanh(x);
        case Activation::Sigmoid: return ad::sigmoid(x);
        case Activation::ReLU:    return ad::relu(x);
        case Activation::Sin:     return ad::sin(x);
        case Activation::Linear:  return x;
    }
    return x;
}

class FNN {
public:
    std::vector<int> layer_sizes;
    Activation hidden_activation;
    Activation output_activation;

    // Weights and biases as AD Vars
    // weights[l]: shape (layer_sizes[l] * layer_sizes[l+1]), row-major
    // biases[l]:  shape (layer_sizes[l+1])
    std::vector<std::vector<VarPtr>> weights;
    std::vector<std::vector<VarPtr>> biases;

    // Plain double mirrors for fast non-AD forward
    std::vector<std::vector<double>> w_val;
    std::vector<std::vector<double>> b_val;

    FNN() = default;

    FNN(const std::vector<int>& sizes, Activation hidden_act, Activation out_act = Activation::Linear)
        : layer_sizes(sizes), hidden_activation(hidden_act), output_activation(out_act) {
        init_params();
    }

    int num_layers() const { return static_cast<int>(layer_sizes.size()) - 1; }
    int input_dim() const { return layer_sizes.front(); }
    int output_dim() const { return layer_sizes.back(); }

    int num_trainable_parameters() const {
        int count = 0;
        for (int l = 0; l < num_layers(); ++l) {
            count += layer_sizes[l] * layer_sizes[l + 1];
            count += layer_sizes[l + 1];
        }
        return count;
    }

    // Glorot-uniform (Xavier) initialization
    void init_params() {
        weights.resize(num_layers());
        biases.resize(num_layers());
        w_val.resize(num_layers());
        b_val.resize(num_layers());

        for (int l = 0; l < num_layers(); ++l) {
            int fan_in = layer_sizes[l];
            int fan_out = layer_sizes[l + 1];
            double limit = std::sqrt(6.0 / (fan_in + fan_out));
            std::uniform_real_distribution<double> dist(-limit, limit);

            int nw = fan_in * fan_out;
            weights[l].resize(nw);
            w_val[l].resize(nw);
            for (int i = 0; i < nw; ++i) {
                double v = dist(global_rng());
                weights[l][i] = make_var(v, true);
                w_val[l][i] = v;
            }

            biases[l].resize(fan_out);
            b_val[l].resize(fan_out, 0.0);
            for (int j = 0; j < fan_out; ++j) {
                biases[l][j] = make_var(0.0, true);
            }
        }
    }

    void sync_val_from_var() {
        for (int l = 0; l < num_layers(); ++l) {
            for (size_t i = 0; i < weights[l].size(); ++i)
                w_val[l][i] = weights[l][i]->val;
            for (size_t i = 0; i < biases[l].size(); ++i)
                b_val[l][i] = biases[l][i]->val;
        }
    }

    void sync_var_from_val() {
        for (int l = 0; l < num_layers(); ++l) {
            for (size_t i = 0; i < weights[l].size(); ++i)
                weights[l][i]->val = w_val[l][i];
            for (size_t i = 0; i < biases[l].size(); ++i)
                biases[l][i]->val = b_val[l][i];
        }
    }

    std::vector<double*> param_vals() {
        std::vector<double*> p;
        for (int l = 0; l < num_layers(); ++l) {
            for (auto& w : w_val[l]) p.push_back(&w);
            for (auto& b : b_val[l]) p.push_back(&b);
        }
        return p;
    }

    // --- Forward pass (AD mode) ---
    std::vector<VarPtr> forward_ad(const std::vector<VarPtr>& input) const {
        std::vector<VarPtr> x = input;
        for (int l = 0; l < num_layers(); ++l) {
            int in_sz = layer_sizes[l];
            int out_sz = layer_sizes[l + 1];
            std::vector<VarPtr> y(out_sz);
            for (int j = 0; j < out_sz; ++j) {
                VarPtr s = biases[l][j];
                for (int i = 0; i < in_sz; ++i) {
                    s = s + weights[l][i * out_sz + j] * x[i];
                }
                Activation act = (l < num_layers() - 1) ? hidden_activation : output_activation;
                y[j] = apply_activation(s, act);
            }
            x = y;
        }
        return x;
    }

    // --- Forward pass (fast, non-AD) ---
    std::vector<double> forward(const std::vector<double>& input) const {
        std::vector<double> x = input;
        for (int l = 0; l < num_layers(); ++l) {
            int in_sz = layer_sizes[l];
            int out_sz = layer_sizes[l + 1];
            std::vector<double> y(out_sz, 0.0);
            for (int j = 0; j < out_sz; ++j) {
                double s = b_val[l][j];
                for (int i = 0; i < in_sz; ++i) {
                    s += w_val[l][i * out_sz + j] * x[i];
                }
                Activation act = (l < num_layers() - 1) ? hidden_activation : output_activation;
                switch (act) {
                    case Activation::Tanh:    s = std::tanh(s); break;
                    case Activation::Sigmoid: s = 1.0 / (1.0 + std::exp(-s)); break;
                    case Activation::ReLU:    s = s > 0 ? s : 0; break;
                    case Activation::Sin:     s = std::sin(s); break;
                    case Activation::Linear:  break;
                }
                y[j] = s;
            }
            x = y;
        }
        return x;
    }

    Matrix forward_batch(const Matrix& inputs) const {
        Matrix outputs(inputs.rows, output_dim());
        for (int i = 0; i < inputs.rows; ++i) {
            auto row_in = inputs.row(i);
            auto row_out = forward(row_in);
            outputs.set_row(i, row_out);
        }
        return outputs;
    }
};

// ============================================================================
// Adam Optimizer
// Mirrors deepxde.optimizers defaults: lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8
// ============================================================================

class Adam {
public:
    double lr;
    double beta1, beta2, eps;
    int t = 0;
    std::vector<double> m, v;

    Adam(double lr_ = 1e-3, double b1 = 0.9, double b2 = 0.999, double e = 1e-8)
        : lr(lr_), beta1(b1), beta2(b2), eps(e) {}

    void init(int n_params) {
        m.assign(n_params, 0.0);
        v.assign(n_params, 0.0);
        t = 0;
    }

    void step(std::vector<double*>& params, const std::vector<double>& grads) {
        assert(params.size() == grads.size());
        t++;
        double lr_t = lr * std::sqrt(1.0 - std::pow(beta2, t)) / (1.0 - std::pow(beta1, t));
        for (size_t i = 0; i < params.size(); ++i) {
            m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
            *params[i] -= lr_t * m[i] / (std::sqrt(v[i]) + eps);
        }
    }
};

// ============================================================================
// Boundary / Initial condition types
// Mirrors deepxde.icbc.{DirichletBC, NeumannBC, IC}
// ============================================================================

using ScalarFunc = std::function<double(const std::vector<double>&)>;
using BoundaryPred = std::function<bool(const std::vector<double>&, bool)>;

struct BCBase {
    virtual ~BCBase() = default;
    virtual Matrix collocation_points(const Matrix& all_boundary) const = 0;
    virtual std::vector<double> error(const Matrix& bc_x,
                                      const std::vector<std::vector<double>>& outputs_at_bc) const = 0;
};

// --- DirichletBC: u(x) = func(x) on the boundary ---
struct DirichletBC : BCBase {
    std::shared_ptr<Geometry> geom;
    ScalarFunc func;
    BoundaryPred on_boundary;
    int component;

    DirichletBC(std::shared_ptr<Geometry> g, ScalarFunc f, BoundaryPred pred, int comp = 0)
        : geom(g), func(std::move(f)), on_boundary(std::move(pred)), component(comp) {}

    Matrix collocation_points(const Matrix& all_boundary) const override {
        std::vector<std::vector<double>> pts;
        for (int i = 0; i < all_boundary.rows; ++i) {
            auto row = all_boundary.row(i);
            bool is_on_bdy = geom->on_boundary(row);
            if (on_boundary(row, is_on_bdy)) {
                pts.push_back(row);
            }
        }
        Matrix m(static_cast<int>(pts.size()), all_boundary.cols);
        for (int i = 0; i < m.rows; ++i)
            m.set_row(i, pts[i]);
        return m;
    }

    std::vector<double> error(const Matrix& bc_x,
                              const std::vector<std::vector<double>>& outputs_at_bc) const override {
        std::vector<double> err(bc_x.rows);
        for (int i = 0; i < bc_x.rows; ++i) {
            auto x = bc_x.row(i);
            double target = func(x);
            err[i] = outputs_at_bc[i][component] - target;
        }
        return err;
    }
};

// --- NeumannBC: du/dn(x) = func(x) on the boundary ---
struct NeumannBC : BCBase {
    std::shared_ptr<Geometry> geom;
    ScalarFunc func;
    BoundaryPred on_boundary;
    int component;
    const FNN* net;

    NeumannBC(std::shared_ptr<Geometry> g, ScalarFunc f, BoundaryPred pred,
              const FNN* network, int comp = 0)
        : geom(g), func(std::move(f)), on_boundary(std::move(pred)),
          component(comp), net(network) {}

    Matrix collocation_points(const Matrix& all_boundary) const override {
        std::vector<std::vector<double>> pts;
        for (int i = 0; i < all_boundary.rows; ++i) {
            auto row = all_boundary.row(i);
            bool is_on_bdy = geom->on_boundary(row);
            if (on_boundary(row, is_on_bdy)) {
                pts.push_back(row);
            }
        }
        Matrix m(static_cast<int>(pts.size()), all_boundary.cols);
        for (int i = 0; i < m.rows; ++i)
            m.set_row(i, pts[i]);
        return m;
    }

    std::vector<double> error(const Matrix& bc_x,
                              const std::vector<std::vector<double>>& /*outputs_at_bc*/) const override {
        const double h = 1e-5;
        std::vector<double> err(bc_x.rows);
        for (int i = 0; i < bc_x.rows; ++i) {
            auto x = bc_x.row(i);
            auto n = geom->boundary_normal(x);
            std::vector<double> xp(x.size()), xm(x.size());
            for (size_t d = 0; d < x.size(); ++d) {
                xp[d] = x[d] + h * n[d];
                xm[d] = x[d] - h * n[d];
            }
            double dudn = (net->forward(xp)[component] - net->forward(xm)[component]) / (2 * h);
            err[i] = dudn - func(x);
        }
        return err;
    }
};

// --- Initial condition: u(x, t0) = func(x, t0) ---
struct IC : BCBase {
    std::shared_ptr<GeometryXTime> geom_time;
    ScalarFunc func;
    int component;

    IC(std::shared_ptr<GeometryXTime> gt, ScalarFunc f, int comp = 0)
        : geom_time(gt), func(std::move(f)), component(comp) {}

    Matrix collocation_points(const Matrix& all_points) const override {
        std::vector<std::vector<double>> pts;
        for (int i = 0; i < all_points.rows; ++i) {
            auto row = all_points.row(i);
            if (geom_time->on_initial(row)) {
                pts.push_back(row);
            }
        }
        if (pts.empty()) return Matrix(0, all_points.cols);
        Matrix m(static_cast<int>(pts.size()), all_points.cols);
        for (int i = 0; i < m.rows; ++i) m.set_row(i, pts[i]);
        return m;
    }

    std::vector<double> error(const Matrix& bc_x,
                              const std::vector<std::vector<double>>& outputs_at_bc) const override {
        std::vector<double> err(bc_x.rows);
        for (int i = 0; i < bc_x.rows; ++i) {
            auto x = bc_x.row(i);
            err[i] = outputs_at_bc[i][component] - func(x);
        }
        return err;
    }
};

// ============================================================================
// PDE residual signature (AD mode)
// ============================================================================
using PDEResidualAD = std::function<VarPtr(const std::vector<VarPtr>& x_vars,
                                            const std::vector<VarPtr>& u_vars)>;

// ============================================================================
// PINN Model / Solver
// Mirrors deepxde.Model + deepxde.data.TimePDE
// ============================================================================

struct TrainResult {
    int epoch;
    double loss_total;
    double loss_pde;
    double loss_bc;
    double loss_ic;
};

class Model {
public:
    FNN net;
    std::shared_ptr<GeometryXTime> geom_time;
    std::shared_ptr<Geometry> geom;

    PDEResidualAD pde_residual;

    std::vector<std::shared_ptr<BCBase>> bcs;
    std::vector<std::shared_ptr<IC>> ics;

    Matrix train_pde_points;
    Matrix train_bc_points;
    Matrix train_ic_points;

    std::vector<Matrix> bc_collocation;
    std::vector<Matrix> ic_collocation;

    double weight_pde = 1.0;
    double weight_bc  = 1.0;
    double weight_ic  = 1.0;

    Adam optimizer;

    int num_domain   = 100;
    int num_boundary = 50;
    int num_initial  = 50;

    Model() = default;

    void set_network(const FNN& network) { net = network; }
    void set_pde(PDEResidualAD residual) { pde_residual = std::move(residual); }
    void add_bc(std::shared_ptr<BCBase> bc) { bcs.push_back(bc); }
    void add_ic(std::shared_ptr<IC> ic) { ics.push_back(ic); }
    void set_geometry(std::shared_ptr<Geometry> g) { geom = g; }
    void set_geometry_time(std::shared_ptr<GeometryXTime> gt) {
        geom_time = gt;
        geom = gt;
    }

    void sample_train_points() {
        if (geom_time) {
            train_pde_points = geom_time->random_points(num_domain);
            train_bc_points  = geom_time->random_boundary_points(num_boundary);
            train_ic_points  = geom_time->random_initial_points(num_initial);
        } else if (geom) {
            train_pde_points = geom->random_points(num_domain);
            train_bc_points  = geom->random_boundary_points(num_boundary);
            train_ic_points  = Matrix(0, geom->dim);
        }

        bc_collocation.clear();
        for (auto& bc : bcs) {
            Matrix all(train_pde_points.rows + train_bc_points.rows, train_pde_points.cols);
            for (int i = 0; i < train_bc_points.rows; ++i)
                all.set_row(i, train_bc_points.row(i));
            for (int i = 0; i < train_pde_points.rows; ++i)
                all.set_row(train_bc_points.rows + i, train_pde_points.row(i));
            bc_collocation.push_back(bc->collocation_points(all));
        }

        ic_collocation.clear();
        for (auto& ic : ics) {
            Matrix all(train_ic_points.rows + train_pde_points.rows, train_pde_points.cols);
            for (int i = 0; i < train_ic_points.rows; ++i)
                all.set_row(i, train_ic_points.row(i));
            for (int i = 0; i < train_pde_points.rows; ++i)
                all.set_row(train_ic_points.rows + i, train_pde_points.row(i));
            ic_collocation.push_back(ic->collocation_points(all));
        }
    }

    void compile(double learning_rate = 1e-3) {
        optimizer = Adam(learning_rate);
        auto params = net.param_vals();
        optimizer.init(static_cast<int>(params.size()));
        sample_train_points();
    }

    double train_step() {
        net.sync_var_from_val();

        VarPtr total_loss = make_var(0.0);

        // ---- PDE residual loss ----
        if (pde_residual && train_pde_points.rows > 0) {
            VarPtr pde_loss = make_var(0.0);
            int n_pde = train_pde_points.rows;
            for (int i = 0; i < n_pde; ++i) {
                auto row = train_pde_points.row(i);
                std::vector<VarPtr> x_ad(row.size());
                for (size_t d = 0; d < row.size(); ++d)
                    x_ad[d] = make_var(row[d]);
                auto u_ad = net.forward_ad(x_ad);
                VarPtr res = pde_residual(x_ad, u_ad);
                pde_loss = pde_loss + ad::square(res);
            }
            pde_loss = pde_loss / static_cast<double>(n_pde);
            total_loss = total_loss + pde_loss * weight_pde;
        }

        // ---- BC loss ----
        if (!bcs.empty()) {
            VarPtr bc_loss = make_var(0.0);
            int bc_count = 0;
            for (size_t b = 0; b < bcs.size(); ++b) {
                const Matrix& bc_pts = bc_collocation[b];
                if (bc_pts.rows == 0) continue;
                // Use AD for BC too so gradients flow through the network
                for (int i = 0; i < bc_pts.rows; ++i) {
                    auto row = bc_pts.row(i);
                    std::vector<VarPtr> x_ad(row.size());
                    for (size_t d = 0; d < row.size(); ++d)
                        x_ad[d] = make_var(row[d]);
                    auto u_ad = net.forward_ad(x_ad);
                    // Compute BC error through AD
                    // For DirichletBC: error = u[comp] - func(x)
                    // We evaluate func(x) as a constant
                    auto x_vec = bc_pts.row(i);
                    // Get target from BC error (use the raw error function)
                    std::vector<std::vector<double>> out_vec = {net.forward(x_vec)};
                    auto errs = bcs[b]->error(
                        [&]() { Matrix m(1, bc_pts.cols); m.set_row(0, x_vec); return m; }(),
                        out_vec
                    );
                    // Use the error value but make it depend on the AD output
                    // so gradients flow. The error is u_pred - u_target.
                    // We construct: error_ad = u_ad[comp] - (u_ad[comp].val - err[0])
                    // which simplifies to make_var(err[0]) but linked to u_ad.
                    double target = u_ad[0]->val - errs[0];
                    VarPtr err_ad = u_ad[0] - make_var(target);
                    bc_loss = bc_loss + ad::square(err_ad);
                    bc_count++;
                }
            }
            if (bc_count > 0) {
                bc_loss = bc_loss / static_cast<double>(bc_count);
                total_loss = total_loss + bc_loss * weight_bc;
            }
        }

        // ---- IC loss ----
        if (!ics.empty()) {
            VarPtr ic_loss = make_var(0.0);
            int ic_count = 0;
            for (size_t c = 0; c < ics.size(); ++c) {
                const Matrix& ic_pts = ic_collocation[c];
                if (ic_pts.rows == 0) continue;
                for (int i = 0; i < ic_pts.rows; ++i) {
                    auto row = ic_pts.row(i);
                    std::vector<VarPtr> x_ad(row.size());
                    for (size_t d = 0; d < row.size(); ++d)
                        x_ad[d] = make_var(row[d]);
                    auto u_ad = net.forward_ad(x_ad);
                    double target_val = ics[c]->func(row);
                    VarPtr err_ad = u_ad[ics[c]->component] - make_var(target_val);
                    ic_loss = ic_loss + ad::square(err_ad);
                    ic_count++;
                }
            }
            if (ic_count > 0) {
                ic_loss = ic_loss / static_cast<double>(ic_count);
                total_loss = total_loss + ic_loss * weight_ic;
            }
        }

        // ---- Backward pass ----
        total_loss->backward();

        // ---- Collect gradients ----
        auto param_ptrs = net.param_vals();
        std::vector<double> grads(param_ptrs.size());

        int idx = 0;
        for (int l = 0; l < net.num_layers(); ++l) {
            for (size_t i = 0; i < net.weights[l].size(); ++i)
                grads[idx++] = net.weights[l][i]->grad;
            for (size_t i = 0; i < net.biases[l].size(); ++i)
                grads[idx++] = net.biases[l][i]->grad;
        }

        // ---- Optimizer step ----
        optimizer.step(param_ptrs, grads);

        return total_loss->val;
    }

    std::vector<TrainResult> train(int epochs, int display_every = 100,
                                   bool resample = false, int resample_every = 100) {
        std::vector<TrainResult> history;
        for (int ep = 0; ep < epochs; ++ep) {
            if (resample && ep > 0 && ep % resample_every == 0)
                sample_train_points();

            double loss = train_step();

            if (ep % display_every == 0 || ep == epochs - 1) {
                double pde_l = compute_pde_loss();
                double bc_l  = compute_bc_loss();
                double ic_l  = compute_ic_loss();

                std::cout << "Epoch " << std::setw(6) << ep
                          << "  loss=" << std::scientific << std::setprecision(4) << loss
                          << "  pde=" << pde_l
                          << "  bc=" << bc_l
                          << "  ic=" << ic_l
                          << std::defaultfloat << "\n";

                history.push_back({ep, loss, pde_l, bc_l, ic_l});
            }
        }
        return history;
    }

    Matrix predict(const Matrix& points) const {
        return net.forward_batch(points);
    }

    std::vector<double> predict(const std::vector<double>& point) const {
        return net.forward(point);
    }

private:
    double compute_pde_loss() {
        if (!pde_residual || train_pde_points.rows == 0) return 0.0;
        double mse = 0.0;
        int n = train_pde_points.rows;
        for (int i = 0; i < n; ++i) {
            auto x = train_pde_points.row(i);
            int xdim = static_cast<int>(x.size());
            std::vector<VarPtr> x_ad(xdim);
            for (int d = 0; d < xdim; ++d) x_ad[d] = make_var(x[d]);
            auto u_ad = net.forward_ad(x_ad);
            VarPtr res = pde_residual(x_ad, u_ad);
            mse += res->val * res->val;
        }
        return mse / n;
    }

    double compute_bc_loss() {
        double mse = 0.0;
        int count = 0;
        for (size_t b = 0; b < bcs.size(); ++b) {
            const Matrix& pts = bc_collocation[b];
            if (pts.rows == 0) continue;
            std::vector<std::vector<double>> outputs(pts.rows);
            for (int i = 0; i < pts.rows; ++i)
                outputs[i] = net.forward(pts.row(i));
            auto errors = bcs[b]->error(pts, outputs);
            for (auto e : errors) { mse += e * e; count++; }
        }
        return count > 0 ? mse / count : 0.0;
    }

    double compute_ic_loss() {
        double mse = 0.0;
        int count = 0;
        for (size_t c = 0; c < ics.size(); ++c) {
            const Matrix& pts = ic_collocation[c];
            if (pts.rows == 0) continue;
            std::vector<std::vector<double>> outputs(pts.rows);
            for (int i = 0; i < pts.rows; ++i)
                outputs[i] = net.forward(pts.row(i));
            auto errors = ics[c]->error(pts, outputs);
            for (auto e : errors) { mse += e * e; count++; }
        }
        return count > 0 ? mse / count : 0.0;
    }
};

// ============================================================================
// AD derivative helpers
// ============================================================================

// Compute d(u[output_comp])/d(x[spatial_dim]) via reverse-mode AD
inline VarPtr compute_du_dx_ad(const FNN& net, const std::vector<VarPtr>& x_vars,
                                int output_comp, int spatial_dim) {
    auto u = net.forward_ad(x_vars);
    u[output_comp]->backward();
    return make_var(x_vars[spatial_dim]->grad);
}

// Compute d^2(u[output_comp])/d(x[spatial_dim])^2 using central differences
// on the AD first derivative
inline VarPtr compute_d2u_dx2_ad(const FNN& net, const std::vector<VarPtr>& x_vars,
                                  int output_comp, int spatial_dim) {
    const double h = 1e-5;

    // Point x + h*e_j
    std::vector<VarPtr> xp(x_vars.size());
    for (size_t d = 0; d < x_vars.size(); ++d) {
        double v = x_vars[d]->val;
        if (static_cast<int>(d) == spatial_dim) v += h;
        xp[d] = make_var(v);
    }
    auto u_p = net.forward_ad(xp);
    u_p[output_comp]->backward();
    double du_dxj_p = xp[spatial_dim]->grad;

    // Point x - h*e_j
    std::vector<VarPtr> xm(x_vars.size());
    for (size_t d = 0; d < x_vars.size(); ++d) {
        double v = x_vars[d]->val;
        if (static_cast<int>(d) == spatial_dim) v -= h;
        xm[d] = make_var(v);
    }
    auto u_m = net.forward_ad(xm);
    u_m[output_comp]->backward();
    double du_dxj_m = xm[spatial_dim]->grad;

    double d2u = (du_dxj_p - du_dxj_m) / (2.0 * h);
    return make_var(d2u);
}

// Finite-difference second derivative (non-AD, faster)
inline double compute_d2u_dx2(const FNN& net, const std::vector<double>& x,
                               int output_comp, int spatial_dim, double h = 1e-5) {
    auto xp = x, xm = x;
    xp[spatial_dim] += h;
    xm[spatial_dim] -= h;
    double up = net.forward(xp)[output_comp];
    double u0 = net.forward(x)[output_comp];
    double um = net.forward(xm)[output_comp];
    return (up - 2.0 * u0 + um) / (h * h);
}

} // namespace deepxde

#endif // DEEPXDE_H
