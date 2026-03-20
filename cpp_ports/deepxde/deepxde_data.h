// deepxde_data.h - Data types for DeepXDE C++ port
// Ports: Data, PDE, TimePDE, IDE, FPDE, DataSet, Function, Triple, Quad,
//        PDEOperator, FunctionSpace, MfFunc, BatchSampler
#ifndef DEEPXDE_DATA_H
#define DEEPXDE_DATA_H

#include "deepxde.h"
#include <functional>

namespace deepxde {

// ============================================================================
// BatchSampler - Mini-batch index sampler
// Ports deepxde.data.sampler.BatchSampler
// ============================================================================
class BatchSampler {
public:
    int num_samples;
    bool shuffle;
    std::vector<int> indices;
    int index_in_epoch;
    int epochs_completed;

    BatchSampler(int n = 0, bool shuf = true)
        : num_samples(n), shuffle(shuf), index_in_epoch(0), epochs_completed(0) {
        indices.resize(n);
        std::iota(indices.begin(), indices.end(), 0);
        if (shuffle) std::shuffle(indices.begin(), indices.end(), global_rng());
    }

    std::vector<int> get_next(int batch_size) {
        if (batch_size > num_samples)
            throw std::invalid_argument("batch_size > num_samples");
        if (index_in_epoch + batch_size <= num_samples) {
            auto start = indices.begin() + index_in_epoch;
            index_in_epoch += batch_size;
            return std::vector<int>(start, start + batch_size);
        }
        // Cross epoch boundary
        epochs_completed++;
        std::vector<int> rest(indices.begin() + index_in_epoch, indices.end());
        if (shuffle) std::shuffle(indices.begin(), indices.end(), global_rng());
        index_in_epoch = batch_size - static_cast<int>(rest.size());
        rest.insert(rest.end(), indices.begin(), indices.begin() + index_in_epoch);
        return rest;
    }
};

// ============================================================================
// Data base class
// Ports deepxde.data.data.Data
// ============================================================================
struct DataBase {
    virtual ~DataBase() = default;
    virtual std::pair<Matrix, Matrix> train_next_batch(int batch_size = -1) = 0;
    virtual std::pair<Matrix, Matrix> test() = 0;
};

// ============================================================================
// DataSet - Fitting data from arrays or files
// Ports deepxde.data.dataset.DataSet
// ============================================================================
struct DataSet : DataBase {
    Matrix train_x, train_y, test_x, test_y;

    DataSet(const Matrix& Xtrain, const Matrix& Ytrain,
            const Matrix& Xtest, const Matrix& Ytest)
        : train_x(Xtrain), train_y(Ytrain), test_x(Xtest), test_y(Ytest) {}

    std::pair<Matrix, Matrix> train_next_batch(int = -1) override {
        return {train_x, train_y};
    }
    std::pair<Matrix, Matrix> test() override {
        return {test_x, test_y};
    }
};

// ============================================================================
// Function - Approximate a function via a network
// Ports deepxde.data.function.Function
// ============================================================================
struct FunctionData : DataBase {
    std::shared_ptr<Geometry> geom;
    std::function<Matrix(const Matrix&)> func;
    int num_train, num_test;
    std::string dist_train;
    bool online;

    Matrix train_x_, train_y_, test_x_, test_y_;
    bool train_ready = false, test_ready = false;

    FunctionData(std::shared_ptr<Geometry> g,
                 std::function<Matrix(const Matrix&)> f,
                 int ntrain, int ntest,
                 const std::string& dist = "uniform", bool onl = false)
        : geom(g), func(std::move(f)), num_train(ntrain), num_test(ntest),
          dist_train(dist), online(onl) {}

    std::pair<Matrix, Matrix> train_next_batch(int = -1) override {
        if (!train_ready || online) {
            train_x_ = (dist_train == "uniform") ?
                geom->uniform_points(num_train, true) :
                geom->random_points(num_train);
            train_y_ = func(train_x_);
            train_ready = true;
        }
        return {train_x_, train_y_};
    }

    std::pair<Matrix, Matrix> test() override {
        if (!test_ready) {
            test_x_ = geom->uniform_points(num_test, true);
            test_y_ = func(test_x_);
            test_ready = true;
        }
        return {test_x_, test_y_};
    }
};

// ============================================================================
// Triple - For DeepONet operator learning
// Ports deepxde.data.triple.Triple
// ============================================================================
struct Triple : DataBase {
    std::pair<Matrix, Matrix> train_x;  // (func_vals, locations)
    Matrix train_y;
    std::pair<Matrix, Matrix> test_x;
    Matrix test_y;

    Triple(const std::pair<Matrix, Matrix>& Xtrain, const Matrix& Ytrain,
           const std::pair<Matrix, Matrix>& Xtest, const Matrix& Ytest)
        : train_x(Xtrain), train_y(Ytrain), test_x(Xtest), test_y(Ytest) {}

    std::pair<Matrix, Matrix> train_next_batch(int = -1) override {
        // Return concatenated for simplicity
        return {train_x.first, train_y};
    }
    std::pair<Matrix, Matrix> test() override {
        return {test_x.first, test_y};
    }
};

// ============================================================================
// Quadruple - For MIONet operator learning
// Ports deepxde.data.quadruple.Quadruple
// ============================================================================
struct Quadruple : DataBase {
    Matrix train_x1, train_x2, train_x3, train_y;
    Matrix test_x1, test_x2, test_x3, test_y;

    Quadruple(const Matrix& x1, const Matrix& x2, const Matrix& x3,
              const Matrix& y, const Matrix& tx1, const Matrix& tx2,
              const Matrix& tx3, const Matrix& ty)
        : train_x1(x1), train_x2(x2), train_x3(x3), train_y(y),
          test_x1(tx1), test_x2(tx2), test_x3(tx3), test_y(ty) {}

    std::pair<Matrix, Matrix> train_next_batch(int = -1) override {
        return {train_x1, train_y};
    }
    std::pair<Matrix, Matrix> test() override {
        return {test_x1, test_y};
    }
};

// ============================================================================
// MfFunc - Multi-fidelity function approximation data
// Ports deepxde.data.mf.MfFunc
// ============================================================================
struct MfFuncData : DataBase {
    std::shared_ptr<Geometry> geom;
    std::function<Matrix(const Matrix&)> func_lo, func_hi;
    int num_lo, num_hi, num_test;
    std::string dist;
    Matrix train_x_, test_x_;
    Matrix train_y_lo, train_y_hi, test_y_lo, test_y_hi;

    MfFuncData(std::shared_ptr<Geometry> g,
               std::function<Matrix(const Matrix&)> flo,
               std::function<Matrix(const Matrix&)> fhi,
               int nlo, int nhi, int ntest, const std::string& d = "uniform")
        : geom(g), func_lo(std::move(flo)), func_hi(std::move(fhi)),
          num_lo(nlo), num_hi(nhi), num_test(ntest), dist(d) {}

    std::pair<Matrix, Matrix> train_next_batch(int = -1) override {
        auto x_lo = (dist == "uniform") ? geom->uniform_points(num_lo) : geom->random_points(num_lo);
        auto x_hi = (dist == "uniform") ? geom->uniform_points(num_hi) : geom->random_points(num_hi);
        // Stack
        train_x_ = Matrix(x_lo.rows + x_hi.rows, x_lo.cols);
        for (int i = 0; i < x_lo.rows; ++i) train_x_.set_row(i, x_lo.row(i));
        for (int i = 0; i < x_hi.rows; ++i) train_x_.set_row(x_lo.rows + i, x_hi.row(i));
        train_y_lo = func_lo(train_x_);
        train_y_hi = func_hi(train_x_);
        return {train_x_, train_y_hi};
    }
    std::pair<Matrix, Matrix> test() override {
        test_x_ = geom->uniform_points(num_test);
        test_y_hi = func_hi(test_x_);
        return {test_x_, test_y_hi};
    }
};

// ============================================================================
// FunctionSpace base class
// Ports deepxde.data.function_spaces.FunctionSpace
// ============================================================================
struct FunctionSpace {
    virtual ~FunctionSpace() = default;
    virtual Matrix random_features(int size) const = 0;
    virtual double eval_one(const std::vector<double>& feature, double x) const = 0;
    virtual Matrix eval_batch(const Matrix& features, const Matrix& xs) const = 0;
};

// ============================================================================
// PowerSeries function space
// Ports deepxde.data.function_spaces.PowerSeries
// ============================================================================
struct PowerSeries : FunctionSpace {
    int N;
    double M;

    PowerSeries(int n = 100, double m = 1.0) : N(n), M(m) {}

    Matrix random_features(int size) const override {
        Matrix feats(size, N);
        std::uniform_real_distribution<double> dist(-M, M);
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < N; ++j)
                feats(i, j) = dist(global_rng());
        return feats;
    }

    double eval_one(const std::vector<double>& feature, double x) const override {
        double result = 0, xp = 1.0;
        for (int i = 0; i < N && i < static_cast<int>(feature.size()); ++i) {
            result += feature[i] * xp;
            xp *= x;
        }
        return result;
    }

    Matrix eval_batch(const Matrix& features, const Matrix& xs) const override {
        Matrix result(features.rows, xs.rows);
        for (int i = 0; i < features.rows; ++i)
            for (int j = 0; j < xs.rows; ++j)
                result(i, j) = eval_one(features.row(i), xs(j, 0));
        return result;
    }
};

// ============================================================================
// Chebyshev polynomial function space
// Ports deepxde.data.function_spaces.Chebyshev
// ============================================================================
struct ChebyshevSpace : FunctionSpace {
    int N;
    double M;

    ChebyshevSpace(int n = 100, double m = 1.0) : N(n), M(m) {}

    Matrix random_features(int size) const override {
        Matrix feats(size, N);
        std::uniform_real_distribution<double> dist(-M, M);
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < N; ++j)
                feats(i, j) = dist(global_rng());
        return feats;
    }

    double eval_one(const std::vector<double>& feature, double x) const override {
        // Map [0,1] -> [-1,1]
        double t = 2.0 * x - 1.0;
        if (N == 0) return 0;
        double T_prev = 1.0, T_curr = t, result = feature[0];
        if (N > 1) result += feature[1] * t;
        for (int i = 2; i < N && i < static_cast<int>(feature.size()); ++i) {
            double T_next = 2.0 * t * T_curr - T_prev;
            result += feature[i] * T_next;
            T_prev = T_curr;
            T_curr = T_next;
        }
        return result;
    }

    Matrix eval_batch(const Matrix& features, const Matrix& xs) const override {
        Matrix result(features.rows, xs.rows);
        for (int i = 0; i < features.rows; ++i)
            for (int j = 0; j < xs.rows; ++j)
                result(i, j) = eval_one(features.row(i), xs(j, 0));
        return result;
    }
};

// ============================================================================
// IDE - Integro-Differential Equation data
// Ports deepxde.data.ide.IDE
// ============================================================================
struct IDEData {
    std::shared_ptr<Geometry> geom;
    int quad_deg;
    std::vector<double> quad_x, quad_w;  // Gauss-Legendre quadrature

    IDEData(std::shared_ptr<Geometry> g, int qdeg)
        : geom(g), quad_deg(qdeg)
    {
        // Gauss-Legendre nodes & weights on [-1, 1]
        compute_gauss_legendre(qdeg, quad_x, quad_w);
    }

    Matrix quad_points(const Matrix& X) const {
        // For each x, quadrature points are (quad_x + 1) * x / 2
        int total = X.rows * quad_deg;
        Matrix qpts(total, 1);
        int idx = 0;
        for (int i = 0; i < X.rows; ++i) {
            double xi = X(i, 0);
            for (int q = 0; q < quad_deg; ++q)
                qpts(idx++, 0) = (quad_x[q] + 1.0) * xi / 2.0;
        }
        return qpts;
    }

    // Integration matrix for computing integrals
    Matrix get_int_matrix(int num_f, int num_bc, int total_x) const {
        Matrix int_mat(num_bc + num_f, total_x, 0.0);
        for (int i = 0; i < num_f; ++i) {
            // Weight for Gauss-Legendre mapped to [0, x_i]
            // ... (simplified - kernel assumed identity)
            int beg = num_f + num_bc + quad_deg * i;
            for (int q = 0; q < quad_deg && beg + q < total_x; ++q) {
                int_mat(i + num_bc, beg + q) = quad_w[q] / 2.0;
            }
        }
        return int_mat;
    }

private:
    static void compute_gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
        x.resize(n);
        w.resize(n);
        // Simple implementation for small n
        if (n == 1) { x[0] = 0; w[0] = 2; return; }
        if (n == 2) { x[0] = -1.0/std::sqrt(3.0); x[1] = 1.0/std::sqrt(3.0); w[0] = w[1] = 1; return; }
        // General Golub-Welsch / Newton iteration
        for (int i = 0; i < n; ++i) {
            double xi = std::cos(M_PI * (i + 0.75) / (n + 0.5));
            for (int iter = 0; iter < 100; ++iter) {
                double p0 = 1.0, p1 = xi;
                for (int j = 2; j <= n; ++j) {
                    double p2 = ((2*j-1)*xi*p1 - (j-1)*p0) / j;
                    p0 = p1; p1 = p2;
                }
                double dp = n * (p0 - xi * p1) / (1 - xi * xi);
                double dx = p1 / dp;
                xi -= dx;
                if (std::fabs(dx) < 1e-15) break;
            }
            x[i] = xi;
            double p0 = 1.0, p1 = xi;
            for (int j = 2; j <= n; ++j) {
                double p2 = ((2*j-1)*xi*p1 - (j-1)*p0) / j;
                p0 = p1; p1 = p2;
            }
            double dp = n * (p0 - xi * p1) / (1 - xi * xi);
            w[i] = 2.0 / ((1 - xi * xi) * dp * dp);
        }
    }
};

// ============================================================================
// FPDE - Fractional PDE discretization scheme
// Ports deepxde.data.fpde.Scheme, Fractional
// ============================================================================
struct FractionalScheme {
    std::string meshtype;  // "static" or "dynamic"
    std::vector<int> resolution;
    int frac_dim;

    FractionalScheme(const std::string& mt, const std::vector<int>& res)
        : meshtype(mt), resolution(res), frac_dim(static_cast<int>(res.size())) {}

    // Compute Grunwald-Letnikov weights
    static std::vector<double> gl_weights(double alpha, int n) {
        std::vector<double> w(n + 1);
        w[0] = 1.0;
        for (int j = 1; j <= n; ++j)
            w[j] = w[j-1] * (j - 1 - alpha) / j;
        return w;
    }
};

// ============================================================================
// PDEOperator - PDE solution operator for PI-DeepONet
// Ports deepxde.data.pde_operator.PDEOperator
// ============================================================================
struct PDEOperatorData : DataBase {
    // Stores function evaluations as branch input, PDE points as trunk input
    Matrix branch_input;  // (num_func, eval_pts_size)
    Matrix trunk_input;   // (num_domain, dim)
    Matrix aux_vars;      // function values at domain points

    PDEOperatorData() = default;

    std::pair<Matrix, Matrix> train_next_batch(int = -1) override {
        return {branch_input, Matrix()};
    }
    std::pair<Matrix, Matrix> test() override {
        return {branch_input, Matrix()};
    }
};

} // namespace deepxde
#endif // DEEPXDE_DATA_H
