// BETSE C++ Port - Core Types and Constants
// Bioelectric Tissue Simulation Engine
// Original: https://github.com/betsee/betse
#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <unordered_map>
#include <map>
#include <functional>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <random>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <optional>
#include <variant>
#include <cstring>
#include <sstream>

namespace betse {

// ============================================================================
// Physical Constants (CODATA 2018)
// ============================================================================
constexpr double F_FARADAY    = 96485.33212;     // Faraday constant [C/mol]
constexpr double R_GAS        = 8.314462618;     // Gas constant [J/(mol*K)]
constexpr double K_BOLTZMANN  = 1.380649e-23;    // Boltzmann constant [J/K]
constexpr double Q_ELECTRON   = 1.602176634e-19; // Elementary charge [C]
constexpr double EPSILON_0    = 8.854187817e-12;  // Vacuum permittivity [F/m]
constexpr double FLOAT_NONCE  = 1.0e-25;         // Numerical stabilizer
constexpr double AVOGADRO     = 6.02214076e23;   // Avogadro number [1/mol]

// ============================================================================
// Ion identifiers
// ============================================================================
enum class Ion : int { Na = 0, K = 1, Cl = 2, Ca = 3, H = 4, P = 5, COUNT = 6 };

inline int ion_index(Ion i) { return static_cast<int>(i); }

inline const char* ion_name(Ion i) {
    static const char* names[] = {"Na", "K", "Cl", "Ca", "H", "P"};
    return names[ion_index(i)];
}

inline int ion_valence(Ion i) {
    static const int z[] = {1, 1, -1, 2, 1, -1};
    return z[ion_index(i)];
}

// ============================================================================
// Simulation Phase
// ============================================================================
enum class SimPhaseKind { SEED, INIT, SIM };

// ============================================================================
// Channel Type enumeration (all channel models from Python BETSE)
// ============================================================================
enum class ChannelType {
    // Sodium channels
    Nav1p2, Nav1p3, Nav1p6, NavRat1, NavRat2, NavRat3, NaLeak,
    // Potassium channels
    Kv1p1, Kv1p2, Kv1p3, Kv1p4, Kv1p5, Kv1p6,
    Kv2p1, Kv2p2, Kv3p1, Kv3p2, Kv3p3, Kv3p4,
    K_Fast, KLeak, Kir2p1,
    // Calcium channels
    Cav1p2, Cav1p3, Cav2p1, Cav2p2, Cav2p3, Cav3p1, Cav3p3,
    Ca_L2, Ca_L3, Cav_G, CaLeak,
    // HCN / Funny current
    HCN1, HCN2, HCN4, HCNLeak, HCN2_cAMP, HCN4_cAMP,
    // Chloride
    ClLeak,
    // Cation leak
    CatLeak, CatLeak2,
    // Morris-Lecar simplified models
    Kv_ML1, Kv2p1_ML, Kv1p3_ML, Kv1p5_ML, Kv1p5S_ML,
    Nav_ML, Cav_L_ML, Cav_L_ML2, Cav_N_ML, Cav_T_ML,
    Kir_ML, HCN2_ML, HCN4_ML,
    // Wound
    TRP
};

// Tissue profile shape
enum class TissueProfileShape { CIRCULAR, RECTANGULAR, BITMAP, ALL };

// ============================================================================
// 2D Vector
// ============================================================================
struct Vec2 {
    double x = 0.0, y = 0.0;
    Vec2() = default;
    Vec2(double x_, double y_) : x(x_), y(y_) {}
    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
    Vec2 operator/(double s) const { return {x / s, y / s}; }
    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
    Vec2& operator-=(const Vec2& o) { x -= o.x; y -= o.y; return *this; }
    Vec2& operator*=(double s) { x *= s; y *= s; return *this; }
    double dot(const Vec2& o) const { return x * o.x + y * o.y; }
    double cross(const Vec2& o) const { return x * o.y - y * o.x; }
    double norm() const { return std::sqrt(x * x + y * y); }
    double norm2() const { return x * x + y * y; }
    Vec2 normalized() const {
        double n = norm();
        return n > 0 ? Vec2{x/n, y/n} : Vec2{0, 0};
    }
    Vec2 perp() const { return {-y, x}; }
    static Vec2 from_polar(double r, double theta) {
        return {r * std::cos(theta), r * std::sin(theta)};
    }
};

inline Vec2 operator*(double s, const Vec2& v) { return v * s; }

// ============================================================================
// Dense Matrix (row-major)
// ============================================================================
struct DenseMatrix {
    int rows = 0, cols = 0;
    std::vector<double> data;

    DenseMatrix() = default;
    DenseMatrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}
    DenseMatrix(int r, int c, double val) : rows(r), cols(c), data(r * c, val) {}

    double& operator()(int i, int j) { return data[i * cols + j]; }
    const double& operator()(int i, int j) const { return data[i * cols + j]; }

    std::vector<double> multiply(const std::vector<double>& x) const {
        std::vector<double> y(rows, 0.0);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                y[i] += data[i * cols + j] * x[j];
        return y;
    }

    // LU solve Ax = b
    std::vector<double> solve(std::vector<double> b) const {
        int n = rows;
        std::vector<double> A(data);
        for (int k = 0; k < n; k++) {
            int max_row = k;
            double max_val = std::abs(A[k * n + k]);
            for (int i = k + 1; i < n; i++) {
                double v = std::abs(A[i * n + k]);
                if (v > max_val) { max_val = v; max_row = i; }
            }
            if (max_row != k) {
                for (int j = 0; j < n; j++)
                    std::swap(A[k * n + j], A[max_row * n + j]);
                std::swap(b[k], b[max_row]);
            }
            double akk = A[k * n + k];
            if (std::abs(akk) < 1e-30) akk = 1e-30;
            for (int i = k + 1; i < n; i++) {
                double factor = A[i * n + k] / akk;
                A[i * n + k] = factor;
                for (int j = k + 1; j < n; j++)
                    A[i * n + j] -= factor * A[k * n + j];
                b[i] -= factor * b[k];
            }
        }
        std::vector<double> x(n);
        for (int i = n - 1; i >= 0; i--) {
            x[i] = b[i];
            for (int j = i + 1; j < n; j++)
                x[i] -= A[i * n + j] * x[j];
            double aii = A[i * n + i];
            if (std::abs(aii) < 1e-30) aii = 1e-30;
            x[i] /= aii;
        }
        return x;
    }
};

// ============================================================================
// Sparse Matrix (CSR)
// ============================================================================
struct SparseMatrix {
    int rows = 0, cols = 0;
    std::vector<int>    row_ptr;
    std::vector<int>    col_idx;
    std::vector<double> values;

    SparseMatrix() = default;
    SparseMatrix(int r, int c) : rows(r), cols(c), row_ptr(r + 1, 0) {}

    std::vector<double> multiply(const std::vector<double>& x) const {
        std::vector<double> y(rows, 0.0);
        for (int i = 0; i < rows; i++)
            for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++)
                y[i] += values[k] * x[col_idx[k]];
        return y;
    }
};

// ============================================================================
// Vector math utilities
// ============================================================================
namespace vecutil {

inline double sum(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0);
}

inline double mean(const std::vector<double>& v) {
    return v.empty() ? 0.0 : sum(v) / (double)v.size();
}

inline double max_abs(const std::vector<double>& v) {
    double m = 0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}

inline void add_scaled(std::vector<double>& dst, const std::vector<double>& src, double s) {
    for (size_t i = 0; i < dst.size(); i++) dst[i] += src[i] * s;
}

inline void clamp_min(std::vector<double>& v, double lo) {
    for (auto& x : v) x = std::max(x, lo);
}

inline std::vector<double> elementwise_mul(const std::vector<double>& a,
                                            const std::vector<double>& b) {
    std::vector<double> r(a.size());
    for (size_t i = 0; i < a.size(); i++) r[i] = a[i] * b[i];
    return r;
}

inline std::vector<double> elementwise_div(const std::vector<double>& a,
                                            const std::vector<double>& b) {
    std::vector<double> r(a.size());
    for (size_t i = 0; i < a.size(); i++) r[i] = a[i] / (b[i] + FLOAT_NONCE);
    return r;
}

} // namespace vecutil


// ============================================================================
// Goldman flux equation (core function used by multiple modules)
// Moved here from betse.h so it can be shared by betse_networks.h etc.
// ============================================================================
inline double electroflux(double cA, double cB, double Dc, double d,
                          int zc, double vBA, double T, double rho = 1.0) {
    double vBA_s = vBA + FLOAT_NONCE;
    double zc_s  = zc + FLOAT_NONCE;
    double alpha = (zc_s * vBA_s * F_FARADAY) / (R_GAS * T);
    double exp_alpha = std::exp(-alpha);
    double denom = -std::expm1(-alpha);
    if (std::abs(denom) < 1e-30) denom = 1e-30;
    return -(Dc * alpha / d) * (cB - cA * exp_alpha) / denom * rho;
}

} // namespace betse
