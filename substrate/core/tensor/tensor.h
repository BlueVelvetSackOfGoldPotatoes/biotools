#pragma once
#include <vector>
#include <cstddef>
#include <cmath>
#include <random>
#include <functional>
#include <algorithm>
#include <numeric>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

// Always-active runtime check (never compiled out unlike assert)
#define TENSOR_CHECK(cond, msg) \
    do { if (!(cond)) throw std::runtime_error(std::string("Tensor: ") + (msg)); } while(0)

class Tensor {
public:
    std::vector<double> data;
    size_t rows = 0, cols = 0;

    // Constructors
    Tensor();
    Tensor(size_t rows, size_t cols);
    Tensor(size_t rows, size_t cols, double val);
    Tensor(size_t rows, size_t cols, const std::vector<double>& data);

    // Element access
    double& operator()(size_t i, size_t j);
    double operator()(size_t i, size_t j) const;
    double& at(size_t i); // flat access
    double at(size_t i) const;

    // Shape
    size_t size() const { return data.size(); }
    std::pair<size_t, size_t> shape() const { return {rows, cols}; }
    Tensor reshape(size_t new_rows, size_t new_cols) const;
    Tensor flatten() const; // to 1 x N
    Tensor slice_rows(size_t start, size_t count) const;
    Tensor slice_cols(size_t start, size_t count) const;
    Tensor row(size_t i) const;
    Tensor col(size_t j) const;
    void set_row(size_t i, const Tensor& row_data);

    // Matrix operations
    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;

    // Element-wise operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const; // Hadamard
    Tensor operator/(const Tensor& other) const;
    Tensor operator*(double scalar) const;
    Tensor operator/(double scalar) const;
    Tensor operator+(double scalar) const;
    Tensor operator-(double scalar) const;
    Tensor operator-() const; // unary minus

    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator*=(double scalar);
    Tensor& operator+=(double scalar);
    Tensor& operator-=(double scalar);

    // Reductions
    Tensor sum_rows() const;    // collapse rows → 1 x cols
    Tensor sum_cols() const;    // collapse cols → rows x 1
    double sum() const;
    double mean() const;
    double max_val() const;
    double min_val() const;
    size_t argmax() const;      // flat argmax
    Tensor argmax_rows() const; // per-row argmax → rows x 1
    Tensor max_rows() const;    // per-row max values → rows x 1

    // Math operations
    Tensor apply(const std::function<double(double)>& fn) const;
    Tensor square() const;
    Tensor sqrt_t() const;
    Tensor exp_t() const;
    Tensor log_t() const;
    Tensor abs_t() const;
    Tensor clamp(double lo, double hi) const;
    Tensor sign() const;
    Tensor pow_t(double p) const;

    // Broadcast add: add 1 x cols to every row
    Tensor add_row_broadcast(const Tensor& row_vec) const;

    // Initialization
    void fill_zeros();
    void fill_ones();
    void fill_val(double v);
    void fill_random_normal(double mean, double stddev, std::mt19937& rng);
    void fill_random_uniform(double lo, double hi, std::mt19937& rng);
    void fill_xavier(size_t fan_in, size_t fan_out, std::mt19937& rng);
    void fill_he(size_t fan_in, std::mt19937& rng);
    void fill_identity();

    // I/O
    void save_csv(const std::string& filepath) const;
    static Tensor load_csv(const std::string& filepath);
    void save_binary(const std::string& filepath) const;
    static Tensor load_binary(const std::string& filepath);
    void print(std::ostream& os = std::cout, int precision = 4) const;

    // Static constructors
    static Tensor zeros(size_t rows, size_t cols);
    static Tensor ones(size_t rows, size_t cols);
    static Tensor eye(size_t n);
    static Tensor randn(size_t rows, size_t cols, std::mt19937& rng);
    static Tensor rand(size_t rows, size_t cols, std::mt19937& rng);
    static Tensor from_vector(const std::vector<double>& v); // column vector
    static Tensor hstack(const Tensor& a, const Tensor& b);
    static Tensor vstack(const Tensor& a, const Tensor& b);

    // Comparison
    bool same_shape(const Tensor& other) const;
    double norm() const; // L2 norm
    double frobenius() const;
};

// Scalar * Tensor
inline Tensor operator*(double scalar, const Tensor& t) { return t * scalar; }
