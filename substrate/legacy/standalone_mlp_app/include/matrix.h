#pragma once
#include <vector>
#include <string>
#include <cstddef>
#include <functional>
#include <iostream>
#include <random>

class Matrix {
public:
    size_t rows;
    size_t cols;
    std::vector<double> data; // row-major: element (i,j) = data[i * cols + j]

    Matrix();
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double val);

    double& operator()(size_t i, size_t j);
    double  operator()(size_t i, size_t j) const;

    // Matrix multiplication (this * other)
    Matrix matmul(const Matrix& other) const;
    Matrix transpose() const;

    // Element-wise operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const; // Hadamard product
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;

    // In-place operations
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double scalar);

    // Reductions
    Matrix sum_rows() const;   // collapse rows -> 1 x cols
    double sum() const;

    // Element-wise function application
    Matrix apply(std::function<double(double)> fn) const;

    // Initialization
    void fill_random_normal(double mean, double stddev, std::mt19937& rng);
    void fill_zeros();

    // I/O
    void save_csv(const std::string& filepath) const;
    void print(std::ostream& os = std::cout) const;

    // Utilities
    Matrix slice_rows(size_t start, size_t count) const;
};
