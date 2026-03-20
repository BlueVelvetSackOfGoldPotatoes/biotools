#include "matrix.h"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <stdexcept>

Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(size_t rows, size_t cols)
    : rows(rows), cols(cols), data(rows * cols, 0.0) {}

Matrix::Matrix(size_t rows, size_t cols, double val)
    : rows(rows), cols(cols), data(rows * cols, val) {}

double& Matrix::operator()(size_t i, size_t j) {
    return data[i * cols + j];
}

double Matrix::operator()(size_t i, size_t j) const {
    return data[i * cols + j];
}

Matrix Matrix::matmul(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::runtime_error("Matrix::matmul shape mismatch");
    }
    Matrix result(rows, other.cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double a_ij = data[i * cols + j];
            for (size_t k = 0; k < other.cols; ++k) {
                result.data[i * other.cols + k] += a_ij * other.data[j * other.cols + k];
            }
        }
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[j * rows + i] = data[i * cols + j];
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::runtime_error("Matrix::operator+ shape mismatch");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::runtime_error("Matrix::operator- shape mismatch");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::runtime_error("Matrix::operator* elementwise shape mismatch");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    Matrix result(rows, cols);
    double inv = 1.0 / scalar;
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * inv;
    }
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::runtime_error("Matrix::operator+= shape mismatch");
    }
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::runtime_error("Matrix::operator-= shape mismatch");
    }
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= other.data[i];
    }
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] *= scalar;
    }
    return *this;
}

Matrix Matrix::sum_rows() const {
    Matrix result(1, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[j] += data[i * cols + j];
        }
    }
    return result;
}

double Matrix::sum() const {
    double s = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        s += data[i];
    }
    return s;
}

Matrix Matrix::apply(std::function<double(double)> fn) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = fn(data[i]);
    }
    return result;
}

void Matrix::fill_random_normal(double mean, double stddev, std::mt19937& rng) {
    std::normal_distribution<double> dist(mean, stddev);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(rng);
    }
}

void Matrix::fill_zeros() {
    std::fill(data.begin(), data.end(), 0.0);
}

void Matrix::save_csv(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (j > 0) file << ",";
            file << std::setprecision(8) << data[i * cols + j];
        }
        file << "\n";
    }
}

void Matrix::print(std::ostream& os) const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (j > 0) os << " ";
            os << std::setw(10) << std::setprecision(4) << data[i * cols + j];
        }
        os << "\n";
    }
}

Matrix Matrix::slice_rows(size_t start, size_t count) const {
    if (start > rows || count > rows - start) {
        throw std::runtime_error("Matrix::slice_rows out of bounds");
    }
    Matrix result(count, cols);
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i * cols + j] = data[(start + i) * cols + j];
        }
    }
    return result;
}
