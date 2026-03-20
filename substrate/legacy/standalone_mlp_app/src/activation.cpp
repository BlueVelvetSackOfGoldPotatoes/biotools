#include "activation.h"
#include <cmath>
#include <algorithm>

Matrix Activation::relu(const Matrix& z) {
    return z.apply([](double x) { return x > 0.0 ? x : 0.0; });
}

Matrix Activation::relu_derivative(const Matrix& z) {
    return z.apply([](double x) { return x > 0.0 ? 1.0 : 0.0; });
}

Matrix Activation::softmax(const Matrix& z) {
    Matrix result(z.rows, z.cols);
    for (size_t i = 0; i < z.rows; ++i) {
        // Numerical stability: subtract row max before exp
        double row_max = z(i, 0);
        for (size_t j = 1; j < z.cols; ++j) {
            if (z(i, j) > row_max) row_max = z(i, j);
        }
        double sum = 0.0;
        for (size_t j = 0; j < z.cols; ++j) {
            result(i, j) = std::exp(z(i, j) - row_max);
            sum += result(i, j);
        }
        for (size_t j = 0; j < z.cols; ++j) {
            result(i, j) /= sum;
        }
    }
    return result;
}
