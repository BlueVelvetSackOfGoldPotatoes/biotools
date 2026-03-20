#include "loss.h"
#include <cmath>

double CrossEntropyLoss::forward(const Matrix& predictions, const Matrix& targets) {
    const double epsilon = 1e-12;
    double loss = 0.0;
    for (size_t i = 0; i < predictions.rows; ++i) {
        for (size_t j = 0; j < predictions.cols; ++j) {
            loss -= targets(i, j) * std::log(predictions(i, j) + epsilon);
        }
    }
    return loss / static_cast<double>(predictions.rows);
}

Matrix CrossEntropyLoss::backward(const Matrix& predictions, const Matrix& targets) {
    double n = static_cast<double>(predictions.rows);
    return (predictions - targets) * (1.0 / n);
}
