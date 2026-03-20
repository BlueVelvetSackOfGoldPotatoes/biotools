#pragma once
#include "matrix.h"

class CrossEntropyLoss {
public:
    // predictions: batch_size x num_classes (softmax output)
    // targets: batch_size x num_classes (one-hot)
    double forward(const Matrix& predictions, const Matrix& targets);

    // Combined softmax + cross-entropy gradient: (predictions - targets) / batch_size
    Matrix backward(const Matrix& predictions, const Matrix& targets);
};
