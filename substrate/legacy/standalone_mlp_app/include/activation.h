#pragma once
#include "matrix.h"

namespace Activation {
    Matrix relu(const Matrix& z);
    Matrix relu_derivative(const Matrix& z);
    Matrix softmax(const Matrix& z); // row-wise
}
