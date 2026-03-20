#include "layer.h"
#include "activation.h"
#include <cmath>

// --- DenseLayer ---

DenseLayer::DenseLayer(size_t input_size, size_t output_size,
                       const std::string& name, std::mt19937& rng)
    : weights_(input_size, output_size),
      biases_(1, output_size),
      weight_grads_(input_size, output_size),
      bias_grads_(1, output_size),
      name_(name)
{
    // He initialization: N(0, sqrt(2 / fan_in))
    double stddev = std::sqrt(2.0 / static_cast<double>(input_size));
    weights_.fill_random_normal(0.0, stddev, rng);
    biases_.fill_zeros();
}

Matrix DenseLayer::forward(const Matrix& input) {
    input_cache_ = input;
    // output = input * weights + biases (broadcast biases across batch)
    Matrix z = input.matmul(weights_);
    for (size_t i = 0; i < z.rows; ++i) {
        for (size_t j = 0; j < z.cols; ++j) {
            z(i, j) += biases_(0, j);
        }
    }
    return z;
}

Matrix DenseLayer::backward(const Matrix& grad_output) {
    // dL/dW = input^T * grad_output
    weight_grads_ = input_cache_.transpose().matmul(grad_output);

    // dL/db = sum of grad_output across batch dimension
    bias_grads_ = grad_output.sum_rows();

    // dL/d(input) = grad_output * W^T
    return grad_output.matmul(weights_.transpose());
}

// --- ReLULayer ---

Matrix ReLULayer::forward(const Matrix& input) {
    input_cache_ = input;
    return Activation::relu(input);
}

Matrix ReLULayer::backward(const Matrix& grad_output) {
    // Element-wise multiply by ReLU derivative mask
    Matrix mask = Activation::relu_derivative(input_cache_);
    return grad_output * mask;
}
