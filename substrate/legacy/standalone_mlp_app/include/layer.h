#pragma once
#include "matrix.h"
#include <string>
#include <random>

class Layer {
public:
    virtual ~Layer() = default;
    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& grad_output) = 0;
    virtual Matrix& get_weights() = 0;
    virtual Matrix& get_biases() = 0;
    virtual Matrix& get_weight_gradients() = 0;
    virtual Matrix& get_bias_gradients() = 0;
    virtual bool has_parameters() const = 0;
    virtual std::string name() const = 0;
};

class DenseLayer : public Layer {
private:
    Matrix weights_;          // input_size x output_size
    Matrix biases_;           // 1 x output_size
    Matrix weight_grads_;
    Matrix bias_grads_;
    Matrix input_cache_;
    std::string name_;

public:
    DenseLayer(size_t input_size, size_t output_size,
               const std::string& name, std::mt19937& rng);

    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;

    Matrix& get_weights() override { return weights_; }
    Matrix& get_biases() override { return biases_; }
    Matrix& get_weight_gradients() override { return weight_grads_; }
    Matrix& get_bias_gradients() override { return bias_grads_; }
    bool has_parameters() const override { return true; }
    std::string name() const override { return name_; }
};

class ReLULayer : public Layer {
private:
    Matrix input_cache_;
    Matrix dummy_;

public:
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& grad_output) override;

    Matrix& get_weights() override { return dummy_; }
    Matrix& get_biases() override { return dummy_; }
    Matrix& get_weight_gradients() override { return dummy_; }
    Matrix& get_bias_gradients() override { return dummy_; }
    bool has_parameters() const override { return false; }
    std::string name() const override { return "ReLU"; }
};
