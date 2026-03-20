#pragma once
#include "../tensor/tensor.h"
#include <vector>
#include <memory>
#include <string>

// Base class for all modules/layers
class Module {
public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;

    // Parameter access
    virtual std::vector<Tensor*> parameters() { return {}; }
    virtual std::vector<Tensor*> gradients() { return {}; }
    virtual std::vector<const Tensor*> parameters() const { return {}; }
    virtual std::vector<const Tensor*> gradients() const { return {}; }
    virtual bool has_parameters() const { return false; }
    virtual std::string name() const { return "Module"; }

    // Training mode
    bool training_ = true;
    virtual void train() { training_ = true; }
    virtual void eval() { training_ = false; }
};

// ============================================================
// Linear (fully connected) layer
// ============================================================
class Linear : public Module {
public:
    Tensor weights, biases;
    Tensor grad_weights, grad_biases;
    Tensor cached_input;
    std::string name_;
    size_t in_features, out_features;

    Linear(size_t in_features, size_t out_features, const std::string& name, std::mt19937& rng);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor*> parameters() override { return {&weights, &biases}; }
    std::vector<Tensor*> gradients() override { return {&grad_weights, &grad_biases}; }
    std::vector<const Tensor*> parameters() const override { return {&weights, &biases}; }
    std::vector<const Tensor*> gradients() const override { return {&grad_weights, &grad_biases}; }
    bool has_parameters() const override { return true; }
    std::string name() const override { return name_; }
};

// ============================================================
// Activation layers
// ============================================================
class ReLU : public Module {
    Tensor cached_input;
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "ReLU"; }
};

class Sigmoid : public Module {
    Tensor cached_output;
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "Sigmoid"; }
};

class Tanh_ : public Module {
    Tensor cached_output;
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "Tanh"; }
};

class GELU : public Module {
    Tensor cached_input;
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "GELU"; }
};

class LeakyReLU : public Module {
    Tensor cached_input;
    double alpha_;
public:
    LeakyReLU(double alpha = 0.01) : alpha_(alpha) {}
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "LeakyReLU"; }
};

class Softmax : public Module {
    Tensor cached_output;
public:
    Tensor forward(const Tensor& input) override; // row-wise softmax
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "Softmax"; }
};

// ============================================================
// Normalization layers
// ============================================================
class BatchNorm1d : public Module {
public:
    size_t num_features;
    Tensor gamma, beta;
    Tensor grad_gamma, grad_beta;
    Tensor running_mean, running_var;
    double momentum_, eps_;
    Tensor cached_input_norm, cached_std, cached_input;

    BatchNorm1d(size_t num_features, double momentum = 0.1, double eps = 1e-5);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor*> parameters() override { return {&gamma, &beta}; }
    std::vector<Tensor*> gradients() override { return {&grad_gamma, &grad_beta}; }
    std::vector<const Tensor*> parameters() const override { return {&gamma, &beta}; }
    std::vector<const Tensor*> gradients() const override { return {&grad_gamma, &grad_beta}; }
    bool has_parameters() const override { return true; }
    std::string name() const override { return "BatchNorm1d"; }
};

// Per-channel batch normalization for conv layers.
// Input:  (batch, C*H*W)   — C channels, each H*W spatial positions
// Output: (batch, C*H*W)   — normalized per channel (mean/var across batch and H*W)
class BatchNorm2d : public Module {
public:
    size_t channels_, h_, w_;
    Tensor gamma, beta;              // (1, C)
    Tensor grad_gamma, grad_beta;    // (1, C)
    Tensor running_mean, running_var; // (1, C)
    double momentum_, eps_;
    Tensor cached_input_norm;        // (N, C*H*W) — normalized input
    Tensor cached_std;               // (1, C)     — sqrt(var + eps)

    BatchNorm2d(size_t channels, size_t h, size_t w, double momentum = 0.1, double eps = 1e-5);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor*> parameters() override { return {&gamma, &beta}; }
    std::vector<Tensor*> gradients() override { return {&grad_gamma, &grad_beta}; }
    std::vector<const Tensor*> parameters() const override { return {&gamma, &beta}; }
    std::vector<const Tensor*> gradients() const override { return {&grad_gamma, &grad_beta}; }
    bool has_parameters() const override { return true; }
    std::string name() const override { return "BatchNorm2d"; }
};

class LayerNorm : public Module {
public:
    size_t normalized_size;
    Tensor gamma, beta;
    Tensor grad_gamma, grad_beta;
    double eps_;
    Tensor cached_input_norm, cached_std, cached_input;

    LayerNorm(size_t normalized_size, double eps = 1e-5);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor*> parameters() override { return {&gamma, &beta}; }
    std::vector<Tensor*> gradients() override { return {&grad_gamma, &grad_beta}; }
    std::vector<const Tensor*> parameters() const override { return {&gamma, &beta}; }
    std::vector<const Tensor*> gradients() const override { return {&grad_gamma, &grad_beta}; }
    bool has_parameters() const override { return true; }
    std::string name() const override { return "LayerNorm"; }
};

// ============================================================
// Dropout
// ============================================================
class Dropout : public Module {
    double rate_;
    Tensor mask_;
    std::mt19937* rng_;
    bool applied_mask_ = false; // whether mask was applied in last forward
public:
    Dropout(double rate, std::mt19937& rng) : rate_(rate), rng_(&rng) {}
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "Dropout"; }
};

// ============================================================
// Embedding layer
// ============================================================
class Embedding : public Module {
public:
    Tensor weights;
    Tensor grad_weights;
    std::vector<size_t> cached_indices;
    size_t vocab_size, embed_dim;
    std::string name_;

    Embedding(size_t vocab_size, size_t embed_dim, const std::string& name, std::mt19937& rng);
    // Input: batch x seq_len (token indices as doubles)
    // Output: (batch * seq_len) x embed_dim
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor*> parameters() override { return {&weights}; }
    std::vector<Tensor*> gradients() override { return {&grad_weights}; }
    std::vector<const Tensor*> parameters() const override { return {&weights}; }
    std::vector<const Tensor*> gradients() const override { return {&grad_weights}; }
    bool has_parameters() const override { return true; }
    std::string name() const override { return name_; }
};

// ============================================================
// Conv2D (im2col approach)
// Expects input as batch of flattened images: batch x (C*H*W)
// with explicit C, H, W stored for reshaping
// ============================================================
class Conv2D : public Module {
public:
    size_t in_channels, out_channels, kernel_size, stride, padding;
    size_t in_h, in_w; // input spatial dims
    Tensor weights; // out_channels x (in_channels * kernel_size * kernel_size)
    Tensor biases;  // 1 x out_channels
    Tensor grad_weights, grad_biases;
    Tensor cached_col; // im2col result
    size_t cached_batch;
    std::string name_;

    Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size,
           size_t in_h, size_t in_w, size_t stride, size_t padding,
           const std::string& name, std::mt19937& rng);

    size_t out_h() const { return (in_h + 2 * padding - kernel_size) / stride + 1; }
    size_t out_w() const { return (in_w + 2 * padding - kernel_size) / stride + 1; }

    Tensor forward(const Tensor& input) override; // batch x (C*H*W) -> batch x (Cout*OH*OW)
    Tensor backward(const Tensor& grad_output) override;

    std::vector<Tensor*> parameters() override { return {&weights, &biases}; }
    std::vector<Tensor*> gradients() override { return {&grad_weights, &grad_biases}; }
    std::vector<const Tensor*> parameters() const override { return {&weights, &biases}; }
    std::vector<const Tensor*> gradients() const override { return {&grad_weights, &grad_biases}; }
    bool has_parameters() const override { return true; }
    std::string name() const override { return name_; }

    // im2col / col2im
    Tensor im2col(const Tensor& input, size_t batch_size) const;
    Tensor col2im(const Tensor& col, size_t batch_size) const;
};

// ============================================================
// MaxPool2D
// ============================================================
class MaxPool2D : public Module {
public:
    size_t pool_size, stride, channels, in_h, in_w;
    Tensor cached_max_indices;
    Tensor cached_input;

    MaxPool2D(size_t pool_size, size_t stride, size_t channels, size_t in_h, size_t in_w);

    size_t out_h() const { return (in_h - pool_size) / stride + 1; }
    size_t out_w() const { return (in_w - pool_size) / stride + 1; }

    Tensor forward(const Tensor& input) override; // batch x (C*H*W) -> batch x (C*OH*OW)
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "MaxPool2D"; }
};

// ============================================================
// AvgPool2D
// ============================================================
class AvgPool2D : public Module {
public:
    size_t pool_size, stride, channels, in_h, in_w;
    Tensor cached_input;

    AvgPool2D(size_t pool_size, size_t stride, size_t channels, size_t in_h, size_t in_w);

    size_t out_h() const { return (in_h - pool_size) / stride + 1; }
    size_t out_w() const { return (in_w - pool_size) / stride + 1; }

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "AvgPool2D"; }
};

// ============================================================
// Flatten (for CNN -> FC transition)
// ============================================================
class Flatten : public Module {
    size_t cached_rows, cached_cols;
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "Flatten"; }
};

// ============================================================
// Sequential container
// ============================================================
class Sequential : public Module {
public:
    std::vector<std::unique_ptr<Module>> layers;

    void add(std::unique_ptr<Module> layer);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::vector<const Tensor*> parameters() const override;
    std::vector<const Tensor*> gradients() const override;
    bool has_parameters() const override;
    std::string name() const override { return "Sequential"; }

    Module& get(size_t idx) { return *layers[idx]; }
    const Module& get(size_t idx) const { return *layers[idx]; }
    size_t size() const { return layers.size(); }

    void train() override;
    void eval() override;
};
