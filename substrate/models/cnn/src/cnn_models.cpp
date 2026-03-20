#include "cnn_models.h"
#include <iostream>

// ============================================================
// GlobalAvgPool2D implementation
// Input:  batch x (C * H * W)
// Output: batch x C
// Each output channel is the mean of all H*W spatial values.
// ============================================================

GlobalAvgPool2D::GlobalAvgPool2D(size_t c, size_t h, size_t w)
    : channels_(c), h_(h), w_(w), cached_batch_(0) {}

Tensor GlobalAvgPool2D::forward(const Tensor& input) {
    cached_batch_ = input.rows;
    size_t spatial = h_ * w_;
    Tensor output(cached_batch_, channels_);

    for (size_t b = 0; b < cached_batch_; ++b) {
        for (size_t c = 0; c < channels_; ++c) {
            double sum = 0.0;
            size_t offset = c * spatial;
            for (size_t s = 0; s < spatial; ++s) {
                sum += input(b, offset + s);
            }
            output(b, c) = sum / (double)spatial;
        }
    }
    return output;
}

Tensor GlobalAvgPool2D::backward(const Tensor& grad_output) {
    size_t spatial = h_ * w_;
    Tensor grad_input(cached_batch_, channels_ * spatial);

    for (size_t b = 0; b < cached_batch_; ++b) {
        for (size_t c = 0; c < channels_; ++c) {
            double grad_val = grad_output(b, c) / (double)spatial;
            size_t offset = c * spatial;
            for (size_t s = 0; s < spatial; ++s) {
                grad_input(b, offset + s) = grad_val;
            }
        }
    }
    return grad_input;
}

// ============================================================
// ResBlock implementation
// Two conv layers with BN and ReLU, plus a skip connection.
// conv1: in_ch -> out_ch, 3x3, pad=1 (preserves spatial dims)
// conv2: out_ch -> out_ch, 3x3, pad=1 (preserves spatial dims)
// If in_ch != out_ch, a 1x1 conv maps the skip connection.
// ============================================================

ResBlock::ResBlock(size_t in_ch, size_t out_ch, size_t h, size_t w, std::mt19937& rng)
    : conv1_(in_ch, out_ch, 3, h, w, 1, 1, "resblock_conv1", rng),
      conv2_(out_ch, out_ch, 3, h, w, 1, 1, "resblock_conv2", rng),
      bn1_(out_ch, h, w),
      bn2_(out_ch, h, w),
      channels_in_(in_ch), channels_out_(out_ch), h_(h), w_(w) {
    if (in_ch != out_ch) {
        // 1x1 convolution to match channel dimensions on the skip path
        downsample_ = std::make_unique<Conv2D>(
            in_ch, out_ch, 1, h, w, 1, 0, "resblock_downsample", rng
        );
        downsample_bn_ = std::make_unique<BatchNorm2d>(out_ch, h, w);
    }
}

Tensor ResBlock::forward(const Tensor& input) {
    // Main path: conv1 -> bn1 -> relu1 -> conv2 -> bn2
    Tensor out = conv1_.forward(input);
    out = bn1_.forward(out);
    out = relu1_.forward(out);
    out = conv2_.forward(out);
    out = bn2_.forward(out);

    // Skip path
    Tensor skip;
    if (downsample_) {
        skip = downsample_->forward(input);
        skip = downsample_bn_->forward(skip);
    } else {
        skip = input;
    }

    // Element-wise addition: residual + skip
    Tensor sum = out + skip;

    // Final activation
    return relu2_.forward(sum);
}

Tensor ResBlock::backward(const Tensor& grad_output) {
    // Backward through final relu2
    Tensor grad_sum = relu2_.backward(grad_output);

    // The addition splits the gradient equally to both branches
    Tensor grad_residual = grad_sum;
    Tensor grad_skip = grad_sum;

    // Backward through main path: bn2 -> conv2 -> relu1 -> bn1 -> conv1
    Tensor grad_main = bn2_.backward(grad_residual);
    grad_main = conv2_.backward(grad_main);
    grad_main = relu1_.backward(grad_main);
    grad_main = bn1_.backward(grad_main);
    grad_main = conv1_.backward(grad_main);

    // Backward through skip path
    Tensor grad_skip_input;
    if (downsample_) {
        grad_skip_input = downsample_bn_->backward(grad_skip);
        grad_skip_input = downsample_->backward(grad_skip_input);
    } else {
        grad_skip_input = grad_skip;
    }

    // Sum gradients from both paths at the input
    return grad_main + grad_skip_input;
}

std::vector<Tensor*> ResBlock::parameters() {
    std::vector<Tensor*> params;

    // conv1 parameters
    for (auto* p : conv1_.parameters()) params.push_back(p);
    // bn1 parameters
    for (auto* p : bn1_.parameters()) params.push_back(p);
    // conv2 parameters
    for (auto* p : conv2_.parameters()) params.push_back(p);
    // bn2 parameters
    for (auto* p : bn2_.parameters()) params.push_back(p);
    // downsample parameters (if present)
    if (downsample_) {
        for (auto* p : downsample_->parameters()) params.push_back(p);
        for (auto* p : downsample_bn_->parameters()) params.push_back(p);
    }

    return params;
}

std::vector<Tensor*> ResBlock::gradients() {
    std::vector<Tensor*> grads;

    for (auto* g : conv1_.gradients()) grads.push_back(g);
    for (auto* g : bn1_.gradients()) grads.push_back(g);
    for (auto* g : conv2_.gradients()) grads.push_back(g);
    for (auto* g : bn2_.gradients()) grads.push_back(g);
    if (downsample_) {
        for (auto* g : downsample_->gradients()) grads.push_back(g);
        for (auto* g : downsample_bn_->gradients()) grads.push_back(g);
    }

    return grads;
}

void ResBlock::train() {
    training_ = true;
    conv1_.train(); bn1_.train(); relu1_.train();
    conv2_.train(); bn2_.train(); relu2_.train();
    if (downsample_) {
        downsample_->train();
        downsample_bn_->train();
    }
}

void ResBlock::eval() {
    training_ = false;
    conv1_.eval(); bn1_.eval(); relu1_.eval();
    conv2_.eval(); bn2_.eval(); relu2_.eval();
    if (downsample_) {
        downsample_->eval();
        downsample_bn_->eval();
    }
}

// ============================================================
// build_simple_conv
// Simple 2-conv + 2-FC classifier for MNIST (1x28x28)
//
// Architecture:
//   Conv2D(1, 16, 3, 28x28, stride=1, pad=1) -> 16x28x28
//   ReLU
//   MaxPool2D(2, 2, 16, 28x28)                -> 16x14x14
//   Conv2D(16, 32, 3, 14x14, stride=1, pad=1) -> 32x14x14
//   ReLU
//   MaxPool2D(2, 2, 32, 14x14)                -> 32x7x7 = 1568
//   Flatten
//   Linear(1568, 128)
//   ReLU
//   Linear(128, 10)
// ============================================================

Sequential build_simple_conv(std::mt19937& rng) {
    Sequential model;

    // Block 1: Conv -> ReLU -> MaxPool
    model.add(std::make_unique<Conv2D>(1, 16, 3, 28, 28, 1, 1, "conv1", rng));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<MaxPool2D>(2, 2, 16, 28, 28));

    // Block 2: Conv -> ReLU -> MaxPool
    model.add(std::make_unique<Conv2D>(16, 32, 3, 14, 14, 1, 1, "conv2", rng));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<MaxPool2D>(2, 2, 32, 14, 14));

    // Flatten (no-op in this framework but keeps intent clear)
    model.add(std::make_unique<Flatten>());

    // FC layers
    model.add(std::make_unique<Linear>(32 * 7 * 7, 128, "fc1", rng));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(128, 10, "fc2", rng));

    return model;
}

// ============================================================
// build_lenet5
// LeNet-5 adapted for 28x28 MNIST input
//
// Architecture:
//   Conv2D(1, 6, 5, 28x28, stride=1, pad=2)   -> 6x28x28
//   ReLU
//   AvgPool2D(2, 2, 6, 28x28)                  -> 6x14x14
//   Conv2D(6, 16, 5, 14x14, stride=1, pad=0)   -> 16x10x10
//   ReLU
//   AvgPool2D(2, 2, 16, 10x10)                 -> 16x5x5 = 400
//   Flatten
//   Linear(400, 120)
//   ReLU
//   Linear(120, 84)
//   ReLU
//   Linear(84, 10)
// ============================================================

Sequential build_lenet5(std::mt19937& rng) {
    Sequential model;

    // C1: Convolutional layer
    model.add(std::make_unique<Conv2D>(1, 6, 5, 28, 28, 1, 2, "c1", rng));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<AvgPool2D>(2, 2, 6, 28, 28));

    // C3: Convolutional layer
    model.add(std::make_unique<Conv2D>(6, 16, 5, 14, 14, 1, 0, "c3", rng));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<AvgPool2D>(2, 2, 16, 10, 10));

    // Flatten
    model.add(std::make_unique<Flatten>());

    // F5: Fully connected
    model.add(std::make_unique<Linear>(16 * 5 * 5, 120, "f5", rng));
    model.add(std::make_unique<ReLU>());

    // F6: Fully connected
    model.add(std::make_unique<Linear>(120, 84, "f6", rng));
    model.add(std::make_unique<ReLU>());

    // Output
    model.add(std::make_unique<Linear>(84, 10, "output", rng));

    return model;
}

// ============================================================
// build_resnet_simple
// A simple ResNet-like model for MNIST
//
// Architecture:
//   Conv2D(1, 16, 3, 28x28, stride=1, pad=1)   -> 16x28x28
//   BatchNorm1d(16*28*28)
//   ReLU
//   ResBlock(16, 16, 28, 28)                     -> 16x28x28
//   MaxPool2D(2, 2, 16, 28x28)                   -> 16x14x14
//   ResBlock(16, 32, 14, 14)                      -> 32x14x14
//   MaxPool2D(2, 2, 32, 14x14)                   -> 32x7x7
//   GlobalAvgPool2D(32, 7, 7)                     -> 32
//   Linear(32, 10)
//
// Note: ResBlock is a custom Module, not part of Sequential's
//       layer iteration, but Sequential handles it via the
//       Module interface (forward/backward/parameters/gradients).
// ============================================================

Sequential build_resnet_simple(std::mt19937& rng) {
    Sequential model;

    // Initial conv + BN + ReLU
    model.add(std::make_unique<Conv2D>(1, 16, 3, 28, 28, 1, 1, "conv_init", rng));
    model.add(std::make_unique<BatchNorm2d>(16, 28, 28));
    model.add(std::make_unique<ReLU>());

    // ResBlock 1: 16 channels, 28x28 spatial
    model.add(std::make_unique<ResBlock>(16, 16, 28, 28, rng));

    // Downsample spatial via MaxPool
    model.add(std::make_unique<MaxPool2D>(2, 2, 16, 28, 28));

    // ResBlock 2: 16 -> 32 channels, 14x14 spatial
    model.add(std::make_unique<ResBlock>(16, 32, 14, 14, rng));

    // Downsample spatial via MaxPool
    model.add(std::make_unique<MaxPool2D>(2, 2, 32, 14, 14));

    // Global average pooling: 32x7x7 -> 32
    model.add(std::make_unique<GlobalAvgPool2D>(32, 7, 7));

    // Classification head
    model.add(std::make_unique<Linear>(32, 10, "fc_out", rng));

    return model;
}
