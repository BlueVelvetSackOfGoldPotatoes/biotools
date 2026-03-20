#pragma once

#include "../../../core/nn/module.h"
#include "../../../core/losses/losses.h"
#include "../../../core/optim/optimizer.h"
#include <random>
#include <memory>
#include <vector>

// ============================================================
// GlobalAvgPool2D
// Reduces spatial dimensions by averaging over H*W for each channel.
// Input:  batch x (C * H * W)
// Output: batch x C
// ============================================================
class GlobalAvgPool2D : public Module {
    size_t channels_, h_, w_;
    size_t cached_batch_;
public:
    GlobalAvgPool2D(size_t c, size_t h, size_t w);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override { return "GlobalAvgPool2D"; }
};

// ============================================================
// ResBlock - Residual block with two conv layers and skip connection
// Input:  batch x (in_ch * H * W)
// Output: batch x (out_ch * H * W)
// If in_ch != out_ch, a 1x1 conv downsample is used on the skip path.
// ============================================================
class ResBlock : public Module {
    Conv2D conv1_, conv2_;
    BatchNorm2d bn1_, bn2_;
    ReLU relu1_, relu2_;
    std::unique_ptr<Conv2D> downsample_;       // 1x1 conv if channels change
    std::unique_ptr<BatchNorm2d> downsample_bn_; // BN after 1x1 downsample
    size_t channels_in_, channels_out_, h_, w_;
public:
    ResBlock(size_t in_ch, size_t out_ch, size_t h, size_t w, std::mt19937& rng);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    bool has_parameters() const override { return true; }
    std::string name() const override { return "ResBlock"; }

    void train() override;
    void eval() override;
};

// ============================================================
// CNN architecture builders
// All models accept MNIST input: batch x 784 (which represents 1x28x28)
// and output batch x 10 (logits for 10 classes).
// ============================================================

// Simple 2-conv + 2-FC classifier for MNIST
// Conv(1,16,3,pad=1) -> ReLU -> MaxPool(2) -> Conv(16,32,3,pad=1) -> ReLU -> MaxPool(2) -> FC(1568,128) -> ReLU -> FC(128,10)
Sequential build_simple_conv(std::mt19937& rng);

// LeNet-5 adapted for 28x28 input
// Conv(1,6,5,pad=2) -> ReLU -> AvgPool(2) -> Conv(6,16,5,pad=0) -> ReLU -> AvgPool(2) -> FC(400,120) -> ReLU -> FC(120,84) -> ReLU -> FC(84,10)
Sequential build_lenet5(std::mt19937& rng);

// Simple ResNet-like model for MNIST
// Conv(1,16,3,pad=1) -> BN -> ReLU -> ResBlock(16,16) -> MaxPool(2) -> ResBlock(16,32) -> MaxPool(2) -> GlobalAvgPool -> FC(32,10)
Sequential build_resnet_simple(std::mt19937& rng);
