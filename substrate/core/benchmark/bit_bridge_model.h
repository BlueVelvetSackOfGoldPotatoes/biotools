#pragma once

#include "core/nn/module.h"
#include "core/online/trainable_model.h"

#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace benchmark {

struct BitBridgeConfig {
    std::size_t input_bits = 0;
    std::size_t bridge_width = 784;
    std::size_t backbone_input_dim = 784;
    std::size_t backbone_output_dim = 10;
    std::size_t action_dim = 0;
};

// Adapter that allows arbitrary bit observations/actions to be consumed by
// existing trainable backbones with fixed input/output dimensions.
class BitBridgeModel final : public TrainableModel {
public:
    BitBridgeModel(std::unique_ptr<TrainableModel> backbone, BitBridgeConfig cfg, std::mt19937& rng);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

    void train() override;
    void eval() override;
    const std::string& id() const override { return id_; }

    const BitBridgeConfig& config() const { return cfg_; }

private:
    std::unique_ptr<TrainableModel> backbone_;
    BitBridgeConfig cfg_;

    Linear input_proj_;
    GELU input_act_;
    Linear bridge_to_backbone_;
    GELU bridge_act_;
    Linear output_proj_;

    std::string id_;
    bool warned_missing_input_grad_ = false;
};

} // namespace benchmark
