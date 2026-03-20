#include "core/benchmark/bit_bridge_model.h"

#include <iostream>
#include <stdexcept>
#include <utility>

namespace benchmark {

BitBridgeModel::BitBridgeModel(std::unique_ptr<TrainableModel> backbone, BitBridgeConfig cfg, std::mt19937& rng)
    : backbone_(std::move(backbone)),
      cfg_(cfg),
      input_proj_(cfg_.input_bits, cfg_.bridge_width, "bit_bridge_in", rng),
      bridge_to_backbone_(cfg_.bridge_width, cfg_.backbone_input_dim, "bit_bridge_backbone_in", rng),
      output_proj_(cfg_.backbone_output_dim, cfg_.action_dim, "bit_bridge_out", rng) {
    if (!backbone_) {
        throw std::runtime_error("BitBridgeModel requires a non-null backbone");
    }
    if (cfg_.input_bits == 0 || cfg_.action_dim == 0) {
        throw std::runtime_error("BitBridgeModel requires input_bits > 0 and action_dim > 0");
    }
    if (cfg_.backbone_output_dim == 0) {
        throw std::runtime_error("BitBridgeModel requires backbone_output_dim > 0");
    }

    id_ = "bit_bridge[" + backbone_->id() + "]";
}

Tensor BitBridgeModel::forward(const Tensor& input) {
    if (input.cols != cfg_.input_bits) {
        throw std::runtime_error("BitBridgeModel::forward input width mismatch");
    }
    Tensor h = input_proj_.forward(input);
    h = input_act_.forward(h);
    h = bridge_to_backbone_.forward(h);
    h = bridge_act_.forward(h);
    Tensor backbone_logits = backbone_->forward(h);
    if (backbone_logits.cols != cfg_.backbone_output_dim) {
        throw std::runtime_error("BitBridgeModel::forward backbone output width mismatch");
    }
    return output_proj_.forward(backbone_logits);
}

Tensor BitBridgeModel::backward(const Tensor& grad_output) {
    if (grad_output.cols != cfg_.action_dim) {
        throw std::runtime_error("BitBridgeModel::backward grad width mismatch");
    }
    Tensor grad = output_proj_.backward(grad_output);
    grad = backbone_->backward(grad);
    if (grad.data.empty() || grad.cols != cfg_.backbone_input_dim) {
        if (grad.data.empty() || grad.cols == 0 || grad.rows == 0) {
            if (!warned_missing_input_grad_) {
                std::cerr
                    << "BitBridgeModel: backbone '" << backbone_->id()
                    << "' did not provide usable input gradients (rows=" << grad.rows
                    << ", cols=" << grad.cols
                    << "); bridge input layers remain frozen."
                    << std::endl;
                warned_missing_input_grad_ = true;
            }
            return Tensor();
        }

        // Surrogate gradient remap for backbones that expose only intermediate
        // feature gradients. This keeps bridge projections trainable while the
        // backbone can still use non-standard local learning internals.
        Tensor remapped(grad.rows, cfg_.backbone_input_dim);
        for (std::size_t r = 0; r < remapped.rows; ++r) {
            for (std::size_t c = 0; c < remapped.cols; ++c) {
                remapped(r, c) = grad(r, c % grad.cols);
            }
        }
        if (!warned_missing_input_grad_) {
            std::cerr
                << "BitBridgeModel: backbone '" << backbone_->id()
                << "' gradient width (" << grad.cols
                << ") differs from expected input width (" << cfg_.backbone_input_dim
                << "); using deterministic surrogate remap for bridge training."
                << std::endl;
            warned_missing_input_grad_ = true;
        }
        grad = std::move(remapped);
    }
    grad = bridge_act_.backward(grad);
    grad = bridge_to_backbone_.backward(grad);
    grad = input_act_.backward(grad);
    return input_proj_.backward(grad);
}

std::vector<Tensor*> BitBridgeModel::parameters() {
    std::vector<Tensor*> out = backbone_->parameters();
    auto p_in = input_proj_.parameters();
    auto p_bridge = bridge_to_backbone_.parameters();
    auto p_out = output_proj_.parameters();
    out.insert(out.end(), p_in.begin(), p_in.end());
    out.insert(out.end(), p_bridge.begin(), p_bridge.end());
    out.insert(out.end(), p_out.begin(), p_out.end());
    return out;
}

std::vector<Tensor*> BitBridgeModel::gradients() {
    std::vector<Tensor*> out = backbone_->gradients();
    auto g_in = input_proj_.gradients();
    auto g_bridge = bridge_to_backbone_.gradients();
    auto g_out = output_proj_.gradients();
    out.insert(out.end(), g_in.begin(), g_in.end());
    out.insert(out.end(), g_bridge.begin(), g_bridge.end());
    out.insert(out.end(), g_out.begin(), g_out.end());
    return out;
}

void BitBridgeModel::zero_grad() {
    backbone_->zero_grad();
    for (auto* g : input_proj_.gradients()) {
        if (g) g->fill_zeros();
    }
    for (auto* g : bridge_to_backbone_.gradients()) {
        if (g) g->fill_zeros();
    }
    for (auto* g : output_proj_.gradients()) {
        if (g) g->fill_zeros();
    }
}

void BitBridgeModel::train() {
    backbone_->train();
}

void BitBridgeModel::eval() {
    backbone_->eval();
}

} // namespace benchmark
