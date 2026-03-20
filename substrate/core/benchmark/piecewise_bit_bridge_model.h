#pragma once

#include "core/benchmark/bit_bridge_model.h"
#include "core/online/trainable_model.h"

#include <cstddef>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace benchmark {

// Routes action logits through multiple specialist bit-bridge models.
// Intended for settings like chess per-piece control, where each action
// belongs to a semantic owner group (pawn/knight/..).
class PiecewiseBitBridgeModel final : public TrainableModel, public ActionOwnerAwareModel {
public:
    PiecewiseBitBridgeModel(std::vector<std::unique_ptr<TrainableModel>> specialists,
                            BitBridgeConfig cfg,
                            std::mt19937& rng,
                            std::vector<std::string> owner_group_names = {});

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

    void train() override;
    void eval() override;
    const std::string& id() const override { return id_; }

    void set_action_owner_map(const std::vector<int>& owner_by_action,
                              int num_owner_groups) override;

private:
    std::vector<std::unique_ptr<BitBridgeModel>> specialists_;
    std::vector<std::string> owner_group_names_;
    std::vector<Tensor> cached_logits_;
    std::vector<int> owner_by_action_;
    int num_owner_groups_ = 1;
    std::size_t action_dim_ = 0;
    std::string id_;

    std::size_t specialist_index_for_action(std::size_t action) const;
};

} // namespace benchmark

