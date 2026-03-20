#pragma once

#include "core/nn/module.h"
#include "core/online/trainable_model.h"

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace hybrid {

struct NamedModel {
    std::string name;
    std::unique_ptr<TrainableModel> model;
};

class HybridSystem : public TrainableModel {
public:
    HybridSystem(std::vector<NamedModel> experts, std::mt19937& rng);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

    void train() override;
    void eval() override;
    const std::string& id() const override { return id_; }

    std::vector<std::string> expert_names() const;
    std::vector<double> expert_fusion_strength() const;
    std::vector<double> expert_accuracy(const Tensor& input, const Tensor& one_hot_labels);

private:
    std::vector<NamedModel> experts_;
    std::size_t expert_width_ = 10;

    // Two-stage fusion:
    // concat(expert_logits) -> fusion_in -> GELU -> fusion_out.
    Linear fusion_in_;
    GELU fusion_act_;
    Linear fusion_out_;

    std::string id_;
};

std::vector<std::string> supported_model_names();
std::vector<std::string> parse_model_spec(const std::string& spec);

std::unique_ptr<TrainableModel> build_model_by_name(const std::string& name, std::mt19937& rng);
std::vector<NamedModel> build_named_models(const std::vector<std::string>& names, std::mt19937& rng);

} // namespace hybrid
