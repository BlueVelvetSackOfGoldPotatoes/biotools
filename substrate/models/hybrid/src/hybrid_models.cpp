#include "models/hybrid/src/hybrid_models.h"

#include "core/losses/losses.h"
#include "core/metrics/metrics.h"
#include "models/hybrid/src/model_registry.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string trim(std::string s) {
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!s.empty() && is_space(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && is_space(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

} // namespace

namespace hybrid {

std::vector<std::string> supported_model_names() {
    return registered_model_adapters();
}

std::vector<std::string> parse_model_spec(const std::string& spec) {
    std::vector<std::string> out;
    std::stringstream ss(spec);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token = to_lower(trim(token));
        if (!token.empty()) out.push_back(token);
    }
    return out;
}

std::unique_ptr<TrainableModel> build_model_by_name(const std::string& name, std::mt19937& rng) {
    return create_model_adapter(name, rng);
}

std::vector<NamedModel> build_named_models(const std::vector<std::string>& names, std::mt19937& rng) {
    std::vector<NamedModel> out;
    std::unordered_map<std::string, int> seen;
    out.reserve(names.size());
    for (const auto& raw : names) {
        const std::string base = to_lower(trim(raw));
        auto model = build_model_by_name(base, rng);
        int& count = seen[base];
        count += 1;
        std::string unique_name = base;
        if (count > 1) unique_name += "_" + std::to_string(count);
        out.push_back({unique_name, std::move(model)});
    }
    return out;
}

HybridSystem::HybridSystem(std::vector<NamedModel> experts, std::mt19937& rng)
    : experts_(std::move(experts)),
      fusion_in_(std::max<std::size_t>(1, experts_.size()) * 10, 64, "hybrid_fusion_in", rng),
      fusion_out_(64, 10, "hybrid_fusion_out", rng) {
    if (experts_.empty()) {
        throw std::runtime_error("HybridSystem requires at least one expert model");
    }
    expert_width_ = 10;

    std::ostringstream oss;
    oss << "hybrid[";
    for (std::size_t i = 0; i < experts_.size(); ++i) {
        if (i) oss << "+";
        oss << experts_[i].name;
    }
    oss << "]";
    id_ = oss.str();
}

Tensor HybridSystem::forward(const Tensor& input) {
    const std::size_t batch = input.rows;
    Tensor concat(batch, experts_.size() * expert_width_);

    for (std::size_t e = 0; e < experts_.size(); ++e) {
        Tensor logits = experts_[e].model->forward(input);
        if (logits.rows != batch || logits.cols != expert_width_) {
            throw std::runtime_error("HybridSystem expert output shape must be batch x 10");
        }

        const std::size_t base = e * expert_width_;
        for (std::size_t r = 0; r < batch; ++r) {
            for (std::size_t c = 0; c < expert_width_; ++c) {
                concat(r, base + c) = logits(r, c);
            }
        }
    }

    Tensor h = fusion_in_.forward(concat);
    h = fusion_act_.forward(h);
    return fusion_out_.forward(h);
}

Tensor HybridSystem::backward(const Tensor& grad_output) {
    Tensor grad = fusion_out_.backward(grad_output);
    grad = fusion_act_.backward(grad);
    Tensor grad_concat = fusion_in_.backward(grad);

    // Accumulate input gradients from all experts for composability.
    // Each expert maps the same input to its logits, so their input
    // gradients are additive.
    Tensor grad_input;
    for (std::size_t e = 0; e < experts_.size(); ++e) {
        const std::size_t base = e * expert_width_;
        Tensor grad_slice = grad_concat.slice_cols(base, expert_width_);
        Tensor expert_grad = experts_[e].model->backward(grad_slice);
        if (expert_grad.data.empty()) continue;
        if (grad_input.data.empty()) {
            grad_input = expert_grad;
        } else if (expert_grad.data.size() == grad_input.data.size()) {
            grad_input = grad_input + expert_grad;
        }
    }

    return grad_input;
}

std::vector<Tensor*> HybridSystem::parameters() {
    std::vector<Tensor*> out;
    for (auto& expert : experts_) {
        auto p = expert.model->parameters();
        out.insert(out.end(), p.begin(), p.end());
    }
    auto p1 = fusion_in_.parameters();
    auto p2 = fusion_out_.parameters();
    out.insert(out.end(), p1.begin(), p1.end());
    out.insert(out.end(), p2.begin(), p2.end());
    return out;
}

std::vector<Tensor*> HybridSystem::gradients() {
    std::vector<Tensor*> out;
    for (auto& expert : experts_) {
        auto g = expert.model->gradients();
        out.insert(out.end(), g.begin(), g.end());
    }
    auto g1 = fusion_in_.gradients();
    auto g2 = fusion_out_.gradients();
    out.insert(out.end(), g1.begin(), g1.end());
    out.insert(out.end(), g2.begin(), g2.end());
    return out;
}

void HybridSystem::zero_grad() {
    for (auto& expert : experts_) expert.model->zero_grad();
    for (auto* g : fusion_in_.gradients()) g->fill_zeros();
    for (auto* g : fusion_out_.gradients()) g->fill_zeros();
}

void HybridSystem::train() {
    for (auto& expert : experts_) expert.model->train();
}

void HybridSystem::eval() {
    for (auto& expert : experts_) expert.model->eval();
}

std::vector<std::string> HybridSystem::expert_names() const {
    std::vector<std::string> out;
    out.reserve(experts_.size());
    for (const auto& expert : experts_) out.push_back(expert.name);
    return out;
}

std::vector<double> HybridSystem::expert_fusion_strength() const {
    std::vector<double> out(experts_.size(), 0.0);
    if (experts_.empty() || fusion_in_.weights.rows == 0 || fusion_in_.weights.cols == 0) return out;

    for (std::size_t e = 0; e < experts_.size(); ++e) {
        const std::size_t r0 = e * expert_width_;
        const std::size_t r1 = std::min(fusion_in_.weights.rows, r0 + expert_width_);
        double sum_abs = 0.0;
        std::size_t n = 0;
        for (std::size_t r = r0; r < r1; ++r) {
            for (std::size_t c = 0; c < fusion_in_.weights.cols; ++c) {
                sum_abs += std::abs(fusion_in_.weights(r, c));
                n++;
            }
        }
        out[e] = n ? (sum_abs / static_cast<double>(n)) : 0.0;
    }
    return out;
}

std::vector<double> HybridSystem::expert_accuracy(const Tensor& input, const Tensor& one_hot_labels) {
    std::vector<double> out(experts_.size(), 0.0);
    for (std::size_t i = 0; i < experts_.size(); ++i) {
        Tensor logits = experts_[i].model->forward(input);
        Tensor probs = LossFunctions::softmax(logits);
        out[i] = Metrics::accuracy_from_tensor(probs, one_hot_labels);
    }
    return out;
}

} // namespace hybrid
