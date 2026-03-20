#include "models/hybrid/src/model_registry.h"

#include "core/nn/module.h"
#include "models/actor_critic/src/ac_models.h"
#include "models/cnn/src/cnn_models.h"
#include "models/hebbian/src/hebbian_models.h"
#include "models/lstm/src/lstm_models.h"
#include "models/rnn/src/rnn_models.h"
#include "models/transformer/src/transformer_models.h"
#include "models/vit/src/vit_models.h"

#include <algorithm>
#include <cctype>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

std::string normalize_name(std::string s) {
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!s.empty() && is_space(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && is_space(static_cast<unsigned char>(s.back()))) s.pop_back();
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

struct RegistryState {
    std::unordered_map<std::string, hybrid::ModelFactory> factories_by_canonical;
    std::unordered_map<std::string, std::string> alias_to_canonical;
    std::vector<std::string> canonical_order;
    std::mutex mu;
};

RegistryState& registry_state() {
    static RegistryState state;
    return state;
}

class MLPTrainable final : public TrainableModel {
public:
    explicit MLPTrainable(std::mt19937& rng) : id_("mlp") {
        model_.add(std::make_unique<Linear>(784, 256, "hybrid_mlp_fc1", rng));
        model_.add(std::make_unique<ReLU>());
        model_.add(std::make_unique<Linear>(256, 128, "hybrid_mlp_fc2", rng));
        model_.add(std::make_unique<ReLU>());
        model_.add(std::make_unique<Linear>(128, 10, "hybrid_mlp_fc3", rng));
    }

    Tensor forward(const Tensor& input) override { return model_.forward(input); }
    Tensor backward(const Tensor& grad_output) override { return model_.backward(grad_output); }

    std::vector<Tensor*> parameters() override { return model_.parameters(); }
    std::vector<Tensor*> gradients() override { return model_.gradients(); }
    void zero_grad() override {
        auto grads = model_.gradients();
        for (auto* g : grads) g->fill_zeros();
    }

    void train() override { model_.train(); }
    void eval() override { model_.eval(); }
    const std::string& id() const override { return id_; }

private:
    Sequential model_;
    std::string id_;
};

class CNNTrainable final : public TrainableModel {
public:
    explicit CNNTrainable(std::mt19937& rng) : model_(build_simple_conv(rng)), id_("cnn") {}

    Tensor forward(const Tensor& input) override { return model_.forward(input); }
    Tensor backward(const Tensor& grad_output) override { return model_.backward(grad_output); }

    std::vector<Tensor*> parameters() override { return model_.parameters(); }
    std::vector<Tensor*> gradients() override { return model_.gradients(); }
    void zero_grad() override {
        auto grads = model_.gradients();
        for (auto* g : grads) g->fill_zeros();
    }

    void train() override { model_.train(); }
    void eval() override { model_.eval(); }
    const std::string& id() const override { return id_; }

private:
    Sequential model_;
    std::string id_;
};

class RNNTrainable final : public TrainableModel {
public:
    explicit RNNTrainable(std::mt19937& rng) : model_(28, 128, 10, 28, false, rng), id_("rnn") {}

    Tensor forward(const Tensor& input) override { return model_.forward(input); }
    Tensor backward(const Tensor& grad_output) override { return model_.backward(grad_output); }

    std::vector<Tensor*> parameters() override { return model_.parameters(); }
    std::vector<Tensor*> gradients() override { return model_.gradients(); }
    void zero_grad() override { model_.zero_grad(); }

    void train() override { model_.train_mode(); }
    void eval() override { model_.eval_mode(); }
    const std::string& id() const override { return id_; }

private:
    RNNClassifier model_;
    std::string id_;
};

class GRUTrainable final : public TrainableModel {
public:
    explicit GRUTrainable(std::mt19937& rng) : model_(28, 128, 10, 28, rng), id_("gru") {}

    Tensor forward(const Tensor& input) override { return model_.forward(input); }
    Tensor backward(const Tensor& grad_output) override { return model_.backward(grad_output); }

    std::vector<Tensor*> parameters() override { return model_.parameters(); }
    std::vector<Tensor*> gradients() override { return model_.gradients(); }
    void zero_grad() override { model_.zero_grad(); }

    void train() override {}
    void eval() override {}
    const std::string& id() const override { return id_; }

private:
    GRUClassifier model_;
    std::string id_;
};

class LSTMTrainable final : public TrainableModel {
public:
    explicit LSTMTrainable(std::mt19937& rng) : model_(28, 128, 10, 28, rng), id_("lstm") {}

    Tensor forward(const Tensor& input) override { return model_.forward(input); }
    Tensor backward(const Tensor& grad_output) override { return model_.backward(grad_output); }

    std::vector<Tensor*> parameters() override { return model_.parameters(); }
    std::vector<Tensor*> gradients() override { return model_.gradients(); }
    void zero_grad() override { model_.zero_grad(); }

    void train() override {}
    void eval() override {}
    const std::string& id() const override { return id_; }

private:
    LSTMClassifier model_;
    std::string id_;
};

class TransformerTrainable final : public TrainableModel {
public:
    explicit TransformerTrainable(std::mt19937& rng) : model_(rng), id_("transformer") {}

    Tensor forward(const Tensor& input) override { return model_.forward(input); }
    Tensor backward(const Tensor& grad_output) override { return model_.backward(grad_output); }

    std::vector<Tensor*> parameters() override { return model_.parameters(); }
    std::vector<Tensor*> gradients() override { return model_.gradients(); }
    void zero_grad() override { model_.zero_grad(); }

    void train() override { model_.train(); }
    void eval() override { model_.eval(); }
    const std::string& id() const override { return id_; }

private:
    TransformerClassifier model_;
    std::string id_;
};

class ViTTrainable final : public TrainableModel {
public:
    explicit ViTTrainable(std::mt19937& rng) : model_(rng), id_("vit") {}

    Tensor forward(const Tensor& input) override { return model_.forward(input); }
    Tensor backward(const Tensor& grad_output) override { return model_.backward(grad_output); }

    std::vector<Tensor*> parameters() override { return model_.parameters(); }
    std::vector<Tensor*> gradients() override { return model_.gradients(); }
    void zero_grad() override { model_.zero_grad(); }

    void train() override { model_.train(); }
    void eval() override { model_.eval(); }
    const std::string& id() const override { return id_; }

private:
    ViTClassifier model_;
    std::string id_;
};

class HebbianTrainable final : public TrainableModel {
public:
    explicit HebbianTrainable(std::mt19937& rng) : model_(rng), id_("hebbian") {}

    Tensor forward(const Tensor& input) override { return model_.forward(input); }
    Tensor backward(const Tensor& grad_output) override { return model_.backward(grad_output); }

    std::vector<Tensor*> parameters() override { return model_.parameters(); }
    std::vector<Tensor*> gradients() override { return model_.gradients(); }
    void zero_grad() override { model_.zero_grad(); }

    void train() override {}
    void eval() override {}
    const std::string& id() const override { return id_; }

private:
    HebbianClassifier model_;
    std::string id_;
};

class ActorCriticPolicyTrainable final : public TrainableModel {
public:
    explicit ActorCriticPolicyTrainable(std::mt19937& rng)
        : model_(784, 256, 10, rng), id_("actor_critic") {}

    Tensor forward(const Tensor& input) override { return model_.forward(input); }
    Tensor backward(const Tensor& grad_output) override { return model_.backward(grad_output); }

    std::vector<Tensor*> parameters() override { return model_.parameters(); }
    std::vector<Tensor*> gradients() override { return model_.gradients(); }
    void zero_grad() override { model_.zero_grad(); }

    void train() override {}
    void eval() override {}
    const std::string& id() const override { return id_; }

private:
    PolicyNetwork model_;
    std::string id_;
};

} // namespace

namespace hybrid {

void register_model_adapter(const std::string& canonical_name,
                            ModelFactory factory,
                            const std::vector<std::string>& aliases) {
    const std::string canonical = normalize_name(canonical_name);
    if (canonical.empty()) {
        throw std::runtime_error("register_model_adapter: canonical name cannot be empty");
    }
    if (!factory) {
        throw std::runtime_error("register_model_adapter: factory cannot be null");
    }

    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mu);

    if (state.factories_by_canonical.count(canonical) != 0) {
        throw std::runtime_error("register_model_adapter: duplicate canonical model '" + canonical + "'");
    }

    state.factories_by_canonical.emplace(canonical, std::move(factory));
    state.canonical_order.push_back(canonical);

    auto bind_alias = [&](const std::string& raw_alias) {
        const std::string alias = normalize_name(raw_alias);
        if (alias.empty()) return;
        const auto it = state.alias_to_canonical.find(alias);
        if (it != state.alias_to_canonical.end() && it->second != canonical) {
            throw std::runtime_error(
                "register_model_adapter: alias '" + alias + "' already bound to '" + it->second + "'"
            );
        }
        state.alias_to_canonical[alias] = canonical;
    };

    bind_alias(canonical);
    for (const auto& alias : aliases) bind_alias(alias);
}

void register_builtin_model_adapters() {
    static std::once_flag once;
    std::call_once(once, []() {
        register_model_adapter("mlp", [](std::mt19937& rng) {
            return std::make_unique<MLPTrainable>(rng);
        });
        register_model_adapter("cnn", [](std::mt19937& rng) {
            return std::make_unique<CNNTrainable>(rng);
        });
        register_model_adapter("rnn", [](std::mt19937& rng) {
            return std::make_unique<RNNTrainable>(rng);
        });
        register_model_adapter("gru", [](std::mt19937& rng) {
            return std::make_unique<GRUTrainable>(rng);
        });
        register_model_adapter("lstm", [](std::mt19937& rng) {
            return std::make_unique<LSTMTrainable>(rng);
        });
        register_model_adapter("transformer", [](std::mt19937& rng) {
            return std::make_unique<TransformerTrainable>(rng);
        });
        register_model_adapter("vit", [](std::mt19937& rng) {
            return std::make_unique<ViTTrainable>(rng);
        });
        register_model_adapter("hebbian", [](std::mt19937& rng) {
            return std::make_unique<HebbianTrainable>(rng);
        });
        register_model_adapter(
            "actor_critic",
            [](std::mt19937& rng) {
                return std::make_unique<ActorCriticPolicyTrainable>(rng);
            },
            {"actorcritic", "ac_policy"}
        );
    });
}

bool has_model_adapter(const std::string& name) {
    register_builtin_model_adapters();
    const std::string lookup = normalize_name(name);
    if (lookup.empty()) return false;
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mu);
    return state.alias_to_canonical.count(lookup) != 0;
}

std::vector<std::string> registered_model_adapters() {
    register_builtin_model_adapters();
    RegistryState& state = registry_state();
    std::lock_guard<std::mutex> lock(state.mu);
    return state.canonical_order;
}

std::unique_ptr<TrainableModel> create_model_adapter(const std::string& name, std::mt19937& rng) {
    register_builtin_model_adapters();
    const std::string lookup = normalize_name(name);
    if (lookup.empty()) {
        throw std::runtime_error("create_model_adapter: model name cannot be empty");
    }

    ModelFactory factory;
    std::vector<std::string> supported;
    {
        RegistryState& state = registry_state();
        std::lock_guard<std::mutex> lock(state.mu);
        const auto alias_it = state.alias_to_canonical.find(lookup);
        if (alias_it != state.alias_to_canonical.end()) {
            const auto fac_it = state.factories_by_canonical.find(alias_it->second);
            if (fac_it != state.factories_by_canonical.end()) {
                factory = fac_it->second;
            }
        }
        supported = state.canonical_order;
    }

    if (!factory) {
        const auto non_adapters = non_adapter_model_families();
        std::ostringstream oss;
        oss << "Unsupported model '" << name << "'. Supported: ";
        for (std::size_t i = 0; i < supported.size(); ++i) {
            if (i) oss << ", ";
            oss << supported[i];
        }
        if (!non_adapters.empty()) {
            oss << ". Non-adapter families (separate training interfaces): ";
            for (std::size_t i = 0; i < non_adapters.size(); ++i) {
                if (i) oss << ", ";
                oss << non_adapters[i];
            }
        }
        throw std::runtime_error(oss.str());
    }

    return factory(rng);
}

std::vector<std::string> non_adapter_model_families() {
    return {
        "markov",
        "reinforcement",
        "gnn",
        "forward_forward",
        "diffusion",
        "trees",
        "forests",
        "clustering",
        "continuous"
    };
}

} // namespace hybrid
