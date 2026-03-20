#pragma once

#include "core/online/trainable_model.h"

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace hybrid {

using ModelFactory = std::function<std::unique_ptr<TrainableModel>(std::mt19937&)>;

// Registers all built-in adapters once (idempotent).
void register_builtin_model_adapters();

// Registers a single adapter under a canonical name with optional aliases.
void register_model_adapter(const std::string& canonical_name,
                            ModelFactory factory,
                            const std::vector<std::string>& aliases = {});

bool has_model_adapter(const std::string& name);
std::vector<std::string> registered_model_adapters();
std::vector<std::string> non_adapter_model_families();
std::unique_ptr<TrainableModel> create_model_adapter(const std::string& name, std::mt19937& rng);

} // namespace hybrid
