#pragma once

#include "core/tensor/tensor.h"

#include <string>
#include <vector>

// Unified training contract used by continuous-learning and hybrid systems.
class TrainableModel {
public:
    virtual ~TrainableModel() = default;

    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;

    virtual std::vector<Tensor*> parameters() = 0;
    virtual std::vector<Tensor*> gradients() = 0;
    virtual void zero_grad() = 0;

    virtual void train() = 0;
    virtual void eval() = 0;

    virtual const std::string& id() const = 0;
};

// Optional interface for models that route actions through specialist heads
// (e.g. per-piece control in chess). The action-owner map is state-dependent
// and can be pushed by environments/runtimes before forward passes.
class ActionOwnerAwareModel {
public:
    virtual ~ActionOwnerAwareModel() = default;
    virtual void set_action_owner_map(const std::vector<int>& owner_by_action,
                                      int num_owner_groups) = 0;
};
