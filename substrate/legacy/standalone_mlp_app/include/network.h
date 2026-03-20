#pragma once
#include "layer.h"
#include "loss.h"
#include "activation.h"
#include <vector>
#include <memory>

class Network {
private:
    std::vector<std::unique_ptr<Layer>> layers_;
    CrossEntropyLoss loss_fn_;
    std::vector<Matrix> layer_outputs_; // index 0 = input, index i = output of layer i-1

public:
    void add_layer(std::unique_ptr<Layer> layer);
    Matrix forward(const Matrix& input);
    void backward(const Matrix& predictions, const Matrix& targets);
    double compute_loss(const Matrix& predictions, const Matrix& targets);
    std::vector<int> predict(const Matrix& input);

    // Access for optimizer and interpretability
    Layer& get_layer(size_t index);
    const std::vector<std::unique_ptr<Layer>>& get_layers() const { return layers_; }
    const std::vector<Matrix>& get_layer_outputs() const { return layer_outputs_; }
    size_t num_layers() const { return layers_.size(); }
};
