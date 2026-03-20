#include "network.h"
#include <algorithm>

void Network::add_layer(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
}

Matrix Network::forward(const Matrix& input) {
    layer_outputs_.clear();
    layer_outputs_.push_back(input); // store input as layer_outputs_[0]

    Matrix current = input;
    for (auto& layer : layers_) {
        current = layer->forward(current);
        layer_outputs_.push_back(current);
    }
    return current;
}

void Network::backward(const Matrix& predictions, const Matrix& targets) {
    Matrix grad = loss_fn_.backward(predictions, targets);
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        grad = layers_[i]->backward(grad);
    }
}

double Network::compute_loss(const Matrix& predictions, const Matrix& targets) {
    return loss_fn_.forward(predictions, targets);
}

std::vector<int> Network::predict(const Matrix& input) {
    Matrix logits = forward(input);
    Matrix probs = Activation::softmax(logits);

    std::vector<int> predictions(probs.rows);
    for (size_t i = 0; i < probs.rows; ++i) {
        int best = 0;
        double best_val = probs(i, 0);
        for (size_t j = 1; j < probs.cols; ++j) {
            if (probs(i, j) > best_val) {
                best_val = probs(i, j);
                best = static_cast<int>(j);
            }
        }
        predictions[i] = best;
    }
    return predictions;
}

Layer& Network::get_layer(size_t index) {
    return *layers_[index];
}
