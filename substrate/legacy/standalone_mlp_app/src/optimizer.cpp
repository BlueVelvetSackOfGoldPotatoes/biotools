#include "optimizer.h"

SGD::SGD(double lr) : learning_rate_(lr) {}

void SGD::step(Network& network) {
    for (size_t i = 0; i < network.num_layers(); ++i) {
        Layer& layer = network.get_layer(i);
        if (layer.has_parameters()) {
            // W -= lr * dW
            layer.get_weights() -= layer.get_weight_gradients() * learning_rate_;
            // b -= lr * db
            layer.get_biases() -= layer.get_bias_gradients() * learning_rate_;
        }
    }
}
