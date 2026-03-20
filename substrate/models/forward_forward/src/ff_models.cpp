#include "ff_models.h"
#include "core/bio/bio_affine.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

// ============================================================
// FFLayer
// ============================================================
FFLayer::FFLayer(size_t in_f, size_t out_f, double thresh, std::mt19937& rng)
    : in_features(in_f), out_features(out_f), threshold(thresh) {
    weights = Tensor(in_f, out_f);
    weights.fill_he(in_f, rng);
    biases = Tensor::zeros(1, out_f);
    grad_weights = Tensor::zeros(in_f, out_f);
    grad_biases = Tensor::zeros(1, out_f);
}

Tensor FFLayer::forward_raw(const Tensor& input) {
    // Linear + ReLU (no caching)
    Tensor z = bio::affine(input, weights, &biases, false);
    return z.apply([](double x) { return x > 0.0 ? x : 0.0; });
}

Tensor FFLayer::layer_normalize(const Tensor& x) const {
    // Per-sample normalization: normalize each row to unit length
    // This prevents later layers from using raw magnitude as a feature
    Tensor result(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i) {
        double norm_sq = 0;
        for (size_t j = 0; j < x.cols; ++j) {
            norm_sq += x(i, j) * x(i, j);
        }
        double norm = std::sqrt(norm_sq + 1e-8);
        for (size_t j = 0; j < x.cols; ++j) {
            result(i, j) = x(i, j) / norm;
        }
    }
    return result;
}

Tensor FFLayer::forward(const Tensor& input) {
    cached_input = input;
    Tensor z = bio::affine(input, weights, &biases, false);
    Tensor activated = z.apply([](double x) { return x > 0.0 ? x : 0.0; });
    cached_output = layer_normalize(activated);
    return cached_output;
}

Tensor FFLayer::goodness(const Tensor& input) {
    // Goodness = sum of squared activations per sample
    // input goes through: linear -> ReLU
    Tensor h = forward_raw(input);
    Tensor g(h.rows, 1);
    for (size_t i = 0; i < h.rows; ++i) {
        double sum_sq = 0;
        for (size_t j = 0; j < h.cols; ++j) {
            sum_sq += h(i, j) * h(i, j);
        }
        g(i, 0) = sum_sq;
    }
    return g;
}

double FFLayer::ff_update(const Tensor& pos_input, const Tensor& neg_input, double lr) {
    size_t batch = pos_input.rows;
    if (batch == 0) {
        throw std::runtime_error("FFLayer::ff_update requires non-empty positive batch");
    }
    if (neg_input.rows != batch) {
        throw std::runtime_error("FFLayer::ff_update positive/negative batch row mismatch");
    }
    if (pos_input.cols != in_features || neg_input.cols != in_features) {
        throw std::runtime_error("FFLayer::ff_update input width must match in_features");
    }

    // Forward pass for positive and negative samples
    Tensor pos_z = bio::affine(pos_input, weights, &biases, false);
    Tensor pos_h = pos_z.apply([](double x) { return x > 0.0 ? x : 0.0; });

    Tensor neg_z = bio::affine(neg_input, weights, &biases, false);
    Tensor neg_h = neg_z.apply([](double x) { return x > 0.0 ? x : 0.0; });

    // Compute goodness for positive and negative
    // goodness_pos[i] = sum_j(pos_h[i][j]^2)
    // goodness_neg[i] = sum_j(neg_h[i][j]^2)
    std::vector<double> goodness_pos(batch), goodness_neg(batch);
    for (size_t i = 0; i < batch; ++i) {
        double gp = 0, gn = 0;
        for (size_t j = 0; j < out_features; ++j) {
            gp += pos_h(i, j) * pos_h(i, j);
            gn += neg_h(i, j) * neg_h(i, j);
        }
        goodness_pos[i] = gp;
        goodness_neg[i] = gn;
    }

    // Compute loss:
    // L = mean_i [ log(1 + exp(-(goodness_pos_i - theta)))
    //            + log(1 + exp(goodness_neg_i - theta)) ]
    double total_loss = 0;
    for (size_t i = 0; i < batch; ++i) {
        double lp = std::log(1.0 + std::exp(-(goodness_pos[i] - threshold)));
        double ln = std::log(1.0 + std::exp(goodness_neg[i] - threshold));
        total_loss += lp + ln;
    }
    total_loss /= (double)batch;

    // Compute gradients
    // For positive: d_loss/d_goodness_pos = -sigma(-(goodness_pos - theta))
    //   where sigma is the sigmoid function
    // For negative: d_loss/d_goodness_neg = sigma(goodness_neg - theta)
    //
    // d_goodness/d_h_j = 2 * h_j  (since goodness = sum h_j^2)
    // d_h_j/d_z_j = 1 if z_j > 0, else 0  (ReLU derivative)
    // d_z/d_W = input^T, d_z/d_b = 1

    grad_weights.fill_zeros();
    grad_biases.fill_zeros();

    for (size_t i = 0; i < batch; ++i) {
        // Positive contribution: wants to increase goodness
        double sig_pos = 1.0 / (1.0 + std::exp(goodness_pos[i] - threshold));
        // d_loss/d_h_pos_j = -sig_pos * 2 * h_pos_j * relu'(z_pos_j)
        // -> gradient on weights: d_loss/d_W += (-sig_pos * 2 * h_j * relu'(z_j)) * x_i^T

        for (size_t j = 0; j < out_features; ++j) {
            double h_val = pos_h(i, j);
            double relu_deriv = (pos_z(i, j) > 0.0) ? 1.0 : 0.0;
            double dL_dh = -sig_pos * 2.0 * h_val * relu_deriv;

            for (size_t k = 0; k < in_features; ++k) {
                grad_weights(k, j) += dL_dh * pos_input(i, k) / (double)batch;
            }
            grad_biases(0, j) += dL_dh / (double)batch;
        }

        // Negative contribution: wants to decrease goodness
        double sig_neg = 1.0 / (1.0 + std::exp(-(goodness_neg[i] - threshold)));
        // d_loss/d_h_neg_j = sig_neg * 2 * h_neg_j * relu'(z_neg_j)

        for (size_t j = 0; j < out_features; ++j) {
            double h_val = neg_h(i, j);
            double relu_deriv = (neg_z(i, j) > 0.0) ? 1.0 : 0.0;
            double dL_dh = sig_neg * 2.0 * h_val * relu_deriv;

            for (size_t k = 0; k < in_features; ++k) {
                grad_weights(k, j) += dL_dh * neg_input(i, k) / (double)batch;
            }
            grad_biases(0, j) += dL_dh / (double)batch;
        }
    }

    // Apply gradient update (SGD step)
    for (size_t i = 0; i < weights.data.size(); ++i) {
        weights.data[i] -= lr * grad_weights.data[i];
    }
    for (size_t j = 0; j < biases.data.size(); ++j) {
        biases.data[j] -= lr * grad_biases.data[j];
    }

    return total_loss;
}

// ============================================================
// ForwardForwardNetwork
// ============================================================
ForwardForwardNetwork::ForwardForwardNetwork(const std::vector<size_t>& layer_sizes,
                                              size_t nc, double thresh,
                                              std::mt19937& rng)
    : num_classes(nc), rng_ptr(&rng) {
    if (layer_sizes.size() < 2) {
        throw std::runtime_error("ForwardForwardNetwork requires at least input and output layer sizes");
    }
    input_dim = layer_sizes[0];

    for (size_t i = 0; i + 1 < layer_sizes.size(); ++i) {
        layers.emplace_back(layer_sizes[i], layer_sizes[i + 1], thresh, rng);
    }
}

ForwardForwardNetwork::ForwardForwardNetwork(std::mt19937& rng)
    : ForwardForwardNetwork({784, 500, 500}, 10, 2.0, rng) {}

Tensor ForwardForwardNetwork::embed_label(const Tensor& images, const Tensor& labels) {
    // Embed one-hot label into the first num_classes pixels of each image.
    // labels: batch x 1 with integer class labels.
    // The one-hot values are scaled to a strong signal (e.g., 1.0).
    size_t batch = images.rows;
    size_t dim = images.cols;
    Tensor result = images; // copy

    for (size_t i = 0; i < batch; ++i) {
        int label = (int)labels(i, 0);
        // Zero out the first 10 pixels
        for (size_t j = 0; j < 10 && j < dim; ++j) {
            result(i, j) = 0.0;
        }
        // Set the one-hot position
        if (label >= 0 && label < 10 && (size_t)label < dim) {
            result(i, (size_t)label) = 1.0;
        }
    }
    return result;
}

Tensor ForwardForwardNetwork::embed_wrong_label(const Tensor& images, const Tensor& labels,
                                                  std::mt19937& rng) {
    // Embed a random wrong label into each image.
    size_t batch = images.rows;
    size_t dim = images.cols;
    Tensor result = images; // copy

    std::uniform_int_distribution<int> dist(0, 8); // 0..8 -> 9 choices

    for (size_t i = 0; i < batch; ++i) {
        int correct = (int)labels(i, 0);
        // Pick a random wrong label
        int wrong = dist(rng);
        if (wrong >= correct) wrong++; // skip the correct label

        // Zero out the first 10 pixels
        for (size_t j = 0; j < 10 && j < dim; ++j) {
            result(i, j) = 0.0;
        }
        // Set the wrong one-hot position
        if (wrong >= 0 && wrong < 10 && (size_t)wrong < dim) {
            result(i, (size_t)wrong) = 1.0;
        }
    }
    return result;
}

double ForwardForwardNetwork::train_step(const Tensor& images, const Tensor& labels, double lr) {
    // Create positive and negative samples
    Tensor pos_data = embed_label(images, labels);
    Tensor neg_data = embed_wrong_label(images, labels, *rng_ptr);

    double total_loss = 0;

    // Train each layer with its own FF update
    Tensor pos_input = pos_data;
    Tensor neg_input = neg_data;

    for (size_t i = 0; i < layers.size(); ++i) {
        // Compute the output of this layer FIRST (with current weights)
        // to use as input for the next layer. This must happen before
        // the weight update so the next layer sees activations computed
        // with the pre-update weights.
        Tensor next_pos_input = layers[i].forward(pos_input);
        Tensor next_neg_input = layers[i].forward(neg_input);

        // FF update for this layer (modifies weights)
        double layer_loss = layers[i].ff_update(pos_input, neg_input, lr);
        total_loss += layer_loss;

        pos_input = next_pos_input;
        neg_input = next_neg_input;
    }

    return total_loss / (double)layers.size();
}

std::vector<int> ForwardForwardNetwork::predict(const Tensor& images) {
    Tensor goodness_scores = predict_goodness(images);
    size_t batch = images.rows;
    std::vector<int> predictions(batch);

    for (size_t i = 0; i < batch; ++i) {
        double best_goodness = -std::numeric_limits<double>::infinity();
        int best_label = 0;
        for (size_t c = 0; c < num_classes; ++c) {
            if (goodness_scores(i, c) > best_goodness) {
                best_goodness = goodness_scores(i, c);
                best_label = (int)c;
            }
        }
        predictions[i] = best_label;
    }
    return predictions;
}

Tensor ForwardForwardNetwork::predict_goodness(const Tensor& images) {
    // For each possible label, embed it and compute total goodness through all layers.
    size_t batch = images.rows;
    Tensor result(batch, num_classes);

    for (size_t c = 0; c < num_classes; ++c) {
        // Create label tensor for this class
        Tensor label_tensor(batch, 1, (double)c);

        // Embed label
        Tensor embedded = embed_label(images, label_tensor);

        // Forward through all layers, summing goodness
        Tensor input = embedded;
        for (size_t i = 0; i < layers.size(); ++i) {
            // Compute goodness at this layer (before layer norm)
            Tensor g = layers[i].goodness(input);

            // Add to result
            for (size_t b = 0; b < batch; ++b) {
                result(b, c) += g(b, 0);
            }

            // Forward through layer (with layer norm) for next layer's input
            input = layers[i].forward(input);
        }
    }

    return result;
}

std::vector<Tensor*> ForwardForwardNetwork::parameters() {
    std::vector<Tensor*> all;
    for (auto& layer : layers) {
        auto p = layer.parameters();
        all.insert(all.end(), p.begin(), p.end());
    }
    return all;
}

std::vector<Tensor*> ForwardForwardNetwork::all_gradients() {
    std::vector<Tensor*> all;
    for (auto& layer : layers) {
        auto g = layer.gradients();
        all.insert(all.end(), g.begin(), g.end());
    }
    return all;
}

void ForwardForwardNetwork::zero_grad() {
    for (auto& layer : layers) {
        layer.grad_weights.fill_zeros();
        layer.grad_biases.fill_zeros();
    }
}
