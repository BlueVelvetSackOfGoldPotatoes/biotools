#include "hebbian_models.h"
#include "core/bio/bio_affine.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ============================================================
// HebbianMLP
// ============================================================
HebbianMLP::HebbianMLP(const std::vector<size_t>& sizes, double learning_rate,
                        const std::string& hebbian_rule, std::mt19937& rng)
    : lr(learning_rate), rule(hebbian_rule), layer_sizes(sizes), bcm_tau(100.0) {
    if (sizes.size() < 2) {
        throw std::runtime_error("HebbianMLP requires at least input and output layers");
    }

    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
        size_t fan_in = sizes[i];
        size_t fan_out = sizes[i + 1];

        // Initialize weights with small random values
        Tensor w(fan_in, fan_out);
        double std = std::sqrt(2.0 / (double)(fan_in + fan_out));
        w.fill_random_normal(0.0, std, rng);
        weights.push_back(w);

        Tensor b = Tensor::zeros(1, fan_out);
        biases.push_back(b);

        // BCM thresholds: initialized to small positive value
        Tensor theta(1, fan_out, 0.1);
        thresholds.push_back(theta);
    }
}

Tensor HebbianMLP::activate(const Tensor& x, bool is_output) const {
    if (is_output) {
        return x; // linear output for last layer
    }
    // ReLU for hidden layers
    return x.apply([](double v) { return v > 0.0 ? v : 0.0; });
}

Tensor HebbianMLP::forward(const Tensor& input) {
    Tensor x = input;
    cached_activations.clear();
    cached_activations.push_back(x);

    for (size_t i = 0; i < weights.size(); ++i) {
        bool is_output = (i == weights.size() - 1);
        x = bio::affine(x, weights[i], &biases[i], false);
        x = activate(x, is_output);
        cached_activations.push_back(x);
    }
    return x;
}

void HebbianMLP::hebbian_update(const Tensor& input) {
    // Forward pass, collecting activations
    Tensor x = input;
    std::vector<Tensor> pre_act;  // pre-activation inputs to each layer
    std::vector<Tensor> post_act; // post-activation outputs

    pre_act.push_back(x);

    for (size_t i = 0; i < weights.size(); ++i) {
        bool is_output = (i == weights.size() - 1);
        Tensor z = bio::affine(x, weights[i], &biases[i], false);
        x = activate(z, is_output);
        post_act.push_back(x);
        if (i + 1 < weights.size()) {
            pre_act.push_back(x);
        }
    }

    // Apply Hebbian update at each layer
    size_t batch = input.rows;
    double scale = lr / (double)batch;

    for (size_t layer = 0; layer < weights.size(); ++layer) {
        Tensor& W = weights[layer];
        Tensor& b = biases[layer];
        const Tensor& x_in = pre_act[layer];   // batch x fan_in
        const Tensor& y = post_act[layer];      // batch x fan_out
        size_t fan_in = W.rows;
        size_t fan_out = W.cols;

        if (rule == "hebb") {
            // Basic Hebbian: dW = lr * x^T * y / batch
            // With weight decay to prevent explosion
            Tensor dW = x_in.transpose().matmul(y) * scale;
            double decay = 0.001;
            for (size_t i = 0; i < fan_in; ++i) {
                for (size_t j = 0; j < fan_out; ++j) {
                    W(i, j) += dW(i, j) - decay * W(i, j) * lr;
                }
            }
            // Bias update: average output
            Tensor db = y.sum_rows() * (scale);
            for (size_t j = 0; j < fan_out; ++j) {
                b(0, j) += db(0, j);
            }
        }
        else if (rule == "sanger") {
            // Sanger's rule (Generalized Hebbian Algorithm):
            // dW_ij = lr * y_j * (x_i - sum_{k<=j} W_ik * y_k) / batch
            // This produces ordered principal components (PC1, PC2, ..., PCn).
            // The triangular structure where higher-index neurons are
            // influenced by lower-index ones distinguishes this from
            // single-neuron Oja's rule.
            //
            // Optimized: compute reconstruction incrementally over j for each (bi, i)
            // to avoid O(fan_out^2) inner loop. Total: O(batch * fan_in * fan_out).
            double lr_scaled = lr / (double)batch;
            for (size_t bi = 0; bi < batch; ++bi) {
                for (size_t i = 0; i < fan_in; ++i) {
                    double reconstruction = 0.0;
                    for (size_t j = 0; j < fan_out; ++j) {
                        double y_j = y(bi, j);
                        reconstruction += W(i, j) * y_j;
                        double update = lr_scaled * y_j * (x_in(bi, i) - reconstruction);
                        W(i, j) += update;
                    }
                }
            }
            // No bias update for PCA-like features
        }
        else if (rule == "bcm") {
            // BCM rule: dW_ij = lr * y_j * (y_j - theta_j) * x_i
            // Threshold update: theta_j = theta_j + (mean(y_j^2) - theta_j) / tau
            // The threshold is updated once per batch using the batch mean of y_j^2,
            // not once per sample.
            Tensor& theta = thresholds[layer];

            // Accumulate y_j^2 across the batch for threshold update
            std::vector<double> y_sq_mean(fan_out, 0.0);

            for (size_t bi = 0; bi < batch; ++bi) {
                for (size_t j = 0; j < fan_out; ++j) {
                    double y_j = y(bi, j);
                    double th_j = theta(0, j);
                    double modulation = y_j * (y_j - th_j);
                    for (size_t i = 0; i < fan_in; ++i) {
                        W(i, j) += (lr / (double)batch) * modulation * x_in(bi, i);
                    }
                    y_sq_mean[j] += y_j * y_j;
                }
            }

            // Single threshold update using batch mean of y_j^2
            for (size_t j = 0; j < fan_out; ++j) {
                y_sq_mean[j] /= (double)batch;
                theta(0, j) += (y_sq_mean[j] - theta(0, j)) / bcm_tau;
            }

            // Bias update
            Tensor db = y.sum_rows() * scale;
            for (size_t j = 0; j < fan_out; ++j) {
                b(0, j) += db(0, j);
            }
        }

        // Normalize weights to prevent explosion (column-wise L2 normalization)
        for (size_t j = 0; j < fan_out; ++j) {
            double norm = 0;
            for (size_t i = 0; i < fan_in; ++i)
                norm += W(i, j) * W(i, j);
            norm = std::sqrt(norm);
            if (norm > 3.0) {
                double scale_factor = 3.0 / norm;
                for (size_t i = 0; i < fan_in; ++i)
                    W(i, j) *= scale_factor;
            }
        }
    }
}

std::vector<Tensor*> HebbianMLP::get_weights() {
    std::vector<Tensor*> ptrs;
    for (auto& w : weights) ptrs.push_back(&w);
    return ptrs;
}

std::vector<Tensor*> HebbianMLP::get_biases() {
    std::vector<Tensor*> ptrs;
    for (auto& b : biases) ptrs.push_back(&b);
    return ptrs;
}

// ============================================================
// HebbianClassifier
// ============================================================
HebbianClassifier::HebbianClassifier(const std::vector<size_t>& feature_layers,
                                      size_t num_classes, double hebbian_lr,
                                      const std::string& rule, std::mt19937& rng)
    : feature_extractor(feature_layers, hebbian_lr, rule, rng),
      probe(feature_layers.back(), num_classes, "hebbian_probe", rng) {
    grad_probe_weights = Tensor::zeros(feature_layers.back(), num_classes);
    grad_probe_biases = Tensor::zeros(1, num_classes);
}

HebbianClassifier::HebbianClassifier(std::mt19937& rng)
    : HebbianClassifier({784, 128}, 10, 0.001, "sanger", rng) {}

void HebbianClassifier::train_features(const Tensor& input) {
    feature_extractor.hebbian_update(input);
}

Tensor HebbianClassifier::forward(const Tensor& input) {
    // Forward through feature extractor (no learning)
    cached_features = feature_extractor.forward(input);
    // Normalize features per-sample to prevent extreme logit values
    for (size_t i = 0; i < cached_features.rows; ++i) {
        double norm = 0;
        for (size_t j = 0; j < cached_features.cols; ++j)
            norm += cached_features(i, j) * cached_features(i, j);
        norm = std::sqrt(norm + 1e-8);
        for (size_t j = 0; j < cached_features.cols; ++j)
            cached_features(i, j) /= norm;
    }
    // Forward through linear probe
    cached_logits = probe.forward(cached_features);
    return cached_logits;
}

Tensor HebbianClassifier::backward(const Tensor& grad_output) {
    // Only backprop through the probe, not the feature extractor
    Tensor grad_features = probe.backward(grad_output);
    // We do not propagate further; Hebbian layers are unsupervised
    return grad_features;
}

std::vector<Tensor*> HebbianClassifier::parameters() {
    return probe.parameters();
}

std::vector<Tensor*> HebbianClassifier::gradients() {
    return probe.gradients();
}

void HebbianClassifier::zero_grad() {
    probe.grad_weights.fill_zeros();
    probe.grad_biases.fill_zeros();
}

// ============================================================
// HopfieldNetwork
// ============================================================
HopfieldNetwork::HopfieldNetwork(int n)
    : num_neurons(n), num_stored(0) {
    weights = Tensor::zeros((size_t)n, (size_t)n);
}

void HopfieldNetwork::store(const Tensor& pattern) {
    // pattern: 1 x N with values +1 or -1
    if (pattern.rows != 1 || static_cast<int>(pattern.cols) != num_neurons) {
        throw std::runtime_error("HopfieldNetwork::store expects pattern shape 1 x num_neurons");
    }

    // Outer product rule: W += pattern^T * pattern
    // Then zero the diagonal (no self-connections)
    for (int i = 0; i < num_neurons; ++i) {
        for (int j = 0; j < num_neurons; ++j) {
            if (i != j) {
                weights((size_t)i, (size_t)j) += pattern(0, (size_t)i) * pattern(0, (size_t)j);
            }
        }
    }
    num_stored++;
}

Tensor HopfieldNetwork::recall(const Tensor& query, int max_steps) {
    // query: 1 x N with values +1 or -1
    if (query.rows != 1 || static_cast<int>(query.cols) != num_neurons) {
        throw std::runtime_error("HopfieldNetwork::recall expects query shape 1 x num_neurons");
    }

    Tensor state = query;

    for (int step = 0; step < max_steps; ++step) {
        bool changed = false;

        // Asynchronous update: update each neuron one at a time
        for (int i = 0; i < num_neurons; ++i) {
            // Compute local field for neuron i: h_i = sum_j(W_ij * s_j)
            double h = 0;
            for (int j = 0; j < num_neurons; ++j) {
                h += weights((size_t)i, (size_t)j) * state(0, (size_t)j);
            }

            // Apply sign activation
            double new_val = (h >= 0) ? 1.0 : -1.0;
            if (new_val != state(0, (size_t)i)) {
                state(0, (size_t)i) = new_val;
                changed = true;
            }
        }

        // If no neuron changed, we have converged
        if (!changed) break;
    }

    return state;
}

double HopfieldNetwork::energy(const Tensor& state) const {
    // E = -0.5 * sum_i sum_j W_ij * s_i * s_j
    if (state.rows != 1 || static_cast<int>(state.cols) != num_neurons) {
        throw std::runtime_error("HopfieldNetwork::energy expects state shape 1 x num_neurons");
    }

    double E = 0;
    for (int i = 0; i < num_neurons; ++i) {
        for (int j = 0; j < num_neurons; ++j) {
            E -= weights((size_t)i, (size_t)j) * state(0, (size_t)i) * state(0, (size_t)j);
        }
    }
    return 0.5 * E;
}

void HopfieldNetwork::reset() {
    weights.fill_zeros();
    num_stored = 0;
}
