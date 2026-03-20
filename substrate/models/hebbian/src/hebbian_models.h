#pragma once
#include "../../../core/nn/module.h"
#include "../../../core/losses/losses.h"
#include <cmath>
#include <vector>
#include <string>

// ============================================================
// Hebbian MLP
// ============================================================
// A multi-layer perceptron trained with local Hebbian learning rules
// instead of backpropagation.
//
// Supported rules:
//   "hebb"   - Basic Hebbian: dW = lr * (y * x^T)
//   "sanger" - Sanger's rule (Generalized Hebbian Algorithm):
//              dW_ij = lr * y_j * (x_i - sum_{k<=j} W_ik * y_k)
//              Extracts ordered principal components (PC1, PC2, ..., PCn).
//              Unlike single-neuron Oja's rule, Sanger's rule has a
//              triangular structure where higher-index neurons are
//              influenced by lower-index ones.
//   "bcm"    - BCM rule:      dW = lr * y * (y - theta) * x^T
//              (Bienenstock-Cooper-Munro with sliding threshold)
//
// For classification on MNIST, the HebbianMLP learns features in
// an unsupervised manner. A separate linear probe can be trained
// on top with standard gradient descent.
// ============================================================
class HebbianMLP {
public:
    std::vector<Tensor> weights;     // weights[i]: layer_sizes[i] x layer_sizes[i+1]
    std::vector<Tensor> biases;      // biases[i]:  1 x layer_sizes[i+1]
    double lr;
    std::string rule;                // "hebb", "sanger", "bcm"
    std::vector<size_t> layer_sizes;

    // For BCM: sliding threshold per neuron per layer
    // thresholds[i]: 1 x layer_sizes[i+1]
    std::vector<Tensor> thresholds;
    double bcm_tau;                  // time constant for threshold update

    // Cached activations for inspection
    std::vector<Tensor> cached_activations;

    HebbianMLP(const std::vector<size_t>& layer_sizes, double lr,
               const std::string& rule, std::mt19937& rng);

    // Forward pass (inference only, no learning)
    // input: batch x input_dim
    // Returns: batch x output_dim
    Tensor forward(const Tensor& input);

    // One-shot forward + Hebbian weight update
    // Processes one sample (or batch) through the network,
    // applying the selected Hebbian rule at each layer.
    // input: batch x input_dim
    void hebbian_update(const Tensor& input);

    // Access weights for inspection / linear probe
    std::vector<Tensor*> get_weights();
    std::vector<Tensor*> get_biases();

private:
    // Apply activation function (ReLU for hidden, identity for output)
    Tensor activate(const Tensor& x, bool is_output) const;
};

// ============================================================
// Hebbian Classifier
// ============================================================
// Combines an unsupervised HebbianMLP feature extractor with a
// supervised linear probe for classification.
//
// Architecture:
//   HebbianMLP [784 -> 128] (unsupervised, Hebbian-trained)
//   Linear [128 -> 10]      (supervised, gradient-trained)
//
// Training procedure:
//   1. Train HebbianMLP unsupervised with hebbian_update()
//   2. Freeze Hebbian layers, train linear probe with backprop
// ============================================================
class HebbianClassifier {
public:
    HebbianMLP feature_extractor;
    Linear probe;
    Tensor grad_probe_weights, grad_probe_biases;

    // Caches for backward through probe
    Tensor cached_features;
    Tensor cached_logits;

    HebbianClassifier(const std::vector<size_t>& feature_layers, size_t num_classes,
                      double hebbian_lr, const std::string& rule, std::mt19937& rng);

    // Convenience constructor for MNIST defaults: [784, 128], 10 classes
    HebbianClassifier(std::mt19937& rng);

    // Unsupervised Hebbian training step
    void train_features(const Tensor& input);

    // Supervised forward (features are frozen)
    // input: batch x 784 -> batch x num_classes (logits)
    Tensor forward(const Tensor& input);

    // Backward through linear probe only
    Tensor backward(const Tensor& grad_output);

    // Parameters/gradients for probe only (for optimizer)
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();
};

// ============================================================
// Hopfield Network
// ============================================================
// A classic Hopfield network for associative memory.
// Stores binary patterns (+1/-1) and recalls them from
// noisy or partial queries.
//
// Weight matrix is symmetric with zero diagonal.
// Storage follows the outer-product (Hebbian) rule:
//   W += pattern * pattern^T - I
//
// Recall uses asynchronous updates until convergence
// or max_steps is reached.
//
// Theoretical capacity: ~0.138 * N patterns.
// ============================================================
class HopfieldNetwork {
public:
    Tensor weights;     // N x N symmetric weight matrix
    int num_neurons;
    int num_stored;     // number of patterns stored so far

    HopfieldNetwork(int num_neurons);

    // Store a pattern in the network.
    // pattern: 1 x N with values +1 or -1
    void store(const Tensor& pattern);

    // Recall the stored pattern closest to the query.
    // query: 1 x N with values +1 or -1
    // Returns: 1 x N with values +1 or -1
    Tensor recall(const Tensor& query, int max_steps = 100);

    // Compute the energy of a given state.
    // E = -0.5 * state * W * state^T
    double energy(const Tensor& state) const;

    // Theoretical capacity
    int capacity() const { return (int)(0.138 * num_neurons); }

    // Reset all stored patterns
    void reset();
};
