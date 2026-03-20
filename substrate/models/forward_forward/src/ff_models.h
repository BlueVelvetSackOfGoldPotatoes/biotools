#pragma once
#include "../../../core/nn/module.h"
#include "../../../core/losses/losses.h"
#include <cmath>
#include <vector>
#include <string>

// ============================================================
// Forward-Forward Layer
// ============================================================
// A single layer in the Forward-Forward network.
//
// Each layer learns independently (layer-local learning):
//   - Given positive data: maximize "goodness" (sum of squared activations)
//   - Given negative data: minimize "goodness"
//   - Goodness threshold theta separates positive from negative
//
// Loss per layer:
//   L = log(1 + exp(-(goodness_pos - theta))) + log(1 + exp(goodness_neg - theta))
//
// This drives positive samples above the threshold and
// negative samples below it.
// ============================================================
class FFLayer {
public:
    Tensor weights, biases;
    Tensor grad_weights, grad_biases;
    size_t in_features, out_features;
    double threshold;

    // Cached values for inspection
    Tensor cached_input, cached_output;

    FFLayer(size_t in_f, size_t out_f, double threshold, std::mt19937& rng);

    // Standard forward pass (with layer normalization of output)
    // input: batch x in_features
    // Returns: batch x out_features (ReLU activated, layer-normalized)
    Tensor forward(const Tensor& input);

    // Compute goodness for a given input (sum of squared activations per sample)
    // Returns: batch x 1
    Tensor goodness(const Tensor& input);

    // Forward-Forward update: one step of layer-local learning.
    // pos_input: batch x in_features (positive samples)
    // neg_input: batch x in_features (negative samples)
    // lr: learning rate
    // Returns: the FF loss for this layer
    double ff_update(const Tensor& pos_input, const Tensor& neg_input, double lr);

    std::vector<Tensor*> parameters() { return {&weights, &biases}; }
    std::vector<Tensor*> gradients() { return {&grad_weights, &grad_biases}; }

private:
    // Forward without caching (for internal use during training)
    Tensor forward_raw(const Tensor& input);

    // Layer-normalize the output (per-sample normalization)
    Tensor layer_normalize(const Tensor& x) const;
};

// ============================================================
// Forward-Forward Network
// ============================================================
// Implements Hinton's Forward-Forward algorithm for MNIST classification.
//
// Architecture:
//   Multiple FFLayers stacked, each trained with layer-local FF learning.
//   For classification, a label is embedded into the first 10 pixels of
//   the input image. During inference, all 10 possible labels are tried,
//   and the label producing the highest total goodness is selected.
//
// Label embedding:
//   The first 10 pixels of the image are replaced with a one-hot
//   encoding of the label (scaled to match image intensity range).
//
// Training procedure:
//   1. Create positive samples: real image with correct label embedded
//   2. Create negative samples: real image with random wrong label embedded
//   3. Each layer independently tries to push positive goodness > theta
//      and negative goodness < theta
//
// Default config: layers [784, 500, 500], num_classes=10, theta=2.0
// ============================================================
class ForwardForwardNetwork {
public:
    std::vector<FFLayer> layers;
    size_t num_classes;
    size_t input_dim;
    std::mt19937* rng_ptr;

    ForwardForwardNetwork(const std::vector<size_t>& layer_sizes, size_t num_classes,
                          double threshold, std::mt19937& rng);

    // Convenience constructor for MNIST defaults
    ForwardForwardNetwork(std::mt19937& rng);

    // Train one step on a batch.
    // images: batch x 784 (raw pixel values)
    // labels: batch x 1 (integer class labels as doubles)
    // lr: learning rate
    // Returns: average FF loss across all layers
    double train_step(const Tensor& images, const Tensor& labels, double lr);

    // Predict labels for a batch of images.
    // images: batch x 784
    // Returns: vector of predicted class labels
    std::vector<int> predict(const Tensor& images);

    // Predict and return goodness scores for all labels.
    // images: batch x 784
    // Returns: batch x num_classes (total goodness per label per sample)
    Tensor predict_goodness(const Tensor& images);

    // Embed a correct label into the image (one-hot in first 10 pixels).
    // images: batch x 784
    // labels: batch x 1 (integer class labels)
    // Returns: batch x 784
    static Tensor embed_label(const Tensor& images, const Tensor& labels);

    // Embed a random wrong label into the image.
    // images: batch x 784
    // labels: batch x 1 (correct labels, to avoid)
    // rng: random number generator
    // Returns: batch x 784
    static Tensor embed_wrong_label(const Tensor& images, const Tensor& labels,
                                     std::mt19937& rng);

    // Get all trainable parameters across all layers
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> all_gradients();
    void zero_grad();
};
