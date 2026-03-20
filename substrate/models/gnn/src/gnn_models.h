#pragma once

#include "../../../core/nn/module.h"
#include "../../../core/losses/losses.h"
#include <random>
#include <vector>
#include <utility>
#include <cmath>

// ============================================================
// Graph - Adjacency list / matrix representation of a graph
// ============================================================
struct Graph {
    Tensor node_features;                      // N x F
    std::vector<std::pair<int, int>> edges;    // edge list
    Tensor adjacency;                          // N x N (dense)
    int num_nodes;
};

// ============================================================
// GCNLayer - Graph Convolutional Network layer
// Implements: H' = sigma(A_norm @ H @ W + b)
// where A_norm = D^{-1/2} (A + I) D^{-1/2} (precomputed outside)
// ============================================================
class GCNLayer {
    Tensor weights;        // in_features x out_features
    Tensor bias;           // 1 x out_features
    Tensor grad_weights;   // in_features x out_features
    Tensor grad_bias;      // 1 x out_features

    // Cached values for backward pass
    Tensor cached_input;   // N x in_features (input node features H)
    Tensor cached_adj_norm; // N x N (normalized adjacency)
    Tensor cached_AH;      // A_norm @ H: N x in_features

    size_t in_features, out_features;

public:
    GCNLayer() : in_features(0), out_features(0) {}
    GCNLayer(size_t in_f, size_t out_f, std::mt19937& rng);

    // Forward: output = A_norm @ X @ W + b   (N x out_features)
    // A_norm: N x N, X: N x in_features
    Tensor forward(const Tensor& X, const Tensor& A_norm);

    // Backward: given grad of output (N x out_features), compute grad_weights,
    // grad_bias, and return grad w.r.t. X (N x in_features)
    Tensor backward(const Tensor& grad, const Tensor& A_norm);

    std::vector<Tensor*> parameters() { return {&weights, &bias}; }
    std::vector<Tensor*> gradients() { return {&grad_weights, &grad_bias}; }
};

// ============================================================
// GATLayer - Graph Attention Network layer
// Implements single-head attention:
//   h'_i = sigma( sum_j alpha_{ij} W h_j )
//   alpha_{ij} = softmax_j( LeakyReLU(a_left^T Wh_i + a_right^T Wh_j) )
// Only computes attention for connected nodes (adj[i][j] > 0 or i==j).
// ============================================================
class GATLayer {
    Tensor W;             // in_features x out_features
    Tensor a_left;        // out_features x 1 (attention vector for source)
    Tensor a_right;       // out_features x 1 (attention vector for target)
    Tensor grad_W;
    Tensor grad_a_left;
    Tensor grad_a_right;

    // Cached values for backward
    Tensor cached_input;  // N x in_features
    Tensor cached_Wh;     // N x out_features (W @ h for each node)
    Tensor cached_attn;   // N x N (attention coefficients after softmax)

    size_t in_features, out_features;
    double negative_slope; // for LeakyReLU

public:
    GATLayer() : in_features(0), out_features(0), negative_slope(0.2) {}
    GATLayer(size_t in_f, size_t out_f, std::mt19937& rng, double neg_slope = 0.2);

    // Forward: X: N x in_features, adj: N x N binary mask
    // Returns: N x out_features
    Tensor forward(const Tensor& X, const Tensor& adj);

    // Backward: given grad of output (N x out_features), return grad w.r.t. X
    Tensor backward(const Tensor& grad, const Tensor& adj);

    std::vector<Tensor*> parameters() { return {&W, &a_left, &a_right}; }
    std::vector<Tensor*> gradients() { return {&grad_W, &grad_a_left, &grad_a_right}; }
};

// ============================================================
// GNNClassifier
// GNN-based classifier for MNIST images.
//
// Strategy: treat each 28x28 image as a graph where each pixel is a node
// connected to its 8 neighbors (grid graph). Node features are pixel values
// (1-dimensional). Run GCN layers to propagate information, then global
// mean pool over all nodes and classify with a linear layer.
//
// For efficiency, the adjacency matrix is precomputed once since every
// MNIST image uses the same 28x28 grid structure.
// ============================================================
class GNNClassifier {
    std::vector<GCNLayer> gcn_layers;
    Linear attn_fc;       // attention readout: hidden -> 1
    Linear fc;            // classification: hidden -> num_classes
    size_t num_nodes;
    size_t num_classes;
    size_t hidden_size;
    Tensor adj_norm;      // precomputed normalized adjacency

    // Cached values for backward
    std::vector<Tensor> cached_gcn_inputs;  // input to each GCN layer (after act)
    Tensor cached_final_H;                   // node features before readout
    Tensor cached_attn_weights;              // attention weights (num_nodes x 1)
    Tensor cached_pooled;                    // after attention readout: 1 x hidden

    // Adam optimizer state
    std::vector<Tensor> adam_m_, adam_v_;
    int adam_t_;

public:
    GNNClassifier(size_t node_features, size_t hidden_size, size_t num_classes,
                  size_t num_gcn_layers, std::mt19937& rng,
                  size_t grid_h = 28, size_t grid_w = 28);

    // Create an 8-connected grid adjacency matrix with self-loops.
    // Returns: (h*w) x (h*w) binary adjacency with self-loops.
    static Tensor create_grid_adjacency(size_t h, size_t w);

    // Symmetric normalization: D^{-1/2} A D^{-1/2}
    // where A already includes self-loops.
    static Tensor normalize_adjacency(const Tensor& adj);

    // Forward pass for a single image.
    // image: 784-element column vector (784 x 1) or (1 x 784)
    // Returns: 1 x num_classes (logits)
    Tensor forward_single(const Tensor& image);

    // Predict class labels for a batch of images.
    // images: batch x 784
    // Returns: vector of predicted class indices
    std::vector<int> predict(const Tensor& images);

    // Training step: forward + backward + SGD update for a batch.
    // images: batch x 784, labels: class indices
    void train_step(const Tensor& images, const std::vector<int>& labels, double lr);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();
};
