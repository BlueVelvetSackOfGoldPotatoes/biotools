#pragma once
#include "network.h"
#include "mnist_loader.h"
#include <string>
#include <vector>

struct NeuronStats {
    double mean_activation;
    double max_activation;
    double activation_frequency; // fraction of inputs where neuron fires (> 0)
};

struct AblationResult {
    size_t neuron_index;
    double accuracy_drop;   // positive = neuron was helpful
    double loss_increase;
};

class Interpretability {
public:
    // Dump all weight matrices and bias vectors as CSV files
    void dump_weights(const Network& net, const std::string& prefix);

    // Dump first-layer weights reshaped as 28x28 images (one per neuron)
    void dump_weight_images(const Network& net, const std::string& prefix);

    // Compute weight magnitude statistics per layer
    void dump_weight_statistics(const Network& net, const std::string& prefix);

    // Record all intermediate activations for a batch of inputs
    void record_activations(Network& net, const Matrix& inputs,
                            const std::string& prefix);

    // Per-neuron activation statistics at a given layer
    std::vector<NeuronStats> compute_neuron_stats(
        Network& net, const Matrix& dataset, size_t layer_index);

    // Find top-K inputs that maximally activate a specific neuron
    std::vector<size_t> max_activating_inputs(
        Network& net, const Matrix& dataset,
        size_t layer_index, size_t neuron_index, size_t top_k = 10);

    // Zero-ablation: set neuron output to zero, return accuracy delta
    double ablate_neuron_zero(Network& net, const MNISTDataset& test_data,
                              size_t layer_index, size_t neuron_index);

    // Ablation sweep: ablate every neuron in a layer one-at-a-time
    std::vector<AblationResult> ablation_sweep(
        Network& net, const MNISTDataset& test_data, size_t layer_index);

    // Class-conditional mean activations: 10 x num_neurons matrix
    Matrix class_conditional_activations(
        Network& net, const MNISTDataset& dataset, size_t layer_index);

    // Selectivity index per neuron in a layer
    std::vector<double> neuron_selectivity(
        Network& net, const MNISTDataset& dataset, size_t layer_index);

    // Run all analyses
    void full_analysis(Network& net, const MNISTDataset& test_data,
                       const std::string& output_dir);
};
