#include "interpretability.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <sstream>

// Helper: compute accuracy given predictions and labels
static double compute_accuracy(const std::vector<int>& preds,
                               const std::vector<uint8_t>& labels) {
    size_t correct = 0;
    for (size_t i = 0; i < preds.size(); ++i) {
        if (preds[i] == static_cast<int>(labels[i])) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(preds.size());
}

// Helper: forward pass and return activations at a specific layer
static Matrix get_activations_at_layer(Network& net, const Matrix& inputs,
                                       size_t layer_index) {
    net.forward(inputs);
    const auto& outputs = net.get_layer_outputs();
    // layer_outputs_[0] = input, layer_outputs_[i] = output of layer i-1
    // So output of layer layer_index is at index layer_index + 1
    return outputs[layer_index + 1];
}

void Interpretability::dump_weights(const Network& net, const std::string& prefix) {
    const auto& layers = net.get_layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i]->has_parameters()) {
            std::string w_path = prefix + "_" + layers[i]->name() + "_weights.csv";
            std::string b_path = prefix + "_" + layers[i]->name() + "_biases.csv";
            layers[i]->get_weights().save_csv(w_path);
            layers[i]->get_biases().save_csv(b_path);
        }
    }
}

void Interpretability::dump_weight_images(const Network& net, const std::string& prefix) {
    // First dense layer: weights are input_size(784) x output_size(128)
    // Each column is one neuron's weight vector -> reshape to 28x28
    const auto& layers = net.get_layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i]->has_parameters()) {
            const Matrix& W = layers[i]->get_weights();
            if (W.rows == 784) {
                // This is the first layer - dump each neuron as 28x28
                for (size_t n = 0; n < W.cols; ++n) {
                    Matrix img(28, 28);
                    for (size_t r = 0; r < 28; ++r) {
                        for (size_t c = 0; c < 28; ++c) {
                            img(r, c) = W(r * 28 + c, n);
                        }
                    }
                    std::string path = prefix + "_neuron_" + std::to_string(n) + ".csv";
                    img.save_csv(path);
                }
            }
            break; // only first dense layer
        }
    }
}

void Interpretability::dump_weight_statistics(const Network& net, const std::string& prefix) {
    std::ofstream file(prefix + "_weight_stats.csv");
    file << "layer,param,mean,stddev,min,max,l2_norm\n";

    const auto& layers = net.get_layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        if (!layers[i]->has_parameters()) continue;
        const Matrix& W = layers[i]->get_weights();

        double sum = 0, sum_sq = 0, mn = W.data[0], mx = W.data[0];
        for (size_t j = 0; j < W.data.size(); ++j) {
            double v = W.data[j];
            sum += v;
            sum_sq += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        double n = static_cast<double>(W.data.size());
        double mean = sum / n;
        double stddev = std::sqrt(sum_sq / n - mean * mean);
        double l2 = std::sqrt(sum_sq);

        file << layers[i]->name() << ",weights,"
             << mean << "," << stddev << "," << mn << "," << mx << "," << l2 << "\n";

        const Matrix& B = layers[i]->get_biases();
        sum = 0; sum_sq = 0; mn = B.data[0]; mx = B.data[0];
        for (size_t j = 0; j < B.data.size(); ++j) {
            double v = B.data[j];
            sum += v;
            sum_sq += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        n = static_cast<double>(B.data.size());
        mean = sum / n;
        stddev = std::sqrt(std::max(0.0, sum_sq / n - mean * mean));
        l2 = std::sqrt(sum_sq);

        file << layers[i]->name() << ",biases,"
             << mean << "," << stddev << "," << mn << "," << mx << "," << l2 << "\n";
    }
}

void Interpretability::record_activations(Network& net, const Matrix& inputs,
                                          const std::string& prefix) {
    net.forward(inputs);
    const auto& outputs = net.get_layer_outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::string label = (i == 0) ? "input" : "layer_" + std::to_string(i - 1);
        outputs[i].save_csv(prefix + "_" + label + ".csv");
    }
}

std::vector<NeuronStats> Interpretability::compute_neuron_stats(
    Network& net, const Matrix& dataset, size_t layer_index) {

    Matrix activations = get_activations_at_layer(net, dataset, layer_index);
    size_t num_neurons = activations.cols;
    size_t num_samples = activations.rows;

    std::vector<NeuronStats> stats(num_neurons);
    for (size_t j = 0; j < num_neurons; ++j) {
        double sum = 0.0;
        double mx = activations(0, j);
        size_t fire_count = 0;
        for (size_t i = 0; i < num_samples; ++i) {
            double v = activations(i, j);
            sum += v;
            if (v > mx) mx = v;
            if (v > 0.0) ++fire_count;
        }
        stats[j].mean_activation = sum / static_cast<double>(num_samples);
        stats[j].max_activation = mx;
        stats[j].activation_frequency = static_cast<double>(fire_count) / static_cast<double>(num_samples);
    }
    return stats;
}

std::vector<size_t> Interpretability::max_activating_inputs(
    Network& net, const Matrix& dataset,
    size_t layer_index, size_t neuron_index, size_t top_k) {

    Matrix activations = get_activations_at_layer(net, dataset, layer_index);

    // Collect (activation_value, sample_index) pairs
    std::vector<std::pair<double, size_t>> activation_index(activations.rows);
    for (size_t i = 0; i < activations.rows; ++i) {
        activation_index[i] = {activations(i, neuron_index), i};
    }

    // Partial sort for top-K
    size_t k = std::min(top_k, activation_index.size());
    std::partial_sort(activation_index.begin(),
                      activation_index.begin() + k,
                      activation_index.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<size_t> result(k);
    for (size_t i = 0; i < k; ++i) {
        result[i] = activation_index[i].second;
    }
    return result;
}

double Interpretability::ablate_neuron_zero(Network& net, const MNISTDataset& test_data,
                                            size_t layer_index, size_t neuron_index) {
    // Baseline accuracy
    auto baseline_preds = net.predict(test_data.images);
    double baseline_acc = compute_accuracy(baseline_preds, test_data.labels);

    // Forward up to the target layer, zero out the neuron, continue
    net.forward(test_data.images);
    const auto& outputs = net.get_layer_outputs();

    // Get activation after target layer
    Matrix modified = outputs[layer_index + 1];
    // Zero out the neuron column
    for (size_t i = 0; i < modified.rows; ++i) {
        modified(i, neuron_index) = 0.0;
    }

    // Continue forward from the layer after the target
    Matrix current = modified;
    for (size_t i = layer_index + 1; i < net.num_layers(); ++i) {
        current = net.get_layer(i).forward(current);
    }

    // Compute ablated predictions
    Matrix probs = Activation::softmax(current);
    std::vector<int> ablated_preds(probs.rows);
    for (size_t i = 0; i < probs.rows; ++i) {
        int best = 0;
        double best_val = probs(i, 0);
        for (size_t j = 1; j < probs.cols; ++j) {
            if (probs(i, j) > best_val) {
                best_val = probs(i, j);
                best = static_cast<int>(j);
            }
        }
        ablated_preds[i] = best;
    }
    double ablated_acc = compute_accuracy(ablated_preds, test_data.labels);

    // Restore: re-run clean forward to reset caches
    net.forward(test_data.images);

    return baseline_acc - ablated_acc; // positive = neuron was helpful
}

std::vector<AblationResult> Interpretability::ablation_sweep(
    Network& net, const MNISTDataset& test_data, size_t layer_index) {

    // Get number of neurons from the target layer's output
    net.forward(test_data.images);
    const auto& outputs = net.get_layer_outputs();
    size_t num_neurons = outputs[layer_index + 1].cols;

    // Baseline
    auto baseline_preds = net.predict(test_data.images);
    double baseline_acc = compute_accuracy(baseline_preds, test_data.labels);
    double baseline_loss = net.compute_loss(
        Activation::softmax(net.get_layer_outputs().back()), test_data.one_hot_labels);

    std::vector<AblationResult> results(num_neurons);

    for (size_t n = 0; n < num_neurons; ++n) {
        // Forward and get activations
        net.forward(test_data.images);
        Matrix modified = net.get_layer_outputs()[layer_index + 1];

        // Zero out neuron n
        for (size_t i = 0; i < modified.rows; ++i) {
            modified(i, n) = 0.0;
        }

        // Continue forward
        Matrix current = modified;
        for (size_t i = layer_index + 1; i < net.num_layers(); ++i) {
            current = net.get_layer(i).forward(current);
        }

        Matrix probs = Activation::softmax(current);

        // Accuracy
        std::vector<int> preds(probs.rows);
        for (size_t i = 0; i < probs.rows; ++i) {
            int best = 0;
            double best_val = probs(i, 0);
            for (size_t j = 1; j < probs.cols; ++j) {
                if (probs(i, j) > best_val) {
                    best_val = probs(i, j);
                    best = static_cast<int>(j);
                }
            }
            preds[i] = best;
        }
        double abl_acc = compute_accuracy(preds, test_data.labels);

        // Loss
        CrossEntropyLoss loss_fn;
        double abl_loss = loss_fn.forward(probs, test_data.one_hot_labels);

        results[n] = {n, baseline_acc - abl_acc, abl_loss - baseline_loss};
    }

    // Restore
    net.forward(test_data.images);

    return results;
}

Matrix Interpretability::class_conditional_activations(
    Network& net, const MNISTDataset& dataset, size_t layer_index) {

    Matrix activations = get_activations_at_layer(net, dataset.images, layer_index);
    size_t num_neurons = activations.cols;

    Matrix result(10, num_neurons); // 10 digit classes x num_neurons
    std::vector<int> class_counts(10, 0);

    for (size_t i = 0; i < dataset.labels.size(); ++i) {
        int cls = dataset.labels[i];
        class_counts[cls]++;
        for (size_t j = 0; j < num_neurons; ++j) {
            result(cls, j) += activations(i, j);
        }
    }

    // Average
    for (int c = 0; c < 10; ++c) {
        if (class_counts[c] > 0) {
            for (size_t j = 0; j < num_neurons; ++j) {
                result(c, j) /= static_cast<double>(class_counts[c]);
            }
        }
    }

    return result;
}

std::vector<double> Interpretability::neuron_selectivity(
    Network& net, const MNISTDataset& dataset, size_t layer_index) {

    Matrix cca = class_conditional_activations(net, dataset, layer_index);
    size_t num_neurons = cca.cols;

    std::vector<double> selectivity(num_neurons);
    for (size_t j = 0; j < num_neurons; ++j) {
        double overall_mean = 0.0;
        double max_class_mean = cca(0, j);
        for (int c = 0; c < 10; ++c) {
            overall_mean += cca(c, j);
            if (cca(c, j) > max_class_mean) max_class_mean = cca(c, j);
        }
        overall_mean /= 10.0;
        const double epsilon = 1e-8;
        selectivity[j] = (max_class_mean - overall_mean) / (overall_mean + epsilon);
    }
    return selectivity;
}

void Interpretability::full_analysis(Network& net, const MNISTDataset& test_data,
                                     const std::string& output_dir) {
    std::cout << "\n=== Mechanistic Interpretability Analysis ===" << std::endl;

    // 1. Weight dumps
    std::cout << "Dumping weights..." << std::endl;
    dump_weights(net, output_dir + "/weights");
    dump_weight_images(net, output_dir + "/weight_images");
    dump_weight_statistics(net, output_dir + "/stats");

    // 2. Activation recording (first 200 test samples)
    std::cout << "Recording activations..." << std::endl;
    Matrix sample = test_data.images.slice_rows(0, std::min((size_t)200, test_data.images.rows));
    record_activations(net, sample, output_dir + "/activations");

    // 3. Neuron stats for each layer with parameters
    const auto& layers = net.get_layers();
    for (size_t i = 0; i < layers.size(); ++i) {
        if (!layers[i]->has_parameters()) continue;

        std::cout << "Computing neuron stats for " << layers[i]->name() << "..." << std::endl;
        auto stats = compute_neuron_stats(net, test_data.images, i);

        std::ofstream file(output_dir + "/neuron_stats_" + layers[i]->name() + ".csv");
        file << "neuron,mean_activation,max_activation,activation_frequency\n";
        size_t dead_count = 0;
        for (size_t j = 0; j < stats.size(); ++j) {
            file << j << "," << stats[j].mean_activation << ","
                 << stats[j].max_activation << ","
                 << stats[j].activation_frequency << "\n";
            if (stats[j].activation_frequency < 0.01) ++dead_count;
        }
        std::cout << "  Dead neurons (<1% firing): " << dead_count
                  << "/" << stats.size() << std::endl;
    }

    // 4. Class-conditional activations for ReLU layers (post-activation)
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i]->name() != "ReLU") continue;

        std::cout << "Class-conditional activations after layer " << i << "..." << std::endl;
        Matrix cca = class_conditional_activations(net, test_data, i);
        cca.save_csv(output_dir + "/class_conditional_layer_" + std::to_string(i) + ".csv");

        std::vector<double> sel = neuron_selectivity(net, test_data, i);
        std::ofstream sel_file(output_dir + "/selectivity_layer_" + std::to_string(i) + ".csv");
        sel_file << "neuron,selectivity\n";
        for (size_t j = 0; j < sel.size(); ++j) {
            sel_file << j << "," << sel[j] << "\n";
        }
    }

    // 5. Ablation sweep on first hidden layer (the ReLU after dense1)
    // Find the first ReLU layer
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i]->name() != "ReLU") continue;

        std::cout << "Ablation sweep on layer " << i << " ("
                  << layers[i]->name() << ")..." << std::endl;
        auto ablation = ablation_sweep(net, test_data, i);

        std::ofstream abl_file(output_dir + "/ablation_layer_" + std::to_string(i) + ".csv");
        abl_file << "neuron,accuracy_drop,loss_increase\n";
        for (const auto& r : ablation) {
            abl_file << r.neuron_index << "," << r.accuracy_drop << ","
                     << r.loss_increase << "\n";
        }

        // Report top-10 most important neurons
        std::vector<AblationResult> sorted_abl = ablation;
        std::sort(sorted_abl.begin(), sorted_abl.end(),
                  [](const auto& a, const auto& b) { return a.accuracy_drop > b.accuracy_drop; });
        std::cout << "  Top-10 most important neurons (by accuracy drop):" << std::endl;
        for (size_t j = 0; j < std::min((size_t)10, sorted_abl.size()); ++j) {
            std::cout << "    Neuron " << sorted_abl[j].neuron_index
                      << ": accuracy drop = " << std::fixed << std::setprecision(4)
                      << (sorted_abl[j].accuracy_drop * 100.0) << "%" << std::endl;
        }

        break; // only first ReLU layer for full sweep
    }

    std::cout << "=== Analysis complete. Results in " << output_dir << "/ ===" << std::endl;
}
