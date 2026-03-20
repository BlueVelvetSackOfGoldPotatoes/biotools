#include "matrix.h"
#include "activation.h"
#include "loss.h"
#include "layer.h"
#include "network.h"
#include "optimizer.h"
#include "mnist_loader.h"
#include "interpretability.h"

#include <iostream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <chrono>

// Extract a mini-batch using shuffled indices
static Matrix extract_batch(const Matrix& data, const std::vector<size_t>& indices,
                            size_t start, size_t batch_size) {
    Matrix batch(batch_size, data.cols);
    for (size_t i = 0; i < batch_size; ++i) {
        size_t idx = indices[start + i];
        for (size_t j = 0; j < data.cols; ++j) {
            batch(i, j) = data(idx, j);
        }
    }
    return batch;
}

static double compute_accuracy(const std::vector<int>& preds,
                               const std::vector<uint8_t>& labels) {
    size_t correct = 0;
    for (size_t i = 0; i < preds.size(); ++i) {
        if (preds[i] == static_cast<int>(labels[i])) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(preds.size());
}

int main() {
    // --- Configuration ---
    const size_t EPOCHS = 20;
    const size_t BATCH_SIZE = 64;
    const double LEARNING_RATE = 0.01;
    const unsigned SEED = 42;

    std::mt19937 rng(SEED);

    std::cout << "=== Pure C++ MNIST MLP - Mechanistic Interpretability Benchmark ===" << std::endl;
    std::cout << "Architecture: 784 -> 128 -> 64 -> 10" << std::endl;
    std::cout << "Epochs: " << EPOCHS << ", Batch size: " << BATCH_SIZE
              << ", LR: " << LEARNING_RATE << std::endl;
    std::cout << std::endl;

    // --- Load data ---
    std::cout << "Loading MNIST dataset..." << std::endl;
    auto train_data = MNISTLoader::load("data/train-images-idx3-ubyte",
                                         "data/train-labels-idx1-ubyte");
    auto test_data = MNISTLoader::load("data/t10k-images-idx3-ubyte",
                                        "data/t10k-labels-idx1-ubyte");
    std::cout << "Training samples: " << train_data.images.rows << std::endl;
    std::cout << "Test samples: " << test_data.images.rows << std::endl;
    std::cout << std::endl;

    // --- Build network ---
    Network net;
    net.add_layer(std::make_unique<DenseLayer>(784, 128, "dense1", rng));
    net.add_layer(std::make_unique<ReLULayer>());
    net.add_layer(std::make_unique<DenseLayer>(128, 64, "dense2", rng));
    net.add_layer(std::make_unique<ReLULayer>());
    net.add_layer(std::make_unique<DenseLayer>(64, 10, "dense3", rng));

    SGD optimizer(LEARNING_RATE);
    Interpretability interp;

    // --- Training ---
    std::cout << "Training..." << std::endl;
    auto train_start = std::chrono::high_resolution_clock::now();

    for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        double epoch_loss = 0.0;
        size_t num_batches = 0;

        // Shuffle training indices
        std::vector<size_t> indices(train_data.images.rows);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (size_t batch_start = 0;
             batch_start + BATCH_SIZE <= train_data.images.rows;
             batch_start += BATCH_SIZE) {

            // Extract mini-batch
            Matrix batch_images = extract_batch(train_data.images, indices,
                                                 batch_start, BATCH_SIZE);
            Matrix batch_labels = extract_batch(train_data.one_hot_labels, indices,
                                                 batch_start, BATCH_SIZE);

            // Forward
            Matrix logits = net.forward(batch_images);
            Matrix probs = Activation::softmax(logits);
            double loss = net.compute_loss(probs, batch_labels);
            epoch_loss += loss;

            // Backward
            net.backward(probs, batch_labels);

            // Update
            optimizer.step(net);
            num_batches++;
        }

        // Evaluate on test set
        auto test_preds = net.predict(test_data.images);
        double accuracy = compute_accuracy(test_preds, test_data.labels);

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_ms = std::chrono::duration<double, std::milli>(epoch_end - epoch_start).count();

        std::cout << "Epoch " << std::setw(2) << (epoch + 1)
                  << "/" << EPOCHS
                  << " | Loss: " << std::fixed << std::setprecision(4)
                  << (epoch_loss / num_batches)
                  << " | Test Acc: " << std::setprecision(2)
                  << (accuracy * 100.0) << "%"
                  << " | Time: " << std::setprecision(0) << epoch_ms << "ms"
                  << std::endl;

        // Periodic interpretability snapshots
        if ((epoch + 1) % 10 == 0 || epoch == EPOCHS - 1) {
            std::string prefix = "output/epoch_" + std::to_string(epoch + 1);
            interp.dump_weights(net, prefix);
            interp.dump_weight_statistics(net, prefix);
        }
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(train_end - train_start).count();
    std::cout << "\nTraining complete in " << std::fixed << std::setprecision(1)
              << total_s << "s" << std::endl;

    // --- Final interpretability analysis ---
    std::cout << "\nRunning mechanistic interpretability analysis..." << std::endl;
    interp.full_analysis(net, test_data, "output/final");

    return 0;
}
