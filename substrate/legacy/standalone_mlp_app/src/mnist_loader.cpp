#include "mnist_loader.h"
#include <stdexcept>
#include <iostream>

uint32_t MNISTLoader::read_uint32_be(std::ifstream& stream) {
    uint8_t bytes[4];
    stream.read(reinterpret_cast<char*>(bytes), 4);
    return (static_cast<uint32_t>(bytes[0]) << 24) |
           (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8)  |
           (static_cast<uint32_t>(bytes[3]));
}

Matrix MNISTLoader::load_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open image file: " + path);
    }

    uint32_t magic = read_uint32_be(file);
    if (magic != 2051) {
        throw std::runtime_error("Invalid image file magic number: " + std::to_string(magic));
    }

    uint32_t num_images = read_uint32_be(file);
    uint32_t num_rows = read_uint32_be(file);
    uint32_t num_cols = read_uint32_be(file);
    size_t pixels_per_image = num_rows * num_cols;

    std::cout << "Loading " << num_images << " images ("
              << num_rows << "x" << num_cols << ")..." << std::endl;

    Matrix images(num_images, pixels_per_image);

    // Read all pixel data at once
    std::vector<uint8_t> buffer(num_images * pixels_per_image);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

    // Normalize to [0, 1]
    for (size_t i = 0; i < buffer.size(); ++i) {
        images.data[i] = static_cast<double>(buffer[i]) / 255.0;
    }

    return images;
}

std::vector<uint8_t> MNISTLoader::load_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open label file: " + path);
    }

    uint32_t magic = read_uint32_be(file);
    if (magic != 2049) {
        throw std::runtime_error("Invalid label file magic number: " + std::to_string(magic));
    }

    uint32_t num_labels = read_uint32_be(file);

    std::cout << "Loading " << num_labels << " labels..." << std::endl;

    std::vector<uint8_t> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    return labels;
}

Matrix MNISTLoader::to_one_hot(const std::vector<uint8_t>& labels, int num_classes) {
    Matrix result(labels.size(), num_classes);
    for (size_t i = 0; i < labels.size(); ++i) {
        result(i, labels[i]) = 1.0;
    }
    return result;
}

MNISTDataset MNISTLoader::load(const std::string& image_path,
                                const std::string& label_path) {
    MNISTDataset dataset;
    dataset.images = load_images(image_path);
    dataset.labels = load_labels(label_path);
    dataset.one_hot_labels = to_one_hot(dataset.labels);
    return dataset;
}
