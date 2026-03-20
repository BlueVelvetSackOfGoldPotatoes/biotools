#pragma once
#include "matrix.h"
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>

struct MNISTDataset {
    Matrix images;              // num_samples x 784, normalized to [0,1]
    std::vector<uint8_t> labels;
    Matrix one_hot_labels;      // num_samples x 10
};

class MNISTLoader {
public:
    static MNISTDataset load(const std::string& image_path,
                             const std::string& label_path);

private:
    static uint32_t read_uint32_be(std::ifstream& stream);
    static Matrix load_images(const std::string& path);
    static std::vector<uint8_t> load_labels(const std::string& path);
    static Matrix to_one_hot(const std::vector<uint8_t>& labels, int num_classes = 10);
};
