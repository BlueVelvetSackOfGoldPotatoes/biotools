#pragma once
#include "../tensor/tensor.h"
#include <vector>
#include <cstdint>
#include <string>
#include <random>
#include <map>
#include <set>
#include <fstream>

// MNIST dataset
struct MNISTData {
    Tensor images;                 // N x 784, normalized [0,1]
    std::vector<uint8_t> labels;   // raw labels
    Tensor one_hot;                // N x 10
};

class MNISTLoader {
public:
    static MNISTData load(const std::string& image_path, const std::string& label_path);
private:
    static uint32_t read_u32(std::ifstream& f);
    static Tensor load_images(const std::string& path);
    static std::vector<uint8_t> load_labels(const std::string& path);
    static Tensor to_one_hot(const std::vector<uint8_t>& labels, int num_classes);
};

// Generic batch extraction
Tensor extract_batch(const Tensor& data, const std::vector<size_t>& indices,
                     size_t start, size_t batch_size);

// Simple DataLoader for iterating batches
class DataLoader {
public:
    Tensor data;
    Tensor targets;
    size_t batch_size;
    bool shuffle;
    bool drop_last;
    std::mt19937& rng;
    std::vector<size_t> indices;

    DataLoader(const Tensor& data, const Tensor& targets, size_t batch_size,
               bool shuffle, std::mt19937& rng, bool drop_last = true);

    void reset(); // re-shuffle if needed
    size_t num_batches() const;

    struct Batch {
        Tensor data;
        Tensor targets;
    };

    Batch get_batch(size_t batch_idx) const;
};

// Sequence data for RNN/LSTM/Transformer
struct TextData {
    std::vector<int> token_ids;
    std::map<char, int> char_to_idx;
    std::map<int, char> idx_to_char;
    int vocab_size;
};

TextData load_text_file(const std::string& path, int max_chars = -1);

// Create sequence batches from text
struct SeqBatch {
    Tensor inputs;  // batch x seq_len
    Tensor targets; // batch x seq_len
};
std::vector<SeqBatch> create_sequence_batches(const TextData& text, int seq_len,
                                               int batch_size);
