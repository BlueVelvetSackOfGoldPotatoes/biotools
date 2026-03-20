#include "dataloader.h"
#include <fstream>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#define DATALOADER_OMP_PAR_FOR _Pragma("omp parallel for schedule(static)")
#else
#define DATALOADER_OMP_PAR_FOR
#endif

// ============================================================
// MNIST Loader
// ============================================================
uint32_t MNISTLoader::read_u32(std::ifstream& f) {
    uint32_t val = 0;
    unsigned char bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    if (f.gcount() != 4) {
        throw std::runtime_error("MNISTLoader: failed to read 4-byte header field");
    }
    val = ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) | ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3];
    return val;
}

Tensor MNISTLoader::load_images(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("MNISTLoader: cannot open image file '" + path + "'");
    }
    uint32_t magic = read_u32(f);
    if (magic != 2051) {
        throw std::runtime_error("MNISTLoader: invalid image magic number in '" + path + "'");
    }
    uint32_t num = read_u32(f);
    uint32_t rows = read_u32(f);
    uint32_t cols = read_u32(f);
    if (num == 0 || rows == 0 || cols == 0) {
        throw std::runtime_error("MNISTLoader: invalid empty image dimensions in '" + path + "'");
    }
    std::cout << "Loading " << num << " images (" << rows << "x" << cols << ")..." << std::endl;

    const std::size_t pixels = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (pixels == 0) {
        throw std::runtime_error("MNISTLoader: invalid image pixel size in '" + path + "'");
    }
    const std::size_t expected_bytes = static_cast<std::size_t>(num) * pixels;
    std::vector<unsigned char> buf(expected_bytes);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    if (f.gcount() != static_cast<std::streamsize>(expected_bytes)) {
        throw std::runtime_error("MNISTLoader: incomplete image payload read in '" + path + "'");
    }

    Tensor images(num, pixels);
    for (size_t i = 0; i < num * pixels; ++i)
        images.data[i] = buf[i] / 255.0;
    return images;
}

std::vector<uint8_t> MNISTLoader::load_labels(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("MNISTLoader: cannot open label file '" + path + "'");
    }
    uint32_t magic = read_u32(f);
    if (magic != 2049) {
        throw std::runtime_error("MNISTLoader: invalid label magic number in '" + path + "'");
    }
    uint32_t num = read_u32(f);
    if (num == 0) {
        throw std::runtime_error("MNISTLoader: zero labels in '" + path + "'");
    }
    std::cout << "Loading " << num << " labels..." << std::endl;

    std::vector<uint8_t> labels(num);
    f.read(reinterpret_cast<char*>(labels.data()), static_cast<std::streamsize>(num));
    if (f.gcount() != static_cast<std::streamsize>(num)) {
        throw std::runtime_error("MNISTLoader: incomplete label payload read in '" + path + "'");
    }
    return labels;
}

Tensor MNISTLoader::to_one_hot(const std::vector<uint8_t>& labels, int nc) {
    if (nc <= 0) {
        throw std::runtime_error("MNISTLoader::to_one_hot: num_classes must be > 0");
    }
    Tensor oh(labels.size(), nc);
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] >= static_cast<uint8_t>(nc)) {
            throw std::runtime_error("MNISTLoader::to_one_hot: label out of range");
        }
        oh(i, labels[i]) = 1.0;
    }
    return oh;
}

MNISTData MNISTLoader::load(const std::string& image_path, const std::string& label_path) {
    MNISTData d;
    d.images = load_images(image_path);
    d.labels = load_labels(label_path);
    if (d.images.rows != d.labels.size()) {
        throw std::runtime_error("MNISTLoader::load: image/label count mismatch");
    }
    d.one_hot = to_one_hot(d.labels, 10);
    return d;
}

// ============================================================
// Batch extraction
// ============================================================
Tensor extract_batch(const Tensor& data, const std::vector<size_t>& indices,
                     size_t start, size_t batch_size) {
    if (batch_size == 0) {
        throw std::runtime_error("extract_batch: batch_size must be > 0");
    }
    if (start + batch_size > indices.size()) {
        throw std::out_of_range("extract_batch: index range out of bounds");
    }
    Tensor batch(batch_size, data.cols);
    DATALOADER_OMP_PAR_FOR
    for (size_t i = 0; i < batch_size; ++i) {
        size_t idx = indices[start + i];
        for (size_t j = 0; j < data.cols; ++j)
            batch(i, j) = data(idx, j);
    }
    return batch;
}

// ============================================================
// DataLoader
// ============================================================
DataLoader::DataLoader(const Tensor& d, const Tensor& t, size_t bs, bool shuf, std::mt19937& r, bool drop_last_batches)
    : data(d), targets(t), batch_size(bs), shuffle(shuf), drop_last(drop_last_batches), rng(r) {
    if (data.rows != targets.rows) {
        throw std::runtime_error("DataLoader: data/target row mismatch");
    }
    if (batch_size == 0) {
        throw std::runtime_error("DataLoader: batch_size must be > 0");
    }
    indices.resize(d.rows);
    std::iota(indices.begin(), indices.end(), 0);
    if (shuffle) std::shuffle(indices.begin(), indices.end(), rng);
}

void DataLoader::reset() {
    if (shuffle) std::shuffle(indices.begin(), indices.end(), rng);
}

size_t DataLoader::num_batches() const {
    if (batch_size == 0) return 0;
    if (drop_last) return data.rows / batch_size;
    return (data.rows + batch_size - 1) / batch_size;
}

DataLoader::Batch DataLoader::get_batch(size_t batch_idx) const {
    const size_t total = num_batches();
    if (batch_idx >= total) {
        throw std::out_of_range("DataLoader::get_batch batch_idx out of range");
    }
    size_t start = batch_idx * batch_size;
    const size_t available = data.rows > start ? (data.rows - start) : 0;
    const size_t this_batch = std::min(batch_size, available);
    if (this_batch == 0) {
        throw std::out_of_range("DataLoader::get_batch computed zero-sized batch");
    }
    Batch b;
    b.data = extract_batch(data, indices, start, this_batch);
    b.targets = extract_batch(targets, indices, start, this_batch);
    return b;
}

// ============================================================
// Text data loading
// ============================================================
TextData load_text_file(const std::string& path, int max_chars) {
    TextData td;
    std::ifstream f(path);
    std::string content;
    if (f.is_open()) {
        std::string line;
        while (std::getline(f, line)) {
            content += line + "\n";
            if (max_chars > 0 && (int)content.size() >= max_chars) break;
        }
    }
    if (max_chars > 0 && (int)content.size() > max_chars)
        content = content.substr(0, max_chars);

    // Build vocabulary
    std::set<char> chars(content.begin(), content.end());
    int idx = 0;
    for (char c : chars) {
        td.char_to_idx[c] = idx;
        td.idx_to_char[idx] = c;
        idx++;
    }
    td.vocab_size = idx;

    // Tokenize
    td.token_ids.resize(content.size());
    for (size_t i = 0; i < content.size(); ++i)
        td.token_ids[i] = td.char_to_idx[content[i]];

    return td;
}

std::vector<SeqBatch> create_sequence_batches(const TextData& text, int seq_len, int batch_size) {
    std::vector<SeqBatch> batches;
    int total_seqs = ((int)text.token_ids.size() - 1) / seq_len;
    int num_batches = total_seqs / batch_size;

    for (int b = 0; b < num_batches; ++b) {
        SeqBatch sb;
        sb.inputs = Tensor(batch_size, seq_len);
        sb.targets = Tensor(batch_size, seq_len);
        for (int i = 0; i < batch_size; ++i) {
            int seq_idx = b * batch_size + i;
            int start = seq_idx * seq_len;
            for (int j = 0; j < seq_len; ++j) {
                sb.inputs(i, j) = text.token_ids[start + j];
                sb.targets(i, j) = text.token_ids[start + j + 1];
            }
        }
        batches.push_back(sb);
    }
    return batches;
}
