// modulus_datapipes.h - Data pipeline components
// Port of Modulus datapipes: data loading, batching, normalization
//
// C++17, no external dependencies.

#ifndef MODULUS_DATAPIPES_H
#define MODULUS_DATAPIPES_H

#include "modulus.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace modulus {
namespace datapipes {

// ============================================================
// Dataset - container for input/output pairs
// ============================================================
class Dataset {
public:
    std::vector<Tensor> inputs;
    std::vector<Tensor> targets;

    int size() const { return (int)inputs.size(); }

    void add(const Tensor& input, const Tensor& target) {
        inputs.push_back(input);
        targets.push_back(target);
    }

    // Split into train/val/test
    struct Split {
        std::vector<Tensor> train_inputs, train_targets;
        std::vector<Tensor> val_inputs, val_targets;
        std::vector<Tensor> test_inputs, test_targets;
        int train_size() const { return (int)train_inputs.size(); }
        int val_size() const { return (int)val_inputs.size(); }
        int test_size() const { return (int)test_inputs.size(); }
    };

    Split split(double train_ratio = 0.8, double val_ratio = 0.1) const {
        int n = size();
        int n_train = (int)(n * train_ratio);
        int n_val = (int)(n * val_ratio);

        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), global_rng());

        Split s;
        for (int i = 0; i < n_train; ++i) {
            s.train_inputs.push_back(inputs[indices[i]]);
            s.train_targets.push_back(targets[indices[i]]);
        }
        for (int i = n_train; i < n_train + n_val; ++i) {
            s.val_inputs.push_back(inputs[indices[i]]);
            s.val_targets.push_back(targets[indices[i]]);
        }
        for (int i = n_train + n_val; i < n; ++i) {
            s.test_inputs.push_back(inputs[indices[i]]);
            s.test_targets.push_back(targets[indices[i]]);
        }
        return s;
    }
};

// ============================================================
// DataLoader - batching and shuffling
// ============================================================
class DataLoader {
public:
    const Dataset& dataset;
    int batch_size;
    bool shuffle;
    std::vector<int> indices;
    int current_idx;

    DataLoader(const Dataset& ds, int bs = 32, bool shuf = true)
        : dataset(ds), batch_size(bs), shuffle(shuf), current_idx(0) {
        reset();
    }

    void reset() {
        indices.resize(dataset.size());
        std::iota(indices.begin(), indices.end(), 0);
        if (shuffle) std::shuffle(indices.begin(), indices.end(), global_rng());
        current_idx = 0;
    }

    bool has_next() const { return current_idx < (int)indices.size(); }

    // Get next batch
    struct Batch {
        std::vector<Tensor> inputs;
        std::vector<Tensor> targets;
        int size() const { return (int)inputs.size(); }
    };

    Batch next_batch() {
        Batch batch;
        int end = std::min(current_idx + batch_size, (int)indices.size());
        for (int i = current_idx; i < end; ++i) {
            batch.inputs.push_back(dataset.inputs[indices[i]]);
            batch.targets.push_back(dataset.targets[indices[i]]);
        }
        current_idx = end;
        return batch;
    }

    int num_batches() const {
        return (dataset.size() + batch_size - 1) / batch_size;
    }
};

// ============================================================
// Data normalization
// ============================================================
class Normalizer {
public:
    Tensor mean, std_dev;
    double eps;
    bool fitted;

    Normalizer() : eps(1e-8), fitted(false) {}

    // Fit normalization statistics from data
    void fit(const std::vector<Tensor>& data) {
        if (data.empty()) return;
        int dim = data[0].shape.back();
        int n = (int)data.size();
        int samples_per = data[0].numel() / dim;

        mean = Tensor::zeros({1, dim});
        std_dev = Tensor::zeros({1, dim});

        // Compute mean
        int total_count = 0;
        for (auto& t : data) {
            int n_pts = t.numel() / dim;
            for (int p = 0; p < n_pts; ++p)
                for (int d = 0; d < dim; ++d)
                    mean.data[d] += t.data[p * dim + d];
            total_count += n_pts;
        }
        for (int d = 0; d < dim; ++d) mean.data[d] /= total_count;

        // Compute std
        for (auto& t : data) {
            int n_pts = t.numel() / dim;
            for (int p = 0; p < n_pts; ++p)
                for (int d = 0; d < dim; ++d) {
                    double diff = t.data[p * dim + d] - mean.data[d];
                    std_dev.data[d] += diff * diff;
                }
        }
        for (int d = 0; d < dim; ++d)
            std_dev.data[d] = std::sqrt(std_dev.data[d] / total_count);

        fitted = true;
    }

    // Normalize a tensor
    Tensor normalize(const Tensor& x) const {
        if (!fitted) return x;
        int dim = mean.numel();
        Tensor out(x.shape);
        int n_pts = x.numel() / dim;
        for (int p = 0; p < n_pts; ++p)
            for (int d = 0; d < dim; ++d)
                out.data[p * dim + d] = (x.data[p * dim + d] - mean.data[d]) /
                                         (std_dev.data[d] + eps);
        return out;
    }

    // Denormalize a tensor
    Tensor denormalize(const Tensor& x) const {
        if (!fitted) return x;
        int dim = mean.numel();
        Tensor out(x.shape);
        int n_pts = x.numel() / dim;
        for (int p = 0; p < n_pts; ++p)
            for (int d = 0; d < dim; ++d)
                out.data[p * dim + d] = x.data[p * dim + d] * (std_dev.data[d] + eps) +
                                         mean.data[d];
        return out;
    }
};

// ============================================================
// Min-Max normalization
// ============================================================
class MinMaxNormalizer {
public:
    Tensor min_vals, max_vals;
    double eps;
    bool fitted;

    MinMaxNormalizer() : eps(1e-8), fitted(false) {}

    void fit(const std::vector<Tensor>& data) {
        if (data.empty()) return;
        int dim = data[0].shape.back();

        min_vals = Tensor({1, dim}, 1e30);
        max_vals = Tensor({1, dim}, -1e30);

        for (auto& t : data) {
            int n_pts = t.numel() / dim;
            for (int p = 0; p < n_pts; ++p)
                for (int d = 0; d < dim; ++d) {
                    min_vals.data[d] = std::min(min_vals.data[d], t.data[p * dim + d]);
                    max_vals.data[d] = std::max(max_vals.data[d], t.data[p * dim + d]);
                }
        }
        fitted = true;
    }

    Tensor normalize(const Tensor& x) const {
        if (!fitted) return x;
        int dim = min_vals.numel();
        Tensor out(x.shape);
        int n_pts = x.numel() / dim;
        for (int p = 0; p < n_pts; ++p)
            for (int d = 0; d < dim; ++d) {
                double range = max_vals.data[d] - min_vals.data[d] + eps;
                out.data[p * dim + d] = (x.data[p * dim + d] - min_vals.data[d]) / range;
            }
        return out;
    }

    Tensor denormalize(const Tensor& x) const {
        if (!fitted) return x;
        int dim = min_vals.numel();
        Tensor out(x.shape);
        int n_pts = x.numel() / dim;
        for (int p = 0; p < n_pts; ++p)
            for (int d = 0; d < dim; ++d) {
                double range = max_vals.data[d] - min_vals.data[d] + eps;
                out.data[p * dim + d] = x.data[p * dim + d] * range + min_vals.data[d];
            }
        return out;
    }
};

// ============================================================
// Data augmentation
// ============================================================
namespace augmentation {

// Add Gaussian noise
inline Tensor add_noise(const Tensor& x, double std_dev = 0.01) {
    Tensor out(x.shape);
    std::normal_distribution<double> noise(0, std_dev);
    for (int i = 0; i < x.numel(); ++i)
        out.data[i] = x.data[i] + noise(global_rng());
    return out;
}

// Random horizontal flip for 2D data (batch, ch, h, w)
inline Tensor random_hflip(const Tensor& x, double prob = 0.5) {
    std::uniform_real_distribution<double> dist(0, 1);
    if (dist(global_rng()) > prob) return x;
    if (x.ndim() != 4) return x;
    int batch = x.shape[0], ch = x.shape[1], h = x.shape[2], w = x.shape[3];
    Tensor out(x.shape);
    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < ch; ++c)
            for (int y = 0; y < h; ++y)
                for (int xi = 0; xi < w; ++xi)
                    out.data[((b * ch + c) * h + y) * w + xi] =
                        x.data[((b * ch + c) * h + y) * w + (w - 1 - xi)];
    return out;
}

// Random scaling
inline Tensor random_scale(const Tensor& x, double lo = 0.9, double hi = 1.1) {
    std::uniform_real_distribution<double> dist(lo, hi);
    double scale = dist(global_rng());
    return x * scale;
}

} // namespace augmentation

// ============================================================
// CSV data loader (simple)
// ============================================================
namespace csv {

inline std::vector<std::vector<double>> read(const std::string& path,
                                              bool has_header = true) {
    std::vector<std::vector<double>> data;
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open CSV: " + path);

    std::string line;
    if (has_header) std::getline(file, line); // skip header

    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try { row.push_back(std::stod(cell)); }
            catch (...) { row.push_back(0.0); }
        }
        if (!row.empty()) data.push_back(row);
    }
    return data;
}

// Convert CSV data to Tensor
inline Tensor to_tensor(const std::vector<std::vector<double>>& data) {
    if (data.empty()) return Tensor();
    int rows = (int)data.size();
    int cols = (int)data[0].size();
    Tensor t({rows, cols});
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols && j < (int)data[i].size(); ++j)
            t.at2(i, j) = data[i][j];
    return t;
}

} // namespace csv

} // namespace datapipes
} // namespace modulus

#endif // MODULUS_DATAPIPES_H
