// esm.h - Comprehensive header-only C++17 port of ESM (Evolutionary Scale Modeling)
//
// Ported from: https://github.com/facebookresearch/esm
// Original: Meta Platforms, Inc. (MIT License)
//
// This reimplementation covers:
//   - Matrix / Tensor infrastructure (2D, 3D, 4D, 5D)
//   - Math utilities (GELU, softmax, layer norm, matmul, rotary embeddings)
//   - Alphabet (ESM-1, ESM-1b/ESM-2, MSA Transformer, Inverse Folding)
//   - Data loading (FastaBatchedDataset, BatchConverter, MSABatchConverter)
//   - Embeddings (Learned positional, Sinusoidal positional, Rotary)
//   - Multi-head attention (with rotary, bias_kv, causal mask, cross-attention)
//   - Axial attention (RowSelfAttention, ColumnSelfAttention)
//   - Transformer layers (standard, axial, encoder, decoder)
//   - ESM-1 / ESM-1b (ProteinBertModel)
//   - ESM-2 (with rotary embeddings, token dropout)
//   - MSA Transformer (axial transformer, row/column attention)
//   - Contact prediction head (symmetrize, APC, logistic regression)
//   - RobertaLMHead
//   - GVP modules (Geometric Vector Perceptron, GVP convolution, GVP conv layer)
//   - GVP Encoder and GVP Transformer (inverse folding)
//   - Transformer Decoder (autoregressive sequence design)
//   - ESMFold components (FoldingTrunk, TriangularSelfAttentionBlock, etc.)
//   - Pretrained model registry
//   - Feature extraction utilities

#ifndef ESM_H
#define ESM_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace esm {

// ===================================================================
// SECTION 1: Dense Tensor Types
// ===================================================================

struct Matrix {
    std::vector<float> data;
    int rows = 0;
    int cols = 0;

    Matrix() = default;
    Matrix(int r, int c) : data(static_cast<size_t>(r) * c, 0.0f), rows(r), cols(c) {}
    Matrix(int r, int c, float val) : data(static_cast<size_t>(r) * c, val), rows(r), cols(c) {}

    float& operator()(int r, int c) { return data[static_cast<size_t>(r) * cols + c]; }
    float  operator()(int r, int c) const { return data[static_cast<size_t>(r) * cols + c]; }

    float* row_ptr(int r) { return data.data() + static_cast<size_t>(r) * cols; }
    const float* row_ptr(int r) const { return data.data() + static_cast<size_t>(r) * cols; }

    int size() const { return rows * cols; }
    void fill(float v) { std::fill(data.begin(), data.end(), v); }

    void resize(int r, int c) {
        rows = r; cols = c;
        data.assign(static_cast<size_t>(r) * c, 0.0f);
    }

    void xavier_uniform(std::mt19937& rng, float gain = 1.0f) {
        float limit = gain * std::sqrt(6.0f / (rows + cols));
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (auto& v : data) v = dist(rng);
    }

    void normal_init(std::mt19937& rng, float mean = 0.0f, float stddev = 1.0f) {
        std::normal_distribution<float> dist(mean, stddev);
        for (auto& v : data) v = dist(rng);
    }

    void zeros() { fill(0.0f); }
    void ones() { fill(1.0f); }

    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result(j, i) = (*this)(i, j);
        return result;
    }
};

struct Tensor3D {
    std::vector<float> data;
    int d0 = 0, d1 = 0, d2 = 0;

    Tensor3D() = default;
    Tensor3D(int a, int b, int c)
        : data(static_cast<size_t>(a) * b * c, 0.0f), d0(a), d1(b), d2(c) {}
    Tensor3D(int a, int b, int c, float val)
        : data(static_cast<size_t>(a) * b * c, val), d0(a), d1(b), d2(c) {}

    float& operator()(int i, int j, int k) {
        return data[(static_cast<size_t>(i) * d1 + j) * d2 + k];
    }
    float operator()(int i, int j, int k) const {
        return data[(static_cast<size_t>(i) * d1 + j) * d2 + k];
    }

    float* slice_ptr(int i, int j) {
        return data.data() + (static_cast<size_t>(i) * d1 + j) * d2;
    }
    const float* slice_ptr(int i, int j) const {
        return data.data() + (static_cast<size_t>(i) * d1 + j) * d2;
    }

    float* batch_ptr(int i) {
        return data.data() + static_cast<size_t>(i) * d1 * d2;
    }
    const float* batch_ptr(int i) const {
        return data.data() + static_cast<size_t>(i) * d1 * d2;
    }

    void fill(float v) { std::fill(data.begin(), data.end(), v); }

    void resize(int a, int b, int c) {
        d0 = a; d1 = b; d2 = c;
        data.assign(static_cast<size_t>(a) * b * c, 0.0f);
    }

    int total_size() const { return d0 * d1 * d2; }
};

struct Tensor4D {
    std::vector<float> data;
    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;

    Tensor4D() = default;
    Tensor4D(int a, int b, int c, int d)
        : data(static_cast<size_t>(a) * b * c * d, 0.0f), d0(a), d1(b), d2(c), d3(d) {}

    float& operator()(int i, int j, int k, int l) {
        return data[((static_cast<size_t>(i) * d1 + j) * d2 + k) * d3 + l];
    }
    float operator()(int i, int j, int k, int l) const {
        return data[((static_cast<size_t>(i) * d1 + j) * d2 + k) * d3 + l];
    }

    void fill(float v) { std::fill(data.begin(), data.end(), v); }

    void resize(int a, int b, int c, int d) {
        d0 = a; d1 = b; d2 = c; d3 = d;
        data.assign(static_cast<size_t>(a) * b * c * d, 0.0f);
    }

    int total_size() const { return d0 * d1 * d2 * d3; }
};

struct Tensor5D {
    std::vector<float> data;
    int d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0;

    Tensor5D() = default;
    Tensor5D(int a, int b, int c, int d, int e)
        : data(static_cast<size_t>(a) * b * c * d * e, 0.0f),
          d0(a), d1(b), d2(c), d3(d), d4(e) {}

    float& operator()(int i, int j, int k, int l, int m) {
        return data[(((static_cast<size_t>(i) * d1 + j) * d2 + k) * d3 + l) * d4 + m];
    }
    float operator()(int i, int j, int k, int l, int m) const {
        return data[(((static_cast<size_t>(i) * d1 + j) * d2 + k) * d3 + l) * d4 + m];
    }

    void fill(float v) { std::fill(data.begin(), data.end(), v); }
    int total_size() const { return d0 * d1 * d2 * d3 * d4; }
};

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;

    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(float s) const { float inv = 1.0f / s; return {x * inv, y * inv, z * inv}; }
    Vec3 operator-() const { return {-x, -y, -z}; }

    float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    float norm() const { return std::sqrt(x * x + y * y + z * z); }
    float norm_sq() const { return x * x + y * y + z * z; }
    Vec3 normalized(float eps = 1e-8f) const {
        float n = norm();
        if (n < eps) return {0, 0, 0};
        return *this / n;
    }
    bool is_finite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }
};

// ===================================================================
// SECTION 2: Math Operations
// ===================================================================

namespace math {

inline float gelu(float x) {
    return x * 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float silu(float x) { return x * sigmoid(x); }

inline void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

inline void matmul_transB(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[i * K + k] * B[j * K + k];
            C[i * N + j] = sum;
        }
    }
}

inline void matmul_acc(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] += sum;
        }
}

inline void linear(const float* x, const Matrix& W, const std::vector<float>& bias,
                   float* out, int seq_len, int in_dim, int out_dim) {
    matmul_transB(x, W.data.data(), out, seq_len, in_dim, out_dim);
    if (!bias.empty()) {
        for (int i = 0; i < seq_len; ++i)
            for (int j = 0; j < out_dim; ++j)
                out[i * out_dim + j] += bias[j];
    }
}

inline void softmax_row(float* row, int len) {
    float max_val = *std::max_element(row, row + len);
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) { row[i] = std::exp(row[i] - max_val); sum += row[i]; }
    float inv = 1.0f / (sum + 1e-12f);
    for (int i = 0; i < len; ++i) row[i] *= inv;
}

inline void layer_norm(float* x, const float* weight, const float* bias, int dim, float eps = 1e-12f) {
    float mean = 0.0f;
    for (int i = 0; i < dim; ++i) mean += x[i];
    mean /= dim;
    float var = 0.0f;
    for (int i = 0; i < dim; ++i) { float d = x[i] - mean; var += d * d; }
    var /= dim;
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < dim; ++i) {
        x[i] = (x[i] - mean) * inv_std;
        if (weight) x[i] *= weight[i];
        if (bias) x[i] += bias[i];
    }
}

inline void layer_norm_pytorch(float* x, const float* weight, const float* bias, int dim, float eps = 1e-5f) {
    layer_norm(x, weight, bias, dim, eps);
}

inline void gelu_inplace(float* buf, int n) { for (int i = 0; i < n; ++i) buf[i] = gelu(buf[i]); }
inline void relu_inplace(float* buf, int n) { for (int i = 0; i < n; ++i) buf[i] = relu(buf[i]); }
inline void sigmoid_inplace(float* buf, int n) { for (int i = 0; i < n; ++i) buf[i] = sigmoid(buf[i]); }

inline void vec_add(float* dst, const float* src, int n) { for (int i = 0; i < n; ++i) dst[i] += src[i]; }
inline void vec_copy(float* dst, const float* src, int n) { std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float)); }
inline void vec_scale(float* buf, float s, int n) { for (int i = 0; i < n; ++i) buf[i] *= s; }
inline void vec_mul(float* dst, const float* src, int n) { for (int i = 0; i < n; ++i) dst[i] *= src[i]; }
inline void vec_zero(float* buf, int n) { std::memset(buf, 0, static_cast<size_t>(n) * sizeof(float)); }
inline float vec_dot(const float* a, const float* b, int n) { float s = 0; for (int i = 0; i < n; ++i) s += a[i] * b[i]; return s; }
inline float vec_norm(const float* v, int n) { return std::sqrt(vec_dot(v, v, n)); }

inline std::vector<float> rbf(float value, float v_min, float v_max, int n_bins = 16) {
    std::vector<float> result(n_bins);
    float bin_width = (v_max - v_min) / n_bins;
    for (int i = 0; i < n_bins; ++i) {
        float center = v_min + (i + 0.5f) * bin_width;
        float z = (value - center) / bin_width;
        result[i] = std::exp(-z * z);
    }
    return result;
}

inline void symmetrize(float* mat, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            float avg = mat[i * n + j] + mat[j * n + i];
            mat[i * n + j] = avg;
            mat[j * n + i] = avg;
        }
}

inline void apc(float* mat, int n) {
    std::vector<float> row_sum(n, 0.0f), col_sum(n, 0.0f);
    float total = 0.0f;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            row_sum[i] += mat[i * n + j];
            col_sum[j] += mat[i * n + j];
            total += mat[i * n + j];
        }
    if (std::abs(total) < 1e-12f) return;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            mat[i * n + j] -= row_sum[i] * col_sum[j] / total;
}

} // namespace math

// ===================================================================
// SECTION 3: Constants
// ===================================================================

namespace constants {
inline const std::vector<std::string>& proteinseq_toks() {
    static const std::vector<std::string> toks = {
        "L","A","G","V","S","E","R","T","I","D",
        "P","K","Q","N","F","Y","M","H","W","C",
        "X","B","U","Z","O",".","-"
    };
    return toks;
}

inline const std::vector<std::string>& restypes_with_x() {
    static const std::vector<std::string> types = {
        "A","R","N","D","C","Q","E","G","H","I",
        "L","K","M","F","P","S","T","W","Y","V","X"
    };
    return types;
}
} // namespace constants

// ===================================================================
// SECTION 4: Alphabet
// ===================================================================

enum class AlphabetArch { ESM1, ESM1b, MSATransformer, InvariantGVP };

class Alphabet {
public:
    std::vector<std::string> all_toks;
    std::unordered_map<std::string, int> tok_to_idx;
    std::vector<std::string> standard_toks;
    std::vector<std::string> prepend_toks_list;
    std::vector<std::string> append_toks_list;

    int unk_idx = 0, pad_idx = 0, cls_idx = 0, mask_idx = 0, eos_idx = 0;
    bool prepend_bos = true, append_eos = true, use_msa = false;

    Alphabet() { init_from_architecture(AlphabetArch::ESM1b); }
    explicit Alphabet(AlphabetArch arch) { init_from_architecture(arch); }

    Alphabet(const std::vector<std::string>& std_toks,
             const std::vector<std::string>& prepend,
             const std::vector<std::string>& append,
             bool p_bos, bool a_eos, bool msa = false) {
        standard_toks = std_toks; prepend_toks_list = prepend;
        append_toks_list = append; prepend_bos = p_bos; append_eos = a_eos; use_msa = msa;
        build_vocab();
    }

    void init_from_architecture(AlphabetArch arch) {
        const auto& std_toks = constants::proteinseq_toks();
        switch (arch) {
        case AlphabetArch::ESM1:
            standard_toks = std_toks;
            prepend_toks_list = {"<null_0>", "<pad>", "<eos>", "<unk>"};
            append_toks_list = {"<cls>", "<mask>", "<sep>"};
            prepend_bos = true; append_eos = false; use_msa = false; break;
        case AlphabetArch::ESM1b:
            standard_toks = std_toks;
            prepend_toks_list = {"<cls>", "<pad>", "<eos>", "<unk>"};
            append_toks_list = {"<mask>"};
            prepend_bos = true; append_eos = true; use_msa = false; break;
        case AlphabetArch::MSATransformer:
            standard_toks = std_toks;
            prepend_toks_list = {"<cls>", "<pad>", "<eos>", "<unk>"};
            append_toks_list = {"<mask>"};
            prepend_bos = true; append_eos = false; use_msa = true; break;
        case AlphabetArch::InvariantGVP:
            standard_toks = std_toks;
            prepend_toks_list = {"<null_0>", "<pad>", "<eos>", "<unk>"};
            append_toks_list = {"<mask>", "<cath>", "<af2>"};
            prepend_bos = true; append_eos = false; use_msa = false; break;
        }
        build_vocab();
    }

    int vocab_size() const { return static_cast<int>(all_toks.size()); }

    int get_idx(const std::string& tok) const {
        auto it = tok_to_idx.find(tok);
        return (it != tok_to_idx.end()) ? it->second : unk_idx;
    }

    const std::string& get_tok(int idx) const { return all_toks[idx]; }

    std::vector<int> encode(const std::string& seq) const {
        std::vector<int> tokens;
        tokens.reserve(seq.size() + 2);
        if (prepend_bos) tokens.push_back(cls_idx);
        for (char c : seq) {
            std::string s(1, c);
            auto it = tok_to_idx.find(s);
            tokens.push_back(it != tok_to_idx.end() ? it->second : unk_idx);
        }
        if (append_eos) tokens.push_back(eos_idx);
        return tokens;
    }

    std::string decode(const std::vector<int>& tokens) const {
        std::string result;
        for (int t : tokens) {
            if (t == cls_idx || t == eos_idx || t == pad_idx) continue;
            const auto& tok = all_toks[t];
            if (tok.size() == 1 && tok[0] != '<') result += tok;
        }
        return result;
    }

private:
    void build_vocab() {
        all_toks.clear(); tok_to_idx.clear();
        for (auto& t : prepend_toks_list) all_toks.push_back(t);
        for (auto& t : standard_toks) all_toks.push_back(t);
        int remainder = static_cast<int>(all_toks.size()) % 8;
        if (remainder != 0) {
            int pad_count = 8 - remainder;
            for (int i = 0; i < pad_count; ++i)
                all_toks.push_back("<null_" + std::to_string(i + 1) + ">");
        }
        for (auto& t : append_toks_list) all_toks.push_back(t);
        for (int i = 0; i < static_cast<int>(all_toks.size()); ++i)
            tok_to_idx[all_toks[i]] = i;
        unk_idx = get_idx("<unk>"); pad_idx = get_idx("<pad>");
        cls_idx = get_idx("<cls>"); eos_idx = get_idx("<eos>");
        mask_idx = get_idx("<mask>");
    }
};

// ===================================================================
// SECTION 5: Data Loading Utilities
// ===================================================================

class FastaBatchedDataset {
public:
    std::vector<std::string> sequence_labels;
    std::vector<std::string> sequence_strs;

    FastaBatchedDataset() = default;
    FastaBatchedDataset(const std::vector<std::string>& labels, const std::vector<std::string>& strs)
        : sequence_labels(labels), sequence_strs(strs) {}

    static FastaBatchedDataset from_file(const std::string& fasta_file) {
        std::vector<std::string> labels, seqs;
        std::string cur_label, buf, line;
        std::ifstream infile(fasta_file);
        if (!infile.is_open()) throw std::runtime_error("Cannot open FASTA file: " + fasta_file);
        int line_idx = 0;
        while (std::getline(infile, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' '))
                line.pop_back();
            if (!line.empty() && line[0] == '>') {
                if (!cur_label.empty()) { labels.push_back(cur_label); seqs.push_back(buf); buf.clear(); }
                std::string label = line.substr(1);
                while (!label.empty() && label[0] == ' ') label = label.substr(1);
                cur_label = label.empty() ? ("seqnum" + std::to_string(line_idx)) : label;
            } else { buf += line; }
            line_idx++;
        }
        if (!cur_label.empty()) { labels.push_back(cur_label); seqs.push_back(buf); }
        return FastaBatchedDataset(labels, seqs);
    }

    int size() const { return static_cast<int>(sequence_labels.size()); }
    std::pair<std::string, std::string> operator[](int idx) const {
        return {sequence_labels[idx], sequence_strs[idx]};
    }

    std::vector<std::vector<int>> get_batch_indices(int toks_per_batch, int extra_toks_per_seq = 0) const {
        std::vector<std::pair<int, int>> sizes;
        for (int i = 0; i < static_cast<int>(sequence_strs.size()); ++i)
            sizes.push_back({static_cast<int>(sequence_strs[i].size()), i});
        std::sort(sizes.begin(), sizes.end());
        std::vector<std::vector<int>> batches;
        std::vector<int> buf;
        int max_len = 0;
        for (auto& [sz, i] : sizes) {
            int eff_sz = sz + extra_toks_per_seq;
            if (std::max(eff_sz, max_len) * (static_cast<int>(buf.size()) + 1) > toks_per_batch) {
                if (!buf.empty()) { batches.push_back(buf); buf.clear(); max_len = 0; }
            }
            max_len = std::max(max_len, eff_sz);
            buf.push_back(i);
        }
        if (!buf.empty()) batches.push_back(buf);
        return batches;
    }
};

class BatchConverter {
public:
    const Alphabet* alphabet;
    int truncation_seq_length;

    BatchConverter() : alphabet(nullptr), truncation_seq_length(0) {}
    explicit BatchConverter(const Alphabet* alpha, int trunc_len = 0)
        : alphabet(alpha), truncation_seq_length(trunc_len) {}

    struct BatchResult {
        std::vector<std::string> labels;
        std::vector<std::string> strs;
        std::vector<std::vector<int>> tokens;
    };

    BatchResult operator()(const std::vector<std::pair<std::string, std::string>>& raw_batch) const {
        BatchResult result;
        std::vector<std::vector<int>> encoded_list;
        for (auto& [label, seq_str] : raw_batch) {
            result.labels.push_back(label);
            result.strs.push_back(seq_str);
            auto enc = alphabet->encode(seq_str);
            if (truncation_seq_length > 0 && static_cast<int>(enc.size()) > truncation_seq_length)
                enc.resize(truncation_seq_length);
            encoded_list.push_back(enc);
        }
        int max_len = 0;
        for (auto& enc : encoded_list) max_len = std::max(max_len, static_cast<int>(enc.size()));
        result.tokens.resize(raw_batch.size(), std::vector<int>(max_len, alphabet->pad_idx));
        for (size_t i = 0; i < encoded_list.size(); ++i)
            for (int j = 0; j < static_cast<int>(encoded_list[i].size()); ++j)
                result.tokens[i][j] = encoded_list[i][j];
        return result;
    }
};

class MSABatchConverter : public BatchConverter {
public:
    using BatchConverter::BatchConverter;
    struct MSABatchResult {
        std::vector<std::vector<std::string>> labels;
        std::vector<std::vector<std::string>> strs;
        Tensor3D tokens;
    };

    MSABatchResult operator()(
            const std::vector<std::vector<std::pair<std::string, std::string>>>& raw_batch) const {
        MSABatchResult result;
        int batch_size = static_cast<int>(raw_batch.size());
        int max_alignments = 0, max_seqlen = 0;
        for (auto& msa : raw_batch) {
            max_alignments = std::max(max_alignments, static_cast<int>(msa.size()));
            if (!msa.empty()) max_seqlen = std::max(max_seqlen, static_cast<int>(msa[0].second.size()));
        }
        int tok_len = max_seqlen + (alphabet->prepend_bos ? 1 : 0) + (alphabet->append_eos ? 1 : 0);
        result.tokens = Tensor3D(batch_size, max_alignments, tok_len);
        result.tokens.fill(static_cast<float>(alphabet->pad_idx));
        result.labels.resize(batch_size); result.strs.resize(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            auto& msa = raw_batch[i];
            for (int j = 0; j < static_cast<int>(msa.size()); ++j) {
                result.labels[i].push_back(msa[j].first);
                result.strs[i].push_back(msa[j].second);
                auto enc = alphabet->encode(msa[j].second);
                for (int k = 0; k < static_cast<int>(enc.size()) && k < tok_len; ++k)
                    result.tokens(i, j, k) = static_cast<float>(enc[k]);
            }
        }
        return result;
    }
};

inline std::vector<std::pair<std::string, std::string>> read_fasta(
        const std::string& path, bool keep_gaps = true, bool keep_insertions = true, bool to_upper = false) {
    std::vector<std::pair<std::string, std::string>> results;
    std::ifstream f(path);
    if (!f.is_open()) return results;
    std::string desc, seq, line;
    auto flush = [&]() {
        if (seq.empty() || desc.empty()) return;
        std::string processed = seq;
        if (!keep_gaps) { std::string tmp; for (char c : processed) if (c != '-') tmp += c; processed = tmp; }
        if (!keep_insertions) { std::string tmp; for (char c : processed) if (c < 'a' || c > 'z') tmp += c; processed = tmp; }
        if (to_upper) for (auto& c : processed) c = static_cast<char>(std::toupper(c));
        results.push_back({desc, processed});
    };
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (!line.empty() && line[0] == '>') {
            flush();
            desc = line.substr(1);
            while (!desc.empty() && desc[0] == ' ') desc = desc.substr(1);
            seq.clear();
        } else { seq += line; }
    }
    flush();
    return results;
}

// ===================================================================
// SECTION 6: Neural Network Building Blocks
// ===================================================================

struct Linear {
    Matrix weight;
    std::vector<float> bias;
    int in_features = 0, out_features = 0;

    Linear() = default;
    Linear(int in_f, int out_f, bool use_bias, std::mt19937& rng)
        : in_features(in_f), out_features(out_f) {
        weight = Matrix(out_f, in_f);
        weight.xavier_uniform(rng);
        if (use_bias) bias.assign(out_f, 0.0f);
    }

    void forward(const float* x, float* out, int seq_len) const {
        math::linear(x, weight, bias, out, seq_len, in_features, out_features);
    }

    void forward_single(const float* x, float* out) const { forward(x, out, 1); }
};

struct LayerNorm {
    std::vector<float> weight, bias;
    int dim = 0;
    float eps = 1e-12f;
    bool affine = true;

    LayerNorm() = default;
    LayerNorm(int d, float e = 1e-12f, bool aff = true) : dim(d), eps(e), affine(aff) {
        if (affine) { weight.assign(d, 1.0f); bias.assign(d, 0.0f); }
    }

    void forward(float* x) const {
        math::layer_norm(x, affine ? weight.data() : nullptr, affine ? bias.data() : nullptr, dim, eps);
    }
    void forward_batch(float* x, int n) const {
        for (int i = 0; i < n; ++i) forward(x + i * dim);
    }
};

using ESM1LayerNorm = LayerNorm;

struct ESM1bLayerNorm {
    std::vector<float> weight, bias;
    int dim = 0;
    float eps = 1e-5f;

    ESM1bLayerNorm() = default;
    ESM1bLayerNorm(int d, float e = 1e-5f) : dim(d), eps(e) {
        weight.assign(d, 1.0f); bias.assign(d, 0.0f);
    }
    void forward(float* x) const {
        math::layer_norm_pytorch(x, weight.data(), bias.data(), dim, eps);
    }
    void forward_batch(float* x, int n) const {
        for (int i = 0; i < n; ++i) forward(x + i * dim);
    }
};

struct Embedding {
    Matrix weight;
    int vocab_size = 0, embed_dim = 0, padding_idx = -1;

    Embedding() = default;
    Embedding(int vs, int ed, int pad_idx, std::mt19937& rng)
        : vocab_size(vs), embed_dim(ed), padding_idx(pad_idx) {
        weight = Matrix(vs, ed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : weight.data) v = dist(rng);
        if (padding_idx >= 0 && padding_idx < vs) {
            float* p = weight.row_ptr(padding_idx);
            for (int i = 0; i < ed; ++i) p[i] = 0.0f;
        }
    }
    const float* lookup(int idx) const { return weight.row_ptr(idx); }
};

struct LearnedPositionalEmbedding {
    Matrix weight;
    int num_embeddings = 0, embed_dim = 0, padding_idx = 0, total_embeddings = 0;

    LearnedPositionalEmbedding() = default;
    LearnedPositionalEmbedding(int num_emb, int ed, int pad_idx, std::mt19937& rng)
        : num_embeddings(num_emb), embed_dim(ed), padding_idx(pad_idx) {
        total_embeddings = num_emb + pad_idx + 1;
        weight = Matrix(total_embeddings, ed);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (auto& v : weight.data) v = dist(rng);
        if (pad_idx >= 0 && pad_idx < total_embeddings) {
            float* p = weight.row_ptr(pad_idx);
            for (int i = 0; i < ed; ++i) p[i] = 0.0f;
        }
    }

    std::vector<int> compute_positions(const std::vector<int>& tokens) const {
        std::vector<int> positions(tokens.size());
        int cumsum = 0;
        for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
            bool is_pad = (tokens[i] == padding_idx);
            if (!is_pad) cumsum++;
            positions[i] = is_pad ? padding_idx : (cumsum + padding_idx);
        }
        return positions;
    }

    const float* lookup(int pos) const {
        if (pos < 0 || pos >= total_embeddings) pos = padding_idx;
        return weight.row_ptr(pos);
    }
};

struct SinusoidalPositionalEmbedding {
    int embed_dim = 0, padding_idx = 0;
    Matrix weights;

    SinusoidalPositionalEmbedding() = default;
    SinusoidalPositionalEmbedding(int ed, int pad_idx) : embed_dim(ed), padding_idx(pad_idx) {}

    void ensure_weights(int max_pos) {
        if (weights.rows >= max_pos) return;
        int half_dim = embed_dim / 2;
        float log_10000 = std::log(10000.0f);
        weights = Matrix(max_pos, embed_dim);
        for (int pos = 0; pos < max_pos; ++pos) {
            for (int i = 0; i < half_dim; ++i) {
                float angle = pos * std::exp(-i * log_10000 / half_dim);
                weights(pos, i) = std::sin(angle);
                weights(pos, half_dim + i) = std::cos(angle);
            }
        }
        if (padding_idx >= 0 && padding_idx < max_pos)
            for (int i = 0; i < embed_dim; ++i) weights(padding_idx, i) = 0.0f;
    }

    std::vector<int> make_positions(const std::vector<int>& tokens, int pad_idx) const {
        std::vector<int> positions(tokens.size());
        for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
            positions[i] = (tokens[i] == pad_idx) ? pad_idx : (i + pad_idx + 1);
        }
        return positions;
    }

    const float* lookup(int pos) { ensure_weights(pos + 1); return weights.row_ptr(pos); }
};

// ===================================================================
// SECTION 7: Rotary Embedding
// ===================================================================

struct RotaryEmbedding {
    int dim = 0;
    std::vector<float> inv_freq;
    int cached_seq_len = -1;
    std::vector<float> cos_cached, sin_cached;

    RotaryEmbedding() = default;
    explicit RotaryEmbedding(int d) : dim(d) {
        inv_freq.resize(d / 2);
        for (int i = 0; i < d / 2; ++i)
            inv_freq[i] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / d);
    }

    void update_cache(int seq_len) {
        if (seq_len == cached_seq_len) return;
        cached_seq_len = seq_len;
        cos_cached.resize(seq_len * dim); sin_cached.resize(seq_len * dim);
        for (int pos = 0; pos < seq_len; ++pos) {
            for (int i = 0; i < dim / 2; ++i) {
                float freq = pos * inv_freq[i];
                float c = std::cos(freq), s = std::sin(freq);
                cos_cached[pos * dim + i] = c;
                cos_cached[pos * dim + dim / 2 + i] = c;
                sin_cached[pos * dim + i] = s;
                sin_cached[pos * dim + dim / 2 + i] = s;
            }
        }
    }

    void apply(float* q, float* k, int seq_len, int head_dim) {
        update_cache(seq_len);
        for (int pos = 0; pos < seq_len; ++pos) {
            float* q_row = q + pos * head_dim;
            float* k_row = k + pos * head_dim;
            int half = head_dim / 2;
            for (int i = 0; i < half; ++i) {
                float q1 = q_row[i], q2 = q_row[half + i];
                float k1 = k_row[i], k2 = k_row[half + i];
                float c = cos_cached[pos * dim + i], s = sin_cached[pos * dim + i];
                float ch = cos_cached[pos * dim + half + i], sh = sin_cached[pos * dim + half + i];
                q_row[i]        = q1 * c + (-q2) * s;
                q_row[half + i] = q2 * ch + q1 * sh;
                k_row[i]        = k1 * c + (-k2) * s;
                k_row[half + i] = k2 * ch + k1 * sh;
            }
        }
    }
};

// ===================================================================
// SECTION 8: Multi-Head Attention
// ===================================================================

struct MultiHeadAttention {
    int embed_dim = 0, num_heads = 0, head_dim = 0, kdim = 0, vdim = 0;
    float scaling = 0.0f;
    bool has_bias_kv = false, add_zero_attn = false, self_attention = false;
    bool encoder_decoder_attention = false, use_rotary = false;

    Linear q_proj, k_proj, v_proj, out_proj;
    std::vector<float> bias_k, bias_v;
    RotaryEmbedding rot_emb;

    MultiHeadAttention() = default;
    MultiHeadAttention(int ed, int nh, bool bias_kv, bool zero_attn, bool self_attn,
                       bool enc_dec, bool rotary, std::mt19937& rng, int kd = 0, int vd = 0)
        : embed_dim(ed), num_heads(nh), has_bias_kv(bias_kv), add_zero_attn(zero_attn),
          self_attention(self_attn), encoder_decoder_attention(enc_dec), use_rotary(rotary) {
        kdim = kd > 0 ? kd : ed; vdim = vd > 0 ? vd : ed;
        head_dim = ed / nh; assert(head_dim * nh == ed);
        scaling = 1.0f / std::sqrt(static_cast<float>(head_dim));
        float gain = 1.0f / std::sqrt(2.0f);
        q_proj = Linear(ed, ed, true, rng); k_proj = Linear(kdim, ed, true, rng);
        v_proj = Linear(vdim, ed, true, rng); out_proj = Linear(ed, ed, true, rng);
        q_proj.weight.xavier_uniform(rng, gain); k_proj.weight.xavier_uniform(rng, gain);
        v_proj.weight.xavier_uniform(rng, gain); out_proj.weight.xavier_uniform(rng);
        if (bias_kv) {
            bias_k.resize(ed); bias_v.resize(ed);
            std::normal_distribution<float> dist(0.0f, 0.02f);
            for (auto& v : bias_k) v = dist(rng);
            for (auto& v : bias_v) v = dist(rng);
        }
        if (rotary) rot_emb = RotaryEmbedding(head_dim);
    }

    static MultiHeadAttention create_self_attn(int ed, int nh, bool bias_kv, bool rotary, std::mt19937& rng) {
        return MultiHeadAttention(ed, nh, bias_kv, false, true, false, rotary, rng);
    }
    static MultiHeadAttention create_cross_attn(int ed, int nh, int kd, int vd, std::mt19937& rng) {
        return MultiHeadAttention(ed, nh, false, false, false, true, false, rng, kd, vd);
    }

    void forward(const float* x, int seq_len, float* out, Tensor3D& attn_weights_out,
                 const std::vector<bool>& padding_mask = {},
                 const float* attn_mask = nullptr, bool need_head_weights = true) const {
        int ed = embed_dim;
        std::vector<float> Q(seq_len * ed), K(seq_len * ed), V(seq_len * ed);
        q_proj.forward(x, Q.data(), seq_len); k_proj.forward(x, K.data(), seq_len);
        v_proj.forward(x, V.data(), seq_len);
        math::vec_scale(Q.data(), scaling, seq_len * ed);

        if (use_rotary) {
            RotaryEmbedding rot = rot_emb;
            for (int h = 0; h < num_heads; ++h) {
                std::vector<float> q_head(seq_len * head_dim), k_head(seq_len * head_dim);
                for (int p = 0; p < seq_len; ++p)
                    for (int d = 0; d < head_dim; ++d) {
                        q_head[p * head_dim + d] = Q[p * ed + h * head_dim + d];
                        k_head[p * head_dim + d] = K[p * ed + h * head_dim + d];
                    }
                rot.apply(q_head.data(), k_head.data(), seq_len, head_dim);
                for (int p = 0; p < seq_len; ++p)
                    for (int d = 0; d < head_dim; ++d) {
                        Q[p * ed + h * head_dim + d] = q_head[p * head_dim + d];
                        K[p * ed + h * head_dim + d] = k_head[p * head_dim + d];
                    }
            }
        }

        int src_len = seq_len;
        std::vector<float> K_ext, V_ext;
        if (has_bias_kv && !bias_k.empty()) {
            src_len = seq_len + 1;
            K_ext.resize(src_len * ed); V_ext.resize(src_len * ed);
            std::memcpy(K_ext.data(), K.data(), seq_len * ed * sizeof(float));
            std::memcpy(V_ext.data(), V.data(), seq_len * ed * sizeof(float));
            std::memcpy(K_ext.data() + seq_len * ed, bias_k.data(), ed * sizeof(float));
            std::memcpy(V_ext.data() + seq_len * ed, bias_v.data(), ed * sizeof(float));
        } else { K_ext = K; V_ext = V; }

        attn_weights_out.resize(num_heads, seq_len, src_len);
        std::vector<float> attn_out(seq_len * ed, 0.0f);

        std::vector<bool> key_pad_mask;
        if (!padding_mask.empty()) {
            key_pad_mask.resize(src_len, false);
            for (int i = 0; i < seq_len; ++i) key_pad_mask[i] = padding_mask[i];
        }

        for (int h = 0; h < num_heads; ++h) {
            std::vector<float> scores(seq_len * src_len);
            for (int qi = 0; qi < seq_len; ++qi) {
                const float* q_ptr = Q.data() + qi * ed + h * head_dim;
                for (int ki = 0; ki < src_len; ++ki) {
                    const float* k_ptr = K_ext.data() + ki * ed + h * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) dot += q_ptr[d] * k_ptr[d];
                    scores[qi * src_len + ki] = dot;
                }
            }
            if (attn_mask) {
                for (int qi = 0; qi < seq_len; ++qi)
                    for (int ki = 0; ki < src_len && ki < seq_len; ++ki)
                        scores[qi * src_len + ki] += attn_mask[qi * seq_len + ki];
            }
            if (!key_pad_mask.empty())
                for (int qi = 0; qi < seq_len; ++qi)
                    for (int ki = 0; ki < src_len; ++ki)
                        if (key_pad_mask[ki]) scores[qi * src_len + ki] = -1e9f;
            for (int qi = 0; qi < seq_len; ++qi) math::softmax_row(scores.data() + qi * src_len, src_len);
            for (int qi = 0; qi < seq_len; ++qi)
                for (int ki = 0; ki < src_len; ++ki)
                    attn_weights_out(h, qi, ki) = scores[qi * src_len + ki];
            for (int qi = 0; qi < seq_len; ++qi) {
                float* out_ptr = attn_out.data() + qi * ed + h * head_dim;
                for (int ki = 0; ki < src_len; ++ki) {
                    float w = scores[qi * src_len + ki];
                    const float* v_ptr = V_ext.data() + ki * ed + h * head_dim;
                    for (int d = 0; d < head_dim; ++d) out_ptr[d] += w * v_ptr[d];
                }
            }
        }
        out_proj.forward(attn_out.data(), out, seq_len);
    }

    void forward_cross(const float* query, int tgt_len, const float* key, const float* value,
                       int src_len, float* out, Tensor3D& attn_weights_out,
                       const std::vector<bool>& key_padding_mask = {},
                       const float* attn_mask = nullptr) const {
        int ed = embed_dim;
        std::vector<float> Q(tgt_len * ed), K(src_len * ed), V(src_len * ed);
        q_proj.forward(query, Q.data(), tgt_len);
        k_proj.forward(key, K.data(), src_len);
        v_proj.forward(value, V.data(), src_len);
        math::vec_scale(Q.data(), scaling, tgt_len * ed);

        attn_weights_out.resize(num_heads, tgt_len, src_len);
        std::vector<float> attn_out(tgt_len * ed, 0.0f);

        for (int h = 0; h < num_heads; ++h) {
            std::vector<float> scores(tgt_len * src_len);
            for (int qi = 0; qi < tgt_len; ++qi) {
                const float* q_ptr = Q.data() + qi * ed + h * head_dim;
                for (int ki = 0; ki < src_len; ++ki) {
                    const float* k_ptr = K.data() + ki * ed + h * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) dot += q_ptr[d] * k_ptr[d];
                    scores[qi * src_len + ki] = dot;
                }
            }
            if (attn_mask)
                for (int qi = 0; qi < tgt_len; ++qi)
                    for (int ki = 0; ki < src_len; ++ki)
                        scores[qi * src_len + ki] += attn_mask[qi * src_len + ki];
            if (!key_padding_mask.empty())
                for (int qi = 0; qi < tgt_len; ++qi)
                    for (int ki = 0; ki < src_len; ++ki)
                        if (key_padding_mask[ki]) scores[qi * src_len + ki] = -1e9f;
            for (int qi = 0; qi < tgt_len; ++qi) math::softmax_row(scores.data() + qi * src_len, src_len);
            for (int qi = 0; qi < tgt_len; ++qi)
                for (int ki = 0; ki < src_len; ++ki)
                    attn_weights_out(h, qi, ki) = scores[qi * src_len + ki];
            for (int qi = 0; qi < tgt_len; ++qi) {
                float* out_ptr = attn_out.data() + qi * ed + h * head_dim;
                for (int ki = 0; ki < src_len; ++ki) {
                    float w = scores[qi * src_len + ki];
                    const float* v_ptr = V.data() + ki * ed + h * head_dim;
                    for (int d = 0; d < head_dim; ++d) out_ptr[d] += w * v_ptr[d];
                }
            }
        }
        out_proj.forward(attn_out.data(), out, tgt_len);
    }
};

// ===================================================================
// SECTION 9: Axial Attention
// ===================================================================

struct RowSelfAttention {
    int embed_dim = 0, num_heads = 0, head_dim = 0;
    float scaling = 0.0f;
    Linear q_proj, k_proj, v_proj, out_proj;

    RowSelfAttention() = default;
    RowSelfAttention(int ed, int nh, std::mt19937& rng)
        : embed_dim(ed), num_heads(nh) {
        head_dim = ed / nh; scaling = 1.0f / std::sqrt(static_cast<float>(head_dim));
        q_proj = Linear(ed, ed, true, rng); k_proj = Linear(ed, ed, true, rng);
        v_proj = Linear(ed, ed, true, rng); out_proj = Linear(ed, ed, true, rng);
    }

    void forward(const Tensor3D& x, Tensor3D& output, Tensor3D& attn_weights) const {
        int num_rows = x.d0, num_cols = x.d1, ed = embed_dim;
        float align_scale = scaling / std::sqrt(static_cast<float>(num_rows));
        Tensor3D attn_scores(num_heads, num_cols, num_cols, 0.0f);
        for (int r = 0; r < num_rows; ++r) {
            std::vector<float> Q(num_cols * ed), K(num_cols * ed);
            q_proj.forward(x.slice_ptr(r, 0), Q.data(), num_cols);
            k_proj.forward(x.slice_ptr(r, 0), K.data(), num_cols);
            math::vec_scale(Q.data(), align_scale, num_cols * ed);
            for (int h = 0; h < num_heads; ++h)
                for (int i = 0; i < num_cols; ++i)
                    for (int j = 0; j < num_cols; ++j) {
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; ++d)
                            dot += Q[i * ed + h * head_dim + d] * K[j * ed + h * head_dim + d];
                        attn_scores(h, i, j) += dot;
                    }
        }
        for (int h = 0; h < num_heads; ++h)
            for (int i = 0; i < num_cols; ++i)
                math::softmax_row(attn_scores.slice_ptr(h, i), num_cols);
        attn_weights = attn_scores;
        output = Tensor3D(num_rows, num_cols, ed, 0.0f);
        for (int r = 0; r < num_rows; ++r) {
            std::vector<float> V(num_cols * ed);
            v_proj.forward(x.slice_ptr(r, 0), V.data(), num_cols);
            std::vector<float> attn_out(num_cols * ed, 0.0f);
            for (int h = 0; h < num_heads; ++h)
                for (int i = 0; i < num_cols; ++i)
                    for (int j = 0; j < num_cols; ++j) {
                        float w = attn_scores(h, i, j);
                        for (int d = 0; d < head_dim; ++d)
                            attn_out[i * ed + h * head_dim + d] += w * V[j * ed + h * head_dim + d];
                    }
            out_proj.forward(attn_out.data(), output.slice_ptr(r, 0), num_cols);
        }
    }
};

struct ColumnSelfAttention {
    int embed_dim = 0, num_heads = 0, head_dim = 0;
    float scaling = 0.0f;
    Linear q_proj, k_proj, v_proj, out_proj;

    ColumnSelfAttention() = default;
    ColumnSelfAttention(int ed, int nh, std::mt19937& rng)
        : embed_dim(ed), num_heads(nh) {
        head_dim = ed / nh; scaling = 1.0f / std::sqrt(static_cast<float>(head_dim));
        q_proj = Linear(ed, ed, true, rng); k_proj = Linear(ed, ed, true, rng);
        v_proj = Linear(ed, ed, true, rng); out_proj = Linear(ed, ed, true, rng);
    }

    void forward(const Tensor3D& x, Tensor3D& output, Tensor4D& attn_weights) const {
        int num_rows = x.d0, num_cols = x.d1, ed = embed_dim;
        output = Tensor3D(num_rows, num_cols, ed, 0.0f);
        attn_weights = Tensor4D(num_heads, num_cols, num_rows, num_rows);
        for (int c = 0; c < num_cols; ++c) {
            std::vector<float> col_data(num_rows * ed);
            for (int r = 0; r < num_rows; ++r) math::vec_copy(col_data.data() + r * ed, x.slice_ptr(r, c), ed);
            std::vector<float> Q(num_rows * ed), K(num_rows * ed), V(num_rows * ed);
            q_proj.forward(col_data.data(), Q.data(), num_rows);
            k_proj.forward(col_data.data(), K.data(), num_rows);
            v_proj.forward(col_data.data(), V.data(), num_rows);
            math::vec_scale(Q.data(), scaling, num_rows * ed);
            std::vector<float> attn_out(num_rows * ed, 0.0f);
            for (int h = 0; h < num_heads; ++h) {
                std::vector<float> scores(num_rows * num_rows);
                for (int i = 0; i < num_rows; ++i)
                    for (int j = 0; j < num_rows; ++j) {
                        float dot = 0.0f;
                        for (int d = 0; d < head_dim; ++d)
                            dot += Q[i * ed + h * head_dim + d] * K[j * ed + h * head_dim + d];
                        scores[i * num_rows + j] = dot;
                    }
                for (int i = 0; i < num_rows; ++i) math::softmax_row(scores.data() + i * num_rows, num_rows);
                for (int i = 0; i < num_rows; ++i)
                    for (int j = 0; j < num_rows; ++j)
                        attn_weights(h, c, i, j) = scores[i * num_rows + j];
                for (int i = 0; i < num_rows; ++i)
                    for (int j = 0; j < num_rows; ++j) {
                        float w = scores[i * num_rows + j];
                        for (int d = 0; d < head_dim; ++d)
                            attn_out[i * ed + h * head_dim + d] += w * V[j * ed + h * head_dim + d];
                    }
            }
            std::vector<float> col_out(num_rows * ed);
            out_proj.forward(attn_out.data(), col_out.data(), num_rows);
            for (int r = 0; r < num_rows; ++r) math::vec_copy(output.slice_ptr(r, c), col_out.data() + r * ed, ed);
        }
    }
};

// ===================================================================
// SECTION 10: Feed-Forward Network
// ===================================================================

struct FeedForwardNetwork {
    int embed_dim = 0, ffn_embed_dim = 0;
    Linear fc1, fc2;

    FeedForwardNetwork() = default;
    FeedForwardNetwork(int ed, int ffn_ed, std::mt19937& rng) : embed_dim(ed), ffn_embed_dim(ffn_ed) {
        fc1 = Linear(ed, ffn_ed, true, rng); fc2 = Linear(ffn_ed, ed, true, rng);
    }
    void forward(const float* x, float* out, int seq_len) const {
        std::vector<float> buf(seq_len * ffn_embed_dim);
        fc1.forward(x, buf.data(), seq_len);
        math::gelu_inplace(buf.data(), seq_len * ffn_embed_dim);
        fc2.forward(buf.data(), out, seq_len);
    }
};

// ===================================================================
// SECTION 11: Transformer Layers
// ===================================================================

struct TransformerLayer {
    int embed_dim = 0, ffn_embed_dim = 0, attention_heads = 0;
    bool use_rotary_embeddings = false, use_esm1b_layer_norm = false;

    MultiHeadAttention self_attn;
    LayerNorm self_attn_layer_norm;
    ESM1bLayerNorm self_attn_ln_1b;
    Linear fc1, fc2;
    LayerNorm final_layer_norm;
    ESM1bLayerNorm final_ln_1b;

    TransformerLayer() = default;
    TransformerLayer(int ed, int ffn_ed, int heads, bool add_bias_kv,
                     bool esm1b_ln, bool rotary, std::mt19937& rng)
        : embed_dim(ed), ffn_embed_dim(ffn_ed), attention_heads(heads),
          use_rotary_embeddings(rotary), use_esm1b_layer_norm(esm1b_ln) {
        self_attn = MultiHeadAttention::create_self_attn(ed, heads, add_bias_kv, rotary, rng);
        if (esm1b_ln) { self_attn_ln_1b = ESM1bLayerNorm(ed); final_ln_1b = ESM1bLayerNorm(ed); }
        else { self_attn_layer_norm = LayerNorm(ed); final_layer_norm = LayerNorm(ed); }
        fc1 = Linear(ed, ffn_ed, true, rng); fc2 = Linear(ffn_ed, ed, true, rng);
    }
    TransformerLayer(int ed, int ffn_ed, int heads, bool add_bias_kv, std::mt19937& rng)
        : TransformerLayer(ed, ffn_ed, heads, add_bias_kv, false, false, rng) {}

    void forward(float* x, int seq_len, Tensor3D& attn_weights,
                 const std::vector<bool>& padding_mask = {},
                 bool need_head_weights = true) const {
        int ed = embed_dim;
        std::vector<float> residual(seq_len * ed);
        math::vec_copy(residual.data(), x, seq_len * ed);
        if (use_esm1b_layer_norm) self_attn_ln_1b.forward_batch(x, seq_len);
        else self_attn_layer_norm.forward_batch(x, seq_len);
        std::vector<float> attn_out(seq_len * ed);
        self_attn.forward(x, seq_len, attn_out.data(), attn_weights, padding_mask, nullptr, need_head_weights);
        for (int i = 0; i < seq_len * ed; ++i) x[i] = residual[i] + attn_out[i];
        math::vec_copy(residual.data(), x, seq_len * ed);
        if (use_esm1b_layer_norm) final_ln_1b.forward_batch(x, seq_len);
        else final_layer_norm.forward_batch(x, seq_len);
        std::vector<float> ffn_buf(seq_len * ffn_embed_dim);
        fc1.forward(x, ffn_buf.data(), seq_len);
        math::gelu_inplace(ffn_buf.data(), seq_len * ffn_embed_dim);
        fc2.forward(ffn_buf.data(), x, seq_len);
        math::vec_add(x, residual.data(), seq_len * ed);
    }
};

struct AxialTransformerLayer {
    int embed_dim = 0, ffn_embed_dim = 0, num_attention_heads = 0;
    RowSelfAttention row_attn;
    ColumnSelfAttention col_attn;
    FeedForwardNetwork ffn;
    ESM1bLayerNorm row_ln, col_ln, ffn_ln;

    AxialTransformerLayer() = default;
    AxialTransformerLayer(int ed, int ffn_ed, int nh, std::mt19937& rng)
        : embed_dim(ed), ffn_embed_dim(ffn_ed), num_attention_heads(nh) {
        row_attn = RowSelfAttention(ed, nh, rng); col_attn = ColumnSelfAttention(ed, nh, rng);
        ffn = FeedForwardNetwork(ed, ffn_ed, rng);
        row_ln = ESM1bLayerNorm(ed); col_ln = ESM1bLayerNorm(ed); ffn_ln = ESM1bLayerNorm(ed);
    }

    void forward(Tensor3D& x, Tensor3D& row_attn_weights, Tensor4D& col_attn_weights) const {
        int nr = x.d0, nc = x.d1, ed = embed_dim;
        {
            Tensor3D normed(nr, nc, ed);
            for (int r = 0; r < nr; ++r)
                for (int c = 0; c < nc; ++c) {
                    math::vec_copy(normed.slice_ptr(r, c), x.slice_ptr(r, c), ed);
                    row_ln.forward(normed.slice_ptr(r, c));
                }
            Tensor3D row_out;
            row_attn.forward(normed, row_out, row_attn_weights);
            for (int i = 0; i < static_cast<int>(x.data.size()); ++i) x.data[i] += row_out.data[i];
        }
        {
            Tensor3D normed(nr, nc, ed);
            for (int r = 0; r < nr; ++r)
                for (int c = 0; c < nc; ++c) {
                    math::vec_copy(normed.slice_ptr(r, c), x.slice_ptr(r, c), ed);
                    col_ln.forward(normed.slice_ptr(r, c));
                }
            Tensor3D col_out;
            col_attn.forward(normed, col_out, col_attn_weights);
            for (int i = 0; i < static_cast<int>(x.data.size()); ++i) x.data[i] += col_out.data[i];
        }
        for (int r = 0; r < nr; ++r)
            for (int c = 0; c < nc; ++c) {
                std::vector<float> normed(ed), ffn_out(ed);
                math::vec_copy(normed.data(), x.slice_ptr(r, c), ed);
                ffn_ln.forward(normed.data());
                ffn.forward(normed.data(), ffn_out.data(), 1);
                math::vec_add(x.slice_ptr(r, c), ffn_out.data(), ed);
            }
    }
};

struct TransformerEncoderLayer {
    int embed_dim = 0, ffn_embed_dim = 0, attention_heads = 0;
    MultiHeadAttention self_attn;
    ESM1bLayerNorm self_attn_layer_norm, final_layer_norm;
    Linear fc1, fc2;

    TransformerEncoderLayer() = default;
    TransformerEncoderLayer(int ed, int ffn_ed, int heads, std::mt19937& rng)
        : embed_dim(ed), ffn_embed_dim(ffn_ed), attention_heads(heads) {
        self_attn = MultiHeadAttention(ed, heads, false, false, true, false, false, rng);
        self_attn_layer_norm = ESM1bLayerNorm(ed); fc1 = Linear(ed, ffn_ed, true, rng);
        fc2 = Linear(ffn_ed, ed, true, rng); final_layer_norm = ESM1bLayerNorm(ed);
    }

    void forward(float* x, int seq_len, const std::vector<bool>& padding_mask = {}) const {
        int ed = embed_dim;
        std::vector<float> residual(seq_len * ed);
        math::vec_copy(residual.data(), x, seq_len * ed);
        self_attn_layer_norm.forward_batch(x, seq_len);
        Tensor3D aw; std::vector<float> ao(seq_len * ed);
        self_attn.forward(x, seq_len, ao.data(), aw, padding_mask);
        for (int i = 0; i < seq_len * ed; ++i) x[i] = residual[i] + ao[i];
        math::vec_copy(residual.data(), x, seq_len * ed);
        final_layer_norm.forward_batch(x, seq_len);
        std::vector<float> buf(seq_len * ffn_embed_dim);
        fc1.forward(x, buf.data(), seq_len);
        math::relu_inplace(buf.data(), seq_len * ffn_embed_dim);
        fc2.forward(buf.data(), x, seq_len);
        math::vec_add(x, residual.data(), seq_len * ed);
    }
};

struct TransformerDecoderLayer {
    int embed_dim = 0, ffn_embed_dim = 0, attention_heads = 0;
    MultiHeadAttention self_attn, encoder_attn;
    ESM1bLayerNorm self_attn_layer_norm, encoder_attn_layer_norm, final_layer_norm;
    Linear fc1, fc2;

    TransformerDecoderLayer() = default;
    TransformerDecoderLayer(int ed, int ffn_ed, int heads, int enc_ed, std::mt19937& rng)
        : embed_dim(ed), ffn_embed_dim(ffn_ed), attention_heads(heads) {
        self_attn = MultiHeadAttention(ed, heads, false, false, true, false, false, rng);
        encoder_attn = MultiHeadAttention::create_cross_attn(ed, heads, enc_ed, enc_ed, rng);
        self_attn_layer_norm = ESM1bLayerNorm(ed); encoder_attn_layer_norm = ESM1bLayerNorm(ed);
        final_layer_norm = ESM1bLayerNorm(ed);
        fc1 = Linear(ed, ffn_ed, true, rng); fc2 = Linear(ffn_ed, ed, true, rng);
    }

    void forward(float* x, int tgt_len, const float* encoder_out, int src_len,
                 const std::vector<bool>& kpm = {}, const float* self_mask = nullptr) const {
        int ed = embed_dim;
        std::vector<float> res(tgt_len * ed);
        math::vec_copy(res.data(), x, tgt_len * ed);
        self_attn_layer_norm.forward_batch(x, tgt_len);
        Tensor3D aw; std::vector<float> ao(tgt_len * ed);
        self_attn.forward(x, tgt_len, ao.data(), aw, {}, self_mask);
        for (int i = 0; i < tgt_len * ed; ++i) x[i] = res[i] + ao[i];
        if (encoder_out) {
            math::vec_copy(res.data(), x, tgt_len * ed);
            encoder_attn_layer_norm.forward_batch(x, tgt_len);
            Tensor3D caw; std::vector<float> co(tgt_len * ed);
            encoder_attn.forward_cross(x, tgt_len, encoder_out, encoder_out, src_len, co.data(), caw, kpm);
            for (int i = 0; i < tgt_len * ed; ++i) x[i] = res[i] + co[i];
        }
        math::vec_copy(res.data(), x, tgt_len * ed);
        final_layer_norm.forward_batch(x, tgt_len);
        std::vector<float> buf(tgt_len * ffn_embed_dim);
        fc1.forward(x, buf.data(), tgt_len);
        math::relu_inplace(buf.data(), tgt_len * ffn_embed_dim);
        fc2.forward(buf.data(), x, tgt_len);
        math::vec_add(x, res.data(), tgt_len * ed);
    }
};

// ===================================================================
// SECTION 12: LM Head and Contact Prediction Head
// ===================================================================

struct RobertaLMHead {
    Linear dense;
    ESM1bLayerNorm layer_norm;
    const Matrix* embed_weight = nullptr;
    std::vector<float> bias;

    RobertaLMHead() = default;
    void init(int embed_dim, int vocab_size, const Matrix* embed_w, std::mt19937& rng) {
        dense = Linear(embed_dim, embed_dim, true, rng);
        layer_norm = ESM1bLayerNorm(embed_dim); embed_weight = embed_w;
        bias.assign(vocab_size, 0.0f);
    }
    void forward(const float* x, float* out, int seq_len) const {
        int ed = dense.in_features, vs = static_cast<int>(bias.size());
        std::vector<float> buf(seq_len * ed);
        dense.forward(x, buf.data(), seq_len);
        math::gelu_inplace(buf.data(), seq_len * ed);
        layer_norm.forward_batch(buf.data(), seq_len);
        math::matmul_transB(buf.data(), embed_weight->data.data(), out, seq_len, ed, vs);
        for (int i = 0; i < seq_len; ++i)
            for (int j = 0; j < vs; ++j) out[i * vs + j] += bias[j];
    }
};

struct ContactPredictionHead {
    int in_features = 0;
    bool prepend_bos = true, append_eos = true;
    int eos_idx = -1;
    Linear regression;

    ContactPredictionHead() = default;
    void init(int in_f, bool bos, bool eos, int eidx, std::mt19937& rng) {
        in_features = in_f; prepend_bos = bos; append_eos = eos; eos_idx = eidx;
        regression = Linear(in_f, 1, true, rng);
    }

    Matrix predict(const std::vector<int>& tokens,
                   const std::vector<Tensor3D>& layer_attentions) const {
        int num_layers = static_cast<int>(layer_attentions.size());
        if (num_layers == 0) return Matrix();
        int seq_len = layer_attentions[0].d1, num_heads = layer_attentions[0].d0;
        int start = prepend_bos ? 1 : 0;
        int end = seq_len;
        if (append_eos) for (int i = seq_len - 1; i >= 0; --i) if (tokens[i] == eos_idx) { end = i; break; }
        int L = end - start;
        int total_heads = num_layers * num_heads;
        std::vector<float> stacked(total_heads * L * L);
        for (int lay = 0; lay < num_layers; ++lay)
            for (int h = 0; h < num_heads; ++h) {
                int fi = lay * num_heads + h;
                for (int i = 0; i < L; ++i)
                    for (int j = 0; j < L; ++j)
                        stacked[fi * L * L + i * L + j] = layer_attentions[lay](h, start + i, start + j);
            }
        for (int f = 0; f < total_heads; ++f) math::symmetrize(stacked.data() + f * L * L, L);
        for (int f = 0; f < total_heads; ++f) math::apc(stacked.data() + f * L * L, L);
        Matrix contacts(L, L);
        for (int i = 0; i < L; ++i)
            for (int j = 0; j < L; ++j) {
                std::vector<float> features(total_heads);
                for (int f = 0; f < total_heads; ++f) features[f] = stacked[f * L * L + i * L + j];
                float logit = 0.0f;
                regression.forward_single(features.data(), &logit);
                contacts(i, j) = math::sigmoid(logit);
            }
        return contacts;
    }
};

// ===================================================================
// SECTION 13: GVP Modules
// ===================================================================

struct GVPLayer {
    int si = 0, vi = 0, so = 0, vo = 0, h_dim = 0;
    Linear wh, ws, wv, wg;
    bool has_vectors_in = false, has_vectors_out = false, vector_gate = false;
    bool use_scalar_act = true, use_vector_act = true;
    float eps = 1e-8f;

    GVPLayer() = default;
    GVPLayer(int si_, int vi_, int so_, int vo_, bool vg, bool sa, bool va, std::mt19937& rng, float e = 1e-8f)
        : si(si_), vi(vi_), so(so_), vo(vo_), vector_gate(vg), use_scalar_act(sa), use_vector_act(va), eps(e) {
        has_vectors_in = (vi > 0); has_vectors_out = (vo > 0);
        if (has_vectors_in) {
            h_dim = std::max(vi, vo);
            wh = Linear(vi, h_dim, false, rng);
            ws = Linear(h_dim + si, so, true, rng);
            if (has_vectors_out) {
                wv = Linear(h_dim, vo, false, rng);
                if (vector_gate) wg = Linear(so, vo, true, rng);
            }
        } else { ws = Linear(si, so, true, rng); }
    }

    void forward_scalar(const float* s_in, float* s_out, int n) const {
        ws.forward(s_in, s_out, n);
        if (use_scalar_act) math::relu_inplace(s_out, n * so);
    }

    void forward(const float* s_in, const float* v_in, float* s_out, float* v_out, int n) const {
        if (!has_vectors_in) {
            forward_scalar(s_in, s_out, n);
            if (has_vectors_out && v_out) math::vec_zero(v_out, n * vo * 3);
            return;
        }
        for (int i = 0; i < n; ++i) {
            const float* vi_ptr = v_in + i * vi * 3;
            std::vector<float> vt(3 * vi);
            for (int j = 0; j < vi; ++j) for (int d = 0; d < 3; ++d) vt[d * vi + j] = vi_ptr[j * 3 + d];
            std::vector<float> vh(3 * h_dim);
            for (int d = 0; d < 3; ++d) wh.forward(vt.data() + d * vi, vh.data() + d * h_dim, 1);
            std::vector<float> vn(h_dim);
            for (int j = 0; j < h_dim; ++j) {
                float sq = 0.0f;
                for (int d = 0; d < 3; ++d) sq += vh[d * h_dim + j] * vh[d * h_dim + j];
                vn[j] = std::sqrt(sq + eps);
            }
            std::vector<float> cat_input(si + h_dim);
            math::vec_copy(cat_input.data(), s_in + i * si, si);
            math::vec_copy(cat_input.data() + si, vn.data(), h_dim);
            ws.forward(cat_input.data(), s_out + i * so, 1);
            if (use_scalar_act) math::relu_inplace(s_out + i * so, so);
            if (has_vectors_out && v_out) {
                std::vector<float> vout_t(3 * vo);
                for (int d = 0; d < 3; ++d) wv.forward(vh.data() + d * h_dim, vout_t.data() + d * vo, 1);
                float* vo_ptr = v_out + i * vo * 3;
                for (int j = 0; j < vo; ++j) for (int d = 0; d < 3; ++d) vo_ptr[j * 3 + d] = vout_t[d * vo + j];
                if (vector_gate) {
                    std::vector<float> gate(vo);
                    wg.forward(s_out + i * so, gate.data(), 1);
                    for (int j = 0; j < vo; ++j) {
                        float g = math::sigmoid(gate[j]);
                        for (int d = 0; d < 3; ++d) vo_ptr[j * 3 + d] *= g;
                    }
                } else {
                    for (int j = 0; j < vo; ++j) {
                        float ns = 0.0f;
                        for (int d = 0; d < 3; ++d) ns += vo_ptr[j * 3 + d] * vo_ptr[j * 3 + d];
                        float g = std::sqrt(ns + eps);
                        if (use_vector_act) g = math::sigmoid(g);
                        for (int d = 0; d < 3; ++d) vo_ptr[j * 3 + d] *= g;
                    }
                }
            }
        }
    }
};

struct GVPLayerNorm {
    int s_dim = 0, v_dim = 0;
    ESM1bLayerNorm scalar_norm;
    float eps = 1e-8f;

    GVPLayerNorm() = default;
    GVPLayerNorm(int sd, int vd, float e = 1e-8f) : s_dim(sd), v_dim(vd), eps(e), scalar_norm(sd) {}

    void forward(float* s, float* v, int n) const {
        scalar_norm.forward_batch(s, n);
        if (v_dim > 0 && v) {
            for (int i = 0; i < n; ++i) {
                float sum_sq = 0.0f; int count = 0;
                for (int j = 0; j < v_dim; ++j) {
                    float ns = 0.0f;
                    for (int d = 0; d < 3; ++d) { float val = v[i * v_dim * 3 + j * 3 + d]; ns += val * val; }
                    if (ns > 2 * eps) { sum_sq += ns; count++; }
                }
                float rms = std::sqrt(sum_sq / std::max(count, 1) + eps);
                for (int j = 0; j < v_dim; ++j) {
                    float ns = 0.0f;
                    for (int d = 0; d < 3; ++d) { float val = v[i * v_dim * 3 + j * 3 + d]; ns += val * val; }
                    if (ns > 2 * eps) for (int d = 0; d < 3; ++d) v[i * v_dim * 3 + j * 3 + d] /= rms;
                }
            }
        }
    }
};

// ===================================================================
// SECTION 14-17: GVP Encoder, TransformerDecoder, GVPTransformer, ESMFold
// ===================================================================

struct GVPEncoderConfig {
    int num_encoder_layers = 3, node_hidden_dim_scalar = 128, node_hidden_dim_vector = 16;
    int edge_hidden_dim_scalar = 32, edge_hidden_dim_vector = 1, top_k_neighbors = 30;
    float dropout = 0.1f;
};

struct GVPEncoder {
    GVPEncoderConfig config;
    GVPLayer embed_node, embed_edge;
    GVPLayerNorm embed_node_norm, embed_edge_norm;
    Linear embed_confidence;

    struct ConvLayer {
        GVPLayer message_gvp1, message_gvp2, message_gvp3;
        GVPLayerNorm norm1, norm2;
        GVPLayer ff1, ff2;
    };
    std::vector<ConvLayer> encoder_layers;

    GVPEncoder() = default;
    GVPEncoder(const GVPEncoderConfig& cfg, std::mt19937& rng) : config(cfg) {
        int ns = cfg.node_hidden_dim_scalar, nv = cfg.node_hidden_dim_vector;
        int es = cfg.edge_hidden_dim_scalar, ev = cfg.edge_hidden_dim_vector;
        embed_node = GVPLayer(7, 3, ns, nv, false, false, false, rng, 1e-4f);
        embed_node_norm = GVPLayerNorm(ns, nv, 1e-4f);
        embed_edge = GVPLayer(34, 1, es, ev, false, false, false, rng, 1e-4f);
        embed_edge_norm = GVPLayerNorm(es, ev, 1e-4f);
        embed_confidence = Linear(16, ns, true, rng);
        encoder_layers.resize(cfg.num_encoder_layers);
        for (int i = 0; i < cfg.num_encoder_layers; ++i) {
            auto& l = encoder_layers[i];
            int ms = 2 * ns + es, mv = 2 * nv + ev;
            l.message_gvp1 = GVPLayer(ms, mv, ns, nv, true, true, true, rng, 1e-4f);
            l.message_gvp2 = GVPLayer(ns, nv, ns, nv, true, true, true, rng, 1e-4f);
            l.message_gvp3 = GVPLayer(ns, nv, ns, nv, false, false, false, rng, 1e-4f);
            l.norm1 = GVPLayerNorm(ns, nv, 1e-4f); l.norm2 = GVPLayerNorm(ns, nv, 1e-4f);
            l.ff1 = GVPLayer(ns, nv, 4 * ns, 2 * nv, true, true, true, rng, 1e-4f);
            l.ff2 = GVPLayer(4 * ns, 2 * nv, ns, nv, false, false, false, rng, 1e-4f);
        }
    }
};

struct TransformerDecoderConfig {
    int decoder_embed_dim = 512, decoder_ffn_embed_dim = 2048, decoder_attention_heads = 8;
    int decoder_layers = 8, encoder_embed_dim = 512;
    float dropout = 0.1f;
};

struct TransformerDecoder {
    TransformerDecoderConfig config;
    Embedding embed_tokens;
    SinusoidalPositionalEmbedding embed_positions;
    std::vector<TransformerDecoderLayer> layers;
    ESM1bLayerNorm layer_norm;
    Linear output_projection;
    int padding_idx = 0;

    TransformerDecoder() = default;
    TransformerDecoder(const TransformerDecoderConfig& cfg, int vs, int pad, std::mt19937& rng) : config(cfg), padding_idx(pad) {
        int ed = cfg.decoder_embed_dim;
        embed_tokens = Embedding(vs, ed, pad, rng);
        embed_positions = SinusoidalPositionalEmbedding(ed, pad);
        layer_norm = ESM1bLayerNorm(ed);
        output_projection = Linear(ed, vs, false, rng);
        layers.reserve(cfg.decoder_layers);
        for (int i = 0; i < cfg.decoder_layers; ++i)
            layers.emplace_back(ed, cfg.decoder_ffn_embed_dim, cfg.decoder_attention_heads, cfg.encoder_embed_dim, rng);
    }
    static std::vector<float> causal_mask(int size) {
        std::vector<float> mask(size * size, 0.0f);
        for (int i = 0; i < size; ++i) for (int j = i + 1; j < size; ++j) mask[i * size + j] = -1e9f;
        return mask;
    }
};

struct GVPTransformerConfig {
    GVPEncoderConfig gvp_config;
    int encoder_embed_dim = 512, encoder_ffn_embed_dim = 2048, encoder_attention_heads = 8, encoder_layers = 8;
    TransformerDecoderConfig decoder_config;
    float dropout = 0.1f;
};

struct GVPTransformerModel {
    GVPTransformerConfig config;
    Alphabet alphabet;
    GVPEncoder gvp_encoder;
    std::vector<TransformerEncoderLayer> enc_layers;
    ESM1bLayerNorm encoder_layer_norm;
    TransformerDecoder decoder;
    Linear embed_gvp_input_features, embed_confidence, embed_gvp_output;
    SinusoidalPositionalEmbedding embed_positions;
    Embedding embed_tokens_enc;

    GVPTransformerModel() = default;
    GVPTransformerModel(const GVPTransformerConfig& cfg, std::mt19937& rng)
        : config(cfg), alphabet(AlphabetArch::InvariantGVP) {
        int vs = alphabet.vocab_size(), ed = cfg.encoder_embed_dim;
        gvp_encoder = GVPEncoder(cfg.gvp_config, rng);
        enc_layers.reserve(cfg.encoder_layers);
        for (int i = 0; i < cfg.encoder_layers; ++i)
            enc_layers.emplace_back(ed, cfg.encoder_ffn_embed_dim, cfg.encoder_attention_heads, rng);
        encoder_layer_norm = ESM1bLayerNorm(ed);
        embed_gvp_input_features = Linear(15, ed, true, rng);
        embed_confidence = Linear(16, ed, true, rng);
        int gvp_out = cfg.gvp_config.node_hidden_dim_scalar + 3 * cfg.gvp_config.node_hidden_dim_vector;
        embed_gvp_output = Linear(gvp_out, ed, true, rng);
        embed_positions = SinusoidalPositionalEmbedding(ed, alphabet.pad_idx);
        embed_tokens_enc = Embedding(vs, ed, alphabet.pad_idx, rng);
        auto dc = cfg.decoder_config; dc.encoder_embed_dim = ed;
        decoder = TransformerDecoder(dc, vs, alphabet.pad_idx, rng);
    }
};

// ESMFold components
struct SequenceToPair {
    ESM1bLayerNorm layernorm;
    Linear proj, o_proj;
    int sequence_dim = 0, inner_dim = 0, pairwise_dim = 0;

    SequenceToPair() = default;
    SequenceToPair(int sd, int inner, int pd, std::mt19937& rng) : sequence_dim(sd), inner_dim(inner), pairwise_dim(pd) {
        layernorm = ESM1bLayerNorm(sd); proj = Linear(sd, inner * 2, true, rng); o_proj = Linear(2 * inner, pd, true, rng);
    }

    void forward(const float* seq_state, int L, float* pair_out) const {
        std::vector<float> normed(L * sequence_dim);
        math::vec_copy(normed.data(), seq_state, L * sequence_dim);
        layernorm.forward_batch(normed.data(), L);
        std::vector<float> projected(L * inner_dim * 2);
        proj.forward(normed.data(), projected.data(), L);
        for (int i = 0; i < L; ++i)
            for (int j = 0; j < L; ++j) {
                std::vector<float> cat_in(2 * inner_dim);
                for (int d = 0; d < inner_dim; ++d) {
                    float qi = projected[i * inner_dim * 2 + d];
                    float kj = projected[j * inner_dim * 2 + inner_dim + d];
                    cat_in[d] = qi * kj; cat_in[inner_dim + d] = qi - kj;
                }
                o_proj.forward(cat_in.data(), pair_out + (i * L + j) * pairwise_dim, 1);
            }
    }
};

struct PairToSequence {
    ESM1bLayerNorm layernorm;
    Linear linear_proj;
    int pairwise_dim = 0, num_heads = 0;

    PairToSequence() = default;
    PairToSequence(int pd, int nh, std::mt19937& rng) : pairwise_dim(pd), num_heads(nh) {
        layernorm = ESM1bLayerNorm(pd); linear_proj = Linear(pd, nh, false, rng);
    }
    void forward(const float* pair_state, int L, float* bias_out) const {
        std::vector<float> normed(L * L * pairwise_dim);
        math::vec_copy(normed.data(), pair_state, L * L * pairwise_dim);
        for (int i = 0; i < L * L; ++i) layernorm.forward(normed.data() + i * pairwise_dim);
        linear_proj.forward(normed.data(), bias_out, L * L);
    }
};

struct GatedAttention {
    int embed_dim = 0, num_heads = 0, head_width = 0;
    float rescale_factor = 0.0f;
    Linear proj, o_proj, g_proj;
    bool gated = false;

    GatedAttention() = default;
    GatedAttention(int ed, int nh, int hw, bool gate, std::mt19937& rng)
        : embed_dim(ed), num_heads(nh), head_width(hw), gated(gate) {
        rescale_factor = 1.0f / std::sqrt(static_cast<float>(hw));
        proj = Linear(ed, ed * 3, false, rng); o_proj = Linear(ed, ed, true, rng);
        if (gate) {
            g_proj = Linear(ed, ed, true, rng);
            for (auto& b : g_proj.bias) b = 1.0f; g_proj.weight.zeros();
        }
        o_proj.weight.zeros(); o_proj.bias.assign(ed, 0.0f);
    }

    void forward(const float* x, int L, float* out, const float* bias = nullptr,
                 const std::vector<bool>& mask = {}) const {
        int ed = embed_dim;
        std::vector<float> qkv(L * ed * 3);
        proj.forward(x, qkv.data(), L);
        std::vector<float> attn_out(L * ed, 0.0f);
        for (int h = 0; h < num_heads; ++h) {
            std::vector<float> scores(L * L);
            for (int qi = 0; qi < L; ++qi)
                for (int ki = 0; ki < L; ++ki) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_width; ++d)
                        dot += qkv[qi * ed * 3 + h * head_width + d] * qkv[ki * ed * 3 + ed + h * head_width + d];
                    dot *= rescale_factor;
                    if (bias) dot += bias[(qi * L + ki) * num_heads + h];
                    scores[qi * L + ki] = dot;
                }
            if (!mask.empty())
                for (int qi = 0; qi < L; ++qi)
                    for (int ki = 0; ki < L; ++ki) if (!mask[ki]) scores[qi * L + ki] = -1e9f;
            for (int qi = 0; qi < L; ++qi) math::softmax_row(scores.data() + qi * L, L);
            for (int qi = 0; qi < L; ++qi)
                for (int ki = 0; ki < L; ++ki) {
                    float w = scores[qi * L + ki];
                    for (int d = 0; d < head_width; ++d)
                        attn_out[qi * ed + h * head_width + d] += w * qkv[ki * ed * 3 + 2 * ed + h * head_width + d];
                }
        }
        if (gated) {
            std::vector<float> gate(L * ed);
            g_proj.forward(x, gate.data(), L);
            math::sigmoid_inplace(gate.data(), L * ed);
            math::vec_mul(attn_out.data(), gate.data(), L * ed);
        }
        o_proj.forward(attn_out.data(), out, L);
    }
};

struct ResidueMLP {
    ESM1bLayerNorm norm;
    Linear fc1, fc2;
    int embed_dim = 0, inner_dim = 0;

    ResidueMLP() = default;
    ResidueMLP(int ed, int inner, std::mt19937& rng) : embed_dim(ed), inner_dim(inner) {
        norm = ESM1bLayerNorm(ed); fc1 = Linear(ed, inner, true, rng); fc2 = Linear(inner, ed, true, rng);
    }
    void forward(float* x, int L) const {
        std::vector<float> res(L * embed_dim);
        math::vec_copy(res.data(), x, L * embed_dim);
        norm.forward_batch(x, L);
        std::vector<float> buf(L * inner_dim);
        fc1.forward(x, buf.data(), L); math::relu_inplace(buf.data(), L * inner_dim);
        fc2.forward(buf.data(), x, L);
        math::vec_add(x, res.data(), L * embed_dim);
    }
};

struct RelativePosition {
    int bins = 32;
    Embedding embedding;
    RelativePosition() = default;
    RelativePosition(int b, int pd, std::mt19937& rng) : bins(b) { embedding = Embedding(2 * b + 2, pd, -1, rng); }
    void forward(const std::vector<int>& residue_index, int L, float* out, int pd) const {
        for (int i = 0; i < L; ++i) for (int j = 0; j < L; ++j) {
            int diff = std::max(-bins, std::min(bins, residue_index[i] - residue_index[j])) + bins + 1;
            math::vec_copy(out + (i * L + j) * pd, embedding.lookup(diff), pd);
        }
    }
};

struct TriangularSelfAttentionBlock {
    int sequence_state_dim = 0, pairwise_state_dim = 0;
    ESM1bLayerNorm layernorm_1;
    SequenceToPair seq_to_pair;
    PairToSequence pair_to_seq;
    GatedAttention seq_attention;
    ResidueMLP mlp_seq, mlp_pair;

    TriangularSelfAttentionBlock() = default;
    TriangularSelfAttentionBlock(int sd, int pd, int shw, int /*phw*/, std::mt19937& rng) : sequence_state_dim(sd), pairwise_state_dim(pd) {
        int sh = sd / shw;
        layernorm_1 = ESM1bLayerNorm(sd); seq_to_pair = SequenceToPair(sd, pd / 2, pd, rng);
        pair_to_seq = PairToSequence(pd, sh, rng); seq_attention = GatedAttention(sd, sh, shw, true, rng);
        mlp_seq = ResidueMLP(sd, 4 * sd, rng); mlp_pair = ResidueMLP(pd, 4 * pd, rng);
    }

    void forward(float* seq_state, float* pair_state, int L, const std::vector<bool>& mask = {}) const {
        int sd = sequence_state_dim, pd = pairwise_state_dim;
        int nh = sd / 32; // approximate num_heads
        std::vector<float> bias(L * L * nh);
        pair_to_seq.forward(pair_state, L, bias.data());
        std::vector<float> normed(L * sd);
        math::vec_copy(normed.data(), seq_state, L * sd);
        layernorm_1.forward_batch(normed.data(), L);
        std::vector<float> ao(L * sd);
        seq_attention.forward(normed.data(), L, ao.data(), bias.data(), mask);
        math::vec_add(seq_state, ao.data(), L * sd);
        mlp_seq.forward(seq_state, L);
        std::vector<float> pu(L * L * pd);
        seq_to_pair.forward(seq_state, L, pu.data());
        math::vec_add(pair_state, pu.data(), L * L * pd);
        mlp_pair.forward(pair_state, L * L);
    }
};

struct FoldingTrunkConfig {
    int num_blocks = 48, sequence_state_dim = 1024, pairwise_state_dim = 128;
    int sequence_head_width = 32, pairwise_head_width = 32, position_bins = 32, max_recycles = 4;
};

struct FoldingTrunk {
    FoldingTrunkConfig config;
    RelativePosition pairwise_pos_emb;
    std::vector<TriangularSelfAttentionBlock> blocks;
    ESM1bLayerNorm recycle_s_norm, recycle_z_norm;
    Linear trunk2sm_s, trunk2sm_z;

    FoldingTrunk() = default;
    FoldingTrunk(const FoldingTrunkConfig& cfg, std::mt19937& rng) : config(cfg) {
        int cs = cfg.sequence_state_dim, cz = cfg.pairwise_state_dim;
        pairwise_pos_emb = RelativePosition(cfg.position_bins, cz, rng);
        blocks.reserve(cfg.num_blocks);
        for (int i = 0; i < cfg.num_blocks; ++i)
            blocks.emplace_back(cs, cz, cfg.sequence_head_width, cfg.pairwise_head_width, rng);
        recycle_s_norm = ESM1bLayerNorm(cs); recycle_z_norm = ESM1bLayerNorm(cz);
        trunk2sm_s = Linear(cs, 384, true, rng); trunk2sm_z = Linear(cz, 128, true, rng);
    }
};

struct CategoricalMixture {
    int num_bins = 50; float start = 0.0f, end = 1.0f;
    CategoricalMixture() = default;
    CategoricalMixture(int bins, float s = 0.0f, float e = 1.0f) : num_bins(bins), start(s), end(e) {}
    float mean(const float* logits) const {
        std::vector<float> probs(num_bins);
        math::vec_copy(probs.data(), logits, num_bins);
        math::softmax_row(probs.data(), num_bins);
        float result = 0.0f;
        for (int i = 0; i < num_bins; ++i) {
            float lo = start + (end - start) * i / num_bins, hi = start + (end - start) * (i + 1) / num_bins;
            result += probs[i] * (lo + hi) / 2.0f;
        }
        return result;
    }
};

struct ESMFoldConfig {
    FoldingTrunkConfig trunk;
    int lddt_head_hid_dim = 128;
    std::string esm_type = "esm2_3B";
    bool use_esm_attn_map = true;
};

// ===================================================================
// SECTION 18: ESM Model Configurations and Outputs
// ===================================================================

struct ESMConfig {
    int num_layers = 6, embed_dim = 320, ffn_embed_dim = 1280, attention_heads = 20, max_positions = 1024;
    bool add_bias_kv = false, token_dropout = true, use_esm1b_layer_norm = true, use_rotary_embeddings = false;
    float embed_scale = 1.0f;
    enum class ModelType { ESM1, ESM1b, ESM2, MSATransformer };
    ModelType model_type = ModelType::ESM1b;
};

struct ESMOutput {
    Matrix logits;
    std::map<int, Matrix> representations;
    std::map<int, Tensor3D> attentions;
    Matrix contacts;
};

struct MSAOutput {
    Tensor3D logits;
    std::map<int, Tensor3D> representations;
    std::vector<Tensor3D> row_attentions;
    std::vector<Tensor4D> col_attentions;
};

// ===================================================================
// SECTION 19: ESM Model
// ===================================================================

class ESMModel {
public:
    ESMConfig config;
    Alphabet alphabet;
    std::mt19937 rng;
    Embedding embed_tokens;
    LearnedPositionalEmbedding embed_positions;
    SinusoidalPositionalEmbedding sinusoidal_positions;
    Matrix embed_out;
    std::vector<float> embed_out_bias;
    std::vector<TransformerLayer> layers;
    ESM1bLayerNorm emb_layer_norm_before, emb_layer_norm_after;
    RobertaLMHead lm_head;
    ContactPredictionHead contact_head;
    bool has_lm_head = true, has_emb_norm_before = false;

    ESMModel() = default;

    explicit ESMModel(const ESMConfig& cfg, unsigned seed = 42) : config(cfg), rng(seed) {
        bool is_esm1 = (cfg.model_type == ESMConfig::ModelType::ESM1);
        bool is_esm2 = (cfg.model_type == ESMConfig::ModelType::ESM2);
        alphabet = is_esm1 ? Alphabet(AlphabetArch::ESM1) : Alphabet(AlphabetArch::ESM1b);
        int vs = alphabet.vocab_size(), ed = cfg.embed_dim;
        embed_tokens = Embedding(vs, ed, alphabet.pad_idx, rng);
        if (is_esm1) {
            sinusoidal_positions = SinusoidalPositionalEmbedding(ed, alphabet.pad_idx);
            embed_out = Matrix(vs, ed); embed_out.xavier_uniform(rng); has_lm_head = false;
        } else {
            embed_positions = LearnedPositionalEmbedding(cfg.max_positions, ed, alphabet.pad_idx, rng);
            has_lm_head = true;
        }
        bool use_rotary = is_esm2 || cfg.use_rotary_embeddings;
        bool use_1b_ln = !is_esm1 || cfg.use_esm1b_layer_norm;
        bool add_bias = is_esm1 || cfg.add_bias_kv;
        int ffn_ed = cfg.ffn_embed_dim > 0 ? cfg.ffn_embed_dim : 4 * ed;
        layers.reserve(cfg.num_layers);
        for (int i = 0; i < cfg.num_layers; ++i)
            layers.emplace_back(ed, ffn_ed, cfg.attention_heads, add_bias, use_1b_ln, use_rotary, rng);
        emb_layer_norm_after = ESM1bLayerNorm(ed);
        if (has_lm_head) lm_head.init(ed, vs, &embed_tokens.weight, rng);
        contact_head.init(cfg.num_layers * cfg.attention_heads, alphabet.prepend_bos, alphabet.append_eos, alphabet.eos_idx, rng);
    }

    ESMOutput forward(const std::vector<int>& tokens, const std::vector<int>& repr_layers = {},
                      bool need_head_weights = false, bool return_contacts = false) const {
        if (return_contacts) need_head_weights = true;
        int seq_len = static_cast<int>(tokens.size()), ed = config.embed_dim, vs = alphabet.vocab_size();
        bool is_esm1 = (config.model_type == ESMConfig::ModelType::ESM1);
        bool is_esm2 = (config.model_type == ESMConfig::ModelType::ESM2);
        std::vector<bool> padding_mask(seq_len, false);
        for (int i = 0; i < seq_len; ++i) if (tokens[i] == alphabet.pad_idx) padding_mask[i] = true;
        std::vector<float> x(seq_len * ed);
        float scale = is_esm1 ? std::sqrt(static_cast<float>(ed)) : config.embed_scale;
        for (int i = 0; i < seq_len; ++i) {
            const float* emb = embed_tokens.lookup(tokens[i]);
            for (int d = 0; d < ed; ++d) x[i * ed + d] = scale * emb[d];
        }
        if (is_esm2 && config.token_dropout) {
            int mc = 0, npc = 0;
            for (int i = 0; i < seq_len; ++i) {
                if (!padding_mask[i]) npc++;
                if (tokens[i] == alphabet.mask_idx) { mc++; for (int d = 0; d < ed; ++d) x[i * ed + d] = 0.0f; }
            }
            if (npc > 0) {
                float f = (1.0f - 0.12f) / (1.0f - static_cast<float>(mc) / npc + 1e-10f);
                for (int i = 0; i < seq_len; ++i) if (tokens[i] != alphabet.mask_idx) for (int d = 0; d < ed; ++d) x[i * ed + d] *= f;
            }
        }
        if (is_esm1) {
            SinusoidalPositionalEmbedding sp = sinusoidal_positions;
            auto pos = sp.make_positions(tokens, alphabet.pad_idx);
            for (int i = 0; i < seq_len; ++i) math::vec_add(x.data() + i * ed, sp.lookup(pos[i]), ed);
        } else {
            auto pos = embed_positions.compute_positions(tokens);
            for (int i = 0; i < seq_len; ++i) math::vec_add(x.data() + i * ed, embed_positions.lookup(pos[i]), ed);
        }
        for (int i = 0; i < seq_len; ++i) if (padding_mask[i]) math::vec_zero(x.data() + i * ed, ed);
        if (has_emb_norm_before) emb_layer_norm_before.forward_batch(x.data(), seq_len);
        std::set<int> repr_set(repr_layers.begin(), repr_layers.end());
        ESMOutput output;
        if (repr_set.count(0)) { Matrix m(seq_len, ed); math::vec_copy(m.data.data(), x.data(), seq_len * ed); output.representations[0] = std::move(m); }
        std::vector<Tensor3D> all_attn;
        for (int li = 0; li < config.num_layers; ++li) {
            Tensor3D aw;
            layers[li].forward(x.data(), seq_len, aw, padding_mask, need_head_weights);
            if (repr_set.count(li + 1)) { Matrix m(seq_len, ed); math::vec_copy(m.data.data(), x.data(), seq_len * ed); output.representations[li + 1] = std::move(m); }
            if (need_head_weights) { output.attentions[li] = aw; all_attn.push_back(std::move(aw)); }
        }
        emb_layer_norm_after.forward_batch(x.data(), seq_len);
        if (repr_set.count(config.num_layers)) { Matrix m(seq_len, ed); math::vec_copy(m.data.data(), x.data(), seq_len * ed); output.representations[config.num_layers] = std::move(m); }
        output.logits = Matrix(seq_len, vs);
        if (has_lm_head) lm_head.forward(x.data(), output.logits.data.data(), seq_len);
        else {
            math::matmul_transB(x.data(), embed_out.data.data(), output.logits.data.data(), seq_len, ed, vs);
            if (!embed_out_bias.empty()) for (int i = 0; i < seq_len; ++i) for (int j = 0; j < vs; ++j) output.logits(i, j) += embed_out_bias[j];
        }
        if (return_contacts && !all_attn.empty()) output.contacts = contact_head.predict(tokens, all_attn);
        return output;
    }

    ESMOutput infer(const std::string& sequence, const std::vector<int>& repr_layers = {},
                    bool need_head_weights = false, bool return_contacts = false) const {
        return forward(alphabet.encode(sequence), repr_layers, need_head_weights, return_contacts);
    }

    void print_summary(std::ostream& os = std::cout) const {
        const char* ts[] = {"ESM-1", "ESM-1b", "ESM-2", "MSA Transformer"};
        os << "ESM Model Summary (" << ts[static_cast<int>(config.model_type)] << ")\n"
           << "  Layers:          " << config.num_layers << "\n"
           << "  Embed dim:       " << config.embed_dim << "\n"
           << "  FFN embed dim:   " << config.ffn_embed_dim << "\n"
           << "  Attention heads: " << config.attention_heads << "\n"
           << "  Max positions:   " << config.max_positions << "\n"
           << "  Vocab size:      " << alphabet.vocab_size() << "\n"
           << "  Bias KV:         " << (config.add_bias_kv ? "yes" : "no") << "\n"
           << "  Rotary:          " << (config.use_rotary_embeddings ? "yes" : "no") << "\n";
        size_t tp = embed_tokens.weight.size();
        if (has_lm_head) tp += embed_positions.weight.size();
        for (auto& l : layers) {
            tp += l.self_attn.q_proj.weight.size() + l.self_attn.q_proj.bias.size()
                + l.self_attn.k_proj.weight.size() + l.self_attn.k_proj.bias.size()
                + l.self_attn.v_proj.weight.size() + l.self_attn.v_proj.bias.size()
                + l.self_attn.out_proj.weight.size() + l.self_attn.out_proj.bias.size()
                + l.fc1.weight.size() + l.fc1.bias.size() + l.fc2.weight.size() + l.fc2.bias.size();
            if (l.use_esm1b_layer_norm) tp += l.self_attn_ln_1b.weight.size() + l.self_attn_ln_1b.bias.size() + l.final_ln_1b.weight.size() + l.final_ln_1b.bias.size();
            else tp += l.self_attn_layer_norm.weight.size() + l.self_attn_layer_norm.bias.size() + l.final_layer_norm.weight.size() + l.final_layer_norm.bias.size();
        }
        tp += emb_layer_norm_after.weight.size() + emb_layer_norm_after.bias.size();
        if (has_lm_head) tp += lm_head.dense.weight.size() + lm_head.dense.bias.size() + lm_head.layer_norm.weight.size() + lm_head.layer_norm.bias.size() + lm_head.bias.size();
        else tp += embed_out.size();
        os << "  Total params:    " << tp << "\n";
    }
};

// ===================================================================
// SECTION 20: Factory Functions
// ===================================================================

inline ESMModel create_esm2(int num_layers, int embed_dim, int attention_heads, unsigned seed = 42) {
    ESMConfig cfg; cfg.num_layers = num_layers; cfg.embed_dim = embed_dim; cfg.ffn_embed_dim = 4 * embed_dim;
    cfg.attention_heads = attention_heads; cfg.max_positions = 1024; cfg.add_bias_kv = false;
    cfg.token_dropout = true; cfg.use_esm1b_layer_norm = true; cfg.use_rotary_embeddings = true;
    cfg.embed_scale = 1.0f; cfg.model_type = ESMConfig::ModelType::ESM2;
    return ESMModel(cfg, seed);
}

inline ESMModel create_esm1b(int nl, int ed, int ffn, int nh, unsigned seed = 42) {
    ESMConfig cfg; cfg.num_layers = nl; cfg.embed_dim = ed; cfg.ffn_embed_dim = ffn;
    cfg.attention_heads = nh; cfg.model_type = ESMConfig::ModelType::ESM1b;
    cfg.add_bias_kv = false; cfg.use_esm1b_layer_norm = true; cfg.embed_scale = 1.0f;
    return ESMModel(cfg, seed);
}

inline ESMModel create_esm1(int nl, int ed, int ffn, int nh, unsigned seed = 42) {
    ESMConfig cfg; cfg.num_layers = nl; cfg.embed_dim = ed; cfg.ffn_embed_dim = ffn;
    cfg.attention_heads = nh; cfg.model_type = ESMConfig::ModelType::ESM1;
    cfg.add_bias_kv = true; cfg.use_esm1b_layer_norm = false;
    cfg.embed_scale = std::sqrt(static_cast<float>(ed));
    return ESMModel(cfg, seed);
}

// ===================================================================
// SECTION 21: MSA Transformer Model
// ===================================================================

struct MSATransformerConfig {
    int num_layers = 12, embed_dim = 768, ffn_embed_dim = 3072, attention_heads = 12, max_positions = 1024;
};

class MSATransformerModel {
public:
    MSATransformerConfig config;
    Alphabet alphabet;
    std::mt19937 rng;
    Embedding embed_tokens;
    LearnedPositionalEmbedding embed_positions;
    ESM1bLayerNorm emb_layer_norm_before, emb_layer_norm_after;
    std::vector<AxialTransformerLayer> layers;
    RobertaLMHead lm_head;
    ContactPredictionHead contact_head;

    MSATransformerModel() = default;
    explicit MSATransformerModel(const MSATransformerConfig& cfg, unsigned seed = 42)
        : config(cfg), alphabet(AlphabetArch::MSATransformer), rng(seed) {
        int vs = alphabet.vocab_size(), ed = cfg.embed_dim;
        embed_tokens = Embedding(vs, ed, alphabet.pad_idx, rng);
        embed_positions = LearnedPositionalEmbedding(cfg.max_positions, ed, alphabet.pad_idx, rng);
        emb_layer_norm_before = ESM1bLayerNorm(ed); emb_layer_norm_after = ESM1bLayerNorm(ed);
        layers.reserve(cfg.num_layers);
        for (int i = 0; i < cfg.num_layers; ++i) layers.emplace_back(ed, cfg.ffn_embed_dim, cfg.attention_heads, rng);
        lm_head.init(ed, vs, &embed_tokens.weight, rng);
        contact_head.init(cfg.num_layers * cfg.attention_heads, alphabet.prepend_bos, alphabet.append_eos, alphabet.eos_idx, rng);
    }

    MSAOutput forward(const std::vector<std::vector<int>>& msa_tokens,
                      const std::vector<int>& repr_layers = {}, bool need_head_weights = false) const {
        int nr = static_cast<int>(msa_tokens.size());
        if (nr == 0) return MSAOutput();
        int sl = static_cast<int>(msa_tokens[0].size()), ed = config.embed_dim, vs = alphabet.vocab_size();
        Tensor3D x(nr, sl, ed);
        for (int r = 0; r < nr; ++r) {
            for (int c = 0; c < sl; ++c) math::vec_copy(x.slice_ptr(r, c), embed_tokens.lookup(msa_tokens[r][c]), ed);
            auto pos = embed_positions.compute_positions(msa_tokens[r]);
            for (int c = 0; c < sl; ++c) math::vec_add(x.slice_ptr(r, c), embed_positions.lookup(pos[c]), ed);
        }
        for (int r = 0; r < nr; ++r) for (int c = 0; c < sl; ++c) emb_layer_norm_before.forward(x.slice_ptr(r, c));
        for (int r = 0; r < nr; ++r) for (int c = 0; c < sl; ++c)
            if (msa_tokens[r][c] == alphabet.pad_idx) math::vec_zero(x.slice_ptr(r, c), ed);
        std::set<int> rs(repr_layers.begin(), repr_layers.end());
        MSAOutput output;
        if (rs.count(0)) output.representations[0] = x;
        for (int li = 0; li < config.num_layers; ++li) {
            Tensor3D raw; Tensor4D caw;
            layers[li].forward(x, raw, caw);
            if (rs.count(li + 1)) output.representations[li + 1] = x;
            if (need_head_weights) { output.row_attentions.push_back(std::move(raw)); output.col_attentions.push_back(std::move(caw)); }
        }
        for (int r = 0; r < nr; ++r) for (int c = 0; c < sl; ++c) emb_layer_norm_after.forward(x.slice_ptr(r, c));
        if (rs.count(config.num_layers)) output.representations[config.num_layers] = x;
        output.logits = Tensor3D(nr, sl, vs);
        for (int r = 0; r < nr; ++r) lm_head.forward(x.slice_ptr(r, 0), output.logits.slice_ptr(r, 0), sl);
        return output;
    }

    void print_summary(std::ostream& os = std::cout) const {
        os << "MSA Transformer Model Summary\n  Layers: " << config.num_layers
           << "\n  Embed dim: " << config.embed_dim << "\n  FFN dim: " << config.ffn_embed_dim
           << "\n  Heads: " << config.attention_heads << "\n  Vocab: " << alphabet.vocab_size() << "\n";
    }
};

// ===================================================================
// SECTION 22: Pretrained Model Registry
// ===================================================================

namespace pretrained {
struct ModelInfo {
    std::string name; int num_layers, embed_dim, ffn_embed_dim, attention_heads;
    ESMConfig::ModelType model_type; std::string url;
};

inline const std::vector<ModelInfo>& model_registry() {
    static const std::vector<ModelInfo> r = {
        {"esm1_t34_670M_UR50S",34,1280,5120,20,ESMConfig::ModelType::ESM1,""},
        {"esm1_t12_85M_UR50S",12,768,3072,12,ESMConfig::ModelType::ESM1,""},
        {"esm1_t6_43M_UR50S",6,768,3072,12,ESMConfig::ModelType::ESM1,""},
        {"esm1b_t33_650M_UR50S",33,1280,5120,20,ESMConfig::ModelType::ESM1b,""},
        {"esm1v_t33_650M_UR90S_1",33,1280,5120,20,ESMConfig::ModelType::ESM1b,""},
        {"esm1v_t33_650M_UR90S_2",33,1280,5120,20,ESMConfig::ModelType::ESM1b,""},
        {"esm1v_t33_650M_UR90S_3",33,1280,5120,20,ESMConfig::ModelType::ESM1b,""},
        {"esm1v_t33_650M_UR90S_4",33,1280,5120,20,ESMConfig::ModelType::ESM1b,""},
        {"esm1v_t33_650M_UR90S_5",33,1280,5120,20,ESMConfig::ModelType::ESM1b,""},
        {"esm2_t6_8M_UR50D",6,320,1280,20,ESMConfig::ModelType::ESM2,""},
        {"esm2_t12_35M_UR50D",12,480,1920,20,ESMConfig::ModelType::ESM2,""},
        {"esm2_t30_150M_UR50D",30,640,2560,20,ESMConfig::ModelType::ESM2,""},
        {"esm2_t33_650M_UR50D",33,1280,5120,20,ESMConfig::ModelType::ESM2,""},
        {"esm2_t36_3B_UR50D",36,2560,10240,40,ESMConfig::ModelType::ESM2,""},
        {"esm2_t48_15B_UR50D",48,5120,20480,40,ESMConfig::ModelType::ESM2,""},
    };
    return r;
}

inline ESMModel create_model(const std::string& name, unsigned seed = 42) {
    for (auto& info : model_registry()) {
        if (info.name == name) {
            ESMConfig cfg; cfg.num_layers = info.num_layers; cfg.embed_dim = info.embed_dim;
            cfg.ffn_embed_dim = info.ffn_embed_dim; cfg.attention_heads = info.attention_heads;
            cfg.model_type = info.model_type; cfg.max_positions = 1024;
            bool is1 = info.model_type == ESMConfig::ModelType::ESM1;
            bool is2 = info.model_type == ESMConfig::ModelType::ESM2;
            cfg.add_bias_kv = is1; cfg.use_esm1b_layer_norm = !is1;
            cfg.use_rotary_embeddings = is2; cfg.token_dropout = is2;
            cfg.embed_scale = is1 ? std::sqrt(static_cast<float>(info.embed_dim)) : 1.0f;
            return ESMModel(cfg, seed);
        }
    }
    throw std::runtime_error("Unknown model: " + name);
}

inline std::vector<std::string> list_models() {
    std::vector<std::string> n; for (auto& i : model_registry()) n.push_back(i.name); return n;
}
} // namespace pretrained

// ===================================================================
// SECTION 23: Feature Extraction Utilities
// ===================================================================

namespace features {

inline Matrix extract_residue_embeddings(const ESMOutput& output, int layer, const Alphabet& alpha) {
    auto it = output.representations.find(layer);
    if (it == output.representations.end()) return Matrix();
    const Matrix& repr = it->second;
    int sl = repr.rows, ed = repr.cols, start = alpha.prepend_bos ? 1 : 0;
    int end = sl - (alpha.append_eos ? 1 : 0), L = end - start;
    Matrix result(L, ed);
    for (int i = 0; i < L; ++i) math::vec_copy(result.row_ptr(i), repr.row_ptr(start + i), ed);
    return result;
}

inline std::vector<float> mean_embedding(const Matrix& embeddings) {
    if (embeddings.rows == 0) return {};
    int ed = embeddings.cols;
    std::vector<float> mean(ed, 0.0f);
    for (int i = 0; i < embeddings.rows; ++i) math::vec_add(mean.data(), embeddings.row_ptr(i), ed);
    for (int d = 0; d < ed; ++d) mean[d] /= embeddings.rows;
    return mean;
}

inline float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    int n = static_cast<int>(a.size());
    float dot = math::vec_dot(a.data(), b.data(), n);
    float na = math::vec_norm(a.data(), n), nb = math::vec_norm(b.data(), n);
    return (na < 1e-10f || nb < 1e-10f) ? 0.0f : dot / (na * nb);
}

inline std::vector<std::array<float, 6>> compute_dihedral_features(const std::vector<std::array<Vec3, 3>>& coords) {
    int L = static_cast<int>(coords.size());
    std::vector<Vec3> atoms; atoms.reserve(3 * L);
    for (auto& c : coords) { atoms.push_back(c[0]); atoms.push_back(c[1]); atoms.push_back(c[2]); }
    int na = static_cast<int>(atoms.size());
    std::vector<Vec3> U(na - 1);
    for (int i = 0; i < na - 1; ++i) U[i] = (atoms[i + 1] - atoms[i]).normalized();
    std::vector<float> D(na - 3, 0.0f);
    for (int i = 0; i < static_cast<int>(U.size()) - 2; ++i) {
        Vec3 n2 = U[i].cross(U[i + 1]).normalized(), n1 = U[i + 1].cross(U[i + 2]).normalized();
        float cosD = std::max(-1.0f + 1e-7f, std::min(1.0f - 1e-7f, n2.dot(n1)));
        D[i] = (U[i].dot(n1) > 0 ? 1.0f : -1.0f) * std::acos(cosD);
    }
    std::vector<std::array<float, 6>> features(L);
    for (int i = 0; i < L; ++i) for (int j = 0; j < 3; ++j) {
        int idx = 3 * i + j; float angle = (idx >= 1 && idx < static_cast<int>(D.size()) + 1) ? D[idx - 1] : 0.0f;
        features[i][j] = std::cos(angle); features[i][j + 3] = std::sin(angle);
    }
    return features;
}

inline std::array<Vec3, 3> get_rotation_frame(Vec3 N, Vec3 CA, Vec3 C) {
    Vec3 e1 = (C - CA).normalized();
    Vec3 v2 = N - CA, u2 = v2 - e1 * e1.dot(v2);
    Vec3 e2 = u2.normalized(), e3 = e1.cross(e2);
    return {e1, e2, e3};
}
} // namespace features

// ===================================================================
// SECTION 24: Inverse Folding Utilities
// ===================================================================

namespace inverse_folding {
using BackboneCoords = std::vector<std::array<Vec3, 3>>;

struct PDBAtom { std::string name; Vec3 coord; int residue_index; char chain_id; };

inline BackboneCoords extract_backbone(const std::vector<PDBAtom>& atoms) {
    std::map<int, std::map<std::string, Vec3>> residues;
    for (auto& a : atoms) residues[a.residue_index][a.name] = a.coord;
    BackboneCoords coords;
    for (auto& [idx, am] : residues)
        if (am.count("N") && am.count("CA") && am.count("C"))
            coords.push_back({am["N"], am["CA"], am["C"]});
    return coords;
}

inline Matrix compute_distance_matrix(const BackboneCoords& coords) {
    int L = static_cast<int>(coords.size());
    Matrix dist(L, L);
    for (int i = 0; i < L; ++i) for (int j = 0; j < L; ++j) dist(i, j) = (coords[i][1] - coords[j][1]).norm();
    return dist;
}

inline std::vector<std::vector<int>> knn(const Matrix& dist, int k) {
    int L = dist.rows;
    std::vector<std::vector<int>> neighbors(L);
    for (int i = 0; i < L; ++i) {
        std::vector<std::pair<float, int>> dists;
        for (int j = 0; j < L; ++j) if (i != j) dists.push_back({dist(i, j), j});
        std::partial_sort(dists.begin(), dists.begin() + std::min(k, static_cast<int>(dists.size())), dists.end());
        for (int j = 0; j < std::min(k, static_cast<int>(dists.size())); ++j) neighbors[i].push_back(dists[j].second);
    }
    return neighbors;
}

inline Vec3 compute_sidechain_vector(Vec3 N, Vec3 CA, Vec3 C) {
    Vec3 cd = (C - CA).normalized(), nd = (N - CA).normalized();
    Vec3 bisector = (cd + nd).normalized(), perp = cd.cross(nd).normalized();
    return bisector * (-std::sqrt(1.0f / 3.0f)) - perp * std::sqrt(2.0f / 3.0f);
}
} // namespace inverse_folding

// ===================================================================
// SECTION 25: ESMFold Utilities
// ===================================================================

namespace esmfold {

inline std::vector<int> encode_sequence(const std::string& seq) {
    static const std::unordered_map<char, int> m = {
        {'A',0},{'R',1},{'N',2},{'D',3},{'C',4},{'Q',5},{'E',6},{'G',7},{'H',8},{'I',9},
        {'L',10},{'K',11},{'M',12},{'F',13},{'P',14},{'S',15},{'T',16},{'W',17},{'Y',18},{'V',19},{'X',20}
    };
    std::vector<int> enc; enc.reserve(seq.size());
    for (char c : seq) { auto it = m.find(c); enc.push_back(it != m.end() ? it->second : 20); }
    return enc;
}

struct ChainEncoding {
    std::vector<int> aatype, residue_index, chain_index;
    std::vector<float> linker_mask;
};

inline ChainEncoding encode_multimer(const std::string& seq, int rio = 512, const std::string& linker = "GGGGGGGGGGGGGGGGGGGGGGGGG") {
    ChainEncoding result;
    std::vector<std::string> chains;
    std::string cur;
    for (char c : seq) { if (c == ':') { chains.push_back(cur); cur.clear(); } else cur += c; }
    chains.push_back(cur);
    std::string full;
    for (size_t i = 0; i < chains.size(); ++i) { if (i > 0) full += linker; full += chains[i]; }
    result.aatype = encode_sequence(full);
    int tl = static_cast<int>(full.size());
    result.residue_index.resize(tl); result.linker_mask.resize(tl, 1.0f); result.chain_index.resize(tl, 0);
    int pos = 0;
    for (int ci = 0; ci < static_cast<int>(chains.size()); ++ci) {
        if (ci > 0) for (int j = 0; j < static_cast<int>(linker.size()); ++j) {
            result.residue_index[pos] = pos + ci * rio; result.linker_mask[pos] = 0.0f; result.chain_index[pos] = ci - 1; pos++;
        }
        for (int j = 0; j < static_cast<int>(chains[ci].size()); ++j) {
            result.residue_index[pos] = pos + ci * rio; result.chain_index[pos] = ci; pos++;
        }
    }
    return result;
}

inline std::vector<std::vector<int>> compute_distogram(const std::vector<Vec3>& ca, float minb = 3.375f, float maxb = 21.375f, int nb = 15) {
    int L = static_cast<int>(ca.size());
    std::vector<float> boundaries(nb - 1);
    for (int i = 0; i < nb - 1; ++i) { float b = minb + (maxb - minb) * i / (nb - 2); boundaries[i] = b * b; }
    std::vector<std::vector<int>> bins(L, std::vector<int>(L, 0));
    for (int i = 0; i < L; ++i) for (int j = 0; j < L; ++j) {
        float ds = (ca[i] - ca[j]).norm_sq();
        int bin = 0; for (int b = 0; b < nb - 1; ++b) if (ds > boundaries[b]) bin++;
        bins[i][j] = bin;
    }
    return bins;
}

inline float categorical_lddt(const float* logits, int n_atoms, int bins = 50) {
    CategoricalMixture mix(bins);
    float sum = 0.0f;
    for (int i = 0; i < n_atoms; ++i) sum += mix.mean(logits + i * bins);
    return sum / std::max(n_atoms, 1);
}

} // namespace esmfold

} // namespace esm

#endif // ESM_H
