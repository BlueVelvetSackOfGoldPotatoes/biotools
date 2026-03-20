#pragma once
#include "../../../core/nn/module.h"
#include "../../../core/losses/losses.h"
#include <cmath>

// ============================================================
// Multi-Head Attention
// ============================================================
// Input/output layout: (batch*seq_len) x d_model
// The seq_len parameter is passed separately so we can reshape
// internally for the attention computation.
// ============================================================
class MultiHeadAttention {
public:
    Linear W_q, W_k, W_v, W_o;
    size_t d_model, num_heads, d_k;

    // Stored for interpretability: (batch * num_heads) x seq_len x seq_len
    // Flattened into a 2D tensor: (batch * num_heads * seq_len) x seq_len
    Tensor cached_attn_weights;

    // Caches for backward pass
    Tensor cached_input;        // (batch*seq_len) x d_model
    Tensor cached_Q;            // (batch*seq_len) x d_model
    Tensor cached_K;            // (batch*seq_len) x d_model
    Tensor cached_V;            // (batch*seq_len) x d_model
    Tensor cached_attn_output;  // (batch*num_heads*seq_len) x d_k -- per-head output pre-concat
    size_t cached_seq_len;
    size_t cached_batch;
    bool cached_causal;

    MultiHeadAttention(size_t d_model, size_t num_heads, std::mt19937& rng);

    // input: (batch*seq_len) x d_model
    // Returns: (batch*seq_len) x d_model
    Tensor forward(const Tensor& input, size_t seq_len, bool causal_mask = false);
    Tensor backward(const Tensor& grad_output, size_t seq_len);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();
};

// ============================================================
// Feed-Forward Network (used inside each encoder block)
// ============================================================
// Two-layer MLP with GELU activation:
//   Linear(d_model -> d_ff) -> GELU -> Linear(d_ff -> d_model)
// ============================================================
class FeedForward {
public:
    Linear fc1, fc2;
    GELU gelu;

    FeedForward(size_t d_model, size_t d_ff, std::mt19937& rng);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();
};

// ============================================================
// Transformer Encoder Block (Post-Norm)
// ============================================================
// Architecture:
//   x -> MHA -> + residual -> LayerNorm -> FFN -> + residual -> LayerNorm
// ============================================================
class TransformerEncoderBlock {
public:
    MultiHeadAttention mha;
    FeedForward ffn;
    LayerNorm ln1, ln2;
    Dropout dropout1, dropout2;

    // Caches for backward
    Tensor cached_mha_input;     // input to MHA (and residual source)
    Tensor cached_mha_out;       // MHA output (before residual add)
    Tensor cached_ln1_out;       // output of first layer norm
    Tensor cached_ffn_out;       // FFN output (before residual add)
    size_t cached_seq_len;

    TransformerEncoderBlock(size_t d_model, size_t num_heads, size_t d_ff,
                            double dropout_rate, std::mt19937& rng);

    // input: (batch*seq_len) x d_model
    Tensor forward(const Tensor& input, size_t seq_len);
    Tensor backward(const Tensor& grad_output, size_t seq_len);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();

    void train();
    void eval();
};

// ============================================================
// Transformer Classifier for MNIST
// ============================================================
// Treats 28x28 image as a sequence of 28 rows, each 28-dim.
//
// Architecture:
//   Input projection: Linear(28 -> d_model)
//   + sinusoidal positional encoding
//   N x TransformerEncoderBlock
//   Mean pooling over sequence dimension
//   Linear(d_model -> num_classes)
//
// Default config: d_model=64, num_heads=4, d_ff=128, 2 blocks
// ============================================================
class TransformerClassifier {
public:
    Linear input_proj;      // 28 -> d_model
    Linear classifier;      // d_model -> num_classes
    std::vector<TransformerEncoderBlock> blocks;

    size_t d_model, num_heads, d_ff, num_blocks;
    size_t input_dim;       // 28 (row width)
    size_t seq_len;         // 28 (number of rows)
    size_t num_classes;     // 10
    double dropout_rate;

    // Sinusoidal positional encoding: seq_len x d_model
    Tensor pos_encoding;

    // Caches for backward
    Tensor cached_input;            // original input: batch x 784
    Tensor cached_proj_out;         // after input projection + pos encoding
    Tensor cached_mean_pooled;      // batch x d_model (after mean pooling)
    size_t cached_batch;

    TransformerClassifier(size_t input_dim, size_t seq_len, size_t num_classes,
                          size_t d_model, size_t num_heads, size_t d_ff,
                          size_t num_blocks, double dropout_rate,
                          std::mt19937& rng);

    // Convenience constructor with defaults for MNIST
    TransformerClassifier(std::mt19937& rng);

    // input: batch x 784
    // output: batch x num_classes (logits)
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();

    void train();
    void eval();

private:
    void build_positional_encoding();
};
