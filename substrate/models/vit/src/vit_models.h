#pragma once
#include "../../../core/nn/module.h"
#include "../../../core/losses/losses.h"
#include <cmath>
#include <vector>

// ============================================================
// Multi-Head Attention (self-contained for ViT)
// ============================================================
// Input/output layout: (batch*seq_len) x d_model
// seq_len is passed separately for internal reshaping.
// ============================================================
class ViTMultiHeadAttention {
public:
    Linear W_q, W_k, W_v, W_o;
    size_t d_model, num_heads, d_k;

    // Caches for backward
    Tensor cached_Q, cached_K, cached_V;
    Tensor cached_attn_weights;   // (batch*num_heads*seq_len) x seq_len
    Tensor cached_attn_output;    // (batch*num_heads*seq_len) x d_k
    size_t cached_seq_len;
    size_t cached_batch;

    ViTMultiHeadAttention(size_t d_model, size_t num_heads, std::mt19937& rng);

    // input: (batch*seq_len) x d_model -> (batch*seq_len) x d_model
    Tensor forward(const Tensor& input, size_t seq_len);
    Tensor backward(const Tensor& grad_output, size_t seq_len);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();
};

// ============================================================
// Feed-Forward Network for ViT encoder blocks
// ============================================================
class ViTFeedForward {
public:
    Linear fc1, fc2;
    GELU gelu;

    ViTFeedForward(size_t d_model, size_t d_ff, std::mt19937& rng);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();
};

// ============================================================
// ViT Transformer Encoder Block (Pre-Norm)
// ============================================================
// Architecture:
//   x -> LayerNorm -> MHA -> + residual -> LayerNorm -> FFN -> + residual
// ============================================================
class ViTEncoderBlock {
public:
    ViTMultiHeadAttention mha;
    ViTFeedForward ffn;
    LayerNorm ln1, ln2;
    Dropout dropout1, dropout2;

    // Caches for backward
    Tensor cached_input;         // residual source
    Tensor cached_ln1_out;       // output of first layer norm
    Tensor cached_mha_out;       // MHA output after dropout
    Tensor cached_after_res1;    // after first residual add
    Tensor cached_ln2_out;       // output of second layer norm
    size_t cached_seq_len;

    ViTEncoderBlock(size_t d_model, size_t num_heads, size_t d_ff,
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
// Patch Embedding
// ============================================================
// Splits a 28x28 image into non-overlapping patches of patch_size x patch_size,
// projects each patch into embed_dim, prepends a learnable CLS token,
// and adds learnable positional embeddings.
//
// With patch_size=7: 28/7 = 4 patches per side -> 16 patches total.
// Sequence length = num_patches + 1 (CLS) = 17.
// ============================================================
class PatchEmbedding {
public:
    Linear projection;    // (in_channels * patch_size * patch_size) -> embed_dim
    size_t img_size, patch_size, in_channels, embed_dim;
    size_t num_patches;   // (img_size/patch_size)^2
    size_t seq_len;       // num_patches + 1

    Tensor pos_embedding; // (num_patches+1) x embed_dim  (learnable)
    Tensor cls_token;     // 1 x embed_dim                (learnable)

    // Gradient accumulators for pos_embedding and cls_token
    Tensor grad_pos_embedding;
    Tensor grad_cls_token;

    // Caches for backward
    Tensor cached_patches;   // batch x num_patches rows, each (C*P*P) cols
    size_t cached_batch;

    PatchEmbedding(size_t img_size, size_t patch_size, size_t in_channels,
                   size_t embed_dim, std::mt19937& rng);

    // Input: batch x (C*H*W)
    // Output: (batch * seq_len) x embed_dim
    //   where seq_len = num_patches + 1
    Tensor forward(const Tensor& input, size_t batch_size);

    // grad: (batch * seq_len) x embed_dim
    // Returns: batch x (C*H*W)  (gradient w.r.t. input image)
    Tensor backward(const Tensor& grad, size_t batch_size);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();
};

// ============================================================
// Vision Transformer Classifier for MNIST
// ============================================================
// Architecture:
//   Input (batch x 784) -> PatchEmbedding -> sequence of ViTEncoderBlocks
//   -> extract CLS token -> Linear classifier -> logits
//
// Default config:
//   img_size=28, patch_size=7, in_channels=1
//   embed_dim=64, num_heads=4, d_ff=128, num_blocks=2
//   dropout_rate=0.1, num_classes=10
// ============================================================
class ViTClassifier {
public:
    PatchEmbedding patch_embed;
    std::vector<ViTEncoderBlock> blocks;
    LayerNorm final_ln;
    Linear classifier;

    size_t img_size, patch_size, in_channels;
    size_t embed_dim, num_heads, d_ff, num_blocks;
    size_t num_classes;
    double dropout_rate;

    // Caches for backward
    Tensor cached_encoder_out;   // (batch * seq_len) x embed_dim
    Tensor cached_cls_tokens;    // batch x embed_dim (extracted CLS)
    Tensor cached_ln_out;        // batch x embed_dim (after final LN)
    size_t cached_batch;
    size_t cached_seq_len;

    // Full constructor
    ViTClassifier(size_t img_size, size_t patch_size, size_t in_channels,
                  size_t num_classes, size_t embed_dim, size_t num_heads,
                  size_t d_ff, size_t num_blocks, double dropout_rate,
                  std::mt19937& rng);

    // Convenience constructor with MNIST defaults
    ViTClassifier(std::mt19937& rng);

    // input: batch x 784
    // output: batch x num_classes (logits)
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();

    void train();
    void eval();
};
