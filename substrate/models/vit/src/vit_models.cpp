#include "vit_models.h"
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#define VIT_OMP_PAR_FOR _Pragma("omp parallel for schedule(static)")
#define VIT_OMP_PAR_FOR_COLLAPSE2 _Pragma("omp parallel for collapse(2) schedule(static)")
#define VIT_OMP_PAR_FOR_COLLAPSE3 _Pragma("omp parallel for collapse(3) schedule(static)")
#else
#define VIT_OMP_PAR_FOR
#define VIT_OMP_PAR_FOR_COLLAPSE2
#define VIT_OMP_PAR_FOR_COLLAPSE3
#endif

// ============================================================
// ViTMultiHeadAttention
// ============================================================
ViTMultiHeadAttention::ViTMultiHeadAttention(size_t dm, size_t nh, std::mt19937& rng)
    : W_q(dm, dm, "vit_attn_q", rng),
      W_k(dm, dm, "vit_attn_k", rng),
      W_v(dm, dm, "vit_attn_v", rng),
      W_o(dm, dm, "vit_attn_o", rng),
      d_model(dm), num_heads(nh), d_k(nh > 0 ? dm / nh : 0),
      cached_seq_len(0), cached_batch(0) {
    if (nh == 0 || dm % nh != 0) {
        throw std::runtime_error("ViTMultiHeadAttention: d_model must be divisible by num_heads and num_heads > 0");
    }
}

Tensor ViTMultiHeadAttention::forward(const Tensor& input, size_t seq_len) {
    // input: (batch*seq_len) x d_model
    if (seq_len == 0) {
        throw std::runtime_error("ViTMultiHeadAttention::forward seq_len must be > 0");
    }
    size_t total = input.rows;
    if (total % seq_len != 0) {
        throw std::runtime_error("ViTMultiHeadAttention::forward input rows must be divisible by seq_len");
    }
    size_t batch = total / seq_len;
    cached_seq_len = seq_len;
    cached_batch = batch;

    // Project Q, K, V
    cached_Q = W_q.forward(input);   // (B*S) x d_model
    cached_K = W_k.forward(input);
    cached_V = W_v.forward(input);

    // Reshape and compute attention per head
    // For each head h, Q_h is (B*S) x d_k taken from columns [h*d_k, (h+1)*d_k)
    // We need to compute attention: softmax(Q_h K_h^T / sqrt(d_k)) V_h

    // Output: (B*num_heads*S) x d_k -- one block per (batch, head) pair
    cached_attn_output = Tensor(batch * num_heads * seq_len, d_k);
    cached_attn_weights = Tensor(batch * num_heads * seq_len, seq_len);

    double scale = 1.0 / std::sqrt((double)d_k);

    VIT_OMP_PAR_FOR_COLLAPSE2
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            // Extract Q_h, K_h, V_h for this batch element and head
            // Q_h: seq_len x d_k
            Tensor Q_h(seq_len, d_k);
            Tensor K_h(seq_len, d_k);
            Tensor V_h(seq_len, d_k);
            for (size_t s = 0; s < seq_len; ++s) {
                size_t src_row = b * seq_len + s;
                for (size_t d = 0; d < d_k; ++d) {
                    Q_h(s, d) = cached_Q(src_row, h * d_k + d);
                    K_h(s, d) = cached_K(src_row, h * d_k + d);
                    V_h(s, d) = cached_V(src_row, h * d_k + d);
                }
            }

            // Attention scores: Q_h * K_h^T * scale -> seq_len x seq_len
            Tensor scores = Q_h.matmul(K_h.transpose()) * scale;

            // Softmax per row
            Tensor attn(seq_len, seq_len);
            for (size_t i = 0; i < seq_len; ++i) {
                double max_val = scores(i, 0);
                for (size_t j = 1; j < seq_len; ++j)
                    max_val = std::max(max_val, scores(i, j));
                double sum_exp = 0;
                for (size_t j = 0; j < seq_len; ++j) {
                    attn(i, j) = std::exp(scores(i, j) - max_val);
                    sum_exp += attn(i, j);
                }
                for (size_t j = 0; j < seq_len; ++j)
                    attn(i, j) /= sum_exp;
            }

            // Store attention weights for backward
            size_t attn_base = (b * num_heads + h) * seq_len;
            for (size_t i = 0; i < seq_len; ++i)
                for (size_t j = 0; j < seq_len; ++j)
                    cached_attn_weights(attn_base + i, j) = attn(i, j);

            // Weighted values: attn * V_h -> seq_len x d_k
            Tensor head_out = attn.matmul(V_h);

            // Store per-head output
            for (size_t s = 0; s < seq_len; ++s)
                for (size_t d = 0; d < d_k; ++d)
                    cached_attn_output((b * num_heads + h) * seq_len + s, d) = head_out(s, d);
        }
    }

    // Concatenate heads: (B*S) x d_model
    Tensor concat(batch * seq_len, d_model);
    VIT_OMP_PAR_FOR_COLLAPSE2
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t d = 0; d < d_k; ++d) {
                    concat(b * seq_len + s, h * d_k + d) =
                        cached_attn_output((b * num_heads + h) * seq_len + s, d);
                }
            }
        }
    }

    // Output projection
    return W_o.forward(concat);
}

Tensor ViTMultiHeadAttention::backward(const Tensor& grad_output, size_t seq_len) {
    size_t batch = cached_batch;
    double scale = 1.0 / std::sqrt((double)d_k);

    // Backward through output projection
    Tensor grad_concat = W_o.backward(grad_output);  // (B*S) x d_model

    // Split grad_concat into per-head gradients
    // grad_attn_output: (B*num_heads*S) x d_k
    Tensor grad_attn_output(batch * num_heads * seq_len, d_k);
    VIT_OMP_PAR_FOR_COLLAPSE2
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t h = 0; h < num_heads; ++h) {
                for (size_t d = 0; d < d_k; ++d) {
                    grad_attn_output((b * num_heads + h) * seq_len + s, d) =
                        grad_concat(b * seq_len + s, h * d_k + d);
                }
            }
        }
    }

    // Accumulate gradients for Q, K, V
    Tensor grad_Q = Tensor::zeros(batch * seq_len, d_model);
    Tensor grad_K = Tensor::zeros(batch * seq_len, d_model);
    Tensor grad_V = Tensor::zeros(batch * seq_len, d_model);

    VIT_OMP_PAR_FOR_COLLAPSE2
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            size_t attn_base = (b * num_heads + h) * seq_len;

            // Reconstruct per-head tensors
            Tensor Q_h(seq_len, d_k), K_h(seq_len, d_k), V_h(seq_len, d_k);
            Tensor attn_h(seq_len, seq_len);
            Tensor grad_out_h(seq_len, d_k);

            for (size_t s = 0; s < seq_len; ++s) {
                size_t src_row = b * seq_len + s;
                for (size_t d = 0; d < d_k; ++d) {
                    Q_h(s, d) = cached_Q(src_row, h * d_k + d);
                    K_h(s, d) = cached_K(src_row, h * d_k + d);
                    V_h(s, d) = cached_V(src_row, h * d_k + d);
                    grad_out_h(s, d) = grad_attn_output(attn_base + s, d);
                }
                for (size_t j = 0; j < seq_len; ++j)
                    attn_h(s, j) = cached_attn_weights(attn_base + s, j);
            }

            // grad_V_h = attn_h^T * grad_out_h  (seq_len x d_k)
            Tensor grad_V_h = attn_h.transpose().matmul(grad_out_h);

            // grad_attn = grad_out_h * V_h^T  (seq_len x seq_len)
            Tensor grad_attn = grad_out_h.matmul(V_h.transpose());

            // Backward through softmax
            // d_scores(i,j) = attn(i,j) * (grad_attn(i,j) - sum_k(grad_attn(i,k)*attn(i,k)))
            Tensor grad_scores(seq_len, seq_len);
            for (size_t i = 0; i < seq_len; ++i) {
                double dot = 0;
                for (size_t j = 0; j < seq_len; ++j)
                    dot += grad_attn(i, j) * attn_h(i, j);
                for (size_t j = 0; j < seq_len; ++j)
                    grad_scores(i, j) = attn_h(i, j) * (grad_attn(i, j) - dot) * scale;
            }

            // grad_Q_h = grad_scores * K_h  (seq_len x d_k)
            Tensor grad_Q_h = grad_scores.matmul(K_h);
            // grad_K_h = grad_scores^T * Q_h  (seq_len x d_k)
            Tensor grad_K_h = grad_scores.transpose().matmul(Q_h);

            // Scatter back into full gradient tensors
            for (size_t s = 0; s < seq_len; ++s) {
                size_t dst_row = b * seq_len + s;
                for (size_t d = 0; d < d_k; ++d) {
                    grad_Q(dst_row, h * d_k + d) += grad_Q_h(s, d);
                    grad_K(dst_row, h * d_k + d) += grad_K_h(s, d);
                    grad_V(dst_row, h * d_k + d) += grad_V_h(s, d);
                }
            }
        }
    }

    // Backward through Q, K, V projections and sum input gradients
    Tensor grad_input = W_q.backward(grad_Q);
    grad_input += W_k.backward(grad_K);
    grad_input += W_v.backward(grad_V);

    return grad_input;
}

std::vector<Tensor*> ViTMultiHeadAttention::parameters() {
    auto p1 = W_q.parameters(), p2 = W_k.parameters();
    auto p3 = W_v.parameters(), p4 = W_o.parameters();
    std::vector<Tensor*> all;
    all.insert(all.end(), p1.begin(), p1.end());
    all.insert(all.end(), p2.begin(), p2.end());
    all.insert(all.end(), p3.begin(), p3.end());
    all.insert(all.end(), p4.begin(), p4.end());
    return all;
}

std::vector<Tensor*> ViTMultiHeadAttention::gradients() {
    auto g1 = W_q.gradients(), g2 = W_k.gradients();
    auto g3 = W_v.gradients(), g4 = W_o.gradients();
    std::vector<Tensor*> all;
    all.insert(all.end(), g1.begin(), g1.end());
    all.insert(all.end(), g2.begin(), g2.end());
    all.insert(all.end(), g3.begin(), g3.end());
    all.insert(all.end(), g4.begin(), g4.end());
    return all;
}

void ViTMultiHeadAttention::zero_grad() {
    for (auto* g : gradients()) g->fill_zeros();
}

// ============================================================
// ViTFeedForward
// ============================================================
ViTFeedForward::ViTFeedForward(size_t d_model, size_t d_ff, std::mt19937& rng)
    : fc1(d_model, d_ff, "vit_ff1", rng),
      fc2(d_ff, d_model, "vit_ff2", rng) {}

Tensor ViTFeedForward::forward(const Tensor& input) {
    Tensor h = fc1.forward(input);
    h = gelu.forward(h);
    return fc2.forward(h);
}

Tensor ViTFeedForward::backward(const Tensor& grad_output) {
    Tensor g = fc2.backward(grad_output);
    g = gelu.backward(g);
    return fc1.backward(g);
}

std::vector<Tensor*> ViTFeedForward::parameters() {
    auto p1 = fc1.parameters(), p2 = fc2.parameters();
    std::vector<Tensor*> all;
    all.insert(all.end(), p1.begin(), p1.end());
    all.insert(all.end(), p2.begin(), p2.end());
    return all;
}

std::vector<Tensor*> ViTFeedForward::gradients() {
    auto g1 = fc1.gradients(), g2 = fc2.gradients();
    std::vector<Tensor*> all;
    all.insert(all.end(), g1.begin(), g1.end());
    all.insert(all.end(), g2.begin(), g2.end());
    return all;
}

void ViTFeedForward::zero_grad() {
    for (auto* g : gradients()) g->fill_zeros();
}

// ============================================================
// ViTEncoderBlock (Pre-Norm)
// ============================================================
ViTEncoderBlock::ViTEncoderBlock(size_t d_model, size_t num_heads, size_t d_ff,
                                  double dropout_rate, std::mt19937& rng)
    : mha(d_model, num_heads, rng),
      ffn(d_model, d_ff, rng),
      ln1(d_model),
      ln2(d_model),
      dropout1(dropout_rate, rng),
      dropout2(dropout_rate, rng),
      cached_seq_len(0) {}

Tensor ViTEncoderBlock::forward(const Tensor& input, size_t seq_len) {
    cached_seq_len = seq_len;
    cached_input = input;

    // Pre-norm: LN -> MHA -> dropout -> + residual
    cached_ln1_out = ln1.forward(input);
    Tensor mha_out = mha.forward(cached_ln1_out, seq_len);
    cached_mha_out = dropout1.forward(mha_out);
    cached_after_res1 = input + cached_mha_out;

    // Pre-norm: LN -> FFN -> dropout -> + residual
    cached_ln2_out = ln2.forward(cached_after_res1);
    Tensor ffn_out = ffn.forward(cached_ln2_out);
    Tensor ffn_drop = dropout2.forward(ffn_out);

    return cached_after_res1 + ffn_drop;
}

Tensor ViTEncoderBlock::backward(const Tensor& grad_output, size_t seq_len) {
    // Backward through second residual connection
    Tensor grad_res1 = grad_output; // gradient flows through residual
    Tensor grad_ffn_drop = grad_output;

    // Backward through dropout2 -> FFN -> LN2
    Tensor grad_ffn = dropout2.backward(grad_ffn_drop);
    grad_ffn = ffn.backward(grad_ffn);
    Tensor grad_ln2 = ln2.backward(grad_ffn);

    // Add residual gradient
    grad_res1 += grad_ln2;

    // Backward through first residual connection
    Tensor grad_input = grad_res1; // gradient flows through residual
    Tensor grad_mha_drop = grad_res1;

    // Backward through dropout1 -> MHA -> LN1
    Tensor grad_mha = dropout1.backward(grad_mha_drop);
    grad_mha = mha.backward(grad_mha, seq_len);
    Tensor grad_ln1 = ln1.backward(grad_mha);

    // Add residual gradient
    grad_input += grad_ln1;

    return grad_input;
}

std::vector<Tensor*> ViTEncoderBlock::parameters() {
    std::vector<Tensor*> all;
    auto p1 = mha.parameters();   all.insert(all.end(), p1.begin(), p1.end());
    auto p2 = ffn.parameters();   all.insert(all.end(), p2.begin(), p2.end());
    auto p3 = ln1.parameters();   all.insert(all.end(), p3.begin(), p3.end());
    auto p4 = ln2.parameters();   all.insert(all.end(), p4.begin(), p4.end());
    return all;
}

std::vector<Tensor*> ViTEncoderBlock::gradients() {
    std::vector<Tensor*> all;
    auto g1 = mha.gradients();    all.insert(all.end(), g1.begin(), g1.end());
    auto g2 = ffn.gradients();    all.insert(all.end(), g2.begin(), g2.end());
    auto g3 = ln1.gradients();    all.insert(all.end(), g3.begin(), g3.end());
    auto g4 = ln2.gradients();    all.insert(all.end(), g4.begin(), g4.end());
    return all;
}

void ViTEncoderBlock::zero_grad() {
    for (auto* g : gradients()) g->fill_zeros();
}

void ViTEncoderBlock::train() {
    ln1.train(); ln2.train();
    dropout1.train();
    dropout2.train();
}

void ViTEncoderBlock::eval() {
    ln1.eval(); ln2.eval();
    dropout1.eval();
    dropout2.eval();
}

// ============================================================
// PatchEmbedding
// ============================================================
PatchEmbedding::PatchEmbedding(size_t is, size_t ps, size_t ic, size_t ed,
                                std::mt19937& rng)
    : projection(ic * ps * ps, ed, "vit_patch_proj", rng),
      img_size(is), patch_size(ps), in_channels(ic), embed_dim(ed),
      cached_batch(0) {
    size_t patches_per_side = img_size / patch_size;
    num_patches = patches_per_side * patches_per_side;
    seq_len = num_patches + 1; // +1 for CLS token

    // Learnable positional embeddings
    pos_embedding = Tensor(seq_len, embed_dim);
    pos_embedding.fill_random_normal(0.0, 0.02, rng);
    grad_pos_embedding = Tensor::zeros(seq_len, embed_dim);

    // Learnable CLS token
    cls_token = Tensor(1, embed_dim);
    cls_token.fill_random_normal(0.0, 0.02, rng);
    grad_cls_token = Tensor::zeros(1, embed_dim);
}

Tensor PatchEmbedding::forward(const Tensor& input, size_t batch_size) {
    // input: batch x (C*H*W), e.g. batch x 784 for MNIST
    cached_batch = batch_size;
    size_t patch_dim = in_channels * patch_size * patch_size;
    size_t patches_per_side = img_size / patch_size;

    // Extract patches: batch*num_patches x patch_dim
    cached_patches = Tensor(batch_size * num_patches, patch_dim);
    VIT_OMP_PAR_FOR_COLLAPSE3
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t py = 0; py < patches_per_side; ++py) {
            for (size_t px = 0; px < patches_per_side; ++px) {
                size_t patch_idx = py * patches_per_side + px;
                size_t dst_row = b * num_patches + patch_idx;
                size_t col_idx = 0;
                for (size_t c = 0; c < in_channels; ++c) {
                    for (size_t i = 0; i < patch_size; ++i) {
                        for (size_t j = 0; j < patch_size; ++j) {
                            size_t row_in_img = py * patch_size + i;
                            size_t col_in_img = px * patch_size + j;
                            size_t src_idx = c * img_size * img_size + row_in_img * img_size + col_in_img;
                            cached_patches(dst_row, col_idx) = input(b, src_idx);
                            col_idx++;
                        }
                    }
                }
            }
        }
    }

    // Project patches through linear layer
    Tensor patch_embeds = projection.forward(cached_patches);
    // patch_embeds: (batch*num_patches) x embed_dim

    // Build output: prepend CLS token and add positional embeddings
    // Output: (batch * seq_len) x embed_dim
    Tensor output(batch_size * seq_len, embed_dim);
    VIT_OMP_PAR_FOR
    for (size_t b = 0; b < batch_size; ++b) {
        // CLS token at position 0
        for (size_t d = 0; d < embed_dim; ++d) {
            output(b * seq_len, d) = cls_token(0, d) + pos_embedding(0, d);
        }
        // Patch embeddings at positions 1..num_patches
        for (size_t p = 0; p < num_patches; ++p) {
            for (size_t d = 0; d < embed_dim; ++d) {
                output(b * seq_len + 1 + p, d) =
                    patch_embeds(b * num_patches + p, d) + pos_embedding(1 + p, d);
            }
        }
    }

    return output;
}

Tensor PatchEmbedding::backward(const Tensor& grad, size_t batch_size) {
    // grad: (batch * seq_len) x embed_dim

    // Accumulate gradient for positional embeddings (summed over batch)
    grad_pos_embedding.fill_zeros();
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            for (size_t d = 0; d < embed_dim; ++d) {
                grad_pos_embedding(s, d) += grad(b * seq_len + s, d);
            }
        }
    }

    // Accumulate gradient for CLS token (position 0 in each batch)
    grad_cls_token.fill_zeros();
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t d = 0; d < embed_dim; ++d) {
            grad_cls_token(0, d) += grad(b * seq_len, d);
        }
    }

    // Extract gradient for patch embeddings (positions 1..num_patches)
    Tensor grad_patch_embeds(batch_size * num_patches, embed_dim);
    VIT_OMP_PAR_FOR_COLLAPSE2
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t p = 0; p < num_patches; ++p) {
            for (size_t d = 0; d < embed_dim; ++d) {
                grad_patch_embeds(b * num_patches + p, d) =
                    grad(b * seq_len + 1 + p, d);
            }
        }
    }

    // Backward through projection
    Tensor grad_patches = projection.backward(grad_patch_embeds);
    // grad_patches: (batch*num_patches) x patch_dim

    // Scatter back to image space
    size_t patches_per_side = img_size / patch_size;
    Tensor grad_input(batch_size, in_channels * img_size * img_size);

    VIT_OMP_PAR_FOR_COLLAPSE3
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t py = 0; py < patches_per_side; ++py) {
            for (size_t px = 0; px < patches_per_side; ++px) {
                size_t patch_idx = py * patches_per_side + px;
                size_t src_row = b * num_patches + patch_idx;
                size_t col_idx = 0;
                for (size_t c = 0; c < in_channels; ++c) {
                    for (size_t i = 0; i < patch_size; ++i) {
                        for (size_t j = 0; j < patch_size; ++j) {
                            size_t row_in_img = py * patch_size + i;
                            size_t col_in_img = px * patch_size + j;
                            size_t dst_idx = c * img_size * img_size + row_in_img * img_size + col_in_img;
                            grad_input(b, dst_idx) = grad_patches(src_row, col_idx);
                            col_idx++;
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

std::vector<Tensor*> PatchEmbedding::parameters() {
    auto p = projection.parameters();
    p.push_back(&pos_embedding);
    p.push_back(&cls_token);
    return p;
}

std::vector<Tensor*> PatchEmbedding::gradients() {
    auto g = projection.gradients();
    g.push_back(&grad_pos_embedding);
    g.push_back(&grad_cls_token);
    return g;
}

void PatchEmbedding::zero_grad() {
    for (auto* g : gradients()) g->fill_zeros();
}

// ============================================================
// ViTClassifier
// ============================================================
ViTClassifier::ViTClassifier(size_t is, size_t ps, size_t ic, size_t nc,
                              size_t ed, size_t nh, size_t df, size_t nb,
                              double dr, std::mt19937& rng)
    : patch_embed(is, ps, ic, ed, rng),
      final_ln(ed),
      classifier(ed, nc, "vit_classifier", rng),
      img_size(is), patch_size(ps), in_channels(ic),
      embed_dim(ed), num_heads(nh), d_ff(df), num_blocks(nb),
      num_classes(nc), dropout_rate(dr),
      cached_batch(0), cached_seq_len(0) {
    for (size_t i = 0; i < nb; ++i) {
        blocks.emplace_back(ed, nh, df, dr, rng);
    }
    cached_seq_len = patch_embed.seq_len;
}

ViTClassifier::ViTClassifier(std::mt19937& rng)
    : ViTClassifier(28, 7, 1, 10, 64, 4, 128, 2, 0.1, rng) {}

Tensor ViTClassifier::forward(const Tensor& input) {
    // input: batch x 784
    size_t batch = input.rows;
    cached_batch = batch;
    cached_seq_len = patch_embed.seq_len;

    // Patch embedding: (batch * seq_len) x embed_dim
    Tensor x = patch_embed.forward(input, batch);

    // Transformer encoder blocks
    for (size_t i = 0; i < blocks.size(); ++i) {
        x = blocks[i].forward(x, cached_seq_len);
    }
    cached_encoder_out = x;

    // Extract CLS token (position 0 in each batch element's sequence)
    cached_cls_tokens = Tensor(batch, embed_dim);
    VIT_OMP_PAR_FOR
    for (size_t b = 0; b < batch; ++b) {
        for (size_t d = 0; d < embed_dim; ++d) {
            cached_cls_tokens(b, d) = x(b * cached_seq_len, d);
        }
    }

    // Final layer norm on CLS tokens
    cached_ln_out = final_ln.forward(cached_cls_tokens);

    // Classification head
    return classifier.forward(cached_ln_out);
}

Tensor ViTClassifier::backward(const Tensor& grad_output) {
    // grad_output: batch x num_classes
    size_t batch = cached_batch;

    // Backward through classifier
    Tensor grad_ln = classifier.backward(grad_output);

    // Backward through final layer norm
    Tensor grad_cls = final_ln.backward(grad_ln);

    // Scatter CLS gradient back into full sequence gradient
    Tensor grad_encoder(batch * cached_seq_len, embed_dim);
    VIT_OMP_PAR_FOR
    for (size_t b = 0; b < batch; ++b) {
        for (size_t d = 0; d < embed_dim; ++d) {
            grad_encoder(b * cached_seq_len, d) = grad_cls(b, d);
        }
        // Other positions get zero gradient (already initialized to 0)
    }

    // Backward through encoder blocks in reverse
    for (int i = (int)blocks.size() - 1; i >= 0; --i) {
        grad_encoder = blocks[i].backward(grad_encoder, cached_seq_len);
    }

    // Backward through patch embedding
    return patch_embed.backward(grad_encoder, batch);
}

std::vector<Tensor*> ViTClassifier::parameters() {
    std::vector<Tensor*> all;
    auto p0 = patch_embed.parameters();   all.insert(all.end(), p0.begin(), p0.end());
    for (auto& block : blocks) {
        auto pb = block.parameters();     all.insert(all.end(), pb.begin(), pb.end());
    }
    auto pln = final_ln.parameters();     all.insert(all.end(), pln.begin(), pln.end());
    auto pc = classifier.parameters();    all.insert(all.end(), pc.begin(), pc.end());
    return all;
}

std::vector<Tensor*> ViTClassifier::gradients() {
    std::vector<Tensor*> all;
    auto g0 = patch_embed.gradients();    all.insert(all.end(), g0.begin(), g0.end());
    for (auto& block : blocks) {
        auto gb = block.gradients();      all.insert(all.end(), gb.begin(), gb.end());
    }
    auto gln = final_ln.gradients();      all.insert(all.end(), gln.begin(), gln.end());
    auto gc = classifier.gradients();     all.insert(all.end(), gc.begin(), gc.end());
    return all;
}

void ViTClassifier::zero_grad() {
    for (auto* g : gradients()) g->fill_zeros();
}

void ViTClassifier::train() {
    for (auto& block : blocks) block.train();
    final_ln.train();
}

void ViTClassifier::eval() {
    for (auto& block : blocks) block.eval();
    final_ln.eval();
}
