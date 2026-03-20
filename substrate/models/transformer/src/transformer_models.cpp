#include "transformer_models.h"
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#define TRANS_OMP_PAR_FOR _Pragma("omp parallel for schedule(static)")
#define TRANS_OMP_PAR_FOR_COLLAPSE2 _Pragma("omp parallel for collapse(2) schedule(static)")
#define TRANS_OMP_PAR_FOR_COLLAPSE3 _Pragma("omp parallel for collapse(3) schedule(static)")
#else
#define TRANS_OMP_PAR_FOR
#define TRANS_OMP_PAR_FOR_COLLAPSE2
#define TRANS_OMP_PAR_FOR_COLLAPSE3
#endif

// ============================================================
// MultiHeadAttention
// ============================================================

MultiHeadAttention::MultiHeadAttention(size_t dm, size_t nh, std::mt19937& rng)
    : W_q(dm, dm, "mha_Wq", rng),
      W_k(dm, dm, "mha_Wk", rng),
      W_v(dm, dm, "mha_Wv", rng),
      W_o(dm, dm, "mha_Wo", rng),
      d_model(dm), num_heads(nh), d_k(nh > 0 ? dm / nh : 0),
      cached_seq_len(0), cached_batch(0), cached_causal(false) {
    if (nh == 0 || dm % nh != 0) {
        throw std::runtime_error("MultiHeadAttention: d_model must be divisible by num_heads and num_heads > 0");
    }

    // Use Xavier initialization for attention projections (more stable than He)
    double xavier_std = std::sqrt(2.0 / (double)(dm + dm));
    W_q.weights.fill_random_normal(0, xavier_std, rng);
    W_k.weights.fill_random_normal(0, xavier_std, rng);
    W_v.weights.fill_random_normal(0, xavier_std, rng);
    W_o.weights.fill_random_normal(0, xavier_std, rng);

    W_q.biases.fill_zeros();
    W_k.biases.fill_zeros();
    W_v.biases.fill_zeros();
    W_o.biases.fill_zeros();
}

Tensor MultiHeadAttention::forward(const Tensor& input, size_t seq_len, bool causal_mask) {
    // input: (batch*seq_len) x d_model
    if (seq_len == 0) {
        throw std::runtime_error("MultiHeadAttention::forward seq_len must be > 0");
    }
    size_t total_tokens = input.rows;
    if (total_tokens % seq_len != 0) {
        throw std::runtime_error("MultiHeadAttention::forward input rows must be divisible by seq_len");
    }
    size_t batch = total_tokens / seq_len;

    cached_input = input;
    cached_seq_len = seq_len;
    cached_batch = batch;
    cached_causal = causal_mask;

    // Project Q, K, V: each (batch*seq_len) x d_model
    cached_Q = W_q.forward(input);
    cached_K = W_k.forward(input);
    cached_V = W_v.forward(input);

    // Compute multi-head attention per batch element and per head.
    // Q, K, V are (batch*seq_len) x d_model.
    // We split d_model into num_heads * d_k and compute attention for each head.
    //
    // Output: (batch*seq_len) x d_model (concatenated heads)
    //
    // We also store attention weights for interpretability:
    //   cached_attn_weights: (batch * num_heads * seq_len) x seq_len
    //   cached_attn_output:  (batch * num_heads * seq_len) x d_k

    cached_attn_weights = Tensor(batch * num_heads * seq_len, seq_len);
    cached_attn_output = Tensor(batch * num_heads * seq_len, d_k);

    double scale = 1.0 / std::sqrt((double)d_k);

    // Concatenated multi-head output before W_o projection
    Tensor concat_out(batch * seq_len, d_model);

    TRANS_OMP_PAR_FOR_COLLAPSE2
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            size_t head_offset = h * d_k;

            // Extract Q_h, K_h, V_h for this batch and head
            // Q_h: seq_len x d_k
            Tensor Q_h(seq_len, d_k);
            Tensor K_h(seq_len, d_k);
            Tensor V_h(seq_len, d_k);

            for (size_t s = 0; s < seq_len; ++s) {
                size_t row = b * seq_len + s;
                for (size_t d = 0; d < d_k; ++d) {
                    Q_h(s, d) = cached_Q(row, head_offset + d);
                    K_h(s, d) = cached_K(row, head_offset + d);
                    V_h(s, d) = cached_V(row, head_offset + d);
                }
            }

            // Attention scores: Q_h * K_h^T * scale -> seq_len x seq_len
            Tensor scores = Q_h.matmul(K_h.transpose()) * scale;

            // Apply causal mask if requested
            if (causal_mask) {
                for (size_t i = 0; i < seq_len; ++i)
                    for (size_t j = i + 1; j < seq_len; ++j)
                        scores(i, j) = -1e9;
            }

            // Softmax over last dimension (each row)
            Tensor attn(seq_len, seq_len);
            for (size_t i = 0; i < seq_len; ++i) {
                double max_val = scores(i, 0);
                for (size_t j = 1; j < seq_len; ++j)
                    max_val = std::max(max_val, scores(i, j));
                double sum_exp = 0.0;
                for (size_t j = 0; j < seq_len; ++j) {
                    attn(i, j) = std::exp(scores(i, j) - max_val);
                    sum_exp += attn(i, j);
                }
                for (size_t j = 0; j < seq_len; ++j)
                    attn(i, j) /= sum_exp;
            }

            // Store attention weights
            size_t attn_base = (b * num_heads + h) * seq_len;
            for (size_t i = 0; i < seq_len; ++i)
                for (size_t j = 0; j < seq_len; ++j)
                    cached_attn_weights(attn_base + i, j) = attn(i, j);

            // Attention output: attn * V_h -> seq_len x d_k
            Tensor head_out = attn.matmul(V_h);

            // Store per-head output for backward
            for (size_t i = 0; i < seq_len; ++i)
                for (size_t d = 0; d < d_k; ++d)
                    cached_attn_output(attn_base + i, d) = head_out(i, d);

            // Write into concatenated output
            for (size_t s = 0; s < seq_len; ++s) {
                size_t row = b * seq_len + s;
                for (size_t d = 0; d < d_k; ++d)
                    concat_out(row, head_offset + d) = head_out(s, d);
            }
        }
    }

    // Output projection: (batch*seq_len) x d_model -> (batch*seq_len) x d_model
    return W_o.forward(concat_out);
}

Tensor MultiHeadAttention::backward(const Tensor& grad_output, size_t seq_len) {
    size_t batch = cached_batch;
    double scale = 1.0 / std::sqrt((double)d_k);

    // Backprop through output projection W_o
    // W_o.forward was called on concat_out, so W_o.backward gives grad_concat_out
    Tensor grad_concat = W_o.backward(grad_output);

    // Prepare gradient accumulators for Q, K, V projections
    Tensor grad_Q(batch * seq_len, d_model);
    Tensor grad_K(batch * seq_len, d_model);
    Tensor grad_V(batch * seq_len, d_model);

    TRANS_OMP_PAR_FOR_COLLAPSE2
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            size_t head_offset = h * d_k;
            size_t attn_base = (b * num_heads + h) * seq_len;

            // Extract per-head grad_concat slice: seq_len x d_k
            Tensor grad_head(seq_len, d_k);
            for (size_t s = 0; s < seq_len; ++s) {
                size_t row = b * seq_len + s;
                for (size_t d = 0; d < d_k; ++d)
                    grad_head(s, d) = grad_concat(row, head_offset + d);
            }

            // Recover attention weights and V_h for this head
            Tensor attn(seq_len, seq_len);
            for (size_t i = 0; i < seq_len; ++i)
                for (size_t j = 0; j < seq_len; ++j)
                    attn(i, j) = cached_attn_weights(attn_base + i, j);

            Tensor V_h(seq_len, d_k);
            for (size_t s = 0; s < seq_len; ++s) {
                size_t row = b * seq_len + s;
                for (size_t d = 0; d < d_k; ++d)
                    V_h(s, d) = cached_V(row, head_offset + d);
            }

            // head_out = attn * V_h  (seq_len x d_k)
            // grad_head is dL/d(head_out): seq_len x d_k

            // dL/d(attn) = grad_head * V_h^T : seq_len x seq_len
            Tensor grad_attn = grad_head.matmul(V_h.transpose());

            // dL/d(V_h) = attn^T * grad_head : seq_len x d_k
            Tensor grad_V_h = attn.transpose().matmul(grad_head);

            // Backprop through softmax:
            // attn_i = softmax(scores_i)
            // d(scores_i,j) = attn_i,j * (grad_attn_i,j - sum_k(grad_attn_i,k * attn_i,k))
            Tensor grad_scores(seq_len, seq_len);
            for (size_t i = 0; i < seq_len; ++i) {
                double dot = 0.0;
                for (size_t j = 0; j < seq_len; ++j)
                    dot += grad_attn(i, j) * attn(i, j);
                for (size_t j = 0; j < seq_len; ++j)
                    grad_scores(i, j) = attn(i, j) * (grad_attn(i, j) - dot);
            }

            // Causal mask: zero out gradients for masked positions
            if (cached_causal) {
                for (size_t i = 0; i < seq_len; ++i)
                    for (size_t j = i + 1; j < seq_len; ++j)
                        grad_scores(i, j) = 0.0;
            }

            // scores = Q_h * K_h^T * scale
            // grad_scores is dL/d(scores): seq_len x seq_len
            // After accounting for scale:
            Tensor grad_scores_scaled = grad_scores * scale;

            // Recover Q_h and K_h
            Tensor Q_h(seq_len, d_k);
            Tensor K_h(seq_len, d_k);
            for (size_t s = 0; s < seq_len; ++s) {
                size_t row = b * seq_len + s;
                for (size_t d = 0; d < d_k; ++d) {
                    Q_h(s, d) = cached_Q(row, head_offset + d);
                    K_h(s, d) = cached_K(row, head_offset + d);
                }
            }

            // dL/d(Q_h) = grad_scores_scaled * K_h : seq_len x d_k
            Tensor grad_Q_h = grad_scores_scaled.matmul(K_h);

            // dL/d(K_h) = grad_scores_scaled^T * Q_h : seq_len x d_k
            Tensor grad_K_h = grad_scores_scaled.transpose().matmul(Q_h);

            // Scatter head gradients back into full d_model gradients
            for (size_t s = 0; s < seq_len; ++s) {
                size_t row = b * seq_len + s;
                for (size_t d = 0; d < d_k; ++d) {
                    grad_Q(row, head_offset + d) = grad_Q_h(s, d);
                    grad_K(row, head_offset + d) = grad_K_h(s, d);
                    grad_V(row, head_offset + d) = grad_V_h(s, d);
                }
            }
        }
    }

    // Backprop through Q, K, V linear projections
    // Each was: cached_Q = W_q.forward(input), etc.
    // We need to sum gradients flowing back to input from all three.
    Tensor grad_input_q = W_q.backward(grad_Q);
    Tensor grad_input_k = W_k.backward(grad_K);
    Tensor grad_input_v = W_v.backward(grad_V);

    return grad_input_q + grad_input_k + grad_input_v;
}

std::vector<Tensor*> MultiHeadAttention::parameters() {
    std::vector<Tensor*> params;
    for (auto* p : W_q.parameters()) params.push_back(p);
    for (auto* p : W_k.parameters()) params.push_back(p);
    for (auto* p : W_v.parameters()) params.push_back(p);
    for (auto* p : W_o.parameters()) params.push_back(p);
    return params;
}

std::vector<Tensor*> MultiHeadAttention::gradients() {
    std::vector<Tensor*> grads;
    for (auto* g : W_q.gradients()) grads.push_back(g);
    for (auto* g : W_k.gradients()) grads.push_back(g);
    for (auto* g : W_v.gradients()) grads.push_back(g);
    for (auto* g : W_o.gradients()) grads.push_back(g);
    return grads;
}

void MultiHeadAttention::zero_grad() {
    W_q.grad_weights.fill_zeros(); W_q.grad_biases.fill_zeros();
    W_k.grad_weights.fill_zeros(); W_k.grad_biases.fill_zeros();
    W_v.grad_weights.fill_zeros(); W_v.grad_biases.fill_zeros();
    W_o.grad_weights.fill_zeros(); W_o.grad_biases.fill_zeros();
}

// ============================================================
// FeedForward
// ============================================================

FeedForward::FeedForward(size_t d_model, size_t d_ff, std::mt19937& rng)
    : fc1(d_model, d_ff, "ffn_fc1", rng),
      fc2(d_ff, d_model, "ffn_fc2", rng) {}

Tensor FeedForward::forward(const Tensor& input) {
    Tensor h = fc1.forward(input);
    h = gelu.forward(h);
    return fc2.forward(h);
}

Tensor FeedForward::backward(const Tensor& grad_output) {
    Tensor grad = fc2.backward(grad_output);
    grad = gelu.backward(grad);
    grad = fc1.backward(grad);
    return grad;
}

std::vector<Tensor*> FeedForward::parameters() {
    return {&fc1.weights, &fc1.biases, &fc2.weights, &fc2.biases};
}

std::vector<Tensor*> FeedForward::gradients() {
    return {&fc1.grad_weights, &fc1.grad_biases, &fc2.grad_weights, &fc2.grad_biases};
}

void FeedForward::zero_grad() {
    fc1.grad_weights.fill_zeros(); fc1.grad_biases.fill_zeros();
    fc2.grad_weights.fill_zeros(); fc2.grad_biases.fill_zeros();
}

// ============================================================
// TransformerEncoderBlock (Post-Norm)
// ============================================================
// Forward:
//   residual = x
//   x = MHA(x) + residual
//   x = LayerNorm1(x)
//   residual = x
//   x = FFN(x) + residual
//   x = LayerNorm2(x)
// ============================================================

TransformerEncoderBlock::TransformerEncoderBlock(size_t d_model, size_t num_heads,
                                                 size_t d_ff, double dropout_rate,
                                                 std::mt19937& rng)
    : mha(d_model, num_heads, rng),
      ffn(d_model, d_ff, rng),
      ln1(d_model),
      ln2(d_model),
      dropout1(dropout_rate, rng),
      dropout2(dropout_rate, rng),
      cached_seq_len(0) {}

Tensor TransformerEncoderBlock::forward(const Tensor& input, size_t seq_len) {
    cached_seq_len = seq_len;
    cached_mha_input = input;

    // Self-attention sublayer
    Tensor attn_out = mha.forward(input, seq_len, false);
    attn_out = dropout1.forward(attn_out);
    cached_mha_out = attn_out;

    // Residual + LayerNorm
    Tensor x = attn_out + input;
    x = ln1.forward(x);
    cached_ln1_out = x;

    // Feed-forward sublayer
    Tensor ffn_out = ffn.forward(x);
    ffn_out = dropout2.forward(ffn_out);
    cached_ffn_out = ffn_out;

    // Residual + LayerNorm
    x = ffn_out + cached_ln1_out;
    x = ln2.forward(x);

    return x;
}

Tensor TransformerEncoderBlock::backward(const Tensor& grad_output, size_t seq_len) {
    // Backward through ln2
    Tensor grad = ln2.backward(grad_output);

    // Backward through second residual: grad flows to both ffn_out and ln1_out
    Tensor grad_ffn_out = grad;
    Tensor grad_ln1_residual = grad;

    // Backward through dropout2
    grad_ffn_out = dropout2.backward(grad_ffn_out);

    // Backward through FFN
    Tensor grad_ln1_from_ffn = ffn.backward(grad_ffn_out);

    // Total gradient at ln1 output
    Tensor grad_ln1 = grad_ln1_from_ffn + grad_ln1_residual;

    // Backward through ln1
    grad = ln1.backward(grad_ln1);

    // Backward through first residual: grad flows to both mha_out and input
    Tensor grad_mha_out = grad;
    Tensor grad_input_residual = grad;

    // Backward through dropout1
    grad_mha_out = dropout1.backward(grad_mha_out);

    // Backward through MHA
    Tensor grad_input_from_mha = mha.backward(grad_mha_out, seq_len);

    // Total gradient at input
    return grad_input_from_mha + grad_input_residual;
}

std::vector<Tensor*> TransformerEncoderBlock::parameters() {
    std::vector<Tensor*> params;
    for (auto* p : mha.parameters()) params.push_back(p);
    for (auto* p : ffn.parameters()) params.push_back(p);
    for (auto* p : ln1.parameters()) params.push_back(p);
    for (auto* p : ln2.parameters()) params.push_back(p);
    return params;
}

std::vector<Tensor*> TransformerEncoderBlock::gradients() {
    std::vector<Tensor*> grads;
    for (auto* g : mha.gradients()) grads.push_back(g);
    for (auto* g : ffn.gradients()) grads.push_back(g);
    for (auto* g : ln1.gradients()) grads.push_back(g);
    for (auto* g : ln2.gradients()) grads.push_back(g);
    return grads;
}

void TransformerEncoderBlock::zero_grad() {
    mha.zero_grad();
    ffn.zero_grad();
    ln1.grad_gamma.fill_zeros(); ln1.grad_beta.fill_zeros();
    ln2.grad_gamma.fill_zeros(); ln2.grad_beta.fill_zeros();
}

void TransformerEncoderBlock::train() {
    dropout1.train();
    dropout2.train();
}

void TransformerEncoderBlock::eval() {
    dropout1.eval();
    dropout2.eval();
}

// ============================================================
// TransformerClassifier
// ============================================================

TransformerClassifier::TransformerClassifier(size_t input_dim, size_t seq_len,
                                             size_t num_classes, size_t d_model,
                                             size_t num_heads, size_t d_ff,
                                             size_t num_blocks, double dropout_rate,
                                             std::mt19937& rng)
    : input_proj(input_dim, d_model, "input_proj", rng),
      classifier(d_model, num_classes, "classifier", rng),
      d_model(d_model), num_heads(num_heads), d_ff(d_ff),
      num_blocks(num_blocks), input_dim(input_dim),
      seq_len(seq_len), num_classes(num_classes),
      dropout_rate(dropout_rate), cached_batch(0) {

    // Initialize input projection with Xavier
    double xavier_std = std::sqrt(2.0 / (double)(input_dim + d_model));
    input_proj.weights.fill_random_normal(0, xavier_std, rng);
    input_proj.biases.fill_zeros();

    // Initialize classifier head
    double cls_std = std::sqrt(2.0 / (double)(d_model + num_classes));
    classifier.weights.fill_random_normal(0, cls_std, rng);
    classifier.biases.fill_zeros();

    // Build encoder blocks
    blocks.reserve(num_blocks);
    for (size_t i = 0; i < num_blocks; ++i)
        blocks.emplace_back(d_model, num_heads, d_ff, dropout_rate, rng);

    // Build sinusoidal positional encoding
    build_positional_encoding();
}

TransformerClassifier::TransformerClassifier(std::mt19937& rng)
    : TransformerClassifier(28, 28, 10, 64, 4, 128, 2, 0.1, rng) {}

void TransformerClassifier::build_positional_encoding() {
    // Sinusoidal positional encoding as in "Attention Is All You Need"
    // PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    // PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    pos_encoding = Tensor(seq_len, d_model);
    TRANS_OMP_PAR_FOR
    for (size_t pos = 0; pos < seq_len; ++pos) {
        for (size_t i = 0; i < d_model; ++i) {
            double angle = (double)pos / std::pow(10000.0, (double)(i / 2 * 2) / (double)d_model);
            if (i % 2 == 0)
                pos_encoding(pos, i) = std::sin(angle);
            else
                pos_encoding(pos, i) = std::cos(angle);
        }
    }
}

Tensor TransformerClassifier::forward(const Tensor& input) {
    // input: batch x 784 (28*28 flattened MNIST image)
    size_t batch = input.rows;
    cached_batch = batch;
    cached_input = input;

    // Reshape to (batch * seq_len) x input_dim
    // Each row of the image is one timestep: 28 rows of 28 pixels
    Tensor reshaped(batch * seq_len, input_dim);
    TRANS_OMP_PAR_FOR_COLLAPSE2
    for (size_t b = 0; b < batch; ++b)
        for (size_t s = 0; s < seq_len; ++s)
            for (size_t d = 0; d < input_dim; ++d)
                reshaped(b * seq_len + s, d) = input(b, s * input_dim + d);

    // Project each 28-dim row into d_model dimensions
    // (batch*seq_len) x input_dim -> (batch*seq_len) x d_model
    Tensor projected = input_proj.forward(reshaped);

    // Add positional encoding (broadcast across batch)
    TRANS_OMP_PAR_FOR_COLLAPSE2
    for (size_t b = 0; b < batch; ++b)
        for (size_t s = 0; s < seq_len; ++s)
            for (size_t d = 0; d < d_model; ++d)
                projected(b * seq_len + s, d) += pos_encoding(s, d);

    cached_proj_out = projected;

    // Pass through encoder blocks
    Tensor x = projected;
    for (size_t i = 0; i < num_blocks; ++i)
        x = blocks[i].forward(x, seq_len);

    // Mean pooling over the sequence dimension
    // x: (batch*seq_len) x d_model -> batch x d_model
    cached_mean_pooled = Tensor(batch, d_model);
    TRANS_OMP_PAR_FOR_COLLAPSE2
    for (size_t b = 0; b < batch; ++b) {
        for (size_t d = 0; d < d_model; ++d) {
            double sum = 0.0;
            for (size_t s = 0; s < seq_len; ++s)
                sum += x(b * seq_len + s, d);
            cached_mean_pooled(b, d) = sum / (double)seq_len;
        }
    }

    // Classify: batch x d_model -> batch x num_classes
    return classifier.forward(cached_mean_pooled);
}

Tensor TransformerClassifier::backward(const Tensor& grad_output) {
    // grad_output: batch x num_classes
    size_t batch = cached_batch;

    // Backward through classifier
    Tensor grad_pooled = classifier.backward(grad_output);
    // grad_pooled: batch x d_model

    // Backward through mean pooling:
    // forward was: pooled(b,d) = (1/seq_len) * sum_s x(b*seq_len+s, d)
    // So grad flows equally to each sequence position, divided by seq_len
    Tensor grad_encoder_out(batch * seq_len, d_model);
    TRANS_OMP_PAR_FOR_COLLAPSE2
    for (size_t b = 0; b < batch; ++b)
        for (size_t s = 0; s < seq_len; ++s)
            for (size_t d = 0; d < d_model; ++d)
                grad_encoder_out(b * seq_len + s, d) = grad_pooled(b, d) / (double)seq_len;

    // Backward through encoder blocks (reverse order)
    Tensor grad = grad_encoder_out;
    for (int i = (int)num_blocks - 1; i >= 0; --i)
        grad = blocks[i].backward(grad, seq_len);

    // grad is now dL/d(projected): (batch*seq_len) x d_model
    // Positional encoding is additive constant, no gradient to it.

    // Backward through input projection
    Tensor grad_reshaped = input_proj.backward(grad);
    // grad_reshaped: (batch*seq_len) x input_dim

    // We don't need to propagate gradient to input pixels, but return it
    // for completeness (e.g., for gradient-based interpretability)
    Tensor grad_input(batch, input_dim * seq_len);
    TRANS_OMP_PAR_FOR_COLLAPSE2
    for (size_t b = 0; b < batch; ++b)
        for (size_t s = 0; s < seq_len; ++s)
            for (size_t d = 0; d < input_dim; ++d)
                grad_input(b, s * input_dim + d) = grad_reshaped(b * seq_len + s, d);

    return grad_input;
}

std::vector<Tensor*> TransformerClassifier::parameters() {
    std::vector<Tensor*> params;
    // Input projection
    for (auto* p : input_proj.parameters()) params.push_back(p);
    // Encoder blocks
    for (auto& block : blocks)
        for (auto* p : block.parameters()) params.push_back(p);
    // Classifier head
    for (auto* p : classifier.parameters()) params.push_back(p);
    return params;
}

std::vector<Tensor*> TransformerClassifier::gradients() {
    std::vector<Tensor*> grads;
    for (auto* g : input_proj.gradients()) grads.push_back(g);
    for (auto& block : blocks)
        for (auto* g : block.gradients()) grads.push_back(g);
    for (auto* g : classifier.gradients()) grads.push_back(g);
    return grads;
}

void TransformerClassifier::zero_grad() {
    input_proj.grad_weights.fill_zeros();
    input_proj.grad_biases.fill_zeros();
    for (auto& block : blocks)
        block.zero_grad();
    classifier.grad_weights.fill_zeros();
    classifier.grad_biases.fill_zeros();
}

void TransformerClassifier::train() {
    for (auto& block : blocks) block.train();
}

void TransformerClassifier::eval() {
    for (auto& block : blocks) block.eval();
}
