#pragma once
#include "../../../core/nn/module.h"
#include "../../../core/losses/losses.h"
#include "../../../core/optim/optimizer.h"

// ============================================================
// Vanilla RNN Cell
// ============================================================
class RNNCell {
public:
    Tensor W_ih, W_hh, b_h;       // input-to-hidden, hidden-to-hidden, bias
    Tensor grad_W_ih, grad_W_hh, grad_b_h;
    size_t input_size, hidden_size;

    // Caches for backward
    std::vector<Tensor> cached_inputs, cached_hiddens, cached_pre_act, cached_tanh_out;

    RNNCell(size_t input_size, size_t hidden_size, std::mt19937& rng);
    void zero_grad();
    Tensor forward_step(const Tensor& input, const Tensor& h_prev);
    // Returns grad_input, grad_h_prev for one timestep
    std::pair<Tensor, Tensor> backward_step(const Tensor& grad_h, int t);
};

// ============================================================
// GRU Cell
// ============================================================
class GRUCell {
public:
    // Gates: z=update, r=reset
    Tensor W_iz, W_hz, b_z;  // update gate
    Tensor W_ir, W_hr, b_r;  // reset gate
    Tensor W_in, W_hn, b_n;  // new gate
    Tensor grad_W_iz, grad_W_hz, grad_b_z;
    Tensor grad_W_ir, grad_W_hr, grad_b_r;
    Tensor grad_W_in, grad_W_hn, grad_b_n;
    size_t input_size, hidden_size;

    struct GRUCache {
        Tensor input, h_prev, z, r, n_tilde, h_new;
    };
    std::vector<GRUCache> caches;

    GRUCell(size_t input_size, size_t hidden_size, std::mt19937& rng);
    void zero_grad();
    Tensor forward_step(const Tensor& input, const Tensor& h_prev);
    std::pair<Tensor, Tensor> backward_step(const Tensor& grad_h, int t);
};

// ============================================================
// RNN Classifier for MNIST (pixel-by-pixel or row-by-row)
// ============================================================
class RNNClassifier {
public:
    RNNCell rnn;
    Linear fc;
    size_t input_size, hidden_size, num_classes, seq_len;
    bool bidirectional;
    std::unique_ptr<RNNCell> rnn_back; // for bidirectional

    RNNClassifier(size_t input_size, size_t hidden_size, size_t num_classes,
                  size_t seq_len, bool bidirectional, std::mt19937& rng);

    // input: batch x (seq_len * input_size), reshaped internally
    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();
    void clip_gradients(double max_norm);

    void train_mode() {}
    void eval_mode() {}
};

// ============================================================
// GRU Classifier for MNIST
// ============================================================
class GRUClassifier {
public:
    GRUCell gru;
    Linear fc;
    size_t input_size, hidden_size, num_classes, seq_len;

    GRUClassifier(size_t input_size, size_t hidden_size, size_t num_classes,
                  size_t seq_len, std::mt19937& rng);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();
    void clip_gradients(double max_norm);
};
