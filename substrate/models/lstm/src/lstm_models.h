#pragma once
#include "../../../core/nn/module.h"
#include "../../../core/losses/losses.h"

// ============================================================
// LSTM Cell
// ============================================================
class LSTMCell {
public:
    // Four gates: input(i), forget(f), cell(g), output(o)
    Tensor W_ii, W_hi, b_i;
    Tensor W_if, W_hf, b_f;
    Tensor W_ig, W_hg, b_g;
    Tensor W_io, W_ho, b_o;
    // Gradients
    Tensor grad_W_ii, grad_W_hi, grad_b_i;
    Tensor grad_W_if, grad_W_hf, grad_b_f;
    Tensor grad_W_ig, grad_W_hg, grad_b_g;
    Tensor grad_W_io, grad_W_ho, grad_b_o;

    size_t input_size, hidden_size;

    struct LSTMCache {
        Tensor input, h_prev, c_prev;
        Tensor i_gate, f_gate, g_gate, o_gate;
        Tensor c_new, h_new, tanh_c;
    };
    std::vector<LSTMCache> caches;

    LSTMCell(size_t input_size, size_t hidden_size, std::mt19937& rng);
    void zero_grad();
    std::pair<Tensor, Tensor> forward_step(const Tensor& input, const Tensor& h_prev, const Tensor& c_prev);
    // Returns grad_input, grad_h_prev, grad_c_prev
    std::tuple<Tensor, Tensor, Tensor> backward_step(const Tensor& grad_h, const Tensor& grad_c, int t);
};

// ============================================================
// LSTM Classifier for MNIST
// ============================================================
class LSTMClassifier {
public:
    LSTMCell lstm;
    Linear fc;
    size_t input_size, hidden_size, num_classes, seq_len;

    LSTMClassifier(size_t input_size, size_t hidden_size, size_t num_classes,
                   size_t seq_len, std::mt19937& rng);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();
    void clip_gradients(double max_norm);
};

// ============================================================
// Bidirectional LSTM
// ============================================================
class BiLSTMClassifier {
public:
    LSTMCell lstm_fwd, lstm_bwd;
    Linear fc;
    size_t input_size, hidden_size, num_classes, seq_len;

    BiLSTMClassifier(size_t input_size, size_t hidden_size, size_t num_classes,
                     size_t seq_len, std::mt19937& rng);

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();
    void clip_gradients(double max_norm);
};
