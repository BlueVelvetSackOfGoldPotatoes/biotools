#include "lstm_models.h"
#include "core/bio/bio_affine.h"
#include <cmath>
#include <tuple>

static Tensor sigmoid_t(const Tensor& x) {
    return x.apply([](double v) { return 1.0 / (1.0 + std::exp(-v)); });
}
static Tensor tanh_t(const Tensor& x) {
    return x.apply([](double v) { return std::tanh(v); });
}

// ============================================================
// LSTM Cell
// ============================================================
LSTMCell::LSTMCell(size_t is, size_t hs, std::mt19937& rng) : input_size(is), hidden_size(hs) {
    double std_val = 1.0 / std::sqrt((double)hs);
    auto init = [&](Tensor& W, size_t r, size_t c) {
        W = Tensor(r, c); W.fill_random_uniform(-std_val, std_val, rng);
    };
    init(W_ii, is, hs); init(W_hi, hs, hs); b_i = Tensor(1, hs);
    init(W_if, is, hs); init(W_hf, hs, hs); b_f = Tensor(1, hs, 1.0); // forget bias = 1
    init(W_ig, is, hs); init(W_hg, hs, hs); b_g = Tensor(1, hs);
    init(W_io, is, hs); init(W_ho, hs, hs); b_o = Tensor(1, hs);

    grad_W_ii = Tensor::zeros(is, hs); grad_W_hi = Tensor::zeros(hs, hs); grad_b_i = Tensor::zeros(1, hs);
    grad_W_if = Tensor::zeros(is, hs); grad_W_hf = Tensor::zeros(hs, hs); grad_b_f = Tensor::zeros(1, hs);
    grad_W_ig = Tensor::zeros(is, hs); grad_W_hg = Tensor::zeros(hs, hs); grad_b_g = Tensor::zeros(1, hs);
    grad_W_io = Tensor::zeros(is, hs); grad_W_ho = Tensor::zeros(hs, hs); grad_b_o = Tensor::zeros(1, hs);
}

void LSTMCell::zero_grad() {
    grad_W_ii.fill_zeros(); grad_W_hi.fill_zeros(); grad_b_i.fill_zeros();
    grad_W_if.fill_zeros(); grad_W_hf.fill_zeros(); grad_b_f.fill_zeros();
    grad_W_ig.fill_zeros(); grad_W_hg.fill_zeros(); grad_b_g.fill_zeros();
    grad_W_io.fill_zeros(); grad_W_ho.fill_zeros(); grad_b_o.fill_zeros();
}

std::pair<Tensor, Tensor> LSTMCell::forward_step(const Tensor& input, const Tensor& h_prev, const Tensor& c_prev) {
    LSTMCache cache;
    cache.input = input;
    cache.h_prev = h_prev;
    cache.c_prev = c_prev;

    auto gate = [&](Tensor& Wx, Tensor& Wh, const Tensor& b) {
        Tensor pre = bio::affine(input, Wx, nullptr, false) +
                     bio::affine(h_prev, Wh, nullptr, false);
        for (size_t i = 0; i < pre.rows; ++i)
            for (size_t j = 0; j < pre.cols; ++j)
                pre(i, j) += b(0, j);
        return pre;
    };

    cache.i_gate = sigmoid_t(gate(W_ii, W_hi, b_i));
    cache.f_gate = sigmoid_t(gate(W_if, W_hf, b_f));
    cache.g_gate = tanh_t(gate(W_ig, W_hg, b_g));
    cache.o_gate = sigmoid_t(gate(W_io, W_ho, b_o));

    cache.c_new = cache.f_gate * c_prev + cache.i_gate * cache.g_gate;
    cache.tanh_c = tanh_t(cache.c_new);
    cache.h_new = cache.o_gate * cache.tanh_c;

    caches.push_back(cache);
    return {cache.h_new, cache.c_new};
}

std::tuple<Tensor, Tensor, Tensor> LSTMCell::backward_step(const Tensor& grad_h, const Tensor& grad_c, int t) {
    const auto& c = caches[t];
    size_t batch = grad_h.rows;
    Tensor ones(batch, hidden_size, 1.0);

    const Tensor& tanh_c = c.tanh_c;
    Tensor d_tanh_c = grad_h * c.o_gate * (ones - tanh_c * tanh_c);
    Tensor dc = d_tanh_c + grad_c;

    // Gate gradients
    Tensor d_o = grad_h * tanh_c * c.o_gate * (ones - c.o_gate);
    Tensor d_f = dc * c.c_prev * c.f_gate * (ones - c.f_gate);
    Tensor d_i = dc * c.g_gate * c.i_gate * (ones - c.i_gate);
    Tensor d_g = dc * c.i_gate * (ones - c.g_gate * c.g_gate);

    // Accumulate weight gradients
    auto accum = [&](Tensor& gWx, Tensor& gWh, Tensor& gb, const Tensor& d) {
        gWx += c.input.transpose().matmul(d);
        gWh += c.h_prev.transpose().matmul(d);
        gb += d.sum_rows();
    };
    accum(grad_W_ii, grad_W_hi, grad_b_i, d_i);
    accum(grad_W_if, grad_W_hf, grad_b_f, d_f);
    accum(grad_W_ig, grad_W_hg, grad_b_g, d_g);
    accum(grad_W_io, grad_W_ho, grad_b_o, d_o);

    // Gradient to input
    Tensor grad_input = d_i.matmul(W_ii.transpose()) + d_f.matmul(W_if.transpose()) +
                        d_g.matmul(W_ig.transpose()) + d_o.matmul(W_io.transpose());

    // Gradient to h_prev
    Tensor grad_h_prev = d_i.matmul(W_hi.transpose()) + d_f.matmul(W_hf.transpose()) +
                         d_g.matmul(W_hg.transpose()) + d_o.matmul(W_ho.transpose());

    // Gradient to c_prev
    Tensor grad_c_prev = dc * c.f_gate;

    return {grad_input, grad_h_prev, grad_c_prev};
}

// ============================================================
// LSTM Classifier
// ============================================================
LSTMClassifier::LSTMClassifier(size_t is, size_t hs, size_t nc, size_t sl, std::mt19937& rng)
    : lstm(is, hs, rng), fc(hs, nc, "lstm_fc", rng),
      input_size(is), hidden_size(hs), num_classes(nc), seq_len(sl) {}

Tensor LSTMClassifier::forward(const Tensor& input) {
    size_t batch = input.rows;
    lstm.caches.clear();
    Tensor h = Tensor::zeros(batch, hidden_size);
    Tensor c = Tensor::zeros(batch, hidden_size);

    for (size_t t = 0; t < seq_len; ++t) {
        Tensor x_t(batch, input_size);
        for (size_t b = 0; b < batch; ++b)
            for (size_t j = 0; j < input_size; ++j)
                x_t(b, j) = input(b, t * input_size + j);
        auto [h_new, c_new] = lstm.forward_step(x_t, h, c);
        h = h_new; c = c_new;
    }
    return fc.forward(h);
}

Tensor LSTMClassifier::backward(const Tensor& grad_output) {
    Tensor grad_h = fc.backward(grad_output);
    size_t batch = grad_output.rows;
    Tensor grad_input_full(batch, seq_len * input_size, 0.0);
    Tensor grad_c = Tensor::zeros(batch, hidden_size);

    for (int t = (int)seq_len - 1; t >= 0; --t) {
        auto [gi, gh, gc] = lstm.backward_step(grad_h, grad_c, t);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < input_size; ++j) {
                grad_input_full(b, static_cast<size_t>(t) * input_size + j) += gi(b, j);
            }
        }
        grad_h = gh; grad_c = gc;
    }
    return grad_input_full;
}

std::vector<Tensor*> LSTMClassifier::parameters() {
    return {&lstm.W_ii, &lstm.W_hi, &lstm.b_i,
            &lstm.W_if, &lstm.W_hf, &lstm.b_f,
            &lstm.W_ig, &lstm.W_hg, &lstm.b_g,
            &lstm.W_io, &lstm.W_ho, &lstm.b_o,
            &fc.weights, &fc.biases};
}

std::vector<Tensor*> LSTMClassifier::gradients() {
    return {&lstm.grad_W_ii, &lstm.grad_W_hi, &lstm.grad_b_i,
            &lstm.grad_W_if, &lstm.grad_W_hf, &lstm.grad_b_f,
            &lstm.grad_W_ig, &lstm.grad_W_hg, &lstm.grad_b_g,
            &lstm.grad_W_io, &lstm.grad_W_ho, &lstm.grad_b_o,
            &fc.grad_weights, &fc.grad_biases};
}

void LSTMClassifier::zero_grad() {
    lstm.zero_grad(); fc.grad_weights.fill_zeros(); fc.grad_biases.fill_zeros();
}

void LSTMClassifier::clip_gradients(double max_norm) {
    auto grads = gradients();
    double total_norm_sq = 0.0;
    for (auto* g : grads) {
        double n = g->norm();
        total_norm_sq += n * n;
    }
    double total_norm = std::sqrt(total_norm_sq);
    if (total_norm > max_norm) {
        double scale = max_norm / total_norm;
        for (auto* g : grads)
            *g *= scale;
    }
}

// ============================================================
// BiLSTM Classifier
// ============================================================
BiLSTMClassifier::BiLSTMClassifier(size_t is, size_t hs, size_t nc, size_t sl, std::mt19937& rng)
    : lstm_fwd(is, hs, rng), lstm_bwd(is, hs, rng), fc(hs * 2, nc, "bilstm_fc", rng),
      input_size(is), hidden_size(hs), num_classes(nc), seq_len(sl) {}

Tensor BiLSTMClassifier::forward(const Tensor& input) {
    size_t batch = input.rows;
    lstm_fwd.caches.clear();
    lstm_bwd.caches.clear();

    Tensor h_f = Tensor::zeros(batch, hidden_size), c_f = Tensor::zeros(batch, hidden_size);
    Tensor h_b = Tensor::zeros(batch, hidden_size), c_b = Tensor::zeros(batch, hidden_size);

    // Forward direction
    for (size_t t = 0; t < seq_len; ++t) {
        Tensor x_t(batch, input_size);
        for (size_t b = 0; b < batch; ++b)
            for (size_t j = 0; j < input_size; ++j)
                x_t(b, j) = input(b, t * input_size + j);
        auto [hn, cn] = lstm_fwd.forward_step(x_t, h_f, c_f);
        h_f = hn; c_f = cn;
    }
    // Backward direction
    for (int t = (int)seq_len - 1; t >= 0; --t) {
        Tensor x_t(batch, input_size);
        for (size_t b = 0; b < batch; ++b)
            for (size_t j = 0; j < input_size; ++j)
                x_t(b, j) = input(b, t * input_size + j);
        auto [hn, cn] = lstm_bwd.forward_step(x_t, h_b, c_b);
        h_b = hn; c_b = cn;
    }

    return fc.forward(Tensor::hstack(h_f, h_b));
}

Tensor BiLSTMClassifier::backward(const Tensor& grad_output) {
    Tensor grad_fc = fc.backward(grad_output);
    size_t batch = grad_output.rows;
    Tensor grad_input_full(batch, seq_len * input_size, 0.0);

    Tensor grad_h_f = grad_fc.slice_cols(0, hidden_size);
    Tensor grad_h_b = grad_fc.slice_cols(hidden_size, hidden_size);
    Tensor grad_c_f = Tensor::zeros(batch, hidden_size);
    Tensor grad_c_b = Tensor::zeros(batch, hidden_size);

    for (int t = (int)seq_len - 1; t >= 0; --t) {
        auto [gi, gh, gc] = lstm_fwd.backward_step(grad_h_f, grad_c_f, t);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < input_size; ++j) {
                grad_input_full(b, static_cast<size_t>(t) * input_size + j) += gi(b, j);
            }
        }
        grad_h_f = gh; grad_c_f = gc;
    }
    for (int t = (int)seq_len - 1; t >= 0; --t) {
        auto [gi, gh, gc] = lstm_bwd.backward_step(grad_h_b, grad_c_b, t);
        const size_t orig_t = seq_len - 1 - static_cast<size_t>(t);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < input_size; ++j) {
                grad_input_full(b, orig_t * input_size + j) += gi(b, j);
            }
        }
        grad_h_b = gh; grad_c_b = gc;
    }
    return grad_input_full;
}

std::vector<Tensor*> BiLSTMClassifier::parameters() {
    std::vector<Tensor*> p = {
        &lstm_fwd.W_ii, &lstm_fwd.W_hi, &lstm_fwd.b_i,
        &lstm_fwd.W_if, &lstm_fwd.W_hf, &lstm_fwd.b_f,
        &lstm_fwd.W_ig, &lstm_fwd.W_hg, &lstm_fwd.b_g,
        &lstm_fwd.W_io, &lstm_fwd.W_ho, &lstm_fwd.b_o,
        &lstm_bwd.W_ii, &lstm_bwd.W_hi, &lstm_bwd.b_i,
        &lstm_bwd.W_if, &lstm_bwd.W_hf, &lstm_bwd.b_f,
        &lstm_bwd.W_ig, &lstm_bwd.W_hg, &lstm_bwd.b_g,
        &lstm_bwd.W_io, &lstm_bwd.W_ho, &lstm_bwd.b_o,
        &fc.weights, &fc.biases
    };
    return p;
}

std::vector<Tensor*> BiLSTMClassifier::gradients() {
    return {
        &lstm_fwd.grad_W_ii, &lstm_fwd.grad_W_hi, &lstm_fwd.grad_b_i,
        &lstm_fwd.grad_W_if, &lstm_fwd.grad_W_hf, &lstm_fwd.grad_b_f,
        &lstm_fwd.grad_W_ig, &lstm_fwd.grad_W_hg, &lstm_fwd.grad_b_g,
        &lstm_fwd.grad_W_io, &lstm_fwd.grad_W_ho, &lstm_fwd.grad_b_o,
        &lstm_bwd.grad_W_ii, &lstm_bwd.grad_W_hi, &lstm_bwd.grad_b_i,
        &lstm_bwd.grad_W_if, &lstm_bwd.grad_W_hf, &lstm_bwd.grad_b_f,
        &lstm_bwd.grad_W_ig, &lstm_bwd.grad_W_hg, &lstm_bwd.grad_b_g,
        &lstm_bwd.grad_W_io, &lstm_bwd.grad_W_ho, &lstm_bwd.grad_b_o,
        &fc.grad_weights, &fc.grad_biases
    };
}

void BiLSTMClassifier::zero_grad() {
    lstm_fwd.zero_grad(); lstm_bwd.zero_grad();
    fc.grad_weights.fill_zeros(); fc.grad_biases.fill_zeros();
}

void BiLSTMClassifier::clip_gradients(double max_norm) {
    auto grads = gradients();
    double total_norm_sq = 0.0;
    for (auto* g : grads) {
        double n = g->norm();
        total_norm_sq += n * n;
    }
    double total_norm = std::sqrt(total_norm_sq);
    if (total_norm > max_norm) {
        double scale = max_norm / total_norm;
        for (auto* g : grads)
            *g *= scale;
    }
}
