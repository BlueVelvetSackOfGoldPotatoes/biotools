#include "rnn_models.h"
#include "core/bio/bio_affine.h"
#include <cmath>

// Helper: sigmoid
static Tensor sigmoid_t(const Tensor& x) {
    return x.apply([](double v) { return 1.0 / (1.0 + std::exp(-v)); });
}
static Tensor tanh_t(const Tensor& x) {
    return x.apply([](double v) { return std::tanh(v); });
}

// ============================================================
// RNN Cell
// ============================================================
RNNCell::RNNCell(size_t is, size_t hs, std::mt19937& rng) : input_size(is), hidden_size(hs) {
    double std_val = 1.0 / std::sqrt((double)hs);
    W_ih = Tensor(is, hs); W_ih.fill_random_uniform(-std_val, std_val, rng);
    W_hh = Tensor(hs, hs); W_hh.fill_random_uniform(-std_val, std_val, rng);
    b_h = Tensor(1, hs);
    grad_W_ih = Tensor::zeros(is, hs);
    grad_W_hh = Tensor::zeros(hs, hs);
    grad_b_h = Tensor::zeros(1, hs);
}

void RNNCell::zero_grad() {
    grad_W_ih.fill_zeros(); grad_W_hh.fill_zeros(); grad_b_h.fill_zeros();
}

Tensor RNNCell::forward_step(const Tensor& input, const Tensor& h_prev) {
    // h_new = tanh(input @ W_ih + h_prev @ W_hh + b_h)
    Tensor pre_act = bio::affine(input, W_ih, nullptr, false) +
                     bio::affine(h_prev, W_hh, nullptr, false);
    for (size_t i = 0; i < pre_act.rows; ++i)
        for (size_t j = 0; j < pre_act.cols; ++j)
            pre_act(i, j) += b_h(0, j);
    Tensor h_new = tanh_t(pre_act);

    cached_inputs.push_back(input);
    cached_hiddens.push_back(h_prev);
    cached_pre_act.push_back(pre_act);
    cached_tanh_out.push_back(h_new);
    return h_new;
}

std::pair<Tensor, Tensor> RNNCell::backward_step(const Tensor& grad_h, int t) {
    const Tensor& input = cached_inputs[t];
    const Tensor& h_prev = cached_hiddens[t];
    const Tensor& tanh_val = cached_tanh_out[t];

    // d_tanh = grad_h * (1 - tanh^2(pre_act)), using cached tanh output
    Tensor d_pre = grad_h * (Tensor(tanh_val.rows, tanh_val.cols, 1.0) - tanh_val * tanh_val);

    grad_W_ih += input.transpose().matmul(d_pre);
    grad_W_hh += h_prev.transpose().matmul(d_pre);
    grad_b_h += d_pre.sum_rows();

    Tensor grad_input = d_pre.matmul(W_ih.transpose());
    Tensor grad_h_prev = d_pre.matmul(W_hh.transpose());
    return {grad_input, grad_h_prev};
}

// ============================================================
// GRU Cell
// ============================================================
GRUCell::GRUCell(size_t is, size_t hs, std::mt19937& rng) : input_size(is), hidden_size(hs) {
    double std_val = 1.0 / std::sqrt((double)hs);
    auto init = [&](Tensor& W, size_t r, size_t c) {
        W = Tensor(r, c); W.fill_random_uniform(-std_val, std_val, rng);
    };
    init(W_iz, is, hs); init(W_hz, hs, hs); b_z = Tensor(1, hs);
    init(W_ir, is, hs); init(W_hr, hs, hs); b_r = Tensor(1, hs);
    init(W_in, is, hs); init(W_hn, hs, hs); b_n = Tensor(1, hs);

    grad_W_iz = Tensor::zeros(is, hs); grad_W_hz = Tensor::zeros(hs, hs); grad_b_z = Tensor::zeros(1, hs);
    grad_W_ir = Tensor::zeros(is, hs); grad_W_hr = Tensor::zeros(hs, hs); grad_b_r = Tensor::zeros(1, hs);
    grad_W_in = Tensor::zeros(is, hs); grad_W_hn = Tensor::zeros(hs, hs); grad_b_n = Tensor::zeros(1, hs);
}

void GRUCell::zero_grad() {
    grad_W_iz.fill_zeros(); grad_W_hz.fill_zeros(); grad_b_z.fill_zeros();
    grad_W_ir.fill_zeros(); grad_W_hr.fill_zeros(); grad_b_r.fill_zeros();
    grad_W_in.fill_zeros(); grad_W_hn.fill_zeros(); grad_b_n.fill_zeros();
}

Tensor GRUCell::forward_step(const Tensor& input, const Tensor& h_prev) {
    GRUCache cache;
    cache.input = input;
    cache.h_prev = h_prev;

    // z = sigmoid(x @ W_iz + h @ W_hz + b_z)
    Tensor z_pre = bio::affine(input, W_iz, nullptr, false) +
                   bio::affine(h_prev, W_hz, nullptr, false);
    for (size_t i = 0; i < z_pre.rows; ++i)
        for (size_t j = 0; j < z_pre.cols; ++j) z_pre(i, j) += b_z(0, j);
    cache.z = sigmoid_t(z_pre);

    // r = sigmoid(x @ W_ir + h @ W_hr + b_r)
    Tensor r_pre = bio::affine(input, W_ir, nullptr, false) +
                   bio::affine(h_prev, W_hr, nullptr, false);
    for (size_t i = 0; i < r_pre.rows; ++i)
        for (size_t j = 0; j < r_pre.cols; ++j) r_pre(i, j) += b_r(0, j);
    cache.r = sigmoid_t(r_pre);

    // n = tanh(x @ W_in + (r * h) @ W_hn + b_n)
    Tensor n_pre = bio::affine(input, W_in, nullptr, false) +
                   bio::affine(cache.r * h_prev, W_hn, nullptr, false);
    for (size_t i = 0; i < n_pre.rows; ++i)
        for (size_t j = 0; j < n_pre.cols; ++j) n_pre(i, j) += b_n(0, j);
    cache.n_tilde = tanh_t(n_pre);

    // h_new = (1 - z) * n + z * h_prev
    Tensor one_minus_z = cache.z.apply([](double v) { return 1.0 - v; });
    cache.h_new = one_minus_z * cache.n_tilde + cache.z * h_prev;

    caches.push_back(cache);
    return cache.h_new;
}

std::pair<Tensor, Tensor> GRUCell::backward_step(const Tensor& grad_h, int t) {
    const auto& c = caches[t];
    size_t batch = grad_h.rows;

    // dh/dz = h_prev - n_tilde
    Tensor d_z = grad_h * (c.h_prev - c.n_tilde);
    // dh/dn = (1 - z)
    Tensor one_minus_z = c.z.apply([](double v) { return 1.0 - v; });
    Tensor d_n = grad_h * one_minus_z;
    // dh/dh_prev (through z)
    Tensor d_h_prev = grad_h * c.z;

    // Backprop through tanh for n
    Tensor d_n_pre = d_n * (Tensor(batch, hidden_size, 1.0) - c.n_tilde * c.n_tilde);

    // Backprop through sigmoid for z
    Tensor d_z_pre = d_z * c.z * (Tensor(batch, hidden_size, 1.0) - c.z);

    // Gradients for n gate
    grad_W_in += c.input.transpose().matmul(d_n_pre);
    Tensor rh = c.r * c.h_prev;
    grad_W_hn += rh.transpose().matmul(d_n_pre);
    grad_b_n += d_n_pre.sum_rows();

    // d_r from n backprop: d(r*h) = d_n_pre @ W_hn^T, then * h_prev for r
    Tensor d_rh = d_n_pre.matmul(W_hn.transpose());
    Tensor d_r = d_rh * c.h_prev;
    d_h_prev += d_rh * c.r;

    // Backprop through sigmoid for r
    Tensor d_r_pre = d_r * c.r * (Tensor(batch, hidden_size, 1.0) - c.r);

    // Gradients for z gate
    grad_W_iz += c.input.transpose().matmul(d_z_pre);
    grad_W_hz += c.h_prev.transpose().matmul(d_z_pre);
    grad_b_z += d_z_pre.sum_rows();

    // Gradients for r gate
    grad_W_ir += c.input.transpose().matmul(d_r_pre);
    grad_W_hr += c.h_prev.transpose().matmul(d_r_pre);
    grad_b_r += d_r_pre.sum_rows();

    // Gradient to input
    Tensor grad_input = d_z_pre.matmul(W_iz.transpose()) +
                        d_r_pre.matmul(W_ir.transpose()) +
                        d_n_pre.matmul(W_in.transpose());

    // Gradient to h_prev
    d_h_prev += d_z_pre.matmul(W_hz.transpose()) +
                d_r_pre.matmul(W_hr.transpose());

    return {grad_input, d_h_prev};
}

// ============================================================
// RNN Classifier
// ============================================================
RNNClassifier::RNNClassifier(size_t is, size_t hs, size_t nc, size_t sl, bool bidir, std::mt19937& rng)
    : rnn(is, hs, rng), fc(bidir ? hs * 2 : hs, nc, "rnn_fc", rng),
      input_size(is), hidden_size(hs), num_classes(nc), seq_len(sl), bidirectional(bidir) {
    if (bidir) rnn_back = std::make_unique<RNNCell>(is, hs, rng);
}

Tensor RNNClassifier::forward(const Tensor& input) {
    size_t batch = input.rows;
    // input: batch x (seq_len * input_size), reshape to seq of batch x input_size
    rnn.cached_inputs.clear();
    rnn.cached_hiddens.clear();
    rnn.cached_pre_act.clear();
    rnn.cached_tanh_out.clear();

    Tensor h = Tensor::zeros(batch, hidden_size);
    for (size_t t = 0; t < seq_len; ++t) {
        Tensor x_t(batch, input_size);
        for (size_t b = 0; b < batch; ++b)
            for (size_t j = 0; j < input_size; ++j)
                x_t(b, j) = input(b, t * input_size + j);
        h = rnn.forward_step(x_t, h);
    }

    Tensor final_h = h;

    if (bidirectional) {
        rnn_back->cached_inputs.clear();
        rnn_back->cached_hiddens.clear();
        rnn_back->cached_pre_act.clear();
        rnn_back->cached_tanh_out.clear();
        Tensor h_back = Tensor::zeros(batch, hidden_size);
        for (int t = (int)seq_len - 1; t >= 0; --t) {
            Tensor x_t(batch, input_size);
            for (size_t b = 0; b < batch; ++b)
                for (size_t j = 0; j < input_size; ++j)
                    x_t(b, j) = input(b, t * input_size + j);
            h_back = rnn_back->forward_step(x_t, h_back);
        }
        final_h = Tensor::hstack(h, h_back);
    }

    return fc.forward(final_h);
}

Tensor RNNClassifier::backward(const Tensor& grad_output) {
    Tensor grad_fc = fc.backward(grad_output);
    const size_t batch = grad_output.rows;
    Tensor grad_input_full(batch, seq_len * input_size, 0.0);

    Tensor grad_h;
    if (bidirectional) {
        grad_h = grad_fc.slice_cols(0, hidden_size);
        Tensor grad_h_back = grad_fc.slice_cols(hidden_size, hidden_size);
        // Backprop backward RNN (BPTT in reverse cache order)
        for (int t = (int)seq_len - 1; t >= 0; --t) {
            auto [gi, gh] = rnn_back->backward_step(grad_h_back, t);
            const size_t orig_t = seq_len - 1 - static_cast<size_t>(t);
            for (size_t b = 0; b < batch; ++b) {
                for (size_t j = 0; j < input_size; ++j) {
                    grad_input_full(b, orig_t * input_size + j) += gi(b, j);
                }
            }
            grad_h_back = gh;
        }
    } else {
        grad_h = grad_fc;
    }

    // Backprop forward RNN through time
    for (int t = (int)seq_len - 1; t >= 0; --t) {
        auto [grad_input, grad_h_prev] = rnn.backward_step(grad_h, t);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < input_size; ++j) {
                grad_input_full(b, static_cast<size_t>(t) * input_size + j) += grad_input(b, j);
            }
        }
        grad_h = grad_h_prev;
    }

    return grad_input_full;
}

std::vector<Tensor*> RNNClassifier::parameters() {
    std::vector<Tensor*> p = {&rnn.W_ih, &rnn.W_hh, &rnn.b_h, &fc.weights, &fc.biases};
    if (bidirectional) {
        p.push_back(&rnn_back->W_ih);
        p.push_back(&rnn_back->W_hh);
        p.push_back(&rnn_back->b_h);
    }
    return p;
}

std::vector<Tensor*> RNNClassifier::gradients() {
    std::vector<Tensor*> g = {&rnn.grad_W_ih, &rnn.grad_W_hh, &rnn.grad_b_h, &fc.grad_weights, &fc.grad_biases};
    if (bidirectional) {
        g.push_back(&rnn_back->grad_W_ih);
        g.push_back(&rnn_back->grad_W_hh);
        g.push_back(&rnn_back->grad_b_h);
    }
    return g;
}

void RNNClassifier::zero_grad() {
    rnn.zero_grad(); fc.grad_weights.fill_zeros(); fc.grad_biases.fill_zeros();
    if (bidirectional) rnn_back->zero_grad();
}

void RNNClassifier::clip_gradients(double max_norm) {
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
// GRU Classifier
// ============================================================
GRUClassifier::GRUClassifier(size_t is, size_t hs, size_t nc, size_t sl, std::mt19937& rng)
    : gru(is, hs, rng), fc(hs, nc, "gru_fc", rng),
      input_size(is), hidden_size(hs), num_classes(nc), seq_len(sl) {}

Tensor GRUClassifier::forward(const Tensor& input) {
    size_t batch = input.rows;
    gru.caches.clear();
    Tensor h = Tensor::zeros(batch, hidden_size);
    for (size_t t = 0; t < seq_len; ++t) {
        Tensor x_t(batch, input_size);
        for (size_t b = 0; b < batch; ++b)
            for (size_t j = 0; j < input_size; ++j)
                x_t(b, j) = input(b, t * input_size + j);
        h = gru.forward_step(x_t, h);
    }
    return fc.forward(h);
}

Tensor GRUClassifier::backward(const Tensor& grad_output) {
    Tensor grad_h = fc.backward(grad_output);
    const size_t batch = grad_output.rows;
    Tensor grad_input_full(batch, seq_len * input_size, 0.0);
    for (int t = (int)seq_len - 1; t >= 0; --t) {
        auto [gi, gh] = gru.backward_step(grad_h, t);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < input_size; ++j) {
                grad_input_full(b, static_cast<size_t>(t) * input_size + j) += gi(b, j);
            }
        }
        grad_h = gh;
    }
    return grad_input_full;
}

std::vector<Tensor*> GRUClassifier::parameters() {
    return {&gru.W_iz, &gru.W_hz, &gru.b_z,
            &gru.W_ir, &gru.W_hr, &gru.b_r,
            &gru.W_in, &gru.W_hn, &gru.b_n,
            &fc.weights, &fc.biases};
}

std::vector<Tensor*> GRUClassifier::gradients() {
    return {&gru.grad_W_iz, &gru.grad_W_hz, &gru.grad_b_z,
            &gru.grad_W_ir, &gru.grad_W_hr, &gru.grad_b_r,
            &gru.grad_W_in, &gru.grad_W_hn, &gru.grad_b_n,
            &fc.grad_weights, &fc.grad_biases};
}

void GRUClassifier::zero_grad() {
    gru.zero_grad(); fc.grad_weights.fill_zeros(); fc.grad_biases.fill_zeros();
}

void GRUClassifier::clip_gradients(double max_norm) {
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
