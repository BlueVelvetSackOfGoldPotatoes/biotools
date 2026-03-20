#include "core/tensor/tensor.h"
#include "core/nn/module.h"
#include "core/losses/losses.h"
#include "core/metrics/metrics.h"
#include "core/optim/optimizer.h"
#include "core/online/continuous_runtime.h"
#include "models/markov/src/markov_models.h"
#include "models/reinforcement/src/rl_models.h"
#include "models/cells/src/cell_models.h"
#include <cmath>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;

static void check(bool cond, const std::string& name) {
    if (cond) {
        std::cout << "  PASS  " << name << std::endl;
        g_pass++;
    } else {
        std::cerr << "  FAIL  " << name << std::endl;
        g_fail++;
    }
}

// =======================================================================
// Test 1: Tensor matmul with known values
// =======================================================================
static void test_matmul() {
    // [1 2 3] * [7  8 ]   = [58  64 ]
    // [4 5 6]   [9  10]     [139 154]
    //           [11 12]
    Tensor a(2, 3);
    a(0,0)=1; a(0,1)=2; a(0,2)=3;
    a(1,0)=4; a(1,1)=5; a(1,2)=6;

    Tensor b(3, 2);
    b(0,0)=7;  b(0,1)=8;
    b(1,0)=9;  b(1,1)=10;
    b(2,0)=11; b(2,1)=12;

    Tensor c = a.matmul(b);
    check(c.rows == 2 && c.cols == 2, "matmul shape");
    check(std::abs(c(0,0) - 58.0) < 1e-9 && std::abs(c(0,1) - 64.0) < 1e-9 &&
          std::abs(c(1,0) - 139.0) < 1e-9 && std::abs(c(1,1) - 154.0) < 1e-9,
          "matmul values");
}

// =======================================================================
// Test 2: Tensor transpose
// =======================================================================
static void test_transpose() {
    Tensor a(2, 3);
    a(0,0)=1; a(0,1)=2; a(0,2)=3;
    a(1,0)=4; a(1,1)=5; a(1,2)=6;

    Tensor t = a.transpose();
    check(t.rows == 3 && t.cols == 2, "transpose shape");
    check(std::abs(t(0,0) - 1.0) < 1e-9 && std::abs(t(2,1) - 6.0) < 1e-9, "transpose values");
}

// =======================================================================
// Test 3: Element-wise add
// =======================================================================
static void test_elementwise_add() {
    Tensor a(2, 3);
    a(0,0)=1; a(0,1)=2; a(0,2)=3;
    a(1,0)=4; a(1,1)=5; a(1,2)=6;

    Tensor b(2, 3);
    b(0,0)=10; b(0,1)=20; b(0,2)=30;
    b(1,0)=10; b(1,1)=20; b(1,2)=30;

    Tensor c = a + b;
    check(c.rows == 2 && c.cols == 3, "elementwise add shape");
    check(std::abs(c(0,0) - 11.0) < 1e-9 && std::abs(c(1,2) - 36.0) < 1e-9, "elementwise add values");
}

// =======================================================================
// Test 4: Conv2D forward with known kernel
// =======================================================================
static void test_conv2d_forward() {
    std::mt19937 rng(123);
    // 1 channel, 4x4 input, 1 output channel, 3x3 kernel
    Conv2D conv(1, 1, 3, 4, 4, 1, 0, "test_conv", rng);

    // Set kernel to all 1s, bias to 0
    auto params = conv.parameters();
    // params[0] = weights: shape (out_ch, in_ch*kH*kW) = (1, 9)
    // params[1] = bias: shape (1, 1)
    for (auto& v : params[0]->data) v = 1.0;
    for (auto& v : params[1]->data) v = 0.0;

    // Input: 1 sample, 1 channel, 4x4 = row (1, 16)
    Tensor x(1, 16);
    for (size_t i = 0; i < 16; ++i) x(0, i) = 1.0;

    Tensor out = conv.forward(x);
    // With all-1 kernel on all-1 4x4 input, stride=1, pad=0:
    // output is 2x2, each value = 9.0
    check(out.cols == 4, "conv2d output size (1 * 2 * 2 = 4)");
    bool all_nine = true;
    for (size_t i = 0; i < out.data.size(); ++i)
        if (std::abs(out.data[i] - 9.0) > 1e-6) all_nine = false;
    check(all_nine, "conv2d output values (all 9.0)");
}

// =======================================================================
// Test 5: BatchNorm1d train vs eval mode
// =======================================================================
static void test_batchnorm1d() {
    BatchNorm1d bn(3);

    // Batch of 4 samples, 3 features
    Tensor x(4, 3);
    x(0,0)=1; x(0,1)=4; x(0,2)=2;
    x(1,0)=3; x(1,1)=6; x(1,2)=4;
    x(2,0)=5; x(2,1)=8; x(2,2)=6;
    x(3,0)=7; x(3,1)=10; x(3,2)=8;

    // Train mode: should normalize each feature to ~zero mean
    bn.train();
    Tensor out = bn.forward(x);
    double mean0 = 0, mean1 = 0, mean2 = 0;
    for (size_t i = 0; i < 4; ++i) {
        mean0 += out(i, 0);
        mean1 += out(i, 1);
        mean2 += out(i, 2);
    }
    mean0 /= 4; mean1 /= 4; mean2 /= 4;
    check(std::abs(mean0) < 1e-5 && std::abs(mean1) < 1e-5 && std::abs(mean2) < 1e-5,
          "batchnorm1d train mode zero mean");

    // Eval mode: uses running stats
    bn.eval();
    Tensor out2 = bn.forward(x);
    bool finite = true;
    for (auto& v : out2.data)
        if (!std::isfinite(v)) finite = false;
    check(finite, "batchnorm1d eval mode finite output");
}

// =======================================================================
// Test 5b: BatchNorm tiny-batch stability
// =======================================================================
static void test_batchnorm_tiny_batch() {
    BatchNorm1d bn1(3);
    bn1.train();
    Tensor x1(1, 3);
    x1(0, 0) = 0.2;
    x1(0, 1) = -0.4;
    x1(0, 2) = 1.7;
    Tensor out1 = bn1.forward(x1);
    bool finite1 = true;
    for (double v : out1.data) if (!std::isfinite(v)) finite1 = false;
    check(finite1, "batchnorm1d tiny-batch finite output");

    BatchNorm2d bn2(2, 1, 1);
    bn2.train();
    Tensor x2(1, 2); // N=1, C=2, H=W=1 => M=1
    x2(0, 0) = 0.3;
    x2(0, 1) = -0.1;
    Tensor out2 = bn2.forward(x2);
    bool finite2 = true;
    for (double v : out2.data) if (!std::isfinite(v)) finite2 = false;
    check(finite2, "batchnorm2d tiny-batch finite output");
}

// =======================================================================
// Test 6: CrossEntropyLoss numerical gradient check
// =======================================================================
static void test_ce_gradient() {
    CrossEntropyLoss loss_fn;
    // 2 samples, 3 classes
    Tensor logits(2, 3);
    logits(0,0)=1.0; logits(0,1)=2.0; logits(0,2)=0.5;
    logits(1,0)=0.1; logits(1,1)=0.3; logits(1,2)=2.0;

    Tensor target(2, 3, 0.0);
    target(0, 1) = 1.0; // class 1
    target(1, 2) = 1.0; // class 2

    Tensor probs = LossFunctions::softmax(logits);
    loss_fn.forward(probs, target);
    Tensor analytic_grad = loss_fn.backward(probs, target);

    // Numerical gradient via finite differences on logits
    const double eps = 1e-5;
    double max_err = 0.0;
    for (size_t i = 0; i < logits.rows; ++i) {
        for (size_t j = 0; j < logits.cols; ++j) {
            Tensor logits_plus = logits;
            logits_plus(i, j) += eps;
            Tensor p_plus = LossFunctions::softmax(logits_plus);
            double loss_plus = loss_fn.forward(p_plus, target);

            Tensor logits_minus = logits;
            logits_minus(i, j) -= eps;
            Tensor p_minus = LossFunctions::softmax(logits_minus);
            double loss_minus = loss_fn.forward(p_minus, target);

            double numerical = (loss_plus - loss_minus) / (2.0 * eps);
            double err = std::abs(analytic_grad(i, j) - numerical);
            max_err = std::max(max_err, err);
        }
    }
    check(max_err < 1e-4, "cross-entropy numerical gradient check (max_err=" + std::to_string(max_err) + ")");
}

// =======================================================================
// Test 6b: Softmax backward chain-rule check for non-CE losses
// =======================================================================
static void test_softmax_backward_chain_rule() {
    MSELoss loss_fn;
    Tensor logits(2, 3);
    logits(0,0)=0.2; logits(0,1)=1.1; logits(0,2)=-0.7;
    logits(1,0)=1.8; logits(1,1)=-0.3; logits(1,2)=0.4;

    Tensor target(2, 3, 0.0);
    target(0, 1) = 1.0;
    target(1, 0) = 1.0;

    Tensor probs = LossFunctions::softmax(logits);
    (void)loss_fn.forward(probs, target);
    Tensor grad_probs = loss_fn.backward(probs, target);
    Tensor analytic_grad = LossFunctions::softmax_backward(probs, grad_probs);

    const double eps = 1e-5;
    double max_err = 0.0;
    for (size_t i = 0; i < logits.rows; ++i) {
        for (size_t j = 0; j < logits.cols; ++j) {
            Tensor logits_plus = logits;
            logits_plus(i, j) += eps;
            Tensor p_plus = LossFunctions::softmax(logits_plus);
            double loss_plus = loss_fn.forward(p_plus, target);

            Tensor logits_minus = logits;
            logits_minus(i, j) -= eps;
            Tensor p_minus = LossFunctions::softmax(logits_minus);
            double loss_minus = loss_fn.forward(p_minus, target);

            double numerical = (loss_plus - loss_minus) / (2.0 * eps);
            double err = std::abs(analytic_grad(i, j) - numerical);
            max_err = std::max(max_err, err);
        }
    }
    check(max_err < 1e-4, "softmax_backward chain-rule check (max_err=" + std::to_string(max_err) + ")");
}

// =======================================================================
// Test 7: Adam single-step update
// =======================================================================
static void test_adam_step() {
    std::mt19937 rng(42);
    Sequential model;
    model.add(std::make_unique<Linear>(2, 1, "test_adam_fc", rng));

    // Save initial weight
    auto params = model.parameters();
    double w0 = params[0]->data[0];

    // Create a simple gradient
    Tensor x(1, 2);
    x(0, 0) = 1.0; x(0, 1) = 0.5;
    Tensor y(1, 1);
    y(0, 0) = 1.0;

    Tensor out = model.forward(x);
    // MSE gradient: 2*(out - y)
    Tensor grad(1, 1);
    grad(0, 0) = 2.0 * (out(0, 0) - y(0, 0));
    model.backward(grad);

    Adam opt(0.01);
    opt.step(model);

    double w1 = params[0]->data[0];
    check(w0 != w1, "adam step changed weight");
    check(std::isfinite(w1), "adam step produced finite weight");
}

// =======================================================================
// Test 8: Adam vector-based interface
// =======================================================================
static void test_adam_vector_interface() {
    Tensor param(2, 3);
    for (auto& v : param.data) v = 1.0;

    Tensor grad(2, 3);
    for (auto& v : grad.data) v = 0.1;

    std::vector<Tensor*> params = {&param};
    std::vector<Tensor*> grads = {&grad};

    Adam opt(0.01);
    opt.step(params, grads);

    // Parameters should have decreased (gradient is positive)
    bool decreased = true;
    for (auto& v : param.data)
        if (v >= 1.0) decreased = false;
    check(decreased, "adam vector step decreased params");

    // Test lr_scale
    double before = param.data[0];
    for (auto& v : grad.data) v = 0.1;
    opt.step(params, grads, 0.5);
    double after = param.data[0];
    check(after < before, "adam vector step with lr_scale");

    // Test zero_grad
    Adam::zero_grad(grads);
    bool all_zero = true;
    for (auto& v : grad.data)
        if (v != 0.0) all_zero = false;
    check(all_zero, "adam static zero_grad");
}

// =======================================================================
// Test 9: ReplayBuffer push/sample
// =======================================================================
static void test_replay_buffer() {
    ReplayBuffer buf(10);
    check(buf.size() == 0, "replay buffer initially empty");

    std::mt19937 rng(42);

    // Push 5 items
    for (int i = 0; i < 5; ++i) {
        RLTransition t;
        t.state = Tensor(1, 4);
        t.state(0, 0) = static_cast<double>(i);
        t.action = i;
        t.reward = static_cast<double>(i) * 0.1;
        t.next_state = Tensor(1, 4);
        t.done = 0;
        buf.push(t);
    }
    check(buf.size() == 5, "replay buffer size after 5 pushes");

    // Sample 3 items
    auto samples = buf.sample(3, rng);
    check(samples.size() == 3, "replay buffer sample returns 3");
    bool valid = true;
    for (auto& s : samples)
        if (s.action < 0 || s.action > 4) valid = false;
    check(valid, "replay buffer sample returns valid actions");

    // Push past capacity to test wrapping
    for (int i = 5; i < 15; ++i) {
        RLTransition t;
        t.state = Tensor(1, 4);
        t.action = i % 10;
        t.reward = 0.0;
        t.next_state = Tensor(1, 4);
        t.done = 0;
        buf.push(t);
    }
    check(buf.size() == 10, "replay buffer wraps at capacity");
}

// =======================================================================
// Test 10: ContinuousReplayBuffer push/sample
// =======================================================================
static void test_continuous_replay_buffer() {
    ContinuousReplayBuffer buf(8);
    check(buf.size() == 0, "continuous replay buffer initially empty");

    std::mt19937 rng(42);
    std::vector<std::size_t> indices = {0, 1, 2, 3, 4};
    buf.push_many(indices);
    check(buf.size() == 5, "continuous replay buffer size after push_many");

    std::vector<std::size_t> out;
    buf.sample(3, rng, out);
    check(out.size() == 3, "continuous replay buffer sample returns 3");
    bool valid = true;
    for (auto idx : out)
        if (idx > 4) valid = false;
    check(valid, "continuous replay buffer sample returns valid indices");

    // Wrap
    for (std::size_t i = 0; i < 10; ++i) buf.push(i + 100);
    check(buf.size() == 8, "continuous replay buffer wraps at capacity");
}

// =======================================================================
// Test 11: Original smoke test (training loop convergence)
// =======================================================================
static void test_training_convergence() {
    std::mt19937 rng(42);

    Tensor x(4, 2);
    x(0,0)=0; x(0,1)=0;
    x(1,0)=0; x(1,1)=1;
    x(2,0)=1; x(2,1)=0;
    x(3,0)=1; x(3,1)=1;

    Tensor y(4, 2);
    y(0,0)=1; y(0,1)=0;
    y(1,0)=1; y(1,1)=0;
    y(2,0)=0; y(2,1)=1;
    y(3,0)=0; y(3,1)=1;

    Sequential model;
    model.add(std::make_unique<Linear>(2, 8, "fc1", rng));
    model.add(std::make_unique<ReLU>());
    model.add(std::make_unique<Linear>(8, 2, "fc2", rng));

    CrossEntropyLoss loss_fn;
    Adam opt(0.05);

    double first_loss = 0.0, last_loss = 0.0;
    for (int i = 0; i < 200; ++i) {
        Tensor logits = model.forward(x);
        Tensor probs = LossFunctions::softmax(logits);
        double loss = loss_fn.forward(probs, y);
        if (i == 0) first_loss = loss;
        last_loss = loss;
        Tensor grad = loss_fn.backward(probs, y);
        model.backward(grad);
        opt.step(model);
        opt.zero_grad(model);
    }

    Tensor logits = model.forward(x);
    Tensor probs = LossFunctions::softmax(logits);
    double acc = 0.0;
    for (size_t i = 0; i < probs.rows; ++i) {
        size_t pred = probs(i, 0) > probs(i, 1) ? 0 : 1;
        size_t truth = y(i, 0) > y(i, 1) ? 0 : 1;
        if (pred == truth) acc += 1.0;
    }
    acc /= probs.rows;

    check(last_loss < first_loss && std::isfinite(last_loss),
          "training loop loss decreased");
    check(acc >= 0.95, "training loop accuracy >= 95%");
}

// =======================================================================
// Test 12: Const-correctness of Module API
// =======================================================================
static void test_const_api() {
    std::mt19937 rng(42);
    Sequential model;
    model.add(std::make_unique<Linear>(4, 2, "fc", rng));

    const Sequential& cmodel = model;
    auto const_params = cmodel.parameters();
    auto const_grads = cmodel.gradients();
    check(const_params.size() == 2, "const parameters() returns weights+bias");
    check(const_grads.size() == 2, "const gradients() returns grad_weights+grad_bias");

    const Module& cmod = cmodel.get(0);
    auto mod_params = cmod.parameters();
    check(mod_params.size() == 2, "const get() returns module with const parameters");
}

// =======================================================================
// Test 13: Markov chain classifier basic fit/predict contract
// =======================================================================
static void test_markov_classifier() {
    Tensor x(6, 4);
    // Class 0: mostly dark pixels.
    x(0,0)=0.0; x(0,1)=0.1; x(0,2)=0.0; x(0,3)=0.2;
    x(1,0)=0.1; x(1,1)=0.0; x(1,2)=0.2; x(1,3)=0.1;
    x(2,0)=0.0; x(2,1)=0.0; x(2,2)=0.1; x(2,3)=0.0;
    // Class 1: mostly bright pixels.
    x(3,0)=0.9; x(3,1)=1.0; x(3,2)=0.8; x(3,3)=0.9;
    x(4,0)=1.0; x(4,1)=0.8; x(4,2)=0.9; x(4,3)=1.0;
    x(5,0)=0.8; x(5,1)=0.9; x(5,2)=1.0; x(5,3)=0.8;

    std::vector<int> y = {0, 0, 0, 1, 1, 1};
    MarkovChainClassifier model(2, 2, 1, 0.1);
    model.partial_fit(x, y, 0, x.rows);

    auto preds = model.predict(x);
    int correct = 0;
    for (std::size_t i = 0; i < preds.size(); ++i) {
        if (preds[i] == y[i]) correct++;
    }
    const double acc = static_cast<double>(correct) / static_cast<double>(preds.size());
    check(acc >= 0.95, "markov classifier fits separable toy data");

    auto nll = model.position_nll(x, y);
    bool finite = nll.size() == x.cols;
    for (double v : nll) {
        if (!std::isfinite(v)) finite = false;
    }
    check(finite, "markov position_nll returns finite per-position values");
}

// =======================================================================
// Test 14: Guard rails for shape/size mismatch paths
// =======================================================================
static void test_guard_rails() {
    bool mse_shape_threw = false;
    try {
        Tensor a(2, 2, 1.0);
        Tensor b(2, 3, 1.0);
        MSELoss mse;
        (void)mse.forward(a, b);
    } catch (const std::exception&) {
        mse_shape_threw = true;
    }
    check(mse_shape_threw, "mse shape mismatch throws");

    bool metrics_size_threw = false;
    try {
        std::vector<int> preds = {0, 1, 1};
        std::vector<int> labels = {0, 1};
        (void)Metrics::accuracy(preds, labels);
    } catch (const std::exception&) {
        metrics_size_threw = true;
    }
    check(metrics_size_threw, "metrics size mismatch throws");
}

// =======================================================================
// Test 15: Cell simulator mesh generation sanity
// =======================================================================
static void test_cells_mesh_counts() {
    cells::CellInit init;
    init.icosphere_subdivisions = 0;
    init.radius = 1.0;
    cells::CellSim sim(init);

    check(sim.vertex_count() == 12, "cells icosphere L0 vertex count");
    check(sim.triangle_count() == 20, "cells icosphere L0 triangle count");
    check(sim.edge_count() == 30, "cells icosphere L0 edge count");
}

// =======================================================================
// Test 16: Cell simulator deterministic trajectory and volume stability
// =======================================================================
static void test_cells_determinism_and_volume() {
    cells::CellInit init;
    init.icosphere_subdivisions = 1;
    init.radius = 1.0;

    cells::CellSim a(init);
    cells::CellSim b(init);

    a.poke_vertex(0, cells::Vec3(0.8, 0.0, 0.0));
    b.poke_vertex(0, cells::Vec3(0.8, 0.0, 0.0));

    for (int i = 0; i < 300; ++i) {
        a.step(0.005);
        b.step(0.005);
    }

    const auto& sa = a.state().membrane_positions;
    const auto& sb = b.state().membrane_positions;
    bool same = sa.size() == sb.size();
    for (std::size_t i = 0; i < sa.size() && same; ++i) {
        const double err = std::abs(sa[i].x - sb[i].x) +
                           std::abs(sa[i].y - sb[i].y) +
                           std::abs(sa[i].z - sb[i].z);
        if (err > 1e-12) same = false;
    }
    check(same, "cells deterministic trajectory");

    const auto& d = a.diagnostics();
    check(std::isfinite(d.volume_drift_pct), "cells finite volume drift");
    check(d.volume_drift_pct < 3.0, "cells volume drift < 3% after perturbation");
}

// =======================================================================
int main() {
    std::cout << "=== Core Unit Tests ===" << std::endl;

    test_matmul();
    test_transpose();
    test_elementwise_add();
    test_conv2d_forward();
    test_batchnorm1d();
    test_batchnorm_tiny_batch();
    test_ce_gradient();
    test_softmax_backward_chain_rule();
    test_adam_step();
    test_adam_vector_interface();
    test_replay_buffer();
    test_continuous_replay_buffer();
    test_training_convergence();
    test_const_api();
    test_markov_classifier();
    test_guard_rails();
    test_cells_mesh_counts();
    test_cells_determinism_and_volume();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail > 0 ? 1 : 0;
}
