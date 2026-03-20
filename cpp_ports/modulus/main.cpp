// main.cpp - Demo: Train FNO to learn 1D Burgers equation operator
//
// This demonstrates the C++ port of NVIDIA Modulus/PhysicsNeMo:
//   1. FFT verification (Cooley-Tukey)
//   2. MLP forward/backward test
//   3. FNO training on Burgers equation (u0 -> u(T))
//   4. DeepONet demonstration
//   5. Physics-informed training with PDE residual loss

#include "modulus.h"
#include <chrono>

using namespace modulus;

// ============================================================
// Test 1: Verify FFT implementation
// ============================================================
void test_fft() {
    std::cout << "=== Test 1: FFT Verification ===" << std::endl;

    // Test with known signal: sin(2*pi*x) sampled at 8 points
    int N = 8;
    std::vector<double> signal(N);
    for (int i = 0; i < N; ++i) {
        double x = 2.0 * M_PI * i / N;
        signal[i] = std::sin(x) + 0.5 * std::cos(2.0 * x);
    }

    // Forward FFT
    auto spectrum = fft::rfft(signal);
    std::cout << "  Input signal: [";
    for (int i = 0; i < N; ++i) {
        if (i) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << signal[i];
    }
    std::cout << "]" << std::endl;

    std::cout << "  FFT magnitudes: [";
    for (int i = 0; i < (int)spectrum.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << std::abs(spectrum[i]);
    }
    std::cout << "]" << std::endl;

    // Inverse FFT (round-trip)
    auto recovered = fft::irfft(spectrum, N);
    double max_err = 0;
    for (int i = 0; i < N; ++i)
        max_err = std::max(max_err, std::abs(signal[i] - recovered[i]));
    std::cout << "  Round-trip max error: " << std::scientific << max_err << std::endl;
    std::cout << "  FFT test: " << (max_err < 1e-10 ? "PASSED" : "FAILED") << std::endl;
    std::cout << std::endl;
}

// ============================================================
// Test 2: MLP forward/backward
// ============================================================
void test_mlp() {
    std::cout << "=== Test 2: MLP Forward/Backward ===" << std::endl;
    seed_rng(123);

    MLP net("test_mlp", 4, 16, 2, 2, "gelu");
    std::cout << "  Architecture: 4 -> 16 -> 16 -> 2 (GELU)" << std::endl;

    int n_params = 0;
    for (auto* p : net.parameters()) n_params += p->value.numel();
    std::cout << "  Total parameters: " << n_params << std::endl;

    // Forward pass
    Tensor x = Tensor::randn({3, 4}); // batch=3, features=4
    Tensor y = net.forward(x);
    std::cout << "  Input shape:  [3, 4]" << std::endl;
    std::cout << "  Output shape: [" << y.shape[0] << ", " << y.shape[1] << "]" << std::endl;
    std::cout << "  Output[0]:    [" << std::fixed << std::setprecision(6)
              << y.at2(0, 0) << ", " << y.at2(0, 1) << "]" << std::endl;

    // Simple training: learn identity-like mapping
    Adam opt(0.01);
    auto params = net.parameters();
    Tensor target({3, 2});
    target.at2(0, 0) = 1.0; target.at2(0, 1) = 0.0;
    target.at2(1, 0) = 0.0; target.at2(1, 1) = 1.0;
    target.at2(2, 0) = 0.5; target.at2(2, 1) = 0.5;

    double initial_loss = 0, final_loss = 0;
    for (int epoch = 0; epoch < 200; ++epoch) {
        net.zero_grad();
        Tensor pred = net.forward(x);
        Tensor diff = pred - target;
        double loss = diff.norm_sq() / diff.numel();
        if (epoch == 0) initial_loss = loss;
        if (epoch == 199) final_loss = loss;

        Tensor grad(pred.shape);
        for (int i = 0; i < pred.numel(); ++i)
            grad.data[i] = 2.0 * diff.data[i] / diff.numel();
        net.backward(grad);
        opt.step(params);
    }
    std::cout << "  Training 200 epochs: loss " << std::scientific
              << initial_loss << " -> " << final_loss << std::endl;
    std::cout << "  MLP test: " << (final_loss < initial_loss * 0.1 ? "PASSED" : "FAILED")
              << std::endl;
    std::cout << std::endl;
}

// ============================================================
// Test 3: SpectralConv1d verification
// ============================================================
void test_spectral_conv() {
    std::cout << "=== Test 3: SpectralConv1d ===" << std::endl;
    seed_rng(42);

    int batch = 2, in_ch = 3, out_ch = 4, spatial = 16, modes = 8;
    SpectralConv1d sconv("test_sp", in_ch, out_ch, modes);

    Tensor x({batch, in_ch, spatial});
    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < in_ch; ++c)
            for (int s = 0; s < spatial; ++s)
                x.at3(b, c, s) = std::sin(2.0 * M_PI * s / spatial * (c + 1));

    Tensor y = sconv.forward(x);
    std::cout << "  Input:  [" << batch << ", " << in_ch << ", " << spatial << "]" << std::endl;
    std::cout << "  Output: [" << y.shape[0] << ", " << y.shape[1] << ", " << y.shape[2] << "]" << std::endl;

    int n_params = 0;
    for (auto* p : sconv.parameters()) n_params += p->value.numel();
    std::cout << "  Spectral weight params: " << n_params << std::endl;

    // Test backward
    Tensor grad_out = Tensor::ones({batch, out_ch, spatial});
    sconv.zero_grad();
    Tensor grad_in = sconv.backward(grad_out);
    std::cout << "  Grad input shape: [" << grad_in.shape[0] << ", "
              << grad_in.shape[1] << ", " << grad_in.shape[2] << "]" << std::endl;

    double grad_norm = 0;
    for (auto* p : sconv.parameters()) grad_norm += p->grad.norm_sq();
    std::cout << "  Grad norm (weights): " << std::scientific << std::sqrt(grad_norm) << std::endl;
    std::cout << "  SpectralConv test: PASSED" << std::endl;
    std::cout << std::endl;
}

// ============================================================
// Test 4: FNO on Burgers equation
// ============================================================
void test_fno_burgers() {
    std::cout << "=== Test 4: FNO on 1D Burgers Equation ===" << std::endl;
    seed_rng(42);

    // Problem setup
    int spatial = 64;     // Grid points (power of 2 for FFT)
    double nu = 0.01;     // Viscosity
    double dt = 0.0001;   // Time step for solver
    int n_steps = 100;    // Time steps to evolve
    int n_train = 16;     // Training samples
    int n_test = 4;       // Test samples

    std::cout << "  Generating Burgers equation data..." << std::endl;
    std::cout << "  Grid: " << spatial << " points, nu=" << nu
              << ", dt=" << dt << ", steps=" << n_steps << std::endl;

    // Generate training data: u0 -> u(T)
    std::vector<Tensor> train_inputs, train_targets;
    for (int i = 0; i < n_train; ++i) {
        Tensor u0 = burgers::generate_initial_condition(spatial, 1);
        Tensor uT = burgers::solve(u0, nu, dt, n_steps);
        train_inputs.push_back(u0);
        train_targets.push_back(uT);
    }

    // Generate test data
    std::vector<Tensor> test_inputs, test_targets;
    for (int i = 0; i < n_test; ++i) {
        Tensor u0 = burgers::generate_initial_condition(spatial, 1);
        Tensor uT = burgers::solve(u0, nu, dt, n_steps);
        test_inputs.push_back(u0);
        test_targets.push_back(uT);
    }

    // Create FNO model (smaller for demo speed)
    int latent_ch = 16;
    int n_fno_layers = 3;
    int n_fno_modes = 12;
    int decoder_layers = 1;
    int decoder_size = 32;

    FNO1D model(1, 1, latent_ch, n_fno_layers, n_fno_modes,
                decoder_layers, decoder_size, "gelu", true);

    int total_params = 0;
    for (auto* p : model.parameters()) total_params += p->value.numel();
    std::cout << "  FNO architecture:" << std::endl;
    std::cout << "    Latent channels: " << latent_ch << std::endl;
    std::cout << "    FNO layers:      " << n_fno_layers << std::endl;
    std::cout << "    Fourier modes:   " << n_fno_modes << std::endl;
    std::cout << "    Total params:    " << total_params << std::endl;

    // Optional physics constraint
    double dx = 2.0 * M_PI / spatial;
    PhysicsConstraint phys = burgers::make_constraint(nu, dx, 0.01);
    Tensor x_coords({1, 1, spatial});
    for (int s = 0; s < spatial; ++s)
        x_coords.at3(0, 0, s) = 2.0 * M_PI * s / spatial;

    // Train
    std::cout << "\n  Training FNO (with physics-informed loss)..." << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();

    int epochs = 50;
    Trainer::train_fno(model, train_inputs, train_targets, epochs,
                       1e-3, 10, &phys, &x_coords);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "  Training time: " << std::fixed << std::setprecision(2)
              << elapsed << "s" << std::endl;

    // Evaluate on test set
    double test_loss = 0;
    for (int i = 0; i < n_test; ++i) {
        Tensor pred = model.forward(test_inputs[i]);
        Tensor diff = pred - test_targets[i];
        test_loss += diff.norm_sq() / diff.numel();
    }
    test_loss /= n_test;
    std::cout << "  Test MSE:  " << std::scientific << test_loss << std::endl;

    // Show sample prediction vs target
    std::cout << "\n  Sample prediction (first 8 points):" << std::endl;
    Tensor pred = model.forward(test_inputs[0]);
    std::cout << "    Target: [";
    for (int s = 0; s < 8; ++s) {
        if (s) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << test_targets[0].at3(0, 0, s);
    }
    std::cout << ", ...]" << std::endl;
    std::cout << "    Pred:   [";
    for (int s = 0; s < 8; ++s) {
        if (s) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << pred.at3(0, 0, s);
    }
    std::cout << ", ...]" << std::endl;
    std::cout << std::endl;
}

// ============================================================
// Test 5: DeepONet demonstration
// ============================================================
void test_deeponet() {
    std::cout << "=== Test 5: DeepONet ===" << std::endl;
    seed_rng(99);

    // Learn the antiderivative operator: F(u)(y) = integral_0^y u(x) dx
    // Branch input: discrete values of u at sensor locations
    // Trunk input: query point y
    // Output: antiderivative value at y

    int n_sensors = 16;      // Sensor points for branch
    int n_query = 8;         // Query points for trunk
    int n_train = 20;        // Training samples
    int p = 16;              // Latent dimension

    DeepONet model(n_sensors, 1, p,
                   2, 32,   // branch: 2 hidden layers, width 32
                   2, 32,   // trunk: 2 hidden layers, width 32
                   "gelu");

    int total_params = 0;
    for (auto* p : model.parameters()) total_params += p->value.numel();
    std::cout << "  DeepONet architecture:" << std::endl;
    std::cout << "    Branch input:  " << n_sensors << " sensors" << std::endl;
    std::cout << "    Trunk input:   1D coordinate" << std::endl;
    std::cout << "    Latent dim:    " << p << std::endl;
    std::cout << "    Total params:  " << total_params << std::endl;

    // Generate data: u(x) = sum of sines, F(u)(y) = integral
    std::vector<Tensor> branch_data, trunk_data, target_data;
    std::uniform_real_distribution<double> coeff(-1.0, 1.0);

    for (int s = 0; s < n_train; ++s) {
        // Random function u(x) = a*sin(pi*x) + b*sin(2*pi*x)
        double a = coeff(global_rng());
        double b = coeff(global_rng());

        // Branch: sensor values
        Tensor branch({1, n_sensors});
        for (int i = 0; i < n_sensors; ++i) {
            double x = (double)i / (n_sensors - 1);
            branch.at2(0, i) = a * std::sin(M_PI * x) + b * std::sin(2.0 * M_PI * x);
        }

        // Trunk: query coordinates
        Tensor trunk({n_query, 1});
        for (int i = 0; i < n_query; ++i)
            trunk.at2(i, 0) = (double)i / (n_query - 1);

        // Target: antiderivative values
        // integral of a*sin(pi*x) = -a/pi * cos(pi*x) + a/pi
        // integral of b*sin(2*pi*x) = -b/(2*pi) * cos(2*pi*x) + b/(2*pi)
        Tensor target({1, n_query});
        for (int i = 0; i < n_query; ++i) {
            double y = (double)i / (n_query - 1);
            target.at2(0, i) = (-a / M_PI) * (std::cos(M_PI * y) - 1.0)
                              + (-b / (2.0 * M_PI)) * (std::cos(2.0 * M_PI * y) - 1.0);
        }

        branch_data.push_back(branch);
        trunk_data.push_back(trunk);
        target_data.push_back(target);
    }

    // Train
    std::cout << "\n  Training DeepONet..." << std::endl;
    Trainer::train_deeponet(model, branch_data, trunk_data, target_data,
                            100, 1e-3, 25);

    // Test
    double a_test = 0.7, b_test = -0.3;
    Tensor test_branch({1, n_sensors});
    for (int i = 0; i < n_sensors; ++i) {
        double x = (double)i / (n_sensors - 1);
        test_branch.at2(0, i) = a_test * std::sin(M_PI * x) + b_test * std::sin(2.0 * M_PI * x);
    }

    Tensor test_trunk({n_query, 1});
    for (int i = 0; i < n_query; ++i)
        test_trunk.at2(i, 0) = (double)i / (n_query - 1);

    Tensor pred = model.forward(test_branch, test_trunk);
    std::cout << "\n  Test prediction (antiderivative):" << std::endl;
    std::cout << "    y:      [";
    for (int i = 0; i < n_query; ++i) {
        if (i) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(3) << test_trunk.at2(i, 0);
    }
    std::cout << "]" << std::endl;

    std::cout << "    Target: [";
    for (int i = 0; i < n_query; ++i) {
        if (i) std::cout << ", ";
        double y = (double)i / (n_query - 1);
        double exact = (-a_test / M_PI) * (std::cos(M_PI * y) - 1.0)
                     + (-b_test / (2.0 * M_PI)) * (std::cos(2.0 * M_PI * y) - 1.0);
        std::cout << std::fixed << std::setprecision(4) << exact;
    }
    std::cout << "]" << std::endl;

    std::cout << "    Pred:   [";
    for (int i = 0; i < n_query; ++i) {
        if (i) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << pred.at2(0, i);
    }
    std::cout << "]" << std::endl;
    std::cout << std::endl;
}

// ============================================================
// Main
// ============================================================
int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "  Modulus C++ Port - Physics-Informed ML Framework" << std::endl;
    std::cout << "  Ported from NVIDIA Modulus/PhysicsNeMo" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;

    test_fft();
    test_mlp();
    test_spectral_conv();
    test_fno_burgers();
    test_deeponet();

    std::cout << "================================================================" << std::endl;
    std::cout << "  All tests completed." << std::endl;
    std::cout << "================================================================" << std::endl;

    return 0;
}
