// main.cpp - Demo: PINN solving the 1D heat equation
//
// PDE: u_t = k * u_xx   on x in [0, 1], t in [0, 1]
//
// Boundary conditions:
//   u(0, t) = 0   (Dirichlet, left)
//   u(1, t) = 0   (Dirichlet, right)
//
// Initial condition:
//   u(x, 0) = sin(pi * x)
//
// Exact solution:
//   u(x, t) = sin(pi * x) * exp(-k * pi^2 * t)
//
// This mirrors a typical DeepXDE TimePDE example.

#include "deepxde.h"
#include <cmath>
#include <iostream>
#include <iomanip>

int main() {
    using namespace deepxde;

    const double k_diff = 0.01;  // thermal diffusivity
    const double PI = 3.14159265358979323846;

    auto exact_solution = [&](double x, double t) -> double {
        return std::sin(PI * x) * std::exp(-k_diff * PI * PI * t);
    };

    // ---- Geometry: [0, 1] x [0, 1] ----
    auto spatial = std::make_shared<Interval>(0.0, 1.0);
    auto temporal = std::make_shared<TimeDomain>(0.0, 1.0);
    auto geom_time = std::make_shared<GeometryXTime>(spatial, temporal);

    // ---- Neural network: 2 -> 16 -> 16 -> 1 ----
    FNN net({2, 16, 16, 1}, Activation::Tanh);

    std::cout << "=== DeepXDE C++ Port: 1D Heat Equation PINN ===\n";
    std::cout << "PDE: u_t = " << k_diff << " * u_xx\n";
    std::cout << "Domain: x in [0,1], t in [0,1]\n";
    std::cout << "BC: u(0,t) = u(1,t) = 0\n";
    std::cout << "IC: u(x,0) = sin(pi*x)\n";
    std::cout << "Exact: u(x,t) = sin(pi*x) * exp(-" << k_diff << "*pi^2*t)\n";
    std::cout << "Network: [2, 16, 16, 1], tanh activation\n";
    std::cout << "Trainable parameters: " << net.num_trainable_parameters() << "\n\n";

    // ---- PDE residual: u_t - k * u_xx = 0 ----
    // Uses AD backward for first derivatives and finite diff on the AD gradient for u_xx
    auto pde_residual = [&net, k_diff](const std::vector<VarPtr>& x_vars,
                                        const std::vector<VarPtr>& u_vars) -> VarPtr {
        // First backward pass gives u_t and u_x
        u_vars[0]->backward();
        double u_t = x_vars[1]->grad;  // du/dt

        // u_xx via central differences on du/dx
        const double h = 1e-4;
        double x_val = x_vars[0]->val;
        double t_val = x_vars[1]->val;

        // du/dx at x+h
        std::vector<VarPtr> xp = {make_var(x_val + h), make_var(t_val)};
        auto up = net.forward_ad(xp);
        up[0]->backward();
        double du_dx_p = xp[0]->grad;

        // du/dx at x-h
        std::vector<VarPtr> xm = {make_var(x_val - h), make_var(t_val)};
        auto um = net.forward_ad(xm);
        um[0]->backward();
        double du_dx_m = xm[0]->grad;

        double u_xx = (du_dx_p - du_dx_m) / (2.0 * h);

        return make_var(u_t - k_diff * u_xx);
    };

    // ---- Boundary conditions ----
    auto bc_left = std::make_shared<DirichletBC>(
        std::static_pointer_cast<Geometry>(geom_time),
        [](const std::vector<double>&) { return 0.0; },
        [](const std::vector<double>& x, bool on_bdy) {
            return on_bdy && std::fabs(x[0]) < 1e-12;
        }
    );

    auto bc_right = std::make_shared<DirichletBC>(
        std::static_pointer_cast<Geometry>(geom_time),
        [](const std::vector<double>&) { return 0.0; },
        [](const std::vector<double>& x, bool on_bdy) {
            return on_bdy && std::fabs(x[0] - 1.0) < 1e-12;
        }
    );

    // ---- Initial condition ----
    auto ic = std::make_shared<IC>(
        geom_time,
        [PI](const std::vector<double>& x) { return std::sin(PI * x[0]); }
    );

    // ---- Build model ----
    Model model;
    model.set_network(net);
    model.set_geometry_time(geom_time);
    model.set_pde(pde_residual);
    model.add_bc(bc_left);
    model.add_bc(bc_right);
    model.add_ic(ic);

    // Keep small for fast execution
    model.num_domain   = 30;
    model.num_boundary = 20;
    model.num_initial  = 20;
    model.weight_pde = 1.0;
    model.weight_bc  = 10.0;
    model.weight_ic  = 10.0;

    model.compile(1e-3);

    std::cout << "Training with Adam optimizer (lr=1e-3)...\n";
    std::cout << "Points: " << model.num_domain << " domain, "
              << model.num_boundary << " boundary, "
              << model.num_initial << " initial\n\n";

    auto history = model.train(
        /*epochs=*/500,
        /*display_every=*/50,
        /*resample=*/true,
        /*resample_every=*/200
    );

    // ---- Evaluate accuracy ----
    std::cout << "\n=== Evaluation on test grid ===\n";
    std::cout << std::setw(8) << "x" << std::setw(8) << "t"
              << std::setw(14) << "u_pred" << std::setw(14) << "u_exact"
              << std::setw(14) << "error" << "\n";
    std::cout << std::string(62, '-') << "\n";

    double max_err = 0;
    double l2_err = 0;
    double l2_exact = 0;
    int n_test = 0;

    for (double x = 0.0; x <= 1.001; x += 0.25) {
        for (double t = 0.0; t <= 1.001; t += 0.25) {
            double xc = std::min(x, 1.0);
            double tc = std::min(t, 1.0);
            std::vector<double> pt = {xc, tc};
            auto pred = model.predict(pt);
            double u_pred = pred[0];
            double u_exact = exact_solution(xc, tc);
            double err = std::fabs(u_pred - u_exact);
            max_err = std::max(max_err, err);
            l2_err += (u_pred - u_exact) * (u_pred - u_exact);
            l2_exact += u_exact * u_exact;
            n_test++;

            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(8) << xc
                      << std::setw(8) << tc
                      << std::setprecision(6)
                      << std::setw(14) << u_pred
                      << std::setw(14) << u_exact
                      << std::scientific << std::setprecision(2)
                      << std::setw(14) << err
                      << std::defaultfloat << "\n";
        }
    }

    double rel_l2 = std::sqrt(l2_err / (l2_exact + 1e-30));
    std::cout << "\nMax absolute error: " << std::scientific << max_err << "\n";
    std::cout << "Relative L2 error:  " << rel_l2 << "\n";
    std::cout << "Test points:        " << n_test << "\n";

    std::cout << "\n=== Training Summary ===\n";
    std::cout << "Final total loss: " << std::scientific << history.back().loss_total << "\n";
    std::cout << "Final PDE loss:   " << history.back().loss_pde << "\n";
    std::cout << "Final BC loss:    " << history.back().loss_bc << "\n";
    std::cout << "Final IC loss:    " << history.back().loss_ic << "\n";

    return 0;
}
