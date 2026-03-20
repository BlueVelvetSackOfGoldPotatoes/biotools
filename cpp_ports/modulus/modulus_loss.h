// modulus_loss.h - Loss functions, aggregation, and constraint losses
// Port of loss aggregation, weighted losses, integral/point/constraint losses
//
// C++17, no external dependencies.

#ifndef MODULUS_LOSS_H
#define MODULUS_LOSS_H

#include "modulus.h"
#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace modulus {
namespace loss {

// ============================================================
// Loss functions
// ============================================================

// MSE loss
inline double mse_loss(const Tensor& pred, const Tensor& target) {
    Tensor diff = pred - target;
    return diff.norm_sq() / diff.numel();
}

// MAE / L1 loss
inline double l1_loss(const Tensor& pred, const Tensor& target) {
    Tensor diff = pred - target;
    double s = 0;
    for (int i = 0; i < diff.numel(); ++i) s += std::abs(diff.data[i]);
    return s / diff.numel();
}

// Huber / Smooth L1 loss
inline double huber_loss(const Tensor& pred, const Tensor& target, double delta = 1.0) {
    Tensor diff = pred - target;
    double s = 0;
    for (int i = 0; i < diff.numel(); ++i) {
        double a = std::abs(diff.data[i]);
        s += (a < delta) ? 0.5 * a * a : delta * (a - 0.5 * delta);
    }
    return s / diff.numel();
}

// Relative L2 loss: ||pred - target||_2 / ||target||_2
inline double relative_l2_loss(const Tensor& pred, const Tensor& target) {
    Tensor diff = pred - target;
    double tgt_norm = std::sqrt(target.norm_sq());
    if (tgt_norm < 1e-15) return std::sqrt(diff.norm_sq());
    return std::sqrt(diff.norm_sq()) / tgt_norm;
}

// Log-cosh loss
inline double log_cosh_loss(const Tensor& pred, const Tensor& target) {
    Tensor diff = pred - target;
    double s = 0;
    for (int i = 0; i < diff.numel(); ++i)
        s += std::log(std::cosh(diff.data[i]));
    return s / diff.numel();
}

// Gradient of MSE loss: d(MSE)/d(pred)
inline Tensor mse_grad(const Tensor& pred, const Tensor& target) {
    Tensor g(pred.shape);
    for (int i = 0; i < pred.numel(); ++i)
        g.data[i] = 2.0 * (pred.data[i] - target.data[i]) / pred.numel();
    return g;
}

// Gradient of L1 loss
inline Tensor l1_grad(const Tensor& pred, const Tensor& target) {
    Tensor g(pred.shape);
    for (int i = 0; i < pred.numel(); ++i) {
        double diff = pred.data[i] - target.data[i];
        g.data[i] = (diff > 0 ? 1.0 : (diff < 0 ? -1.0 : 0.0)) / pred.numel();
    }
    return g;
}

// ============================================================
// Weighted Loss
// ============================================================
struct WeightedLoss {
    std::string name;
    double weight;
    std::function<double(const Tensor&, const Tensor&)> loss_fn;

    WeightedLoss() : weight(1.0) {}
    WeightedLoss(const std::string& n, double w,
                 std::function<double(const Tensor&, const Tensor&)> fn)
        : name(n), weight(w), loss_fn(std::move(fn)) {}
};

// ============================================================
// Loss Aggregator - combines multiple loss terms
// ============================================================
class LossAggregator {
public:
    std::vector<WeightedLoss> losses;
    std::unordered_map<std::string, double> loss_history; // last computed values
    int step_count;

    LossAggregator() : step_count(0) {}

    void add_loss(const std::string& name, double weight,
                  std::function<double(const Tensor&, const Tensor&)> fn) {
        losses.push_back({name, weight, std::move(fn)});
    }

    void add_mse(const std::string& name, double weight = 1.0) {
        add_loss(name, weight, mse_loss);
    }

    void add_l1(const std::string& name, double weight = 1.0) {
        add_loss(name, weight, l1_loss);
    }

    void add_relative_l2(const std::string& name, double weight = 1.0) {
        add_loss(name, weight, relative_l2_loss);
    }

    // Compute total weighted loss
    double compute(const std::vector<std::pair<Tensor, Tensor>>& pred_target_pairs) {
        double total = 0;
        for (int i = 0; i < (int)losses.size() && i < (int)pred_target_pairs.size(); ++i) {
            auto& [pred, target] = pred_target_pairs[i];
            double val = losses[i].weight * losses[i].loss_fn(pred, target);
            loss_history[losses[i].name] = val;
            total += val;
        }
        step_count++;
        return total;
    }

    // Print loss breakdown
    void print_losses() const {
        std::cout << "  Loss breakdown (step " << step_count << "):" << std::endl;
        for (auto& [name, val] : loss_history)
            std::cout << "    " << name << ": " << std::scientific << val << std::endl;
    }
};

// ============================================================
// Point Constraint Loss
// Enforces model output matches target at specific points
// ============================================================
class PointConstraintLoss {
public:
    Tensor coords;   // (n_points, dim)
    Tensor targets;   // (n_points, out_dim)
    double weight;

    PointConstraintLoss() : weight(1.0) {}

    PointConstraintLoss(const Tensor& c, const Tensor& t, double w = 1.0)
        : coords(c), targets(t), weight(w) {}

    double compute(const Tensor& predictions) const {
        return weight * mse_loss(predictions, targets);
    }
};

// ============================================================
// Integral Constraint Loss
// Enforces integral of output over domain matches target
// Uses trapezoidal rule for numerical integration
// ============================================================
class IntegralConstraintLoss {
public:
    double target_integral;
    double weight;
    double dx; // grid spacing

    IntegralConstraintLoss() : target_integral(0), weight(1.0), dx(1.0) {}

    IntegralConstraintLoss(double target, double w, double grid_spacing)
        : target_integral(target), weight(w), dx(grid_spacing) {}

    double compute(const Tensor& predictions) const {
        // Trapezoidal integration
        double integral = 0;
        for (int i = 0; i < predictions.numel(); ++i) {
            double coeff = (i == 0 || i == predictions.numel() - 1) ? 0.5 : 1.0;
            integral += coeff * predictions.data[i] * dx;
        }
        double diff = integral - target_integral;
        return weight * diff * diff;
    }
};

// ============================================================
// PDE Residual Loss (extended from PhysicsConstraint)
// ============================================================
class PDEResidualLoss {
public:
    using PDEFn = std::function<Tensor(const Tensor& coords, const Tensor& solution)>;
    PDEFn pde_residual;
    double weight;

    PDEResidualLoss() : weight(1.0) {}

    PDEResidualLoss(PDEFn fn, double w = 1.0)
        : pde_residual(std::move(fn)), weight(w) {}

    double compute(const Tensor& coords, const Tensor& solution) const {
        Tensor residual = pde_residual(coords, solution);
        return weight * residual.norm_sq() / residual.numel();
    }
};

// ============================================================
// Boundary Condition Loss
// ============================================================
class BoundaryLoss {
public:
    enum Type { DIRICHLET, NEUMANN };
    Type bc_type;
    Tensor boundary_coords;
    Tensor boundary_values;
    double weight;

    BoundaryLoss() : bc_type(DIRICHLET), weight(1.0) {}

    BoundaryLoss(Type type, const Tensor& coords, const Tensor& values, double w = 1.0)
        : bc_type(type), boundary_coords(coords), boundary_values(values), weight(w) {}

    double compute(const Tensor& predictions) const {
        return weight * mse_loss(predictions, boundary_values);
    }
};

// ============================================================
// Combined Physics-Informed Loss
// Combines data loss, PDE residual, BCs, and constraints
// ============================================================
class PhysicsInformedLoss {
public:
    double data_weight;
    std::vector<PDEResidualLoss> pde_losses;
    std::vector<BoundaryLoss> bc_losses;
    std::vector<PointConstraintLoss> point_constraints;
    std::vector<IntegralConstraintLoss> integral_constraints;

    PhysicsInformedLoss() : data_weight(1.0) {}

    void add_pde(PDEResidualLoss::PDEFn fn, double weight = 1.0) {
        pde_losses.push_back({std::move(fn), weight});
    }

    void add_bc(BoundaryLoss::Type type, const Tensor& coords,
                const Tensor& values, double weight = 1.0) {
        bc_losses.push_back({type, coords, values, weight});
    }

    void add_point_constraint(const Tensor& coords, const Tensor& values, double weight = 1.0) {
        point_constraints.push_back({coords, values, weight});
    }

    void add_integral_constraint(double target, double weight, double dx) {
        integral_constraints.push_back({target, weight, dx});
    }

    struct LossResult {
        double total;
        double data_loss;
        double pde_loss;
        double bc_loss;
        double constraint_loss;
    };

    LossResult compute(const Tensor& pred, const Tensor& target,
                       const Tensor& coords = Tensor()) const {
        LossResult result = {0, 0, 0, 0, 0};

        // Data loss
        result.data_loss = data_weight * mse_loss(pred, target);
        result.total += result.data_loss;

        // PDE residual losses
        for (auto& pde : pde_losses) {
            if (!coords.empty()) {
                double l = pde.compute(coords, pred);
                result.pde_loss += l;
                result.total += l;
            }
        }

        // Boundary losses
        for (auto& bc : bc_losses) {
            double l = bc.compute(pred);
            result.bc_loss += l;
            result.total += l;
        }

        // Point constraints
        for (auto& pc : point_constraints) {
            double l = pc.compute(pred);
            result.constraint_loss += l;
            result.total += l;
        }

        // Integral constraints
        for (auto& ic : integral_constraints) {
            double l = ic.compute(pred);
            result.constraint_loss += l;
            result.total += l;
        }

        return result;
    }
};

} // namespace loss
} // namespace modulus

#endif // MODULUS_LOSS_H
