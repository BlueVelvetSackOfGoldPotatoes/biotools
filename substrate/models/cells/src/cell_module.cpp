#include "models/cells/src/cell_module.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace cells {
namespace {

inline double sigmoid(double x) {
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(x);
    return z / (1.0 + z);
}

inline double clamp01(double x) {
    return std::max(0.0, std::min(1.0, x));
}

inline double compute_mse(CellModule module, const std::vector<double>& inputs,
                          const std::vector<double>& targets, double dt) {
    if (inputs.empty()) return 0.0;
    double se = 0.0;
    const std::size_t n = std::min(inputs.size(), targets.size());
    for (std::size_t i = 0; i < n; ++i) {
        const double y = module.step(inputs[i], dt);
        const double e = y - targets[i];
        se += e * e;
    }
    return se / static_cast<double>(n);
}

} // namespace

CellModule::CellModule(const CellModuleParams& params, unsigned int seed)
    : params_(params), rng_(seed) {
    state_.E = params_.energy_max;
}

double CellModule::step(double input, double dt) {
    const double dte = dt / std::max(1e-6, params_.tau_e);
    const double dtb = dt / std::max(1e-6, params_.tau_b);
    const double dts = dt / std::max(1e-6, params_.tau_s);

    state_.e += dte * (-state_.e + params_.gain_input * input + params_.coupling_b * state_.b - params_.coupling_s * state_.s);
    state_.b += dtb * (-state_.b + params_.b_drive * std::tanh(state_.e));
    state_.s += dts * (-state_.s + params_.s_drive * state_.b);

    const double base = sigmoid(state_.e);
    const double gain = clamp01(state_.E / std::max(1e-8, params_.energy_max));

    std::normal_distribution<double> nd(0.0, 1.0);
    const double noise = params_.noise_scale * (1.0 - gain) * nd(rng_);

    state_.output = std::max(0.0, std::min(1.0, base * gain + noise));

    state_.E += dt * (params_.energy_replenish - params_.energy_cost * std::abs(state_.output));
    state_.E = std::max(0.0, std::min(params_.energy_max, state_.E));

    const double activity_dev = std::abs(state_.output - params_.activity_target);
    const double over = std::max(0.0, activity_dev - params_.activity_tol);
    const double e_floor = 0.2 * params_.energy_max;
    const double e_over = std::max(0.0, e_floor - state_.E);
    state_.H = params_.homeostasis_weight * (over * over + e_over * e_over);

    return state_.output;
}

void CellModule::reset() {
    state_ = CellModuleState{};
    state_.E = params_.energy_max;
}

CellModuleBank::CellModuleBank(std::size_t n, const CellModuleParams& params, unsigned int seed) {
    modules_.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        modules_.emplace_back(params, seed + static_cast<unsigned int>(i * 17 + 1));
    }
}

std::vector<double> CellModuleBank::step(const std::vector<double>& inputs, double dt) {
    std::vector<double> out(modules_.size(), 0.0);
    const std::size_t n = std::min(modules_.size(), inputs.size());
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = modules_[i].step(inputs[i], dt);
    }
    for (std::size_t i = n; i < modules_.size(); ++i) {
        out[i] = modules_[i].step(0.0, dt);
    }
    return out;
}

void CellModuleBank::reset() {
    for (CellModule& m : modules_) {
        m.reset();
    }
}

double CellModuleBank::mean_energy() const {
    if (modules_.empty()) return 0.0;
    double s = 0.0;
    for (const CellModule& m : modules_) s += m.state().E;
    return s / static_cast<double>(modules_.size());
}

double CellModuleBank::mean_homeostasis() const {
    if (modules_.empty()) return 0.0;
    double s = 0.0;
    for (const CellModule& m : modules_) s += m.state().H;
    return s / static_cast<double>(modules_.size());
}

CalibrationResult calibrate_module_gain_bias(
    CellModuleParams& params,
    const std::vector<double>& inputs,
    const std::vector<double>& targets,
    double dt,
    unsigned int seed) {
    CalibrationResult out;
    if (inputs.empty() || targets.empty()) {
        out.fitted_gain = params.gain_input;
        return out;
    }

    const std::size_t n = std::min(inputs.size(), targets.size());
    double sx = 0.0;
    double sy = 0.0;
    double sxx = 0.0;
    double sxy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sx += inputs[i];
        sy += targets[i];
        sxx += inputs[i] * inputs[i];
        sxy += inputs[i] * targets[i];
    }

    const double denom = std::max(1e-8, n * sxx - sx * sx);
    const double gain = (n * sxy - sx * sy) / denom;
    const double bias = (sy - gain * sx) / static_cast<double>(n);

    CellModuleParams before = params;
    CellModule m_before(before, seed);
    out.mse_before = compute_mse(m_before, inputs, targets, dt);

    params.gain_input = gain;
    params.coupling_b += 0.05 * bias;

    CellModule m_after(params, seed);
    out.mse_after = compute_mse(m_after, inputs, targets, dt);
    out.fitted_gain = gain;
    out.fitted_bias = bias;
    return out;
}

} // namespace cells

