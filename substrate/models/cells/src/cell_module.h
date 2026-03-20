#pragma once

#include <cstddef>
#include <random>
#include <vector>

namespace cells {

struct CellModuleParams {
    double tau_e = 0.05;
    double tau_b = 0.5;
    double tau_s = 5.0;

    double gain_input = 1.0;
    double coupling_b = 0.15;
    double coupling_s = 0.05;
    double b_drive = 0.4;
    double s_drive = 0.1;

    double energy_max = 1.0;
    double energy_replenish = 0.01;
    double energy_cost = 0.02;

    double activity_target = 0.2;
    double activity_tol = 0.12;
    double homeostasis_weight = 1.0;

    double noise_scale = 0.02;
};

struct CellModuleState {
    double e = 0.0;
    double b = 0.0;
    double s = 0.0;
    double E = 1.0;
    double H = 0.0;
    double output = 0.0;
};

class CellModule {
public:
    explicit CellModule(const CellModuleParams& params = CellModuleParams{}, unsigned int seed = 42);

    double step(double input, double dt);
    void reset();

    const CellModuleState& state() const { return state_; }
    CellModuleState& state() { return state_; }

    const CellModuleParams& params() const { return params_; }
    CellModuleParams& params() { return params_; }

private:
    CellModuleParams params_;
    CellModuleState state_;
    std::mt19937 rng_;
};

class CellModuleBank {
public:
    CellModuleBank(std::size_t n, const CellModuleParams& params = CellModuleParams{}, unsigned int seed = 42);

    std::vector<double> step(const std::vector<double>& inputs, double dt);
    void reset();

    std::size_t size() const { return modules_.size(); }
    const CellModule& module(std::size_t i) const { return modules_[i]; }
    CellModule& module(std::size_t i) { return modules_[i]; }

    double mean_energy() const;
    double mean_homeostasis() const;

private:
    std::vector<CellModule> modules_;
};

struct CalibrationResult {
    double mse_before = 0.0;
    double mse_after = 0.0;
    double fitted_gain = 1.0;
    double fitted_bias = 0.0;
};

CalibrationResult calibrate_module_gain_bias(
    CellModuleParams& params,
    const std::vector<double>& inputs,
    const std::vector<double>& targets,
    double dt,
    unsigned int seed = 42);

} // namespace cells

