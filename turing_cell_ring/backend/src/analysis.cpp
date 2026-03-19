#include "turing/analysis.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace turing {

namespace {

SimulationConfig with_gamma(const SimulationConfig& config, double gamma) {
    auto adjusted = config;
    adjusted.schedule.id = "analysis_constant_gamma";
    adjusted.schedule.name = "Analysis constant gamma";
    adjusted.schedule.segments = {{0.0, std::max(1.0, config.total_time), gamma, gamma}};
    adjusted.schedule.post_segment_value = gamma;
    return adjusted;
}

double max_growth(
    const PaperModel& model,
    const SimulationConfig& config,
    double gamma) {
    const auto analysis = LinearStabilityAnalyzer{}.analyze(model, with_gamma(config, gamma), gamma);
    return analysis.dominant_growth;
}

Eigen::MatrixXd to_eigen_matrix(const std::vector<std::vector<double>>& values) {
    if (values.empty()) {
        return Eigen::MatrixXd{};
    }
    Eigen::MatrixXd matrix(values.size(), values.front().size());
    for (std::size_t row = 0; row < values.size(); ++row) {
        for (std::size_t col = 0; col < values[row].size(); ++col) {
            matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) = values[row][col];
        }
    }
    return matrix;
}

}  // namespace

StabilityAnalysis LinearStabilityAnalyzer::analyze(
    const PaperModel& model,
    const SimulationConfig& config,
    double gamma) const {
    StabilityAnalysis analysis;
    analysis.gamma = gamma;
    analysis.family_id = config.family_id;
    analysis.analysis_mode = config.analysis_mode;
    const auto& family = model.require_family(config.family_id);
    analysis.species_order = family.species_order;

    const auto local_config = with_gamma(config, gamma);
    analysis.equilibrium = model.solve_equilibrium(local_config);
    analysis.jacobian = model.jacobian(local_config, analysis.equilibrium);

    analysis.modes.reserve(config.ring.cell_count / 2 + 1);

    double best_growth = -std::numeric_limits<double>::infinity();
    for (std::size_t mode = 0; mode <= config.ring.cell_count / 2; ++mode) {
        const auto matrix = model.mode_matrix(local_config, mode, gamma);
        const auto eigen_matrix = to_eigen_matrix(matrix);
        if (eigen_matrix.rows() == 0) {
            continue;
        }

        Eigen::EigenSolver<Eigen::MatrixXd> solver(eigen_matrix, false);
        const auto eigenvalues = solver.eigenvalues();
        auto best_index = Eigen::Index{0};
        auto dominant = eigenvalues[0];
        for (Eigen::Index index = 1; index < eigenvalues.size(); ++index) {
            if (eigenvalues[index].real() > dominant.real()) {
                dominant = eigenvalues[index];
                best_index = index;
            }
        }
        (void)best_index;

        ModeGrowth mode_growth;
        mode_growth.mode = mode;
        mode_growth.real = dominant.real();
        mode_growth.imag = dominant.imag();
        mode_growth.oscillatory = std::abs(mode_growth.imag) > 1.0e-12;
        mode_growth.wavelength_cells =
            mode == 0 ? 0.0 : static_cast<double>(config.ring.cell_count) / static_cast<double>(mode);
        mode_growth.frequency_cycles_per_time = mode_growth.imag / (2.0 * M_PI);
        mode_growth.phase_velocity_cells_per_time =
            mode_growth.wavelength_cells * mode_growth.frequency_cycles_per_time;
        analysis.modes.push_back(mode_growth);

        if (mode_growth.real > best_growth) {
            best_growth = mode_growth.real;
            analysis.dominant_mode = mode;
            analysis.dominant_growth = mode_growth.real;
            analysis.dominant_frequency_cycles_per_time = mode_growth.frequency_cycles_per_time;
            analysis.dominant_phase_velocity_cells_per_time = mode_growth.phase_velocity_cells_per_time;
            analysis.dominant_classification =
                mode_growth.oscillatory ? "oscillatory" : "stationary";
            analysis.has_instability = mode_growth.real > 0.0;
        }
    }

    if (analysis.modes.size() > 3) {
        analysis.mode3_growth = analysis.modes[3].real;
    }
    if (analysis.modes.size() > 4) {
        analysis.mode4_growth = analysis.modes[4].real;
    }
    analysis.mode34_gap = analysis.mode3_growth - analysis.mode4_growth;
    if (analysis.mode34_gap > 0.0) {
        analysis.mode34_efold_time_special_units = 1.0 / analysis.mode34_gap;
        analysis.mode34_efold_time_hours =
            analysis.mode34_efold_time_special_units * config.ring.time_unit_seconds / 3600.0;
    }

    if (family.kind == FamilyKind::section10_example1
        && config.schedule.id != "analysis_constant_gamma") {
        analysis.threshold_gamma = find_threshold_gamma(model, config, -0.25, 0.25);
    }
    return analysis;
}

double LinearStabilityAnalyzer::find_threshold_gamma(
    const PaperModel& model,
    const SimulationConfig& config,
    double gamma_low,
    double gamma_high) const {
    if (model.require_family(config.family_id).kind != FamilyKind::section10_example1) {
        return 0.0;
    }

    auto low = gamma_low;
    auto high = gamma_high;
    auto low_value = max_growth(model, config, low);
    auto high_value = max_growth(model, config, high);

    if (low_value > 0.0) {
        return low;
    }
    if (high_value < 0.0) {
        return high;
    }

    for (int iteration = 0; iteration < 80; ++iteration) {
        const auto mid = 0.5 * (low + high);
        const auto mid_value = max_growth(model, config, mid);
        if (mid_value > 0.0) {
            high = mid;
        } else {
            low = mid;
        }
        if (std::abs(high - low) < 1.0e-12 || std::abs(mid_value) < 1.0e-12) {
            break;
        }
    }
    return 0.5 * (low + high);
}

}  // namespace turing
