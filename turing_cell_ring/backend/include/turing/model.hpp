#pragma once

#include "turing/types.hpp"

#include <optional>
#include <string>
#include <vector>

namespace turing {

enum class FamilyKind {
    section10_example1,
    example2_table2,
    oscillatory_case_e,
    oscillatory_case_f,
};

struct ModelFamily {
    std::string id;
    std::string name;
    std::string description;
    std::string analysis_mode;
    FamilyKind kind = FamilyKind::section10_example1;
    std::vector<std::string> species_order;
    std::vector<double> diffusion_by_species;
    std::vector<std::vector<double>> linear_matrix;
    std::size_t primary_species_index = 0;
    bool supports_full_chemistry = false;
    bool supports_historical_execution = false;
    double mode_u_scale = 1.0;
    std::vector<ReferencePattern> reference_patterns;
};

struct CellChannels {
    double source_x = 0.0;
    double xy_to_yy = 0.0;
    double xx_to_yy = 0.0;
    double y_to_x = 0.0;
    double sink_y = 0.0;
};

struct CellReaction {
    double dx = 0.0;
    double dy = 0.0;
    CellChannels channels;
};

class PaperModel {
public:
    PaperModel();

    const ModelFamily* find_family(const std::string& family_id) const;
    const ModelFamily& require_family(const std::string& family_id) const;
    const std::vector<ModelFamily>& families() const;

    Equilibrium solve_equilibrium(const SimulationConfig& config) const;
    Jacobian jacobian(const SimulationConfig& config, const Equilibrium& equilibrium) const;
    CellReaction evaluate_section10_cell(double x, double y, double gamma) const;
    std::vector<double> evaluate_generic_cell(
        const SimulationConfig& config,
        const std::vector<double>& cell_species,
        double gamma) const;
    std::vector<ReferencePattern> reference_patterns(const SimulationConfig& config) const;
    std::vector<NamedSeries> reference_species_for_preset(const SimulationConfig& config) const;
    bool supports_full_chemistry(const SimulationConfig& config) const;
    bool supports_historical_execution(const SimulationConfig& config) const;
    std::size_t primary_species_index(const SimulationConfig& config) const;
    double mode_u_value(const SimulationConfig& config, std::size_t mode) const;
    std::vector<std::vector<double>> mode_matrix(
        const SimulationConfig& config,
        std::size_t mode,
        double gamma) const;

private:
    std::vector<ModelFamily> families_;
};

class PresetRegistry {
public:
    PresetRegistry();

    const std::vector<Preset>& presets() const;
    const Preset* find(const std::string& preset_id) const;
    const std::vector<ModelFamily>& families() const;
    const ModelFamily* find_family(const std::string& family_id) const;

private:
    PaperModel model_;
    std::vector<Preset> presets_;
};

ControlSchedule quick_cooking_schedule();
ControlSchedule slow_cooking_schedule();
ControlSchedule zero_schedule();

}  // namespace turing
