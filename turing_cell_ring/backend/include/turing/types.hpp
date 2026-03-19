#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace turing {

struct NamedValue {
    std::string name;
    double value = 0.0;
};

struct NamedSeries {
    std::string name;
    std::vector<double> values;
};

struct RingConfig {
    std::size_t cell_count = 20;
    double cell_diameter_cm = 0.01;
    double diffusion_x_special = 0.5;
    double diffusion_y_special = 0.25;
    double time_unit_seconds = 1000.0;
    double concentration_unit_mol_per_cm3 = 1.0e-11;
};

struct ControlSegment {
    double start_time = 0.0;
    double end_time = 0.0;
    double start_value = 0.0;
    double end_value = 0.0;
};

struct ControlSchedule {
    std::string id;
    std::string name;
    std::vector<ControlSegment> segments;
    double post_segment_value = 0.0;

    double value_at(double time) const;
    double start_value() const;
    double end_time() const;
};

struct SimulationConfig {
    std::string preset_id = "paper-quick";
    std::string family_id = "section10_example1";
    std::string engine = "reduced_xy";
    std::string analysis_mode = "stationary_two_species";
    std::string execution_mode = "modern";
    std::string execution_profile_id = "modern_default";
    std::string incipient_capture_mode = "gamma_threshold";
    std::vector<std::string> species_order;
    std::vector<NamedValue> family_parameters;
    RingConfig ring;
    ControlSchedule schedule;
    double dt = 0.01;
    double total_time = 80.0;
    bool enable_noise = true;
    double noise_scale = 1.0;
    double initial_perturbation_scale = 1.0e-4;
    std::uint64_t seed = 1;
    std::size_t capture_stride = 25;
    std::size_t max_mode = 10;
    double incipient_capture_gamma = 1.0 / 16.0;
    double incipient_capture_start_time = 0.0;
    double incipient_mode234_threshold = 0.0;
};

struct Equilibrium {
    double x = 1.0;
    double y = 1.0;
    std::vector<std::string> species_order;
    std::vector<double> species_values;
};

struct Jacobian {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double d = 0.0;
    std::vector<std::vector<double>> matrix;
};

struct ModeGrowth {
    std::size_t mode = 0;
    double real = 0.0;
    double imag = 0.0;
    bool oscillatory = false;
    double wavelength_cells = 0.0;
    double frequency_cycles_per_time = 0.0;
    double phase_velocity_cells_per_time = 0.0;
};

struct StabilityAnalysis {
    double gamma = 0.0;
    std::string family_id;
    std::string analysis_mode = "stationary_two_species";
    std::vector<std::string> species_order;
    Equilibrium equilibrium;
    Jacobian jacobian;
    std::size_t dominant_mode = 0;
    double dominant_growth = 0.0;
    double dominant_frequency_cycles_per_time = 0.0;
    double dominant_phase_velocity_cells_per_time = 0.0;
    std::string dominant_classification = "stationary";
    double mode3_growth = 0.0;
    double mode4_growth = 0.0;
    double mode34_gap = 0.0;
    double mode34_efold_time_special_units = 0.0;
    double mode34_efold_time_hours = 0.0;
    double threshold_gamma = 0.0;
    bool has_instability = false;
    std::vector<ModeGrowth> modes;
};

struct ChemistrySnapshot {
    double a_reservoir = 1000.0;
    double c_total = 0.0;
    std::vector<double> b;
    std::vector<double> c;
    std::vector<double> c_prime;
    std::vector<double> w;
};

struct ModeTraceSample {
    double time = 0.0;
    double gamma = 0.0;
    std::size_t dominant_mode = 0;
    double dominant_phase_radians = 0.0;
    std::vector<double> amplitudes;
    std::vector<NamedValue> species_phases;
};

struct Snapshot {
    double time = 0.0;
    double gamma = 0.0;
    std::vector<std::string> species_order;
    std::vector<NamedSeries> species;
    std::vector<NamedSeries> species_mode_amplitudes;
    std::vector<double> primary_mode_amplitudes;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> y_mode_amplitudes;
    std::optional<ChemistrySnapshot> chemistry;
};

struct ReferencePattern {
    std::string id;
    std::string label;
    std::vector<NamedSeries> species;
    std::optional<std::vector<double>> x;
    std::optional<std::vector<double>> y;
};

struct ReferenceComparison {
    std::string reference_id;
    std::string label;
    int best_shift = 0;
    double rmse_x = -1.0;
    double rmse_y = -1.0;
    double rmse_combined = -1.0;
};

struct SimulationMetrics {
    std::size_t dominant_mode = 0;
    std::string dominant_classification = "stationary";
    double dominant_amplitude = 0.0;
    double mode3_amplitude = 0.0;
    double mode4_amplitude = 0.0;
    double mode3_to_mode4_ratio = 0.0;
    double regularity_index = 0.0;
    double dominant_frequency_cycles_per_time = 0.0;
    double dominant_phase_velocity_cells_per_time = 0.0;
    double neighbor_phase_offset_radians = 0.0;
    double travelling_consistency = 0.0;
    std::vector<ReferenceComparison> comparisons;
};

struct ArrestDiagnostics {
    bool observed = false;
    double first_y_zero_time = -1.0;
    std::vector<std::size_t> first_y_zero_cells;
    std::size_t final_zero_cell_count = 0;
};

struct SimulationResult {
    SimulationConfig config;
    StabilityAnalysis initial_analysis;
    StabilityAnalysis incipient_analysis;
    Snapshot initial_snapshot;
    Snapshot incipient_snapshot;
    Snapshot final_snapshot;
    std::vector<ModeTraceSample> mode_trace;
    std::vector<Snapshot> snapshot_trace;
    SimulationMetrics metrics;
    ArrestDiagnostics arrest;
};

struct BatchRunSummary {
    std::uint64_t seed = 0;
    std::size_t dominant_mode = 0;
    std::string dominant_classification = "stationary";
    double dominant_amplitude = 0.0;
    double regularity_index = 0.0;
    double mode3_to_mode4_ratio = 0.0;
};

struct BatchResult {
    SimulationConfig config;
    std::size_t replicates = 0;
    StabilityAnalysis incipient_analysis;
    std::vector<std::size_t> dominant_mode_histogram;
    double mean_regularity_index = 0.0;
    double mean_mode3_to_mode4_ratio = 0.0;
    double share_mode3 = 0.0;
    double share_mode4 = 0.0;
    double share_mode3_ci_low = 0.0;
    double share_mode3_ci_high = 0.0;
    double share_mode4_ci_low = 0.0;
    double share_mode4_ci_high = 0.0;
    std::vector<BatchRunSummary> samples;
};

struct Preset {
    std::string id;
    std::string family_id;
    std::string name;
    std::string description;
    std::string hypothesis;
    SimulationConfig config;
};

double parameter_value(
    const std::vector<NamedValue>& values,
    const std::string& name,
    double fallback);

void upsert_parameter(
    std::vector<NamedValue>& values,
    const std::string& name,
    double value);

void validate_config(const SimulationConfig& config);

}  // namespace turing
