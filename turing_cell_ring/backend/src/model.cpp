#include "turing/model.hpp"

#include <cmath>
#include <stdexcept>

namespace turing {

namespace {

NamedSeries make_series(const std::string& name, std::vector<double> values) {
    return {name, std::move(values)};
}

std::vector<ReferencePattern> section10_reference_patterns() {
    return {
        {
            "paper_first_incipient",
            "Table 1 — first specimen incipient pattern",
            {
                make_series("X", {
                    1.130, 1.123, 1.154, 1.215, 1.249, 1.158, 1.074, 1.078, 1.148, 1.231,
                    1.204, 1.149, 1.156, 1.170, 1.131, 1.090, 1.109, 1.201, 1.306, 1.217
                }),
                make_series("Y", {
                    0.929, 0.940, 0.885, 0.810, 0.753, 0.873, 1.003, 1.000, 0.896, 0.775,
                    0.820, 0.907, 0.886, 0.854, 0.904, 0.976, 0.957, 0.820, 0.675, 0.811
                }),
            },
            std::vector<double>{
                1.130, 1.123, 1.154, 1.215, 1.249, 1.158, 1.074, 1.078, 1.148, 1.231,
                1.204, 1.149, 1.156, 1.170, 1.131, 1.090, 1.109, 1.201, 1.306, 1.217
            },
            std::vector<double>{
                0.929, 0.940, 0.885, 0.810, 0.753, 0.873, 1.003, 1.000, 0.896, 0.775,
                0.820, 0.907, 0.886, 0.854, 0.904, 0.976, 0.957, 0.820, 0.675, 0.811
            }
        },
        {
            "paper_first_final",
            "Table 1 — first specimen final pattern",
            {
                make_series("X", {
                    0.741, 0.761, 0.954, 1.711, 1.707, 0.875, 0.700, 0.699, 0.885, 1.704,
                    1.708, 0.944, 0.766, 0.744, 0.756, 0.935, 1.711, 1.706, 0.927, 0.746
                }),
                make_series("Y", {
                    1.463, 1.469, 1.255, 0.000, 0.000, 1.385, 1.622, 1.615, 1.382, 0.000,
                    0.000, 1.273, 1.451, 1.442, 1.478, 1.308, 0.000, 0.000, 1.309, 1.487
                }),
            },
            std::vector<double>{
                0.741, 0.761, 0.954, 1.711, 1.707, 0.875, 0.700, 0.699, 0.885, 1.704,
                1.708, 0.944, 0.766, 0.744, 0.756, 0.935, 1.711, 1.706, 0.927, 0.746
            },
            std::vector<double>{
                1.463, 1.469, 1.255, 0.000, 0.000, 1.385, 1.622, 1.615, 1.382, 0.000,
                0.000, 1.273, 1.451, 1.442, 1.478, 1.308, 0.000, 0.000, 1.309, 1.487
            }
        },
        {
            "paper_second_incipient_y",
            "Table 1 — second specimen incipient Y",
            {make_series("Y", {
                0.834, 0.833, 0.766, 0.836, 0.930, 0.898, 0.770, 0.740, 0.846, 0.937,
                0.986, 1.019, 0.899, 0.431, 0.485, 0.919, 1.035, 1.003, 0.899, 0.820
            })},
            std::nullopt,
            std::vector<double>{
                0.834, 0.833, 0.766, 0.836, 0.930, 0.898, 0.770, 0.740, 0.846, 0.937,
                0.986, 1.019, 0.899, 0.431, 0.485, 0.919, 1.035, 1.003, 0.899, 0.820
            }
        },
        {
            "paper_slow_incipient_y",
            "Table 1 — slow-cooking incipient Y",
            {make_series("Y", {
                1.057, 0.903, 0.813, 0.882, 1.088, 1.222, 1.173, 0.956, 0.775, 0.775,
                0.969, 1.170, 1.203, 1.048, 0.868, 0.813, 0.910, 1.050, 1.175, 1.181
            })},
            std::nullopt,
            std::vector<double>{
                1.057, 0.903, 0.813, 0.882, 1.088, 1.222, 1.173, 0.956, 0.775, 0.775,
                0.969, 1.170, 1.203, 1.048, 0.868, 0.813, 0.910, 1.050, 1.175, 1.181
            }
        },
        {
            "paper_four_lobed_equilibrium",
            "Table 1 — four-lobed equilibrium",
            {
                make_series("X", {
                    1.747, 1.685, 1.445, 0.445, 1.685, 1.747, 1.685, 0.445, 0.445, 1.685,
                    1.747, 1.685, 0.445, 0.445, 1.685, 1.747, 1.685, 0.445, 0.445, 1.685
                }),
                make_series("Y", {
                    0.000, 0.000, 2.500, 2.500, 0.000, 0.000, 0.000, 2.500, 2.500, 0.000,
                    0.000, 0.000, 2.500, 2.500, 0.000, 0.000, 0.000, 2.500, 2.500, 0.000
                }),
            },
            std::vector<double>{
                1.747, 1.685, 1.445, 0.445, 1.685, 1.747, 1.685, 0.445, 0.445, 1.685,
                1.747, 1.685, 0.445, 0.445, 1.685, 1.747, 1.685, 0.445, 0.445, 1.685
            },
            std::vector<double>{
                0.000, 0.000, 2.500, 2.500, 0.000, 0.000, 0.000, 2.500, 2.500, 0.000,
                0.000, 0.000, 2.500, 2.500, 0.000, 0.000, 0.000, 2.500, 2.500, 0.000
            }
        }
    };
}

std::vector<ReferencePattern> example2_reference_patterns() {
    return {
        {
            "paper_table2_stable_equilibrium",
            "Table 2 — six-cell stable equilibrium",
            {
                make_series("X", {7.5, 3.5, 2.5, 2.5, 3.5, 7.5}),
                make_series("Y", {0.0, 8.0, 8.0, 8.0, 8.0, 0.0}),
            },
            std::vector<double>{7.5, 3.5, 2.5, 2.5, 3.5, 7.5},
            std::vector<double>{0.0, 8.0, 8.0, 8.0, 8.0, 0.0}
        }
    };
}

ControlSchedule make_zero_schedule() {
    ControlSchedule schedule;
    schedule.id = "constant_zero";
    schedule.name = "Constant zero";
    schedule.segments = {{0.0, 1.0, 0.0, 0.0}};
    schedule.post_segment_value = 0.0;
    return schedule;
}

Preset make_quick_preset() {
    Preset preset;
    preset.id = "paper-quick";
    preset.family_id = "section10_example1";
    preset.name = "Turing 1952 numerical example — quick cooking";
    preset.description =
        "Section 10, twenty-cell ring, exact special units, stochastic disturbances on, "
        "gamma ramped from -1/4 to +1/16 and then returned to 0.";
    preset.hypothesis =
        "A nearly homogeneous ring should leave homogeneity through a stationary finite-wave-length "
        "instability, with three lobes the most likely pattern and four lobes the closest competitor.";
    preset.config.preset_id = preset.id;
    preset.config.family_id = preset.family_id;
    preset.config.analysis_mode = "stationary_two_species";
    preset.config.execution_mode = "modern";
    preset.config.execution_profile_id = "modern_default";
    preset.config.species_order = {"X", "Y"};
    preset.config.schedule = quick_cooking_schedule();
    preset.config.total_time = 80.0;
    preset.config.dt = 0.01;
    preset.config.enable_noise = true;
    preset.config.noise_scale = 1.0;
    preset.config.initial_perturbation_scale = 1.0e-4;
    preset.config.capture_stride = 20;
    preset.config.incipient_capture_gamma = 1.0 / 16.0;
    preset.config.incipient_capture_mode = "mode234_threshold";
    preset.config.incipient_capture_start_time = 32.0;
    preset.config.incipient_mode234_threshold = 0.18;
    return preset;
}

Preset make_slow_preset() {
    Preset preset;
    preset.id = "paper-slow";
    preset.family_id = "section10_example1";
    preset.name = "Turing 1952 numerical example — slow cooking";
    preset.description =
        "Section 10 slow-cooking variant, gamma ramped from -0.010 to 0.003 at 10^-5 per special time unit.";
    preset.hypothesis =
        "A slower control-parameter ramp should suppress irregular transient mode competition and "
        "make the three-lobed stationary pattern more clearly visible.";
    preset.config.preset_id = preset.id;
    preset.config.family_id = preset.family_id;
    preset.config.analysis_mode = "stationary_two_species";
    preset.config.execution_mode = "modern";
    preset.config.execution_profile_id = "modern_default";
    preset.config.species_order = {"X", "Y"};
    preset.config.schedule = slow_cooking_schedule();
    preset.config.total_time = 1300.0;
    preset.config.dt = 0.05;
    preset.config.enable_noise = true;
    preset.config.noise_scale = 1.0;
    preset.config.initial_perturbation_scale = 1.0e-4;
    preset.config.capture_stride = 50;
    preset.config.incipient_capture_gamma = 0.003;
    upsert_parameter(
        preset.config.family_parameters,
        "fullChemistryIncipientStartTime",
        900.0);
    upsert_parameter(
        preset.config.family_parameters,
        "fullChemistryIncipientMode234Threshold",
        0.2);
    return preset;
}

Preset make_table2_stable_preset() {
    Preset preset;
    preset.id = "paper-table2-stable-ring";
    preset.family_id = "example2_table2";
    preset.name = "Second chemical example — Table 2 stable six-cell ring";
    preset.description =
        "Paper page 65, second chemical example, six-cell stable equilibrium matching Table 2.";
    preset.hypothesis =
        "At k = 12 the second chemical example admits a stable six-cell ring equilibrium matching Table 2.";
    preset.config.preset_id = preset.id;
    preset.config.family_id = preset.family_id;
    preset.config.analysis_mode = "stationary_two_species";
    preset.config.execution_mode = "modern";
    preset.config.execution_profile_id = "modern_default";
    preset.config.species_order = {"X", "Y"};
    preset.config.ring.cell_count = 6;
    preset.config.schedule = zero_schedule();
    preset.config.total_time = 40.0;
    preset.config.dt = 0.02;
    preset.config.enable_noise = false;
    preset.config.initial_perturbation_scale = 1.0e-3;
    preset.config.capture_stride = 10;
    preset.config.max_mode = 3;
    upsert_parameter(preset.config.family_parameters, "example2F", 4.0);
    upsert_parameter(preset.config.family_parameters, "table2ProfileGain", 0.12);
    return preset;
}

Preset make_example2_scan_preset() {
    Preset preset;
    preset.id = "paper-example2-scan";
    preset.family_id = "example2_table2";
    preset.name = "Second chemical example — stability scan preset";
    preset.description =
        "Paper page 65, second chemical example with adjustable f and k = 16 - f for stability scans.";
    preset.hypothesis =
        "The second chemical example transitions through the paper's case-(d) and instability windows as k changes.";
    preset.config.preset_id = preset.id;
    preset.config.family_id = preset.family_id;
    preset.config.analysis_mode = "stationary_two_species";
    preset.config.execution_mode = "modern";
    preset.config.execution_profile_id = "modern_default";
    preset.config.species_order = {"X", "Y"};
    preset.config.ring.cell_count = 20;
    preset.config.schedule = zero_schedule();
    preset.config.total_time = 60.0;
    preset.config.dt = 0.02;
    preset.config.enable_noise = true;
    preset.config.capture_stride = 20;
    upsert_parameter(preset.config.family_parameters, "example2F", 7.0);
    return preset;
}

Preset make_wave_e_preset() {
    Preset preset;
    preset.id = "paper-wave-e-travelling";
    preset.family_id = "oscillatory_case_e";
    preset.name = "Section 8 case (e) — finite-wavelength travelling wave";
    preset.description =
        "Three-morphogen oscillatory example for genuine travelling waves, using the paper's case (e) coefficients.";
    preset.hypothesis =
        "A three-morphogen ring can support a dominant oscillatory finite-wavelength travelling wave.";
    preset.config.preset_id = preset.id;
    preset.config.family_id = preset.family_id;
    preset.config.analysis_mode = "oscillatory_three_species";
    preset.config.execution_mode = "modern";
    preset.config.execution_profile_id = "modern_default";
    preset.config.species_order = {"M1", "M2", "M3"};
    preset.config.schedule = zero_schedule();
    preset.config.total_time = 40.0;
    preset.config.dt = 0.01;
    preset.config.enable_noise = true;
    preset.config.noise_scale = 1.0;
    preset.config.initial_perturbation_scale = 5.0e-4;
    preset.config.capture_stride = 10;
    preset.config.max_mode = 10;
    upsert_parameter(preset.config.family_parameters, "oscillatoryBias", 0.06);
    return preset;
}

Preset make_wave_f_preset() {
    Preset preset;
    preset.id = "paper-wave-f-short-oscillation";
    preset.family_id = "oscillatory_case_f";
    preset.name = "Section 8 case (f) — short-wave oscillation";
    preset.description =
        "Three-morphogen oscillatory example with neighbouring cells nearly 180 degrees out of phase.";
    preset.hypothesis =
        "A three-morphogen ring can exhibit extreme-short-wave metabolic oscillation with near-antiphase neighbours.";
    preset.config.preset_id = preset.id;
    preset.config.family_id = preset.family_id;
    preset.config.analysis_mode = "oscillatory_three_species";
    preset.config.execution_mode = "modern";
    preset.config.execution_profile_id = "modern_default";
    preset.config.species_order = {"M1", "M2", "M3"};
    preset.config.schedule = zero_schedule();
    preset.config.total_time = 40.0;
    preset.config.dt = 0.01;
    preset.config.enable_noise = true;
    preset.config.noise_scale = 1.0;
    preset.config.initial_perturbation_scale = 5.0e-4;
    preset.config.capture_stride = 10;
    preset.config.max_mode = 10;
    upsert_parameter(preset.config.family_parameters, "oscillatoryBias", 0.05);
    return preset;
}

double example2_f_value(const SimulationConfig& config) {
    return parameter_value(config.family_parameters, "example2F", 4.0);
}

std::vector<std::vector<double>> example2_jacobian_matrix(const Equilibrium& equilibrium) {
    const auto x = equilibrium.species_values.empty() ? equilibrium.x : equilibrium.species_values[0];
    const auto y = equilibrium.species_values.size() > 1 ? equilibrium.species_values[1] : equilibrium.y;
    return {
        {-y / 16.0, -x / 16.0},
        {y / 16.0, (x - 1.0) / 16.0}
    };
}

}  // namespace

PaperModel::PaperModel()
    : families_{
        {
            "section10_example1",
            "Section 10 numerical example",
            "Twenty-cell ring with the Section 10 reaction law and Table 1 references.",
            "stationary_two_species",
            FamilyKind::section10_example1,
            {"X", "Y"},
            {0.5, 0.25},
            {},
            1,
            true,
            true,
            1.0,
            section10_reference_patterns(),
        },
        {
            "example2_table2",
            "Second chemical example and Table 2",
            "Two-species paper example following Section 10, with Table 2 stable equilibrium.",
            "stationary_two_species",
            FamilyKind::example2_table2,
            {"X", "Y"},
            {0.25, 0.375},
            {},
            1,
            false,
            true,
            1.0,
            example2_reference_patterns(),
        },
        {
            "oscillatory_case_e",
            "Section 8 case (e)",
            "Three-morphogen travelling-wave example.",
            "oscillatory_three_species",
            FamilyKind::oscillatory_case_e,
            {"M1", "M2", "M3"},
            {1.0, 0.5, 0.0},
            {
                {-10.0 / 3.0, 3.0, -1.0},
                {-2.0, 7.0 / 3.0, 0.0},
                {3.0, -4.0, 0.0},
            },
            0,
            false,
            false,
            1.0,
            {},
        },
        {
            "oscillatory_case_f",
            "Section 8 case (f)",
            "Three-morphogen extreme-short-wave oscillatory example.",
            "oscillatory_three_species",
            FamilyKind::oscillatory_case_f,
            {"M1", "M2", "M3"},
            {1.0, 0.0, 0.0},
            {
                {-1.0, -1.0, 0.0},
                {1.0, 0.0, -1.0},
                {0.0, 1.0, 0.0},
            },
            0,
            false,
            false,
            2.5,
            {},
        }
    } {}

const ModelFamily* PaperModel::find_family(const std::string& family_id) const {
    for (const auto& family : families_) {
        if (family.id == family_id) {
            return &family;
        }
    }
    return nullptr;
}

const ModelFamily& PaperModel::require_family(const std::string& family_id) const {
    const auto* family = find_family(family_id);
    if (family == nullptr) {
        throw std::invalid_argument("Unknown family: " + family_id);
    }
    return *family;
}

const std::vector<ModelFamily>& PaperModel::families() const {
    return families_;
}

Equilibrium PaperModel::solve_equilibrium(const SimulationConfig& config) const {
    Equilibrium equilibrium;
    const auto& family = require_family(config.family_id);
    equilibrium.species_order = family.species_order;

    switch (family.kind) {
        case FamilyKind::section10_example1: {
            const auto gamma = config.schedule.start_value();
            const auto discriminant = 2500.0 + 28.0 * (57.0 + 55.0 * gamma);
            if (discriminant <= 0.0) {
                throw std::runtime_error("Homogeneous equilibrium does not exist for the supplied gamma.");
            }
            equilibrium.y = 1.0;
            equilibrium.x = (-50.0 + std::sqrt(discriminant)) / 14.0;
            equilibrium.species_values = {equilibrium.x, equilibrium.y};
            return equilibrium;
        }
        case FamilyKind::example2_table2: {
            const auto f = example2_f_value(config);
            const auto k = 16.0 - f;
            equilibrium.x = 16.0 / k;
            equilibrium.y = k;
            equilibrium.species_values = {equilibrium.x, equilibrium.y};
            return equilibrium;
        }
        case FamilyKind::oscillatory_case_e:
        case FamilyKind::oscillatory_case_f:
            equilibrium.x = 0.0;
            equilibrium.y = 0.0;
            equilibrium.species_values.assign(family.species_order.size(), 0.0);
            return equilibrium;
    }

    throw std::runtime_error("Unhandled family equilibrium.");
}

Jacobian PaperModel::jacobian(const SimulationConfig& config, const Equilibrium& equilibrium) const {
    Jacobian value;
    const auto& family = require_family(config.family_id);
    value.matrix.clear();

    switch (family.kind) {
        case FamilyKind::section10_example1: {
            const auto x = equilibrium.x;
            const auto y = equilibrium.y;
            value.a = (-14.0 * x - 50.0 * y) / 32.0;
            value.b = (-50.0 * x) / 32.0;
            value.c = (14.0 * x + 50.0 * y) / 32.0;
            value.d = (50.0 * x - 2.0 * y) / 32.0;
            value.matrix = {
                {value.a, value.b},
                {value.c, value.d},
            };
            return value;
        }
        case FamilyKind::example2_table2: {
            value.matrix = example2_jacobian_matrix(equilibrium);
            value.a = value.matrix[0][0];
            value.b = value.matrix[0][1];
            value.c = value.matrix[1][0];
            value.d = value.matrix[1][1];
            return value;
        }
        case FamilyKind::oscillatory_case_e:
        case FamilyKind::oscillatory_case_f:
            value.matrix = family.linear_matrix;
            return value;
    }

    throw std::runtime_error("Unhandled family Jacobian.");
}

CellReaction PaperModel::evaluate_section10_cell(double x, double y, double gamma) const {
    CellReaction reaction;
    reaction.channels.source_x = 1.0 / 16.0;
    reaction.channels.xy_to_yy = (25.0 / 16.0) * x * y;
    reaction.channels.xx_to_yy = (7.0 / 64.0) * x * x;
    reaction.channels.y_to_x = y > 0.0 ? (55.0 / 32.0) * (1.0 + gamma) : 0.0;
    reaction.channels.sink_y = y / 16.0;

    reaction.dx = reaction.channels.source_x
        - reaction.channels.xy_to_yy
        - 2.0 * reaction.channels.xx_to_yy
        + reaction.channels.y_to_x;
    reaction.dy = reaction.channels.xy_to_yy
        + 2.0 * reaction.channels.xx_to_yy
        - reaction.channels.y_to_x
        - reaction.channels.sink_y;

    if (y <= 0.0 && reaction.dy < 0.0) {
        reaction.dx -= reaction.channels.y_to_x;
        reaction.dy += reaction.channels.y_to_x;
        reaction.channels.y_to_x = 0.0;
    }
    return reaction;
}

std::vector<double> PaperModel::evaluate_generic_cell(
    const SimulationConfig& config,
    const std::vector<double>& cell_species,
    double gamma) const {
    (void)gamma;
    const auto& family = require_family(config.family_id);

    switch (family.kind) {
        case FamilyKind::section10_example1: {
            const auto reaction = evaluate_section10_cell(cell_species[0], cell_species[1], gamma);
            return {reaction.dx, reaction.dy};
        }
        case FamilyKind::example2_table2: {
            const auto x = cell_species[0];
            const auto y = cell_species[1];
            const auto f = example2_f_value(config);
            auto dx = (16.0 - x * y) / 16.0;
            auto dy = (x * y - y - f) / 16.0;
            if (y <= 0.0 && dy < 0.0) {
                dy = 0.0;
            }
            return {dx, dy};
        }
        case FamilyKind::oscillatory_case_e:
        case FamilyKind::oscillatory_case_f: {
            std::vector<double> drift(cell_species.size(), 0.0);
            for (std::size_t row = 0; row < family.linear_matrix.size(); ++row) {
                for (std::size_t col = 0; col < family.linear_matrix[row].size(); ++col) {
                    drift[row] += family.linear_matrix[row][col] * cell_species[col];
                }
            }
            const auto instability_bias =
                parameter_value(config.family_parameters, "oscillatoryBias", 0.0);
            if (instability_bias != 0.0) {
                for (std::size_t index = 0; index < drift.size(); ++index) {
                    drift[index] += instability_bias * cell_species[index];
                }
            }
            return drift;
        }
    }

    throw std::runtime_error("Unhandled family cell drift.");
}

std::vector<ReferencePattern> PaperModel::reference_patterns(const SimulationConfig& config) const {
    return require_family(config.family_id).reference_patterns;
}

std::vector<NamedSeries> PaperModel::reference_species_for_preset(const SimulationConfig& config) const {
    if (config.preset_id == "paper-table2-stable-ring") {
        return {
            make_series("X", {7.5, 3.5, 2.5, 2.5, 3.5, 7.5}),
            make_series("Y", {0.0, 8.0, 8.0, 8.0, 8.0, 0.0}),
        };
    }
    return {};
}

bool PaperModel::supports_full_chemistry(const SimulationConfig& config) const {
    return require_family(config.family_id).supports_full_chemistry;
}

bool PaperModel::supports_historical_execution(const SimulationConfig& config) const {
    return require_family(config.family_id).supports_historical_execution;
}

std::size_t PaperModel::primary_species_index(const SimulationConfig& config) const {
    return require_family(config.family_id).primary_species_index;
}

double PaperModel::mode_u_value(const SimulationConfig& config, std::size_t mode) const {
    const auto& family = require_family(config.family_id);
    const auto n = static_cast<double>(config.ring.cell_count);
    const auto s = static_cast<double>(mode);
    const auto sin_term = std::sin(M_PI * s / n);
    return family.mode_u_scale * 4.0 * sin_term * sin_term;
}

std::vector<std::vector<double>> PaperModel::mode_matrix(
    const SimulationConfig& config,
    std::size_t mode,
    double gamma) const {
    (void)gamma;
    const auto& family = require_family(config.family_id);
    if (family.kind == FamilyKind::oscillatory_case_e || family.kind == FamilyKind::oscillatory_case_f) {
        auto matrix = family.linear_matrix;
        const auto u_value = mode_u_value(config, mode);
        for (std::size_t index = 0; index < matrix.size() && index < family.diffusion_by_species.size(); ++index) {
            matrix[index][index] -= family.diffusion_by_species[index] * u_value;
        }
        const auto instability_bias =
            parameter_value(config.family_parameters, "oscillatoryBias", 0.0);
        if (instability_bias != 0.0) {
            for (std::size_t index = 0; index < matrix.size(); ++index) {
                matrix[index][index] += instability_bias;
            }
        }
        return matrix;
    }

    const auto equilibrium = solve_equilibrium(config);
    const auto jac = jacobian(config, equilibrium);
    const auto u_value = mode_u_value(config, mode);
    auto matrix = jac.matrix;
    const auto diffusions = family.diffusion_by_species;
    for (std::size_t index = 0; index < matrix.size() && index < diffusions.size(); ++index) {
        matrix[index][index] -= diffusions[index] * u_value;
    }
    return matrix;
}

PresetRegistry::PresetRegistry()
    : presets_{
        make_quick_preset(),
        make_slow_preset(),
        make_table2_stable_preset(),
        make_example2_scan_preset(),
        make_wave_e_preset(),
        make_wave_f_preset(),
    } {}

const std::vector<Preset>& PresetRegistry::presets() const {
    return presets_;
}

const Preset* PresetRegistry::find(const std::string& preset_id) const {
    for (const auto& preset : presets_) {
        if (preset.id == preset_id) {
            return &preset;
        }
    }
    return nullptr;
}

const std::vector<ModelFamily>& PresetRegistry::families() const {
    return model_.families();
}

const ModelFamily* PresetRegistry::find_family(const std::string& family_id) const {
    return model_.find_family(family_id);
}

ControlSchedule quick_cooking_schedule() {
    ControlSchedule schedule;
    schedule.id = "quick_cooking";
    schedule.name = "Quick cooking";
    schedule.segments = {
        {0.0, 40.0, -0.25, 1.0 / 16.0},
        {40.0, 48.0, 1.0 / 16.0, 0.0}
    };
    schedule.post_segment_value = 0.0;
    return schedule;
}

ControlSchedule slow_cooking_schedule() {
    ControlSchedule schedule;
    schedule.id = "slow_cooking";
    schedule.name = "Slow cooking";
    schedule.segments = {
        {0.0, 1300.0, -0.010, 0.003}
    };
    schedule.post_segment_value = 0.003;
    return schedule;
}

ControlSchedule zero_schedule() {
    return make_zero_schedule();
}

}  // namespace turing
