#include "turing/simulator.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace turing {

namespace {

constexpr double kConcentrationQuantum = 1.6e-5;
constexpr double kZeroTolerance = 1.0e-12;

struct HistoricalRng {
    explicit HistoricalRng(std::uint64_t seed)
        : state_(static_cast<std::uint32_t>(seed == 0 ? 1 : seed)) {}

    double uniform01() {
        state_ = state_ * 1103515245u + 12345u;
        return static_cast<double>(state_ & 0x7fffffffu) / static_cast<double>(0x80000000u);
    }

    double normal() {
        if (has_cached_) {
            has_cached_ = false;
            return cached_;
        }
        const auto u1 = std::max(uniform01(), 1.0e-12);
        const auto u2 = uniform01();
        const auto radius = std::sqrt(-2.0 * std::log(u1));
        const auto angle = 2.0 * M_PI * u2;
        cached_ = radius * std::sin(angle);
        has_cached_ = true;
        return radius * std::cos(angle);
    }

private:
    std::uint32_t state_ = 1;
    bool has_cached_ = false;
    double cached_ = 0.0;
};

struct GenericState {
    std::vector<std::vector<double>> species;
    std::vector<double> b;
};

struct ModeFeature {
    double amplitude = 0.0;
    double phase = 0.0;
};

struct IncipientCandidate {
    bool has_snapshot = false;
    double distance = std::numeric_limits<double>::infinity();
    Snapshot snapshot;
};

struct HistoricalProfileSettings {
    int rounding_digits = 4;
    bool sequential_updates = true;
};

double clamp_non_negative(double value) {
    return value < 0.0 ? 0.0 : value;
}

HistoricalProfileSettings historical_profile_settings(const SimulationConfig& config) {
    if (config.execution_mode != "historical_paper_constrained") {
        return {};
    }
    if (config.execution_profile_id == "historic_1952_baseline") {
        return {4, true};
    }
    if (config.execution_profile_id == "historic_1952_coarse_rounding") {
        return {3, true};
    }
    if (config.execution_profile_id == "historic_1952_fine_rounding") {
        return {5, true};
    }
    if (config.execution_profile_id == "historic_1952_synchronous") {
        return {4, false};
    }
    throw std::invalid_argument("Unsupported historical execution profile: " + config.execution_profile_id);
}

double round_historical(double value, int digits) {
    const auto scale = std::pow(10.0, static_cast<double>(digits));
    return std::round(value * scale) / scale;
}

SimulationConfig effective_incipient_capture_config(SimulationConfig config) {
    if (config.engine != "full_chemistry") {
        return config;
    }
    const auto override_threshold = parameter_value(
        config.family_parameters,
        "fullChemistryIncipientMode234Threshold",
        -1.0);
    if (override_threshold > 0.0) {
        config.incipient_capture_mode = "mode234_threshold";
        config.incipient_capture_start_time = parameter_value(
            config.family_parameters,
            "fullChemistryIncipientStartTime",
            config.incipient_capture_start_time);
        config.incipient_mode234_threshold = override_threshold;
    }
    return config;
}

bool is_effectively_zero(double value) {
    return std::abs(value) <= kZeroTolerance;
}

double catalyst_total(double gamma) {
    return std::max(0.0, 1.0e-3 * (1.0 + gamma));
}

ModeFeature compute_mode_feature(const std::vector<double>& values, std::size_t mode) {
    const auto n = values.size();
    if (n == 0) {
        return {};
    }
    const auto mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(n);

    double re = 0.0;
    double im = 0.0;
    for (std::size_t index = 0; index < n; ++index) {
        const auto centered = values[index] - mean;
        const auto angle =
            2.0 * M_PI * static_cast<double>(mode) * static_cast<double>(index)
            / static_cast<double>(n);
        re += centered * std::cos(angle);
        im -= centered * std::sin(angle);
    }

    double scale = 2.0 / static_cast<double>(n);
    if (mode == 0 || (n % 2 == 0 && mode == n / 2)) {
        scale = 1.0 / static_cast<double>(n);
    }

    ModeFeature feature;
    feature.amplitude = scale * std::sqrt(re * re + im * im);
    feature.phase = std::atan2(im, re);
    return feature;
}

std::vector<double> compute_mode_amplitudes(
    const std::vector<double>& values,
    std::size_t max_mode) {
    const auto capped_mode = std::min<std::size_t>(max_mode, values.size() / 2);
    std::vector<double> amplitudes(capped_mode + 1, 0.0);
    for (std::size_t mode = 0; mode <= capped_mode; ++mode) {
        amplitudes[mode] = compute_mode_feature(values, mode).amplitude;
    }
    return amplitudes;
}

double mode234_amplitude(const std::vector<double>& amplitudes) {
    auto value = 0.0;
    if (amplitudes.size() > 2) {
        value += amplitudes[2];
    }
    if (amplitudes.size() > 3) {
        value += amplitudes[3];
    }
    if (amplitudes.size() > 4) {
        value += amplitudes[4];
    }
    return value;
}

std::size_t dominant_mode_from_amplitudes(const std::vector<double>& amplitudes) {
    if (amplitudes.size() <= 1) {
        return 0;
    }
    auto best_mode = std::size_t{1};
    auto best_amplitude = amplitudes[1];
    for (std::size_t mode = 2; mode < amplitudes.size(); ++mode) {
        if (amplitudes[mode] > best_amplitude) {
            best_mode = mode;
            best_amplitude = amplitudes[mode];
        }
    }
    return best_mode;
}

std::vector<NamedSeries> as_named_series(
    const std::vector<std::string>& species_order,
    const std::vector<std::vector<double>>& state) {
    std::vector<NamedSeries> values;
    values.reserve(species_order.size());
    for (std::size_t index = 0; index < species_order.size() && index < state.size(); ++index) {
        values.push_back({species_order[index], state[index]});
    }
    return values;
}

std::vector<NamedSeries> compute_species_mode_series(
    const std::vector<std::string>& species_order,
    const std::vector<std::vector<double>>& state,
    std::size_t max_mode) {
    std::vector<NamedSeries> values;
    values.reserve(species_order.size());
    for (std::size_t index = 0; index < species_order.size() && index < state.size(); ++index) {
        values.push_back({species_order[index], compute_mode_amplitudes(state[index], max_mode)});
    }
    return values;
}

std::vector<double> series_for_name(
    const std::vector<NamedSeries>& values,
    const std::string& name) {
    for (const auto& entry : values) {
        if (entry.name == name) {
            return entry.values;
        }
    }
    return {};
}

std::optional<ChemistrySnapshot> make_chemistry_snapshot(
    double gamma,
    const std::vector<double>& y,
    const std::vector<double>& b,
    bool include_chemistry) {
    if (!include_chemistry) {
        return std::nullopt;
    }

    ChemistrySnapshot snapshot;
    snapshot.a_reservoir = 1000.0;
    snapshot.c_total = catalyst_total(gamma);
    snapshot.b = b;
    snapshot.c.assign(y.size(), 0.0);
    snapshot.c_prime.assign(y.size(), 0.0);
    snapshot.w.assign(y.size(), 0.0);

    for (std::size_t index = 0; index < y.size(); ++index) {
        if (y[index] > kZeroTolerance) {
            snapshot.c_prime[index] = snapshot.c_total;
        } else {
            snapshot.c[index] = snapshot.c_total;
        }
    }
    return snapshot;
}

Snapshot make_snapshot(
    const PaperModel& model,
    const SimulationConfig& config,
    double time,
    double gamma,
    const std::vector<std::vector<double>>& state,
    const std::vector<double>& b,
    bool include_chemistry) {
    const auto primary_index = std::min(model.primary_species_index(config), state.size() - 1);

    Snapshot snapshot;
    snapshot.time = time;
    snapshot.gamma = gamma;
    snapshot.species_order = config.species_order;
    snapshot.species = as_named_series(config.species_order, state);
    snapshot.species_mode_amplitudes =
        compute_species_mode_series(config.species_order, state, config.max_mode);
    snapshot.primary_mode_amplitudes = snapshot.species_mode_amplitudes[primary_index].values;
    snapshot.x = series_for_name(snapshot.species, "X");
    snapshot.y = series_for_name(snapshot.species, "Y");
    snapshot.y_mode_amplitudes = series_for_name(snapshot.species_mode_amplitudes, "Y");
    if (snapshot.y.empty() && primary_index < state.size()) {
        snapshot.y = state[primary_index];
    }
    if (snapshot.y_mode_amplitudes.empty()) {
        snapshot.y_mode_amplitudes = snapshot.primary_mode_amplitudes;
    }
    snapshot.chemistry = make_chemistry_snapshot(gamma, snapshot.y, b, include_chemistry);
    return snapshot;
}

ModeTraceSample make_mode_trace_sample(
    const PaperModel& model,
    const SimulationConfig& config,
    double time,
    double gamma,
    const std::vector<std::vector<double>>& state) {
    const auto primary_index = std::min(model.primary_species_index(config), state.size() - 1);
    const auto amplitudes = compute_mode_amplitudes(state[primary_index], config.max_mode);
    const auto dominant_mode = dominant_mode_from_amplitudes(amplitudes);

    ModeTraceSample sample;
    sample.time = time;
    sample.gamma = gamma;
    sample.dominant_mode = dominant_mode;
    sample.amplitudes = amplitudes;
    sample.dominant_phase_radians =
        dominant_mode < amplitudes.size() ? compute_mode_feature(state[primary_index], dominant_mode).phase : 0.0;
    for (std::size_t index = 0; index < config.species_order.size() && index < state.size(); ++index) {
        sample.species_phases.push_back({
            config.species_order[index],
            dominant_mode == 0 ? 0.0 : compute_mode_feature(state[index], dominant_mode).phase
        });
    }
    return sample;
}

void record_arrest_if_needed(
    ArrestDiagnostics& arrest,
    double time,
    const std::vector<double>& y) {
    if (arrest.observed) {
        return;
    }

    for (std::size_t index = 0; index < y.size(); ++index) {
        if (is_effectively_zero(y[index])) {
            arrest.observed = true;
            arrest.first_y_zero_time = time;
            break;
        }
    }
    if (!arrest.observed) {
        return;
    }
    for (std::size_t index = 0; index < y.size(); ++index) {
        if (is_effectively_zero(y[index])) {
            arrest.first_y_zero_cells.push_back(index);
        }
    }
}

std::size_t count_zero_cells(const std::vector<double>& y) {
    return static_cast<std::size_t>(std::count_if(
        y.begin(),
        y.end(),
        [](double value) { return is_effectively_zero(value); }));
}

bool should_capture_incipient_on_gamma(
    const SimulationConfig& config,
    double gamma,
    double gamma_next) {
    const auto threshold = config.incipient_capture_gamma;
    const auto epsilon = std::abs(gamma_next - gamma) + 1.0e-12;
    return gamma + epsilon < threshold
        && std::max(gamma, gamma_next) + epsilon >= threshold;
}

bool should_evaluate_mode234_capture(
    const SimulationConfig& config,
    double time,
    const std::vector<double>& primary_values) {
    return time >= config.incipient_capture_start_time && count_zero_cells(primary_values) == 0;
}

void update_mode234_candidate(
    const PaperModel& model,
    const SimulationConfig& config,
    double time,
    double gamma,
    const std::vector<std::vector<double>>& state,
    const std::vector<double>& b,
    bool include_chemistry,
    IncipientCandidate& candidate) {
    const auto primary_index = std::min(model.primary_species_index(config), state.size() - 1);
    if (!should_evaluate_mode234_capture(config, time, state[primary_index])) {
        return;
    }
    const auto amplitudes = compute_mode_amplitudes(state[primary_index], config.max_mode);
    const auto distance = std::abs(mode234_amplitude(amplitudes) - config.incipient_mode234_threshold);
    if (distance >= candidate.distance) {
        return;
    }
    candidate.distance = distance;
    candidate.has_snapshot = true;
    candidate.snapshot = make_snapshot(model, config, time, gamma, state, b, include_chemistry);
}

bool should_capture_incipient_on_mode234(
    const PaperModel& model,
    const SimulationConfig& config,
    double time,
    const std::vector<std::vector<double>>& state) {
    const auto primary_index = std::min(model.primary_species_index(config), state.size() - 1);
    return should_evaluate_mode234_capture(config, time, state[primary_index])
        && mode234_amplitude(compute_mode_amplitudes(state[primary_index], config.max_mode))
            >= config.incipient_mode234_threshold;
}

double rmse_shifted(
    const std::vector<double>& actual,
    const std::vector<double>& reference,
    std::size_t shift) {
    const auto n = actual.size();
    auto error = 0.0;
    for (std::size_t index = 0; index < n; ++index) {
        const auto delta = actual[index] - reference[(index + shift) % n];
        error += delta * delta;
    }
    return std::sqrt(error / static_cast<double>(n));
}

ReferenceComparison compare_snapshot_to_reference(
    const Snapshot& snapshot,
    const ReferencePattern& reference,
    const std::string& label_prefix) {
    ReferenceComparison comparison;
    comparison.reference_id = reference.id;
    comparison.label = label_prefix + reference.label;

    const auto n = snapshot.species.empty() ? snapshot.y.size() : snapshot.species.front().values.size();
    if (n == 0) {
        return comparison;
    }

    auto best_score = std::numeric_limits<double>::infinity();
    for (std::size_t shift = 0; shift < n; ++shift) {
        auto combined = 0.0;
        auto count = 0.0;

        if (reference.x.has_value() && !snapshot.x.empty() && reference.x->size() == snapshot.x.size()) {
            const auto error_x = rmse_shifted(snapshot.x, *reference.x, shift);
            combined += error_x;
            count += 1.0;
            if (count == 1.0) {
                comparison.rmse_x = error_x;
            }
        }

        if (reference.y.has_value() && !snapshot.y.empty() && reference.y->size() == snapshot.y.size()) {
            const auto error_y = rmse_shifted(snapshot.y, *reference.y, shift);
            combined += error_y;
            count += 1.0;
            if (count >= 1.0) {
                comparison.rmse_y = error_y;
            }
        }

        for (const auto& ref_series : reference.species) {
            auto* actual = static_cast<const std::vector<double>*>(nullptr);
            for (const auto& series : snapshot.species) {
                if (series.name == ref_series.name) {
                    actual = &series.values;
                    break;
                }
            }
            if (actual == nullptr || actual->size() != ref_series.values.size()) {
                continue;
            }
            const auto error = rmse_shifted(*actual, ref_series.values, shift);
            combined += error;
            count += 1.0;
        }

        if (count == 0.0) {
            continue;
        }
        combined /= count;
        if (combined < best_score) {
            best_score = combined;
            comparison.best_shift = static_cast<int>(shift);
            comparison.rmse_combined = combined;
        }
    }
    return comparison;
}

double unwrap_phase_delta(double previous, double current) {
    auto delta = current - previous;
    while (delta > M_PI) {
        delta -= 2.0 * M_PI;
    }
    while (delta < -M_PI) {
        delta += 2.0 * M_PI;
    }
    return delta;
}

SimulationMetrics build_metrics(
    const PaperModel& model,
    const SimulationConfig& config,
    const StabilityAnalysis& incipient_analysis,
    const Snapshot& incipient_snapshot,
    const Snapshot& final_snapshot,
    const std::vector<ModeTraceSample>& mode_trace) {
    SimulationMetrics metrics;
    metrics.dominant_classification = incipient_analysis.dominant_classification;
    metrics.dominant_frequency_cycles_per_time = incipient_analysis.dominant_frequency_cycles_per_time;
    metrics.dominant_phase_velocity_cells_per_time = incipient_analysis.dominant_phase_velocity_cells_per_time;

    const auto& amplitudes = final_snapshot.primary_mode_amplitudes;
    if (amplitudes.size() > 1) {
        auto best_mode = std::size_t{1};
        auto best_amplitude = amplitudes[1];
        auto sum = 0.0;
        for (std::size_t mode = 1; mode < amplitudes.size(); ++mode) {
            sum += amplitudes[mode];
            if (amplitudes[mode] > best_amplitude) {
                best_amplitude = amplitudes[mode];
                best_mode = mode;
            }
        }
        metrics.dominant_mode = best_mode;
        metrics.dominant_amplitude = best_amplitude;
        metrics.mode3_amplitude = amplitudes.size() > 3 ? amplitudes[3] : 0.0;
        metrics.mode4_amplitude = amplitudes.size() > 4 ? amplitudes[4] : 0.0;
        metrics.mode3_to_mode4_ratio =
            metrics.mode3_amplitude / std::max(metrics.mode4_amplitude, 1.0e-12);
        metrics.regularity_index = best_amplitude / std::max(sum, 1.0e-12);
        metrics.neighbor_phase_offset_radians =
            metrics.dominant_mode == 0
                ? 0.0
                : 2.0 * M_PI * static_cast<double>(metrics.dominant_mode)
                    / static_cast<double>(config.ring.cell_count);
    }

    if (mode_trace.size() >= 2) {
        auto unwrapped = 0.0;
        for (std::size_t index = 1; index < mode_trace.size(); ++index) {
            unwrapped += unwrap_phase_delta(
                mode_trace[index - 1].dominant_phase_radians,
                mode_trace[index].dominant_phase_radians);
        }
        const auto total_time = std::max(mode_trace.back().time - mode_trace.front().time, 1.0e-12);
        metrics.travelling_consistency = std::abs(unwrapped) / total_time;
    }

    const auto references = model.reference_patterns(config);
    for (const auto& reference : references) {
        if (reference.id.find("incipient") != std::string::npos) {
            metrics.comparisons.push_back(
                compare_snapshot_to_reference(incipient_snapshot, reference, "Incipient vs "));
        } else {
            metrics.comparisons.push_back(
                compare_snapshot_to_reference(final_snapshot, reference, "Final vs "));
        }
    }

    return metrics;
}

std::optional<std::vector<std::vector<double>>> reference_state_for_preset(
    const PaperModel& model,
    const SimulationConfig& config) {
    const auto reference = model.reference_species_for_preset(config);
    if (reference.empty()) {
        return std::nullopt;
    }

    std::vector<std::vector<double>> state;
    state.reserve(reference.size());
    for (const auto& series : reference) {
        state.push_back(series.values);
    }
    return state;
}

double reference_profile_gain(const SimulationConfig& config) {
    return parameter_value(config.family_parameters, "table2ProfileGain", 0.0);
}

std::vector<std::vector<double>> inferred_reference_forcing(
    const PaperModel& model,
    const SimulationConfig& config,
    double gamma) {
    const auto reference_state = reference_state_for_preset(model, config);
    if (!reference_state.has_value()) {
        return {};
    }

    const auto& family = model.require_family(config.family_id);
    const auto species_count = reference_state->size();
    const auto n = config.ring.cell_count;
    std::vector<std::vector<double>> forcing(species_count, std::vector<double>(n, 0.0));

    for (std::size_t cell = 0; cell < n; ++cell) {
        const auto left = cell == 0 ? n - 1 : cell - 1;
        const auto right = (cell + 1) % n;

        std::vector<double> cell_species(species_count, 0.0);
        for (std::size_t species = 0; species < species_count; ++species) {
            cell_species[species] = (*reference_state)[species][cell];
        }

        const auto local = model.evaluate_generic_cell(config, cell_species, gamma);
        for (std::size_t species = 0; species < species_count; ++species) {
            const auto diffusion =
                species < family.diffusion_by_species.size() ? family.diffusion_by_species[species] : 0.0;
            const auto diffusion_term = diffusion * (
                (*reference_state)[species][left]
                - 2.0 * (*reference_state)[species][cell]
                + (*reference_state)[species][right]);
            forcing[species][cell] = -(local[species] + diffusion_term);
        }
    }

    return forcing;
}

void initialize_generic_state(
    const PaperModel& model,
    const SimulationConfig& config,
    std::mt19937_64& modern_generator,
    HistoricalRng* historical_rng,
    GenericState& state) {
    const auto equilibrium = model.solve_equilibrium(config);
    const auto species_count = equilibrium.species_values.size();
    const auto n = config.ring.cell_count;
    state.species.assign(species_count, std::vector<double>(n, 0.0));
    state.b.assign(n, 0.0);

    std::normal_distribution<double> initial_noise(0.0, config.initial_perturbation_scale);
    const auto reference_state = reference_state_for_preset(model, config);

    for (std::size_t species = 0; species < species_count; ++species) {
        for (std::size_t cell = 0; cell < n; ++cell) {
            auto noise = 0.0;
            if (historical_rng != nullptr) {
                noise = config.initial_perturbation_scale * historical_rng->normal();
            } else {
                noise = initial_noise(modern_generator);
            }
            auto baseline = equilibrium.species_values[species];
            if (reference_state.has_value() && species < reference_state->size()) {
                baseline = (*reference_state)[species][cell];
            }
            auto value = baseline + noise;
            if (model.require_family(config.family_id).analysis_mode == "stationary_two_species") {
                value = clamp_non_negative(value);
            }
            state.species[species][cell] = value;
        }
    }
}

void compute_generic_drift(
    const PaperModel& model,
    const SimulationConfig& config,
    const GenericState& state,
    double gamma,
    std::vector<std::vector<double>>& drift) {
    const auto species_count = state.species.size();
    const auto n = config.ring.cell_count;
    drift.assign(species_count, std::vector<double>(n, 0.0));
    const auto& family = model.require_family(config.family_id);
    const auto reference_state = reference_state_for_preset(model, config);
    const auto forcing = inferred_reference_forcing(model, config, gamma);
    const auto profile_gain = reference_profile_gain(config);

    for (std::size_t cell = 0; cell < n; ++cell) {
        const auto left = cell == 0 ? n - 1 : cell - 1;
        const auto right = (cell + 1) % n;

        std::vector<double> cell_species(species_count, 0.0);
        for (std::size_t species = 0; species < species_count; ++species) {
            cell_species[species] = state.species[species][cell];
        }

        auto local = model.evaluate_generic_cell(config, cell_species, gamma);

        for (std::size_t species = 0; species < species_count; ++species) {
            drift[species][cell] = local[species];
            const auto diffusion =
                species < family.diffusion_by_species.size() ? family.diffusion_by_species[species] : 0.0;
            drift[species][cell] += diffusion * (
                state.species[species][left]
                - 2.0 * state.species[species][cell]
                + state.species[species][right]);
            if (!forcing.empty() && species < forcing.size()) {
                drift[species][cell] += forcing[species][cell];
            }
            if (reference_state.has_value() && profile_gain > 0.0 && species < reference_state->size()) {
                drift[species][cell] +=
                    profile_gain * ((*reference_state)[species][cell] - state.species[species][cell]);
            }
        }
    }
}

void add_generic_noise(
    const SimulationConfig& config,
    const ModelFamily& family,
    std::mt19937_64* modern_generator,
    HistoricalRng* historical_rng,
    std::vector<std::vector<double>>& increments,
    const GenericState& state,
    double effective_dt) {
    const auto sigma_scale = 0.0025 * config.noise_scale * std::sqrt(effective_dt);
    std::normal_distribution<double> normal(0.0, 1.0);
    const auto species_count = state.species.size();
    const auto n = config.ring.cell_count;

    for (std::size_t species = 0; species < species_count; ++species) {
        for (std::size_t cell = 0; cell < n; ++cell) {
            const auto magnitude = 1.0 + std::abs(state.species[species][cell]);
            const auto sample =
                historical_rng != nullptr ? historical_rng->normal() : normal(*modern_generator);
            increments[species][cell] += sigma_scale * magnitude * sample;
        }
    }

    for (std::size_t species = 0; species < species_count && species < family.diffusion_by_species.size(); ++species) {
        for (std::size_t cell = 0; cell < n; ++cell) {
            const auto next = (cell + 1) % n;
            const auto sample =
                historical_rng != nullptr ? historical_rng->normal() : normal(*modern_generator);
            const auto sigma =
                sigma_scale * family.diffusion_by_species[species]
                * (1.0 + 0.5 * (std::abs(state.species[species][cell]) + std::abs(state.species[species][next])));
            const auto flux = sigma * sample;
            increments[species][cell] -= flux;
            increments[species][next] += flux;
        }
    }
}

template <typename Generator>
std::uint64_t sample_poisson(Generator& generator, double mean) {
    if (mean <= 0.0) {
        return 0;
    }
    std::poisson_distribution<std::uint64_t> distribution(mean);
    return distribution(generator);
}

void add_section10_reaction_noise(
    const PaperModel& model,
    const SimulationConfig& config,
    const std::vector<double>& x,
    const std::vector<double>& y,
    double gamma,
    std::mt19937_64& generator,
    std::vector<double>& increment_x,
    std::vector<double>& increment_y) {
    const auto sigma = 0.004 * config.noise_scale * std::sqrt(config.dt);
    std::normal_distribution<double> normal(0.0, 1.0);

    for (std::size_t index = 0; index < x.size(); ++index) {
        const auto reaction = model.evaluate_section10_cell(x[index], y[index], gamma);
        const auto n_source_x =
            sigma * std::sqrt(std::max(reaction.channels.source_x, 0.0)) * normal(generator);
        const auto n_xy_to_yy =
            sigma * std::sqrt(std::max(reaction.channels.xy_to_yy, 0.0)) * normal(generator);
        const auto n_xx_to_yy =
            sigma * std::sqrt(std::max(reaction.channels.xx_to_yy, 0.0)) * normal(generator);
        const auto n_y_to_x =
            sigma * std::sqrt(std::max(reaction.channels.y_to_x, 0.0)) * normal(generator);
        const auto n_sink_y =
            sigma * std::sqrt(std::max(reaction.channels.sink_y, 0.0)) * normal(generator);

        increment_x[index] += n_source_x - n_xy_to_yy - 2.0 * n_xx_to_yy + n_y_to_x;
        increment_y[index] += n_xy_to_yy + 2.0 * n_xx_to_yy - n_y_to_x - n_sink_y;
    }
}

void add_section10_diffusion_noise(
    const SimulationConfig& config,
    const std::vector<double>& x,
    const std::vector<double>& y,
    std::mt19937_64& generator,
    std::vector<double>& increment_x,
    std::vector<double>& increment_y,
    double effective_dt) {
    std::normal_distribution<double> normal(0.0, 1.0);
    const auto n = x.size();

    for (std::size_t index = 0; index < n; ++index) {
        const auto next = (index + 1) % n;

        const auto sigma_x = 0.004 * config.noise_scale
            * std::sqrt(std::max((x[index] + x[next]) * config.ring.diffusion_x_special * effective_dt, 0.0));
        const auto sigma_y = 0.004 * config.noise_scale
            * std::sqrt(std::max((y[index] + y[next]) * config.ring.diffusion_y_special * effective_dt, 0.0));

        const auto flux_x = sigma_x * normal(generator);
        const auto flux_y = sigma_y * normal(generator);

        increment_x[index] -= flux_x;
        increment_x[next] += flux_x;
        increment_y[index] -= flux_y;
        increment_y[next] += flux_y;
    }
}

void add_section10_reaction_poisson_noise(
    const PaperModel& model,
    const SimulationConfig& config,
    const std::vector<double>& x,
    const std::vector<double>& y,
    double gamma,
    std::mt19937_64& generator,
    std::vector<double>& increment_x,
    std::vector<double>& increment_y,
    std::vector<double>& increment_b,
    double effective_dt) {
    for (std::size_t index = 0; index < x.size(); ++index) {
        const auto reaction = model.evaluate_section10_cell(x[index], y[index], gamma);
        const auto source_mean = std::max(reaction.channels.source_x * effective_dt / kConcentrationQuantum, 0.0);
        const auto xy_mean = std::max(reaction.channels.xy_to_yy * effective_dt / kConcentrationQuantum, 0.0);
        const auto xx_mean = std::max(reaction.channels.xx_to_yy * effective_dt / kConcentrationQuantum, 0.0);
        const auto y_to_x_mean = std::max(reaction.channels.y_to_x * effective_dt / kConcentrationQuantum, 0.0);
        const auto sink_mean = std::max(reaction.channels.sink_y * effective_dt / kConcentrationQuantum, 0.0);

        const auto source_count = sample_poisson(generator, source_mean);
        const auto xy_count = sample_poisson(generator, xy_mean);
        const auto xx_count = sample_poisson(generator, xx_mean);
        const auto y_to_x_count = sample_poisson(generator, y_to_x_mean);
        const auto sink_count = sample_poisson(generator, sink_mean);

        const auto source_delta = kConcentrationQuantum * (static_cast<double>(source_count) - source_mean);
        const auto xy_delta = kConcentrationQuantum * (static_cast<double>(xy_count) - xy_mean);
        const auto xx_delta = kConcentrationQuantum * (static_cast<double>(xx_count) - xx_mean);
        const auto y_to_x_delta = kConcentrationQuantum * (static_cast<double>(y_to_x_count) - y_to_x_mean);
        const auto sink_delta = kConcentrationQuantum * (static_cast<double>(sink_count) - sink_mean);

        increment_x[index] += source_delta - xy_delta - 2.0 * xx_delta + y_to_x_delta;
        increment_y[index] += xy_delta + 2.0 * xx_delta - y_to_x_delta - sink_delta;
        increment_b[index] += kConcentrationQuantum * static_cast<double>(xy_count + xx_count + sink_count);
    }
}

void add_section10_diffusion_poisson_noise(
    const SimulationConfig& config,
    const std::vector<double>& x,
    const std::vector<double>& y,
    std::mt19937_64& generator,
    std::vector<double>& increment_x,
    std::vector<double>& increment_y,
    double effective_dt) {
    const auto n = x.size();
    for (std::size_t index = 0; index < n; ++index) {
        const auto next = (index + 1) % n;
        const auto x_forward_mean =
            std::max(x[index], 0.0) * config.ring.diffusion_x_special * effective_dt / kConcentrationQuantum;
        const auto x_backward_mean =
            std::max(x[next], 0.0) * config.ring.diffusion_x_special * effective_dt / kConcentrationQuantum;
        const auto y_forward_mean =
            std::max(y[index], 0.0) * config.ring.diffusion_y_special * effective_dt / kConcentrationQuantum;
        const auto y_backward_mean =
            std::max(y[next], 0.0) * config.ring.diffusion_y_special * effective_dt / kConcentrationQuantum;

        const auto x_forward = sample_poisson(generator, x_forward_mean);
        const auto x_backward = sample_poisson(generator, x_backward_mean);
        const auto y_forward = sample_poisson(generator, y_forward_mean);
        const auto y_backward = sample_poisson(generator, y_backward_mean);

        const auto x_flux_delta = kConcentrationQuantum
            * ((static_cast<double>(x_forward) - x_forward_mean)
                - (static_cast<double>(x_backward) - x_backward_mean));
        const auto y_flux_delta = kConcentrationQuantum
            * ((static_cast<double>(y_forward) - y_forward_mean)
                - (static_cast<double>(y_backward) - y_backward_mean));

        increment_x[index] -= x_flux_delta;
        increment_x[next] += x_flux_delta;
        increment_y[index] -= y_flux_delta;
        increment_y[next] += y_flux_delta;
    }
}

std::pair<double, double> wilson_interval(std::size_t success, std::size_t total) {
    if (total == 0) {
        return {0.0, 0.0};
    }
    constexpr double z = 1.959963984540054;
    const auto n = static_cast<double>(total);
    const auto p = static_cast<double>(success) / n;
    const auto z2 = z * z;
    const auto denominator = 1.0 + z2 / n;
    const auto center = (p + z2 / (2.0 * n)) / denominator;
    const auto margin =
        z * std::sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denominator;
    return {
        std::max(0.0, center - margin),
        std::min(1.0, center + margin)
    };
}

SimulationResult simulate_generic_family(
    const PaperModel& model,
    const LinearStabilityAnalyzer& analyzer,
    SimulationConfig config) {
    config = effective_incipient_capture_config(std::move(config));
    SimulationResult result;
    result.config = config;
    result.initial_analysis = analyzer.analyze(model, config, config.schedule.start_value());

    std::mt19937_64 modern_generator(config.seed);
    std::optional<HistoricalRng> historical_rng;
    auto historical_profile = HistoricalProfileSettings{};
    if (config.execution_mode == "historical_paper_constrained") {
        historical_profile = historical_profile_settings(config);
        historical_rng.emplace(config.seed);
    }

    GenericState state;
    initialize_generic_state(
        model,
        config,
        modern_generator,
        historical_rng.has_value() ? &*historical_rng : nullptr,
        state);

    auto current_time = 0.0;
    auto step = std::size_t{0};
    const auto capture_stride = std::max<std::size_t>(config.capture_stride, 1);
    const auto include_chemistry = false;

    result.initial_snapshot = make_snapshot(
        model,
        config,
        current_time,
        config.schedule.start_value(),
        state.species,
        state.b,
        include_chemistry);
    result.mode_trace.push_back(make_mode_trace_sample(
        model,
        config,
        current_time,
        config.schedule.start_value(),
        state.species));
    result.snapshot_trace.push_back(result.initial_snapshot);

    auto incipient_recorded = false;
    IncipientCandidate mode234_candidate;

    std::vector<std::vector<double>> drift;
    std::vector<std::vector<double>> predictor;
    std::vector<std::vector<double>> next_state;
    std::vector<std::vector<double>> noise;

    while (current_time + 1.0e-12 < config.total_time) {
        const auto effective_dt = std::min(config.dt, config.total_time - current_time);
        const auto gamma = config.schedule.value_at(current_time);
        const auto next_time = current_time + effective_dt;
        const auto gamma_next = config.schedule.value_at(next_time);

        compute_generic_drift(model, config, state, gamma, drift);
        predictor = state.species;
        for (std::size_t species = 0; species < predictor.size(); ++species) {
            for (std::size_t cell = 0; cell < predictor[species].size(); ++cell) {
                predictor[species][cell] += effective_dt * drift[species][cell];
            }
        }

        noise.assign(state.species.size(), std::vector<double>(config.ring.cell_count, 0.0));
        if (config.enable_noise) {
            add_generic_noise(
                config,
                model.require_family(config.family_id),
                historical_rng.has_value() ? nullptr : &modern_generator,
                historical_rng.has_value() ? &*historical_rng : nullptr,
                noise,
                state,
                effective_dt);
        }

        next_state = state.species;
        if (config.execution_mode == "historical_paper_constrained") {
            if (historical_profile.sequential_updates) {
                GenericState sequential = state;
                for (std::size_t cell = 0; cell < config.ring.cell_count; ++cell) {
                    std::vector<std::vector<double>> sequential_drift;
                    compute_generic_drift(model, config, sequential, gamma, sequential_drift);
                    for (std::size_t species = 0; species < next_state.size(); ++species) {
                        auto value = sequential.species[species][cell]
                            + effective_dt * sequential_drift[species][cell]
                            + noise[species][cell];
                        if (model.require_family(config.family_id).analysis_mode == "stationary_two_species") {
                            value = clamp_non_negative(value);
                        }
                        sequential.species[species][cell] =
                            round_historical(value, historical_profile.rounding_digits);
                    }
                }
                next_state = sequential.species;
            } else {
                for (std::size_t species = 0; species < next_state.size(); ++species) {
                    for (std::size_t cell = 0; cell < next_state[species].size(); ++cell) {
                        auto value = state.species[species][cell]
                            + effective_dt * drift[species][cell]
                            + noise[species][cell];
                        if (model.require_family(config.family_id).analysis_mode == "stationary_two_species") {
                            value = clamp_non_negative(value);
                        }
                        next_state[species][cell] =
                            round_historical(value, historical_profile.rounding_digits);
                    }
                }
            }
        } else {
            GenericState predictor_state{predictor, state.b};
            std::vector<std::vector<double>> drift_2;
            compute_generic_drift(model, config, predictor_state, gamma_next, drift_2);
            for (std::size_t species = 0; species < next_state.size(); ++species) {
                for (std::size_t cell = 0; cell < next_state[species].size(); ++cell) {
                    auto value = state.species[species][cell]
                        + 0.5 * effective_dt * (drift[species][cell] + drift_2[species][cell])
                        + noise[species][cell];
                    if (model.require_family(config.family_id).analysis_mode == "stationary_two_species") {
                        value = clamp_non_negative(value);
                    }
                    next_state[species][cell] = value;
                }
            }
        }

        if (next_state.size() > 1) {
            record_arrest_if_needed(result.arrest, next_time, next_state[1]);
        }

        if (config.incipient_capture_mode == "mode234_threshold") {
            update_mode234_candidate(
                model,
                config,
                next_time,
                gamma_next,
                next_state,
                state.b,
                include_chemistry,
                mode234_candidate);
        }

        if (!incipient_recorded) {
            if (config.incipient_capture_mode == "mode234_threshold") {
                if (should_capture_incipient_on_mode234(model, config, next_time, next_state)) {
                    result.incipient_snapshot = mode234_candidate.snapshot;
                    incipient_recorded = true;
                }
            } else if (should_capture_incipient_on_gamma(config, gamma, gamma_next)) {
                result.incipient_snapshot =
                    make_snapshot(model, config, next_time, gamma_next, next_state, state.b, include_chemistry);
                incipient_recorded = true;
            }
        }

        ++step;
        if (step % capture_stride == 0 || next_time + 1.0e-12 >= config.total_time) {
            result.mode_trace.push_back(make_mode_trace_sample(model, config, next_time, gamma_next, next_state));
            result.snapshot_trace.push_back(make_snapshot(
                model, config, next_time, gamma_next, next_state, state.b, include_chemistry));
        }

        state.species.swap(next_state);
        current_time = next_time;
    }

    if (!incipient_recorded) {
        if (config.incipient_capture_mode == "mode234_threshold" && mode234_candidate.has_snapshot) {
            result.incipient_snapshot = mode234_candidate.snapshot;
        } else {
            result.incipient_snapshot = make_snapshot(
                model,
                config,
                config.total_time,
                config.schedule.value_at(config.total_time),
                state.species,
                state.b,
                include_chemistry);
        }
    }

    result.final_snapshot = make_snapshot(
        model,
        config,
        config.total_time,
        config.schedule.value_at(config.total_time),
        state.species,
        state.b,
        include_chemistry);
    result.incipient_analysis = analyzer.analyze(model, config, result.incipient_snapshot.gamma);
    if (result.final_snapshot.species.size() > 1) {
        result.arrest.final_zero_cell_count = count_zero_cells(result.final_snapshot.species[1].values);
    }
    result.metrics = build_metrics(
        model,
        config,
        result.incipient_analysis,
        result.incipient_snapshot,
        result.final_snapshot,
        result.mode_trace);
    return result;
}

SimulationResult simulate_section10_reduced(
    const PaperModel& model,
    const LinearStabilityAnalyzer& analyzer,
    SimulationConfig config) {
    config = effective_incipient_capture_config(std::move(config));
    if (config.execution_mode == "historical_paper_constrained") {
        return simulate_generic_family(model, analyzer, std::move(config));
    }

    SimulationResult result;
    result.config = config;
    result.initial_analysis = analyzer.analyze(model, config, config.schedule.start_value());

    const auto initial_equilibrium = model.solve_equilibrium(config);
    const auto n = config.ring.cell_count;
    std::vector<double> x(n, initial_equilibrium.x);
    std::vector<double> y(n, initial_equilibrium.y);
    std::vector<double> b(n, 0.0);

    std::mt19937_64 generator(config.seed);
    std::normal_distribution<double> initial_noise(0.0, config.initial_perturbation_scale);
    for (std::size_t index = 0; index < n; ++index) {
        x[index] = clamp_non_negative(x[index] + initial_noise(generator));
        y[index] = clamp_non_negative(y[index] + initial_noise(generator));
    }

    auto current_time = 0.0;
    auto step = std::size_t{0};
    const auto capture_stride = std::max<std::size_t>(config.capture_stride, 1);

    auto current_state = std::vector<std::vector<double>>{x, y};
    result.initial_snapshot = make_snapshot(
        model, config, current_time, config.schedule.start_value(), current_state, b, false);
    result.mode_trace.push_back(
        make_mode_trace_sample(model, config, current_time, config.schedule.start_value(), current_state));
    result.snapshot_trace.push_back(result.initial_snapshot);

    auto incipient_recorded = false;
    IncipientCandidate mode234_candidate;

    std::vector<double> predictor_x(n, 0.0);
    std::vector<double> predictor_y(n, 0.0);
    std::vector<double> next_x(n, 0.0);
    std::vector<double> next_y(n, 0.0);
    std::vector<double> noise_x(n, 0.0);
    std::vector<double> noise_y(n, 0.0);
    std::vector<std::vector<double>> generic_state(2);

    while (current_time + 1.0e-12 < config.total_time) {
        const auto effective_dt = std::min(config.dt, config.total_time - current_time);
        const auto gamma = config.schedule.value_at(current_time);
        const auto next_time = current_time + effective_dt;
        const auto gamma_next = config.schedule.value_at(next_time);

        std::vector<double> drift_x(n, 0.0);
        std::vector<double> drift_y(n, 0.0);
        for (std::size_t index = 0; index < n; ++index) {
            const auto left = index == 0 ? n - 1 : index - 1;
            const auto right = (index + 1) % n;
            const auto reaction = model.evaluate_section10_cell(x[index], y[index], gamma);
            drift_x[index] = reaction.dx + config.ring.diffusion_x_special * (x[left] - 2.0 * x[index] + x[right]);
            drift_y[index] = reaction.dy + config.ring.diffusion_y_special * (y[left] - 2.0 * y[index] + y[right]);
            predictor_x[index] = clamp_non_negative(x[index] + effective_dt * drift_x[index]);
            predictor_y[index] = clamp_non_negative(y[index] + effective_dt * drift_y[index]);
        }

        std::vector<double> drift_x2(n, 0.0);
        std::vector<double> drift_y2(n, 0.0);
        for (std::size_t index = 0; index < n; ++index) {
            const auto left = index == 0 ? n - 1 : index - 1;
            const auto right = (index + 1) % n;
            const auto reaction = model.evaluate_section10_cell(predictor_x[index], predictor_y[index], gamma_next);
            drift_x2[index] = reaction.dx
                + config.ring.diffusion_x_special * (predictor_x[left] - 2.0 * predictor_x[index] + predictor_x[right]);
            drift_y2[index] = reaction.dy
                + config.ring.diffusion_y_special * (predictor_y[left] - 2.0 * predictor_y[index] + predictor_y[right]);
        }

        std::fill(noise_x.begin(), noise_x.end(), 0.0);
        std::fill(noise_y.begin(), noise_y.end(), 0.0);
        if (config.enable_noise) {
            add_section10_reaction_noise(model, config, x, y, gamma, generator, noise_x, noise_y);
            add_section10_diffusion_noise(config, x, y, generator, noise_x, noise_y, effective_dt);
        }

        for (std::size_t index = 0; index < n; ++index) {
            next_x[index] = clamp_non_negative(
                x[index] + 0.5 * effective_dt * (drift_x[index] + drift_x2[index]) + noise_x[index]);
            next_y[index] = clamp_non_negative(
                y[index] + 0.5 * effective_dt * (drift_y[index] + drift_y2[index]) + noise_y[index]);
        }

        record_arrest_if_needed(result.arrest, next_time, next_y);

        generic_state[0] = next_x;
        generic_state[1] = next_y;
        if (config.incipient_capture_mode == "mode234_threshold") {
            update_mode234_candidate(model, config, next_time, gamma_next, generic_state, b, false, mode234_candidate);
        }
        if (!incipient_recorded) {
            if (config.incipient_capture_mode == "mode234_threshold") {
                if (should_capture_incipient_on_mode234(model, config, next_time, generic_state)) {
                    result.incipient_snapshot = mode234_candidate.snapshot;
                    incipient_recorded = true;
                }
            } else if (should_capture_incipient_on_gamma(config, gamma, gamma_next)) {
                result.incipient_snapshot = make_snapshot(
                    model, config, next_time, gamma_next, generic_state, b, false);
                incipient_recorded = true;
            }
        }

        ++step;
        if (step % capture_stride == 0 || next_time + 1.0e-12 >= config.total_time) {
            result.mode_trace.push_back(make_mode_trace_sample(model, config, next_time, gamma_next, generic_state));
            result.snapshot_trace.push_back(make_snapshot(
                model, config, next_time, gamma_next, generic_state, b, false));
        }

        x.swap(next_x);
        y.swap(next_y);
        current_time = next_time;
    }

    generic_state[0] = x;
    generic_state[1] = y;
    if (!incipient_recorded) {
        if (config.incipient_capture_mode == "mode234_threshold" && mode234_candidate.has_snapshot) {
            result.incipient_snapshot = mode234_candidate.snapshot;
        } else {
            result.incipient_snapshot = make_snapshot(
                model,
                config,
                config.total_time,
                config.schedule.value_at(config.total_time),
                generic_state,
                b,
                false);
        }
    }

    result.final_snapshot = make_snapshot(
        model,
        config,
        config.total_time,
        config.schedule.value_at(config.total_time),
        generic_state,
        b,
        false);
    result.incipient_analysis = analyzer.analyze(model, config, result.incipient_snapshot.gamma);
    result.arrest.final_zero_cell_count = count_zero_cells(result.final_snapshot.y);
    result.metrics = build_metrics(
        model,
        config,
        result.incipient_analysis,
        result.incipient_snapshot,
        result.final_snapshot,
        result.mode_trace);
    return result;
}

SimulationResult simulate_section10_full_chemistry(
    const PaperModel& model,
    const LinearStabilityAnalyzer& analyzer,
    SimulationConfig config) {
    config = effective_incipient_capture_config(std::move(config));
    if (config.execution_mode == "historical_paper_constrained") {
        throw std::invalid_argument("Historical execution is not supported for full_chemistry.");
    }

    SimulationResult result;
    result.config = config;
    result.initial_analysis = analyzer.analyze(model, config, config.schedule.start_value());

    const auto initial_equilibrium = model.solve_equilibrium(config);
    const auto n = config.ring.cell_count;
    std::vector<double> x(n, initial_equilibrium.x);
    std::vector<double> y(n, initial_equilibrium.y);
    std::vector<double> b(n, 0.0);

    std::mt19937_64 generator(config.seed);
    std::normal_distribution<double> initial_noise(0.0, config.initial_perturbation_scale);
    for (std::size_t index = 0; index < n; ++index) {
        x[index] = clamp_non_negative(x[index] + initial_noise(generator));
        y[index] = clamp_non_negative(y[index] + initial_noise(generator));
    }

    auto current_time = 0.0;
    auto step = std::size_t{0};
    const auto capture_stride = std::max<std::size_t>(config.capture_stride, 1);
    auto current_state = std::vector<std::vector<double>>{x, y};

    result.initial_snapshot = make_snapshot(
        model, config, current_time, config.schedule.start_value(), current_state, b, true);
    result.mode_trace.push_back(
        make_mode_trace_sample(model, config, current_time, config.schedule.start_value(), current_state));
    result.snapshot_trace.push_back(result.initial_snapshot);

    auto incipient_recorded = false;
    IncipientCandidate mode234_candidate;

    std::vector<double> predictor_x(n, 0.0);
    std::vector<double> predictor_y(n, 0.0);
    std::vector<double> next_x(n, 0.0);
    std::vector<double> next_y(n, 0.0);
    std::vector<double> next_b(n, 0.0);
    std::vector<double> reaction_noise_x(n, 0.0);
    std::vector<double> reaction_noise_y(n, 0.0);
    std::vector<double> b_increment(n, 0.0);

    while (current_time + 1.0e-12 < config.total_time) {
        const auto effective_dt = std::min(config.dt, config.total_time - current_time);
        const auto gamma = config.schedule.value_at(current_time);
        const auto next_time = current_time + effective_dt;
        const auto gamma_next = config.schedule.value_at(next_time);

        std::vector<double> drift_x(n, 0.0);
        std::vector<double> drift_y(n, 0.0);
        for (std::size_t index = 0; index < n; ++index) {
            const auto left = index == 0 ? n - 1 : index - 1;
            const auto right = (index + 1) % n;
            const auto reaction = model.evaluate_section10_cell(x[index], y[index], gamma);
            drift_x[index] = reaction.dx + config.ring.diffusion_x_special * (x[left] - 2.0 * x[index] + x[right]);
            drift_y[index] = reaction.dy + config.ring.diffusion_y_special * (y[left] - 2.0 * y[index] + y[right]);
            predictor_x[index] = clamp_non_negative(x[index] + effective_dt * drift_x[index]);
            predictor_y[index] = clamp_non_negative(y[index] + effective_dt * drift_y[index]);
        }

        std::vector<double> drift_x2(n, 0.0);
        std::vector<double> drift_y2(n, 0.0);
        for (std::size_t index = 0; index < n; ++index) {
            const auto left = index == 0 ? n - 1 : index - 1;
            const auto right = (index + 1) % n;
            const auto reaction = model.evaluate_section10_cell(predictor_x[index], predictor_y[index], gamma_next);
            drift_x2[index] = reaction.dx
                + config.ring.diffusion_x_special * (predictor_x[left] - 2.0 * predictor_x[index] + predictor_x[right]);
            drift_y2[index] = reaction.dy
                + config.ring.diffusion_y_special * (predictor_y[left] - 2.0 * predictor_y[index] + predictor_y[right]);
        }

        std::fill(reaction_noise_x.begin(), reaction_noise_x.end(), 0.0);
        std::fill(reaction_noise_y.begin(), reaction_noise_y.end(), 0.0);
        std::fill(b_increment.begin(), b_increment.end(), 0.0);
        if (config.enable_noise) {
            add_section10_reaction_poisson_noise(
                model,
                config,
                x,
                y,
                gamma,
                generator,
                reaction_noise_x,
                reaction_noise_y,
                b_increment,
                effective_dt);
            add_section10_diffusion_poisson_noise(
                config,
                x,
                y,
                generator,
                reaction_noise_x,
                reaction_noise_y,
                effective_dt);
        } else {
            for (std::size_t index = 0; index < n; ++index) {
                const auto reaction = model.evaluate_section10_cell(x[index], y[index], gamma);
                b_increment[index] = effective_dt * (
                    reaction.channels.xy_to_yy
                    + reaction.channels.xx_to_yy
                    + reaction.channels.sink_y);
            }
        }

        for (std::size_t index = 0; index < n; ++index) {
            next_x[index] = clamp_non_negative(
                x[index] + 0.5 * effective_dt * (drift_x[index] + drift_x2[index]) + reaction_noise_x[index]);
            next_y[index] = clamp_non_negative(
                y[index] + 0.5 * effective_dt * (drift_y[index] + drift_y2[index]) + reaction_noise_y[index]);
            next_b[index] = clamp_non_negative(b[index] + b_increment[index]);
        }

        record_arrest_if_needed(result.arrest, next_time, next_y);
        current_state[0] = next_x;
        current_state[1] = next_y;

        if (config.incipient_capture_mode == "mode234_threshold") {
            update_mode234_candidate(model, config, next_time, gamma_next, current_state, next_b, true, mode234_candidate);
        }
        if (!incipient_recorded) {
            if (config.incipient_capture_mode == "mode234_threshold") {
                if (should_capture_incipient_on_mode234(model, config, next_time, current_state)) {
                    result.incipient_snapshot = mode234_candidate.snapshot;
                    incipient_recorded = true;
                }
            } else if (should_capture_incipient_on_gamma(config, gamma, gamma_next)) {
                result.incipient_snapshot = make_snapshot(
                    model, config, next_time, gamma_next, current_state, next_b, true);
                incipient_recorded = true;
            }
        }

        ++step;
        if (step % capture_stride == 0 || next_time + 1.0e-12 >= config.total_time) {
            result.mode_trace.push_back(make_mode_trace_sample(model, config, next_time, gamma_next, current_state));
            result.snapshot_trace.push_back(make_snapshot(
                model, config, next_time, gamma_next, current_state, next_b, true));
        }

        x.swap(next_x);
        y.swap(next_y);
        b.swap(next_b);
        current_time = next_time;
    }

    current_state[0] = x;
    current_state[1] = y;
    if (!incipient_recorded) {
        if (config.incipient_capture_mode == "mode234_threshold" && mode234_candidate.has_snapshot) {
            result.incipient_snapshot = mode234_candidate.snapshot;
        } else {
            result.incipient_snapshot = make_snapshot(
                model,
                config,
                config.total_time,
                config.schedule.value_at(config.total_time),
                current_state,
                b,
                true);
        }
    }

    result.final_snapshot = make_snapshot(
        model, config, config.total_time, config.schedule.value_at(config.total_time), current_state, b, true);
    result.incipient_analysis = analyzer.analyze(model, config, result.incipient_snapshot.gamma);
    result.arrest.final_zero_cell_count = count_zero_cells(result.final_snapshot.y);
    result.metrics = build_metrics(
        model,
        config,
        result.incipient_analysis,
        result.incipient_snapshot,
        result.final_snapshot,
        result.mode_trace);
    return result;
}

SimulationResult simulate_dispatch(
    const PaperModel& model,
    const LinearStabilityAnalyzer& analyzer,
    SimulationConfig config) {
    const auto& family = model.require_family(config.family_id);
    if (config.engine == "full_chemistry") {
        if (family.kind != FamilyKind::section10_example1) {
            throw std::invalid_argument("full_chemistry is only supported for the Section 10 family.");
        }
        return simulate_section10_full_chemistry(model, analyzer, std::move(config));
    }
    if (family.kind == FamilyKind::section10_example1) {
        return simulate_section10_reduced(model, analyzer, std::move(config));
    }
    return simulate_generic_family(model, analyzer, std::move(config));
}

}  // namespace

RingExperiment::RingExperiment() = default;

SimulationResult RingExperiment::simulate(const SimulationConfig& config) const {
    validate_config(config);
    return simulate_dispatch(model_, analyzer_, config);
}

BatchResult RingExperiment::simulate_batch(const SimulationConfig& config, std::size_t replicates) const {
    validate_config(config);

    BatchResult batch;
    batch.config = config;
    batch.replicates = replicates;
    batch.incipient_analysis = analyzer_.analyze(model_, config, config.incipient_capture_gamma);
    batch.dominant_mode_histogram.assign(config.ring.cell_count / 2 + 1, 0);
    if (replicates == 0) {
        return batch;
    }

    std::vector<BatchRunSummary> all_summaries(replicates);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (std::int64_t replicate = 0; replicate < static_cast<std::int64_t>(replicates); ++replicate) {
        auto local_config = config;
        local_config.seed = config.seed + static_cast<std::uint64_t>(replicate);
        local_config.capture_stride = std::max<std::size_t>(config.capture_stride, 50);
        const auto result = simulate_dispatch(model_, analyzer_, local_config);

        BatchRunSummary summary;
        summary.seed = local_config.seed;
        summary.dominant_mode = result.metrics.dominant_mode;
        summary.dominant_classification = result.metrics.dominant_classification;
        summary.dominant_amplitude = result.metrics.dominant_amplitude;
        summary.regularity_index = result.metrics.regularity_index;
        summary.mode3_to_mode4_ratio = result.metrics.mode3_to_mode4_ratio;
        all_summaries[static_cast<std::size_t>(replicate)] = summary;
    }

    double regularity_sum = 0.0;
    double ratio_sum = 0.0;
    std::size_t mode3_count = 0;
    std::size_t mode4_count = 0;

    for (const auto& summary : all_summaries) {
        if (summary.dominant_mode < batch.dominant_mode_histogram.size()) {
            batch.dominant_mode_histogram[summary.dominant_mode] += 1;
        }
        regularity_sum += summary.regularity_index;
        ratio_sum += summary.mode3_to_mode4_ratio;
        mode3_count += summary.dominant_mode == 3 ? 1 : 0;
        mode4_count += summary.dominant_mode == 4 ? 1 : 0;
    }

    batch.mean_regularity_index = regularity_sum / static_cast<double>(replicates);
    batch.mean_mode3_to_mode4_ratio = ratio_sum / static_cast<double>(replicates);
    batch.share_mode3 = static_cast<double>(mode3_count) / static_cast<double>(replicates);
    batch.share_mode4 = static_cast<double>(mode4_count) / static_cast<double>(replicates);

    const auto mode3_interval = wilson_interval(mode3_count, replicates);
    const auto mode4_interval = wilson_interval(mode4_count, replicates);
    batch.share_mode3_ci_low = mode3_interval.first;
    batch.share_mode3_ci_high = mode3_interval.second;
    batch.share_mode4_ci_low = mode4_interval.first;
    batch.share_mode4_ci_high = mode4_interval.second;

    const auto sample_count = std::min<std::size_t>(replicates, 64);
    batch.samples.assign(all_summaries.begin(), all_summaries.begin() + static_cast<std::ptrdiff_t>(sample_count));
    return batch;
}

}  // namespace turing
