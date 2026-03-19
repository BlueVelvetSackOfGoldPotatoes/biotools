#include "turing/analysis.hpp"
#include "turing/json.hpp"
#include "turing/model.hpp"
#include "turing/simulator.hpp"

#include <boost/json.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct FailureCounter {
    int value = 0;

    void require(bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << "FAIL: " << message << '\n';
            ++value;
        }
    }

    void require_close(double actual, double expected, double tolerance, const std::string& message) {
        require(std::abs(actual - expected) <= tolerance, message);
    }
};

const turing::ReferenceComparison* find_comparison(
    const turing::SimulationResult& result,
    const std::string& reference_id) {
    for (const auto& comparison : result.metrics.comparisons) {
        if (comparison.reference_id == reference_id) {
            return &comparison;
        }
    }
    return nullptr;
}

double histogram_mass_mode3_mode4(const turing::BatchResult& batch) {
    if (batch.replicates == 0 || batch.dominant_mode_histogram.size() <= 4) {
        return 0.0;
    }
    return static_cast<double>(batch.dominant_mode_histogram[3] + batch.dominant_mode_histogram[4])
        / static_cast<double>(batch.replicates);
}

const std::vector<double>* find_series(
    const std::vector<turing::NamedSeries>& series,
    const std::string& name) {
    for (const auto& entry : series) {
        if (entry.name == name) {
            return &entry.values;
        }
    }
    return nullptr;
}

double max_abs_difference(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    double best = 0.0;
    const auto limit = std::min(lhs.size(), rhs.size());
    for (std::size_t index = 0; index < limit; ++index) {
        best = std::max(best, std::abs(lhs[index] - rhs[index]));
    }
    return best;
}

}  // namespace

int main() {
    FailureCounter failures;
    const auto pi = std::acos(-1.0);

    turing::PresetRegistry registry;
    turing::PaperModel model;
    turing::LinearStabilityAnalyzer analyzer;
    turing::RingExperiment experiment;

    const auto* quick_preset = registry.find("paper-quick");
    const auto* slow_preset = registry.find("paper-slow");
    const auto* table2_preset = registry.find("paper-table2-stable-ring");
    const auto* wave_e_preset = registry.find("paper-wave-e-travelling");
    const auto* wave_f_preset = registry.find("paper-wave-f-short-oscillation");

    failures.require(quick_preset != nullptr, "paper-quick preset exists");
    failures.require(slow_preset != nullptr, "paper-slow preset exists");
    failures.require(table2_preset != nullptr, "paper-table2-stable-ring preset exists");
    failures.require(wave_e_preset != nullptr, "paper-wave-e-travelling preset exists");
    failures.require(wave_f_preset != nullptr, "paper-wave-f-short-oscillation preset exists");
    if (quick_preset == nullptr || slow_preset == nullptr || table2_preset == nullptr
        || wave_e_preset == nullptr || wave_f_preset == nullptr) {
        return 1;
    }

    const auto analysis = analyzer.analyze(model, quick_preset->config, 0.0);
    failures.require(analysis.dominant_mode == 3, "gamma=0 dominant mode is 3");
    failures.require_close(analysis.equilibrium.x, 1.0, 1.0e-12, "gamma=0 equilibrium X is 1");
    failures.require_close(analysis.equilibrium.y, 1.0, 1.0e-12, "gamma=0 equilibrium Y is 1");
    failures.require_close(analysis.jacobian.a, -2.0, 1.0e-12, "jacobian a matches paper");
    failures.require_close(analysis.jacobian.b, -1.5625, 1.0e-12, "jacobian b matches paper");
    failures.require_close(analysis.jacobian.c, 2.0, 1.0e-12, "jacobian c matches paper");
    failures.require_close(analysis.jacobian.d, 1.5, 1.0e-12, "jacobian d matches paper");
    failures.require_close(analysis.mode34_gap, 0.0084, 0.0006, "mode 3/4 gap is near paper value");
    failures.require(
        analysis.mode34_efold_time_special_units >= 110.0
            && analysis.mode34_efold_time_special_units <= 125.0,
        "mode 3/4 e-fold time is within expected range");

    auto quick_reduced = quick_preset->config;
    quick_reduced.seed = 1;
    const auto reduced_result = experiment.simulate(quick_reduced);
    const auto* reduced_final = find_comparison(reduced_result, "paper_first_final");
    failures.require(reduced_result.metrics.dominant_mode == 3, "reduced quick seed 1 ends in mode 3");
    failures.require(reduced_result.arrest.observed, "reduced quick seed 1 records arrest");
    failures.require(
        reduced_final != nullptr && reduced_final->rmse_combined <= 0.03,
        "reduced quick seed 1 matches the first final pattern closely");

    auto quick_full = quick_preset->config;
    quick_full.engine = "full_chemistry";
    quick_full.seed = 1;
    const auto full_result = experiment.simulate(quick_full);
    const auto* full_final = find_comparison(full_result, "paper_first_final");
    failures.require(full_result.metrics.dominant_mode == 3, "full chemistry quick seed 1 ends in mode 3");
    failures.require(full_result.arrest.observed, "full chemistry quick seed 1 records arrest");
    failures.require(
        full_final != nullptr && full_final->rmse_combined <= 0.03,
        "full chemistry quick seed 1 matches the first final pattern closely");

    const auto reduced_quick_batch = experiment.simulate_batch(quick_preset->config, 32);
    failures.require(
        reduced_quick_batch.share_mode3 > reduced_quick_batch.share_mode4,
        "reduced quick batch prefers mode 3 over mode 4");
    failures.require(
        histogram_mass_mode3_mode4(reduced_quick_batch) >= 0.90,
        "reduced quick batch is concentrated in modes 3 and 4");

    auto full_quick_batch_config = quick_preset->config;
    full_quick_batch_config.engine = "full_chemistry";
    const auto full_quick_batch = experiment.simulate_batch(full_quick_batch_config, 32);
    failures.require(
        full_quick_batch.share_mode3 > full_quick_batch.share_mode4,
        "full chemistry quick batch prefers mode 3 over mode 4");

    auto full_slow_batch_config = slow_preset->config;
    full_slow_batch_config.engine = "full_chemistry";
    const auto full_slow_batch = experiment.simulate_batch(full_slow_batch_config, 32);
    failures.require(
        full_slow_batch.share_mode3 >= 0.90,
        "full chemistry slow batch is strongly mode-3 dominated");
    failures.require(
        full_slow_batch.mean_regularity_index > full_quick_batch.mean_regularity_index,
        "full chemistry slow batch is cleaner than quick batch");

    const auto table2_analysis = analyzer.analyze(model, table2_preset->config, 0.0);
    failures.require(
        table2_analysis.family_id == "example2_table2",
        "Table 2 preset resolves to the second paper family");
    failures.require_close(
        table2_analysis.equilibrium.x,
        16.0 / 12.0,
        1.0e-9,
        "Table 2 example homogeneous equilibrium X matches k = 12");
    failures.require_close(
        table2_analysis.equilibrium.y,
        12.0,
        1.0e-9,
        "Table 2 example homogeneous equilibrium Y matches k = 12");

    auto table2_config = table2_preset->config;
    table2_config.enable_noise = false;
    const auto table2_result = experiment.simulate(table2_config);
    const auto table2_reference = model.reference_species_for_preset(table2_config);
    const auto* table2_x = find_series(table2_result.final_snapshot.species, "X");
    const auto* table2_y = find_series(table2_result.final_snapshot.species, "Y");
    const auto* table2_ref_x = find_series(table2_reference, "X");
    const auto* table2_ref_y = find_series(table2_reference, "Y");
    failures.require(table2_x != nullptr && table2_ref_x != nullptr, "Table 2 X series available");
    failures.require(table2_y != nullptr && table2_ref_y != nullptr, "Table 2 Y series available");
    if (table2_x != nullptr && table2_ref_x != nullptr) {
        failures.require(
            max_abs_difference(*table2_x, *table2_ref_x) <= 0.35,
            "Table 2 stable ring stays close to the reference X profile");
    }
    if (table2_y != nullptr && table2_ref_y != nullptr) {
        failures.require(
            max_abs_difference(*table2_y, *table2_ref_y) <= 0.35,
            "Table 2 stable ring stays close to the reference Y profile");
    }
    const auto* table2_initial_x = find_series(table2_result.initial_snapshot.species, "X");
    const auto* table2_initial_y = find_series(table2_result.initial_snapshot.species, "Y");
    if (table2_initial_x != nullptr && table2_ref_x != nullptr && table2_x != nullptr) {
        failures.require(
            max_abs_difference(*table2_x, *table2_ref_x)
                <= max_abs_difference(*table2_initial_x, *table2_ref_x),
            "Table 2 modern run relaxes X toward the stable reference");
    }
    if (table2_initial_y != nullptr && table2_ref_y != nullptr && table2_y != nullptr) {
        failures.require(
            max_abs_difference(*table2_y, *table2_ref_y)
                <= max_abs_difference(*table2_initial_y, *table2_ref_y),
            "Table 2 modern run relaxes Y toward the stable reference");
    }

    auto historical_table2_config = table2_config;
    historical_table2_config.execution_mode = "historical_paper_constrained";
    historical_table2_config.execution_profile_id = "historic_1952_baseline";
    const auto historical_table2_result = experiment.simulate(historical_table2_config);
    const auto* historical_table2_x = find_series(historical_table2_result.final_snapshot.species, "X");
    const auto* historical_table2_y = find_series(historical_table2_result.final_snapshot.species, "Y");
    if (historical_table2_x != nullptr && table2_ref_x != nullptr) {
        failures.require(
            max_abs_difference(*historical_table2_x, *table2_ref_x) <= 0.01,
            "Table 2 historical run stays close to the reference X profile");
    }
    if (historical_table2_y != nullptr && table2_ref_y != nullptr) {
        failures.require(
            max_abs_difference(*historical_table2_y, *table2_ref_y) <= 0.01,
            "Table 2 historical run stays close to the reference Y profile");
    }

    const auto wave_e_analysis = analyzer.analyze(model, wave_e_preset->config, 0.0);
    failures.require(
        wave_e_analysis.dominant_classification == "oscillatory",
        "Case (e) classifies as oscillatory");
    failures.require(
        wave_e_analysis.dominant_frequency_cycles_per_time > 0.01,
        "Case (e) has nonzero oscillation frequency");
    failures.require(
        wave_e_analysis.dominant_phase_velocity_cells_per_time > 0.01,
        "Case (e) has positive travelling-wave speed");

    const auto wave_f_analysis = analyzer.analyze(model, wave_f_preset->config, 0.0);
    failures.require(
        wave_f_analysis.dominant_classification == "oscillatory",
        "Case (f) classifies as oscillatory");
    failures.require(
        wave_f_analysis.dominant_mode >= 4,
        "Case (f) prefers a short-wave oscillatory mode");
    failures.require(
        wave_f_analysis.dominant_growth > 0.0,
        "Case (f) now has a genuinely unstable short-wave oscillatory mode");

    auto wave_f_simulation = wave_f_preset->config;
    wave_f_simulation.enable_noise = false;
    const auto wave_f_result = experiment.simulate(wave_f_simulation);
    failures.require(
        std::abs(std::abs(wave_f_result.metrics.neighbor_phase_offset_radians) - pi) <= 0.8,
        "Case (f) neighbour phase offset is near antiphase");

    auto historical = quick_preset->config;
    historical.execution_mode = "historical_paper_constrained";
    historical.execution_profile_id = "historic_1952_baseline";
    historical.seed = 7;
    const auto historical_first = experiment.simulate(historical);
    const auto historical_second = experiment.simulate(historical);
    failures.require(
        max_abs_difference(historical_first.final_snapshot.x, historical_second.final_snapshot.x) <= 1.0e-12,
        "historical execution is reproducible for fixed seed");
    failures.require(
        max_abs_difference(historical_first.final_snapshot.y, historical_second.final_snapshot.y) <= 1.0e-12,
        "historical execution reproduces Y profiles exactly for fixed seed");

    auto historical_coarse = historical;
    historical_coarse.execution_profile_id = "historic_1952_coarse_rounding";
    const auto historical_coarse_result = experiment.simulate(historical_coarse);
    failures.require(
        max_abs_difference(historical_first.final_snapshot.y, historical_coarse_result.final_snapshot.y) > 1.0e-6,
        "historical coarse rounding diverges from the baseline profile");

    auto slow_full = slow_preset->config;
    slow_full.engine = "full_chemistry";
    slow_full.seed = 6;
    const auto slow_full_result = experiment.simulate(slow_full);
    const auto* slow_full_incipient = find_comparison(slow_full_result, "paper_slow_incipient_y");
    failures.require(
        slow_full_incipient != nullptr && slow_full_incipient->rmse_combined <= 0.12,
        "full chemistry slow incipient capture now matches the Table 1 slow row");

    const boost::json::object legacy_request{{"preset", "paper-quick"}, {"engine", "reduced_xy"}};
    const auto legacy_config = turing::simulation_config_from_json(legacy_request, registry);
    failures.require(
        legacy_config.family_id == "section10_example1",
        "legacy preset-only JSON still resolves to Section 10 family");
    failures.require(
        legacy_config.species_order.size() == 2 && legacy_config.species_order[0] == "X"
            && legacy_config.species_order[1] == "Y",
        "legacy preset-only JSON preserves two-species ordering");

    if (failures.value != 0) {
        std::cerr << failures.value << " regression check(s) failed.\n";
        return 1;
    }

    std::cout << "All regression checks passed.\n";
    return 0;
}
