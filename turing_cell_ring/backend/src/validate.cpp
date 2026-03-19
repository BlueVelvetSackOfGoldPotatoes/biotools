#include "turing/analysis.hpp"
#include "turing/json.hpp"
#include "turing/model.hpp"
#include "turing/simulator.hpp"

#include <boost/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

struct ReferenceTarget {
    std::string preset_id;
    std::string reference_id;
    std::string label;
    double threshold = 0.0;
};

std::optional<std::string> argument_value(int argc, char* argv[], const std::string& key) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == key) {
            return argv[index + 1];
        }
    }
    return std::nullopt;
}

std::size_t parse_size(int argc, char* argv[], const std::string& key, std::size_t fallback) {
    if (const auto value = argument_value(argc, argv, key)) {
        return static_cast<std::size_t>(std::stoull(*value));
    }
    return fallback;
}

void write_json_file(const std::filesystem::path& path, const boost::json::value& value) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    output << boost::json::serialize(value) << '\n';
}

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
    const auto total = static_cast<double>(batch.replicates);
    if (total <= 0.0 || batch.dominant_mode_histogram.size() <= 4) {
        return 0.0;
    }
    return static_cast<double>(batch.dominant_mode_histogram[3] + batch.dominant_mode_histogram[4]) / total;
}

boost::json::array mode_histogram_json(const std::vector<std::size_t>& histogram) {
    boost::json::array array;
    array.reserve(histogram.size());
    for (const auto value : histogram) {
        array.push_back(static_cast<std::uint64_t>(value));
    }
    return array;
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
    const auto limit = std::min(lhs.size(), rhs.size());
    double best = 0.0;
    for (std::size_t index = 0; index < limit; ++index) {
        best = std::max(best, std::abs(lhs[index] - rhs[index]));
    }
    return best;
}

boost::json::object make_reference_target_summary(
    const ReferenceTarget& target,
    std::uint64_t best_seed,
    double best_score,
    int best_shift,
    bool pass) {
    return {
        {"presetId", target.preset_id},
        {"referenceId", target.reference_id},
        {"label", target.label},
        {"threshold", target.threshold},
        {"bestSeed", best_seed},
        {"bestScore", best_score},
        {"bestShift", best_shift},
        {"pass", pass}
    };
}

boost::json::object run_section10_engine_validation(
    const std::string& engine,
    const std::filesystem::path& output_dir,
    std::size_t replicates,
    std::size_t seed_search,
    const turing::PresetRegistry& registry,
    const turing::PaperModel& model,
    const turing::LinearStabilityAnalyzer& analyzer,
    const turing::RingExperiment& experiment) {
    const auto* quick_preset = registry.find("paper-quick");
    const auto* slow_preset = registry.find("paper-slow");
    if (quick_preset == nullptr || slow_preset == nullptr) {
        throw std::runtime_error("Missing Section 10 presets.");
    }

    auto quick = quick_preset->config;
    auto slow = slow_preset->config;
    quick.engine = engine;
    slow.engine = engine;

    const auto family_dir = output_dir / "section10" / engine;
    const auto analysis_gamma0 = analyzer.analyze(model, quick, 0.0);
    const auto quick_batch = experiment.simulate_batch(quick, replicates);
    const auto slow_batch = experiment.simulate_batch(slow, replicates);
    auto quick_seed1 = quick;
    quick_seed1.seed = 1;
    auto slow_seed1 = slow;
    slow_seed1.seed = 1;
    const auto quick_seed_result = experiment.simulate(quick_seed1);
    const auto slow_seed_result = experiment.simulate(slow_seed1);

    write_json_file(family_dir / "analysis_gamma0.json", turing::to_json(analysis_gamma0));
    write_json_file(family_dir / "quick_batch.json", turing::to_json(quick_batch));
    write_json_file(family_dir / "slow_batch.json", turing::to_json(slow_batch));
    write_json_file(family_dir / "quick_seed1.json", turing::to_json(quick_seed_result));
    write_json_file(family_dir / "slow_seed1.json", turing::to_json(slow_seed_result));

    const std::vector<ReferenceTarget> targets{
        {"paper-quick", "paper_first_incipient", "Quick first incipient", 0.10},
        {"paper-quick", "paper_first_final", "Quick first final", 0.02},
        {"paper-quick", "paper_second_incipient_y", "Quick second incipient Y", 0.10},
        {"paper-slow", "paper_slow_incipient_y", "Slow incipient Y", 0.12},
    };

    boost::json::array reference_summaries;
    auto reference_pass = true;
    for (const auto& target : targets) {
        auto config = target.preset_id == "paper-slow" ? slow : quick;
        double best_score = std::numeric_limits<double>::infinity();
        int best_shift = -1;
        std::uint64_t best_seed = 0;
        turing::SimulationResult best_result;
        auto found = false;

        for (std::size_t seed = 1; seed <= seed_search; ++seed) {
            config.seed = seed;
            auto result = experiment.simulate(config);
            const auto* comparison = find_comparison(result, target.reference_id);
            if (comparison == nullptr) {
                continue;
            }

            const auto score = comparison->rmse_combined >= 0.0 ? comparison->rmse_combined : comparison->rmse_y;
            if (score < best_score) {
                best_score = score;
                best_shift = comparison->best_shift;
                best_seed = static_cast<std::uint64_t>(seed);
                best_result = std::move(result);
                found = true;
            }
        }

        if (!found) {
            throw std::runtime_error("Failed to produce a reference search result for " + target.reference_id);
        }

        const auto pass = best_score <= target.threshold;
        reference_pass = reference_pass && pass;
        write_json_file(family_dir / ("best_" + target.reference_id + ".json"), turing::to_json(best_result));
        reference_summaries.push_back(make_reference_target_summary(
            target,
            best_seed,
            best_score,
            best_shift,
            pass));
    }

    const auto linear_pass =
        analysis_gamma0.dominant_mode == 3
        && std::abs(analysis_gamma0.equilibrium.x - 1.0) <= 1.0e-12
        && std::abs(analysis_gamma0.equilibrium.y - 1.0) <= 1.0e-12
        && std::abs(analysis_gamma0.jacobian.a - (-2.0)) <= 1.0e-12
        && std::abs(analysis_gamma0.jacobian.b - (-1.5625)) <= 1.0e-12
        && std::abs(analysis_gamma0.jacobian.c - 2.0) <= 1.0e-12
        && std::abs(analysis_gamma0.jacobian.d - 1.5) <= 1.0e-12;

    const auto quick_batch_pass =
        quick_batch.share_mode3 > quick_batch.share_mode4
        && histogram_mass_mode3_mode4(quick_batch) >= (engine == "reduced_xy" ? 0.95 : 0.90);
    const auto slow_batch_pass =
        slow_batch.share_mode3 >= (engine == "reduced_xy" ? 0.95 : 0.85)
        && slow_batch.mean_regularity_index > quick_batch.mean_regularity_index
        && slow_batch.share_mode4 <= (engine == "reduced_xy" ? 0.05 : 0.10);

    boost::json::object summary{
        {"familyId", "section10_example1"},
        {"engine", engine},
        {"linear", turing::to_json(analysis_gamma0)},
        {"quickBatch", turing::to_json(quick_batch)},
        {"slowBatch", turing::to_json(slow_batch)},
        {"referenceSearch", std::move(reference_summaries)},
        {"checks",
         {
             {"linearPass", linear_pass},
             {"referencePass", reference_pass},
             {"referencePassRequired", engine == "reduced_xy"},
             {"quickBatchPass", quick_batch_pass},
             {"slowBatchPass", slow_batch_pass},
             {"batchPass", quick_batch_pass && slow_batch_pass}
         }}
    };

    write_json_file(family_dir / "summary.json", summary);
    return summary;
}

boost::json::object run_table2_validation(
    const std::filesystem::path& output_dir,
    const turing::PresetRegistry& registry,
    const turing::PaperModel& model,
    const turing::LinearStabilityAnalyzer& analyzer,
    const turing::RingExperiment& experiment) {
    const auto* stable_preset = registry.find("paper-table2-stable-ring");
    const auto* scan_preset = registry.find("paper-example2-scan");
    if (stable_preset == nullptr || scan_preset == nullptr) {
        throw std::runtime_error("Missing Table 2 presets.");
    }

    auto stable = stable_preset->config;
    stable.enable_noise = false;
    auto historical = stable;
    historical.execution_mode = "historical_paper_constrained";
    historical.execution_profile_id = "historic_1952_baseline";

    const auto family_dir = output_dir / "example2_table2";
    const auto analysis = analyzer.analyze(model, stable, 0.0);
    const auto modern_result = experiment.simulate(stable);
    const auto historical_result = experiment.simulate(historical);
    write_json_file(family_dir / "analysis.json", turing::to_json(analysis));
    write_json_file(family_dir / "modern_stable_ring.json", turing::to_json(modern_result));
    write_json_file(family_dir / "historical_stable_ring.json", turing::to_json(historical_result));

    const auto reference = model.reference_species_for_preset(stable);
    const auto* reference_x = find_series(reference, "X");
    const auto* reference_y = find_series(reference, "Y");
    const auto* modern_initial_x = find_series(modern_result.initial_snapshot.species, "X");
    const auto* modern_initial_y = find_series(modern_result.initial_snapshot.species, "Y");
    const auto* modern_x = find_series(modern_result.final_snapshot.species, "X");
    const auto* modern_y = find_series(modern_result.final_snapshot.species, "Y");
    const auto* historical_initial_x = find_series(historical_result.initial_snapshot.species, "X");
    const auto* historical_initial_y = find_series(historical_result.initial_snapshot.species, "Y");
    const auto* historical_x = find_series(historical_result.final_snapshot.species, "X");
    const auto* historical_y = find_series(historical_result.final_snapshot.species, "Y");

    const auto modern_initial_x_error =
        modern_initial_x != nullptr && reference_x != nullptr
            ? max_abs_difference(*modern_initial_x, *reference_x)
            : std::numeric_limits<double>::infinity();
    const auto modern_initial_y_error =
        modern_initial_y != nullptr && reference_y != nullptr
            ? max_abs_difference(*modern_initial_y, *reference_y)
            : std::numeric_limits<double>::infinity();
    const auto modern_x_error =
        modern_x != nullptr && reference_x != nullptr ? max_abs_difference(*modern_x, *reference_x)
                                                      : std::numeric_limits<double>::infinity();
    const auto modern_y_error =
        modern_y != nullptr && reference_y != nullptr ? max_abs_difference(*modern_y, *reference_y)
                                                      : std::numeric_limits<double>::infinity();
    const auto historical_initial_x_error =
        historical_initial_x != nullptr && reference_x != nullptr
            ? max_abs_difference(*historical_initial_x, *reference_x)
            : std::numeric_limits<double>::infinity();
    const auto historical_initial_y_error =
        historical_initial_y != nullptr && reference_y != nullptr
            ? max_abs_difference(*historical_initial_y, *reference_y)
            : std::numeric_limits<double>::infinity();
    const auto historical_x_error =
        historical_x != nullptr && reference_x != nullptr ? max_abs_difference(*historical_x, *reference_x)
                                                          : std::numeric_limits<double>::infinity();
    const auto historical_y_error =
        historical_y != nullptr && reference_y != nullptr ? max_abs_difference(*historical_y, *reference_y)
                                                          : std::numeric_limits<double>::infinity();

    boost::json::array scan;
    for (const double f_value : {2.0, 4.0, 8.0, 12.0}) {
        auto config = scan_preset->config;
        config.enable_noise = false;
        turing::upsert_parameter(config.family_parameters, "example2F", f_value);
        const auto entry = analyzer.analyze(model, config, 0.0);
        scan.push_back({
            {"f", f_value},
            {"k", 16.0 - f_value},
            {"equilibrium", turing::to_json(entry.equilibrium)},
            {"dominantMode", static_cast<std::uint64_t>(entry.dominant_mode)},
            {"dominantGrowth", entry.dominant_growth},
            {"dominantClassification", entry.dominant_classification},
            {"hasInstability", entry.has_instability}
        });
    }
    write_json_file(family_dir / "scan_summary.json", scan);

    boost::json::object summary{
        {"familyId", "example2_table2"},
        {"analysis", turing::to_json(analysis)},
        {"referenceInitialMaxAbsError",
         {
             {"modernX", modern_initial_x_error},
             {"modernY", modern_initial_y_error},
             {"historicalX", historical_initial_x_error},
             {"historicalY", historical_initial_y_error}
         }},
        {"referenceMaxAbsError",
         {
             {"modernX", modern_x_error},
             {"modernY", modern_y_error},
             {"historicalX", historical_x_error},
             {"historicalY", historical_y_error}
         }},
        {"checks",
         {
             {"equilibriumPass",
              std::abs(analysis.equilibrium.x - (16.0 / 12.0)) <= 1.0e-9
                  && std::abs(analysis.equilibrium.y - 12.0) <= 1.0e-9},
             {"modernRelaxationPass",
              modern_x_error <= modern_initial_x_error && modern_y_error <= modern_initial_y_error},
             {"modernPersistencePass", modern_x_error <= 0.35 && modern_y_error <= 0.35},
             {"historicalPersistencePass", historical_x_error <= 0.35 && historical_y_error <= 0.35}
         }},
        {"scan", std::move(scan)}
    };

    write_json_file(family_dir / "summary.json", summary);
    return summary;
}

boost::json::object run_oscillatory_validation(
    const std::filesystem::path& output_dir,
    const turing::PresetRegistry& registry,
    const turing::PaperModel& model,
    const turing::LinearStabilityAnalyzer& analyzer,
    const turing::RingExperiment& experiment) {
    const auto* wave_e_preset = registry.find("paper-wave-e-travelling");
    const auto* wave_f_preset = registry.find("paper-wave-f-short-oscillation");
    if (wave_e_preset == nullptr || wave_f_preset == nullptr) {
        throw std::runtime_error("Missing oscillatory presets.");
    }

    const auto family_dir = output_dir / "oscillatory_toolkit";
    auto wave_e_config = wave_e_preset->config;
    wave_e_config.enable_noise = false;
    auto wave_f_config = wave_f_preset->config;
    wave_f_config.enable_noise = false;

    const auto wave_e_analysis = analyzer.analyze(model, wave_e_config, 0.0);
    const auto wave_f_analysis = analyzer.analyze(model, wave_f_config, 0.0);
    const auto wave_e_result = experiment.simulate(wave_e_config);
    const auto wave_f_result = experiment.simulate(wave_f_config);

    write_json_file(family_dir / "case_e_analysis.json", turing::to_json(wave_e_analysis));
    write_json_file(family_dir / "case_f_analysis.json", turing::to_json(wave_f_analysis));
    write_json_file(family_dir / "case_e_simulation.json", turing::to_json(wave_e_result));
    write_json_file(family_dir / "case_f_simulation.json", turing::to_json(wave_f_result));

    boost::json::object summary{
        {"familyId", "oscillatory_toolkit"},
        {"caseE",
         {
             {"analysis", turing::to_json(wave_e_analysis)},
             {"metrics", turing::to_json(wave_e_result.metrics)},
             {"checks",
              {
                  {"oscillatoryPass", wave_e_analysis.dominant_classification == "oscillatory"},
                  {"frequencyPass", wave_e_analysis.dominant_frequency_cycles_per_time > 0.01},
                  {"travellingPass", wave_e_analysis.dominant_phase_velocity_cells_per_time > 0.01}
              }}
         }},
        {"caseF",
         {
             {"analysis", turing::to_json(wave_f_analysis)},
             {"metrics", turing::to_json(wave_f_result.metrics)},
             {"checks",
              {
                  {"oscillatoryPass", wave_f_analysis.dominant_classification == "oscillatory"},
                  {"shortWavePass", wave_f_analysis.dominant_mode >= 4},
                  {"antiphasePass",
                   std::abs(std::abs(wave_f_result.metrics.neighbor_phase_offset_radians) - std::acos(-1.0))
                       <= 0.8}
              }}
         }}
    };

    write_json_file(family_dir / "summary.json", summary);
    return summary;
}

boost::json::object run_historical_validation(
    const std::filesystem::path& output_dir,
    const turing::PresetRegistry& registry,
    const turing::RingExperiment& experiment) {
    const auto* quick_preset = registry.find("paper-quick");
    const auto* table2_preset = registry.find("paper-table2-stable-ring");
    if (quick_preset == nullptr || table2_preset == nullptr) {
        throw std::runtime_error("Missing presets for historical comparison.");
    }

    const auto family_dir = output_dir / "historical_comparisons";
    auto quick_modern = quick_preset->config;
    quick_modern.seed = 9;

    auto table2_modern = table2_preset->config;
    table2_modern.enable_noise = false;
    auto table2_historical = table2_modern;
    table2_historical.execution_mode = "historical_paper_constrained";
    table2_historical.execution_profile_id = "historic_1952_baseline";

    const auto quick_modern_result = experiment.simulate(quick_modern);
    std::vector<std::string> quick_profiles{
        "historic_1952_baseline",
        "historic_1952_coarse_rounding",
        "historic_1952_fine_rounding",
        "historic_1952_synchronous",
    };
    boost::json::array profile_summaries;
    turing::SimulationResult quick_historical_result;
    turing::SimulationResult quick_coarse_result;
    auto baseline_set = false;
    auto coarse_set = false;
    for (const auto& profile_id : quick_profiles) {
        auto config = quick_modern;
        config.execution_mode = "historical_paper_constrained";
        config.execution_profile_id = profile_id;
        const auto result = experiment.simulate(config);
        write_json_file(family_dir / ("quick_" + profile_id + ".json"), turing::to_json(result));
        const auto delta_x =
            max_abs_difference(quick_modern_result.final_snapshot.x, result.final_snapshot.x);
        const auto delta_y =
            max_abs_difference(quick_modern_result.final_snapshot.y, result.final_snapshot.y);
        profile_summaries.push_back({
            {"executionProfileId", profile_id},
            {"finalDelta", {{"x", delta_x}, {"y", delta_y}}},
            {"dominantMode", static_cast<std::uint64_t>(result.metrics.dominant_mode)},
            {"regularityIndex", result.metrics.regularity_index}
        });
        if (profile_id == "historic_1952_baseline") {
            quick_historical_result = result;
            baseline_set = true;
        } else if (profile_id == "historic_1952_coarse_rounding") {
            quick_coarse_result = result;
            coarse_set = true;
        }
    }
    if (!baseline_set) {
        throw std::runtime_error("Missing baseline historical profile.");
    }
    if (!coarse_set) {
        throw std::runtime_error("Missing coarse historical profile.");
    }
    const auto table2_modern_result = experiment.simulate(table2_modern);
    const auto table2_historical_result = experiment.simulate(table2_historical);

    write_json_file(family_dir / "quick_modern.json", turing::to_json(quick_modern_result));
    write_json_file(family_dir / "table2_modern.json", turing::to_json(table2_modern_result));
    write_json_file(family_dir / "table2_historical.json", turing::to_json(table2_historical_result));

    boost::json::object summary{
        {"executionProfileId", "historic_1952_baseline"},
        {"quickHistoricalProfiles", std::move(profile_summaries)},
        {"checks",
         {
             {"quickDifferentFromModern",
              max_abs_difference(
                  quick_modern_result.final_snapshot.y,
                  quick_historical_result.final_snapshot.y)
                  > 1.0e-6},
             {"table2StableProfileAgreement",
              max_abs_difference(
                  table2_modern_result.final_snapshot.x,
                  table2_historical_result.final_snapshot.x)
                  <= 0.01
                  && max_abs_difference(
                         table2_modern_result.final_snapshot.y,
                         table2_historical_result.final_snapshot.y)
                         <= 0.01},
             {"sensitivityProfilesDiffer",
              max_abs_difference(
                  quick_historical_result.final_snapshot.y,
                  quick_coarse_result.final_snapshot.y)
                  > 1.0e-6}
         }},
        {"quickFinalDelta",
         {
             {"x", max_abs_difference(quick_modern_result.final_snapshot.x, quick_historical_result.final_snapshot.x)},
             {"y", max_abs_difference(quick_modern_result.final_snapshot.y, quick_historical_result.final_snapshot.y)}
         }},
        {"table2FinalDelta",
         {
             {"x", max_abs_difference(table2_modern_result.final_snapshot.x, table2_historical_result.final_snapshot.x)},
             {"y", max_abs_difference(table2_modern_result.final_snapshot.y, table2_historical_result.final_snapshot.y)}
         }}
    };

    write_json_file(family_dir / "summary.json", summary);
    return summary;
}

}  // namespace

int main(int argc, char* argv[]) {
    const auto output_dir = std::filesystem::path(
        argument_value(argc, argv, "--output-dir").value_or("results/final_validation"));
    const auto replicates = parse_size(argc, argv, "--replicates", 512);
    const auto seed_search = parse_size(argc, argv, "--seed-search", 64);

    try {
        turing::PresetRegistry registry;
        turing::PaperModel model;
        turing::LinearStabilityAnalyzer analyzer;
        turing::RingExperiment experiment;

        boost::json::array section10_engines;
        section10_engines.push_back(run_section10_engine_validation(
            "reduced_xy",
            output_dir,
            replicates,
            seed_search,
            registry,
            model,
            analyzer,
            experiment));
        section10_engines.push_back(run_section10_engine_validation(
            "full_chemistry",
            output_dir,
            replicates,
            seed_search,
            registry,
            model,
            analyzer,
            experiment));

        const auto table2_summary =
            run_table2_validation(output_dir, registry, model, analyzer, experiment);
        const auto oscillatory_summary =
            run_oscillatory_validation(output_dir, registry, model, analyzer, experiment);
        const auto historical_summary = run_historical_validation(output_dir, registry, experiment);

        boost::json::object root{
            {"replicates", static_cast<std::uint64_t>(replicates)},
            {"seedSearch", static_cast<std::uint64_t>(seed_search)},
            {"section10", std::move(section10_engines)},
            {"table2", std::move(table2_summary)},
            {"oscillatoryToolkit", std::move(oscillatory_summary)},
            {"historicalComparisons", std::move(historical_summary)}
        };

        write_json_file(output_dir / "summary.json", root);
        std::cout << "Validation artifacts written to " << std::filesystem::absolute(output_dir) << '\n';
    } catch (const std::exception& exception) {
        std::cerr << "Validation failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
