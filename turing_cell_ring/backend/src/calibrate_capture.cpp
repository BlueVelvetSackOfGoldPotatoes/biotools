#include "turing/json.hpp"
#include "turing/model.hpp"
#include "turing/simulator.hpp"

#include <boost/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

struct CandidateConfig {
    std::string mode = "gamma_threshold";
    double start_time = 0.0;
    double threshold = 0.0;
};

struct ReferenceTarget {
    std::string reference_id;
    std::string label;
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

boost::json::object evaluate_candidate(
    const std::string& preset_id,
    const std::vector<ReferenceTarget>& targets,
    const CandidateConfig& candidate,
    std::size_t seed_search,
    const turing::PresetRegistry& registry,
    const turing::RingExperiment& experiment) {
    const auto* preset = registry.find(preset_id);
    if (preset == nullptr) {
        throw std::runtime_error("Missing preset: " + preset_id);
    }

    boost::json::array engine_summaries;
    auto objective = 0.0;

    for (const std::string engine : {"reduced_xy", "full_chemistry"}) {
        auto config = preset->config;
        config.engine = engine;
        config.incipient_capture_mode = candidate.mode;
        config.incipient_capture_start_time = candidate.start_time;
        config.incipient_mode234_threshold = candidate.threshold;

        boost::json::array target_summaries;
        auto engine_score = 0.0;

        for (const auto& target : targets) {
            auto best_score = std::numeric_limits<double>::infinity();
            std::uint64_t best_seed = 0;
            auto best_time = -1.0;
            auto best_gamma = 0.0;

            for (std::size_t seed = 1; seed <= seed_search; ++seed) {
                config.seed = seed;
                const auto result = experiment.simulate(config);
                const auto* comparison = find_comparison(result, target.reference_id);
                if (comparison == nullptr) {
                    continue;
                }

                const auto score = comparison->rmse_combined >= 0.0
                    ? comparison->rmse_combined
                    : comparison->rmse_y;
                if (score < best_score) {
                    best_score = score;
                    best_seed = static_cast<std::uint64_t>(seed);
                    best_time = result.incipient_snapshot.time;
                    best_gamma = result.incipient_snapshot.gamma;
                }
            }

            if (!std::isfinite(best_score)) {
                throw std::runtime_error("Failed to produce score for " + target.reference_id);
            }

            engine_score += best_score;
            target_summaries.push_back({
                {"referenceId", target.reference_id},
                {"label", target.label},
                {"bestSeed", best_seed},
                {"bestScore", best_score},
                {"bestTime", best_time},
                {"bestGamma", best_gamma}
            });
        }

        objective += engine_score;
        engine_summaries.push_back({
            {"engine", engine},
            {"score", engine_score},
            {"targets", std::move(target_summaries)}
        });
    }

    return {
        {"presetId", preset_id},
        {"mode", candidate.mode},
        {"startTime", candidate.start_time},
        {"mode234Threshold", candidate.threshold},
        {"objective", objective},
        {"engines", std::move(engine_summaries)}
    };
}

}  // namespace

int main(int argc, char* argv[]) {
    const auto preset_id = argument_value(argc, argv, "--preset").value_or("paper-quick");
    const auto output_dir = std::filesystem::path(
        argument_value(argc, argv, "--output-dir").value_or("results/capture_calibration"));
    const auto seed_search = parse_size(argc, argv, "--seed-search", 32);
    const auto single_mode = argument_value(argc, argv, "--single-mode");
    const auto single_start_time = argument_value(argc, argv, "--single-start-time");
    const auto single_threshold = argument_value(argc, argv, "--single-threshold");

    try {
        turing::PresetRegistry registry;
        turing::RingExperiment experiment;

        std::vector<ReferenceTarget> targets;
        std::vector<CandidateConfig> candidates;

        if (preset_id == "paper-quick") {
            targets = {
                {"paper_first_incipient", "Quick first incipient"},
                {"paper_second_incipient_y", "Quick second incipient Y"},
            };
        } else if (preset_id == "paper-slow") {
            targets = {
                {"paper_slow_incipient_y", "Slow incipient Y"},
            };
        } else {
            throw std::runtime_error("Unsupported preset for capture calibration: " + preset_id);
        }

        if (single_mode.has_value()) {
            candidates.push_back({
                *single_mode,
                single_start_time.has_value() ? std::stod(*single_start_time) : 0.0,
                single_threshold.has_value() ? std::stod(*single_threshold) : 0.0
            });
        } else {
            candidates.push_back({"gamma_threshold", 0.0, 0.0});
            if (preset_id == "paper-quick") {
                for (const double start_time : {32.0, 34.0, 36.0, 38.0, 40.0}) {
                    for (const double threshold : {0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26}) {
                        candidates.push_back({"mode234_threshold", start_time, threshold});
                    }
                }
            } else {
                for (const double start_time : {900.0, 1000.0, 1100.0, 1200.0}) {
                    for (const double threshold : {0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24}) {
                        candidates.push_back({"mode234_threshold", start_time, threshold});
                    }
                }
            }
        }

        boost::json::array results;
        auto best_index = std::size_t{0};
        auto best_objective = std::numeric_limits<double>::infinity();

        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const auto summary = evaluate_candidate(
                preset_id,
                targets,
                candidates[index],
                seed_search,
                registry,
                experiment);
            const auto objective = summary.at("objective").as_double();
            if (objective < best_objective) {
                best_objective = objective;
                best_index = index;
            }
            results.push_back(summary);
        }

        boost::json::object output{
            {"presetId", preset_id},
            {"seedSearch", static_cast<std::uint64_t>(seed_search)},
            {"candidateCount", static_cast<std::uint64_t>(candidates.size())},
            {"bestCandidate", results[best_index]},
            {"candidates", std::move(results)}
        };

        write_json_file(output_dir / (preset_id + "_capture_calibration.json"), output);
        std::cout << boost::json::serialize(output.at("bestCandidate")) << '\n';
    } catch (const std::exception& exception) {
        std::cerr << "Calibration failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
