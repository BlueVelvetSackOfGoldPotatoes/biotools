#include "turing/json.hpp"

#include "turing/model.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace turing {

namespace {

boost::json::array to_json_array(const std::vector<double>& values) {
    boost::json::array array;
    array.reserve(values.size());
    for (const auto value : values) {
        array.push_back(value);
    }
    return array;
}

boost::json::array to_json_array(const std::vector<std::size_t>& values) {
    boost::json::array array;
    array.reserve(values.size());
    for (const auto value : values) {
        array.push_back(static_cast<std::uint64_t>(value));
    }
    return array;
}

boost::json::array to_json_array(const std::vector<std::string>& values) {
    boost::json::array array;
    array.reserve(values.size());
    for (const auto& value : values) {
        array.push_back(value.c_str());
    }
    return array;
}

const boost::json::value* find_if_present(
    const boost::json::object& object,
    const std::string& key) {
    if (const auto* entry = object.if_contains(key)) {
        return entry;
    }
    return nullptr;
}

double read_double(
    const boost::json::object& object,
    const std::string& key,
    double fallback) {
    if (const auto* value = find_if_present(object, key)) {
        if (value->is_double()) {
            return value->as_double();
        }
        if (value->is_int64()) {
            return static_cast<double>(value->as_int64());
        }
        if (value->is_uint64()) {
            return static_cast<double>(value->as_uint64());
        }
    }
    return fallback;
}

std::size_t read_size(
    const boost::json::object& object,
    const std::string& key,
    std::size_t fallback) {
    if (const auto* value = find_if_present(object, key)) {
        if (value->is_uint64()) {
            return static_cast<std::size_t>(value->as_uint64());
        }
        if (value->is_int64()) {
            return static_cast<std::size_t>(std::max<std::int64_t>(value->as_int64(), 0));
        }
    }
    return fallback;
}

bool read_bool(
    const boost::json::object& object,
    const std::string& key,
    bool fallback) {
    if (const auto* value = find_if_present(object, key); value && value->is_bool()) {
        return value->as_bool();
    }
    return fallback;
}

std::string read_string(
    const boost::json::object& object,
    const std::string& key,
    const std::string& fallback) {
    if (const auto* value = find_if_present(object, key); value && value->is_string()) {
        return std::string(value->as_string().c_str());
    }
    return fallback;
}

boost::json::object series_object(const std::vector<NamedSeries>& values) {
    boost::json::object object;
    for (const auto& series : values) {
        object[series.name] = to_json_array(series.values);
    }
    return object;
}

boost::json::object named_values_object(const std::vector<NamedValue>& values) {
    boost::json::object object;
    for (const auto& value : values) {
        object[value.name] = value.value;
    }
    return object;
}

std::vector<std::string> read_string_array(const boost::json::object& object, const std::string& key) {
    std::vector<std::string> result;
    if (const auto* value = find_if_present(object, key); value && value->is_array()) {
        for (const auto& entry : value->as_array()) {
            if (entry.is_string()) {
                result.push_back(std::string(entry.as_string().c_str()));
            }
        }
    }
    return result;
}

void read_named_values_from_json(
    std::vector<NamedValue>& values,
    const boost::json::object& object,
    const std::string& key) {
    if (const auto* entry = find_if_present(object, key); entry && entry->is_object()) {
        values.clear();
        for (const auto& item : entry->as_object()) {
            if (item.value().is_double()) {
                values.push_back({std::string(item.key()), item.value().as_double()});
            } else if (item.value().is_int64()) {
                values.push_back({std::string(item.key()), static_cast<double>(item.value().as_int64())});
            } else if (item.value().is_uint64()) {
                values.push_back({std::string(item.key()), static_cast<double>(item.value().as_uint64())});
            }
        }
    }
}

}  // namespace

boost::json::object to_json(const RingConfig& value) {
    return {
        {"cellCount", static_cast<std::uint64_t>(value.cell_count)},
        {"cellDiameterCm", value.cell_diameter_cm},
        {"diffusionXSpecial", value.diffusion_x_special},
        {"diffusionYSpecial", value.diffusion_y_special},
        {"timeUnitSeconds", value.time_unit_seconds},
        {"concentrationUnitMolPerCm3", value.concentration_unit_mol_per_cm3}
    };
}

boost::json::object to_json(const ControlSegment& value) {
    return {
        {"startTime", value.start_time},
        {"endTime", value.end_time},
        {"startValue", value.start_value},
        {"endValue", value.end_value}
    };
}

boost::json::object to_json(const ControlSchedule& value) {
    return {
        {"id", value.id},
        {"name", value.name},
        {"segments", [&]() {
            boost::json::array array;
            for (const auto& segment : value.segments) {
                array.push_back(to_json(segment));
            }
            return array;
        }()},
        {"postSegmentValue", value.post_segment_value}
    };
}

boost::json::object to_json(const SimulationConfig& value) {
    return {
        {"presetId", value.preset_id},
        {"familyId", value.family_id},
        {"engine", value.engine},
        {"analysisMode", value.analysis_mode},
        {"executionMode", value.execution_mode},
        {"executionProfileId", value.execution_profile_id},
        {"incipientCaptureMode", value.incipient_capture_mode},
        {"speciesOrder", to_json_array(value.species_order)},
        {"familyParameters", named_values_object(value.family_parameters)},
        {"ring", to_json(value.ring)},
        {"schedule", to_json(value.schedule)},
        {"dt", value.dt},
        {"totalTime", value.total_time},
        {"enableNoise", value.enable_noise},
        {"noiseScale", value.noise_scale},
        {"initialPerturbationScale", value.initial_perturbation_scale},
        {"seed", value.seed},
        {"captureStride", static_cast<std::uint64_t>(value.capture_stride)},
        {"maxMode", static_cast<std::uint64_t>(value.max_mode)},
        {"incipientCaptureGamma", value.incipient_capture_gamma},
        {"incipientCaptureStartTime", value.incipient_capture_start_time},
        {"incipientMode234Threshold", value.incipient_mode234_threshold}
    };
}

boost::json::object to_json(const Equilibrium& value) {
    return {
        {"x", value.x},
        {"y", value.y},
        {"speciesOrder", to_json_array(value.species_order)},
        {"speciesValues", to_json_array(value.species_values)}
    };
}

boost::json::object to_json(const Jacobian& value) {
    boost::json::array matrix;
    for (const auto& row : value.matrix) {
        matrix.push_back(to_json_array(row));
    }
    return {
        {"a", value.a},
        {"b", value.b},
        {"c", value.c},
        {"d", value.d},
        {"matrix", std::move(matrix)}
    };
}

boost::json::object to_json(const ModeGrowth& value) {
    return {
        {"mode", static_cast<std::uint64_t>(value.mode)},
        {"real", value.real},
        {"imag", value.imag},
        {"oscillatory", value.oscillatory},
        {"wavelengthCells", value.wavelength_cells},
        {"frequencyCyclesPerTime", value.frequency_cycles_per_time},
        {"phaseVelocityCellsPerTime", value.phase_velocity_cells_per_time}
    };
}

boost::json::object to_json(const StabilityAnalysis& value) {
    boost::json::array modes;
    for (const auto& mode : value.modes) {
        modes.push_back(to_json(mode));
    }
    return {
        {"gamma", value.gamma},
        {"familyId", value.family_id},
        {"analysisMode", value.analysis_mode},
        {"speciesOrder", to_json_array(value.species_order)},
        {"equilibrium", to_json(value.equilibrium)},
        {"jacobian", to_json(value.jacobian)},
        {"dominantMode", static_cast<std::uint64_t>(value.dominant_mode)},
        {"dominantGrowth", value.dominant_growth},
        {"dominantFrequencyCyclesPerTime", value.dominant_frequency_cycles_per_time},
        {"dominantPhaseVelocityCellsPerTime", value.dominant_phase_velocity_cells_per_time},
        {"dominantClassification", value.dominant_classification},
        {"mode3Growth", value.mode3_growth},
        {"mode4Growth", value.mode4_growth},
        {"mode34Gap", value.mode34_gap},
        {"mode34EFoldTimeSpecialUnits", value.mode34_efold_time_special_units},
        {"mode34EFoldTimeHours", value.mode34_efold_time_hours},
        {"thresholdGamma", value.threshold_gamma},
        {"hasInstability", value.has_instability},
        {"modes", std::move(modes)}
    };
}

boost::json::object to_json(const ChemistrySnapshot& value) {
    return {
        {"aReservoir", value.a_reservoir},
        {"cTotal", value.c_total},
        {"b", to_json_array(value.b)},
        {"c", to_json_array(value.c)},
        {"cPrime", to_json_array(value.c_prime)},
        {"w", to_json_array(value.w)}
    };
}

boost::json::object to_json(const ModeTraceSample& value) {
    return {
        {"time", value.time},
        {"gamma", value.gamma},
        {"dominantMode", static_cast<std::uint64_t>(value.dominant_mode)},
        {"dominantPhaseRadians", value.dominant_phase_radians},
        {"amplitudes", to_json_array(value.amplitudes)},
        {"speciesPhases", named_values_object(value.species_phases)}
    };
}

boost::json::object to_json(const Snapshot& value) {
    boost::json::object object{
        {"time", value.time},
        {"gamma", value.gamma},
        {"speciesOrder", to_json_array(value.species_order)},
        {"species", series_object(value.species)},
        {"speciesModeAmplitudes", series_object(value.species_mode_amplitudes)},
        {"primaryModeAmplitudes", to_json_array(value.primary_mode_amplitudes)},
        {"x", to_json_array(value.x)},
        {"y", to_json_array(value.y)},
        {"yModeAmplitudes", to_json_array(value.y_mode_amplitudes)}
    };
    if (value.chemistry.has_value()) {
        object["chemistry"] = to_json(*value.chemistry);
    }
    return object;
}

boost::json::object to_json(const ReferenceComparison& value) {
    return {
        {"referenceId", value.reference_id},
        {"label", value.label},
        {"bestShift", value.best_shift},
        {"rmseX", value.rmse_x},
        {"rmseY", value.rmse_y},
        {"rmseCombined", value.rmse_combined}
    };
}

boost::json::object to_json(const SimulationMetrics& value) {
    boost::json::array comparisons;
    for (const auto& comparison : value.comparisons) {
        comparisons.push_back(to_json(comparison));
    }
    return {
        {"dominantMode", static_cast<std::uint64_t>(value.dominant_mode)},
        {"dominantClassification", value.dominant_classification},
        {"dominantAmplitude", value.dominant_amplitude},
        {"mode3Amplitude", value.mode3_amplitude},
        {"mode4Amplitude", value.mode4_amplitude},
        {"mode3ToMode4Ratio", value.mode3_to_mode4_ratio},
        {"regularityIndex", value.regularity_index},
        {"dominantFrequencyCyclesPerTime", value.dominant_frequency_cycles_per_time},
        {"dominantPhaseVelocityCellsPerTime", value.dominant_phase_velocity_cells_per_time},
        {"neighborPhaseOffsetRadians", value.neighbor_phase_offset_radians},
        {"travellingConsistency", value.travelling_consistency},
        {"comparisons", std::move(comparisons)}
    };
}

boost::json::object to_json(const ArrestDiagnostics& value) {
    boost::json::array cells;
    cells.reserve(value.first_y_zero_cells.size());
    for (const auto cell : value.first_y_zero_cells) {
        cells.push_back(static_cast<std::uint64_t>(cell));
    }

    return {
        {"observed", value.observed},
        {"firstYZeroTime", value.first_y_zero_time},
        {"firstYZeroCells", std::move(cells)},
        {"finalZeroCellCount", static_cast<std::uint64_t>(value.final_zero_cell_count)}
    };
}

boost::json::object to_json(const SimulationResult& value) {
    boost::json::array mode_trace;
    for (const auto& sample : value.mode_trace) {
        mode_trace.push_back(to_json(sample));
    }
    boost::json::array snapshot_trace;
    for (const auto& sample : value.snapshot_trace) {
        snapshot_trace.push_back(to_json(sample));
    }
    return {
        {"config", to_json(value.config)},
        {"initialAnalysis", to_json(value.initial_analysis)},
        {"incipientAnalysis", to_json(value.incipient_analysis)},
        {"initialSnapshot", to_json(value.initial_snapshot)},
        {"incipientSnapshot", to_json(value.incipient_snapshot)},
        {"finalSnapshot", to_json(value.final_snapshot)},
        {"modeTrace", std::move(mode_trace)},
        {"snapshotTrace", std::move(snapshot_trace)},
        {"metrics", to_json(value.metrics)},
        {"arrest", to_json(value.arrest)}
    };
}

boost::json::object to_json(const BatchRunSummary& value) {
    return {
        {"seed", value.seed},
        {"dominantMode", static_cast<std::uint64_t>(value.dominant_mode)},
        {"dominantClassification", value.dominant_classification},
        {"dominantAmplitude", value.dominant_amplitude},
        {"regularityIndex", value.regularity_index},
        {"mode3ToMode4Ratio", value.mode3_to_mode4_ratio}
    };
}

boost::json::object to_json(const BatchResult& value) {
    boost::json::array samples;
    for (const auto& sample : value.samples) {
        samples.push_back(to_json(sample));
    }
    return {
        {"config", to_json(value.config)},
        {"replicates", static_cast<std::uint64_t>(value.replicates)},
        {"incipientAnalysis", to_json(value.incipient_analysis)},
        {"dominantModeHistogram", to_json_array(value.dominant_mode_histogram)},
        {"meanRegularityIndex", value.mean_regularity_index},
        {"meanMode3ToMode4Ratio", value.mean_mode3_to_mode4_ratio},
        {"shareMode3", value.share_mode3},
        {"shareMode4", value.share_mode4},
        {"shareMode3CiLow", value.share_mode3_ci_low},
        {"shareMode3CiHigh", value.share_mode3_ci_high},
        {"shareMode4CiLow", value.share_mode4_ci_low},
        {"shareMode4CiHigh", value.share_mode4_ci_high},
        {"samples", std::move(samples)}
    };
}

boost::json::object to_json(const ModelFamily& value) {
    boost::json::array diffusion;
    diffusion.reserve(value.diffusion_by_species.size());
    for (const auto coefficient : value.diffusion_by_species) {
        diffusion.push_back(coefficient);
    }

    boost::json::array linear_matrix;
    linear_matrix.reserve(value.linear_matrix.size());
    for (const auto& row : value.linear_matrix) {
        linear_matrix.push_back(to_json_array(row));
    }

    return {
        {"id", value.id},
        {"name", value.name},
        {"description", value.description},
        {"analysisMode", value.analysis_mode},
        {"speciesOrder", to_json_array(value.species_order)},
        {"diffusionBySpecies", std::move(diffusion)},
        {"linearMatrix", std::move(linear_matrix)},
        {"primarySpeciesIndex", static_cast<std::uint64_t>(value.primary_species_index)},
        {"supportsFullChemistry", value.supports_full_chemistry},
        {"supportsHistoricalExecution", value.supports_historical_execution},
        {"modeUScale", value.mode_u_scale}
    };
}

boost::json::object to_json(const Preset& value) {
    return {
        {"id", value.id},
        {"familyId", value.family_id},
        {"name", value.name},
        {"description", value.description},
        {"hypothesis", value.hypothesis},
        {"config", to_json(value.config)}
    };
}

SimulationConfig simulation_config_from_json(
    const boost::json::object& value,
    const PresetRegistry& presets) {
    const auto preset_id = read_string(value, "preset", "paper-quick");
    const auto* preset = presets.find(preset_id);
    if (preset == nullptr) {
        throw std::invalid_argument("Unknown preset: " + preset_id);
    }

    auto config = preset->config;
    config.preset_id = preset->id;
    config.family_id = read_string(value, "familyId", config.family_id);
    config.engine = read_string(value, "engine", config.engine);
    config.analysis_mode = read_string(value, "analysisMode", config.analysis_mode);
    config.execution_mode = read_string(value, "executionMode", config.execution_mode);
    config.execution_profile_id = read_string(value, "executionProfileId", config.execution_profile_id);
    config.incipient_capture_mode =
        read_string(value, "incipientCaptureMode", config.incipient_capture_mode);
    config.dt = read_double(value, "dt", config.dt);
    config.total_time = read_double(value, "totalTime", config.total_time);
    config.enable_noise = read_bool(value, "enableNoise", config.enable_noise);
    config.noise_scale = read_double(value, "noiseScale", config.noise_scale);
    config.initial_perturbation_scale =
        read_double(value, "initialPerturbationScale", config.initial_perturbation_scale);
    config.seed = read_size(value, "seed", config.seed);
    config.capture_stride = read_size(value, "captureStride", config.capture_stride);
    config.max_mode = read_size(value, "maxMode", config.max_mode);
    config.incipient_capture_gamma =
        read_double(value, "incipientCaptureGamma", config.incipient_capture_gamma);
    config.incipient_capture_start_time =
        read_double(value, "incipientCaptureStartTime", config.incipient_capture_start_time);
    config.incipient_mode234_threshold =
        read_double(value, "incipientMode234Threshold", config.incipient_mode234_threshold);

    const auto species_order = read_string_array(value, "speciesOrder");
    if (!species_order.empty()) {
        config.species_order = species_order;
    }
    read_named_values_from_json(config.family_parameters, value, "familyParameters");

    if (const auto* ring = find_if_present(value, "ring"); ring && ring->is_object()) {
        const auto& ring_object = ring->as_object();
        config.ring.cell_count = read_size(ring_object, "cellCount", config.ring.cell_count);
        config.ring.cell_diameter_cm =
            read_double(ring_object, "cellDiameterCm", config.ring.cell_diameter_cm);
        config.ring.diffusion_x_special =
            read_double(ring_object, "diffusionXSpecial", config.ring.diffusion_x_special);
        config.ring.diffusion_y_special =
            read_double(ring_object, "diffusionYSpecial", config.ring.diffusion_y_special);
        config.ring.time_unit_seconds =
            read_double(ring_object, "timeUnitSeconds", config.ring.time_unit_seconds);
        config.ring.concentration_unit_mol_per_cm3 = read_double(
            ring_object,
            "concentrationUnitMolPerCm3",
            config.ring.concentration_unit_mol_per_cm3);
    }

    if (const auto* schedule = find_if_present(value, "schedule"); schedule && schedule->is_object()) {
        const auto& schedule_object = schedule->as_object();
        config.schedule.post_segment_value =
            read_double(schedule_object, "postSegmentValue", config.schedule.post_segment_value);
        config.schedule.id = read_string(schedule_object, "id", config.schedule.id);
        config.schedule.name = read_string(schedule_object, "name", config.schedule.name);

        if (const auto* segments = schedule_object.if_contains("segments");
            segments != nullptr && segments->is_array()) {
            config.schedule.segments.clear();
            for (const auto& entry : segments->as_array()) {
                if (!entry.is_object()) {
                    continue;
                }
                const auto& segment_object = entry.as_object();
                config.schedule.segments.push_back(ControlSegment{
                    read_double(segment_object, "startTime", 0.0),
                    read_double(segment_object, "endTime", 0.0),
                    read_double(segment_object, "startValue", 0.0),
                    read_double(segment_object, "endValue", 0.0)
                });
            }
        }
    }

    if (const auto* family = presets.find_family(config.family_id); family != nullptr) {
        if (config.species_order.empty()) {
            config.species_order = family->species_order;
        }
        if (config.analysis_mode.empty()) {
            config.analysis_mode = family->analysis_mode;
        }
    }

    return config;
}

std::size_t batch_replicates_from_json(const boost::json::object& value, std::size_t fallback) {
    return read_size(value, "replicates", fallback);
}

}  // namespace turing
