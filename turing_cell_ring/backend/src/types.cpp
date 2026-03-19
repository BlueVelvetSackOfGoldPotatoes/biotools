#include "turing/types.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace turing {

double ControlSchedule::value_at(double time) const {
    if (segments.empty()) {
        return post_segment_value;
    }
    if (time <= segments.front().start_time) {
        return segments.front().start_value;
    }
    for (const auto& segment : segments) {
        if (time <= segment.end_time) {
            const auto width = segment.end_time - segment.start_time;
            if (width <= 0.0) {
                return segment.end_value;
            }
            const auto alpha = std::clamp((time - segment.start_time) / width, 0.0, 1.0);
            return segment.start_value + alpha * (segment.end_value - segment.start_value);
        }
    }
    return post_segment_value;
}

double ControlSchedule::start_value() const {
    if (segments.empty()) {
        return post_segment_value;
    }
    return segments.front().start_value;
}

double ControlSchedule::end_time() const {
    if (segments.empty()) {
        return 0.0;
    }
    return segments.back().end_time;
}

double parameter_value(
    const std::vector<NamedValue>& values,
    const std::string& name,
    double fallback) {
    for (const auto& value : values) {
        if (value.name == name) {
            return value.value;
        }
    }
    return fallback;
}

void upsert_parameter(
    std::vector<NamedValue>& values,
    const std::string& name,
    double value) {
    for (auto& entry : values) {
        if (entry.name == name) {
            entry.value = value;
            return;
        }
    }
    values.push_back({name, value});
}

void validate_config(const SimulationConfig& config) {
    if (config.engine != "reduced_xy" && config.engine != "full_chemistry") {
        throw std::invalid_argument("Unsupported engine: " + config.engine);
    }
    if (
        config.incipient_capture_mode != "gamma_threshold"
        && config.incipient_capture_mode != "mode234_threshold") {
        throw std::invalid_argument(
            "Unsupported incipient capture mode: " + config.incipient_capture_mode);
    }
    if (
        config.execution_mode != "modern"
        && config.execution_mode != "historical_paper_constrained") {
        throw std::invalid_argument("Unsupported execution mode: " + config.execution_mode);
    }
    if (config.ring.cell_count < 3) {
        throw std::invalid_argument("Ring cell count must be at least 3.");
    }
    if (config.dt <= 0.0) {
        throw std::invalid_argument("dt must be positive.");
    }
    if (config.total_time <= 0.0) {
        throw std::invalid_argument("total_time must be positive.");
    }
    if (config.max_mode > config.ring.cell_count / 2) {
        throw std::invalid_argument("max_mode must be at most cell_count / 2.");
    }
    if (config.incipient_capture_start_time < 0.0) {
        throw std::invalid_argument("incipient_capture_start_time must be non-negative.");
    }
    if (config.incipient_capture_start_time > config.total_time) {
        throw std::invalid_argument("incipient_capture_start_time must not exceed total_time.");
    }
    if (
        config.incipient_capture_mode == "mode234_threshold"
        && config.incipient_mode234_threshold <= 0.0) {
        throw std::invalid_argument(
            "incipient_mode234_threshold must be positive for mode234_threshold capture.");
    }
    if (config.family_id.empty()) {
        throw std::invalid_argument("family_id must not be empty.");
    }
    if (config.analysis_mode.empty()) {
        throw std::invalid_argument("analysis_mode must not be empty.");
    }
    if (config.execution_profile_id.empty()) {
        throw std::invalid_argument("execution_profile_id must not be empty.");
    }
    if (config.execution_mode == "historical_paper_constrained"
        && config.execution_profile_id != "historic_1952_baseline"
        && config.execution_profile_id != "historic_1952_coarse_rounding"
        && config.execution_profile_id != "historic_1952_fine_rounding"
        && config.execution_profile_id != "historic_1952_synchronous") {
        throw std::invalid_argument(
            "Unsupported historical execution profile: " + config.execution_profile_id);
    }

    auto previous_end = -std::numeric_limits<double>::infinity();
    for (const auto& segment : config.schedule.segments) {
        if (segment.end_time < segment.start_time) {
            throw std::invalid_argument("Control schedule segment end_time must be >= start_time.");
        }
        if (segment.start_time < previous_end) {
            throw std::invalid_argument("Control schedule segments must be sorted and non-overlapping.");
        }
        previous_end = segment.end_time;
    }
}

}  // namespace turing
