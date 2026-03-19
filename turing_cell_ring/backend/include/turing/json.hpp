#pragma once

#include "turing/model.hpp"
#include "turing/types.hpp"

#include <boost/json.hpp>

namespace turing {

boost::json::object to_json(const RingConfig& value);
boost::json::object to_json(const ControlSegment& value);
boost::json::object to_json(const ControlSchedule& value);
boost::json::object to_json(const SimulationConfig& value);
boost::json::object to_json(const Equilibrium& value);
boost::json::object to_json(const Jacobian& value);
boost::json::object to_json(const ModeGrowth& value);
boost::json::object to_json(const StabilityAnalysis& value);
boost::json::object to_json(const ChemistrySnapshot& value);
boost::json::object to_json(const ModeTraceSample& value);
boost::json::object to_json(const Snapshot& value);
boost::json::object to_json(const ReferenceComparison& value);
boost::json::object to_json(const SimulationMetrics& value);
boost::json::object to_json(const ArrestDiagnostics& value);
boost::json::object to_json(const SimulationResult& value);
boost::json::object to_json(const BatchRunSummary& value);
boost::json::object to_json(const BatchResult& value);
boost::json::object to_json(const ModelFamily& value);
boost::json::object to_json(const Preset& value);

SimulationConfig simulation_config_from_json(
    const boost::json::object& value,
    const PresetRegistry& presets);

std::size_t batch_replicates_from_json(const boost::json::object& value, std::size_t fallback);

}  // namespace turing
