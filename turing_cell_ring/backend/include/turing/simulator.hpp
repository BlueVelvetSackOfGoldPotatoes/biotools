#pragma once

#include "turing/analysis.hpp"
#include "turing/model.hpp"
#include "turing/types.hpp"

namespace turing {

class RingExperiment {
public:
    RingExperiment();

    SimulationResult simulate(const SimulationConfig& config) const;
    BatchResult simulate_batch(const SimulationConfig& config, std::size_t replicates) const;

private:
    PaperModel model_;
    LinearStabilityAnalyzer analyzer_;
};

}  // namespace turing
