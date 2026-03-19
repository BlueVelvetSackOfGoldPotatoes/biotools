#pragma once

#include "turing/model.hpp"
#include "turing/types.hpp"

namespace turing {

class LinearStabilityAnalyzer {
public:
    StabilityAnalysis analyze(
        const PaperModel& model,
        const SimulationConfig& config,
        double gamma) const;

    double find_threshold_gamma(
        const PaperModel& model,
        const SimulationConfig& config,
        double gamma_low,
        double gamma_high) const;
};

}  // namespace turing
