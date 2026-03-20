#pragma once

#include "models/cells/src/sim/sim_state.h"

#include <vector>

namespace cells {
namespace membrane {

double project_stretch(
    std::vector<Vec3>& predicted_positions,
    const std::vector<Edge>& edges,
    const std::vector<double>& inv_masses,
    double compliance,
    double dt);

double project_area(
    std::vector<Vec3>& predicted_positions,
    const std::vector<Triangle>& triangles,
    const std::vector<double>& inv_masses,
    double target_area,
    double compliance,
    double dt);

double project_volume(
    std::vector<Vec3>& predicted_positions,
    const std::vector<Triangle>& triangles,
    const std::vector<double>& inv_masses,
    double target_volume,
    double compliance,
    double dt);

double project_bend(
    std::vector<Vec3>& predicted_positions,
    const std::vector<Triangle>& triangles,
    const std::vector<Edge>& edges,
    const std::vector<double>& inv_masses,
    double compliance,
    double dt);

} // namespace membrane
} // namespace cells
