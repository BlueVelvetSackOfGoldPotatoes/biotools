#pragma once

#include "models/cells/src/core/span.h"
#include "models/cells/src/core/vec3.h"
#include "models/cells/src/sim/sph/neighbor_grid.h"

#include <cstdint>
#include <vector>

namespace cells {

struct Triangle {
    std::uint32_t a;
    std::uint32_t b;
    std::uint32_t c;
};

struct Edge {
    std::uint32_t i;
    std::uint32_t j;
    double rest_length;
    std::int32_t tri0 = -1;
    std::int32_t tri1 = -1;
    std::uint32_t opp0 = 0;
    std::uint32_t opp1 = 0;
    double rest_dihedral_cos = 1.0;
};

struct Diagnostics {
    double current_volume = 0.0;
    double current_area = 0.0;
    double volume_drift_pct = 0.0;
    double area_drift_pct = 0.0;

    double stretch_residual = 0.0;
    double area_residual = 0.0;
    double volume_residual = 0.0;
    double bend_residual = 0.0;

    double fluid_density_error_avg = 0.0;
    double fluid_density_error_max = 0.0;

    int penetration_count = 0;
    double max_penetration_depth = 0.0;

    double energy = 1.0;
    double homeostasis_score = 0.0;
    double centroid_speed = 0.0;

    double step_time_ms = 0.0;
    double predict_time_ms = 0.0;
    double neighbor_build_time_ms = 0.0;
    double fluid_solve_time_ms = 0.0;
    double coupling_time_ms = 0.0;
    double constraint_solve_time_ms = 0.0;
};

struct BoundarySdfGrid {
    Vec3 origin = Vec3(0.0, 0.0, 0.0);
    double cell_size = 0.2;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    bool valid = false;

    std::vector<float> sdf; // signed distance
    std::vector<std::uint32_t> nearest_tri;
    std::vector<Vec3> nearest_normal;
};

struct SimState {
    std::vector<Vec3> membrane_positions;
    std::vector<Vec3> membrane_velocities;
    std::vector<Vec3> membrane_pred_positions;
    std::vector<Vec3> membrane_normals;

    std::vector<Triangle> membrane_triangles;
    std::vector<Edge> membrane_edges;
    std::vector<std::uint32_t> membrane_indices;

    std::vector<double> membrane_inv_masses;
    std::vector<double> membrane_stretch_lambdas;
    double membrane_area_lambda = 0.0;
    double membrane_volume_lambda = 0.0;

    // Fluid particles (AoS for now, with packed view exposed directly).
    std::vector<Vec3> particle_positions;
    std::vector<Vec3> particle_velocities;
    std::vector<Vec3> particle_pred_positions;
    std::vector<Vec3> particle_position_prev;
    std::vector<double> particle_masses;
    std::vector<double> particle_densities;
    std::vector<double> particle_pressures;
    std::vector<double> particle_lambdas;
    sph::NeighborGrid particle_grid;
    BoundarySdfGrid boundary_sdf;

    Vec3 membrane_centroid = Vec3(0.0, 0.0, 0.0);
    double membrane_mean_radius = 1.0;

    Vec3 polarity = Vec3(1.0, 0.0, 0.0);
    double energy = 1.0;

    Diagnostics diagnostics;
};

struct SimView {
    Span<const Vec3> membrane_positions;
    Span<const Vec3> membrane_normals;
    Span<const std::uint32_t> membrane_indices;
    Span<const Vec3> particle_positions;
};

} // namespace cells
