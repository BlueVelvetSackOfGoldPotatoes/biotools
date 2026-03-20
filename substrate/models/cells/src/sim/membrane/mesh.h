#pragma once

#include "models/cells/src/sim/sim_state.h"

#include <vector>

namespace cells {
namespace membrane {

void generate_icosphere(
    int subdivisions,
    double radius,
    std::vector<Vec3>& out_vertices,
    std::vector<Triangle>& out_triangles);

void build_edges(
    const std::vector<Vec3>& vertices,
    const std::vector<Triangle>& triangles,
    std::vector<Edge>& out_edges);

void flatten_indices(
    const std::vector<Triangle>& triangles,
    std::vector<std::uint32_t>& out_indices);

void compute_vertex_normals(
    const std::vector<Vec3>& vertices,
    const std::vector<Triangle>& triangles,
    std::vector<Vec3>& out_normals);

double compute_area(
    const std::vector<Vec3>& vertices,
    const std::vector<Triangle>& triangles);

double compute_volume(
    const std::vector<Vec3>& vertices,
    const std::vector<Triangle>& triangles);

void compute_vertex_inv_masses(
    const std::vector<Vec3>& vertices,
    const std::vector<Triangle>& triangles,
    double surface_density,
    double thickness,
    std::vector<double>& out_inv_masses);

} // namespace membrane
} // namespace cells

