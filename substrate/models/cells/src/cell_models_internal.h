#pragma once

#include "models/cells/src/cell_models.h"

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

double project_stretch(
    std::vector<Vec3>& predicted_positions,
    const std::vector<Edge>& edges,
    const std::vector<double>& inv_masses,
    std::vector<double>& lambdas,
    double compliance,
    double dt);

double project_bend(
    std::vector<Vec3>& predicted_positions,
    const std::vector<Triangle>& triangles,
    const std::vector<Edge>& edges,
    const std::vector<double>& inv_masses,
    double compliance,
    double dt);

double project_area(
    std::vector<Vec3>& predicted_positions,
    const std::vector<Triangle>& triangles,
    const std::vector<double>& inv_masses,
    double target_area,
    double& lambda,
    double compliance,
    double dt);

double project_volume(
    std::vector<Vec3>& predicted_positions,
    const std::vector<Triangle>& triangles,
    const std::vector<double>& inv_masses,
    double target_volume,
    double& lambda,
    double compliance,
    double dt);

} // namespace membrane
} // namespace cells
