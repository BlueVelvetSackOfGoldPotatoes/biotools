// modulus_graph.h - Mesh graph construction, partitioning, edge features
// Port of physicsnemo graph construction utilities
//
// C++17, no external dependencies.

#ifndef MODULUS_GRAPH_H
#define MODULUS_GRAPH_H

#include "modulus.h"
#include "modulus_geometry.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <numeric>
#include <vector>

namespace modulus {
namespace graph {

using geometry::Vec3;

// ============================================================
// Edge feature computation
// ============================================================

// Compute relative position edge features
inline Tensor compute_edge_features_relative(const Tensor& positions,
                                              const std::vector<int>& src,
                                              const std::vector<int>& dst) {
    int n_edges = (int)src.size();
    int dim = positions.shape[1];
    // Features: relative position (dim) + distance (1)
    Tensor ef({n_edges, dim + 1});
    for (int e = 0; e < n_edges; ++e) {
        double dist = 0;
        for (int d = 0; d < dim; ++d) {
            double diff = positions.at2(dst[e], d) - positions.at2(src[e], d);
            ef.at2(e, d) = diff;
            dist += diff * diff;
        }
        ef.at2(e, dim) = std::sqrt(dist);
    }
    return ef;
}

// Compute edge features with angle information (for 3D)
inline Tensor compute_edge_features_with_angles(const Tensor& positions,
                                                 const std::vector<int>& src,
                                                 const std::vector<int>& dst) {
    int n_edges = (int)src.size();
    // Features: dx, dy, dz, distance, azimuth, elevation (6 features for 3D)
    Tensor ef({n_edges, 6});
    for (int e = 0; e < n_edges; ++e) {
        double dx = positions.at2(dst[e], 0) - positions.at2(src[e], 0);
        double dy = positions.at2(dst[e], 1) - positions.at2(src[e], 1);
        double dz = positions.at2(dst[e], 2) - positions.at2(src[e], 2);
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        double azimuth = std::atan2(dy, dx);
        double elevation = (dist > 1e-15) ? std::asin(dz / dist) : 0.0;

        ef.at2(e, 0) = dx;
        ef.at2(e, 1) = dy;
        ef.at2(e, 2) = dz;
        ef.at2(e, 3) = dist;
        ef.at2(e, 4) = azimuth;
        ef.at2(e, 5) = elevation;
    }
    return ef;
}

// ============================================================
// Graph construction from mesh
// ============================================================

// Build graph from triangle mesh
struct MeshGraph {
    int num_nodes;
    int num_edges;
    std::vector<int> edge_src, edge_dst;
    Tensor node_positions;  // (n_nodes, 3)
    Tensor edge_features;   // (n_edges, feat_dim)

    // From list of triangles (shares vertices by proximity)
    static MeshGraph from_triangles(const std::vector<geometry::Triangle>& triangles,
                                    double merge_tolerance = 1e-8) {
        MeshGraph g;

        // Collect unique vertices
        std::vector<Vec3> vertices;
        auto find_or_add = [&](const Vec3& v) -> int {
            for (int i = 0; i < (int)vertices.size(); ++i) {
                if ((vertices[i] - v).length() < merge_tolerance)
                    return i;
            }
            vertices.push_back(v);
            return (int)vertices.size() - 1;
        };

        // Build connectivity
        std::set<std::pair<int,int>> edge_set;
        for (auto& tri : triangles) {
            int i0 = find_or_add(tri.v0);
            int i1 = find_or_add(tri.v1);
            int i2 = find_or_add(tri.v2);

            auto add_edge = [&](int a, int b) {
                if (a != b) {
                    edge_set.insert({a, b});
                    edge_set.insert({b, a}); // bidirectional
                }
            };
            add_edge(i0, i1);
            add_edge(i1, i2);
            add_edge(i2, i0);
        }

        g.num_nodes = (int)vertices.size();
        g.node_positions = Tensor({g.num_nodes, 3});
        for (int i = 0; i < g.num_nodes; ++i) {
            g.node_positions.at2(i, 0) = vertices[i].x;
            g.node_positions.at2(i, 1) = vertices[i].y;
            g.node_positions.at2(i, 2) = vertices[i].z;
        }

        for (auto& [s, d] : edge_set) {
            g.edge_src.push_back(s);
            g.edge_dst.push_back(d);
        }
        g.num_edges = (int)g.edge_src.size();

        // Compute edge features
        g.edge_features = compute_edge_features_relative(
            g.node_positions, g.edge_src, g.edge_dst);

        return g;
    }

    // From regular grid (2D)
    static MeshGraph from_grid_2d(int nx, int ny, double dx = 1.0, double dy = 1.0) {
        MeshGraph g;
        g.num_nodes = nx * ny;
        g.node_positions = Tensor({g.num_nodes, 2});

        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                int idx = i * ny + j;
                g.node_positions.at2(idx, 0) = i * dx;
                g.node_positions.at2(idx, 1) = j * dy;
            }
        }

        // 4-connectivity
        auto valid = [&](int i, int j) { return i >= 0 && i < nx && j >= 0 && j < ny; };
        int offsets[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};

        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                int src = i * ny + j;
                for (auto& [di, dj] : offsets) {
                    if (valid(i + di, j + dj)) {
                        int dst = (i + di) * ny + (j + dj);
                        g.edge_src.push_back(src);
                        g.edge_dst.push_back(dst);
                    }
                }
            }
        }
        g.num_edges = (int)g.edge_src.size();
        g.edge_features = compute_edge_features_relative(
            g.node_positions, g.edge_src, g.edge_dst);
        return g;
    }
};

// ============================================================
// Graph partitioning (simple spatial partitioning)
// ============================================================

struct GraphPartition {
    std::vector<std::vector<int>> node_assignments; // partition -> list of node indices
    int num_partitions;

    // Simple spatial bisection partitioning
    static GraphPartition spatial_partition(const Tensor& positions, int n_parts) {
        GraphPartition gp;
        gp.num_partitions = n_parts;
        gp.node_assignments.resize(n_parts);

        int n = positions.shape[0];
        int dim = positions.shape[1];

        // Find dimension with largest extent
        double max_extent = 0;
        int split_dim = 0;
        for (int d = 0; d < dim; ++d) {
            double lo = 1e30, hi = -1e30;
            for (int i = 0; i < n; ++i) {
                lo = std::min(lo, positions.at2(i, d));
                hi = std::max(hi, positions.at2(i, d));
            }
            if (hi - lo > max_extent) {
                max_extent = hi - lo;
                split_dim = d;
            }
        }

        // Sort nodes by position along split dimension
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return positions.at2(a, split_dim) < positions.at2(b, split_dim);
        });

        // Assign to partitions
        int per_part = (n + n_parts - 1) / n_parts;
        for (int i = 0; i < n; ++i) {
            int part = std::min(i / per_part, n_parts - 1);
            gp.node_assignments[part].push_back(indices[i]);
        }

        return gp;
    }
};

// ============================================================
// Icosahedral mesh generation (for GraphCast)
// ============================================================
class IcosahedralMesh {
public:
    Tensor positions;  // (n_nodes, 3) on unit sphere
    std::vector<int> edge_src, edge_dst;
    int num_nodes;

    IcosahedralMesh() : num_nodes(0) {}

    // Generate icosahedral mesh at given refinement level
    static IcosahedralMesh generate(int level) {
        IcosahedralMesh mesh;

        // Start with base icosahedron (12 vertices, 20 faces)
        double phi = (1.0 + std::sqrt(5.0)) / 2.0;
        std::vector<Vec3> verts = {
            {-1, phi, 0}, {1, phi, 0}, {-1, -phi, 0}, {1, -phi, 0},
            {0, -1, phi}, {0, 1, phi}, {0, -1, -phi}, {0, 1, -phi},
            {phi, 0, -1}, {phi, 0, 1}, {-phi, 0, -1}, {-phi, 0, 1}
        };
        // Normalize to unit sphere
        for (auto& v : verts) v = v.normalized();

        // Icosahedron faces (20 triangles)
        std::vector<std::array<int, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

        // Subdivide
        for (int l = 0; l < level; ++l) {
            std::vector<std::array<int, 3>> new_faces;
            std::map<std::pair<int,int>, int> edge_midpoints;

            auto get_midpoint = [&](int a, int b) -> int {
                auto key = std::make_pair(std::min(a, b), std::max(a, b));
                auto it = edge_midpoints.find(key);
                if (it != edge_midpoints.end()) return it->second;
                Vec3 mid = ((verts[a] + verts[b]) * 0.5).normalized();
                int idx = (int)verts.size();
                verts.push_back(mid);
                edge_midpoints[key] = idx;
                return idx;
            };

            for (auto& f : faces) {
                int a = f[0], b = f[1], c = f[2];
                int ab = get_midpoint(a, b);
                int bc = get_midpoint(b, c);
                int ca = get_midpoint(c, a);
                new_faces.push_back({a, ab, ca});
                new_faces.push_back({b, bc, ab});
                new_faces.push_back({c, ca, bc});
                new_faces.push_back({ab, bc, ca});
            }
            faces = new_faces;
        }

        mesh.num_nodes = (int)verts.size();
        mesh.positions = Tensor({mesh.num_nodes, 3});
        for (int i = 0; i < mesh.num_nodes; ++i) {
            mesh.positions.at2(i, 0) = verts[i].x;
            mesh.positions.at2(i, 1) = verts[i].y;
            mesh.positions.at2(i, 2) = verts[i].z;
        }

        // Build edges from faces
        std::set<std::pair<int,int>> edge_set;
        for (auto& f : faces) {
            auto add = [&](int a, int b) {
                edge_set.insert({a, b});
                edge_set.insert({b, a});
            };
            add(f[0], f[1]);
            add(f[1], f[2]);
            add(f[2], f[0]);
        }

        for (auto& [s, d] : edge_set) {
            mesh.edge_src.push_back(s);
            mesh.edge_dst.push_back(d);
        }

        return mesh;
    }
};

} // namespace graph
} // namespace modulus

#endif // MODULUS_GRAPH_H
