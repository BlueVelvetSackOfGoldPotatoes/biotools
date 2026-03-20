#pragma once

#include "models/cells/src/sim/sim_params.h"
#include "models/cells/src/sim/sim_state.h"

#include <random>
#include <string>

namespace cells {

struct CellInit {
    int icosphere_subdivisions = 3;
    double radius = 1.0;
    std::size_t fluid_particles = 5000;
    unsigned int seed = 42;
    SimParams params;
};

struct CellControlInput {
    Vec3 chemo_gradient = Vec3(0.0, 0.0, 0.0);
    Vec3 body_force = Vec3(0.0, 0.0, 0.0);
    Vec3 polarity_hint = Vec3(0.0, 0.0, 0.0);
    double active_force_scale = 1.0;
    double substrate_friction_scale = 1.0;
    double polarity_blend = 0.0;
};

class CellSim {
public:
    explicit CellSim(const CellInit& init = CellInit{});

    void step(double dt = -1.0);
    void step(const CellControlInput& control, double dt = -1.0);
    void poke_vertex(std::size_t index, const Vec3& impulse);

    std::size_t vertex_count() const;
    std::size_t triangle_count() const;
    std::size_t edge_count() const;
    std::size_t particle_count() const;

    // For controlled perturbation studies.
    std::size_t damage_random_vertices(double frac, unsigned int seed = 0);
    std::size_t damage_random_particles(double frac, unsigned int seed = 0);

    const SimState& state() const;
    const Diagnostics& diagnostics() const;
    SimView view() const;

    SimParams& params();
    const SimParams& params() const;

private:
    void init_particles(std::size_t n, double radius, unsigned int seed);
    void recompute_geometry();
    void step_fluid(double dt);
    void step_coupling(double dt);
    void apply_active_behavior(const CellControlInput& control, double dt);
    void update_homeostasis(double dt);
    void build_boundary_sdf();
    bool sample_boundary_sdf(const Vec3& p, double& signed_distance, Vec3& gradient, std::size_t& tri_id) const;
    bool segment_hits_membrane(const Vec3& p0, const Vec3& p1, Vec3& hit, Vec3& normal, Vec3& bary, std::size_t& tri_id) const;
    bool point_inside_membrane(const Vec3& p) const;
    std::size_t nearest_triangle(const Vec3& p, Vec3& closest, Vec3& normal, Vec3& bary) const;

    SimParams params_;
    SimState state_;
    std::mt19937 rng_;
};

} // namespace cells
