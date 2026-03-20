#pragma once

#include <string>

namespace cells {

struct SimParams {
    // Membrane constraints.
    double stretch_compliance = 1e-6;
    double area_compliance = 1e-4;
    double volume_compliance = 1e-7;
    double bend_compliance = 1e-3;
    bool enable_bending = false;

    // Global targets (set from initial mesh if <= 0).
    double target_volume = -1.0;
    double target_area = -1.0;

    // Membrane material.
    double membrane_surface_density = 1.0;
    double membrane_thickness = 0.01;

    // Iteration budgets.
    int membrane_iterations = 10;
    int fluid_iterations = 4;
    int coupling_iterations = 1;

    // Integration.
    double dt = 0.005;
    double global_damping = 0.98;

    // Fluid (PBF).
    double rest_density = 1000.0;
    double fluid_particle_mass = 1.0;
    double smoothing_length = 0.12;
    double pbf_epsilon = 100.0;
    double viscosity = 0.01;
    double gravity_z = 0.0;

    // Coupling.
    double noleak_inset = 0.1;
    double coupling_damping = 0.8;
    double coupling_impulse_cap = 0.5;
    bool coupling_enable_ccd = true;
    double ccd_displacement_threshold = 0.5; // in units of smoothing length

    // Active behavior.
    bool enable_active_forces = false;
    double active_force = 0.0;
    double polarity_noise = 0.02;
    double polarity_persistence = 0.9;
    double polarity_chemo = 0.0;
    double substrate_z = -1.0;
    double substrate_band = 0.05;
    double substrate_friction = 0.0;

    // Homeostasis and energy.
    double energy_max = 1.0;
    double energy_replenish = 0.01;
    double energy_cost_active = 0.005;
    double energy_cost_motion = 0.002;
    double homeostasis_tol_volume_pct = 2.0;
    double homeostasis_tol_area_pct = 3.0;
    double homeostasis_target_activity = 0.1;
    double homeostasis_tol_activity = 0.08;

    // Phase presets.
    static SimParams soft_blob();
    static SimParams stiff_blob();
    static SimParams watery_cytoplasm();
    static SimParams viscous_cytoplasm();
    static SimParams preset(const std::string& name);
};

} // namespace cells
