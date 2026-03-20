#include "models/cells/src/cell_models.h"
#include "models/cells/src/core/log.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int env_int(const char* key, int fallback) {
    const char* v = std::getenv(key);
    if (v == nullptr || *v == '\0') {
        return fallback;
    }
    try {
        return std::stoi(v);
    } catch (...) {
        return fallback;
    }
}

double env_double(const char* key, double fallback) {
    const char* v = std::getenv(key);
    if (v == nullptr || *v == '\0') {
        return fallback;
    }
    try {
        return std::stod(v);
    } catch (...) {
        return fallback;
    }
}

std::string fmt(double x) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << x;
    return oss.str();
}

} // namespace

int main() {
    const int steps = env_int("CELLS_STEPS", 10000);
    const int subdivisions = env_int("CELLS_SUBDIVISIONS", 3);
    const double radius = env_double("CELLS_RADIUS", 1.0);
    const double dt = env_double("CELLS_DT", 0.005);
    const int particles = std::max(0, env_int("CELLS_PARTICLES", 5000));
    const int seed = env_int("CELLS_SEED", 42);
    const int log_every = std::max(1, env_int("CELLS_LOG_EVERY", 1));
    const int poke_every = env_int("CELLS_POKE_EVERY", 500);
    const double poke_mag = env_double("CELLS_POKE_MAG", 0.35);
    const std::string preset = []{
        const char* p = std::getenv("CELLS_PRESET");
        return (p && *p) ? std::string(p) : std::string();
    }();
    const bool active_mode = env_int("CELLS_ACTIVE", 0) != 0;
    const bool damage_mode = env_int("CELLS_DAMAGE", 0) != 0;
    const double damage_frac = env_double("CELLS_DAMAGE_FRAC", 0.2);
    const int damage_step = env_int("CELLS_DAMAGE_STEP", steps / 2);

    cells::CellInit init;
    init.icosphere_subdivisions = subdivisions;
    init.radius = radius;
    init.fluid_particles = static_cast<std::size_t>(particles);
    init.seed = static_cast<unsigned int>(std::max(0, seed));
    if (!preset.empty()) {
        init.params = cells::SimParams::preset(preset);
    }
    init.params.dt = dt;
    init.params.enable_active_forces = active_mode;
    if (active_mode) {
        init.params.active_force = env_double("CELLS_ACTIVE_FORCE", 0.25);
        init.params.substrate_friction = env_double("CELLS_SUBSTRATE_FRICTION", 0.8);
        init.params.substrate_z = env_double("CELLS_SUBSTRATE_Z", -radius * 0.8);
    }

    cells::CellSim sim(init);

    std::filesystem::create_directories("output/cells");
    const std::string out_path = "output/cells/cells_diagnostics.csv";
    cells::CsvLogger logger;
    if (!logger.open(out_path,
                     "step,volume,area,volume_drift_pct,area_drift_pct,stretch_residual,area_residual,volume_residual,bend_residual,fluid_density_error_avg,fluid_density_error_max,penetration_count,max_penetration_depth,energy,homeostasis_score,centroid_speed,predict_time_ms,neighbor_build_time_ms,fluid_solve_time_ms,constraint_solve_time_ms,coupling_time_ms,step_time_ms")) {
        std::cerr << "Failed to open diagnostics output: " << out_path << std::endl;
        return 1;
    }

    std::cout << "[cells] start"
              << " steps=" << steps
              << " subdivisions=" << subdivisions
              << " vertices=" << sim.vertex_count()
              << " triangles=" << sim.triangle_count()
              << " edges=" << sim.edge_count()
              << " particles=" << sim.particle_count()
              << std::endl;

    for (int step = 0; step < steps; ++step) {
        if (poke_every > 0 && step > 0 && (step % poke_every) == 0) {
            sim.poke_vertex(0, cells::Vec3(poke_mag, 0.0, 0.0));
        }
        if (damage_mode && step == damage_step) {
            const std::size_t dv = sim.damage_random_vertices(damage_frac, static_cast<unsigned int>(seed + 111));
            const std::size_t dp = sim.damage_random_particles(damage_frac, static_cast<unsigned int>(seed + 222));
            std::cout << "[cells] damage injected step=" << step
                      << " damaged_vertices=" << dv
                      << " damaged_particles=" << dp << std::endl;
        }

        sim.step(dt);
        const auto& d = sim.diagnostics();

        if ((step % log_every) == 0) {
            logger.write_row(
                std::to_string(step) + "," +
                fmt(d.current_volume) + "," +
                fmt(d.current_area) + "," +
                fmt(d.volume_drift_pct) + "," +
                fmt(d.area_drift_pct) + "," +
                fmt(d.stretch_residual) + "," +
                fmt(d.area_residual) + "," +
                fmt(d.volume_residual) + "," +
                fmt(d.bend_residual) + "," +
                fmt(d.fluid_density_error_avg) + "," +
                fmt(d.fluid_density_error_max) + "," +
                std::to_string(d.penetration_count) + "," +
                fmt(d.max_penetration_depth) + "," +
                fmt(d.energy) + "," +
                fmt(d.homeostasis_score) + "," +
                fmt(d.centroid_speed) + "," +
                fmt(d.predict_time_ms) + "," +
                fmt(d.neighbor_build_time_ms) + "," +
                fmt(d.fluid_solve_time_ms) + "," +
                fmt(d.constraint_solve_time_ms) + "," +
                fmt(d.coupling_time_ms) + "," +
                fmt(d.step_time_ms));
        }

        if ((step % 1000) == 0 || step == steps - 1) {
            std::cout << "[cells] step=" << step
                      << " vol_drift=" << fmt(d.volume_drift_pct) << "%"
                      << " area_drift=" << fmt(d.area_drift_pct) << "%"
                      << " rho_err=" << fmt(d.fluid_density_error_avg)
                      << " leaks=" << d.penetration_count
                      << " energy=" << fmt(d.energy)
                      << " stretch_res=" << fmt(d.stretch_residual)
                      << " vol_res=" << fmt(d.volume_residual)
                      << std::endl;
        }
    }

    logger.flush();

    const auto& d = sim.diagnostics();
    std::cout << "[cells] done"
              << " final_volume=" << fmt(d.current_volume)
              << " final_area=" << fmt(d.current_area)
              << " volume_drift_pct=" << fmt(d.volume_drift_pct)
              << " area_drift_pct=" << fmt(d.area_drift_pct)
              << " csv=" << out_path
              << std::endl;

    return 0;
}
