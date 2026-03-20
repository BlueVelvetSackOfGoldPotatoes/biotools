#include "core/io/run_logger.h"
#include "models/cells/src/cell_tasks.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

int env_int(const char* key, int fallback) {
    const char* v = std::getenv(key);
    if (!v || !*v) return fallback;
    try { return std::stoi(v); } catch (...) { return fallback; }
}

double env_double(const char* key, double fallback) {
    const char* v = std::getenv(key);
    if (!v || !*v) return fallback;
    try { return std::stod(v); } catch (...) { return fallback; }
}

std::string json_params(int steps, int seed, double dt, double damage_frac) {
    std::ostringstream oss;
    oss << "{" << '"' << "steps" << '"' << ":" << steps << ","
        << '"' << "seed" << '"' << ":" << seed << ","
        << '"' << "dt" << '"' << ":" << dt << ","
        << '"' << "damage_frac" << '"' << ":" << damage_frac << "}";
    return oss.str();
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char c) {
            return c == ' ' || c == '\t' || c == '\n' || c == '\r';
        }), token.end());
        if (!token.empty()) out.push_back(token);
    }
    return out;
}

bool contains(const std::vector<std::string>& xs, const std::string& v) {
    return std::find(xs.begin(), xs.end(), v) != xs.end();
}

} // namespace

int main() {
    const int seed = env_int("CELL_TASK_SEED", 42);
    const int steps = env_int("CELL_TASK_STEPS", 1000);
    const double dt = env_double("CELL_TASK_DT", 0.005);
    const double damage_frac = env_double("CELL_TASK_DAMAGE_FRAC", 0.18);

    cells::CellInit init;
    init.seed = static_cast<unsigned int>(std::max(0, seed));
    init.icosphere_subdivisions = std::max(1, env_int("CELL_TASK_SUBDIVISIONS", 2));
    init.fluid_particles = static_cast<std::size_t>(std::max(100, env_int("CELL_TASK_PARTICLES", 800)));
    init.params.dt = dt;
    init.params.enable_active_forces = true;
    init.params.active_force = env_double("CELL_TASK_ACTIVE_FORCE", 0.28);
    init.params.substrate_friction = env_double("CELL_TASK_SUBSTRATE_FRICTION", 0.9);
    init.params.substrate_z = env_double("CELL_TASK_SUBSTRATE_Z", -0.75);
    init.params.polarity_noise = env_double("CELL_TASK_POLARITY_NOISE", 0.015);

    RunLogger logger("cells", "cell_task_suite", seed, "sim-v1",
                     json_params(steps, seed, dt, damage_frac),
                     "cell_tasks", "Cell Tasks", "control");

    std::filesystem::create_directories("output/cells");

    const std::string controllers_env = [] {
        const char* v = std::getenv("CELL_TASK_CONTROLLERS");
        return (v && *v) ? std::string(v) : std::string("homeostatic,rnn,gru,lstm,transformer");
    }();
    const std::vector<std::string> selected = split_csv(controllers_env);

    cells::HomeostaticCellController homeo;
    cells::RecurrentBaselineController rnn(static_cast<std::size_t>(std::max(2, env_int("CELL_TASK_RNN_HIDDEN", 4))));
    cells::GruBaselineController gru(static_cast<std::size_t>(std::max(2, env_int("CELL_TASK_GRU_HIDDEN", 4))));
    cells::LstmBaselineController lstm(static_cast<std::size_t>(std::max(2, env_int("CELL_TASK_LSTM_HIDDEN", 4))));
    cells::TransformerBaselineController transformer(
        static_cast<std::size_t>(std::max(4, env_int("CELL_TASK_TRANSFORMER_DIM", 12))),
        static_cast<std::size_t>(std::max(2, env_int("CELL_TASK_TRANSFORMER_WINDOW", 6))));

    std::vector<cells::CellTaskController*> controllers;
    if (contains(selected, "homeostatic")) controllers.push_back(&homeo);
    if (contains(selected, "rnn")) controllers.push_back(&rnn);
    if (contains(selected, "gru")) controllers.push_back(&gru);
    if (contains(selected, "lstm")) controllers.push_back(&lstm);
    if (contains(selected, "transformer")) controllers.push_back(&transformer);
    if (controllers.empty()) {
        std::cerr << "CELL_TASK_CONTROLLERS selected no valid controllers. Use any of: "
                  << "homeostatic,rnn,gru,lstm,transformer" << std::endl;
        return 2;
    }
    const std::vector<cells::CellTaskKind> tasks{
        cells::CellTaskKind::Chemotaxis,
        cells::CellTaskKind::DistributionShift,
        cells::CellTaskKind::DamageRecovery,
    };

    for (cells::CellTaskKind kind : tasks) {
        for (cells::CellTaskController* controller : controllers) {
            cells::CellTaskConfig cfg;
            cfg.kind = kind;
            cfg.init = init;
            cfg.steps = static_cast<std::size_t>(std::max(1, steps));
            cfg.dt = dt;
            cfg.seed = static_cast<unsigned int>(seed);
            cfg.damage_fraction = damage_frac;
            cfg.damage_step = std::max(10, steps / 3);
            cfg.shift_every = std::max(20, steps / 4);
            cfg.target_radius = env_double("CELL_TASK_TARGET_RADIUS", 2.0);

            const cells::CellTaskEpisode episode = cells::run_cell_task_episode(cfg, *controller);
            const std::string task_name = cells::cell_task_kind_name(kind);
            const std::string trace_path = "output/cells/task_" + task_name + "_" + controller->name() + ".csv";
            cells::write_cell_task_csv(trace_path, episode);

            logger.append_csv_row(
                "model_specific/cells/task_summary.csv",
                {"run_id","task","controller","controller_parameters","cumulative_reward","mean_distance",
                 "final_distance","mean_energy","mean_homeostasis","max_volume_drift_pct","max_area_drift_pct",
                 "shift_count","recovery_steps","trace_csv"},
                {logger.run_id(), task_name, controller->name(),
                 std::to_string(episode.metrics.controller_parameters),
                 std::to_string(episode.metrics.cumulative_reward),
                 std::to_string(episode.metrics.mean_distance),
                 std::to_string(episode.metrics.final_distance),
                 std::to_string(episode.metrics.mean_energy),
                 std::to_string(episode.metrics.mean_homeostasis),
                 std::to_string(episode.metrics.max_volume_drift_pct),
                 std::to_string(episode.metrics.max_area_drift_pct),
                 std::to_string(episode.metrics.shift_count),
                 std::to_string(episode.metrics.recovery_steps),
                 trace_path});

            std::cout << "[cell-task] task=" << task_name
                      << " controller=" << controller->name()
                      << " reward=" << episode.metrics.cumulative_reward
                      << " final_distance=" << episode.metrics.final_distance
                      << " recovery_steps=" << episode.metrics.recovery_steps
                      << " edges=n/a"
                      << std::endl;
        }
    }

    logger.write_manifest_end();
    return 0;
}
