#include "models/cells/cellengine/cellengine.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int env_int(const char* key, const int fallback) {
    const char* v = std::getenv(key);
    if (!v || !*v) return fallback;
    try { return std::stoi(v); } catch (...) { return fallback; }
}

double env_double(const char* key, const double fallback) {
    const char* v = std::getenv(key);
    if (!v || !*v) return fallback;
    try { return std::stod(v); } catch (...) { return fallback; }
}

std::string env_string(const char* key, const std::string& fallback) {
    const char* v = std::getenv(key);
    return (v && *v) ? std::string(v) : fallback;
}

} // namespace

int main() {
    cells::cellengine::CellEngineConfig cfg;
    cfg.mode = env_string("CELLENGINE_MODE", cfg.mode);
    cfg.task_name = env_string("CELLENGINE_TASK", cfg.task_name);
    cfg.body_mode = env_string("CELLENGINE_BODY_MODE", cfg.body_mode);
    cfg.rl_algorithm = env_string("CELLENGINE_RL_ALGO", cfg.rl_algorithm);
    cfg.load_genome_csv = env_string("CELLENGINE_LOAD_GENOME", cfg.load_genome_csv);
    cfg.output_dir = env_string("CELLENGINE_OUT", "reports/cellengine_latest");
    cfg.seed = env_int("CELLENGINE_SEED", cfg.seed);
    cfg.population_size = env_int("CELLENGINE_POP", cfg.population_size);
    cfg.generations = env_int("CELLENGINE_GENS", cfg.generations);
    cfg.tournament_size = env_int("CELLENGINE_TOURNAMENT", cfg.tournament_size);
    cfg.elitism = env_int("CELLENGINE_ELITISM", cfg.elitism);
    cfg.mutation_rate = env_double("CELLENGINE_MUT_RATE", cfg.mutation_rate);
    cfg.mutation_sigma = env_double("CELLENGINE_MUT_SIGMA", cfg.mutation_sigma);
    cfg.crossover_rate = env_double("CELLENGINE_CROSSOVER", cfg.crossover_rate);
    cfg.cell_ticks_per_physics = env_int("CELLENGINE_CELL_TICKS", cfg.cell_ticks_per_physics);
    cfg.physics_dt = env_double("CELLENGINE_DT", cfg.physics_dt);
    cfg.stress_relax_iters = env_int("CELLENGINE_STRESS_ITERS", cfg.stress_relax_iters);
    cfg.auto_attempts = env_int("CELLENGINE_AUTO_ATTEMPTS", cfg.auto_attempts);
    cfg.hint_genomes = env_int("CELLENGINE_HINTS", cfg.hint_genomes);
    cfg.rl_train_episodes = env_int("CELLENGINE_RL_EPISODES", cfg.rl_train_episodes);
    cfg.rl_damage_finetune_episodes = env_int("CELLENGINE_RL_FINETUNE", cfg.rl_damage_finetune_episodes);
    cfg.rl_eval_every = env_int("CELLENGINE_RL_EVAL_EVERY", cfg.rl_eval_every);
    cfg.rl_force_mag = env_double("CELLENGINE_RL_FORCE", cfg.rl_force_mag);
    cfg.solve_ratio = env_double("CELLENGINE_SOLVE_RATIO", cfg.solve_ratio);
    cfg.damage_fraction = env_double("CELLENGINE_DAMAGE_FRAC", cfg.damage_fraction);
    cfg.damage_after_trial = env_int("CELLENGINE_DAMAGE_AFTER_TRIAL", cfg.damage_after_trial);
    cfg.replay_sleep_ms = env_int("CELLENGINE_REPLAY_SLEEP_MS", cfg.replay_sleep_ms);
    cfg.replay_frame_stride = env_int("CELLENGINE_REPLAY_FRAME_STRIDE", cfg.replay_frame_stride);
    cfg.replay_damage_tick = env_int("CELLENGINE_REPLAY_DAMAGE_TICK", cfg.replay_damage_tick);
    cfg.evolve_body_mode = env_int("CELLENGINE_EVOLVE_BODY_MODE", cfg.evolve_body_mode ? 1 : 0) != 0;
    cfg.evolve_development_steps = env_int("CELLENGINE_EVOLVE_DEVELOP_STEPS", cfg.evolve_development_steps ? 1 : 0) != 0;
    cfg.evolve_development_seed_half_width = env_int("CELLENGINE_EVOLVE_SEED_HALF_WIDTH", cfg.evolve_development_seed_half_width ? 1 : 0) != 0;
    cfg.evolve_max_cells = env_int("CELLENGINE_EVOLVE_MAX_CELLS", cfg.evolve_max_cells ? 1 : 0) != 0;
    cfg.evolve_body_extent_x = env_int("CELLENGINE_EVOLVE_BODY_EXTENT_X", cfg.evolve_body_extent_x ? 1 : 0) != 0;
    cfg.evolve_body_extent_y = env_int("CELLENGINE_EVOLVE_BODY_EXTENT_Y", cfg.evolve_body_extent_y ? 1 : 0) != 0;
    cfg.evolve_body_extent_z = env_int("CELLENGINE_EVOLVE_BODY_EXTENT_Z", cfg.evolve_body_extent_z ? 1 : 0) != 0;
    cfg.evolve_chemical_diffusion_steps = env_int("CELLENGINE_EVOLVE_CHEM_STEPS", cfg.evolve_chemical_diffusion_steps ? 1 : 0) != 0;
    cfg.evolve_growth_threshold = env_int("CELLENGINE_EVOLVE_GROWTH_THRESHOLD", cfg.evolve_growth_threshold ? 1 : 0) != 0;
    cfg.evolve_chemical_diffusion_rate = env_int("CELLENGINE_EVOLVE_CHEM_RATE", cfg.evolve_chemical_diffusion_rate ? 1 : 0) != 0;
    cfg.evolve_chemical_decay = env_int("CELLENGINE_EVOLVE_CHEM_DECAY", cfg.evolve_chemical_decay ? 1 : 0) != 0;
    cfg.development_steps = env_int("CELLENGINE_DEVELOP_STEPS", cfg.development_steps);
    cfg.development_seed_half_width = env_int("CELLENGINE_SEED_HALF_WIDTH", cfg.development_seed_half_width);
    cfg.max_cells = env_int("CELLENGINE_MAX_CELLS", cfg.max_cells);
    cfg.body_extent_x = env_int("CELLENGINE_BODY_EXTENT_X", cfg.body_extent_x);
    cfg.body_extent_y = env_int("CELLENGINE_BODY_EXTENT_Y", cfg.body_extent_y);
    cfg.body_extent_z = env_int("CELLENGINE_BODY_EXTENT_Z", cfg.body_extent_z);
    cfg.chemical_diffusion_steps = env_int("CELLENGINE_CHEM_DIFF_STEPS", cfg.chemical_diffusion_steps);
    cfg.replay_clear_screen = env_int("CELLENGINE_REPLAY_CLEAR", cfg.replay_clear_screen ? 1 : 0) != 0;
    cfg.replay_emit_stdout = env_int("CELLENGINE_REPLAY_STDOUT", cfg.replay_emit_stdout ? 1 : 0) != 0;
    cfg.odd_enabled = env_int("CELLENGINE_ODD", cfg.odd_enabled ? 1 : 0) != 0;
    cfg.development_growth_threshold = env_double("CELLENGINE_GROWTH_THRESHOLD", cfg.development_growth_threshold);
    cfg.chemical_diffusion_rate = env_double("CELLENGINE_CHEM_DIFF_RATE", cfg.chemical_diffusion_rate);
    cfg.chemical_decay = env_double("CELLENGINE_CHEM_DECAY", cfg.chemical_decay);
    cfg.replay_theta_deg = env_double("CELLENGINE_REPLAY_THETA_DEG", cfg.replay_theta_deg);
    cfg.replay_task_a = env_double("CELLENGINE_REPLAY_TASK_A", cfg.replay_task_a);
    cfg.replay_task_b = env_double("CELLENGINE_REPLAY_TASK_B", cfg.replay_task_b);
    cfg.worm_goal_distance = env_double("CELLENGINE_WORM_GOAL_DISTANCE", cfg.worm_goal_distance);
    cfg.worm_max_backward = env_double("CELLENGINE_WORM_MAX_BACKWARD", cfg.worm_max_backward);
    cfg.pong_target_hits = env_int("CELLENGINE_PONG_TARGET_HITS", cfg.pong_target_hits);
    cfg.pong_ball_speed = env_double("CELLENGINE_PONG_BALL_SPEED", cfg.pong_ball_speed);
    cfg.pong_paddle_half_height = env_double("CELLENGINE_PONG_PADDLE_HALF", cfg.pong_paddle_half_height);
    cfg.search_budget.num_trials = env_int("CELLENGINE_SEARCH_TRIALS", cfg.search_budget.num_trials);
    cfg.search_budget.max_ticks = env_int("CELLENGINE_SEARCH_TICKS", cfg.search_budget.max_ticks);
    cfg.final_budget.num_trials = env_int("CELLENGINE_FINAL_TRIALS", cfg.final_budget.num_trials);
    cfg.final_budget.max_ticks = env_int("CELLENGINE_FINAL_TICKS", cfg.final_budget.max_ticks);
    cfg.damage_budget.num_trials = env_int("CELLENGINE_DAMAGE_TRIALS", cfg.damage_budget.num_trials);
    cfg.damage_budget.max_ticks = env_int("CELLENGINE_DAMAGE_TICKS", cfg.damage_budget.max_ticks);
    cfg.odd_budget.num_trials = env_int("CELLENGINE_ODD_TRIALS", cfg.odd_budget.num_trials);
    cfg.odd_budget.max_ticks = env_int("CELLENGINE_ODD_TICKS", cfg.odd_budget.max_ticks);

    const std::string out_dir = cfg.output_dir;

    std::cout << "=== CellEngine Benchmark (GA cells vs RL comparators) ===\n";
    std::cout << "mode=" << cfg.mode
              << " task=" << cfg.task_name
              << " body=" << cfg.body_mode
              << " seed=" << cfg.seed
              << " pop=" << cfg.population_size
              << " gens=" << cfg.generations
              << " search_trials=" << cfg.search_budget.num_trials
              << " final_trials=" << cfg.final_budget.num_trials
              << std::endl;

    const auto summary = cells::cellengine::run_full_benchmark(cfg);
    std::cout << cells::cellengine::format_summary(summary) << std::endl;

    if (!cells::cellengine::write_report(summary, cfg, out_dir)) {
        std::cerr << "failed to write report to " << out_dir << std::endl;
        return 2;
    }

    return summary.solved ? 0 : 1;
}
