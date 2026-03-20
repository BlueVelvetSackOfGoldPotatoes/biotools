#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace cells {
namespace cellengine {

constexpr int kGeneExprDim = 8;
constexpr int kGenomeSize = 96;

struct TrialBudget {
    int num_trials = 24;
    int max_ticks = 250;
};

struct CandidateParams {
    std::string body_mode = "grown3d";
    int development_steps = 8;
    int development_seed_half_width = 2;
    int max_cells = 128;
    int body_extent_x = 7;
    int body_extent_y = 16;
    int body_extent_z = 2;
    int chemical_diffusion_steps = 2;
    double development_growth_threshold = 0.54;
    double chemical_diffusion_rate = 0.32;
    double chemical_decay = 0.08;
};

struct CellEngineConfig {
    std::string mode = "full";
    std::string task_name = "cartpole_balance";
    std::string body_mode = "grown3d";
    std::string rl_algorithm = "all";
    std::string load_genome_csv;
    std::string output_dir;
    int seed = 42;
    int population_size = 48;
    int generations = 50;
    int tournament_size = 4;
    int elitism = 4;
    double mutation_rate = 0.12;
    double mutation_sigma = 0.10;
    double crossover_rate = 0.35;
    int cell_ticks_per_physics = 4;
    double physics_dt = 0.02;
    int stress_relax_iters = 4;
    int auto_attempts = 3;
    int hint_genomes = 12;
    int threads = 0;
    int rl_train_episodes = 650;
    int rl_damage_finetune_episodes = 180;
    int rl_eval_every = 50;
    double rl_force_mag = 10.0;
    double solve_ratio = 0.80;
    double damage_fraction = 0.15;
    int damage_after_trial = 8;
    int replay_sleep_ms = 30;
    int replay_frame_stride = 1;
    int replay_damage_tick = -1;
    bool evolve_body_mode = false;
    bool evolve_development_steps = false;
    bool evolve_development_seed_half_width = false;
    bool evolve_max_cells = false;
    bool evolve_body_extent_x = false;
    bool evolve_body_extent_y = false;
    bool evolve_body_extent_z = false;
    bool evolve_chemical_diffusion_steps = false;
    bool evolve_growth_threshold = false;
    bool evolve_chemical_diffusion_rate = false;
    bool evolve_chemical_decay = false;
    int development_steps = 8;
    int development_seed_half_width = 2;
    int max_cells = 128;
    int body_extent_x = 7;
    int body_extent_y = 16;
    int body_extent_z = 2;
    int chemical_diffusion_steps = 2;
    bool replay_clear_screen = true;
    bool replay_emit_stdout = true;
    bool odd_enabled = true;
    double development_growth_threshold = 0.54;
    double chemical_diffusion_rate = 0.32;
    double chemical_decay = 0.08;
    double replay_theta_deg = 0.0 / 0.0;
    double replay_task_a = 0.0 / 0.0;
    double replay_task_b = 0.0 / 0.0;
    double worm_goal_distance = 6.0;
    double worm_max_backward = 1.5;
    int pong_target_hits = 6;
    double pong_ball_speed = 1.10;
    double pong_paddle_half_height = 0.22;
    TrialBudget search_budget;
    TrialBudget final_budget{100, 500};
    TrialBudget damage_budget{40, 500};
    TrialBudget odd_budget{24, 500};
};

struct EvaluationSummary {
    double total_ticks = 0.0;
    double max_ticks = 0.0;
    double survival_ratio = 0.0;
    double mean_ticks = 0.0;
    double success_rate = 0.0;
    double task_primary = 0.0;
    double task_secondary = 0.0;
    double sensor_responsiveness = 0.0;
    double signal_connectivity = 0.0;
    double force_correlation = 0.0;
    double adaptation_gain = 0.0;
    double mean_force_abs = 0.0;
    double mean_energy = 0.0;
    double late_mean_ticks = 0.0;
    double early_mean_ticks = 0.0;
    std::string task_primary_label;
    std::string task_secondary_label;
};

struct RLDamageSummary {
    EvaluationSummary clean;
    EvaluationSummary damaged;
    EvaluationSummary recovered;
};

struct ODDDeltaSummary {
    std::string key;
    std::string label;
    double baseline_survival_ratio = 0.0;
    double variant_survival_ratio = 0.0;
    double delta_survival_ratio = 0.0;
    double baseline_success_rate = 0.0;
    double variant_success_rate = 0.0;
    double delta_success_rate = 0.0;
};

struct ODDCorrelationSummary {
    std::string left;
    std::string right;
    double correlation = 0.0;
};

struct ODDSummary {
    bool available = false;
    TrialBudget budget;
    EvaluationSummary baseline_clean;
    std::vector<ODDDeltaSummary> mechanism_deltas;
    std::vector<ODDDeltaSummary> gene_channel_deltas;
    std::vector<ODDDeltaSummary> cell_force_deltas;
    std::vector<ODDDeltaSummary> edge_deltas;
    std::vector<ODDCorrelationSummary> temporal_correlations;
    std::vector<ODDCorrelationSummary> structural_confounds;
    std::string notes;
};

struct ComparisonSummary {
    int attempt = 0;
    int champion_generation = 0;
    int champion_cell_count = 0;
    int champion_body_depth = 0;
    double champion_search_score = 0.0;
    std::array<double, kGenomeSize> champion_genes{};
    CandidateParams champion_params;
    std::string task_name;
    std::string body_mode;
    EvaluationSummary cell_clean;
    EvaluationSummary cell_damaged;
    EvaluationSummary pd_gold;
    std::string rl_algorithm;
    RLDamageSummary rl;
    RLDamageSummary rl_dqn;
    RLDamageSummary rl_a2c;
    ODDSummary odd;
    bool solved = false;
    std::string notes;
};

ComparisonSummary run_full_benchmark(const CellEngineConfig& cfg);
std::string format_summary(const ComparisonSummary& summary);
bool write_report(const ComparisonSummary& summary,
                  const CellEngineConfig& cfg,
                  const std::string& output_dir);

} // namespace cellengine
} // namespace cells
