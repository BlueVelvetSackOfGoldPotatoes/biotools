
#ifndef CELLENGINE_INTERNAL_TYPES_DEFINED
constexpr double kPi = 3.14159265358979323846;
constexpr double kFailAngleRad = 15.0 * kPi / 180.0;
constexpr int kMaxNeighborDirs = 26;
#endif

double clampd(const double v, const double lo, const double hi) {
    return std::max(lo, std::min(hi, v));
}

double lerp(const double a, const double b, const double t) {
    return a + (b - a) * t;
}

double sigmoid(const double x) {
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(x);
    return z / (1.0 + z);
}

double sqr(const double x) {
    return x * x;
}

double gene01(const double g) {
    return clampd(0.5 * (g + 1.0), 0.0, 1.0);
}

double map_gene(const double g, const double lo, const double hi) {
    return lerp(lo, hi, gene01(g));
}

double mean_of(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double corr(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.size() < 2) return 0.0;
    const double ma = mean_of(a);
    const double mb = mean_of(b);
    double num = 0.0;
    double da = 0.0;
    double db = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double xa = a[i] - ma;
        const double xb = b[i] - mb;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if (da < 1e-12 || db < 1e-12) return 0.0;
    return num / std::sqrt(da * db);
}

std::string env_lower(const char* key, const std::string& fallback) {
    const char* v = std::getenv(key);
    std::string s = (v && *v) ? std::string(v) : fallback;
    std::transform(s.begin(), s.end(), s.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

#ifndef CELLENGINE_INTERNAL_TYPES_DEFINED
struct Genome {
    std::array<double, kGenomeSize> genes{};
    CandidateParams params{};
    bool has_params = false;
};
#endif

int body_mode_code(const std::string& body_mode) {
    if (body_mode == "fixed2d") return 0;
    if (body_mode == "grown2d") return 1;
    return 2;
}

std::string body_mode_from_code(const int code) {
    if (code <= 0) return "fixed2d";
    if (code == 1) return "grown2d";
    return "grown3d";
}

CandidateParams default_candidate_params(const CellEngineConfig& cfg) {
    CandidateParams params;
    params.body_mode = cfg.body_mode;
    params.development_steps = cfg.development_steps;
    params.development_seed_half_width = cfg.development_seed_half_width;
    params.max_cells = cfg.max_cells;
    params.body_extent_x = cfg.body_extent_x;
    params.body_extent_y = cfg.body_extent_y;
    params.body_extent_z = cfg.body_extent_z;
    params.chemical_diffusion_steps = cfg.chemical_diffusion_steps;
    params.development_growth_threshold = cfg.development_growth_threshold;
    params.chemical_diffusion_rate = cfg.chemical_diffusion_rate;
    params.chemical_decay = cfg.chemical_decay;
    return params;
}

CellEngineConfig enable_all_candidate_param_overrides(const CellEngineConfig& cfg) {
    CellEngineConfig out = cfg;
    out.evolve_body_mode = true;
    out.evolve_development_steps = true;
    out.evolve_development_seed_half_width = true;
    out.evolve_max_cells = true;
    out.evolve_body_extent_x = true;
    out.evolve_body_extent_y = true;
    out.evolve_body_extent_z = true;
    out.evolve_chemical_diffusion_steps = true;
    out.evolve_growth_threshold = true;
    out.evolve_chemical_diffusion_rate = true;
    out.evolve_chemical_decay = true;
    return out;
}

CellEngineConfig resolve_candidate_cfg(const CellEngineConfig& cfg, const Genome& genome) {
    CellEngineConfig resolved = cfg;
    const CandidateParams& params = genome.params;
    if (cfg.evolve_body_mode) {
        resolved.body_mode = body_mode_from_code(body_mode_code(params.body_mode));
    }
    if (cfg.evolve_development_steps) {
        resolved.development_steps = std::max(1, params.development_steps);
    }
    if (cfg.evolve_development_seed_half_width) {
        resolved.development_seed_half_width = std::max(1, params.development_seed_half_width);
    }
    if (cfg.evolve_max_cells) {
        resolved.max_cells = std::max(18, params.max_cells);
    }
    if (cfg.evolve_body_extent_x) {
        resolved.body_extent_x = std::max(2, params.body_extent_x);
    }
    if (cfg.evolve_body_extent_y) {
        resolved.body_extent_y = std::max(4, params.body_extent_y);
    }
    if (cfg.evolve_body_extent_z) {
        resolved.body_extent_z = std::max(0, params.body_extent_z);
    }
    if (cfg.evolve_chemical_diffusion_steps) {
        resolved.chemical_diffusion_steps = std::max(1, params.chemical_diffusion_steps);
    }
    if (cfg.evolve_growth_threshold) {
        resolved.development_growth_threshold = clampd(params.development_growth_threshold, 0.15, 0.90);
    }
    if (cfg.evolve_chemical_diffusion_rate) {
        resolved.chemical_diffusion_rate = clampd(params.chemical_diffusion_rate, 0.05, 0.60);
    }
    if (cfg.evolve_chemical_decay) {
        resolved.chemical_decay = clampd(params.chemical_decay, 0.0, 0.25);
    }

    resolved.max_cells = std::max(18, resolved.max_cells);
    resolved.development_steps = std::max(1, resolved.development_steps);
    resolved.development_seed_half_width = std::max(1, resolved.development_seed_half_width);
    resolved.body_extent_x = std::max(2, resolved.body_extent_x);
    resolved.body_extent_y = std::max(4, resolved.body_extent_y);
    resolved.body_extent_z = std::max(0, resolved.body_extent_z);
    resolved.chemical_diffusion_steps = std::max(1, resolved.chemical_diffusion_steps);
    resolved.development_growth_threshold = clampd(resolved.development_growth_threshold, 0.15, 0.90);
    resolved.chemical_diffusion_rate = clampd(resolved.chemical_diffusion_rate, 0.05, 0.60);
    resolved.chemical_decay = clampd(resolved.chemical_decay, 0.0, 0.25);

    if (resolved.body_mode == "fixed2d") {
        resolved.body_extent_z = 0;
    }
    if (resolved.body_mode == "grown2d") {
        resolved.body_extent_z = 0;
    }
    if (resolved.body_mode == "grown3d") {
        resolved.body_extent_z = std::max(1, resolved.body_extent_z);
    }
    return resolved;
}

CellEngineConfig resolve_loaded_genome_cfg(const CellEngineConfig& cfg, const Genome& genome) {
    if (!genome.has_params) return resolve_candidate_cfg(cfg, genome);
    return resolve_candidate_cfg(enable_all_candidate_param_overrides(cfg), genome);
}

#ifndef CELLENGINE_INTERNAL_TYPES_DEFINED
enum class TaskKind {
    cartpole_balance,
    mass_spring_balance,
    worm_drag_race,
    pong_return,
};
#endif

TaskKind parse_task_kind(const std::string& task_name) {
    if (task_name == "worm_drag_race") return TaskKind::worm_drag_race;
    if (task_name == "pong_return") return TaskKind::pong_return;
    if (task_name == "mass_spring_balance") return TaskKind::mass_spring_balance;
    return TaskKind::cartpole_balance;
}

std::string task_primary_label(const TaskKind kind) {
    switch (kind) {
        case TaskKind::worm_drag_race: return "goal_progress";
        case TaskKind::pong_return: return "rally_completion";
        case TaskKind::mass_spring_balance: return "spring_stability";
        case TaskKind::cartpole_balance:
        default: return "balance_ratio";
    }
}

std::string task_secondary_label(const TaskKind kind) {
    switch (kind) {
        case TaskKind::worm_drag_race: return "forward_velocity";
        case TaskKind::pong_return: return "successful_returns";
        case TaskKind::mass_spring_balance: return "mean_abs_displacement";
        case TaskKind::cartpole_balance:
        default: return "mean_abs_angle";
    }
}

#ifndef CELLENGINE_INTERNAL_TYPES_DEFINED
struct TaskTelemetry {
    double hinge_signal = 0.0;
    double lateral_signal = 0.0;
    double support_load = 0.0;
    double drift_signal = 0.0;
};

struct BodyCell {
    int x = 0;
    int y = 0;
    int z = 0;
    int birth_step = 0;
    bool hinge = false;
    bool ground = false;
    bool motor = false;
    std::array<double, 5> features{};
    double mech_advantage = 0.0;
    double force_direction_x = 0.0;
};

struct BodyTemplate {
    std::vector<BodyCell> cells;
    std::vector<int> neighbor_offsets;
    std::vector<int> neighbor_flat;
    std::vector<int> cell_to_grid;
    std::vector<int> grid_to_cell;
    std::vector<int> diffusion_offsets;
    std::vector<int> diffusion_flat;
    std::vector<int> sensor_cells;
    std::vector<int> motor_cells;
    int grid_min_x = 0;
    int grid_max_x = 0;
    int grid_min_y = 0;
    int grid_max_y = 0;
    int grid_min_z = 0;
    int grid_max_z = 0;
    int grid_sx = 0;
    int grid_sy = 0;
    int grid_sz = 0;

    int cell_count() const { return static_cast<int>(cells.size()); }
    int body_depth() const {
        if (cells.empty()) return 0;
        int min_z = cells.front().z;
        int max_z = cells.front().z;
        for (const auto& cell : cells) {
            min_z = std::min(min_z, cell.z);
            max_z = std::max(max_z, cell.z);
        }
        return std::max(1, max_z - min_z + 1);
    }
    int grid_cell_count() const { return static_cast<int>(grid_to_cell.size()); }
};

struct DecodedGenome {
    std::array<std::array<double, 5>, kGeneExprDim> W{};
    std::array<double, kGeneExprDim> bias{};
    double tau_v_base = 1.0;
    double tau_v_gain = 0.0;
    double epsilon_base = 0.08;
    double epsilon_gain = 0.0;
    double fhn_a = 0.7;
    double fhn_b = 0.8;
    double tau_ca = 0.8;
    double alpha_ca = 0.08;
    double v_ca_thresh = 0.12;
    double coupling_base = 0.24;
    double hebb_rate = 0.08;
    double decay_rate = 0.03;
    double adhesion_coeff = 0.16;
    double chem_coeff = 0.12;
    double mech_base = 1.4;
    double sigma_target = 0.38;
    double sigma_alpha = 0.025;
    double sigma_beta = 0.08;
    double energy_replenish = 0.05;
    double cost_vm = 0.01;
    double cost_ca = 0.01;
    double cost_mech = 0.014;
    double g_drift_rate = 0.002;
    double force_scale = 12.0;
    double stress_scale = 1.3;
    double teacher_frac = 0.18;
    double teacher_force_scale = 1.0;
    double teacher_hebb_boost = 1.4;
    double stiffness_softness = 0.85;
    double basal_contractility = 0.05;
    double activity_decay = 0.92;
    double sensor_threshold = 0.06;
    double teacher_kp = 72.0;
    double teacher_kd = 14.0;
    double chemical_balance = 0.0;
    double muscle_bias = 0.04;
    double mech_filter = 0.45;
    double energy_output_scale = 1.0;
    double recovery_reset = 0.06;
    double trial_fast_decay = 0.45;
    double damage_resilience = 0.9;
    double force_smoothing = 0.25;
    double stress_smoothing = 0.25;
};

struct CellPopulation {
    int count = 0;
    std::vector<int> active;
    std::vector<double> V;
    std::vector<double> Wrec;
    std::vector<double> Ca;
    std::vector<double> sigma;
    std::vector<double> energy;
    std::vector<double> a_mech;
    std::vector<double> chem_out;
    std::vector<double> stress;
    std::vector<double> activity_avg;
    std::vector<double> fatigue;
    std::vector<double> G;
    std::vector<double> gap;
    std::vector<double> gap_base;
    std::vector<double> chem_field;
    std::vector<double> stress_hinge;
    std::vector<double> stress_lateral;
    std::vector<double> stress_gravity;
    std::vector<double> stress_drift;
    std::vector<double> stress_neighbor;
    std::vector<double> mech_a_gate;
    std::vector<double> mech_contract;
    std::vector<double> mech_prefactor;
    std::vector<double> mech_force_contrib;
    std::vector<double> gap_corr_last;
    std::vector<double> gap_hebb_delta_last;
    std::vector<double> gap_decay_delta_last;
    std::vector<double> gap_old_last;
    std::vector<double> gap_new_last;
};
#endif
