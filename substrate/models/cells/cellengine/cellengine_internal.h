#pragma once

#include "models/cells/cellengine/cellengine.h"

#include "models/reinforcement/src/rl_models.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace cells {
namespace cellengine {
namespace detail {

inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kFailAngleRad = 15.0 * kPi / 180.0;
inline constexpr int kMaxNeighborDirs = 26;

struct Genome {
    std::array<double, kGenomeSize> genes{};
    CandidateParams params{};
    bool has_params = false;
};

enum class TaskKind {
    cartpole_balance,
    mass_spring_balance,
    worm_drag_race,
    pong_return,
};

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

struct ODDIntervention {
    std::string key;
    std::string label;
    bool disable_electrical = false;
    bool disable_chemical = false;
    bool disable_diffusion = false;
    bool disable_self_chemical = false;
    bool disable_mechanical = false;
    bool freeze_sigma = false;
    bool freeze_gap_plasticity = false;
    bool disable_slow_drift = false;
    bool disable_teacher = false;
    int neutralize_gene_channel = -1;
    int freeze_gene_channel = -1;
    int perturb_gene_channel = -1;
    double perturb_gene_delta = 0.0;
    int ablate_force_cell = -1;
    int ablate_edge_src = -1;
    int ablate_edge_dst = -1;
};

struct TrialStats {
    int ticks = 0;
    double mean_abs_force = 0.0;
    double mean_energy = 0.0;
    double mean_abs_theta = 0.0;
    double task_primary = 0.0;
    double task_secondary = 0.0;
    bool success = false;
};

struct ReplayFrame {
    int tick = 0;
    double x = 0.0;
    double theta = 0.0;
    double x_dot = 0.0;
    double theta_dot = 0.0;
    double x_ddot = 0.0;
    double theta_ddot = 0.0;
    double organism_force = 0.0;
    double teacher_force = 0.0;
    double total_force = 0.0;
    double mean_energy = 0.0;
    double mean_stress = 0.0;
    double mean_activity = 0.0;
    double mean_calcium = 0.0;
    double active_fraction = 0.0;
    double task_aux_a = 0.0;
    double task_aux_b = 0.0;
    double task_primary = 0.0;
    double task_secondary = 0.0;
    int task_counter = 0;
    bool task_success = false;
    bool damage_event = false;
    bool terminal = false;
};

struct ReplayCellState {
    int tick = 0;
    int cell_id = 0;
    int active = 0;
    double V = 0.0;
    double Wrec = 0.0;
    double Ca = 0.0;
    double sigma = 0.0;
    double energy = 0.0;
    double a_mech = 0.0;
    double chem_out = 0.0;
    double stress = 0.0;
    double activity = 0.0;
    double I_elec = 0.0;
    double I_chem = 0.0;
    double I_mech = 0.0;
    double I_self_chem = 0.0;
    double I_total = 0.0;
    double stress_hinge = 0.0;
    double stress_lateral = 0.0;
    double stress_gravity = 0.0;
    double stress_drift = 0.0;
    double stress_neighbor = 0.0;
    double stress_final = 0.0;
    double mech_a_gate = 0.0;
    double mech_contract = 0.0;
    double mech_prefactor = 0.0;
    double mech_force_contrib = 0.0;
};

struct ReplayEdgeState {
    int tick = 0;
    int src_cell_id = 0;
    int dst_cell_id = 0;
    int src_active = 0;
    int dst_active = 0;
    double gap_forward = 0.0;
    double gap_reverse = 0.0;
    double gap_mean = 0.0;
    double corr_forward = 0.0;
    double corr_reverse = 0.0;
    double hebb_delta_forward = 0.0;
    double hebb_delta_reverse = 0.0;
    double decay_delta_forward = 0.0;
    double decay_delta_reverse = 0.0;
    double old_gap_forward = 0.0;
    double old_gap_reverse = 0.0;
    double new_gap_forward = 0.0;
    double new_gap_reverse = 0.0;
};

struct RLReplayResult {
    std::string algorithm;
    EvaluationSummary summary;
    std::vector<ReplayFrame> frames;
    std::vector<int> actions;
};

struct CellEvalResult {
    EvaluationSummary summary;
    double score = 0.0;
};

struct EvolutionResult {
    Genome best_genome;
    double best_search_score = -1.0;
    int best_generation = 0;
};

struct TaskState {
    double x = 0.0;
    double x_dot = 0.0;
    double theta = 0.0;
    double theta_dot = 0.0;
};

double clampd(double v, double lo, double hi);
double sigmoid(double x);

struct TaskEnv {
    TaskKind kind = TaskKind::cartpole_balance;
    TaskState s;
    double x_ddot = 0.0;
    double theta_ddot = 0.0;
    double organism_force = 0.0;
    double teacher_force = 0.0;
    double last_reward = 0.0;
    double task_aux_a = 0.0;
    double task_aux_b = 0.0;
    double task_goal = 1.0;
    double task_limit = 1.0;
    int task_target_count = 1;
    int task_counter = 0;
    int ticks = 0;
    bool terminal = false;
    bool success = false;
    bool teacher_active = false;
    double teacher_kp = 72.0;
    double teacher_kd = 14.0;
    double teacher_scale = 1.0;
    double force_limit = 12.0;

    static std::array<double, 4> cartpole_derivatives(const TaskState& st, const double force) {
        constexpr double M = 1.0;
        constexpr double m = 0.1;
        constexpr double l = 0.5;
        constexpr double g = 9.81;

        const double sin_t = std::sin(st.theta);
        const double cos_t = std::cos(st.theta);
        const double denom = l * (4.0 / 3.0 - (m * cos_t * cos_t) / (M + m));
        const double theta_dd =
            (g * sin_t - cos_t * (force + m * l * st.theta_dot * st.theta_dot * sin_t) / (M + m)) /
            std::max(1e-9, denom);
        const double x_dd =
            (force + m * l * (st.theta_dot * st.theta_dot * sin_t - theta_dd * cos_t)) /
            (M + m);
        return {st.x_dot, x_dd, st.theta_dot, theta_dd};
    }

    static std::array<double, 4> mass_spring_derivatives(const TaskState& st, const double force) {
        constexpr double mass = 1.0;
        constexpr double damping = 0.55;
        constexpr double spring = 2.0;
        constexpr double target_scale = 2.4 / kFailAngleRad;
        const double x_dd = (force / mass) - damping * st.x_dot - spring * st.x;
        const double theta = st.x / target_scale;
        const double theta_dot = st.x_dot / target_scale;
        const double theta_dd = x_dd / target_scale;
        (void)theta;
        return {st.x_dot, x_dd, theta_dot, theta_dd};
    }

    std::array<double, 4> derivatives(const TaskState& st, const double force) const {
        if (kind == TaskKind::worm_drag_race) {
            const double anchor = sigmoid(4.2 * (st.theta - 0.05));
            const double x_dd =
                1.35 * anchor * std::max(0.0, st.theta_dot) -
                0.42 * st.x_dot -
                0.10 * st.x_dot * std::abs(st.x_dot) -
                0.16 * std::max(0.0, -st.theta) * (1.0 - anchor);
            const double theta_dd =
                1.55 * force -
                0.95 * st.theta_dot -
                2.35 * st.theta -
                0.10 * st.x_dot;
            return {st.x_dot, x_dd, st.theta_dot, theta_dd};
        }
        return (kind == TaskKind::mass_spring_balance)
                   ? mass_spring_derivatives(st, force)
                   : cartpole_derivatives(st, force);
    }

    double teacher_control_force() const {
        if (kind == TaskKind::worm_drag_race) {
            const double phase = 0.22 * static_cast<double>(ticks);
            const double gait = std::sin(phase);
            return clampd(
                2.1 * (task_goal - s.x) - 1.0 * s.x_dot - 2.0 * s.theta - 0.8 * s.theta_dot + 4.0 * gait,
                -force_limit,
                force_limit);
        }
        if (kind == TaskKind::pong_return) {
            const double anticipation =
                s.theta + 0.30 * s.theta_dot + 0.18 * task_aux_b * std::max(0.0, 1.2 - task_aux_a);
            return clampd((teacher_kp * (anticipation - s.x) - teacher_kd * s.x_dot) * teacher_scale,
                          -force_limit,
                          force_limit);
        }
        if (kind == TaskKind::mass_spring_balance) {
            return clampd(-(teacher_kp * s.x + teacher_kd * s.x_dot) * teacher_scale, -force_limit, force_limit);
        }
        return clampd(-(teacher_kp * s.theta + teacher_kd * s.theta_dot) * teacher_scale, -force_limit, force_limit);
    }

    TaskTelemetry telemetry() const {
        if (kind == TaskKind::worm_drag_race) {
            return {
                0.70 * s.theta + 0.25 * s.theta_dot,
                x_ddot,
                1.0 + std::max(0.0, s.theta),
                s.x_dot
            };
        }
        if (kind == TaskKind::pong_return) {
            return {
                s.theta - s.x,
                task_aux_b,
                std::max(0.25, 1.2 - task_aux_a),
                s.x_dot
            };
        }
        if (kind == TaskKind::mass_spring_balance) {
            return {
                0.80 * s.x + 0.25 * s.x_dot,
                x_ddot,
                9.81,
                s.x_dot
            };
        }
        constexpr double M = 1.0;
        constexpr double m = 0.1;
        constexpr double l = 0.5;
        constexpr double g = 9.81;
        const double hinge_moment = m * g * l * std::sin(s.theta) + m * l * l * theta_ddot;
        const double lateral_force = m * l * (theta_ddot * std::cos(s.theta) - s.theta_dot * s.theta_dot * std::sin(s.theta));
        return {hinge_moment, lateral_force, (M + m) * g, s.x_dot};
    }

    void reset(std::mt19937& rng, const std::optional<double> theta_override = std::nullopt) {
        std::uniform_real_distribution<double> small_angle(-3.0 * kPi / 180.0, 3.0 * kPi / 180.0);
        std::uniform_real_distribution<double> small_x(-0.25, 0.25);
        std::uniform_real_distribution<double> worm_bend(-0.18, 0.18);
        std::uniform_real_distribution<double> pong_y(-0.55, 0.55);
        std::uniform_real_distribution<double> pong_vy(-0.70, 0.70);
        s = {};
        if (kind == TaskKind::pong_return) {
            s.x = 0.0;
            s.x_dot = 0.0;
            s.theta = theta_override.has_value() ? *theta_override : pong_y(rng);
            s.theta_dot = task_aux_b;
            task_aux_a = -1.15;
            if (!std::isfinite(task_aux_b) || std::abs(task_aux_b) < 0.12) {
                task_aux_b = pong_vy(rng);
            }
            task_aux_b = clampd(task_aux_b, -1.10, 1.10);
        } else if (kind == TaskKind::worm_drag_race) {
            s.x = 0.0;
            s.x_dot = 0.0;
            s.theta = theta_override.has_value() ? *theta_override : worm_bend(rng);
            s.theta_dot = 0.0;
        } else if (kind == TaskKind::mass_spring_balance) {
            s.x = theta_override.has_value() ? (*theta_override * (2.4 / kFailAngleRad)) : small_x(rng);
            s.theta = s.x * (kFailAngleRad / 2.4);
        } else {
            s.theta = theta_override.has_value() ? *theta_override : small_angle(rng);
        }
        x_ddot = 0.0;
        theta_ddot = 0.0;
        organism_force = 0.0;
        teacher_force = 0.0;
        last_reward = 0.0;
        task_counter = 0;
        ticks = 0;
        terminal = false;
        success = false;
    }

    void set_teacher(const bool active, const double kp, const double kd, const double scale) {
        teacher_active = active;
        teacher_kp = kp;
        teacher_kd = kd;
        teacher_scale = scale;
    }

    void step(const double commanded_force, const double dt) {
        organism_force = clampd(commanded_force, -force_limit, force_limit);
        teacher_force = teacher_active ? teacher_control_force() : 0.0;
        const double total_force = clampd(organism_force + teacher_force, -force_limit, force_limit);

        if (kind == TaskKind::pong_return) {
            const double paddle_dd = 4.8 * total_force - 2.4 * s.x_dot;
            s.x += dt * s.x_dot;
            s.x_dot += dt * paddle_dd;
            s.x = clampd(s.x, -1.05, 1.05);
            s.theta += dt * s.theta_dot;
            task_aux_a += dt * task_limit;
            if (s.theta >= 1.0) {
                s.theta = 1.0;
                s.theta_dot = -std::abs(s.theta_dot);
            } else if (s.theta <= -1.0) {
                s.theta = -1.0;
                s.theta_dot = std::abs(s.theta_dot);
            }
            if (task_aux_a <= -1.15) {
                task_aux_a = -1.15;
                task_limit = std::abs(task_limit);
            }
            if (task_aux_a >= 1.15) {
                const double miss = std::abs(s.theta - s.x);
                if (miss <= task_goal) {
                    task_counter += 1;
                    task_aux_a = 1.15;
                    task_limit = -std::abs(task_limit);
                    s.theta_dot = clampd(
                        s.theta_dot + 0.65 * clampd((s.theta - s.x) / std::max(0.05, task_goal), -1.0, 1.0),
                        -1.25,
                        1.25);
                } else {
                    terminal = true;
                }
            }
            x_ddot = paddle_dd;
            theta_ddot = 0.0;
            ticks += 1;
            success = (task_counter >= std::max(1, task_target_count));
            last_reward =
                terminal ? -1.0 : (0.18 * (1.0 - std::min(1.0, std::abs(s.theta - s.x))) + (success ? 1.5 : 0.02 * task_counter));
            return;
        }

        const auto k1 = derivatives(s, total_force);
        TaskState s2{s.x + 0.5 * dt * k1[0], s.x_dot + 0.5 * dt * k1[1],
                     s.theta + 0.5 * dt * k1[2], s.theta_dot + 0.5 * dt * k1[3]};
        const auto k2 = derivatives(s2, total_force);
        TaskState s3{s.x + 0.5 * dt * k2[0], s.x_dot + 0.5 * dt * k2[1],
                     s.theta + 0.5 * dt * k2[2], s.theta_dot + 0.5 * dt * k2[3]};
        const auto k3 = derivatives(s3, total_force);
        TaskState s4{s.x + dt * k3[0], s.x_dot + dt * k3[1],
                     s.theta + dt * k3[2], s.theta_dot + dt * k3[3]};
        const auto k4 = derivatives(s4, total_force);

        s.x += dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0;
        s.x_dot += dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0;
        s.theta += dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0;
        s.theta_dot += dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0;

        const auto deriv_now = derivatives(s, total_force);
        x_ddot = deriv_now[1];
        theta_ddot = deriv_now[3];
        if (kind == TaskKind::mass_spring_balance) {
            s.theta = clampd(s.x * (kFailAngleRad / 2.4), -2.5 * kFailAngleRad, 2.5 * kFailAngleRad);
            s.theta_dot = clampd(s.x_dot * (kFailAngleRad / 2.4), -6.0, 6.0);
        }
        ticks += 1;
        if (kind == TaskKind::worm_drag_race) {
            terminal = (!std::isfinite(s.x) || !std::isfinite(s.theta) || std::abs(s.theta) > 1.65 || s.x < -std::max(0.5, task_limit));
            success = s.x >= task_goal;
            last_reward = terminal ? -1.0 : (0.22 * clampd(s.x / std::max(0.5, task_goal), -1.0, 2.0) + 0.14 * clampd(s.x_dot, -1.0, 1.0));
        } else if (kind == TaskKind::mass_spring_balance) {
            terminal = (!std::isfinite(s.x) || !std::isfinite(s.x_dot) || std::abs(s.x) > 2.4);
            success = (!terminal && ticks > 0);
            last_reward = terminal ? -1.0 : 1.0;
        } else {
            terminal = (!std::isfinite(s.theta) || !std::isfinite(s.theta_dot) || std::abs(s.theta) > kFailAngleRad);
            success = (!terminal && ticks > 0);
            last_reward = terminal ? -1.0 : 1.0;
        }
    }

    double primary_metric(const TrialBudget& budget) const {
        switch (kind) {
            case TaskKind::worm_drag_race:
                return clampd(s.x / std::max(0.5, task_goal), -1.0, 2.0);
            case TaskKind::pong_return:
                return clampd(static_cast<double>(task_counter) / static_cast<double>(std::max(1, task_target_count)), 0.0, 2.0);
            case TaskKind::mass_spring_balance:
            case TaskKind::cartpole_balance:
            default:
                return clampd(static_cast<double>(ticks) / std::max(1, budget.max_ticks), 0.0, 1.0);
        }
    }

    double secondary_metric() const {
        switch (kind) {
            case TaskKind::worm_drag_race:
                return s.x_dot;
            case TaskKind::pong_return:
                return static_cast<double>(task_counter);
            case TaskKind::mass_spring_balance:
                return std::abs(s.x);
            case TaskKind::cartpole_balance:
            default:
                return std::abs(s.theta);
        }
    }

    bool trial_success(const TrialBudget& budget) const {
        switch (kind) {
            case TaskKind::worm_drag_race:
                return !terminal && s.x >= task_goal;
            case TaskKind::pong_return:
                return !terminal && task_counter >= std::max(1, task_target_count);
            case TaskKind::mass_spring_balance:
            case TaskKind::cartpole_balance:
            default:
                return !terminal && ticks >= budget.max_ticks;
        }
    }
};

struct ActorCriticAgent {
    Sequential policy;
    Sequential value;
    Adam policy_opt;
    Adam value_opt;
    double gamma = 0.99;

    explicit ActorCriticAgent(std::mt19937& rng)
        : policy_opt(0.0008), value_opt(0.0012) {
        policy.add(std::make_unique<Linear>(4, 64, "ac_pi_fc1", rng));
        policy.add(std::make_unique<ReLU>());
        policy.add(std::make_unique<Linear>(64, 64, "ac_pi_fc2", rng));
        policy.add(std::make_unique<ReLU>());
        policy.add(std::make_unique<Linear>(64, 2, "ac_pi_out", rng));

        value.add(std::make_unique<Linear>(4, 64, "ac_v_fc1", rng));
        value.add(std::make_unique<ReLU>());
        value.add(std::make_unique<Linear>(64, 64, "ac_v_fc2", rng));
        value.add(std::make_unique<ReLU>());
        value.add(std::make_unique<Linear>(64, 1, "ac_v_out", rng));
    }
};

using Coord3 = std::tuple<int, int, int>;

double lerp(double a, double b, double t);
double sqr(double x);
double gene01(double g);
double map_gene(double g, double lo, double hi);
double mean_of(const std::vector<double>& v);
double corr(const std::vector<double>& a, const std::vector<double>& b);
std::string env_lower(const char* key, const std::string& fallback);
int body_mode_code(const std::string& body_mode);
std::string body_mode_from_code(int code);
CandidateParams default_candidate_params(const CellEngineConfig& cfg);
CellEngineConfig enable_all_candidate_param_overrides(const CellEngineConfig& cfg);
CellEngineConfig resolve_candidate_cfg(const CellEngineConfig& cfg, const Genome& genome);
CellEngineConfig resolve_loaded_genome_cfg(const CellEngineConfig& cfg, const Genome& genome);
TaskKind parse_task_kind(const std::string& task_name);
std::string task_primary_label(TaskKind kind);
std::string task_secondary_label(TaskKind kind);
std::string gene_channel_name(int idx);
bool edge_ablation_matches(const ODDIntervention* intervention, int a, int b);
void configure_task_env(TaskEnv& env, const CellEngineConfig& cfg, double force_limit);
Tensor make_state_tensor(const TaskEnv& env);
int grid_linear_index(const BodyTemplate& body, int x, int y, int z);
std::array<double, kGeneExprDim> provisional_gene_expr(const Genome& genome, const std::array<double, 5>& features);
void finalize_body_template(BodyTemplate& body);
BodyTemplate make_fixed_body_template();
BodyTemplate make_developed_body_template(const Genome& genome, const CellEngineConfig& cfg, bool allow_depth);
BodyTemplate make_body_template(const Genome& genome, const CellEngineConfig& cfg);
DecodedGenome decode_genome(const Genome& genome);
Genome random_genome(const CellEngineConfig& cfg, std::mt19937& rng);
bool load_genome_csv(const std::string& path, Genome& genome);
void set_gene(Genome& g, int idx, double value);
Genome make_hint_genome(const CellEngineConfig& cfg, int variant);
CellPopulation build_population(const Genome& genome, const BodyTemplate& body, const ODDIntervention* intervention = nullptr);
void reset_fast_state(CellPopulation& pop, const DecodedGenome& d);
void apply_damage(CellPopulation& pop, const BodyTemplate& body, double frac, std::mt19937& rng);
void write_stress(CellPopulation& pop, const BodyTemplate& body, const DecodedGenome& d, const TaskEnv& env, int relax_iters);
bool try_cuda_cell_tick(CellPopulation& pop,
                        const BodyTemplate& body,
                        const DecodedGenome& d,
                        const CellEngineConfig& cfg,
                        double dt,
                        bool teaching_active,
                        const ODDIntervention* intervention);
void record_last_tick_decomposition(CellPopulation& pop,
                                    const BodyTemplate& body,
                                    const DecodedGenome& d,
                                    const std::vector<double>& pre_V,
                                    const std::vector<double>& pre_gap,
                                    double dt,
                                    bool teaching_active,
                                    const ODDIntervention* intervention);
void cell_tick(CellPopulation& pop,
               const BodyTemplate& body,
               const DecodedGenome& d,
               const CellEngineConfig& cfg,
               double dt,
               bool teaching_active,
               const ODDIntervention* intervention = nullptr,
               bool capture_tick_decomposition = false);
void slow_update(CellPopulation& pop, const BodyTemplate& body, const DecodedGenome& d, const ODDIntervention* intervention = nullptr);
double compute_force(const CellPopulation& pop, const BodyTemplate& body, const ODDIntervention* intervention = nullptr);
ReplayFrame make_replay_frame(const CellPopulation& pop, const TaskEnv& env, bool damage_event = false);
std::string ascii_frame(const ReplayFrame& frame, int max_ticks);
bool write_replay_trace_csv(const std::vector<ReplayFrame>& frames, const std::string& path);
bool write_replay_body_csv(const BodyTemplate& body, const CellPopulation& pop, const std::string& path);
void write_candidate_params_json(std::ostream& json, const CellEngineConfig& cfg);
void write_eval_summary_json(std::ostream& json, const EvaluationSummary& s);
void write_body_cells_json(std::ostream& json, const BodyTemplate& body, const CellPopulation& pop);
bool write_live_discovery_snapshot(const Genome& genome,
                                   const CellEngineConfig& cfg,
                                   int current_generation,
                                   int best_generation,
                                   double best_search_score,
                                   const EvaluationSummary* best_summary,
                                   const std::string& output_dir);
bool write_live_discovery_population_snapshot(const std::vector<Genome>& population,
                                              const std::vector<double>& fitness,
                                              const std::vector<EvaluationSummary>& summaries,
                                              const std::vector<int>& order,
                                              const CellEngineConfig& cfg,
                                              int current_generation,
                                              int best_generation,
                                              double best_search_score,
                                              const std::string& output_dir);
bool write_replay_cells_csv(const std::vector<ReplayCellState>& rows, const std::string& path);
bool write_replay_edges_csv(const std::vector<ReplayEdgeState>& rows, const std::string& path);
bool write_rl_replay_trace_csv(const std::vector<ReplayFrame>& frames,
                               const std::vector<int>& actions,
                               const std::string& algorithm,
                               const std::string& path);
void append_replay_cell_state(const CellPopulation& pop,
                              const BodyTemplate& body,
                              const DecodedGenome& d,
                              int tick,
                              const ODDIntervention* intervention,
                              std::vector<ReplayCellState>& rows);
void append_replay_edge_state(const CellPopulation& pop,
                              const BodyTemplate& body,
                              int tick,
                              std::vector<ReplayEdgeState>& rows);
std::tuple<TrialStats, std::vector<ReplayFrame>, std::vector<ReplayCellState>, std::vector<ReplayEdgeState>>
run_trial_with_trace(CellPopulation& pop,
                     const BodyTemplate& body,
                     const DecodedGenome& d,
                     const CellEngineConfig& cfg,
                     const TrialBudget& budget,
                     std::mt19937& rng,
                     bool teaching_active,
                     TaskEnv& env,
                     const ODDIntervention* intervention = nullptr);
TrialStats run_trial(CellPopulation& pop,
                     const BodyTemplate& body,
                     const DecodedGenome& d,
                     const CellEngineConfig& cfg,
                     const TrialBudget& budget,
                     std::mt19937& rng,
                     bool teaching_active,
                     TaskEnv& env,
                     const ODDIntervention* intervention = nullptr);
EvaluationSummary summarize_trials(const std::vector<TrialStats>& trials, const TrialBudget& budget);
void annotate_task_summary(EvaluationSummary& summary, TaskKind kind);
double task_summary_score(const EvaluationSummary& s, TaskKind kind);
double task_refinement_score(const EvaluationSummary& s, TaskKind kind);
bool task_is_solved(const EvaluationSummary& s, TaskKind kind, const CellEngineConfig& cfg);
std::tuple<double, double, double> run_probe(const Genome& genome, const CellEngineConfig& cfg, std::mt19937& rng);
CellEvalResult evaluate_genome(const Genome& genome,
                               const CellEngineConfig& cfg,
                               const TrialBudget& budget,
                               int eval_seed,
                               bool with_damage,
                               bool include_probe,
                               const ODDIntervention* intervention = nullptr);
void mutate(Genome& genome, const CellEngineConfig& cfg, double rate, double sigma, std::mt19937& rng);
Genome crossover(const Genome& a, const Genome& b, const CellEngineConfig& cfg, double rate, std::mt19937& rng);
int tournament_pick(const std::vector<double>& fitness, int tournament_size, std::mt19937& rng);
Genome local_refine(const Genome& start, const CellEngineConfig& cfg, int seed, double& best_score);
Genome long_horizon_refine(const Genome& start, const CellEngineConfig& cfg, int seed);
EvolutionResult evolve_cells(const CellEngineConfig& cfg, int attempt_seed);
EvaluationSummary evaluate_pd(const CellEngineConfig& cfg, const TrialBudget& budget, int seed);
EvaluationSummary evaluate_dqn_agent(DQNAgent& agent, const CellEngineConfig& cfg, const TrialBudget& budget, int seed);
std::array<double, 2> softmax2(const Tensor& logits, std::size_t row);
int sample_action(const std::array<double, 2>& probs, std::mt19937& rng);
EvaluationSummary summarize_single_trial(const TrialStats& stats, const TrialBudget& budget);
ODDDeltaSummary make_odd_delta(const std::string& key,
                               const std::string& label,
                               const EvaluationSummary& baseline,
                               const EvaluationSummary& variant);
std::vector<ODDCorrelationSummary> top_abs_correlations(std::vector<ODDCorrelationSummary> items, std::size_t limit);
ODDSummary run_odd_experiment(const Genome& genome, const CellEngineConfig& cfg, int seed);
int greedy_a2c_action(ActorCriticAgent& agent, const TaskEnv& env);
int greedy_dqn_action(DQNAgent& agent, const TaskEnv& env);
void train_a2c_agent(ActorCriticAgent& agent, const CellEngineConfig& cfg, int seed);
void finetune_a2c_agent(ActorCriticAgent& agent, const CellEngineConfig& cfg, int seed);
EvaluationSummary evaluate_a2c_agent(ActorCriticAgent& agent, const CellEngineConfig& cfg, const TrialBudget& budget, int seed);
void train_dqn_agent(DQNAgent& agent, const CellEngineConfig& cfg, int seed);
void finetune_dqn_agent(DQNAgent& agent, const CellEngineConfig& cfg, int seed);
RLDamageSummary train_and_evaluate_a2c(const CellEngineConfig& cfg, int seed);
void damage_agent_parameters(std::vector<Tensor*> params, double frac, std::mt19937& rng);
RLDamageSummary train_and_evaluate_dqn(const CellEngineConfig& cfg, int seed);
std::pair<std::string, RLDamageSummary>
train_and_select_rl(const CellEngineConfig& cfg, int seed, RLDamageSummary& dqn_out, RLDamageSummary& a2c_out);
double resolve_replay_theta_deg(const CellEngineConfig& cfg, std::mt19937& rng);
TrialStats replay_a2c_episode(ActorCriticAgent& agent,
                              const CellEngineConfig& cfg,
                              const TrialBudget& budget,
                              double theta_deg,
                              int seed,
                              std::vector<ReplayFrame>& frames,
                              std::vector<int>& actions);
TrialStats replay_dqn_episode(DQNAgent& agent,
                              const CellEngineConfig& cfg,
                              const TrialBudget& budget,
                              double theta_deg,
                              int seed,
                              std::vector<ReplayFrame>& frames,
                              std::vector<int>& actions);
RLReplayResult build_rl_replay(const CellEngineConfig& cfg, const TrialBudget& budget, double theta_deg, int seed);
ComparisonSummary preview_loaded_genome(const Genome& genome, const CellEngineConfig& cfg);
ComparisonSummary replay_loaded_genome(const Genome& genome, const CellEngineConfig& cfg);
std::string summary_line(const std::string& label, const EvaluationSummary& s);
double odd_relative_importance(const std::vector<ODDDeltaSummary>& items, std::size_t index);
std::string json_escape(const std::string& input);
bool write_decoded_genome_json(const Genome& genome, const CandidateParams& params, const std::string& path);
bool write_trial_context_artifacts(const std::vector<ReplayCellState>& pre_reset_rows,
                                   const std::vector<ReplayCellState>& post_reset_rows,
                                   bool teacher_active,
                                   bool damage_active,
                                   const std::string& output_dir,
                                   const std::string& file_prefix);
bool write_named_trial_context_bundle(const Genome& genome,
                                      const CellEngineConfig& cfg,
                                      bool teacher_active,
                                      bool damage_active,
                                      int seed_offset,
                                      const std::string& output_dir,
                                      const std::string& file_prefix);

} // namespace detail
} // namespace cellengine
} // namespace cells

#define CELLENGINE_INTERNAL_TYPES_DEFINED 1
