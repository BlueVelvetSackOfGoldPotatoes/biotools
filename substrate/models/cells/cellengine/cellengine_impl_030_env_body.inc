std::string gene_channel_name(const int idx) {
    switch (idx) {
        case 0: return "g0_excitability";
        case 1: return "g1_contractility";
        case 2: return "g2_mechanosense";
        case 3: return "g3_secretion_mag";
        case 4: return "g4_secretion_sign";
        case 5: return "g5_adhesion";
        case 6: return "g6_coupling";
        case 7: return "g7_adaptation";
        default: return "g_unknown";
    }
}

bool edge_ablation_matches(const ODDIntervention* intervention, const int a, const int b) {
    if (!intervention) return false;
    if (intervention->ablate_edge_src < 0 || intervention->ablate_edge_dst < 0) return false;
    const int src = intervention->ablate_edge_src;
    const int dst = intervention->ablate_edge_dst;
    return (a == src && b == dst) || (a == dst && b == src);
}

#ifndef CELLENGINE_INTERNAL_TYPES_DEFINED
struct TaskState {
    double x = 0.0;
    double x_dot = 0.0;
    double theta = 0.0;
    double theta_dot = 0.0;
};

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
            const double anticipation = s.theta + 0.30 * s.theta_dot + 0.18 * task_aux_b * std::max(0.0, 1.2 - task_aux_a);
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
                    s.theta_dot = clampd(s.theta_dot + 0.65 * clampd((s.theta - s.x) / std::max(0.05, task_goal), -1.0, 1.0), -1.25, 1.25);
                } else {
                    terminal = true;
                }
            }
            x_ddot = paddle_dd;
            theta_ddot = 0.0;
            ticks += 1;
            success = (task_counter >= std::max(1, task_target_count));
            last_reward = terminal ? -1.0 : (0.18 * (1.0 - std::min(1.0, std::abs(s.theta - s.x))) + (success ? 1.5 : 0.02 * task_counter));
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
#endif

void configure_task_env(TaskEnv& env, const CellEngineConfig& cfg, const double force_limit) {
    env.kind = parse_task_kind(cfg.task_name);
    env.force_limit = force_limit;
    env.task_goal = 1.0;
    env.task_limit = 1.0;
    env.task_aux_a = 0.0;
    env.task_aux_b = 0.0;
    env.task_target_count = 1;
    if (env.kind == TaskKind::worm_drag_race) {
        env.task_goal = std::max(1.0, cfg.worm_goal_distance);
        env.task_limit = std::max(0.5, cfg.worm_max_backward);
    } else if (env.kind == TaskKind::pong_return) {
        env.task_goal = clampd(cfg.pong_paddle_half_height, 0.08, 0.45);
        env.task_limit = clampd(cfg.pong_ball_speed, 0.45, 1.60);
        env.task_aux_b = std::isfinite(cfg.replay_task_b) ? cfg.replay_task_b : 0.0;
        env.task_target_count = std::max(1, cfg.pong_target_hits);
    }
}

Tensor make_state_tensor(const TaskEnv& env) {
    Tensor t(1, 4);
    if (env.kind == TaskKind::worm_drag_race) {
        t(0, 0) = clampd(env.s.x / std::max(1.0, env.task_goal), -2.0, 2.0);
        t(0, 1) = std::tanh(env.s.x_dot / 1.2);
        t(0, 2) = clampd(env.s.theta / 1.2, -2.0, 2.0);
        t(0, 3) = std::tanh(env.s.theta_dot / 2.2);
        return t;
    }
    if (env.kind == TaskKind::pong_return) {
        t(0, 0) = clampd(env.s.x / 1.0, -2.0, 2.0);
        t(0, 1) = std::tanh(env.s.x_dot / 2.0);
        t(0, 2) = clampd(env.s.theta / 1.0, -2.0, 2.0);
        t(0, 3) = clampd(env.task_aux_a / 1.15, -2.0, 2.0);
        return t;
    }
    t(0, 0) = clampd(env.s.x / 2.4, -3.0, 3.0);
    t(0, 1) = std::tanh(env.s.x_dot / 3.0);
    t(0, 2) = clampd(env.s.theta / kFailAngleRad, -2.0, 2.0);
    t(0, 3) = std::tanh(env.s.theta_dot / 3.5);
    return t;
}

using Coord3 = std::tuple<int, int, int>;

int grid_linear_index(const BodyTemplate& body, const int x, const int y, const int z) {
    const int gx = x - body.grid_min_x;
    const int gy = y - body.grid_min_y;
    const int gz = z - body.grid_min_z;
    if (gx < 0 || gx >= body.grid_sx || gy < 0 || gy >= body.grid_sy || gz < 0 || gz >= body.grid_sz) {
        return -1;
    }
    return gx + body.grid_sx * (gy + body.grid_sy * gz);
}

std::array<double, kGeneExprDim> provisional_gene_expr(const Genome& genome, const std::array<double, 5>& features) {
    std::array<double, kGeneExprDim> out{};
    for (int gidx = 0; gidx < kGeneExprDim; ++gidx) {
        double acc = map_gene(genome.genes[static_cast<std::size_t>(gidx) * 5], -3.0, 3.0) * features[0] +
                     map_gene(genome.genes[static_cast<std::size_t>(gidx) * 5 + 1], -3.0, 3.0) * features[1] +
                     map_gene(genome.genes[static_cast<std::size_t>(gidx) * 5 + 2], -3.0, 3.0) * features[2] +
                     map_gene(genome.genes[static_cast<std::size_t>(gidx) * 5 + 3], -3.0, 3.0) * features[3] +
                     map_gene(genome.genes[static_cast<std::size_t>(gidx) * 5 + 4], -3.0, 3.0) * features[4] +
                     map_gene(genome.genes[40 + gidx], -2.2, 2.2);
        out[gidx] = sigmoid(acc);
    }
    return out;
}

void finalize_body_template(BodyTemplate& body) {
    if (body.cells.empty()) return;

    std::sort(body.cells.begin(), body.cells.end(), [](const BodyCell& a, const BodyCell& b) {
        if (a.y != b.y) return a.y < b.y;
        if (a.z != b.z) return a.z < b.z;
        return a.x < b.x;
    });

    std::map<Coord3, int> index_of;
    int min_x = body.cells.front().x;
    int max_x = body.cells.front().x;
    int min_y = body.cells.front().y;
    int max_y = body.cells.front().y;
    int min_z = body.cells.front().z;
    int max_z = body.cells.front().z;
    for (std::size_t i = 0; i < body.cells.size(); ++i) {
        const auto& c = body.cells[i];
        index_of[{c.x, c.y, c.z}] = static_cast<int>(i);
        min_x = std::min(min_x, c.x);
        max_x = std::max(max_x, c.x);
        min_y = std::min(min_y, c.y);
        max_y = std::max(max_y, c.y);
        min_z = std::min(min_z, c.z);
        max_z = std::max(max_z, c.z);
    }

    body.neighbor_offsets.assign(body.cells.size() + 1, 0);
    body.neighbor_flat.clear();
    body.sensor_cells.clear();
    body.motor_cells.clear();

    int flat_count = 0;
    for (std::size_t i = 0; i < body.cells.size(); ++i) {
        const auto& c = body.cells[i];
        body.neighbor_offsets[i] = flat_count;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    const auto it = index_of.find({c.x + dx, c.y + dy, c.z + dz});
                    if (it != index_of.end()) {
                        body.neighbor_flat.push_back(it->second);
                        flat_count += 1;
                    }
                }
            }
        }
    }
    body.neighbor_offsets[body.cells.size()] = flat_count;

    body.grid_min_x = min_x - 1;
    body.grid_max_x = max_x + 1;
    body.grid_min_y = std::max(0, min_y - 1);
    body.grid_max_y = max_y + 1;
    body.grid_min_z = min_z - 1;
    body.grid_max_z = max_z + 1;
    body.grid_sx = body.grid_max_x - body.grid_min_x + 1;
    body.grid_sy = body.grid_max_y - body.grid_min_y + 1;
    body.grid_sz = body.grid_max_z - body.grid_min_z + 1;
    const int grid_count = body.grid_sx * body.grid_sy * body.grid_sz;
    body.grid_to_cell.assign(static_cast<std::size_t>(grid_count), -1);
    body.cell_to_grid.assign(body.cells.size(), -1);
    for (std::size_t i = 0; i < body.cells.size(); ++i) {
        const int gi = grid_linear_index(body, body.cells[i].x, body.cells[i].y, body.cells[i].z);
        body.cell_to_grid[i] = gi;
        body.grid_to_cell[static_cast<std::size_t>(gi)] = static_cast<int>(i);
    }

    body.diffusion_offsets.assign(static_cast<std::size_t>(grid_count) + 1, 0);
    body.diffusion_flat.clear();
    int diff_flat = 0;
    for (int gz = body.grid_min_z; gz <= body.grid_max_z; ++gz) {
        for (int gy = body.grid_min_y; gy <= body.grid_max_y; ++gy) {
            for (int gx = body.grid_min_x; gx <= body.grid_max_x; ++gx) {
                const int gi = grid_linear_index(body, gx, gy, gz);
                body.diffusion_offsets[static_cast<std::size_t>(gi)] = diff_flat;
                const std::array<Coord3, 6> nb = {
                    Coord3{gx - 1, gy, gz}, Coord3{gx + 1, gy, gz},
                    Coord3{gx, gy - 1, gz}, Coord3{gx, gy + 1, gz},
                    Coord3{gx, gy, gz - 1}, Coord3{gx, gy, gz + 1}
                };
                for (const auto& [nx, ny, nz] : nb) {
                    const int ni = grid_linear_index(body, nx, ny, nz);
                    if (ni >= 0) {
                        body.diffusion_flat.push_back(ni);
                        diff_flat += 1;
                    }
                }
            }
        }
    }
    body.diffusion_offsets[static_cast<std::size_t>(grid_count)] = diff_flat;

    const double max_abs_x = std::max(1, std::max(std::abs(min_x), std::abs(max_x)));
    const double max_abs_z = std::max(1, std::max(std::abs(min_z), std::abs(max_z)));
    const double max_height = std::max(1, max_y - min_y);
    const double hinge_y = static_cast<double>(min_y) + 1.5;
    const double max_radial = std::max(1.0, std::sqrt(sqr(max_abs_x) + sqr(max_abs_z)));

    for (std::size_t i = 0; i < body.cells.size(); ++i) {
        auto& c = body.cells[i];
        const int neighbors = body.neighbor_offsets[i + 1] - body.neighbor_offsets[i];
        const double x_norm = c.x / max_abs_x;
        const double y_norm = (c.y - min_y) / max_height;
        const double radial = std::sqrt(sqr(static_cast<double>(c.x)) + sqr(static_cast<double>(c.z)));
        const double dist_hinge = std::sqrt(sqr(static_cast<double>(c.x)) + sqr(static_cast<double>(c.z)) + sqr(static_cast<double>(c.y) - hinge_y));
        const double hinge_prox = std::exp(-dist_hinge / 2.1);
        const double surface_prox = 1.0 - static_cast<double>(neighbors) / static_cast<double>(kMaxNeighborDirs);
        const double neighbor_density = static_cast<double>(neighbors) / static_cast<double>(kMaxNeighborDirs);
        c.ground = (c.y == min_y);
        c.hinge = (radial <= 1.5 && c.y <= min_y + 2);
        c.motor = c.ground && std::abs(c.x) >= std::max(1, static_cast<int>(std::round(max_abs_x / 3.0)));
        c.features = {x_norm, clampd(y_norm, 0.0, 1.0), hinge_prox, clampd(surface_prox, 0.0, 1.0), neighbor_density};
        c.force_direction_x = c.ground ? clampd(-x_norm, -1.0, 1.0) * (0.9 + 0.1 * (1.0 - std::abs(c.z) / max_abs_z)) : 0.0;
        c.mech_advantage = c.ground
                               ? (0.62 + 0.38 * std::abs(c.force_direction_x))
                               : ((c.y <= min_y + 1)
                                      ? (0.28 + 0.24 * (1.0 - radial / max_radial))
                                      : (0.08 + 0.20 * hinge_prox * (0.9 + 0.1 * (1.0 - std::abs(c.z) / max_abs_z))));
        if (c.hinge) body.sensor_cells.push_back(static_cast<int>(i));
        if (c.motor) body.motor_cells.push_back(static_cast<int>(i));
    }
}

BodyTemplate make_fixed_body_template() {
    BodyTemplate body;
    for (int y = 0; y <= 1; ++y) {
        for (int x = -4; x <= 4; ++x) {
            body.cells.push_back({x, y, 0, 0});
        }
    }
    for (int y = 2; y <= 12; ++y) {
        body.cells.push_back({0, y, 0, 0});
    }
    finalize_body_template(body);
    return body;
}

BodyTemplate make_developed_body_template(const Genome& genome,
                                          const CellEngineConfig& cfg,
                                          const bool allow_depth) {
    BodyTemplate body;
    std::set<Coord3> occupied;
    const int extent_x = std::max(2, cfg.body_extent_x);
    const int extent_y = std::max(4, cfg.body_extent_y);
    const int extent_z = allow_depth ? std::max(0, cfg.body_extent_z) : 0;
    const int seed_half_width = std::max(1, cfg.development_seed_half_width);
    const int seed_half_depth = allow_depth ? std::min(1, extent_z) : 0;
    const int target_min = std::max(18, 2 * seed_half_width + 3);
    const int target_cells = std::min(
        std::max(target_min,
                 static_cast<int>(std::round(18.0 + gene01(genome.genes[83]) *
                                                        static_cast<double>(std::max(0, cfg.max_cells - 18))))),
        std::max(target_min, cfg.max_cells));
    const double threshold = cfg.development_growth_threshold + 0.18 * (gene01(genome.genes[84]) - 0.5);
    const double vertical_bias = 2.0 * (gene01(genome.genes[80]) - 0.5);
    const double lateral_bias = 2.0 * (gene01(genome.genes[81]) - 0.5);
    const double depth_bias = 2.0 * (gene01(genome.genes[82]) - 0.5);
    const double branch_bias = 2.0 * (gene01(genome.genes[85]) - 0.5);
    const double bridge_bias = 2.0 * (gene01(genome.genes[86]) - 0.5);

    auto add_cell = [&](const int x, const int y, const int z, const int birth_step) {
        if (!occupied.insert({x, y, z}).second) return;
        body.cells.push_back({x, y, z, birth_step});
    };
    auto is_occupied = [&](const int x, const int y, const int z) {
        return occupied.find({x, y, z}) != occupied.end();
    };

    for (int z = -seed_half_depth; z <= seed_half_depth; ++z) {
        for (int x = -seed_half_width; x <= seed_half_width; ++x) {
            add_cell(x, 0, z, 0);
        }
    }
    add_cell(0, 1, 0, 0);

    for (int step = 1; step <= std::max(1, cfg.development_steps) && static_cast<int>(body.cells.size()) < target_cells; ++step) {
        struct GrowthCandidate {
            BodyCell cell;
            double prob = 0.0;
        };
        std::vector<GrowthCandidate> candidates;
        std::set<Coord3> seen_candidates;

        for (const auto& cell : body.cells) {
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        const int nx = cell.x + dx;
                        const int ny = cell.y + dy;
                        const int nz = cell.z + dz;
                        if (nx < -extent_x || nx > extent_x || ny < 0 || ny > extent_y || nz < -extent_z || nz > extent_z) continue;
                        if (!seen_candidates.insert({nx, ny, nz}).second) continue;
                        if (is_occupied(nx, ny, nz)) continue;

                        int neighbors = 0;
                        bool support = (ny == 0);
                        bool bridge = false;
                        for (int ndz = -1; ndz <= 1; ++ndz) {
                            for (int ndy = -1; ndy <= 1; ++ndy) {
                                for (int ndx = -1; ndx <= 1; ++ndx) {
                                    if (ndx == 0 && ndy == 0 && ndz == 0) continue;
                                    if (is_occupied(nx + ndx, ny + ndy, nz + ndz)) neighbors += 1;
                                }
                            }
                        }
                        support = support || is_occupied(nx, ny - 1, nz);
                        bridge = (is_occupied(nx - 1, ny, nz) && is_occupied(nx + 1, ny, nz)) ||
                                 (is_occupied(nx, ny, nz - 1) && is_occupied(nx, ny, nz + 1));
                        if (neighbors == 0) continue;

                        const double x_norm = static_cast<double>(nx) / static_cast<double>(std::max(1, extent_x));
                        const double y_norm = static_cast<double>(ny) / static_cast<double>(std::max(1, extent_y));
                        const double z_norm = static_cast<double>(nz) / static_cast<double>(std::max(1, extent_z == 0 ? 1 : extent_z));
                        const double hinge_prox = std::exp(-std::sqrt(sqr(static_cast<double>(nx)) + sqr(static_cast<double>(nz)) + sqr(static_cast<double>(ny) - 1.5)) / 2.4);
                        const double surface = 1.0 - static_cast<double>(neighbors) / static_cast<double>(kMaxNeighborDirs);
                        const double density = static_cast<double>(neighbors) / static_cast<double>(kMaxNeighborDirs);
                        const auto provisional = provisional_gene_expr(genome, {x_norm, y_norm, hinge_prox, clampd(surface, 0.0, 1.0), density});
                        const double radial_norm = std::sqrt(sqr(x_norm) + sqr(z_norm));
                        const double raw_score =
                            0.30 * provisional[5] +
                            0.24 * provisional[6] +
                            0.22 * provisional[7] +
                            0.16 * provisional[1] +
                            0.14 * provisional[2] +
                            0.18 * hinge_prox +
                            0.14 * clampd(surface, 0.0, 1.0) +
                            0.10 * density +
                            (support ? 0.18 : -0.12) +
                            (bridge ? 0.10 * (1.0 + bridge_bias) : 0.0) +
                            0.18 * vertical_bias * y_norm +
                            0.12 * lateral_bias * std::abs(x_norm) +
                            0.10 * depth_bias * std::abs(z_norm) +
                            0.10 * branch_bias * clampd((neighbors - 2.0) / 6.0, 0.0, 1.0) -
                            0.12 * radial_norm * (1.0 - provisional[1]);
                        const double prob = sigmoid(3.0 * (raw_score - threshold));
                        if (prob >= 0.44 || (support && neighbors >= 3)) {
                            GrowthCandidate candidate;
                            candidate.cell = {nx, ny, nz, step};
                            candidate.prob = prob;
                            candidates.push_back(candidate);
                        }
                    }
                }
            }
        }

        if (candidates.empty()) break;
        std::sort(candidates.begin(), candidates.end(), [](const GrowthCandidate& a, const GrowthCandidate& b) {
            if (a.prob != b.prob) return a.prob > b.prob;
            if (a.cell.y != b.cell.y) return a.cell.y < b.cell.y;
            if (a.cell.z != b.cell.z) return a.cell.z < b.cell.z;
            return a.cell.x < b.cell.x;
        });

        const int remaining = std::max(0, target_cells - static_cast<int>(body.cells.size()));
        const int remaining_steps = std::max(1, cfg.development_steps - step + 1);
        const int growth_budget = std::min(
            remaining,
            std::max(1,
                     std::min(static_cast<int>(candidates.size()),
                              static_cast<int>(std::ceil(static_cast<double>(remaining) / static_cast<double>(remaining_steps))))));
        for (int i = 0; i < growth_budget; ++i) {
            const auto& c = candidates[static_cast<std::size_t>(i)].cell;
            add_cell(c.x, c.y, c.z, c.birth_step);
        }
    }

    if (static_cast<int>(body.cells.size()) < target_min) {
        return make_fixed_body_template();
    }
    finalize_body_template(body);
    return body;
}

BodyTemplate make_body_template(const Genome& genome, const CellEngineConfig& cfg) {
    if (cfg.body_mode == "fixed2d") return make_fixed_body_template();
    if (cfg.body_mode == "grown2d") return make_developed_body_template(genome, cfg, false);
    return make_developed_body_template(genome, cfg, true);
}
