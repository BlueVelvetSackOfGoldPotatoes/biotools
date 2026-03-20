int greedy_a2c_action(ActorCriticAgent& agent, const TaskEnv& env) {
    const Tensor st = make_state_tensor(env);
    const Tensor logits = agent.policy.forward(st);
    return (logits(0, 0) >= logits(0, 1)) ? 0 : 1;
}

int greedy_dqn_action(DQNAgent& agent, const TaskEnv& env) {
    const Tensor st = make_state_tensor(env);
    return agent.predict_actions(st).front();
}

void train_a2c_agent(ActorCriticAgent& agent, const CellEngineConfig& cfg, const int seed) {
    std::mt19937 rng(static_cast<unsigned int>(seed));
    const double force_mag = cfg.rl_force_mag;
    for (int episode = 0; episode < cfg.rl_train_episodes; ++episode) {
        TaskEnv env;
        configure_task_env(env, cfg, force_mag);
        env.reset(rng);

        std::vector<Tensor> states;
        std::vector<int> actions;
        std::vector<double> rewards;
        states.reserve(static_cast<std::size_t>(cfg.final_budget.max_ticks));
        actions.reserve(static_cast<std::size_t>(cfg.final_budget.max_ticks));
        rewards.reserve(static_cast<std::size_t>(cfg.final_budget.max_ticks));

        while (!env.terminal && env.ticks < cfg.final_budget.max_ticks) {
            const Tensor st = make_state_tensor(env);
            const Tensor logits = agent.policy.forward(st);
            const auto probs = softmax2(logits, 0);
            const int action = sample_action(probs, rng);
            const double force = (action == 0) ? -force_mag : force_mag;
            env.step(force, cfg.physics_dt);
            states.push_back(st);
            actions.push_back(action);
            rewards.push_back(env.last_reward);
        }

        if (states.empty()) continue;

        std::vector<double> returns(states.size(), 0.0);
        double ret = 0.0;
        for (int i = static_cast<int>(states.size()) - 1; i >= 0; --i) {
            ret = rewards[static_cast<std::size_t>(i)] + agent.gamma * ret;
            returns[static_cast<std::size_t>(i)] = ret;
        }

        Tensor batch(states.size(), 4);
        for (std::size_t i = 0; i < states.size(); ++i) {
            for (std::size_t j = 0; j < 4; ++j) batch(i, j) = states[i](0, j);
        }

        agent.policy_opt.zero_grad(agent.policy);
        agent.value_opt.zero_grad(agent.value);
        const Tensor logits = agent.policy.forward(batch);
        const Tensor values = agent.value.forward(batch);

        std::vector<double> advantages(states.size(), 0.0);
        double adv_mean = 0.0;
        for (std::size_t i = 0; i < states.size(); ++i) {
            advantages[i] = returns[i] - values(i, 0);
            adv_mean += advantages[i];
        }
        adv_mean /= static_cast<double>(advantages.size());
        double adv_var = 0.0;
        for (double a : advantages) adv_var += (a - adv_mean) * (a - adv_mean);
        const double adv_std = std::sqrt(adv_var / static_cast<double>(advantages.size()) + 1e-8);
        for (double& a : advantages) a = (a - adv_mean) / adv_std;

        Tensor grad_logits(states.size(), 2, 0.0);
        Tensor grad_values(states.size(), 1, 0.0);
        const double inv_n = 1.0 / static_cast<double>(states.size());
        for (std::size_t i = 0; i < states.size(); ++i) {
            const auto probs = softmax2(logits, i);
            grad_logits(i, 0) = probs[0];
            grad_logits(i, 1) = probs[1];
            grad_logits(i, static_cast<std::size_t>(actions[i])) -= 1.0;
            grad_logits(i, 0) *= advantages[i] * inv_n;
            grad_logits(i, 1) *= advantages[i] * inv_n;
            grad_values(i, 0) = (values(i, 0) - returns[i]) * inv_n;
        }

        agent.policy.backward(grad_logits);
        agent.value.backward(grad_values);
        agent.policy_opt.step(agent.policy);
        agent.value_opt.step(agent.value);
    }
}

void finetune_a2c_agent(ActorCriticAgent& agent, const CellEngineConfig& cfg, const int seed) {
    std::mt19937 rng(static_cast<unsigned int>(seed));
    const double force_mag = cfg.rl_force_mag;
    for (int episode = 0; episode < cfg.rl_damage_finetune_episodes; ++episode) {
        TaskEnv env;
        configure_task_env(env, cfg, force_mag);
        env.reset(rng);

        std::vector<Tensor> states;
        std::vector<int> actions;
        std::vector<double> rewards;
        while (!env.terminal && env.ticks < cfg.damage_budget.max_ticks) {
            const Tensor st = make_state_tensor(env);
            const Tensor logits = agent.policy.forward(st);
            const auto probs = softmax2(logits, 0);
            const int action = sample_action(probs, rng);
            const double force = (action == 0) ? -force_mag : force_mag;
            env.step(force, cfg.physics_dt);
            states.push_back(st);
            actions.push_back(action);
            rewards.push_back(env.last_reward);
        }
        if (states.empty()) continue;

        std::vector<double> returns(states.size(), 0.0);
        double ret = 0.0;
        for (int i = static_cast<int>(states.size()) - 1; i >= 0; --i) {
            ret = rewards[static_cast<std::size_t>(i)] + agent.gamma * ret;
            returns[static_cast<std::size_t>(i)] = ret;
        }

        Tensor batch(states.size(), 4);
        for (std::size_t i = 0; i < states.size(); ++i) {
            for (std::size_t j = 0; j < 4; ++j) batch(i, j) = states[i](0, j);
        }
        agent.policy_opt.zero_grad(agent.policy);
        agent.value_opt.zero_grad(agent.value);
        const Tensor logits = agent.policy.forward(batch);
        const Tensor values = agent.value.forward(batch);
        Tensor grad_logits(states.size(), 2, 0.0);
        Tensor grad_values(states.size(), 1, 0.0);
        const double inv_n = 1.0 / static_cast<double>(states.size());
        for (std::size_t i = 0; i < states.size(); ++i) {
            const double advantage = returns[i] - values(i, 0);
            const auto probs = softmax2(logits, i);
            grad_logits(i, 0) = probs[0];
            grad_logits(i, 1) = probs[1];
            grad_logits(i, static_cast<std::size_t>(actions[i])) -= 1.0;
            grad_logits(i, 0) *= advantage * inv_n;
            grad_logits(i, 1) *= advantage * inv_n;
            grad_values(i, 0) = (values(i, 0) - returns[i]) * inv_n;
        }
        agent.policy.backward(grad_logits);
        agent.value.backward(grad_values);
        agent.policy_opt.step(agent.policy);
        agent.value_opt.step(agent.value);
    }
}

EvaluationSummary evaluate_a2c_agent(ActorCriticAgent& agent,
                                     const CellEngineConfig& cfg,
                                     const TrialBudget& budget,
                                     const int seed) {
    std::mt19937 eval_rng(static_cast<unsigned int>(seed));
    std::vector<TrialStats> trials;
    trials.reserve(static_cast<std::size_t>(budget.num_trials));
    const double force_mag = cfg.rl_force_mag;
    for (int t = 0; t < budget.num_trials; ++t) {
        TaskEnv env;
        configure_task_env(env, cfg, force_mag);
        env.reset(eval_rng);
        TrialStats stats;
        while (!env.terminal && env.ticks < budget.max_ticks) {
            const int action = greedy_a2c_action(agent, env);
            const double force = (action == 0) ? -force_mag : force_mag;
            env.step(force, cfg.physics_dt);
            stats.mean_abs_force += std::abs(force);
            stats.mean_abs_theta += std::abs(env.s.theta);
        }
        stats.ticks = env.ticks;
        stats.success = env.trial_success(budget);
        stats.task_primary = env.primary_metric(budget);
        stats.task_secondary = env.secondary_metric();
        stats.mean_energy = 1.0;
        if (stats.ticks > 0) {
            stats.mean_abs_force /= static_cast<double>(stats.ticks);
            stats.mean_abs_theta /= static_cast<double>(stats.ticks);
        }
        trials.push_back(stats);
    }
    auto out = summarize_trials(trials, budget);
    annotate_task_summary(out, parse_task_kind(cfg.task_name));
    out.sensor_responsiveness = 0.0;
    out.signal_connectivity = 0.0;
    out.force_correlation = 0.0;
    return out;
}

void train_dqn_agent(DQNAgent& agent, const CellEngineConfig& cfg, const int seed) {
    std::mt19937 train_rng(static_cast<unsigned int>(seed + 11));
    const double force_mag = cfg.rl_force_mag;
    double best_eval = 0.0;
    for (int episode = 1; episode <= cfg.rl_train_episodes; ++episode) {
        TaskEnv env;
        configure_task_env(env, cfg, force_mag);
        env.reset(train_rng);
        int steps = 0;
        while (!env.terminal && steps < cfg.final_budget.max_ticks) {
            const Tensor st = make_state_tensor(env);
            const int action = agent.select_action(st, train_rng);
            const double force = (action == 0) ? -force_mag : force_mag;
            env.step(force, cfg.physics_dt);
            const Tensor next = make_state_tensor(env);
            const RLTransition tr{st, action, env.last_reward, next, env.terminal ? 1 : 0};
            agent.observe(tr);
            (void)agent.train_step(train_rng);
            agent.decay_epsilon();
            steps += 1;
        }

        if (episode % cfg.rl_eval_every == 0) {
            const auto eval = evaluate_dqn_agent(agent, cfg, cfg.search_budget, seed + 500 + episode);
            best_eval = std::max(best_eval, eval.survival_ratio);
            if (best_eval >= 0.98) break;
        }
    }
}

void finetune_dqn_agent(DQNAgent& agent, const CellEngineConfig& cfg, const int seed) {
    agent.set_epsilon(0.25);
    std::mt19937 fine_rng(static_cast<unsigned int>(seed));
    const double force_mag = cfg.rl_force_mag;
    for (int episode = 0; episode < cfg.rl_damage_finetune_episodes; ++episode) {
        TaskEnv env;
        configure_task_env(env, cfg, force_mag);
        env.reset(fine_rng);
        int steps = 0;
        while (!env.terminal && steps < cfg.damage_budget.max_ticks) {
            const Tensor st = make_state_tensor(env);
            const int action = agent.select_action(st, fine_rng);
            const double force = (action == 0) ? -force_mag : force_mag;
            env.step(force, cfg.physics_dt);
            const Tensor next = make_state_tensor(env);
            const RLTransition tr{st, action, env.last_reward, next, env.terminal ? 1 : 0};
            agent.observe(tr);
            (void)agent.train_step(fine_rng);
            agent.decay_epsilon();
            steps += 1;
        }
    }
}

RLDamageSummary train_and_evaluate_a2c(const CellEngineConfig& cfg, const int seed) {
    std::mt19937 init_rng(static_cast<unsigned int>(seed));
    ActorCriticAgent agent(init_rng);
    train_a2c_agent(agent, cfg, seed);

    RLDamageSummary out;
    out.clean = evaluate_a2c_agent(agent, cfg, cfg.final_budget, seed + 12000);

    std::mt19937 damage_rng(static_cast<unsigned int>(seed + 41));
    damage_agent_parameters(agent.policy.parameters(), cfg.damage_fraction, damage_rng);
    damage_agent_parameters(agent.value.parameters(), cfg.damage_fraction, damage_rng);
    out.damaged = evaluate_a2c_agent(agent, cfg, cfg.damage_budget, seed + 12100);

    finetune_a2c_agent(agent, cfg, seed + 12200);
    out.recovered = evaluate_a2c_agent(agent, cfg, cfg.damage_budget, seed + 12300);
    return out;
}

void damage_agent_parameters(std::vector<Tensor*> params, const double frac, std::mt19937& rng) {
    std::vector<std::pair<Tensor*, std::size_t>> slots;
    for (Tensor* t : params) {
        for (std::size_t i = 0; i < t->data.size(); ++i) slots.emplace_back(t, i);
    }
    std::shuffle(slots.begin(), slots.end(), rng);
    const std::size_t n = std::max<std::size_t>(1, static_cast<std::size_t>(std::round(frac * slots.size())));
    for (std::size_t i = 0; i < n && i < slots.size(); ++i) {
        slots[i].first->data[slots[i].second] = 0.0;
    }
}

RLDamageSummary train_and_evaluate_dqn(const CellEngineConfig& cfg, const int seed) {
    std::mt19937 rng(static_cast<unsigned int>(seed));
    DQNAgent agent(/*state_dim=*/4,
                   /*hidden_dim=*/64,
                   /*num_actions=*/2,
                   /*lr=*/0.001,
                   /*gamma=*/0.995,
                   /*replay_capacity=*/20000,
                   /*batch_size=*/64,
                   /*target_update_interval=*/120,
                   /*double_q=*/true,
                   rng,
                   /*eps_start=*/1.0,
                   /*eps_end=*/0.03,
                   /*eps_decay=*/0.995);
    train_dqn_agent(agent, cfg, seed);

    RLDamageSummary out;
    out.clean = evaluate_dqn_agent(agent, cfg, cfg.final_budget, seed + 9000);

    std::mt19937 damage_rng(static_cast<unsigned int>(seed + 21));
    damage_agent_parameters(agent.parameters(), cfg.damage_fraction, damage_rng);
    out.damaged = evaluate_dqn_agent(agent, cfg, cfg.damage_budget, seed + 9100);

    finetune_dqn_agent(agent, cfg, seed + 31);
    out.recovered = evaluate_dqn_agent(agent, cfg, cfg.damage_budget, seed + 9200);
    return out;
}

std::pair<std::string, RLDamageSummary> train_and_select_rl(const CellEngineConfig& cfg, const int seed,
                                                            RLDamageSummary& dqn_out,
                                                            RLDamageSummary& a2c_out) {
    const bool want_dqn = (cfg.rl_algorithm == "all" || cfg.rl_algorithm == "dqn");
    const bool want_a2c = (cfg.rl_algorithm == "all" || cfg.rl_algorithm == "a2c");

    if (want_dqn) {
        dqn_out = train_and_evaluate_dqn(cfg, seed);
    }
    if (want_a2c) {
        a2c_out = train_and_evaluate_a2c(cfg, seed + 1000);
    }

    if (want_dqn && !want_a2c) return {"dqn", dqn_out};
    if (!want_dqn && want_a2c) return {"a2c", a2c_out};

    if (a2c_out.clean.survival_ratio >= dqn_out.clean.survival_ratio) {
        return {"a2c", a2c_out};
    }
    return {"dqn", dqn_out};
}

double resolve_replay_theta_deg(const CellEngineConfig& cfg, std::mt19937& rng) {
    const TaskKind kind = parse_task_kind(cfg.task_name);
    if (kind == TaskKind::worm_drag_race) {
        if (std::isfinite(cfg.replay_task_a)) return cfg.replay_task_a;
        std::uniform_real_distribution<double> bend_dist(-0.18, 0.18);
        return bend_dist(rng);
    }
    if (kind == TaskKind::pong_return) {
        if (std::isfinite(cfg.replay_task_a)) return cfg.replay_task_a;
        std::uniform_real_distribution<double> serve_dist(-0.55, 0.55);
        return serve_dist(rng);
    }
    if (std::isfinite(cfg.replay_theta_deg)) return cfg.replay_theta_deg;
    std::uniform_real_distribution<double> theta_dist(-3.0, 3.0);
    return theta_dist(rng);
}

TrialStats replay_a2c_episode(ActorCriticAgent& agent,
                              const CellEngineConfig& cfg,
                              const TrialBudget& budget,
                              const double theta_deg,
                              const int seed,
                              std::vector<ReplayFrame>& frames,
                              std::vector<int>& actions) {
    std::mt19937 env_rng(static_cast<unsigned int>(seed));
    std::mt19937 damage_rng(static_cast<unsigned int>(seed + 91));
    TaskEnv env;
    configure_task_env(env, cfg, cfg.rl_force_mag);
    if (env.kind == TaskKind::cartpole_balance || env.kind == TaskKind::mass_spring_balance) {
        env.reset(env_rng, theta_deg * kPi / 180.0);
    } else {
        env.reset(env_rng, theta_deg);
    }
    TrialStats stats;
    frames.clear();
    actions.clear();
    frames.reserve(static_cast<std::size_t>(budget.max_ticks + 1));
    actions.reserve(static_cast<std::size_t>(budget.max_ticks + 1));
    frames.push_back(make_replay_frame(CellPopulation{}, env));
    actions.push_back(-1);

    bool damage_applied = false;
    bool damage_event_pending = false;
    while (!env.terminal && env.ticks < budget.max_ticks) {
        if (!damage_applied && cfg.replay_damage_tick >= 0 && env.ticks == cfg.replay_damage_tick) {
            damage_agent_parameters(agent.policy.parameters(), cfg.damage_fraction, damage_rng);
            damage_applied = true;
            damage_event_pending = true;
        }
        const int action = greedy_a2c_action(agent, env);
        const double force = (action == 0) ? -cfg.rl_force_mag : cfg.rl_force_mag;
        env.step(force, cfg.physics_dt);
        stats.mean_abs_force += std::abs(force);
        stats.mean_abs_theta += std::abs(env.s.theta);
        ReplayFrame frame = make_replay_frame(CellPopulation{}, env, damage_event_pending);
        frame.terminal = env.terminal;
        frames.push_back(frame);
        actions.push_back(action);
        damage_event_pending = false;
    }
    stats.ticks = env.ticks;
    stats.success = env.trial_success(budget);
    stats.task_primary = env.primary_metric(budget);
    stats.task_secondary = env.secondary_metric();
    stats.mean_energy = 1.0;
    if (stats.ticks > 0) {
        stats.mean_abs_force /= static_cast<double>(stats.ticks);
        stats.mean_abs_theta /= static_cast<double>(stats.ticks);
    }
    return stats;
}

TrialStats replay_dqn_episode(DQNAgent& agent,
                              const CellEngineConfig& cfg,
                              const TrialBudget& budget,
                              const double theta_deg,
                              const int seed,
                              std::vector<ReplayFrame>& frames,
                              std::vector<int>& actions) {
    std::mt19937 env_rng(static_cast<unsigned int>(seed));
    std::mt19937 damage_rng(static_cast<unsigned int>(seed + 61));
    TaskEnv env;
    configure_task_env(env, cfg, cfg.rl_force_mag);
    if (env.kind == TaskKind::cartpole_balance || env.kind == TaskKind::mass_spring_balance) {
        env.reset(env_rng, theta_deg * kPi / 180.0);
    } else {
        env.reset(env_rng, theta_deg);
    }
    TrialStats stats;
    frames.clear();
    actions.clear();
    frames.reserve(static_cast<std::size_t>(budget.max_ticks + 1));
    actions.reserve(static_cast<std::size_t>(budget.max_ticks + 1));
    frames.push_back(make_replay_frame(CellPopulation{}, env));
    actions.push_back(-1);

    bool damage_applied = false;
    bool damage_event_pending = false;
    while (!env.terminal && env.ticks < budget.max_ticks) {
        if (!damage_applied && cfg.replay_damage_tick >= 0 && env.ticks == cfg.replay_damage_tick) {
            damage_agent_parameters(agent.parameters(), cfg.damage_fraction, damage_rng);
            damage_applied = true;
            damage_event_pending = true;
        }
        const int action = greedy_dqn_action(agent, env);
        const double force = (action == 0) ? -cfg.rl_force_mag : cfg.rl_force_mag;
        env.step(force, cfg.physics_dt);
        stats.mean_abs_force += std::abs(force);
        stats.mean_abs_theta += std::abs(env.s.theta);
        ReplayFrame frame = make_replay_frame(CellPopulation{}, env, damage_event_pending);
        frame.terminal = env.terminal;
        frames.push_back(frame);
        actions.push_back(action);
        damage_event_pending = false;
    }
    stats.ticks = env.ticks;
    stats.success = env.trial_success(budget);
    stats.task_primary = env.primary_metric(budget);
    stats.task_secondary = env.secondary_metric();
    stats.mean_energy = 1.0;
    if (stats.ticks > 0) {
        stats.mean_abs_force /= static_cast<double>(stats.ticks);
        stats.mean_abs_theta /= static_cast<double>(stats.ticks);
    }
    return stats;
}

RLReplayResult build_rl_replay(const CellEngineConfig& cfg,
                               const TrialBudget& budget,
                               const double theta_deg,
                               const int seed) {
    const std::string algorithm = (cfg.rl_algorithm == "dqn") ? "dqn" : "a2c";
    RLReplayResult result;
    result.algorithm = algorithm;
    if (algorithm == "dqn") {
        std::mt19937 rng(static_cast<unsigned int>(seed));
        DQNAgent agent(/*state_dim=*/4,
                       /*hidden_dim=*/64,
                       /*num_actions=*/2,
                       /*lr=*/0.001,
                       /*gamma=*/0.995,
                       /*replay_capacity=*/20000,
                       /*batch_size=*/64,
                       /*target_update_interval=*/120,
                       /*double_q=*/true,
                       rng,
                       /*eps_start=*/1.0,
                       /*eps_end=*/0.03,
                       /*eps_decay=*/0.995);
        train_dqn_agent(agent, cfg, seed);
        const TrialStats stats = replay_dqn_episode(agent, cfg, budget, theta_deg, seed + 7000, result.frames, result.actions);
        result.summary = summarize_single_trial(stats, budget);
    } else {
        std::mt19937 rng(static_cast<unsigned int>(seed));
        ActorCriticAgent agent(rng);
        train_a2c_agent(agent, cfg, seed);
        const TrialStats stats = replay_a2c_episode(agent, cfg, budget, theta_deg, seed + 8000, result.frames, result.actions);
        result.summary = summarize_single_trial(stats, budget);
    }
    annotate_task_summary(result.summary, parse_task_kind(cfg.task_name));
    return result;
}

bool write_decoded_genome_json(const Genome& genome,
                               const CandidateParams& params,
                               const std::string& path);

bool write_trial_context_artifacts(const std::vector<ReplayCellState>& pre_reset_rows,
                                   const std::vector<ReplayCellState>& post_reset_rows,
                                   const bool teacher_active,
                                   const bool damage_active,
                                   const std::string& output_dir,
                                   const std::string& file_prefix = "");

bool write_named_trial_context_bundle(const Genome& genome,
                                      const CellEngineConfig& cfg,
                                      const bool teacher_active,
                                      const bool damage_active,
                                      const int seed_offset,
                                      const std::string& output_dir,
                                      const std::string& file_prefix);

ComparisonSummary preview_loaded_genome(const Genome& genome, const CellEngineConfig& cfg) {
    const CellEngineConfig resolved_cfg = resolve_loaded_genome_cfg(cfg, genome);
    const BodyTemplate body = make_body_template(genome, resolved_cfg);
    CellPopulation pop = build_population(genome, body);
    write_replay_body_csv(body, pop, (std::filesystem::path(cfg.output_dir) / "preview_body.csv").string());
    write_decoded_genome_json(genome,
                              default_candidate_params(resolved_cfg),
                              (std::filesystem::path(cfg.output_dir) / "decoded_genome.json").string());

    ComparisonSummary summary;
    summary.attempt = 1;
    summary.champion_generation = -1;
    summary.champion_cell_count = body.cell_count();
    summary.champion_body_depth = body.body_depth();
    summary.champion_search_score = std::numeric_limits<double>::quiet_NaN();
    summary.champion_genes = genome.genes;
    summary.champion_params = default_candidate_params(resolved_cfg);
    summary.task_name = resolved_cfg.task_name;
    summary.body_mode = resolved_cfg.body_mode;
    summary.solved = true;
    summary.notes = "preview mode, body=" + (std::filesystem::path(cfg.output_dir) / "preview_body.csv").string();
    return summary;
}

ComparisonSummary replay_loaded_genome(const Genome& genome, const CellEngineConfig& cfg) {
    const CellEngineConfig resolved_cfg = resolve_loaded_genome_cfg(cfg, genome);
    const BodyTemplate body = make_body_template(genome, resolved_cfg);
    const DecodedGenome d = decode_genome(genome);
    CellPopulation pop = build_population(genome, body);
    const CellPopulation pre_reset_pop = pop;
    std::mt19937 rng(static_cast<unsigned int>(resolved_cfg.seed));
    TaskEnv env;
    env.kind = parse_task_kind(resolved_cfg.task_name);
    TrialBudget replay_budget = resolved_cfg.final_budget;
    replay_budget.num_trials = 1;
    CellEngineConfig replay_cfg = resolved_cfg;
    replay_cfg.replay_theta_deg = resolve_replay_theta_deg(resolved_cfg, rng);

    const auto [stats, frames, cell_rows, edge_rows] =
        run_trial_with_trace(pop, body, d, replay_cfg, replay_budget, rng, false, env);
    write_replay_trace_csv(frames, (std::filesystem::path(cfg.output_dir) / "replay_trace.csv").string());
    write_replay_body_csv(body, pop, (std::filesystem::path(cfg.output_dir) / "replay_body.csv").string());
    write_replay_cells_csv(cell_rows, (std::filesystem::path(cfg.output_dir) / "replay_cells.csv").string());
    write_replay_edges_csv(edge_rows, (std::filesystem::path(cfg.output_dir) / "replay_edges.csv").string());
    write_decoded_genome_json(genome,
                              default_candidate_params(resolved_cfg),
                              (std::filesystem::path(cfg.output_dir) / "decoded_genome.json").string());
    std::vector<ReplayCellState> pre_reset_rows;
    append_replay_cell_state(pre_reset_pop, body, d, 0, nullptr, pre_reset_rows);
    std::vector<ReplayCellState> post_reset_rows;
    const std::size_t snapshot_n = std::min<std::size_t>(static_cast<std::size_t>(pop.count), cell_rows.size());
    post_reset_rows.insert(post_reset_rows.end(), cell_rows.begin(), cell_rows.begin() + static_cast<std::ptrdiff_t>(snapshot_n));
    write_trial_context_artifacts(pre_reset_rows,
                                  post_reset_rows,
                                  /*teacher_active=*/false,
                                  /*damage_active=*/(replay_cfg.replay_damage_tick >= 0),
                                  cfg.output_dir,
                                  "");

    const bool want_rl_replay = (resolved_cfg.rl_algorithm != "none");
    RLReplayResult rl_replay;
    if (want_rl_replay) {
        rl_replay = build_rl_replay(replay_cfg, replay_budget, replay_cfg.replay_theta_deg, resolved_cfg.seed + 60000);
        write_rl_replay_trace_csv(rl_replay.frames,
                                  rl_replay.actions,
                                  rl_replay.algorithm,
                                  (std::filesystem::path(cfg.output_dir) / "rl_replay_trace.csv").string());
    }

    if (cfg.replay_emit_stdout) {
        const int stride = std::max(1, cfg.replay_frame_stride);
        for (std::size_t i = 0; i < frames.size(); i += static_cast<std::size_t>(stride)) {
            if (cfg.replay_clear_screen) {
                std::cout << "\x1b[2J\x1b[H";
            }
            std::cout << ascii_frame(frames[i], replay_budget.max_ticks) << std::flush;
            if (cfg.replay_sleep_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(cfg.replay_sleep_ms));
            }
        }
        if (!frames.empty() && ((frames.size() - 1) % static_cast<std::size_t>(stride) != 0)) {
            if (cfg.replay_clear_screen) {
                std::cout << "\x1b[2J\x1b[H";
            }
            std::cout << ascii_frame(frames.back(), replay_budget.max_ticks) << std::flush;
        }
        std::cout << "\nreplay_trace_csv=" << (std::filesystem::path(cfg.output_dir) / "replay_trace.csv").string() << "\n";
    }

    ComparisonSummary summary;
    summary.attempt = 1;
    summary.champion_generation = -1;
    summary.champion_cell_count = body.cell_count();
    summary.champion_body_depth = body.body_depth();
    summary.champion_search_score = std::numeric_limits<double>::quiet_NaN();
    summary.champion_genes = genome.genes;
    summary.champion_params = default_candidate_params(resolved_cfg);
    summary.task_name = resolved_cfg.task_name;
    summary.body_mode = resolved_cfg.body_mode;
    summary.cell_clean.total_ticks = stats.ticks;
    summary.cell_clean.max_ticks = replay_budget.max_ticks;
    summary.cell_clean.mean_ticks = stats.ticks;
    summary.cell_clean.survival_ratio = static_cast<double>(stats.ticks) / static_cast<double>(replay_budget.max_ticks);
    summary.cell_clean.success_rate = stats.success ? 1.0 : 0.0;
    summary.cell_clean.task_primary = stats.task_primary;
    summary.cell_clean.task_secondary = stats.task_secondary;
    annotate_task_summary(summary.cell_clean, parse_task_kind(resolved_cfg.task_name));
    summary.cell_clean.mean_force_abs = stats.mean_abs_force;
    summary.cell_clean.mean_energy = stats.mean_energy;
    if (want_rl_replay) {
        summary.rl_algorithm = rl_replay.algorithm;
        summary.rl.clean = rl_replay.summary;
        if (rl_replay.algorithm == "dqn") {
            summary.rl_dqn.clean = rl_replay.summary;
        } else {
            summary.rl_a2c.clean = rl_replay.summary;
        }
    }
    summary.solved = task_is_solved(summary.cell_clean, parse_task_kind(resolved_cfg.task_name), resolved_cfg);
    std::ostringstream note;
    note << "replay mode, ticks=" << stats.ticks
         << ", success=" << (stats.success ? "true" : "false")
         << ", theta_deg=" << replay_cfg.replay_theta_deg
         << ", rl_replay=" << (want_rl_replay ? rl_replay.algorithm : "disabled")
         << ", trace=" << (std::filesystem::path(cfg.output_dir) / "replay_trace.csv").string();
    summary.notes = note.str();
    return summary;
}

std::string summary_line(const std::string& label, const EvaluationSummary& s) {
    auto fmt = [](const double value, const int digits) {
        if (!std::isfinite(value)) return std::string("n/a");
        std::ostringstream os;
        os << std::fixed << std::setprecision(digits) << value;
        return os.str();
    };
    std::ostringstream oss;
    oss << label << ": survival=" << fmt(s.survival_ratio, 3)
        << " mean_ticks=" << fmt(s.mean_ticks, 1)
        << " success=" << fmt(s.success_rate, 3)
        << " force=" << fmt(s.mean_force_abs, 3)
        << " energy=" << fmt(s.mean_energy, 3);
    return oss.str();
}

double odd_relative_importance(const std::vector<ODDDeltaSummary>& items, const std::size_t index) {
    if (index >= items.size()) return 0.0;
    double denom = 0.0;
    for (const auto& item : items) denom += std::abs(item.delta_survival_ratio);
    if (denom <= 1e-12) return 0.0;
    return std::abs(items[index].delta_survival_ratio) / denom;
}

std::string json_escape(const std::string& input) {
    std::ostringstream out;
    for (const char ch : input) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

bool write_decoded_genome_json(const Genome& genome,
                               const CandidateParams& params,
                               const std::string& path) {
    try {
        namespace fs = std::filesystem;
        fs::create_directories(fs::path(path).parent_path());
        const DecodedGenome d = decode_genome(genome);
        std::ofstream json(path);
        json << std::fixed << std::setprecision(6);
        json << "{\n";
        json << "  \"candidate_params\": {\n";
        json << "    \"body_mode\": \"" << json_escape(params.body_mode) << "\",\n";
        json << "    \"development_steps\": " << params.development_steps << ",\n";
        json << "    \"development_seed_half_width\": " << params.development_seed_half_width << ",\n";
        json << "    \"max_cells\": " << params.max_cells << ",\n";
        json << "    \"body_extent_x\": " << params.body_extent_x << ",\n";
        json << "    \"body_extent_y\": " << params.body_extent_y << ",\n";
        json << "    \"body_extent_z\": " << params.body_extent_z << ",\n";
        json << "    \"chemical_diffusion_steps\": " << params.chemical_diffusion_steps << ",\n";
        json << "    \"development_growth_threshold\": " << params.development_growth_threshold << ",\n";
        json << "    \"chemical_diffusion_rate\": " << params.chemical_diffusion_rate << ",\n";
        json << "    \"chemical_decay\": " << params.chemical_decay << "\n";
        json << "  },\n";
        json << "  \"decoded_scalars\": {\n";
        json << "    \"tau_v_base\": " << d.tau_v_base << ",\n";
        json << "    \"tau_v_gain\": " << d.tau_v_gain << ",\n";
        json << "    \"epsilon_base\": " << d.epsilon_base << ",\n";
        json << "    \"epsilon_gain\": " << d.epsilon_gain << ",\n";
        json << "    \"fhn_a\": " << d.fhn_a << ",\n";
        json << "    \"fhn_b\": " << d.fhn_b << ",\n";
        json << "    \"tau_ca\": " << d.tau_ca << ",\n";
        json << "    \"alpha_ca\": " << d.alpha_ca << ",\n";
        json << "    \"v_ca_thresh\": " << d.v_ca_thresh << ",\n";
        json << "    \"coupling_base\": " << d.coupling_base << ",\n";
        json << "    \"hebb_rate\": " << d.hebb_rate << ",\n";
        json << "    \"decay_rate\": " << d.decay_rate << ",\n";
        json << "    \"adhesion_coeff\": " << d.adhesion_coeff << ",\n";
        json << "    \"chem_coeff\": " << d.chem_coeff << ",\n";
        json << "    \"mech_base\": " << d.mech_base << ",\n";
        json << "    \"sigma_target\": " << d.sigma_target << ",\n";
        json << "    \"sigma_alpha\": " << d.sigma_alpha << ",\n";
        json << "    \"sigma_beta\": " << d.sigma_beta << ",\n";
        json << "    \"energy_replenish\": " << d.energy_replenish << ",\n";
        json << "    \"cost_vm\": " << d.cost_vm << ",\n";
        json << "    \"cost_ca\": " << d.cost_ca << ",\n";
        json << "    \"cost_mech\": " << d.cost_mech << ",\n";
        json << "    \"g_drift_rate\": " << d.g_drift_rate << ",\n";
        json << "    \"force_scale\": " << d.force_scale << ",\n";
        json << "    \"stress_scale\": " << d.stress_scale << ",\n";
        json << "    \"teacher_frac\": " << d.teacher_frac << ",\n";
        json << "    \"teacher_force_scale\": " << d.teacher_force_scale << ",\n";
        json << "    \"teacher_hebb_boost\": " << d.teacher_hebb_boost << ",\n";
        json << "    \"stiffness_softness\": " << d.stiffness_softness << ",\n";
        json << "    \"basal_contractility\": " << d.basal_contractility << ",\n";
        json << "    \"activity_decay\": " << d.activity_decay << ",\n";
        json << "    \"sensor_threshold\": " << d.sensor_threshold << ",\n";
        json << "    \"teacher_kp\": " << d.teacher_kp << ",\n";
        json << "    \"teacher_kd\": " << d.teacher_kd << ",\n";
        json << "    \"chemical_balance\": " << d.chemical_balance << ",\n";
        json << "    \"muscle_bias\": " << d.muscle_bias << ",\n";
        json << "    \"mech_filter\": " << d.mech_filter << ",\n";
        json << "    \"energy_output_scale\": " << d.energy_output_scale << ",\n";
        json << "    \"recovery_reset\": " << d.recovery_reset << ",\n";
        json << "    \"trial_fast_decay\": " << d.trial_fast_decay << ",\n";
        json << "    \"damage_resilience\": " << d.damage_resilience << ",\n";
        json << "    \"force_smoothing\": " << d.force_smoothing << ",\n";
        json << "    \"stress_smoothing\": " << d.stress_smoothing << "\n";
        json << "  },\n";
        json << "  \"bias\": [";
        for (int gidx = 0; gidx < kGeneExprDim; ++gidx) {
            if (gidx != 0) json << ", ";
            json << d.bias[static_cast<std::size_t>(gidx)];
        }
        json << "],\n";
        json << "  \"W\": [\n";
        for (int gidx = 0; gidx < kGeneExprDim; ++gidx) {
            json << "    [";
            for (int f = 0; f < 5; ++f) {
                if (f != 0) json << ", ";
                json << d.W[static_cast<std::size_t>(gidx)][static_cast<std::size_t>(f)];
            }
            json << "]" << (gidx + 1 < kGeneExprDim ? "," : "") << "\n";
        }
        json << "  ]\n";
        json << "}\n";
    } catch (...) {
        return false;
    }
    return true;
}

bool write_trial_context_artifacts(const std::vector<ReplayCellState>& pre_reset_rows,
                                   const std::vector<ReplayCellState>& post_reset_rows,
                                   const bool teacher_active,
                                   const bool damage_active,
                                   const std::string& output_dir,
                                   const std::string& file_prefix) {
    namespace fs = std::filesystem;
    try {
        fs::create_directories(output_dir);
        const std::string prefix = file_prefix.empty() ? std::string() : (file_prefix + "_");
        const std::string pre_name = prefix + "trial_pre_reset_cells.csv";
        const std::string post_name = prefix + "trial_post_reset_cells.csv";
        const std::string json_name = prefix + "trial_context.json";
        if (!write_replay_cells_csv(pre_reset_rows, (fs::path(output_dir) / pre_name).string())) {
            return false;
        }
        if (!write_replay_cells_csv(post_reset_rows, (fs::path(output_dir) / post_name).string())) {
            return false;
        }
        std::ofstream json(fs::path(output_dir) / json_name);
        json << "{\n";
        json << "  \"teacher_active\": " << (teacher_active ? "true" : "false") << ",\n";
        json << "  \"damage_active\": " << (damage_active ? "true" : "false") << ",\n";
        json << "  \"pre_reset_snapshot\": \"" << pre_name << "\",\n";
        json << "  \"post_reset_snapshot\": \"" << post_name << "\"\n";
        json << "}\n";
    } catch (...) {
        return false;
    }
    return true;
}

bool write_named_trial_context_bundle(const Genome& genome,
                                      const CellEngineConfig& cfg,
                                      const bool teacher_active,
                                      const bool damage_active,
                                      const int seed_offset,
                                      const std::string& output_dir,
                                      const std::string& file_prefix) {
    const CellEngineConfig resolved_cfg = resolve_loaded_genome_cfg(cfg, genome);
    const BodyTemplate body = make_body_template(genome, resolved_cfg);
    const DecodedGenome d = decode_genome(genome);
    CellPopulation pop = build_population(genome, body);
    std::mt19937 rng(static_cast<unsigned int>(resolved_cfg.seed + seed_offset));
    if (damage_active) {
        apply_damage(pop, body, resolved_cfg.damage_fraction, rng);
    }
    const CellPopulation pre_reset_pop = pop;
    TaskEnv env;
    env.kind = parse_task_kind(resolved_cfg.task_name);
    const TrialBudget base_budget = damage_active ? resolved_cfg.damage_budget : resolved_cfg.final_budget;
    const TrialBudget context_budget{1, std::max(1, base_budget.max_ticks)};
    auto [stats, frames, cell_rows, edge_rows] =
        run_trial_with_trace(pop, body, d, resolved_cfg, context_budget, rng, teacher_active, env);
    (void)stats;
    (void)frames;
    (void)edge_rows;
    std::vector<ReplayCellState> pre_reset_rows;
    append_replay_cell_state(pre_reset_pop, body, d, 0, nullptr, pre_reset_rows);
    std::vector<ReplayCellState> post_reset_rows;
    const std::size_t snapshot_n = std::min<std::size_t>(static_cast<std::size_t>(body.cell_count()), cell_rows.size());
    post_reset_rows.insert(post_reset_rows.end(), cell_rows.begin(), cell_rows.begin() + static_cast<std::ptrdiff_t>(snapshot_n));
    return write_trial_context_artifacts(pre_reset_rows,
                                         post_reset_rows,
                                         teacher_active,
                                         damage_active,
                                         output_dir,
                                         file_prefix);
}

