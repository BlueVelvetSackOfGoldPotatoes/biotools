EvaluationSummary summarize_trials(const std::vector<TrialStats>& trials, const TrialBudget& budget) {
    EvaluationSummary s;
    s.max_ticks = static_cast<double>(budget.num_trials * budget.max_ticks);
    if (trials.empty() || s.max_ticks <= 0.0) return s;
    int successes = 0;
    double total_force = 0.0;
    double total_energy = 0.0;
    double total_task_primary = 0.0;
    double total_task_secondary = 0.0;
    for (const auto& t : trials) {
        s.total_ticks += static_cast<double>(t.ticks);
        total_force += t.mean_abs_force;
        total_energy += t.mean_energy;
        total_task_primary += t.task_primary;
        total_task_secondary += t.task_secondary;
        successes += t.success ? 1 : 0;
    }
    s.survival_ratio = s.total_ticks / s.max_ticks;
    s.mean_ticks = s.total_ticks / static_cast<double>(trials.size());
    s.success_rate = static_cast<double>(successes) / static_cast<double>(trials.size());
    s.mean_force_abs = total_force / static_cast<double>(trials.size());
    s.mean_energy = total_energy / static_cast<double>(trials.size());
    s.task_primary = total_task_primary / static_cast<double>(trials.size());
    s.task_secondary = total_task_secondary / static_cast<double>(trials.size());

    const std::size_t quarter = std::max<std::size_t>(1, trials.size() / 4);
    for (std::size_t i = 0; i < quarter; ++i) s.early_mean_ticks += trials[i].ticks;
    for (std::size_t i = trials.size() - quarter; i < trials.size(); ++i) s.late_mean_ticks += trials[i].ticks;
    s.early_mean_ticks /= static_cast<double>(quarter);
    s.late_mean_ticks /= static_cast<double>(quarter);
    s.adaptation_gain = clampd((s.late_mean_ticks - s.early_mean_ticks) / static_cast<double>(budget.max_ticks), -1.0, 1.0);
    return s;
}

void annotate_task_summary(EvaluationSummary& summary, const TaskKind kind) {
    summary.task_primary_label = task_primary_label(kind);
    summary.task_secondary_label = task_secondary_label(kind);
}

double task_summary_score(const EvaluationSummary& s, const TaskKind kind) {
    switch (kind) {
        case TaskKind::worm_drag_race:
            return 0.58 * clampd(s.task_primary, -1.0, 2.0) +
                   0.20 * s.success_rate +
                   0.10 * s.survival_ratio +
                   0.07 * clampd(0.5 + 0.5 * std::tanh(s.task_secondary), 0.0, 1.0) +
                   0.05 * s.mean_energy;
        case TaskKind::pong_return:
            return 0.56 * clampd(s.task_primary, 0.0, 2.0) +
                   0.20 * s.success_rate +
                   0.14 * s.survival_ratio +
                   0.05 * clampd(s.task_secondary / 8.0, 0.0, 1.0) +
                   0.05 * s.mean_energy;
        case TaskKind::mass_spring_balance:
        case TaskKind::cartpole_balance:
        default:
            return s.survival_ratio +
                   0.18 * s.sensor_responsiveness +
                   0.18 * s.signal_connectivity +
                   0.25 * s.force_correlation +
                   0.12 * clampd(0.5 + 0.5 * s.adaptation_gain, 0.0, 1.0) +
                   0.05 * s.mean_energy -
                   0.03 * s.mean_force_abs;
    }
}

double task_refinement_score(const EvaluationSummary& s, const TaskKind kind) {
    switch (kind) {
        case TaskKind::worm_drag_race:
        case TaskKind::pong_return:
            return 0.55 * clampd(s.task_primary, -1.0, 2.0) +
                   0.25 * s.success_rate +
                   0.10 * s.survival_ratio +
                   0.06 * s.mean_energy -
                   0.01 * s.mean_force_abs;
        case TaskKind::mass_spring_balance:
        case TaskKind::cartpole_balance:
        default:
            return s.survival_ratio +
                   0.20 * s.success_rate +
                   0.08 * clampd(0.5 + 0.5 * s.adaptation_gain, 0.0, 1.0) +
                   0.04 * s.mean_energy -
                   0.015 * s.mean_force_abs;
    }
}

bool task_is_solved(const EvaluationSummary& s, const TaskKind kind, const CellEngineConfig& cfg) {
    switch (kind) {
        case TaskKind::worm_drag_race:
        case TaskKind::pong_return:
            return s.task_primary >= cfg.solve_ratio && s.success_rate >= 0.55;
        case TaskKind::mass_spring_balance:
        case TaskKind::cartpole_balance:
        default:
            return s.survival_ratio >= cfg.solve_ratio;
    }
}

std::tuple<double, double, double> run_probe(const Genome& genome,
                                             const CellEngineConfig& cfg,
                                             std::mt19937& rng) {
    const CellEngineConfig resolved_cfg = resolve_candidate_cfg(cfg, genome);
    const BodyTemplate body = make_body_template(genome, resolved_cfg);
    const DecodedGenome d = decode_genome(genome);
    const double cell_dt = resolved_cfg.physics_dt / static_cast<double>(resolved_cfg.cell_ticks_per_physics);
    double sensor_resp = 0.0;
    double connectivity = 0.0;
    double force_corr = 0.0;
    int count = 0;

    for (const double sign : {-1.0, 1.0}) {
        CellPopulation pop = build_population(genome, body);
        TaskEnv env;
        configure_task_env(env, resolved_cfg, d.force_scale);
        if (env.kind == TaskKind::cartpole_balance || env.kind == TaskKind::mass_spring_balance) {
            env.reset(rng, sign * 5.0 * kPi / 180.0);
        } else if (env.kind == TaskKind::worm_drag_race) {
            env.reset(rng, sign * 0.22);
        } else {
            env.reset(rng, sign * 0.35);
        }
        env.set_teacher(false, d.teacher_kp, d.teacher_kd, d.teacher_force_scale);
        write_stress(pop, body, d, env, resolved_cfg.stress_relax_iters);

        std::vector<double> sensor_trace;
        std::vector<double> motor_trace;
        std::vector<double> force_trace;

        for (int tick = 0; tick < 60 && !env.terminal; ++tick) {
            for (int k = 0; k < resolved_cfg.cell_ticks_per_physics; ++k) {
                cell_tick(pop, body, d, resolved_cfg, cell_dt, false);
            }
            const double force_cmd = compute_force(pop, body);
            env.step(force_cmd, resolved_cfg.physics_dt);
            write_stress(pop, body, d, env, resolved_cfg.stress_relax_iters);

            double sensor_mean = 0.0;
            for (const int idx : body.sensor_cells) {
                sensor_mean += std::abs(pop.V[static_cast<std::size_t>(idx)]);
            }
            sensor_mean /= std::max<std::size_t>(1, body.sensor_cells.size());
            double motor_mean = 0.0;
            for (const int idx : body.motor_cells) {
                motor_mean += std::abs(pop.a_mech[static_cast<std::size_t>(idx)]);
            }
            motor_mean /= std::max<std::size_t>(1, body.motor_cells.size());
            sensor_trace.push_back(sensor_mean);
            motor_trace.push_back(motor_mean);
            force_trace.push_back(env.organism_force);
        }

        if (!sensor_trace.empty()) {
            const double base_sensor = sensor_trace.front();
            const double peak_sensor = *std::max_element(sensor_trace.begin(), sensor_trace.end());
            sensor_resp += std::tanh((peak_sensor - base_sensor) / std::max(0.03, d.sensor_threshold));
            connectivity += clampd(0.5 + 0.5 * corr(sensor_trace, motor_trace), 0.0, 1.0);
            force_corr += clampd(0.5 + 0.5 * sign * mean_of(force_trace) / std::max(1.0, d.force_scale), 0.0, 1.0);
            count += 1;
        }
    }

    if (count == 0) return {0.0, 0.0, 0.0};
    return {sensor_resp / static_cast<double>(count), connectivity / static_cast<double>(count), force_corr / static_cast<double>(count)};
}

CellEvalResult evaluate_genome(const Genome& genome,
                               const CellEngineConfig& cfg,
                               const TrialBudget& budget,
                               const int eval_seed,
                               const bool with_damage,
                               const bool include_probe,
                               const ODDIntervention* intervention) {
    const CellEngineConfig resolved_cfg = resolve_candidate_cfg(cfg, genome);
    const BodyTemplate body = make_body_template(genome, resolved_cfg);
    const DecodedGenome d = decode_genome(genome);
    CellPopulation pop = build_population(genome, body, intervention);
    std::mt19937 rng(static_cast<unsigned int>(eval_seed));
    TaskEnv env;
    env.kind = parse_task_kind(resolved_cfg.task_name);
    std::vector<TrialStats> trials;
    trials.reserve(static_cast<std::size_t>(budget.num_trials));

    bool damage_applied = false;
    const int teaching_trials =
        (intervention && intervention->disable_teacher)
            ? 0
            : clampd(std::round(d.teacher_frac * budget.num_trials), 0.0, static_cast<double>(budget.num_trials));

    for (int trial = 0; trial < budget.num_trials; ++trial) {
        if (with_damage && !damage_applied && trial == cfg.damage_after_trial) {
            apply_damage(pop, body, resolved_cfg.damage_fraction, rng);
            damage_applied = true;
        }
        const bool teaching_active = trial < teaching_trials;
        trials.push_back(run_trial(pop, body, d, resolved_cfg, budget, rng, teaching_active, env, intervention));
    }

    CellEvalResult result;
    result.summary = summarize_trials(trials, budget);
    annotate_task_summary(result.summary, parse_task_kind(resolved_cfg.task_name));
    if (include_probe) {
        auto [sensor_resp, connectivity, force_corr] = run_probe(genome, resolved_cfg, rng);
        result.summary.sensor_responsiveness = sensor_resp;
        result.summary.signal_connectivity = connectivity;
        result.summary.force_correlation = force_corr;
    }
    result.score = task_summary_score(result.summary, parse_task_kind(resolved_cfg.task_name)) -
                   0.03 * (result.summary.mean_force_abs / std::max(1.0, d.force_scale));
    return result;
}

void damage_agent_parameters(std::vector<Tensor*> params, const double frac, std::mt19937& rng);

void mutate(Genome& genome, const CellEngineConfig& cfg, const double rate, const double sigma, std::mt19937& rng) {
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::normal_distribution<double> noise(0.0, sigma);
    for (double& g : genome.genes) {
        if (u01(rng) < rate) g = clampd(g + noise(rng), -1.0, 1.0);
    }
    auto mutate_int = [&](int& value, const int lo, const int hi, const int step = 1) {
        if (u01(rng) >= rate || hi <= lo) return;
        std::uniform_int_distribution<int> delta_pick(-step, step);
        value = std::max(lo, std::min(hi, value + delta_pick(rng)));
    };
    auto mutate_double = [&](double& value, const double lo, const double hi, const double local_sigma) {
        if (u01(rng) >= rate || hi <= lo) return;
        std::normal_distribution<double> local_noise(0.0, local_sigma);
        value = clampd(value + local_noise(rng), lo, hi);
    };
    if (cfg.evolve_body_mode && u01(rng) < rate) {
        const int max_code = cfg.body_extent_z > 0 ? 2 : 1;
        std::uniform_int_distribution<int> mode_pick(0, max_code);
        genome.params.body_mode = body_mode_from_code(mode_pick(rng));
    }
    if (cfg.evolve_development_steps) mutate_int(genome.params.development_steps, 1, std::max(1, cfg.development_steps), 2);
    if (cfg.evolve_development_seed_half_width) mutate_int(genome.params.development_seed_half_width, 1, std::max(1, cfg.development_seed_half_width), 1);
    if (cfg.evolve_max_cells) mutate_int(genome.params.max_cells, 18, std::max(18, cfg.max_cells), 10);
    if (cfg.evolve_body_extent_x) mutate_int(genome.params.body_extent_x, 2, std::max(2, cfg.body_extent_x), 1);
    if (cfg.evolve_body_extent_y) mutate_int(genome.params.body_extent_y, 4, std::max(4, cfg.body_extent_y), 2);
    if (cfg.evolve_body_extent_z) mutate_int(genome.params.body_extent_z, 0, std::max(0, cfg.body_extent_z), 1);
    if (cfg.evolve_chemical_diffusion_steps) {
        mutate_int(genome.params.chemical_diffusion_steps, 1, std::max(1, cfg.chemical_diffusion_steps), 1);
    }
    if (cfg.evolve_growth_threshold) mutate_double(genome.params.development_growth_threshold, 0.15, 0.90, 0.04);
    if (cfg.evolve_chemical_diffusion_rate) mutate_double(genome.params.chemical_diffusion_rate, 0.05, 0.60, 0.03);
    if (cfg.evolve_chemical_decay) mutate_double(genome.params.chemical_decay, 0.0, 0.25, 0.02);
}

Genome crossover(const Genome& a, const Genome& b, const CellEngineConfig& cfg, const double rate, std::mt19937& rng) {
    Genome out = a;
    out.has_params = a.has_params || b.has_params;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (std::size_t i = 0; i < out.genes.size(); ++i) {
        if (u01(rng) < rate) out.genes[i] = b.genes[i];
    }
    if (cfg.evolve_body_mode && u01(rng) < 0.5) out.params.body_mode = b.params.body_mode;
    if (cfg.evolve_development_steps && u01(rng) < 0.5) out.params.development_steps = b.params.development_steps;
    if (cfg.evolve_development_seed_half_width && u01(rng) < 0.5) out.params.development_seed_half_width = b.params.development_seed_half_width;
    if (cfg.evolve_max_cells && u01(rng) < 0.5) out.params.max_cells = b.params.max_cells;
    if (cfg.evolve_body_extent_x && u01(rng) < 0.5) out.params.body_extent_x = b.params.body_extent_x;
    if (cfg.evolve_body_extent_y && u01(rng) < 0.5) out.params.body_extent_y = b.params.body_extent_y;
    if (cfg.evolve_body_extent_z && u01(rng) < 0.5) out.params.body_extent_z = b.params.body_extent_z;
    if (cfg.evolve_chemical_diffusion_steps && u01(rng) < 0.5) out.params.chemical_diffusion_steps = b.params.chemical_diffusion_steps;
    if (cfg.evolve_growth_threshold && u01(rng) < 0.5) out.params.development_growth_threshold = b.params.development_growth_threshold;
    if (cfg.evolve_chemical_diffusion_rate && u01(rng) < 0.5) out.params.chemical_diffusion_rate = b.params.chemical_diffusion_rate;
    if (cfg.evolve_chemical_decay && u01(rng) < 0.5) out.params.chemical_decay = b.params.chemical_decay;
    return out;
}

int tournament_pick(const std::vector<double>& fitness,
                    const int tournament_size,
                    std::mt19937& rng) {
    std::uniform_int_distribution<int> pick(0, static_cast<int>(fitness.size()) - 1);
    int best = pick(rng);
    for (int i = 1; i < tournament_size; ++i) {
        const int idx = pick(rng);
        if (fitness[static_cast<std::size_t>(idx)] > fitness[static_cast<std::size_t>(best)]) best = idx;
    }
    return best;
}

Genome local_refine(const Genome& start,
                    const CellEngineConfig& cfg,
                    const int seed,
                    double& best_score) {
    Genome best = start;
    std::mt19937 rng(static_cast<unsigned int>(seed));
    for (int step = 0; step < 36; ++step) {
        Genome cand = best;
        mutate(cand, cfg, 0.22, std::max(0.03, cfg.mutation_sigma * 0.55), rng);
        const auto eval = evaluate_genome(cand, cfg, cfg.search_budget, seed + 100 + step, false, true);
        if (eval.score > best_score) {
            best = cand;
            best_score = eval.score;
        }
    }
    return best;
}

Genome long_horizon_refine(const Genome& start,
                           const CellEngineConfig& cfg,
                           const int seed) {
    TrialBudget budget = cfg.final_budget;
    budget.num_trials = std::min(std::max(12, cfg.search_budget.num_trials), cfg.final_budget.num_trials);
    budget.max_ticks = std::min(std::max(300, cfg.search_budget.max_ticks + 120), cfg.final_budget.max_ticks);

    Genome best = start;
    std::mt19937 rng(static_cast<unsigned int>(seed));
    EvaluationSummary best_summary = evaluate_genome(best, cfg, budget, seed + 17000, false, true).summary;
    const TaskKind task_kind = parse_task_kind(cfg.task_name);
    double best_long_score = task_refinement_score(best_summary, task_kind);

    for (int step = 0; step < 64; ++step) {
        Genome cand = best;
        const double sigma = (step < 24) ? std::max(0.02, cfg.mutation_sigma * 0.45)
                                         : std::max(0.01, cfg.mutation_sigma * 0.25);
        const double rate = (step < 24) ? 0.20 : 0.10;
        mutate(cand, cfg, rate, sigma, rng);
        const EvaluationSummary cand_summary =
            evaluate_genome(cand, cfg, budget, seed + 17100 + step * 13, false, true).summary;
        const double cand_score = task_refinement_score(cand_summary, task_kind);
        if (cand_score > best_long_score) {
            best = cand;
            best_summary = cand_summary;
            best_long_score = cand_score;
        }
    }
    return best;
}

EvolutionResult evolve_cells(const CellEngineConfig& cfg, const int attempt_seed) {
    std::mt19937 rng(static_cast<unsigned int>(attempt_seed));
    std::vector<Genome> population(static_cast<std::size_t>(cfg.population_size));
    std::vector<double> fitness(static_cast<std::size_t>(cfg.population_size), 0.0);
    std::vector<EvaluationSummary> search_summaries(static_cast<std::size_t>(cfg.population_size));
    std::vector<int> order(static_cast<std::size_t>(cfg.population_size), 0);
    std::iota(order.begin(), order.end(), 0);

    for (int i = 0; i < cfg.population_size; ++i) {
        population[static_cast<std::size_t>(i)] = (i < cfg.hint_genomes) ? make_hint_genome(cfg, i) : random_genome(cfg, rng);
    }

    EvolutionResult best;
    for (int gen = 0; gen < cfg.generations; ++gen) {
        #pragma omp parallel for schedule(dynamic) if(cfg.population_size > 8)
        for (int i = 0; i < cfg.population_size; ++i) {
            const int eval_seed = attempt_seed + gen * 997 + i * 17;
            const CellEvalResult eval =
                evaluate_genome(population[static_cast<std::size_t>(i)], cfg, cfg.search_budget, eval_seed, false, true);
            fitness[static_cast<std::size_t>(i)] = eval.score;
            search_summaries[static_cast<std::size_t>(i)] = eval.summary;
        }

        std::sort(order.begin(), order.end(), [&](const int a, const int b) {
            return fitness[static_cast<std::size_t>(a)] > fitness[static_cast<std::size_t>(b)];
        });

        const int current_best_idx = order.front();
        const double current_best = fitness[static_cast<std::size_t>(current_best_idx)];
        if (current_best > best.best_search_score) {
            best.best_search_score = current_best;
            best.best_genome = population[static_cast<std::size_t>(current_best_idx)];
            best.best_generation = gen + 1;
        }

        if (!cfg.output_dir.empty() && best.best_search_score >= 0.0) {
            write_live_discovery_snapshot(best.best_genome,
                                          cfg,
                                          gen + 1,
                                          best.best_generation,
                                          best.best_search_score,
                                          &search_summaries[static_cast<std::size_t>(current_best_idx)],
                                          cfg.output_dir);
            write_live_discovery_population_snapshot(population,
                                                     fitness,
                                                     search_summaries,
                                                     order,
                                                     cfg,
                                                     gen + 1,
                                                     best.best_generation,
                                                     best.best_search_score,
                                                     cfg.output_dir);
        }

        std::vector<Genome> next_pop;
        next_pop.reserve(static_cast<std::size_t>(cfg.population_size));
        for (int e = 0; e < cfg.elitism && e < cfg.population_size; ++e) {
            next_pop.push_back(population[static_cast<std::size_t>(order[static_cast<std::size_t>(e)])]);
        }

        const double sigma = std::max(0.03, cfg.mutation_sigma * (0.75 + 0.25 * (1.0 - static_cast<double>(gen) / std::max(1, cfg.generations))));
        while (static_cast<int>(next_pop.size()) < cfg.population_size) {
            const int ia = tournament_pick(fitness, cfg.tournament_size, rng);
            const int ib = tournament_pick(fitness, cfg.tournament_size, rng);
            Genome child = crossover(population[static_cast<std::size_t>(ia)], population[static_cast<std::size_t>(ib)], cfg, cfg.crossover_rate, rng);
            mutate(child, cfg, cfg.mutation_rate, sigma, rng);
            next_pop.push_back(child);
        }
        population.swap(next_pop);
    }

    best.best_genome = local_refine(best.best_genome, cfg, attempt_seed + 7000, best.best_search_score);
    best.best_genome = long_horizon_refine(best.best_genome, cfg, attempt_seed + 9000);
    const CellEvalResult refined_eval =
        evaluate_genome(best.best_genome, cfg, cfg.search_budget, attempt_seed + 9500, false, true);
    best.best_search_score = refined_eval.score;
    if (!cfg.output_dir.empty() && best.best_search_score >= 0.0) {
        write_live_discovery_snapshot(best.best_genome,
                                      cfg,
                                      cfg.generations,
                                      best.best_generation,
                                      best.best_search_score,
                                      &refined_eval.summary,
                                      cfg.output_dir);
        write_live_discovery_population_snapshot(population,
                                                 fitness,
                                                 search_summaries,
                                                 order,
                                                 cfg,
                                                 cfg.generations,
                                                 best.best_generation,
                                                 best.best_search_score,
                                                 cfg.output_dir);
    }
    return best;
}

EvaluationSummary evaluate_pd(const CellEngineConfig& cfg, const TrialBudget& budget, const int seed) {
    std::mt19937 rng(static_cast<unsigned int>(seed));
    std::vector<TrialStats> trials;
    trials.reserve(static_cast<std::size_t>(budget.num_trials));
    for (int t = 0; t < budget.num_trials; ++t) {
        TaskEnv env;
        configure_task_env(env, cfg, 18.0);
        env.reset(rng);
        TrialStats stats;
        while (!env.terminal && env.ticks < budget.max_ticks) {
            const double force = env.teacher_control_force();
            env.set_teacher(false, 0.0, 0.0, 0.0);
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
    return out;
}

EvaluationSummary evaluate_dqn_agent(DQNAgent& agent,
                                     const CellEngineConfig& cfg,
                                     const TrialBudget& budget,
                                     const int seed) {
    std::mt19937 rng(static_cast<unsigned int>(seed));
    std::vector<TrialStats> trials;
    trials.reserve(static_cast<std::size_t>(budget.num_trials));
    const double force_mag = cfg.rl_force_mag;
    for (int t = 0; t < budget.num_trials; ++t) {
        TaskEnv env;
        configure_task_env(env, cfg, force_mag);
        env.reset(rng);
        TrialStats stats;
        while (!env.terminal && env.ticks < budget.max_ticks) {
            Tensor st = make_state_tensor(env);
            const int act = agent.predict_actions(st).front();
            const double force = (act == 0) ? -force_mag : force_mag;
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

std::array<double, 2> softmax2(const Tensor& logits, const std::size_t row) {
    const double a = logits(row, 0);
    const double b = logits(row, 1);
    const double m = std::max(a, b);
    const double ea = std::exp(a - m);
    const double eb = std::exp(b - m);
    const double z = ea + eb;
    return {ea / z, eb / z};
}

int sample_action(const std::array<double, 2>& probs, std::mt19937& rng) {
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    return (u01(rng) < probs[0]) ? 0 : 1;
}

EvaluationSummary summarize_single_trial(const TrialStats& stats, const TrialBudget& budget) {
    return summarize_trials(std::vector<TrialStats>{stats}, budget);
}

ODDDeltaSummary make_odd_delta(const std::string& key,
                               const std::string& label,
                               const EvaluationSummary& baseline,
                               const EvaluationSummary& variant) {
    ODDDeltaSummary out;
    out.key = key;
    out.label = label;
    out.baseline_survival_ratio = baseline.survival_ratio;
    out.variant_survival_ratio = variant.survival_ratio;
    out.delta_survival_ratio = baseline.survival_ratio - variant.survival_ratio;
    out.baseline_success_rate = baseline.success_rate;
    out.variant_success_rate = variant.success_rate;
    out.delta_success_rate = baseline.success_rate - variant.success_rate;
    return out;
}

void append_correlation(std::vector<ODDCorrelationSummary>& out,
                        const std::string& left,
                        const std::string& right,
                        const std::vector<double>& a,
                        const std::vector<double>& b) {
    out.push_back({left, right, corr(a, b)});
}

std::vector<ODDCorrelationSummary> top_abs_correlations(std::vector<ODDCorrelationSummary> items, const std::size_t limit) {
    std::sort(items.begin(), items.end(), [](const ODDCorrelationSummary& a, const ODDCorrelationSummary& b) {
        return std::abs(a.correlation) > std::abs(b.correlation);
    });
    if (items.size() > limit) items.resize(limit);
    return items;
}

ODDSummary run_odd_experiment(const Genome& genome,
                              const CellEngineConfig& cfg,
                              const int seed) {
    ODDSummary odd;
    if (!cfg.odd_enabled) {
        odd.notes = "disabled";
        return odd;
    }

    odd.available = true;
    odd.budget = cfg.odd_budget;
    odd.budget.max_ticks = std::max(1, odd.budget.max_ticks);
    odd.budget.num_trials = std::max(1, odd.budget.num_trials);

    const CellEngineConfig resolved_cfg = resolve_candidate_cfg(cfg, genome);
    const BodyTemplate body = make_body_template(genome, resolved_cfg);
    const EvaluationSummary baseline =
        evaluate_genome(genome, cfg, odd.budget, seed + 81000, false, false).summary;
    odd.baseline_clean = baseline;

    const std::vector<ODDIntervention> mechanisms = {
        {"no_electrical", "Disable electrical coupling", true, false, false, false, false, false, false, false, false, -1},
        {"no_chemical", "Disable chemical signaling", false, true, false, false, false, false, false, false, false, -1},
        {"no_diffusion", "Disable spatial chemical diffusion", false, false, true, false, false, false, false, false, false, -1},
        {"no_self_chemical", "Disable self-chemical feedback", false, false, false, true, false, false, false, false, false, -1},
        {"no_mechanical", "Disable mechanical sensing", false, false, false, false, true, false, false, false, false, -1},
        {"freeze_sigma", "Freeze sigma homeostasis", false, false, false, false, false, true, false, false, false, -1},
        {"freeze_gap", "Freeze gap plasticity", false, false, false, false, false, false, true, false, false, -1},
        {"no_slow_drift", "Disable slow gene drift", false, false, false, false, false, false, false, true, false, -1},
        {"no_teacher", "Disable teaching trials", false, false, false, false, false, false, false, false, true, -1}
    };
    for (std::size_t i = 0; i < mechanisms.size(); ++i) {
        const EvaluationSummary variant =
            evaluate_genome(genome, cfg, odd.budget, seed + 81100 + static_cast<int>(i) * 17, false, false, &mechanisms[i]).summary;
        odd.mechanism_deltas.push_back(make_odd_delta(mechanisms[i].key, mechanisms[i].label, baseline, variant));
    }
    std::sort(odd.mechanism_deltas.begin(), odd.mechanism_deltas.end(), [](const ODDDeltaSummary& a, const ODDDeltaSummary& b) {
        return std::abs(a.delta_survival_ratio) > std::abs(b.delta_survival_ratio);
    });

    for (int gidx = 0; gidx < kGeneExprDim; ++gidx) {
        ODDIntervention zero_intervention;
        zero_intervention.key = gene_channel_name(gidx) + "_zero";
        zero_intervention.label = "Neutralize " + gene_channel_name(gidx);
        zero_intervention.neutralize_gene_channel = gidx;
        const EvaluationSummary zero_variant =
            evaluate_genome(genome, cfg, odd.budget, seed + 81300 + gidx * 29, false, false, &zero_intervention).summary;
        odd.gene_channel_deltas.push_back(make_odd_delta(zero_intervention.key, zero_intervention.label, baseline, zero_variant));

        ODDIntervention freeze_intervention;
        freeze_intervention.key = gene_channel_name(gidx) + "_freeze";
        freeze_intervention.label = "Freeze " + gene_channel_name(gidx);
        freeze_intervention.freeze_gene_channel = gidx;
        const EvaluationSummary freeze_variant =
            evaluate_genome(genome, cfg, odd.budget, seed + 81301 + gidx * 29, false, false, &freeze_intervention).summary;
        odd.gene_channel_deltas.push_back(make_odd_delta(freeze_intervention.key, freeze_intervention.label, baseline, freeze_variant));

        ODDIntervention perturb_intervention;
        perturb_intervention.key = gene_channel_name(gidx) + "_perturb";
        perturb_intervention.label = "Perturb " + gene_channel_name(gidx);
        perturb_intervention.perturb_gene_channel = gidx;
        perturb_intervention.perturb_gene_delta = 0.10;
        const EvaluationSummary perturb_variant =
            evaluate_genome(genome, cfg, odd.budget, seed + 81302 + gidx * 29, false, false, &perturb_intervention).summary;
        odd.gene_channel_deltas.push_back(make_odd_delta(perturb_intervention.key, perturb_intervention.label, baseline, perturb_variant));
    }
    CellEngineConfig fixed_body_cfg = cfg;
    fixed_body_cfg.body_mode = "fixed2d";
    fixed_body_cfg.evolve_body_mode = false;
    fixed_body_cfg.evolve_max_cells = false;
    fixed_body_cfg.evolve_body_extent_x = false;
    fixed_body_cfg.evolve_body_extent_y = false;
    fixed_body_cfg.evolve_body_extent_z = false;
    fixed_body_cfg.evolve_development_steps = false;
    fixed_body_cfg.evolve_development_seed_half_width = false;
    fixed_body_cfg.evolve_growth_threshold = false;
    const EvaluationSummary fixed_body_variant =
        evaluate_genome(genome, fixed_body_cfg, odd.budget, seed + 81470, false, false).summary;
    odd.mechanism_deltas.push_back(
        make_odd_delta("fixed2d_body", "Force legacy fixed 2D body", baseline, fixed_body_variant));
    std::sort(odd.mechanism_deltas.begin(), odd.mechanism_deltas.end(), [](const ODDDeltaSummary& a, const ODDDeltaSummary& b) {
        return std::abs(a.delta_survival_ratio) > std::abs(b.delta_survival_ratio);
    });
    std::sort(odd.gene_channel_deltas.begin(), odd.gene_channel_deltas.end(), [](const ODDDeltaSummary& a, const ODDDeltaSummary& b) {
        return std::abs(a.delta_survival_ratio) > std::abs(b.delta_survival_ratio);
    });

    for (int cell_id = 0; cell_id < body.cell_count(); ++cell_id) {
        ODDIntervention cell_intervention;
        cell_intervention.key = "cell_force_" + std::to_string(cell_id);
        cell_intervention.label = "Ablate force contribution of cell " + std::to_string(cell_id);
        cell_intervention.ablate_force_cell = cell_id;
        const EvaluationSummary variant =
            evaluate_genome(genome, cfg, odd.budget, seed + 81500 + cell_id * 7, false, false, &cell_intervention).summary;
        odd.cell_force_deltas.push_back(make_odd_delta(cell_intervention.key, cell_intervention.label, baseline, variant));
    }
    std::sort(odd.cell_force_deltas.begin(), odd.cell_force_deltas.end(), [](const ODDDeltaSummary& a, const ODDDeltaSummary& b) {
        return std::abs(a.delta_survival_ratio) > std::abs(b.delta_survival_ratio);
    });

    const CellPopulation baseline_pop_for_edges = build_population(genome, body);
    struct EdgeCandidate {
        int src = -1;
        int dst = -1;
        double score = 0.0;
    };
    std::vector<EdgeCandidate> edge_candidates;
    for (int i = 0; i < body.cell_count(); ++i) {
        for (int p = body.neighbor_offsets[static_cast<std::size_t>(i)];
             p < body.neighbor_offsets[static_cast<std::size_t>(i + 1)];
             ++p) {
            const int j = body.neighbor_flat[static_cast<std::size_t>(p)];
            if (i >= j) continue;
            double reverse_base = 0.0;
            const int rev_begin = body.neighbor_offsets[static_cast<std::size_t>(j)];
            const int rev_end = body.neighbor_offsets[static_cast<std::size_t>(j + 1)];
            for (int q = rev_begin; q < rev_end; ++q) {
                if (body.neighbor_flat[static_cast<std::size_t>(q)] == i) {
                    reverse_base = baseline_pop_for_edges.gap_base[static_cast<std::size_t>(q)];
                    break;
                }
            }
            const double forward_base = baseline_pop_for_edges.gap_base[static_cast<std::size_t>(p)];
            edge_candidates.push_back({i, j, 0.5 * (forward_base + reverse_base)});
        }
    }
    std::sort(edge_candidates.begin(), edge_candidates.end(), [](const EdgeCandidate& a, const EdgeCandidate& b) {
        return a.score > b.score;
    });
    const std::size_t max_edge_interventions = std::min<std::size_t>(32, edge_candidates.size());
    for (std::size_t idx = 0; idx < max_edge_interventions; ++idx) {
        const auto& edge = edge_candidates[idx];
        ODDIntervention edge_intervention;
        edge_intervention.key = "edge_" + std::to_string(edge.src) + "_" + std::to_string(edge.dst);
        edge_intervention.label = "Ablate edge " + std::to_string(edge.src) + "-" + std::to_string(edge.dst);
        edge_intervention.ablate_edge_src = edge.src;
        edge_intervention.ablate_edge_dst = edge.dst;
        const EvaluationSummary variant =
            evaluate_genome(genome, cfg, odd.budget, seed + 82000 + static_cast<int>(idx) * 11, false, false, &edge_intervention).summary;
        odd.edge_deltas.push_back(make_odd_delta(edge_intervention.key, edge_intervention.label, baseline, variant));
    }
    std::sort(odd.edge_deltas.begin(), odd.edge_deltas.end(), [](const ODDDeltaSummary& a, const ODDDeltaSummary& b) {
        return std::abs(a.delta_survival_ratio) > std::abs(b.delta_survival_ratio);
    });

    const DecodedGenome d = decode_genome(genome);
    CellPopulation traced_pop = build_population(genome, body);
    std::mt19937 trace_rng(static_cast<unsigned int>(seed + 81500));
    TaskEnv trace_env;
    configure_task_env(trace_env, resolved_cfg, d.force_scale);
    const TrialBudget trace_budget{1, odd.budget.max_ticks};
    auto [trace_stats, frames, cell_rows, edge_rows] =
        run_trial_with_trace(traced_pop, body, d, resolved_cfg, trace_budget, trace_rng, false, trace_env, nullptr);
    (void)trace_stats;
    (void)cell_rows;
    (void)edge_rows;

    std::vector<double> abs_theta;
    std::vector<double> abs_force;
    std::vector<double> mean_stress;
    std::vector<double> mean_activity;
    std::vector<double> mean_calcium;
    std::vector<double> mean_energy;
    std::vector<double> active_fraction;
    abs_theta.reserve(frames.size());
    abs_force.reserve(frames.size());
    mean_stress.reserve(frames.size());
    mean_activity.reserve(frames.size());
    mean_calcium.reserve(frames.size());
    mean_energy.reserve(frames.size());
    active_fraction.reserve(frames.size());
    for (const auto& frame : frames) {
        abs_theta.push_back(std::abs(frame.theta * 180.0 / kPi));
        abs_force.push_back(std::abs(frame.organism_force));
        mean_stress.push_back(frame.mean_stress);
        mean_activity.push_back(frame.mean_activity);
        mean_calcium.push_back(frame.mean_calcium);
        mean_energy.push_back(frame.mean_energy);
        active_fraction.push_back(frame.active_fraction);
    }
    append_correlation(odd.temporal_correlations, "abs_theta_deg", "mean_stress", abs_theta, mean_stress);
    append_correlation(odd.temporal_correlations, "abs_theta_deg", "mean_activity", abs_theta, mean_activity);
    append_correlation(odd.temporal_correlations, "abs_theta_deg", "mean_calcium", abs_theta, mean_calcium);
    append_correlation(odd.temporal_correlations, "abs_theta_deg", "active_fraction", abs_theta, active_fraction);
    append_correlation(odd.temporal_correlations, "abs_force", "mean_stress", abs_force, mean_stress);
    append_correlation(odd.temporal_correlations, "abs_force", "mean_activity", abs_force, mean_activity);
    append_correlation(odd.temporal_correlations, "abs_force", "mean_calcium", abs_force, mean_calcium);
    append_correlation(odd.temporal_correlations, "mean_activity", "mean_stress", mean_activity, mean_stress);
    append_correlation(odd.temporal_correlations, "mean_energy", "abs_force", mean_energy, abs_force);
    odd.temporal_correlations = top_abs_correlations(std::move(odd.temporal_correlations), 8);

    const CellPopulation structural_pop = build_population(genome, body);
    std::vector<double> x_norm;
    std::vector<double> y_norm;
    std::vector<double> hinge_prox;
    std::vector<double> surface_prox;
    std::vector<double> neighbor_density;
    std::vector<double> force_direction_x;
    std::vector<double> mech_advantage;
    std::array<std::vector<double>, kGeneExprDim> gene_values;
    const std::size_t n_cells = body.cells.size();
    x_norm.reserve(n_cells);
    y_norm.reserve(n_cells);
    hinge_prox.reserve(n_cells);
    surface_prox.reserve(n_cells);
    neighbor_density.reserve(n_cells);
    force_direction_x.reserve(n_cells);
    mech_advantage.reserve(n_cells);
    for (std::size_t i = 0; i < n_cells; ++i) {
        const auto& cell = body.cells[i];
        x_norm.push_back(cell.features[0]);
        y_norm.push_back(cell.features[1]);
        hinge_prox.push_back(cell.features[2]);
        surface_prox.push_back(cell.features[3]);
        neighbor_density.push_back(cell.features[4]);
        force_direction_x.push_back(cell.force_direction_x);
        mech_advantage.push_back(cell.mech_advantage);
        for (int gidx = 0; gidx < kGeneExprDim; ++gidx) {
            gene_values[static_cast<std::size_t>(gidx)].push_back(
                structural_pop.G[i * static_cast<std::size_t>(kGeneExprDim) + static_cast<std::size_t>(gidx)]);
        }
    }
    const std::vector<std::pair<std::string, const std::vector<double>*>> structure_vars = {
        {"feature_x_norm", &x_norm},
        {"feature_y_norm", &y_norm},
        {"feature_hinge_prox", &hinge_prox},
        {"feature_surface_prox", &surface_prox},
        {"feature_neighbor_density", &neighbor_density},
        {"force_direction_x", &force_direction_x},
        {"mech_advantage", &mech_advantage}
    };
    for (const auto& [name, values] : structure_vars) {
        for (int gidx = 0; gidx < kGeneExprDim; ++gidx) {
            odd.structural_confounds.push_back({name, gene_channel_name(gidx), corr(*values, gene_values[static_cast<std::size_t>(gidx)])});
        }
    }
    odd.structural_confounds = top_abs_correlations(std::move(odd.structural_confounds), 10);

    std::ostringstream note;
    note << "odd_budget=" << odd.budget.num_trials << "x" << odd.budget.max_ticks
         << ", temporal correlations from single autonomous trace"
         << ", delta_sign=baseline_minus_variant"
         << ", baseline_survival=" << std::fixed << std::setprecision(4) << odd.baseline_clean.survival_ratio;
    odd.notes = note.str();
    return odd;
}
