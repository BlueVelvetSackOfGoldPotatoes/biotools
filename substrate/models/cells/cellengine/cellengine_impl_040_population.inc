DecodedGenome decode_genome(const Genome& genome) {
    DecodedGenome d;
    int gi = 0;
    for (int g = 0; g < kGeneExprDim; ++g) {
        for (int f = 0; f < 5; ++f) {
            d.W[g][f] = map_gene(genome.genes[gi++], -3.0, 3.0);
        }
    }
    for (int g = 0; g < kGeneExprDim; ++g) {
        d.bias[g] = map_gene(genome.genes[gi++], -2.2, 2.2);
    }
    d.tau_v_base = map_gene(genome.genes[gi++], 0.8, 1.4);
    d.tau_v_gain = map_gene(genome.genes[gi++], -0.6, 0.6);
    d.epsilon_base = map_gene(genome.genes[gi++], 0.04, 0.16);
    d.epsilon_gain = map_gene(genome.genes[gi++], -0.05, 0.08);
    d.fhn_a = map_gene(genome.genes[gi++], 0.55, 0.90);
    d.fhn_b = map_gene(genome.genes[gi++], 0.60, 1.20);
    d.tau_ca = map_gene(genome.genes[gi++], 0.40, 1.40);
    d.alpha_ca = map_gene(genome.genes[gi++], 0.04, 0.16);
    d.v_ca_thresh = map_gene(genome.genes[gi++], 0.00, 0.45);
    d.coupling_base = map_gene(genome.genes[gi++], 0.05, 0.55);
    d.hebb_rate = map_gene(genome.genes[gi++], 0.02, 0.18);
    d.decay_rate = map_gene(genome.genes[gi++], 0.01, 0.10);
    d.adhesion_coeff = map_gene(genome.genes[gi++], 0.08, 0.40);
    d.chem_coeff = map_gene(genome.genes[gi++], 0.00, 0.50);
    d.mech_base = map_gene(genome.genes[gi++], 0.5, 2.5);
    d.sigma_target = map_gene(genome.genes[gi++], 0.15, 0.65);
    d.sigma_alpha = map_gene(genome.genes[gi++], 0.01, 0.08);
    d.sigma_beta = map_gene(genome.genes[gi++], 0.02, 0.18);
    d.energy_replenish = map_gene(genome.genes[gi++], 0.025, 0.12);
    d.cost_vm = map_gene(genome.genes[gi++], 0.004, 0.03);
    d.cost_ca = map_gene(genome.genes[gi++], 0.002, 0.03);
    d.cost_mech = map_gene(genome.genes[gi++], 0.004, 0.035);
    d.g_drift_rate = map_gene(genome.genes[gi++], 0.0005, 0.010);
    d.force_scale = map_gene(genome.genes[gi++], 5.0, 18.0);
    d.stress_scale = map_gene(genome.genes[gi++], 0.5, 2.5);
    d.teacher_frac = map_gene(genome.genes[gi++], 0.05, 0.30);
    d.teacher_force_scale = map_gene(genome.genes[gi++], 0.7, 1.3);
    d.teacher_hebb_boost = map_gene(genome.genes[gi++], 1.0, 2.5);
    d.stiffness_softness = map_gene(genome.genes[gi++], 0.2, 1.4);
    d.basal_contractility = map_gene(genome.genes[gi++], 0.00, 0.20);
    d.activity_decay = map_gene(genome.genes[gi++], 0.80, 0.98);
    d.sensor_threshold = map_gene(genome.genes[gi++], 0.02, 0.25);
    d.teacher_kp = map_gene(genome.genes[gi++], 20.0, 110.0);
    d.teacher_kd = map_gene(genome.genes[gi++], 6.0, 32.0);
    d.chemical_balance = map_gene(genome.genes[gi++], -0.35, 0.35);
    d.muscle_bias = map_gene(genome.genes[gi++], 0.0, 0.2);
    d.mech_filter = map_gene(genome.genes[gi++], 0.15, 0.80);
    d.energy_output_scale = map_gene(genome.genes[gi++], 0.6, 1.6);
    d.recovery_reset = map_gene(genome.genes[gi++], 0.0, 0.2);
    d.trial_fast_decay = map_gene(genome.genes[gi++], 0.2, 0.8);
    d.damage_resilience = map_gene(genome.genes[gi++], 0.65, 1.0);
    d.force_smoothing = map_gene(genome.genes[gi++], 0.0, 0.75);
    d.stress_smoothing = map_gene(genome.genes[gi++], 0.0, 0.75);
    return d;
}

Genome random_genome(const CellEngineConfig& cfg, std::mt19937& rng) {
    Genome g;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (double& v : g.genes) v = dist(rng);
    g.params = default_candidate_params(cfg);
    g.has_params = true;
    std::uniform_int_distribution<int> mode_pick(0, cfg.body_extent_z > 0 ? 2 : 1);
    auto uniform_int = [&](const int lo, const int hi, const int fallback) {
        if (hi <= lo) return fallback;
        std::uniform_int_distribution<int> pick(lo, hi);
        return pick(rng);
    };
    auto uniform_double = [&](const double lo, const double hi, const double fallback) {
        if (hi <= lo) return fallback;
        std::uniform_real_distribution<double> pick(lo, hi);
        return pick(rng);
    };
    if (cfg.evolve_body_mode) g.params.body_mode = body_mode_from_code(mode_pick(rng));
    if (cfg.evolve_development_steps) g.params.development_steps = uniform_int(1, std::max(1, cfg.development_steps), cfg.development_steps);
    if (cfg.evolve_development_seed_half_width) g.params.development_seed_half_width = uniform_int(1, std::max(1, cfg.development_seed_half_width), cfg.development_seed_half_width);
    if (cfg.evolve_max_cells) g.params.max_cells = uniform_int(18, std::max(18, cfg.max_cells), cfg.max_cells);
    if (cfg.evolve_body_extent_x) g.params.body_extent_x = uniform_int(2, std::max(2, cfg.body_extent_x), cfg.body_extent_x);
    if (cfg.evolve_body_extent_y) g.params.body_extent_y = uniform_int(4, std::max(4, cfg.body_extent_y), cfg.body_extent_y);
    if (cfg.evolve_body_extent_z) g.params.body_extent_z = uniform_int(0, std::max(0, cfg.body_extent_z), cfg.body_extent_z);
    if (cfg.evolve_chemical_diffusion_steps) {
        g.params.chemical_diffusion_steps = uniform_int(1, std::max(1, cfg.chemical_diffusion_steps), cfg.chemical_diffusion_steps);
    }
    if (cfg.evolve_growth_threshold) {
        g.params.development_growth_threshold = uniform_double(0.15, 0.90, cfg.development_growth_threshold);
    }
    if (cfg.evolve_chemical_diffusion_rate) {
        g.params.chemical_diffusion_rate = uniform_double(0.05, 0.60, cfg.chemical_diffusion_rate);
    }
    if (cfg.evolve_chemical_decay) {
        g.params.chemical_decay = uniform_double(0.0, 0.25, cfg.chemical_decay);
    }
    return g;
}

bool load_genome_csv(const std::string& path, Genome& genome) {
    std::ifstream in(path);
    if (!in) return false;
    genome.genes.fill(0.0);
    genome.params = CandidateParams{};
    genome.has_params = false;
    std::string line;
    bool seen = false;
    while (std::getline(in, line)) {
        if (line.empty() || line == "gene,value" || line == "gene,value,param_name") continue;
        std::stringstream ss(line);
        std::string gene_s;
        std::string value_s;
        std::string param_name;
        if (!std::getline(ss, gene_s, ',')) continue;
        if (!std::getline(ss, value_s, ',')) continue;
        std::getline(ss, param_name);
        try {
            const int idx = std::stoi(gene_s);
            const double value = std::stod(value_s);
            if (idx >= 0 && idx < kGenomeSize) {
                genome.genes[static_cast<std::size_t>(idx)] = value;
                seen = true;
            } else if (idx < 0) {
                genome.has_params = true;
                if (param_name == "body_mode_code") genome.params.body_mode = body_mode_from_code(static_cast<int>(std::round(value)));
                else if (param_name == "development_steps") genome.params.development_steps = static_cast<int>(std::round(value));
                else if (param_name == "development_seed_half_width") genome.params.development_seed_half_width = static_cast<int>(std::round(value));
                else if (param_name == "max_cells") genome.params.max_cells = static_cast<int>(std::round(value));
                else if (param_name == "body_extent_x") genome.params.body_extent_x = static_cast<int>(std::round(value));
                else if (param_name == "body_extent_y") genome.params.body_extent_y = static_cast<int>(std::round(value));
                else if (param_name == "body_extent_z") genome.params.body_extent_z = static_cast<int>(std::round(value));
                else if (param_name == "chemical_diffusion_steps") genome.params.chemical_diffusion_steps = static_cast<int>(std::round(value));
                else if (param_name == "development_growth_threshold") genome.params.development_growth_threshold = value;
                else if (param_name == "chemical_diffusion_rate") genome.params.chemical_diffusion_rate = value;
                else if (param_name == "chemical_decay") genome.params.chemical_decay = value;
            }
        } catch (...) {
        }
    }
    return seen;
}

void set_gene(Genome& g, const int idx, const double value) {
    g.genes[static_cast<std::size_t>(idx)] = clampd(value, -1.0, 1.0);
}

Genome make_hint_genome(const CellEngineConfig& cfg, const int variant) {
    Genome g{};
    g.genes.fill(0.0);
    g.params = default_candidate_params(cfg);
    g.has_params = true;

    auto set_expr = [&](const int expr, const std::array<double, 5>& w, const double bias) {
        for (int f = 0; f < 5; ++f) {
            set_gene(g, expr * 5 + f, w[f] / 3.0);
        }
        set_gene(g, 40 + expr, bias / 2.2);
    };

    set_expr(0, {0.0, 1.8, 2.4, 0.2, 0.8}, 0.4);   // excitability
    set_expr(1, {0.0, -2.4, -1.0, 3.0, -0.4}, 0.5); // contractility
    set_expr(2, {0.0, -1.6, 1.8, 2.8, -0.2}, 0.7);  // mechanosensitivity
    set_expr(3, {0.0, 1.3, 1.8, 0.0, 1.2}, 0.0);    // secretion magnitude
    set_expr(4, {0.0, 0.0, 0.4, 0.0, 0.0}, 0.8);    // secretion flavor
    set_expr(5, {1.2, 0.0, 0.0, 0.3, 0.0}, 0.0);    // adhesion class
    set_expr(6, {0.0, 1.1, 1.8, -0.4, 1.5}, 0.4);   // coupling
    set_expr(7, {0.0, -0.2, 0.8, 1.3, 0.0}, -0.2);  // adaptation rate

    set_gene(g, 48, -0.10);
    set_gene(g, 49, 0.0);
    set_gene(g, 50, -0.25);
    set_gene(g, 51, 0.15);
    set_gene(g, 52, 0.10);
    set_gene(g, 53, -0.15);
    set_gene(g, 54, -0.10);
    set_gene(g, 55, 0.20);
    set_gene(g, 56, -0.80);
    set_gene(g, 57, -0.15);
    set_gene(g, 58, -0.10);
    set_gene(g, 59, -0.35);
    set_gene(g, 60, -0.10);
    set_gene(g, 61, -0.55);
    set_gene(g, 62, 0.35);
    set_gene(g, 63, -0.05);
    set_gene(g, 64, -0.40);
    set_gene(g, 65, -0.05);
    set_gene(g, 66, -0.25);
    set_gene(g, 67, -0.65);
    set_gene(g, 68, -0.70);
    set_gene(g, 69, -0.35);
    set_gene(g, 70, -0.60);
    set_gene(g, 71, 0.55);
    set_gene(g, 72, -0.15);
    set_gene(g, 73, -0.10);
    set_gene(g, 74, 0.0);
    set_gene(g, 75, -0.10);
    set_gene(g, 76, 0.10);
    set_gene(g, 77, -0.40);
    set_gene(g, 78, 0.30);
    set_gene(g, 79, -0.40);
    set_gene(g, 80, 0.15);
    set_gene(g, 81, -0.20);
    set_gene(g, 82, 0.0);
    set_gene(g, 83, 0.30);
    set_gene(g, 84, -0.10);
    set_gene(g, 85, 0.35);
    set_gene(g, 86, -0.30);
    set_gene(g, 87, -0.10);
    set_gene(g, 88, 0.60);
    set_gene(g, 89, -0.20);
    set_gene(g, 90, -0.20);

    const double shift = 0.10 * static_cast<double>(variant % 6 - 3);
    set_gene(g, 58, g.genes[58] + 0.15 * shift);
    set_gene(g, 71, g.genes[71] + 0.35 * shift);
    set_gene(g, 74, g.genes[74] - 0.25 * shift);
    set_gene(g, 80, g.genes[80] + 0.20 * shift);
    set_gene(g, 81, g.genes[81] - 0.15 * shift);
    set_gene(g, 49, g.genes[49] + 0.20 * shift);
    set_gene(g, 61, g.genes[61] - 0.25 * shift);
    if (cfg.evolve_max_cells) {
        g.params.max_cells = std::max(18, cfg.max_cells - std::max(0, (variant % 4) * 8));
    }
    if (cfg.evolve_development_steps) {
        g.params.development_steps = std::max(1, cfg.development_steps - (variant % 3));
    }
    if (cfg.evolve_body_extent_x) {
        g.params.body_extent_x = std::max(2, cfg.body_extent_x - (variant % 2));
    }
    if (cfg.evolve_body_extent_y) {
        g.params.body_extent_y = std::max(4, cfg.body_extent_y - (variant % 3));
    }
    if (cfg.evolve_body_extent_z) {
        g.params.body_extent_z = std::max(0, cfg.body_extent_z - (variant % 2));
    }
    if (cfg.evolve_body_mode) {
        g.params.body_mode = body_mode_from_code((variant % (cfg.body_extent_z > 0 ? 3 : 2)));
    }
    return g;
}

CellPopulation build_population(const Genome& genome, const BodyTemplate& body, const ODDIntervention* intervention) {
    const DecodedGenome d = decode_genome(genome);
    CellPopulation pop;
    pop.count = body.cell_count();
    pop.active.assign(pop.count, 1);
    pop.V.assign(pop.count, 0.0);
    pop.Wrec.assign(pop.count, 0.0);
    pop.Ca.assign(pop.count, 0.0);
    pop.sigma.assign(pop.count, 1.0);
    pop.energy.assign(pop.count, 1.0);
    pop.a_mech.assign(pop.count, 0.0);
    pop.chem_out.assign(pop.count, 0.0);
    pop.stress.assign(pop.count, 0.0);
    pop.activity_avg.assign(pop.count, 0.0);
    pop.fatigue.assign(pop.count, 0.0);
    pop.G.assign(static_cast<std::size_t>(pop.count) * kGeneExprDim, 0.0);
    pop.gap.assign(body.neighbor_flat.size(), 0.0);
    pop.gap_base.assign(body.neighbor_flat.size(), 0.0);
    pop.chem_field.assign(static_cast<std::size_t>(std::max(1, body.grid_cell_count())), 0.0);
    pop.stress_hinge.assign(pop.count, 0.0);
    pop.stress_lateral.assign(pop.count, 0.0);
    pop.stress_gravity.assign(pop.count, 0.0);
    pop.stress_drift.assign(pop.count, 0.0);
    pop.stress_neighbor.assign(pop.count, 0.0);
    pop.mech_a_gate.assign(pop.count, 0.0);
    pop.mech_contract.assign(pop.count, 0.0);
    pop.mech_prefactor.assign(pop.count, 0.0);
    pop.mech_force_contrib.assign(pop.count, 0.0);
    pop.gap_corr_last.assign(body.neighbor_flat.size(), 0.0);
    pop.gap_hebb_delta_last.assign(body.neighbor_flat.size(), 0.0);
    pop.gap_decay_delta_last.assign(body.neighbor_flat.size(), 0.0);
    pop.gap_old_last.assign(body.neighbor_flat.size(), 0.0);
    pop.gap_new_last.assign(body.neighbor_flat.size(), 0.0);

    for (int i = 0; i < pop.count; ++i) {
        for (int gidx = 0; gidx < kGeneExprDim; ++gidx) {
            double acc = d.bias[gidx];
            for (int f = 0; f < 5; ++f) {
                acc += d.W[gidx][f] * body.cells[static_cast<std::size_t>(i)].features[f];
            }
            pop.G[static_cast<std::size_t>(i) * kGeneExprDim + gidx] = sigmoid(acc);
        }
    }

    if (intervention && intervention->neutralize_gene_channel >= 0 && intervention->neutralize_gene_channel < kGeneExprDim) {
        for (int i = 0; i < pop.count; ++i) {
            pop.G[static_cast<std::size_t>(i) * kGeneExprDim + static_cast<std::size_t>(intervention->neutralize_gene_channel)] = 0.5;
        }
    }
    if (intervention && intervention->perturb_gene_channel >= 0 && intervention->perturb_gene_channel < kGeneExprDim) {
        for (int i = 0; i < pop.count; ++i) {
            const std::size_t idx =
                static_cast<std::size_t>(i) * kGeneExprDim + static_cast<std::size_t>(intervention->perturb_gene_channel);
            pop.G[idx] = clampd(pop.G[idx] + intervention->perturb_gene_delta, 0.0, 1.0);
        }
    }

    for (int i = 0; i < pop.count; ++i) {
        const int begin = body.neighbor_offsets[static_cast<std::size_t>(i)];
        const int end = body.neighbor_offsets[static_cast<std::size_t>(i + 1)];
        for (int p = begin; p < end; ++p) {
            const int j = body.neighbor_flat[static_cast<std::size_t>(p)];
            double diff = 0.0;
            for (int gidx = 0; gidx < kGeneExprDim; ++gidx) {
                const double gi = pop.G[static_cast<std::size_t>(i) * kGeneExprDim + gidx];
                const double gj = pop.G[static_cast<std::size_t>(j) * kGeneExprDim + gidx];
                diff += std::abs(gi - gj);
            }
            diff /= static_cast<double>(kGeneExprDim);
            const double coupling_program =
                0.5 * (pop.G[static_cast<std::size_t>(i) * kGeneExprDim + 6] +
                       pop.G[static_cast<std::size_t>(j) * kGeneExprDim + 6]);
            const double adhesion = 1.0 - std::abs(
                pop.G[static_cast<std::size_t>(i) * kGeneExprDim + 5] -
                pop.G[static_cast<std::size_t>(j) * kGeneExprDim + 5]);
            const double base = clampd(
                d.coupling_base * (0.25 + 0.75 * coupling_program) * (0.45 + 0.55 * (1.0 - diff)) +
                    d.adhesion_coeff * adhesion,
                0.02,
                1.0);
            const double edge_base = edge_ablation_matches(intervention, i, j) ? 0.0 : base;
            pop.gap_base[static_cast<std::size_t>(p)] = edge_base;
            pop.gap[static_cast<std::size_t>(p)] = edge_base;
        }
    }
    return pop;
}

void reset_fast_state(CellPopulation& pop, const DecodedGenome& d) {
    for (int i = 0; i < pop.count; ++i) {
        pop.V[static_cast<std::size_t>(i)] = 0.0;
        pop.Wrec[static_cast<std::size_t>(i)] = d.recovery_reset;
        pop.Ca[static_cast<std::size_t>(i)] *= (1.0 - d.trial_fast_decay);
        pop.a_mech[static_cast<std::size_t>(i)] = 0.0;
        pop.chem_out[static_cast<std::size_t>(i)] = 0.0;
        pop.stress[static_cast<std::size_t>(i)] *= 0.5;
    }
    std::fill(pop.chem_field.begin(), pop.chem_field.end(), 0.0);
}

void apply_damage(CellPopulation& pop, const BodyTemplate& body, const double frac, std::mt19937& rng) {
    std::vector<int> candidates;
    for (int i = 0; i < pop.count; ++i) {
        if (pop.active[static_cast<std::size_t>(i)] == 0) continue;
        candidates.push_back(i);
    }
    if (candidates.empty()) return;
    std::shuffle(candidates.begin(), candidates.end(), rng);
    const int n_remove = std::max(1, static_cast<int>(std::round(frac * static_cast<double>(candidates.size()))));
    for (int k = 0; k < n_remove && k < static_cast<int>(candidates.size()); ++k) {
        const int i = candidates[static_cast<std::size_t>(k)];
        pop.active[static_cast<std::size_t>(i)] = 0;
        pop.V[static_cast<std::size_t>(i)] = 0.0;
        pop.Wrec[static_cast<std::size_t>(i)] = 0.0;
        pop.Ca[static_cast<std::size_t>(i)] = 0.0;
        pop.sigma[static_cast<std::size_t>(i)] = 0.1;
        pop.energy[static_cast<std::size_t>(i)] = 0.0;
        pop.a_mech[static_cast<std::size_t>(i)] = 0.0;
        pop.chem_out[static_cast<std::size_t>(i)] = 0.0;
        pop.stress[static_cast<std::size_t>(i)] = 0.0;
        for (int p = body.neighbor_offsets[static_cast<std::size_t>(i)];
             p < body.neighbor_offsets[static_cast<std::size_t>(i + 1)];
             ++p) {
            pop.gap[static_cast<std::size_t>(p)] = 0.0;
        }
    }
    for (int i = 0; i < pop.count; ++i) {
        for (int p = body.neighbor_offsets[static_cast<std::size_t>(i)];
             p < body.neighbor_offsets[static_cast<std::size_t>(i + 1)];
             ++p) {
            if (pop.active[static_cast<std::size_t>(body.neighbor_flat[static_cast<std::size_t>(p)])] == 0) {
                pop.gap[static_cast<std::size_t>(p)] = 0.0;
            }
        }
    }
}

void write_stress(CellPopulation& pop,
                  const BodyTemplate& body,
                  const DecodedGenome& d,
                  const TaskEnv& env,
                  const int relax_iters) {
    std::vector<double> external(pop.count, 0.0);
    std::vector<double> next_stress(pop.count, 0.0);
    std::vector<double> neighbor_term(pop.count, 0.0);
    const TaskTelemetry telemetry = env.telemetry();

    pop.stress_hinge.assign(pop.count, 0.0);
    pop.stress_lateral.assign(pop.count, 0.0);
    pop.stress_gravity.assign(pop.count, 0.0);
    pop.stress_drift.assign(pop.count, 0.0);
    pop.stress_neighbor.assign(pop.count, 0.0);

    for (int i = 0; i < pop.count; ++i) {
        if (pop.active[static_cast<std::size_t>(i)] == 0) {
            external[static_cast<std::size_t>(i)] = 0.0;
            continue;
        }
        const auto& c = body.cells[static_cast<std::size_t>(i)];
        const double x = c.features[0];
        const double y = c.features[1];
        const double hinge = c.features[2];
        const double cart_transmit = 0.35 + 0.65 * (1.0 - y);
        const double hinge_term =
            d.stress_scale * (-22.0 * telemetry.hinge_signal * x * (0.30 * hinge + 0.70 * cart_transmit));
        const double lateral_term =
            d.stress_scale * (9.0 * telemetry.lateral_signal * (0.20 + 0.80 * cart_transmit) *
                              (0.35 + 0.65 * std::abs(x)));
        const double gravity_term =
            d.stress_scale * (0.12 * telemetry.support_load * (1.0 - y) * (c.ground ? 1.0 : 0.15));
        const double drift_term =
            d.stress_scale * (0.35 * telemetry.drift_signal * (c.ground ? -x : 0.0));
        const double ext = hinge_term + lateral_term + gravity_term + drift_term;
        pop.stress_hinge[static_cast<std::size_t>(i)] = hinge_term;
        pop.stress_lateral[static_cast<std::size_t>(i)] = lateral_term;
        pop.stress_gravity[static_cast<std::size_t>(i)] = gravity_term;
        pop.stress_drift[static_cast<std::size_t>(i)] = drift_term;
        external[static_cast<std::size_t>(i)] = ext;
    }

    for (int i = 0; i < pop.count; ++i) next_stress[static_cast<std::size_t>(i)] = external[static_cast<std::size_t>(i)];

    for (int iter = 0; iter < relax_iters; ++iter) {
        for (int i = 0; i < pop.count; ++i) {
            if (pop.active[static_cast<std::size_t>(i)] == 0) {
                next_stress[static_cast<std::size_t>(i)] = 0.0;
                continue;
            }
            const double g0 = pop.G[static_cast<std::size_t>(i) * kGeneExprDim + 0];
            const double g1 = pop.G[static_cast<std::size_t>(i) * kGeneExprDim + 1];
            const double stiffness = d.stiffness_softness * (0.35 + 0.65 * (1.0 - g0)) * (0.75 + 0.25 * (1.0 - g1));
            double acc = 0.0;
            double wsum = 0.0;
            for (int p = body.neighbor_offsets[static_cast<std::size_t>(i)];
                 p < body.neighbor_offsets[static_cast<std::size_t>(i + 1)];
                 ++p) {
                const int j = body.neighbor_flat[static_cast<std::size_t>(p)];
                if (pop.active[static_cast<std::size_t>(j)] == 0) continue;
                const double g0n = pop.G[static_cast<std::size_t>(j) * kGeneExprDim + 0];
                const double gn_stiff = d.stiffness_softness * (0.35 + 0.65 * (1.0 - g0n));
                const double w = 0.5 * (stiffness + gn_stiff);
                acc += w * pop.stress[static_cast<std::size_t>(j)];
                wsum += w;
            }
            const double neigh = (wsum > 1e-9) ? (acc / wsum) : 0.0;
            neighbor_term[static_cast<std::size_t>(i)] = neigh;
            const double blended = 0.45 * external[static_cast<std::size_t>(i)] + 0.55 * neigh;
            next_stress[static_cast<std::size_t>(i)] = blended;
        }
        for (int i = 0; i < pop.count; ++i) {
            pop.stress_neighbor[static_cast<std::size_t>(i)] = neighbor_term[static_cast<std::size_t>(i)];
            pop.stress[static_cast<std::size_t>(i)] =
                d.stress_smoothing * pop.stress[static_cast<std::size_t>(i)] +
                (1.0 - d.stress_smoothing) * next_stress[static_cast<std::size_t>(i)];
        }
    }
}

#ifndef CELLENGINE_USE_EXTERNAL_CUDA_DISPATCH
bool try_cuda_cell_tick(CellPopulation& pop,
                        const BodyTemplate& body,
                        const DecodedGenome& d,
                        const CellEngineConfig& cfg,
                        const double dt,
                        const bool teaching_active,
                        const ODDIntervention* intervention) {
#ifdef USE_CUDA
    auto& runtime = cuda_cell_tick_runtime();
    if (!runtime.should_try(pop.count, static_cast<int>(body.neighbor_flat.size()), body.grid_cell_count())) {
        return false;
    }
    return runtime.tick(pop, body, d, cfg, dt, teaching_active, intervention);
#else
    (void)pop;
    (void)body;
    (void)d;
    (void)cfg;
    (void)dt;
    (void)teaching_active;
    (void)intervention;
    return false;
#endif
}
#endif

void record_last_tick_decomposition(CellPopulation& pop,
                                    const BodyTemplate& body,
                                    const DecodedGenome& d,
                                    const std::vector<double>& pre_V,
                                    const std::vector<double>& pre_gap,
                                    const double dt,
                                    const bool teaching_active,
                                    const ODDIntervention* intervention) {
    pop.mech_a_gate.assign(pop.count, 0.0);
    pop.mech_contract.assign(pop.count, 0.0);
    pop.mech_prefactor.assign(pop.count, 0.0);
    pop.mech_force_contrib.assign(pop.count, 0.0);
    pop.gap_corr_last.assign(body.neighbor_flat.size(), 0.0);
    pop.gap_hebb_delta_last.assign(body.neighbor_flat.size(), 0.0);
    pop.gap_decay_delta_last.assign(body.neighbor_flat.size(), 0.0);
    pop.gap_old_last.assign(body.neighbor_flat.size(), 0.0);
    pop.gap_new_last.assign(body.neighbor_flat.size(), 0.0);

    for (int i = 0; i < pop.count; ++i) {
        if (pop.active[static_cast<std::size_t>(i)] == 0) continue;
        const std::size_t idx = static_cast<std::size_t>(i);
        const double g1 = pop.G[idx * kGeneExprDim + 1];
        const double g2 = pop.G[idx * kGeneExprDim + 2];
        const double energy_gate = std::max(0.05, pop.energy[idx]);
        const double a_gate = 0.10 + 0.90 * sigmoid(2.5 * pop.V[idx]);
        const double mech_contract = std::max(0.0, pop.stress[idx]) * (0.15 + 1.10 * g2);
        const double prefactor =
            0.20 + d.basal_contractility + 1.80 * g1 + d.muscle_bias * body.cells[idx].features[3];
        pop.mech_a_gate[idx] = a_gate;
        pop.mech_contract[idx] = mech_contract;
        pop.mech_prefactor[idx] = prefactor;
        pop.mech_force_contrib[idx] =
            16.0 * pop.a_mech[idx] * body.cells[idx].force_direction_x * body.cells[idx].mech_advantage;
        (void)energy_gate;
    }

    for (int i = 0; i < pop.count; ++i) {
        if (pop.active[static_cast<std::size_t>(i)] == 0) continue;
        const double g7 = pop.G[static_cast<std::size_t>(i) * kGeneExprDim + 7];
        const double hebb_scale = d.hebb_rate * (0.35 + 1.2 * g7) * (teaching_active ? d.teacher_hebb_boost : 1.0);
        const int begin = body.neighbor_offsets[static_cast<std::size_t>(i)];
        const int end = body.neighbor_offsets[static_cast<std::size_t>(i + 1)];
        for (int p = begin; p < end; ++p) {
            const int j = body.neighbor_flat[static_cast<std::size_t>(p)];
            const std::size_t edge_idx = static_cast<std::size_t>(p);
            const double old_gap = pre_gap[edge_idx];
            pop.gap_old_last[edge_idx] = old_gap;
            pop.gap_new_last[edge_idx] = pop.gap[edge_idx];
            if (pop.active[static_cast<std::size_t>(j)] == 0) continue;
            const double corr_ij = clampd(pre_V[static_cast<std::size_t>(i)] * pre_V[static_cast<std::size_t>(j)], -1.0, 1.0);
            pop.gap_corr_last[edge_idx] = corr_ij;
            if (intervention && intervention->freeze_gap_plasticity) continue;
            pop.gap_hebb_delta_last[edge_idx] = dt * (hebb_scale * corr_ij);
            pop.gap_decay_delta_last[edge_idx] = dt * (-d.decay_rate * (old_gap - pop.gap_base[edge_idx]));
        }
    }
}

void cell_tick(CellPopulation& pop,
               const BodyTemplate& body,
               const DecodedGenome& d,
               const CellEngineConfig& cfg,
               const double dt,
               const bool teaching_active,
               const ODDIntervention* intervention,
               const bool capture_tick_decomposition) {
    const std::vector<double> pre_V = capture_tick_decomposition ? pop.V : std::vector<double>{};
    const std::vector<double> pre_gap = capture_tick_decomposition ? pop.gap : std::vector<double>{};
    const bool allow_cuda =
        !(intervention && intervention->ablate_edge_src >= 0 && intervention->ablate_edge_dst >= 0);
    if (allow_cuda && try_cuda_cell_tick(pop, body, d, cfg, dt, teaching_active, intervention)) {
        if (capture_tick_decomposition) {
            record_last_tick_decomposition(pop, body, d, pre_V, pre_gap, dt, teaching_active, intervention);
        }
        return;
    }
    std::vector<double> nextV = pop.V;
    std::vector<double> nextW = pop.Wrec;
    std::vector<double> nextCa = pop.Ca;
    std::vector<double> nextSigma = pop.sigma;
    std::vector<double> nextEnergy = pop.energy;
    std::vector<double> nextA = pop.a_mech;
    std::vector<double> nextChem = pop.chem_out;
    std::vector<double> nextActivity = pop.activity_avg;
    std::vector<double> nextGap = pop.gap;
    std::vector<double> nextField = pop.chem_field;

    for (int i = 0; i < pop.count; ++i) {
        if (pop.active[static_cast<std::size_t>(i)] == 0) {
            nextV[static_cast<std::size_t>(i)] = 0.0;
            nextW[static_cast<std::size_t>(i)] = 0.0;
            nextCa[static_cast<std::size_t>(i)] = 0.0;
            nextSigma[static_cast<std::size_t>(i)] = 0.1;
            nextEnergy[static_cast<std::size_t>(i)] = 0.0;
            nextA[static_cast<std::size_t>(i)] = 0.0;
            nextChem[static_cast<std::size_t>(i)] = 0.0;
            nextActivity[static_cast<std::size_t>(i)] = 0.0;
            continue;
        }

        const double g0 = pop.G[static_cast<std::size_t>(i) * kGeneExprDim + 0];
        const double g1 = pop.G[static_cast<std::size_t>(i) * kGeneExprDim + 1];
        const double g2 = pop.G[static_cast<std::size_t>(i) * kGeneExprDim + 2];
        const double g3 = pop.G[static_cast<std::size_t>(i) * kGeneExprDim + 3];
        const double g4 = pop.G[static_cast<std::size_t>(i) * kGeneExprDim + 4];
        const double g7 = pop.G[static_cast<std::size_t>(i) * kGeneExprDim + 7];

        const double V = pop.V[static_cast<std::size_t>(i)];
        const double W = pop.Wrec[static_cast<std::size_t>(i)];
        const double energy = std::max(0.05, pop.energy[static_cast<std::size_t>(i)]);
        const int grid_idx = (i < static_cast<int>(body.cell_to_grid.size())) ? body.cell_to_grid[static_cast<std::size_t>(i)] : -1;

        double I_elec = 0.0;
        double I_chem = (grid_idx >= 0 && !(intervention && intervention->disable_chemical) &&
                         !(intervention && intervention->disable_diffusion))
                            ? pop.chem_field[static_cast<std::size_t>(grid_idx)]
                            : 0.0;
        for (int p = body.neighbor_offsets[static_cast<std::size_t>(i)];
             p < body.neighbor_offsets[static_cast<std::size_t>(i + 1)];
             ++p) {
            const int j = body.neighbor_flat[static_cast<std::size_t>(p)];
            if (edge_ablation_matches(intervention, i, j)) continue;
            if (pop.active[static_cast<std::size_t>(j)] == 0) continue;
            const double wgap = pop.gap[static_cast<std::size_t>(p)];
            if (!(intervention && intervention->disable_electrical)) {
                I_elec += wgap * (pop.V[static_cast<std::size_t>(j)] - V);
            }
            if ((intervention && intervention->disable_diffusion) && !(intervention && intervention->disable_chemical)) {
                I_chem += wgap * pop.chem_out[static_cast<std::size_t>(j)];
            }
        }

        const double I_mech =
            (intervention && intervention->disable_mechanical)
                ? 0.0
                : d.mech_base * (0.50 + 2.20 * g2) * pop.stress[static_cast<std::size_t>(i)];
        const double chem_self =
            (intervention && (intervention->disable_chemical || intervention->disable_self_chemical))
                ? 0.0
                : d.chemical_balance * pop.chem_out[static_cast<std::size_t>(i)];
        const double I_total =
            pop.sigma[static_cast<std::size_t>(i)] *
            (I_elec + d.chem_coeff * I_chem + I_mech + chem_self) *
            energy;

        const double tau_v = std::max(0.35, d.tau_v_base + d.tau_v_gain * (g0 - 0.5));
        const double eps = std::max(0.015, d.epsilon_base + d.epsilon_gain * (g0 - 0.5));
        const double dV = (V - (V * V * V) / 3.0 - W + I_total) / tau_v;
        const double dW = eps * (V + d.fhn_a - d.fhn_b * W);
        const double Vn = clampd(V + dV * dt, -2.5, 2.5);
        const double Wn = clampd(W + dW * dt, -2.5, 2.5);
        const double Can = clampd(pop.Ca[static_cast<std::size_t>(i)] +
                                      dt * ((-pop.Ca[static_cast<std::size_t>(i)] / d.tau_ca) +
                                            d.alpha_ca * std::max(0.0, Vn - d.v_ca_thresh)),
                                  0.0,
                                  2.5);
        const double activity = d.activity_decay * pop.activity_avg[static_cast<std::size_t>(i)] +
                                (1.0 - d.activity_decay) * std::abs(Vn);
        const double sigma_n = (intervention && intervention->freeze_sigma)
                                   ? pop.sigma[static_cast<std::size_t>(i)]
                                   : clampd(
                                         pop.sigma[static_cast<std::size_t>(i)] +
                                             dt * (d.sigma_alpha * (d.sigma_target - pop.sigma[static_cast<std::size_t>(i)]) +
                                                   d.sigma_beta * (d.sigma_target - activity)),
                                         0.10,
                                         5.0);
        const double a_gate = 0.10 + 0.90 * sigmoid(2.5 * Vn);
        const double mech_contract = std::max(0.0, pop.stress[static_cast<std::size_t>(i)]) * (0.15 + 1.10 * g2);
        const double a_mech =
            clampd((0.20 + d.basal_contractility + 1.80 * g1 +
                    d.muscle_bias * body.cells[static_cast<std::size_t>(i)].features[3]) *
                       (std::tanh(2.2 * Can) + 0.70 * mech_contract) * a_gate * energy * d.energy_output_scale,
                   -2.5,
                   2.5);
        const double chem = (intervention && intervention->disable_chemical)
                                ? 0.0
                                : (0.15 + g3) * std::tanh(1.6 * Can) * ((g4 >= 0.5) ? 1.0 : -1.0);
        const double energy_n = clampd(
            pop.energy[static_cast<std::size_t>(i)] +
                dt * (d.energy_replenish - d.cost_vm * std::abs(Vn) * std::abs(Vn) - d.cost_ca * Can -
                      d.cost_mech * std::abs(a_mech)),
            0.0,
            1.0);

        nextV[static_cast<std::size_t>(i)] = Vn;
        nextW[static_cast<std::size_t>(i)] = Wn;
        nextCa[static_cast<std::size_t>(i)] = Can;
        nextSigma[static_cast<std::size_t>(i)] = sigma_n;
        nextEnergy[static_cast<std::size_t>(i)] = energy_n;
        nextA[static_cast<std::size_t>(i)] = a_mech;
        nextChem[static_cast<std::size_t>(i)] = chem;
        nextActivity[static_cast<std::size_t>(i)] = activity;

        const double hebb_scale = d.hebb_rate * (0.35 + 1.2 * g7) * (teaching_active ? d.teacher_hebb_boost : 1.0);
        for (int p = body.neighbor_offsets[static_cast<std::size_t>(i)];
             p < body.neighbor_offsets[static_cast<std::size_t>(i + 1)];
             ++p) {
            const int j = body.neighbor_flat[static_cast<std::size_t>(p)];
            if (edge_ablation_matches(intervention, i, j)) {
                nextGap[static_cast<std::size_t>(p)] = 0.0;
                continue;
            }
            if (pop.active[static_cast<std::size_t>(j)] == 0) {
                nextGap[static_cast<std::size_t>(p)] = 0.0;
                continue;
            }
            const double corr_ij = clampd(pop.V[static_cast<std::size_t>(i)] * pop.V[static_cast<std::size_t>(j)], -1.0, 1.0);
            if (intervention && intervention->freeze_gap_plasticity) {
                nextGap[static_cast<std::size_t>(p)] = pop.gap[static_cast<std::size_t>(p)];
            } else {
                const double gap = pop.gap[static_cast<std::size_t>(p)] +
                                   dt * (hebb_scale * corr_ij - d.decay_rate * (pop.gap[static_cast<std::size_t>(p)] - pop.gap_base[static_cast<std::size_t>(p)]));
                nextGap[static_cast<std::size_t>(p)] = clampd(gap, 0.0, 1.0);
            }
        }
    }

    pop.V.swap(nextV);
    pop.Wrec.swap(nextW);
    pop.Ca.swap(nextCa);
    pop.sigma.swap(nextSigma);
    pop.energy.swap(nextEnergy);
    pop.a_mech.swap(nextA);
    pop.chem_out.swap(nextChem);
    pop.activity_avg.swap(nextActivity);
    pop.gap.swap(nextGap);

    if ((intervention && intervention->disable_chemical) || (intervention && intervention->disable_diffusion)) {
        std::fill(nextField.begin(), nextField.end(), 0.0);
    } else if (!nextField.empty()) {
        const int diffusion_steps = std::max(1, cfg.chemical_diffusion_steps);
        for (int step = 0; step < diffusion_steps; ++step) {
            std::vector<double> work = nextField;
            const double local_dt = dt / static_cast<double>(diffusion_steps);
            for (std::size_t gi = 0; gi < work.size(); ++gi) {
                const int begin = body.diffusion_offsets[gi];
                const int end = body.diffusion_offsets[gi + 1];
                double neigh_sum = 0.0;
                int neigh_n = 0;
                for (int p = begin; p < end; ++p) {
                    neigh_sum += nextField[static_cast<std::size_t>(body.diffusion_flat[static_cast<std::size_t>(p)])];
                    neigh_n += 1;
                }
                const double self = nextField[gi];
                const double neigh_mean = (neigh_n > 0) ? (neigh_sum / static_cast<double>(neigh_n)) : 0.0;
                const double lap = neigh_mean - self;
                const int cell_id = body.grid_to_cell[gi];
                const double source = (cell_id >= 0 && pop.active[static_cast<std::size_t>(cell_id)] != 0)
                                          ? pop.chem_out[static_cast<std::size_t>(cell_id)]
                                          : 0.0;
                work[gi] = clampd(self + local_dt * (cfg.chemical_diffusion_rate * lap - cfg.chemical_decay * self + source),
                                  -6.0,
                                  6.0);
            }
            nextField.swap(work);
        }
    }
    pop.chem_field.swap(nextField);
    if (capture_tick_decomposition) {
        record_last_tick_decomposition(pop, body, d, pre_V, pre_gap, dt, teaching_active, intervention);
    }
}

void slow_update(CellPopulation& pop, const BodyTemplate& body, const DecodedGenome& d, const ODDIntervention* intervention) {
    for (int i = 0; i < pop.count; ++i) {
        if (pop.active[static_cast<std::size_t>(i)] == 0) continue;
        const int grid_idx = (i < static_cast<int>(body.cell_to_grid.size())) ? body.cell_to_grid[static_cast<std::size_t>(i)] : -1;
        double neighbor_signal = 0.0;
        int n = 0;
        for (int p = body.neighbor_offsets[static_cast<std::size_t>(i)];
             p < body.neighbor_offsets[static_cast<std::size_t>(i + 1)];
             ++p) {
            const int j = body.neighbor_flat[static_cast<std::size_t>(p)];
            if (pop.active[static_cast<std::size_t>(j)] == 0) continue;
            neighbor_signal += pop.Ca[static_cast<std::size_t>(j)];
            n += 1;
        }
        neighbor_signal = (n > 0) ? neighbor_signal / static_cast<double>(n) : 0.0;
        const double field_signal =
            (grid_idx >= 0 && !pop.chem_field.empty()) ? sigmoid(pop.chem_field[static_cast<std::size_t>(grid_idx)]) : 0.5;
        for (int gidx = 0; gidx < kGeneExprDim; ++gidx) {
            const std::size_t idx = static_cast<std::size_t>(i) * kGeneExprDim + gidx;
            if (intervention && intervention->neutralize_gene_channel == gidx) {
                pop.G[idx] = 0.5;
                continue;
            }
            if (intervention && intervention->freeze_gene_channel == gidx) {
                continue;
            }
            if (!(intervention && intervention->disable_slow_drift)) {
                const double target =
                    clampd(0.45 * pop.G[idx] + 0.23 * pop.Ca[static_cast<std::size_t>(i)] +
                               0.18 * neighbor_signal + 0.14 * field_signal,
                           0.0,
                           1.0);
                pop.G[idx] = clampd(pop.G[idx] + d.g_drift_rate * (target - pop.G[idx]), 0.0, 1.0);
            }
        }
        pop.fatigue[static_cast<std::size_t>(i)] = clampd(
            0.97 * pop.fatigue[static_cast<std::size_t>(i)] + 0.03 * (1.0 - pop.energy[static_cast<std::size_t>(i)]),
            0.0,
            1.0);
    }
}
