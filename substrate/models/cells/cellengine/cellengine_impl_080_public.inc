ComparisonSummary run_full_benchmark(const CellEngineConfig& cfg) {
    ComparisonSummary best_summary;
    best_summary.notes = "No completed run.";
    best_summary.task_name = cfg.task_name;
    best_summary.body_mode = cfg.body_mode;

    const bool do_cell = (cfg.mode != "rl_only");
    const bool do_rl = (cfg.mode != "cell_only");

    auto attach_rl = [&](ComparisonSummary& summary, const CellEngineConfig& tuned, const int seed) {
        if (!do_rl) return;
        auto [best_alg, best_rl] = train_and_select_rl(tuned, seed, summary.rl_dqn, summary.rl_a2c);
        summary.rl_algorithm = best_alg;
        summary.rl = best_rl;
    };

    if (cfg.mode == "replay") {
        if (cfg.load_genome_csv.empty()) {
            best_summary.notes = "Replay mode requires CELLENGINE_LOAD_GENOME";
            return best_summary;
        }
        Genome loaded{};
        if (!load_genome_csv(cfg.load_genome_csv, loaded)) {
            best_summary.notes = "Failed to load genome CSV: " + cfg.load_genome_csv;
            return best_summary;
        }
        return replay_loaded_genome(loaded, cfg);
    }

    if (cfg.mode == "preview") {
        if (cfg.load_genome_csv.empty()) {
            best_summary.notes = "Preview mode requires CELLENGINE_LOAD_GENOME";
            return best_summary;
        }
        Genome loaded{};
        if (!load_genome_csv(cfg.load_genome_csv, loaded)) {
            best_summary.notes = "Failed to load genome CSV: " + cfg.load_genome_csv;
            return best_summary;
        }
        return preview_loaded_genome(loaded, cfg);
    }

    if (do_cell && !cfg.load_genome_csv.empty()) {
        Genome loaded{};
        if (!load_genome_csv(cfg.load_genome_csv, loaded)) {
            best_summary.notes = "Failed to load genome CSV: " + cfg.load_genome_csv;
            return best_summary;
        }
        const CellEngineConfig resolved_cfg = resolve_loaded_genome_cfg(cfg, loaded);
        best_summary.attempt = 1;
        best_summary.champion_generation = -1;
        best_summary.champion_search_score = std::numeric_limits<double>::quiet_NaN();
        best_summary.champion_genes = loaded.genes;
        best_summary.champion_params = default_candidate_params(resolved_cfg);
        const BodyTemplate body = make_body_template(loaded, resolved_cfg);
        best_summary.champion_cell_count = body.cell_count();
        best_summary.champion_body_depth = body.body_depth();
        best_summary.task_name = resolved_cfg.task_name;
        best_summary.body_mode = resolved_cfg.body_mode;
        best_summary.cell_clean = evaluate_genome(loaded, resolved_cfg, resolved_cfg.final_budget, resolved_cfg.seed + 50000, false, true).summary;
        best_summary.cell_damaged = evaluate_genome(loaded, resolved_cfg, resolved_cfg.damage_budget, resolved_cfg.seed + 51000, true, true).summary;
        best_summary.pd_gold = evaluate_pd(resolved_cfg, resolved_cfg.final_budget, resolved_cfg.seed + 52000);
        attach_rl(best_summary, resolved_cfg, resolved_cfg.seed + 53000);
        best_summary.odd = run_odd_experiment(loaded, resolved_cfg, resolved_cfg.seed + 54000);
        best_summary.solved = task_is_solved(best_summary.cell_clean, parse_task_kind(resolved_cfg.task_name), resolved_cfg);
        best_summary.notes = "loaded_genome=" + cfg.load_genome_csv;
        return best_summary;
    }

    if (!do_cell && do_rl) {
        best_summary.attempt = 1;
        best_summary.task_name = cfg.task_name;
        best_summary.body_mode = cfg.body_mode;
        attach_rl(best_summary, cfg, cfg.seed + 53000);
        best_summary.notes = "rl_only mode";
        best_summary.solved = task_is_solved(best_summary.rl.clean, parse_task_kind(cfg.task_name), cfg);
        return best_summary;
    }

    for (int attempt = 0; attempt < std::max(1, cfg.auto_attempts); ++attempt) {
        CellEngineConfig tuned = cfg;
        tuned.population_size = cfg.population_size + 12 * attempt;
        tuned.generations = cfg.generations + 12 * attempt;
        tuned.search_budget.num_trials = cfg.search_budget.num_trials + 2 * attempt;
        tuned.search_budget.max_ticks = cfg.search_budget.max_ticks + 20 * attempt;
        tuned.mutation_sigma = std::max(0.04, cfg.mutation_sigma * (1.0 - 0.12 * attempt));

        const int attempt_seed = cfg.seed + attempt * 1000;
        const EvolutionResult evo = evolve_cells(tuned, attempt_seed);
        const CellEngineConfig resolved_tuned = resolve_candidate_cfg(tuned, evo.best_genome);

        ComparisonSummary summary;
        summary.attempt = attempt + 1;
        summary.champion_generation = evo.best_generation;
        const BodyTemplate body = make_body_template(evo.best_genome, resolved_tuned);
        summary.champion_cell_count = body.cell_count();
        summary.champion_body_depth = body.body_depth();
        summary.champion_search_score = evo.best_search_score;
        summary.champion_genes = evo.best_genome.genes;
        summary.champion_params = default_candidate_params(resolved_tuned);
        summary.task_name = resolved_tuned.task_name;
        summary.body_mode = resolved_tuned.body_mode;
        summary.cell_clean = evaluate_genome(evo.best_genome, tuned, tuned.final_budget, attempt_seed + 50000, false, true).summary;
        summary.cell_damaged = evaluate_genome(evo.best_genome, tuned, tuned.damage_budget, attempt_seed + 51000, true, true).summary;
        summary.pd_gold = evaluate_pd(resolved_tuned, resolved_tuned.final_budget, attempt_seed + 52000);
        attach_rl(summary, resolved_tuned, attempt_seed + 53000);
        summary.solved = task_is_solved(summary.cell_clean, parse_task_kind(resolved_tuned.task_name), resolved_tuned);

        std::ostringstream note;
        note << "attempt=" << summary.attempt
             << " pop=" << tuned.population_size
             << " gens=" << tuned.generations
             << " search_trials=" << tuned.search_budget.num_trials
             << " search_ticks=" << tuned.search_budget.max_ticks;
        summary.notes = note.str();

        if (task_summary_score(summary.cell_clean, parse_task_kind(tuned.task_name)) >
            task_summary_score(best_summary.cell_clean, parse_task_kind(tuned.task_name))) {
            best_summary = summary;
        }
        if (summary.solved) break;
    }

    if (best_summary.notes.empty()) best_summary.notes = "completed";
    if (cfg.odd_enabled && best_summary.cell_clean.max_ticks > 0.0) {
        Genome champion{};
        champion.genes = best_summary.champion_genes;
        champion.params = best_summary.champion_params;
        champion.has_params = true;
        best_summary.odd = run_odd_experiment(champion, cfg, cfg.seed + 54000 + best_summary.attempt * 1000);
    }
    return best_summary;
}

std::string format_summary(const ComparisonSummary& summary) {
    std::ostringstream oss;
    const auto search_score =
        std::isfinite(summary.champion_search_score)
            ? [&]() {
                  std::ostringstream score;
                  score << std::fixed << std::setprecision(4) << summary.champion_search_score;
                  return score.str();
              }()
            : std::string("n/a");
    oss << "CellEngine benchmark\n";
    oss << "attempt=" << summary.attempt
        << " task=" << summary.task_name
        << " body=" << summary.body_mode
        << " cells=" << summary.champion_cell_count
        << " depth=" << summary.champion_body_depth
        << " champion_generation=" << summary.champion_generation
        << " search_score=" << search_score
        << " solved=" << (summary.solved ? "yes" : "no") << "\n";
    oss << summary_line("cell_clean", summary.cell_clean) << "\n";
    oss << summary_line("cell_damaged", summary.cell_damaged) << "\n";
    oss << summary_line("pd_gold", summary.pd_gold) << "\n";
    if (!summary.rl_algorithm.empty()) {
        oss << "rl_best_algorithm=" << summary.rl_algorithm << "\n";
    }
    oss << summary_line("rl_best_clean", summary.rl.clean) << "\n";
    oss << summary_line("rl_best_damaged", summary.rl.damaged) << "\n";
    oss << summary_line("rl_best_recovered", summary.rl.recovered) << "\n";
    if (summary.rl_dqn.clean.max_ticks > 0.0) {
        oss << summary_line("rl_dqn_clean", summary.rl_dqn.clean) << "\n";
        oss << summary_line("rl_dqn_damaged", summary.rl_dqn.damaged) << "\n";
        oss << summary_line("rl_dqn_recovered", summary.rl_dqn.recovered) << "\n";
    }
    if (summary.rl_a2c.clean.max_ticks > 0.0) {
        oss << summary_line("rl_a2c_clean", summary.rl_a2c.clean) << "\n";
        oss << summary_line("rl_a2c_damaged", summary.rl_a2c.damaged) << "\n";
        oss << summary_line("rl_a2c_recovered", summary.rl_a2c.recovered) << "\n";
    }
    if (summary.odd.available) {
        oss << summary_line("odd_baseline_clean", summary.odd.baseline_clean) << "\n";
        if (!summary.odd.mechanism_deltas.empty()) {
            const auto& top = summary.odd.mechanism_deltas.front();
            oss << "odd_top_mechanism=" << top.key
                << " delta_survival=" << std::fixed << std::setprecision(4) << top.delta_survival_ratio
                << " delta_success=" << top.delta_success_rate << "\n";
        }
        if (!summary.odd.gene_channel_deltas.empty()) {
            const auto& top = summary.odd.gene_channel_deltas.front();
            oss << "odd_top_gene=" << top.key
                << " delta_survival=" << std::fixed << std::setprecision(4) << top.delta_survival_ratio
                << " delta_success=" << top.delta_success_rate << "\n";
        }
        if (!summary.odd.cell_force_deltas.empty()) {
            const auto& top = summary.odd.cell_force_deltas.front();
            oss << "odd_top_cell_force=" << top.key
                << " delta_survival=" << std::fixed << std::setprecision(4) << top.delta_survival_ratio
                << " delta_success=" << top.delta_success_rate << "\n";
        }
        if (!summary.odd.edge_deltas.empty()) {
            const auto& top = summary.odd.edge_deltas.front();
            oss << "odd_top_edge=" << top.key
                << " delta_survival=" << std::fixed << std::setprecision(4) << top.delta_survival_ratio
                << " delta_success=" << top.delta_success_rate << "\n";
        }
        if (!summary.odd.temporal_correlations.empty()) {
            const auto& top = summary.odd.temporal_correlations.front();
            oss << "odd_top_temporal_corr=" << top.left << ":" << top.right
                << " corr=" << std::fixed << std::setprecision(4) << top.correlation << "\n";
        }
        if (!summary.odd.structural_confounds.empty()) {
            const auto& top = summary.odd.structural_confounds.front();
            oss << "odd_top_structural_confound=" << top.left << ":" << top.right
                << " corr=" << std::fixed << std::setprecision(4) << top.correlation << "\n";
        }
    }
    return oss.str();
}

bool write_report(const ComparisonSummary& summary,
                  const CellEngineConfig& cfg,
                  const std::string& output_dir) {
    namespace fs = std::filesystem;
    try {
        fs::create_directories(output_dir);
        {
            std::ofstream md(fs::path(output_dir) / "report.md");
            md << "# CellEngine Benchmark Report\n\n";
            md << "- Date: 2026-03-09\n";
            md << "- Seed: " << cfg.seed << "\n";
            md << "- Solve threshold: " << cfg.solve_ratio << "\n";
            md << "- Mode: " << cfg.mode << "\n";
            md << "- Task: " << summary.task_name << "\n";
            md << "- Body mode: " << summary.body_mode << "\n";
            md << "- Champion cells: " << summary.champion_cell_count << "\n";
            md << "- Champion body depth: " << summary.champion_body_depth << "\n";
            md << "- RL selector: " << cfg.rl_algorithm << "\n";
            md << "- Notes: " << summary.notes << "\n\n";
            md << "## Champion Parameters\n\n";
            md << "| Parameter | Value | Discoverable |\n";
            md << "| --- | ---: | --- |\n";
            md << "| body_mode | `" << summary.champion_params.body_mode << "` | " << (cfg.evolve_body_mode ? "yes" : "no") << " |\n";
            md << "| development_steps | " << summary.champion_params.development_steps << " | " << (cfg.evolve_development_steps ? "yes" : "no") << " |\n";
            md << "| development_seed_half_width | " << summary.champion_params.development_seed_half_width << " | " << (cfg.evolve_development_seed_half_width ? "yes" : "no") << " |\n";
            md << "| max_cells | " << summary.champion_params.max_cells << " | " << (cfg.evolve_max_cells ? "yes" : "no") << " |\n";
            md << "| body_extent_x | " << summary.champion_params.body_extent_x << " | " << (cfg.evolve_body_extent_x ? "yes" : "no") << " |\n";
            md << "| body_extent_y | " << summary.champion_params.body_extent_y << " | " << (cfg.evolve_body_extent_y ? "yes" : "no") << " |\n";
            md << "| body_extent_z | " << summary.champion_params.body_extent_z << " | " << (cfg.evolve_body_extent_z ? "yes" : "no") << " |\n";
            md << "| chemical_diffusion_steps | " << summary.champion_params.chemical_diffusion_steps << " | " << (cfg.evolve_chemical_diffusion_steps ? "yes" : "no") << " |\n";
            md << "| development_growth_threshold | " << std::fixed << std::setprecision(4) << summary.champion_params.development_growth_threshold << " | " << (cfg.evolve_growth_threshold ? "yes" : "no") << " |\n";
            md << "| chemical_diffusion_rate | " << summary.champion_params.chemical_diffusion_rate << " | " << (cfg.evolve_chemical_diffusion_rate ? "yes" : "no") << " |\n";
            md << "| chemical_decay | " << summary.champion_params.chemical_decay << " | " << (cfg.evolve_chemical_decay ? "yes" : "no") << " |\n\n";
            md << "## Summary\n\n";
            md << "```text\n" << format_summary(summary) << "```\n\n";
            md << "## Observations\n\n";
            md << "- The cell controller is considered solved when full-protocol survival exceeds the configured threshold.\n";
            md << "- `pd_gold` is the reference upper bound on this environment.\n";
            md << "- `rl_best_*` identifies the strongest trained RL comparator among the requested algorithms.\n";
            md << "- `rl_*_damaged` and `rl_*_recovered` show immediate degradation and recovery after parameter deletion.\n";
            if (!summary.cell_clean.task_primary_label.empty()) {
                md << "- Primary task metric: `" << summary.cell_clean.task_primary_label
                   << "` = " << std::fixed << std::setprecision(4) << summary.cell_clean.task_primary << "\n";
            }
            if (!summary.cell_clean.task_secondary_label.empty()) {
                md << "- Secondary task metric: `" << summary.cell_clean.task_secondary_label
                   << "` = " << std::fixed << std::setprecision(4) << summary.cell_clean.task_secondary << "\n";
            }
            if (summary.odd.available) {
                auto write_delta_table = [&](const std::string& title, const std::vector<ODDDeltaSummary>& items) {
                    md << "### " << title << "\n\n";
                    md << "| Rank | Variable | Variant survival | Delta survival | Variant success | Delta success | Relative importance |\n";
                    md << "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n";
                    for (std::size_t i = 0; i < items.size(); ++i) {
                        const auto& item = items[i];
                        md << "| " << (i + 1)
                           << " | " << item.label
                           << " | " << std::fixed << std::setprecision(4) << item.variant_survival_ratio
                           << " | " << item.delta_survival_ratio
                           << " | " << item.variant_success_rate
                           << " | " << item.delta_success_rate
                           << " | " << (100.0 * odd_relative_importance(items, i)) << "% |\n";
                    }
                    md << "\n";
                    if (!items.empty()) {
                        const std::size_t start = items.size() > 3 ? items.size() - 3 : 0;
                        md << "Least-impact entries:";
                        for (std::size_t i = start; i < items.size(); ++i) {
                            md << " `" << items[i].key << "` (" << std::fixed << std::setprecision(4)
                               << items[i].delta_survival_ratio << ")";
                        }
                        md << "\n\n";
                    }
                };
                auto write_corr_table = [&](const std::string& title, const std::vector<ODDCorrelationSummary>& items) {
                    md << "### " << title << "\n\n";
                    md << "| Rank | Left | Right | Correlation |\n";
                    md << "| --- | --- | --- | ---: |\n";
                    for (std::size_t i = 0; i < items.size(); ++i) {
                        const auto& item = items[i];
                        md << "| " << (i + 1)
                           << " | " << item.left
                           << " | " << item.right
                           << " | " << std::fixed << std::setprecision(4) << item.correlation << " |\n";
                    }
                    md << "\n";
                };

                md << "## ODD Results\n\n";
                md << "- ODD budget: " << summary.odd.budget.num_trials << " trials x "
                   << summary.odd.budget.max_ticks << " ticks\n";
                md << "- Baseline clean survival: " << std::fixed << std::setprecision(4)
                   << summary.odd.baseline_clean.survival_ratio << "\n";
                md << "- Baseline clean success: " << summary.odd.baseline_clean.success_rate << "\n";
                if (!summary.odd.mechanism_deltas.empty()) {
                    const auto& top = summary.odd.mechanism_deltas.front();
                    md << "- Strongest mechanism effect: `" << top.key << "` with survival delta "
                       << top.delta_survival_ratio << "\n";
                }
                if (!summary.odd.gene_channel_deltas.empty()) {
                    const auto& top = summary.odd.gene_channel_deltas.front();
                    md << "- Strongest gene-channel effect: `" << top.key << "` with survival delta "
                       << top.delta_survival_ratio << "\n";
                }
                if (!summary.odd.cell_force_deltas.empty()) {
                    const auto& top = summary.odd.cell_force_deltas.front();
                    md << "- Strongest single-cell force effect: `" << top.key << "` with survival delta "
                       << top.delta_survival_ratio << "\n";
                }
                if (!summary.odd.edge_deltas.empty()) {
                    const auto& top = summary.odd.edge_deltas.front();
                    md << "- Strongest edge effect: `" << top.key << "` with survival delta "
                       << top.delta_survival_ratio << "\n";
                }
                md << "- Delta sign convention: `delta = baseline - variant`; positive means removing the variable hurt performance, negative means the ablation improved performance under this budget.\n";
                md << "- Notes: " << summary.odd.notes << "\n\n";

                write_delta_table("Mechanism Ablations", summary.odd.mechanism_deltas);
                write_delta_table("Gene Channel Ablations", summary.odd.gene_channel_deltas);
                write_delta_table("Per-Cell Force Ablations", summary.odd.cell_force_deltas);
                write_delta_table("Edge Ablations", summary.odd.edge_deltas);
                write_corr_table("Temporal Correlations", summary.odd.temporal_correlations);
                write_corr_table("Structural Confounds", summary.odd.structural_confounds);
            }
        }
        {
            std::ofstream json(fs::path(output_dir) / "summary.json");
            json << std::fixed << std::setprecision(6);
            const auto emit_number = [&](const double value) {
                if (std::isfinite(value)) {
                    json << value;
                } else {
                    json << "null";
                }
            };
            const auto emit_string = [&](const std::string& value) {
                json << "\"" << json_escape(value) << "\"";
            };
            json << "{\n";
            json << "  \"attempt\": " << summary.attempt << ",\n";
            json << "  \"champion_generation\": " << summary.champion_generation << ",\n";
            json << "  \"champion_cell_count\": " << summary.champion_cell_count << ",\n";
            json << "  \"champion_body_depth\": " << summary.champion_body_depth << ",\n";
            json << "  \"champion_search_score\": ";
            emit_number(summary.champion_search_score);
            json << ",\n";
            json << "  \"task_name\": ";
            emit_string(summary.task_name);
            json << ",\n";
            json << "  \"body_mode\": ";
            emit_string(summary.body_mode);
            json << ",\n";
            json << "  \"champion_params\": {\n";
            json << "    \"body_mode\": ";
            emit_string(summary.champion_params.body_mode);
            json << ",\n";
            json << "    \"development_steps\": " << summary.champion_params.development_steps << ",\n";
            json << "    \"development_seed_half_width\": " << summary.champion_params.development_seed_half_width << ",\n";
            json << "    \"max_cells\": " << summary.champion_params.max_cells << ",\n";
            json << "    \"body_extent_x\": " << summary.champion_params.body_extent_x << ",\n";
            json << "    \"body_extent_y\": " << summary.champion_params.body_extent_y << ",\n";
            json << "    \"body_extent_z\": " << summary.champion_params.body_extent_z << ",\n";
            json << "    \"chemical_diffusion_steps\": " << summary.champion_params.chemical_diffusion_steps << ",\n";
            json << "    \"development_growth_threshold\": ";
            emit_number(summary.champion_params.development_growth_threshold);
            json << ",\n";
            json << "    \"chemical_diffusion_rate\": ";
            emit_number(summary.champion_params.chemical_diffusion_rate);
            json << ",\n";
            json << "    \"chemical_decay\": ";
            emit_number(summary.champion_params.chemical_decay);
            json << "\n";
            json << "  },\n";
            json << "  \"discoverability\": {\n";
            json << "    \"body_mode\": " << (cfg.evolve_body_mode ? "true" : "false") << ",\n";
            json << "    \"development_steps\": " << (cfg.evolve_development_steps ? "true" : "false") << ",\n";
            json << "    \"development_seed_half_width\": " << (cfg.evolve_development_seed_half_width ? "true" : "false") << ",\n";
            json << "    \"max_cells\": " << (cfg.evolve_max_cells ? "true" : "false") << ",\n";
            json << "    \"body_extent_x\": " << (cfg.evolve_body_extent_x ? "true" : "false") << ",\n";
            json << "    \"body_extent_y\": " << (cfg.evolve_body_extent_y ? "true" : "false") << ",\n";
            json << "    \"body_extent_z\": " << (cfg.evolve_body_extent_z ? "true" : "false") << ",\n";
            json << "    \"chemical_diffusion_steps\": " << (cfg.evolve_chemical_diffusion_steps ? "true" : "false") << ",\n";
            json << "    \"development_growth_threshold\": " << (cfg.evolve_growth_threshold ? "true" : "false") << ",\n";
            json << "    \"chemical_diffusion_rate\": " << (cfg.evolve_chemical_diffusion_rate ? "true" : "false") << ",\n";
            json << "    \"chemical_decay\": " << (cfg.evolve_chemical_decay ? "true" : "false") << "\n";
            json << "  },\n";
            json << "  \"mode\": ";
            emit_string(cfg.mode);
            json << ",\n";
            json << "  \"rl_algorithm_requested\": ";
            emit_string(cfg.rl_algorithm);
            json << ",\n";
            json << "  \"rl_algorithm_selected\": ";
            emit_string(summary.rl_algorithm);
            json << ",\n";
            json << "  \"solved\": " << (summary.solved ? "true" : "false") << ",\n";
            json << "  \"notes\": ";
            emit_string(summary.notes);
            json << ",\n";
            auto emit_eval = [&](const char* key, const EvaluationSummary& s, const bool comma) {
                json << "  \"" << key << "\": {\n";
                json << "    \"total_ticks\": ";
                emit_number(s.total_ticks);
                json << ",\n";
                json << "    \"max_ticks\": ";
                emit_number(s.max_ticks);
                json << ",\n";
                json << "    \"survival_ratio\": ";
                emit_number(s.survival_ratio);
                json << ",\n";
                json << "    \"mean_ticks\": ";
                emit_number(s.mean_ticks);
                json << ",\n";
                json << "    \"success_rate\": ";
                emit_number(s.success_rate);
                json << ",\n";
                json << "    \"task_primary\": ";
                emit_number(s.task_primary);
                json << ",\n";
                json << "    \"task_secondary\": ";
                emit_number(s.task_secondary);
                json << ",\n";
                json << "    \"task_primary_label\": ";
                emit_string(s.task_primary_label);
                json << ",\n";
                json << "    \"task_secondary_label\": ";
                emit_string(s.task_secondary_label);
                json << ",\n";
                json << "    \"sensor_responsiveness\": ";
                emit_number(s.sensor_responsiveness);
                json << ",\n";
                json << "    \"signal_connectivity\": ";
                emit_number(s.signal_connectivity);
                json << ",\n";
                json << "    \"force_correlation\": ";
                emit_number(s.force_correlation);
                json << ",\n";
                json << "    \"adaptation_gain\": ";
                emit_number(s.adaptation_gain);
                json << ",\n";
                json << "    \"mean_force_abs\": ";
                emit_number(s.mean_force_abs);
                json << ",\n";
                json << "    \"mean_energy\": ";
                emit_number(s.mean_energy);
                json << "\n";
                json << "  }" << (comma ? "," : "") << "\n";
            };
            auto emit_delta_array = [&](const char* key, const std::vector<ODDDeltaSummary>& items, const bool comma) {
                json << "    \"" << key << "\": [\n";
                for (std::size_t i = 0; i < items.size(); ++i) {
                    const auto& item = items[i];
                    json << "      {\n";
                    json << "        \"key\": ";
                    emit_string(item.key);
                    json << ",\n";
                    json << "        \"label\": ";
                    emit_string(item.label);
                    json << ",\n";
                    json << "        \"baseline_survival_ratio\": ";
                    emit_number(item.baseline_survival_ratio);
                    json << ",\n";
                    json << "        \"variant_survival_ratio\": ";
                    emit_number(item.variant_survival_ratio);
                    json << ",\n";
                    json << "        \"delta_survival_ratio\": ";
                    emit_number(item.delta_survival_ratio);
                    json << ",\n";
                    json << "        \"baseline_success_rate\": ";
                    emit_number(item.baseline_success_rate);
                    json << ",\n";
                    json << "        \"variant_success_rate\": ";
                    emit_number(item.variant_success_rate);
                    json << ",\n";
                    json << "        \"delta_success_rate\": ";
                    emit_number(item.delta_success_rate);
                    json << ",\n";
                    json << "        \"relative_importance\": ";
                    emit_number(odd_relative_importance(items, i));
                    json << "\n";
                    json << "      }" << ((i + 1 < items.size()) ? "," : "") << "\n";
                }
                json << "    ]" << (comma ? "," : "") << "\n";
            };
            auto emit_corr_array = [&](const char* key, const std::vector<ODDCorrelationSummary>& items, const bool comma) {
                json << "    \"" << key << "\": [\n";
                for (std::size_t i = 0; i < items.size(); ++i) {
                    const auto& item = items[i];
                    json << "      {\n";
                    json << "        \"left\": ";
                    emit_string(item.left);
                    json << ",\n";
                    json << "        \"right\": ";
                    emit_string(item.right);
                    json << ",\n";
                    json << "        \"correlation\": ";
                    emit_number(item.correlation);
                    json << "\n";
                    json << "      }" << ((i + 1 < items.size()) ? "," : "") << "\n";
                }
                json << "    ]" << (comma ? "," : "") << "\n";
            };
            emit_eval("cell_clean", summary.cell_clean, true);
            emit_eval("cell_damaged", summary.cell_damaged, true);
            emit_eval("pd_gold", summary.pd_gold, true);
            emit_eval("rl_best_clean", summary.rl.clean, true);
            emit_eval("rl_best_damaged", summary.rl.damaged, true);
            emit_eval("rl_best_recovered", summary.rl.recovered, true);
            emit_eval("rl_dqn_clean", summary.rl_dqn.clean, true);
            emit_eval("rl_dqn_damaged", summary.rl_dqn.damaged, true);
            emit_eval("rl_dqn_recovered", summary.rl_dqn.recovered, true);
            emit_eval("rl_a2c_clean", summary.rl_a2c.clean, true);
            emit_eval("rl_a2c_damaged", summary.rl_a2c.damaged, true);
            emit_eval("rl_a2c_recovered", summary.rl_a2c.recovered, true);
            json << "  \"odd\": {\n";
            json << "    \"available\": " << (summary.odd.available ? "true" : "false") << ",\n";
            json << "    \"budget\": {\n";
            json << "      \"num_trials\": " << summary.odd.budget.num_trials << ",\n";
            json << "      \"max_ticks\": " << summary.odd.budget.max_ticks << "\n";
            json << "    },\n";
            json << "    \"baseline_clean\": {\n";
            json << "      \"survival_ratio\": ";
            emit_number(summary.odd.baseline_clean.survival_ratio);
            json << ",\n";
            json << "      \"success_rate\": ";
            emit_number(summary.odd.baseline_clean.success_rate);
            json << ",\n";
            json << "      \"mean_ticks\": ";
            emit_number(summary.odd.baseline_clean.mean_ticks);
            json << "\n";
            json << "    },\n";
            emit_delta_array("mechanism_deltas", summary.odd.mechanism_deltas, true);
            emit_delta_array("gene_channel_deltas", summary.odd.gene_channel_deltas, true);
            emit_delta_array("cell_force_deltas", summary.odd.cell_force_deltas, true);
            emit_delta_array("edge_deltas", summary.odd.edge_deltas, true);
            emit_corr_array("temporal_correlations", summary.odd.temporal_correlations, true);
            emit_corr_array("structural_confounds", summary.odd.structural_confounds, true);
            json << "    \"notes\": ";
            emit_string(summary.odd.notes);
            json << "\n";
            json << "  }\n";
            json << "}\n";
        }
        {
            std::ofstream csv(fs::path(output_dir) / "champion_genome.csv");
            csv << "gene,value,param_name\n";
            for (std::size_t i = 0; i < summary.champion_genes.size(); ++i) {
                csv << i << ',' << summary.champion_genes[i] << ",\n";
            }
            csv << "-1," << body_mode_code(summary.champion_params.body_mode) << ",body_mode_code\n";
            csv << "-2," << summary.champion_params.development_steps << ",development_steps\n";
            csv << "-3," << summary.champion_params.development_seed_half_width << ",development_seed_half_width\n";
            csv << "-4," << summary.champion_params.max_cells << ",max_cells\n";
            csv << "-5," << summary.champion_params.body_extent_x << ",body_extent_x\n";
            csv << "-6," << summary.champion_params.body_extent_y << ",body_extent_y\n";
            csv << "-7," << summary.champion_params.body_extent_z << ",body_extent_z\n";
            csv << "-8," << summary.champion_params.chemical_diffusion_steps << ",chemical_diffusion_steps\n";
            csv << "-9," << summary.champion_params.development_growth_threshold << ",development_growth_threshold\n";
            csv << "-10," << summary.champion_params.chemical_diffusion_rate << ",chemical_diffusion_rate\n";
            csv << "-11," << summary.champion_params.chemical_decay << ",chemical_decay\n";
        }
        {
            Genome champion{};
            champion.genes = summary.champion_genes;
            champion.params = summary.champion_params;
            champion.has_params = true;
            write_decoded_genome_json(champion,
                                      summary.champion_params,
                                      (fs::path(output_dir) / "decoded_genome.json").string());
            const bool clean_teacher_ok =
                write_named_trial_context_bundle(champion, cfg, true, false, 61000, output_dir, "clean_teacher");
            const bool clean_autonomous_ok =
                write_named_trial_context_bundle(champion, cfg, false, false, 61100, output_dir, "clean_autonomous");
            const bool damaged_autonomous_ok =
                write_named_trial_context_bundle(champion, cfg, false, true, 61200, output_dir, "damaged_autonomous");
            if (!(clean_teacher_ok && clean_autonomous_ok && damaged_autonomous_ok)) {
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

