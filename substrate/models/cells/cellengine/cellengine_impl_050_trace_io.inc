double compute_force(const CellPopulation& pop, const BodyTemplate& body, const ODDIntervention* intervention) {
    double force = 0.0;
    for (int i = 0; i < pop.count; ++i) {
        if (intervention && intervention->ablate_force_cell == i) continue;
        if (pop.active[static_cast<std::size_t>(i)] == 0) continue;
        const auto& c = body.cells[static_cast<std::size_t>(i)];
        force += pop.a_mech[static_cast<std::size_t>(i)] * c.force_direction_x * c.mech_advantage;
    }
    return 16.0 * force;
}

ReplayFrame make_replay_frame(const CellPopulation& pop, const TaskEnv& env, const bool damage_event) {
    ReplayFrame frame;
    frame.tick = env.ticks;
    frame.x = env.s.x;
    frame.theta = env.s.theta;
    frame.x_dot = env.s.x_dot;
    frame.theta_dot = env.s.theta_dot;
    frame.x_ddot = env.x_ddot;
    frame.theta_ddot = env.theta_ddot;
    frame.organism_force = env.organism_force;
    frame.teacher_force = env.teacher_force;
    frame.total_force = clampd(env.organism_force + env.teacher_force, -env.force_limit, env.force_limit);
    double energy_sum = 0.0;
    double stress_sum = 0.0;
    double activity_sum = 0.0;
    double calcium_sum = 0.0;
    int active_count = 0;
    for (int i = 0; i < pop.count; ++i) {
        if (pop.active[static_cast<std::size_t>(i)] == 0) continue;
        energy_sum += pop.energy[static_cast<std::size_t>(i)];
        stress_sum += pop.stress[static_cast<std::size_t>(i)];
        activity_sum += pop.activity_avg[static_cast<std::size_t>(i)];
        calcium_sum += pop.Ca[static_cast<std::size_t>(i)];
        active_count += 1;
    }
    if (active_count > 0) {
        const double denom = static_cast<double>(active_count);
        frame.mean_energy = energy_sum / denom;
        frame.mean_stress = stress_sum / denom;
        frame.mean_activity = activity_sum / denom;
        frame.mean_calcium = calcium_sum / denom;
    }
    frame.active_fraction = (pop.count > 0) ? static_cast<double>(active_count) / static_cast<double>(pop.count) : 0.0;
    frame.task_aux_a = env.task_aux_a;
    frame.task_aux_b = env.task_aux_b;
    if (env.kind == TaskKind::worm_drag_race || env.kind == TaskKind::pong_return) {
        frame.task_primary = env.primary_metric(TrialBudget{1, 1});
        frame.task_secondary = env.secondary_metric();
    } else {
        frame.task_primary = std::abs(env.s.theta);
        frame.task_secondary = std::abs(env.s.x);
    }
    frame.task_counter = env.task_counter;
    frame.task_success = env.success;
    frame.damage_event = damage_event;
    frame.terminal = env.terminal;
    return frame;
}

void draw_line(std::vector<std::string>& canvas, int x0, int y0, int x1, int y1, char ch) {
    const int dx = std::abs(x1 - x0);
    const int sx = (x0 < x1) ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;
    while (true) {
        if (y0 >= 0 && y0 < static_cast<int>(canvas.size()) && x0 >= 0 && x0 < static_cast<int>(canvas[0].size())) {
            canvas[static_cast<std::size_t>(y0)][static_cast<std::size_t>(x0)] = ch;
        }
        if (x0 == x1 && y0 == y1) break;
        const int e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

std::string ascii_frame(const ReplayFrame& frame, const int max_ticks) {
    constexpr int width = 67;
    constexpr int height = 18;
    constexpr double world_half_window = 4.0;
    std::vector<std::string> canvas(static_cast<std::size_t>(height), std::string(static_cast<std::size_t>(width), ' '));

    const int rail_y = height - 4;
    for (int x = 0; x < width; ++x) canvas[static_cast<std::size_t>(rail_y)][static_cast<std::size_t>(x)] = '=';

    auto world_to_col = [&](const double world_x, const double camera_x) {
        const double rel = clampd((world_x - camera_x) / world_half_window, -1.0, 1.0);
        return static_cast<int>(std::llround((width - 1) * (0.5 + 0.48 * rel)));
    };

    const double camera_x = frame.x;
    const int cart_cx = world_to_col(frame.x, camera_x);
    const int origin_x = world_to_col(0.0, camera_x);
    const int cart_w = 7;
    const int cart_h = 2;
    const int cart_left = std::max(1, cart_cx - cart_w / 2);
    const int cart_right = std::min(width - 2, cart_left + cart_w - 1);
    const int cart_top = rail_y - cart_h;
    for (int x = cart_left; x <= cart_right; ++x) {
        canvas[static_cast<std::size_t>(cart_top)][static_cast<std::size_t>(x)] = '#';
        canvas[static_cast<std::size_t>(cart_top + 1)][static_cast<std::size_t>(x)] = '#';
    }
    canvas[static_cast<std::size_t>(cart_top + 1)][static_cast<std::size_t>(std::max(cart_left, cart_cx - 2))] = '[';
    canvas[static_cast<std::size_t>(cart_top + 1)][static_cast<std::size_t>(std::min(cart_right, cart_cx + 2))] = ']';
    if (origin_x >= 0 && origin_x < width) {
        canvas[static_cast<std::size_t>(rail_y)][static_cast<std::size_t>(origin_x)] = '|';
    }

    const int pivot_x = cart_cx;
    const int pivot_y = cart_top;
    const int pole_len = 9;
    const int tip_x = static_cast<int>(std::llround(static_cast<double>(pivot_x) + pole_len * std::sin(frame.theta)));
    const int tip_y = static_cast<int>(std::llround(static_cast<double>(pivot_y) - pole_len * std::cos(frame.theta)));
    const char pole_char = (frame.theta < -0.10) ? '/' : ((frame.theta > 0.10) ? '\\' : '|');
    draw_line(canvas, pivot_x, pivot_y, tip_x, tip_y, pole_char);
    if (tip_y >= 0 && tip_y < height && tip_x >= 0 && tip_x < width) {
        canvas[static_cast<std::size_t>(tip_y)][static_cast<std::size_t>(tip_x)] = 'o';
    }

    std::ostringstream oss;
    oss << "tick " << frame.tick << "/" << max_ticks
        << "  theta_deg=" << std::fixed << std::setprecision(2) << (frame.theta * 180.0 / kPi)
        << "  x=" << std::setprecision(3) << frame.x
        << "  force=" << std::setprecision(3) << frame.organism_force
        << "  energy=" << std::setprecision(3) << frame.mean_energy
        << (frame.terminal ? "  TERMINAL" : "");
    std::ostringstream out;
    out << oss.str() << "\n";
    for (const auto& row : canvas) out << row << "\n";
    return out.str();
}

bool write_replay_trace_csv(const std::vector<ReplayFrame>& frames, const std::string& path) {
    try {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream csv(path);
        csv << "tick,x,theta_rad,theta_deg,x_dot,theta_dot,x_ddot,theta_ddot,organism_force,teacher_force,total_force,mean_energy,mean_stress,mean_activity,mean_calcium,active_fraction,task_aux_a,task_aux_b,task_primary,task_secondary,task_counter,task_success,damage_event,terminal\n";
        for (const auto& f : frames) {
            csv << f.tick << ','
                << f.x << ','
                << f.theta << ','
                << (f.theta * 180.0 / kPi) << ','
                << f.x_dot << ','
                << f.theta_dot << ','
                << f.x_ddot << ','
                << f.theta_ddot << ','
                << f.organism_force << ','
                << f.teacher_force << ','
                << f.total_force << ','
                << f.mean_energy << ','
                << f.mean_stress << ','
                << f.mean_activity << ','
                << f.mean_calcium << ','
                << f.active_fraction << ','
                << f.task_aux_a << ','
                << f.task_aux_b << ','
                << f.task_primary << ','
                << f.task_secondary << ','
                << f.task_counter << ','
                << (f.task_success ? 1 : 0) << ','
                << (f.damage_event ? 1 : 0) << ','
                << (f.terminal ? 1 : 0) << '\n';
        }
    } catch (...) {
        return false;
    }
    return true;
}

bool write_replay_body_csv(const BodyTemplate& body, const CellPopulation& pop, const std::string& path) {
    try {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream csv(path);
        csv << "cell_id,x,y,z,birth_step,hinge,ground,motor,mech_advantage,force_direction_x,feature_x_norm,feature_y_norm,feature_hinge_prox,feature_surface_prox,feature_neighbor_density";
        for (int gidx = 0; gidx < kGeneExprDim; ++gidx) {
            csv << ",gene_expr_" << gidx;
        }
        csv << '\n';
        for (std::size_t i = 0; i < body.cells.size(); ++i) {
            const auto& c = body.cells[i];
            csv << i << ','
                << c.x << ','
                << c.y << ','
                << c.z << ','
                << c.birth_step << ','
                << (c.hinge ? 1 : 0) << ','
                << (c.ground ? 1 : 0) << ','
                << (c.motor ? 1 : 0) << ','
                << c.mech_advantage << ','
                << c.force_direction_x << ','
                << c.features[0] << ','
                << c.features[1] << ','
                << c.features[2] << ','
                << c.features[3] << ','
                << c.features[4];
            for (int gidx = 0; gidx < kGeneExprDim; ++gidx) {
                csv << ',' << pop.G[i * kGeneExprDim + static_cast<std::size_t>(gidx)];
            }
            csv << '\n';
        }
    } catch (...) {
        return false;
    }
    return true;
}

void write_candidate_params_json(std::ostream& json, const CellEngineConfig& cfg) {
    json << "{";
    json << "\"body_mode\":\"" << cfg.body_mode << "\",";
    json << "\"development_steps\":" << cfg.development_steps << ",";
    json << "\"development_seed_half_width\":" << cfg.development_seed_half_width << ",";
    json << "\"max_cells\":" << cfg.max_cells << ",";
    json << "\"body_extent_x\":" << cfg.body_extent_x << ",";
    json << "\"body_extent_y\":" << cfg.body_extent_y << ",";
    json << "\"body_extent_z\":" << cfg.body_extent_z << ",";
    json << "\"chemical_diffusion_steps\":" << cfg.chemical_diffusion_steps << ",";
    json << "\"development_growth_threshold\":" << cfg.development_growth_threshold << ",";
    json << "\"chemical_diffusion_rate\":" << cfg.chemical_diffusion_rate << ",";
    json << "\"chemical_decay\":" << cfg.chemical_decay;
    json << "}";
}

void write_eval_summary_json(std::ostream& json, const EvaluationSummary& s) {
    auto emit_num = [&](const double value) {
        if (std::isfinite(value)) json << value;
        else json << "null";
    };
    json << "{";
    json << "\"total_ticks\":";
    emit_num(s.total_ticks);
    json << ",\"max_ticks\":";
    emit_num(s.max_ticks);
    json << ",\"survival_ratio\":";
    emit_num(s.survival_ratio);
    json << ",\"mean_ticks\":";
    emit_num(s.mean_ticks);
    json << ",\"success_rate\":";
    emit_num(s.success_rate);
    json << ",\"task_primary\":";
    emit_num(s.task_primary);
    json << ",\"task_secondary\":";
    emit_num(s.task_secondary);
    json << ",\"task_primary_label\":\"" << s.task_primary_label << "\"";
    json << ",\"task_secondary_label\":\"" << s.task_secondary_label << "\"";
    json << "}";
}

void write_body_cells_json(std::ostream& json, const BodyTemplate& body, const CellPopulation& pop) {
    json << "[";
    for (std::size_t i = 0; i < body.cells.size(); ++i) {
        if (i > 0) json << ",";
        const auto& c = body.cells[i];
        json << "{";
        json << "\"cell_id\":" << i << ",";
        json << "\"x\":" << c.x << ",";
        json << "\"y\":" << c.y << ",";
        json << "\"z\":" << c.z << ",";
        json << "\"birth_step\":" << c.birth_step << ",";
        json << "\"hinge\":" << (c.hinge ? 1 : 0) << ",";
        json << "\"ground\":" << (c.ground ? 1 : 0) << ",";
        json << "\"motor\":" << (c.motor ? 1 : 0) << ",";
        json << "\"mech_advantage\":" << c.mech_advantage << ",";
        json << "\"force_direction_x\":" << c.force_direction_x << ",";
        json << "\"feature_x_norm\":" << c.features[0] << ",";
        json << "\"feature_y_norm\":" << c.features[1] << ",";
        json << "\"feature_hinge_prox\":" << c.features[2] << ",";
        json << "\"feature_surface_prox\":" << c.features[3] << ",";
        json << "\"feature_neighbor_density\":" << c.features[4];
        for (int gidx = 0; gidx < kGeneExprDim; ++gidx) {
            json << ",\"gene_expr_" << gidx << "\":" << pop.G[i * kGeneExprDim + static_cast<std::size_t>(gidx)];
        }
        json << "}";
    }
    json << "]";
}

bool write_live_discovery_snapshot(const Genome& genome,
                                   const CellEngineConfig& cfg,
                                   const int current_generation,
                                   const int best_generation,
                                   const double best_search_score,
                                   const EvaluationSummary* best_summary,
                                   const std::string& output_dir) {
    try {
        namespace fs = std::filesystem;
        const CellEngineConfig resolved_cfg = resolve_candidate_cfg(cfg, genome);
        const BodyTemplate body = make_body_template(genome, resolved_cfg);
        const CellPopulation pop = build_population(genome, body);
        const fs::path out_dir(output_dir);
        fs::create_directories(out_dir);
        write_replay_body_csv(body, pop, (out_dir / "live_body.csv").string());
        std::ofstream json(out_dir / "live_meta.json");
        json << "{\n";
        json << "  \"current_generation\": " << current_generation << ",\n";
        json << "  \"best_generation\": " << best_generation << ",\n";
        json << "  \"best_search_score\": ";
        if (std::isfinite(best_search_score)) json << best_search_score;
        else json << "null";
        json << ",\n";
        json << "  \"body_mode\": \"" << resolved_cfg.body_mode << "\",\n";
        json << "  \"cell_count\": " << body.cell_count() << ",\n";
        json << "  \"body_depth\": " << body.body_depth() << ",\n";
        json << "  \"candidate_params\": ";
        write_candidate_params_json(json, resolved_cfg);
        json << ",\n";
        json << "  \"best_summary\": ";
        if (best_summary) write_eval_summary_json(json, *best_summary);
        else json << "null";
        json << "\n";
        json << "}\n";
    } catch (...) {
        return false;
    }
    return true;
}

bool write_live_discovery_population_snapshot(const std::vector<Genome>& population,
                                              const std::vector<double>& fitness,
                                              const std::vector<EvaluationSummary>& summaries,
                                              const std::vector<int>& order,
                                              const CellEngineConfig& cfg,
                                              const int current_generation,
                                              const int best_generation,
                                              const double best_search_score,
                                              const std::string& output_dir) {
    try {
        namespace fs = std::filesystem;
        const fs::path out_dir(output_dir);
        fs::create_directories(out_dir);
        const fs::path tmp_path = out_dir / "live_population.json.tmp";
        const fs::path final_path = out_dir / "live_population.json";
        std::ofstream json(tmp_path);
        json << "{\n";
        json << "  \"current_generation\": " << current_generation << ",\n";
        json << "  \"best_generation\": " << best_generation << ",\n";
        json << "  \"best_search_score\": ";
        if (std::isfinite(best_search_score)) json << best_search_score;
        else json << "null";
        json << ",\n";
        json << "  \"population_size\": " << order.size() << ",\n";
        json << "  \"candidates\": [\n";
        for (std::size_t rank = 0; rank < order.size(); ++rank) {
            const int candidate_index = order[rank];
            const Genome& genome = population[static_cast<std::size_t>(candidate_index)];
            const CellEngineConfig resolved_cfg = resolve_candidate_cfg(cfg, genome);
            const BodyTemplate body = make_body_template(genome, resolved_cfg);
            const CellPopulation candidate_pop = build_population(genome, body);
            if (rank > 0) json << ",\n";
            json << "    {\n";
            json << "      \"candidate_index\": " << candidate_index << ",\n";
            json << "      \"rank\": " << (rank + 1) << ",\n";
            json << "      \"fitness\": ";
            if (candidate_index >= 0 &&
                static_cast<std::size_t>(candidate_index) < fitness.size() &&
                std::isfinite(fitness[static_cast<std::size_t>(candidate_index)])) {
                json << fitness[static_cast<std::size_t>(candidate_index)];
            } else {
                json << "null";
            }
            json << ",\n";
            json << "      \"summary\": ";
            if (candidate_index >= 0 &&
                static_cast<std::size_t>(candidate_index) < summaries.size()) {
                write_eval_summary_json(json, summaries[static_cast<std::size_t>(candidate_index)]);
            } else {
                json << "null";
            }
            json << ",\n";
            json << "      \"body_mode\": \"" << resolved_cfg.body_mode << "\",\n";
            json << "      \"cell_count\": " << body.cell_count() << ",\n";
            json << "      \"body_depth\": " << body.body_depth() << ",\n";
            json << "      \"candidate_params\": ";
            write_candidate_params_json(json, resolved_cfg);
            json << ",\n";
            json << "      \"body_cells\": ";
            write_body_cells_json(json, body, candidate_pop);
            json << "\n";
            json << "    }";
        }
        json << "\n  ]\n";
        json << "}\n";
        json.close();
        std::error_code ec;
        fs::remove(final_path, ec);
        fs::rename(tmp_path, final_path, ec);
        if (ec) return false;
    } catch (...) {
        return false;
    }
    return true;
}

bool write_replay_cells_csv(const std::vector<ReplayCellState>& rows, const std::string& path) {
    try {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream csv(path);
        csv << "tick,cell_id,active,V,Wrec,Ca,sigma,energy,a_mech,chem_out,stress,activity,"
               "I_elec,I_chem,I_mech,I_self_chem,I_total,"
               "stress_hinge,stress_lateral,stress_gravity,stress_drift,stress_neighbor,stress_final,"
               "mech_a_gate,mech_contract,mech_prefactor,mech_force_contrib\n";
        for (const auto& row : rows) {
            csv << row.tick << ','
                << row.cell_id << ','
                << row.active << ','
                << row.V << ','
                << row.Wrec << ','
                << row.Ca << ','
                << row.sigma << ','
                << row.energy << ','
                << row.a_mech << ','
                << row.chem_out << ','
                << row.stress << ','
                << row.activity << ','
                << row.I_elec << ','
                << row.I_chem << ','
                << row.I_mech << ','
                << row.I_self_chem << ','
                << row.I_total << ','
                << row.stress_hinge << ','
                << row.stress_lateral << ','
                << row.stress_gravity << ','
                << row.stress_drift << ','
                << row.stress_neighbor << ','
                << row.stress_final << ','
                << row.mech_a_gate << ','
                << row.mech_contract << ','
                << row.mech_prefactor << ','
                << row.mech_force_contrib << '\n';
        }
    } catch (...) {
        return false;
    }
    return true;
}

bool write_replay_edges_csv(const std::vector<ReplayEdgeState>& rows, const std::string& path) {
    try {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream csv(path);
        csv << "tick,src_cell_id,dst_cell_id,src_active,dst_active,gap_forward,gap_reverse,gap_mean,"
               "corr_forward,corr_reverse,hebb_delta_forward,hebb_delta_reverse,"
               "decay_delta_forward,decay_delta_reverse,old_gap_forward,old_gap_reverse,"
               "new_gap_forward,new_gap_reverse\n";
        for (const auto& row : rows) {
            csv << row.tick << ','
                << row.src_cell_id << ','
                << row.dst_cell_id << ','
                << row.src_active << ','
                << row.dst_active << ','
                << row.gap_forward << ','
                << row.gap_reverse << ','
                << row.gap_mean << ','
                << row.corr_forward << ','
                << row.corr_reverse << ','
                << row.hebb_delta_forward << ','
                << row.hebb_delta_reverse << ','
                << row.decay_delta_forward << ','
                << row.decay_delta_reverse << ','
                << row.old_gap_forward << ','
                << row.old_gap_reverse << ','
                << row.new_gap_forward << ','
                << row.new_gap_reverse << '\n';
        }
    } catch (...) {
        return false;
    }
    return true;
}

bool write_rl_replay_trace_csv(const std::vector<ReplayFrame>& frames,
                               const std::vector<int>& actions,
                               const std::string& algorithm,
                               const std::string& path) {
    try {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream csv(path);
        csv << "tick,x,theta_rad,theta_deg,x_dot,theta_dot,x_ddot,theta_ddot,organism_force,teacher_force,total_force,mean_energy,mean_stress,mean_activity,mean_calcium,active_fraction,task_aux_a,task_aux_b,task_primary,task_secondary,task_counter,task_success,damage_event,terminal,action,algorithm\n";
        for (std::size_t i = 0; i < frames.size(); ++i) {
            const auto& f = frames[i];
            csv << f.tick << ','
                << f.x << ','
                << f.theta << ','
                << (f.theta * 180.0 / kPi) << ','
                << f.x_dot << ','
                << f.theta_dot << ','
                << f.x_ddot << ','
                << f.theta_ddot << ','
                << f.organism_force << ','
                << f.teacher_force << ','
                << f.total_force << ','
                << f.mean_energy << ','
                << f.mean_stress << ','
                << f.mean_activity << ','
                << f.mean_calcium << ','
                << f.active_fraction << ','
                << f.task_aux_a << ','
                << f.task_aux_b << ','
                << f.task_primary << ','
                << f.task_secondary << ','
                << f.task_counter << ','
                << (f.task_success ? 1 : 0) << ','
                << (f.damage_event ? 1 : 0) << ','
                << (f.terminal ? 1 : 0) << ','
                << (i < actions.size() ? actions[i] : -1) << ','
                << algorithm << '\n';
        }
    } catch (...) {
        return false;
    }
    return true;
}

struct ReplayCurrentDecomposition {
    double I_elec = 0.0;
    double I_chem = 0.0;
    double I_mech = 0.0;
    double I_self_chem = 0.0;
    double I_total = 0.0;
};

ReplayCurrentDecomposition compute_replay_current_decomposition(const CellPopulation& pop,
                                                                const BodyTemplate& body,
                                                                const DecodedGenome& d,
                                                                const int cell_id,
                                                                const ODDIntervention* intervention) {
    ReplayCurrentDecomposition terms;
    if (cell_id < 0 || cell_id >= pop.count || pop.active[static_cast<std::size_t>(cell_id)] == 0) {
        return terms;
    }

    const std::size_t idx = static_cast<std::size_t>(cell_id);
    const double g2 = pop.G[idx * kGeneExprDim + 2];
    const double V = pop.V[idx];
    const double energy_gate = std::max(0.05, pop.energy[idx]);
    const int grid_idx = (cell_id < static_cast<int>(body.cell_to_grid.size()))
                             ? body.cell_to_grid[idx]
                             : -1;

    if (!(intervention && intervention->disable_chemical) &&
        !(intervention && intervention->disable_diffusion) &&
        grid_idx >= 0) {
        terms.I_chem += pop.chem_field[static_cast<std::size_t>(grid_idx)];
    }

    const int begin = body.neighbor_offsets[idx];
    const int end = body.neighbor_offsets[idx + 1];
    for (int p = begin; p < end; ++p) {
        const int j = body.neighbor_flat[static_cast<std::size_t>(p)];
        if (edge_ablation_matches(intervention, cell_id, j)) continue;
        if (pop.active[static_cast<std::size_t>(j)] == 0) continue;
        const double wgap = pop.gap[static_cast<std::size_t>(p)];
        if (!(intervention && intervention->disable_electrical)) {
            terms.I_elec += wgap * (pop.V[static_cast<std::size_t>(j)] - V);
        }
        if ((intervention && intervention->disable_diffusion) &&
            !(intervention && intervention->disable_chemical)) {
            terms.I_chem += wgap * pop.chem_out[static_cast<std::size_t>(j)];
        }
    }

    if (!(intervention && intervention->disable_mechanical)) {
        terms.I_mech = d.mech_base * (0.50 + 2.20 * g2) * pop.stress[idx];
    }
    if (!(intervention && (intervention->disable_chemical || intervention->disable_self_chemical))) {
        terms.I_self_chem = d.chemical_balance * pop.chem_out[idx];
    }

    terms.I_total =
        pop.sigma[idx] *
        (terms.I_elec + d.chem_coeff * terms.I_chem + terms.I_mech + terms.I_self_chem) *
        energy_gate;
    return terms;
}

void append_replay_cell_state(const CellPopulation& pop,
                              const BodyTemplate& body,
                              const DecodedGenome& d,
                              const int tick,
                              const ODDIntervention* intervention,
                              std::vector<ReplayCellState>& rows) {
    rows.reserve(rows.size() + static_cast<std::size_t>(pop.count));
    for (int i = 0; i < pop.count; ++i) {
        const ReplayCurrentDecomposition terms =
            compute_replay_current_decomposition(pop, body, d, i, intervention);
        rows.push_back({
            tick,
            i,
            pop.active[static_cast<std::size_t>(i)],
            pop.V[static_cast<std::size_t>(i)],
            pop.Wrec[static_cast<std::size_t>(i)],
            pop.Ca[static_cast<std::size_t>(i)],
            pop.sigma[static_cast<std::size_t>(i)],
            pop.energy[static_cast<std::size_t>(i)],
            pop.a_mech[static_cast<std::size_t>(i)],
            pop.chem_out[static_cast<std::size_t>(i)],
            pop.stress[static_cast<std::size_t>(i)],
            pop.activity_avg[static_cast<std::size_t>(i)],
            terms.I_elec,
            terms.I_chem,
            terms.I_mech,
            terms.I_self_chem,
            terms.I_total,
            pop.stress_hinge[static_cast<std::size_t>(i)],
            pop.stress_lateral[static_cast<std::size_t>(i)],
            pop.stress_gravity[static_cast<std::size_t>(i)],
            pop.stress_drift[static_cast<std::size_t>(i)],
            pop.stress_neighbor[static_cast<std::size_t>(i)],
            pop.stress[static_cast<std::size_t>(i)],
            pop.mech_a_gate[static_cast<std::size_t>(i)],
            pop.mech_contract[static_cast<std::size_t>(i)],
            pop.mech_prefactor[static_cast<std::size_t>(i)],
            pop.mech_force_contrib[static_cast<std::size_t>(i)]
        });
    }
}

void append_replay_edge_state(const CellPopulation& pop,
                              const BodyTemplate& body,
                              const int tick,
                              std::vector<ReplayEdgeState>& rows) {
    for (int i = 0; i < pop.count; ++i) {
        const int begin = body.neighbor_offsets[static_cast<std::size_t>(i)];
        const int end = body.neighbor_offsets[static_cast<std::size_t>(i + 1)];
        for (int p = begin; p < end; ++p) {
            const int j = body.neighbor_flat[static_cast<std::size_t>(p)];
            if (i >= j) continue;
            double reverse_gap = 0.0;
            int rev_idx = -1;
            const int rev_begin = body.neighbor_offsets[static_cast<std::size_t>(j)];
            const int rev_end = body.neighbor_offsets[static_cast<std::size_t>(j + 1)];
            for (int q = rev_begin; q < rev_end; ++q) {
                if (body.neighbor_flat[static_cast<std::size_t>(q)] == i) {
                    rev_idx = q;
                    reverse_gap = pop.gap[static_cast<std::size_t>(q)];
                    break;
                }
            }
            const double forward_gap = pop.gap[static_cast<std::size_t>(p)];
            const double reverse_corr = (rev_idx >= 0) ? pop.gap_corr_last[static_cast<std::size_t>(rev_idx)] : 0.0;
            const double reverse_hebb = (rev_idx >= 0) ? pop.gap_hebb_delta_last[static_cast<std::size_t>(rev_idx)] : 0.0;
            const double reverse_decay = (rev_idx >= 0) ? pop.gap_decay_delta_last[static_cast<std::size_t>(rev_idx)] : 0.0;
            const double reverse_old = (rev_idx >= 0) ? pop.gap_old_last[static_cast<std::size_t>(rev_idx)] : 0.0;
            const double reverse_new = (rev_idx >= 0) ? pop.gap_new_last[static_cast<std::size_t>(rev_idx)] : 0.0;
            rows.push_back({
                tick,
                i,
                j,
                pop.active[static_cast<std::size_t>(i)],
                pop.active[static_cast<std::size_t>(j)],
                forward_gap,
                reverse_gap,
                0.5 * (forward_gap + reverse_gap),
                pop.gap_corr_last[static_cast<std::size_t>(p)],
                reverse_corr,
                pop.gap_hebb_delta_last[static_cast<std::size_t>(p)],
                reverse_hebb,
                pop.gap_decay_delta_last[static_cast<std::size_t>(p)],
                reverse_decay,
                pop.gap_old_last[static_cast<std::size_t>(p)],
                reverse_old,
                pop.gap_new_last[static_cast<std::size_t>(p)],
                reverse_new
            });
        }
    }
}

std::tuple<TrialStats, std::vector<ReplayFrame>, std::vector<ReplayCellState>, std::vector<ReplayEdgeState>>
run_trial_with_trace(CellPopulation& pop,
                     const BodyTemplate& body,
                     const DecodedGenome& d,
                     const CellEngineConfig& cfg,
                     const TrialBudget& budget,
                     std::mt19937& rng,
                     const bool teaching_active,
                     TaskEnv& env,
                     const ODDIntervention* intervention) {
    reset_fast_state(pop, d);
    configure_task_env(env, cfg, d.force_scale);
    if (env.kind == TaskKind::cartpole_balance || env.kind == TaskKind::mass_spring_balance) {
        if (std::isfinite(cfg.replay_theta_deg)) {
            env.reset(rng, cfg.replay_theta_deg * kPi / 180.0);
        } else {
            env.reset(rng);
        }
    } else if (std::isfinite(cfg.replay_task_a)) {
        env.reset(rng, cfg.replay_task_a);
    } else {
        env.reset(rng);
    }
    const bool effective_teaching = teaching_active && !(intervention && intervention->disable_teacher);
    env.set_teacher(effective_teaching, d.teacher_kp, d.teacher_kd, d.teacher_force_scale);
    write_stress(pop, body, d, env, cfg.stress_relax_iters);

    TrialStats stats;
    std::vector<ReplayFrame> frames;
    std::vector<ReplayCellState> cell_rows;
    std::vector<ReplayEdgeState> edge_rows;
    frames.reserve(static_cast<std::size_t>(budget.max_ticks + 1));
    cell_rows.reserve(static_cast<std::size_t>((budget.max_ticks + 1) * std::max(1, pop.count)));
    edge_rows.reserve(static_cast<std::size_t>(budget.max_ticks + 1) *
                      std::max<std::size_t>(1, body.neighbor_flat.size() / 2));
    frames.push_back(make_replay_frame(pop, env));
    append_replay_cell_state(pop, body, d, env.ticks, intervention, cell_rows);
    append_replay_edge_state(pop, body, env.ticks, edge_rows);

    const double cell_dt = cfg.physics_dt / static_cast<double>(cfg.cell_ticks_per_physics);
    double force_prev = 0.0;
    bool damage_applied = false;
    bool damage_event_pending = false;

    while (!env.terminal && env.ticks < budget.max_ticks) {
        if (!damage_applied && cfg.replay_damage_tick >= 0 && env.ticks == cfg.replay_damage_tick) {
            apply_damage(pop, body, cfg.damage_fraction, rng);
            damage_applied = true;
            damage_event_pending = true;
        }
        for (int k = 0; k < cfg.cell_ticks_per_physics; ++k) {
            cell_tick(pop, body, d, cfg, cell_dt, effective_teaching, intervention, true);
        }
        double force_cmd = compute_force(pop, body, intervention);
        force_cmd = d.force_smoothing * force_prev + (1.0 - d.force_smoothing) * force_cmd;
        force_prev = force_cmd;
        env.step(force_cmd, cfg.physics_dt);
        write_stress(pop, body, d, env, cfg.stress_relax_iters);
        stats.mean_abs_force += std::abs(env.organism_force);
        stats.mean_abs_theta += std::abs(env.s.theta);
        stats.mean_energy += mean_of(pop.energy);
        frames.push_back(make_replay_frame(pop, env, damage_event_pending));
        append_replay_cell_state(pop, body, d, env.ticks, intervention, cell_rows);
        append_replay_edge_state(pop, body, env.ticks, edge_rows);
        damage_event_pending = false;
    }

    stats.ticks = env.ticks;
    stats.success = env.trial_success(budget);
    stats.task_primary = env.primary_metric(budget);
    stats.task_secondary = env.secondary_metric();
    if (stats.ticks > 0) {
        stats.mean_abs_force /= static_cast<double>(stats.ticks);
        stats.mean_abs_theta /= static_cast<double>(stats.ticks);
        stats.mean_energy /= static_cast<double>(stats.ticks);
    }
    slow_update(pop, body, d, intervention);
    return {stats, frames, cell_rows, edge_rows};
}

TrialStats run_trial(CellPopulation& pop,
                     const BodyTemplate& body,
                     const DecodedGenome& d,
                     const CellEngineConfig& cfg,
                     const TrialBudget& budget,
                     std::mt19937& rng,
                     const bool teaching_active,
                     TaskEnv& env,
                     const ODDIntervention* intervention) {
    reset_fast_state(pop, d);
    configure_task_env(env, cfg, d.force_scale);
    env.reset(rng);
    const bool effective_teaching = teaching_active && !(intervention && intervention->disable_teacher);
    env.set_teacher(effective_teaching, d.teacher_kp, d.teacher_kd, d.teacher_force_scale);
    write_stress(pop, body, d, env, cfg.stress_relax_iters);

    TrialStats stats;
    const double cell_dt = cfg.physics_dt / static_cast<double>(cfg.cell_ticks_per_physics);
    double force_prev = 0.0;

    while (!env.terminal && env.ticks < budget.max_ticks) {
        for (int k = 0; k < cfg.cell_ticks_per_physics; ++k) {
            cell_tick(pop, body, d, cfg, cell_dt, effective_teaching, intervention);
        }
        double force_cmd = compute_force(pop, body, intervention);
        force_cmd = d.force_smoothing * force_prev + (1.0 - d.force_smoothing) * force_cmd;
        force_prev = force_cmd;
        env.step(force_cmd, cfg.physics_dt);
        write_stress(pop, body, d, env, cfg.stress_relax_iters);
        stats.mean_abs_force += std::abs(env.organism_force);
        stats.mean_abs_theta += std::abs(env.s.theta);
        stats.mean_energy += mean_of(pop.energy);
    }

    stats.ticks = env.ticks;
    stats.success = env.trial_success(budget);
    stats.task_primary = env.primary_metric(budget);
    stats.task_secondary = env.secondary_metric();
    if (stats.ticks > 0) {
        stats.mean_abs_force /= static_cast<double>(stats.ticks);
        stats.mean_abs_theta /= static_cast<double>(stats.ticks);
        stats.mean_energy /= static_cast<double>(stats.ticks);
    }
    slow_update(pop, body, d, intervention);
    return stats;
}
