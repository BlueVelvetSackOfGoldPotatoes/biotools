void BioRuntime::begin_epoch(int epoch) {
    epoch_counter_ = epoch;
    epoch_energy_ = 0.0;
    epoch_wiring_cost_ = 0.0;
    epoch_modulator_sum_ = 0.0;
    epoch_reward_sum_ = 0.0;
    epoch_reward_count_ = 0;
    epoch_plasticity_abs_sum_ = 0.0;
    epoch_plasticity_count_ = 0;
    epoch_homeostasis_error_sum_ = 0.0;
    epoch_homeostasis_count_ = 0;
    epoch_batches_ = 0;
    for (auto& state : states_) {
        state.grad_l2_sum = 0.0;
        state.abs_grad_sum = 0.0;
        state.grad_obs = 0;
    }
}

bool BioRuntime::apply_linear_input_transport(const Tensor* weight_param,
                                              const Tensor& input,
                                              Tensor& effective_input) {
    effective_input = input;
    if (!cfg_.active_parameter_updates || !cfg_.enable_myelination || !weight_param) return false;

    const auto it = param_to_state_.find(weight_param);
    if (it == param_to_state_.end()) return false;
    ParamState& state = states_[it->second];
    if (!state.bio_edge_eligible) return false;

    bool source_rows = state.source_is_rows;
    if (input.cols == state.param->rows) {
        source_rows = true;
    } else if (input.cols == state.param->cols) {
        source_rows = false;
    } else {
        return false;
    }
    if (source_rows != state.source_is_rows) {
        state.source_is_rows = source_rows;
        refresh_source_delay_bins(state);
    } else if (state.src_delay_bin.size() != input.cols) {
        refresh_source_delay_bins(state);
    }
    if (state.src_delay_bin.empty()) return false;

    const std::size_t bins = static_cast<std::size_t>(std::max(1, cfg_.conduction_delay_bins));
    if (state.input_history.size() != bins ||
        state.input_history.empty() ||
        state.input_history.front().rows != input.rows ||
        state.input_history.front().cols != input.cols) {
        state.input_history.assign(bins, input);
    }

    state.input_history.insert(state.input_history.begin(), input);
    if (state.input_history.size() > bins) {
        state.input_history.resize(bins);
    }

    Tensor transported(input.rows, input.cols, 0.0);
    const double base_mix = clamp(cfg_.conduction_delay_mix, 0.0, 1.0);
    for (std::size_t c = 0; c < input.cols; ++c) {
        const std::size_t src_bin =
            c < state.src_delay_bin.size() ? std::min(state.src_delay_bin[c], bins - 1) : 0;
        const Tensor& delayed = state.input_history[src_bin];
        const double bin_frac =
            bins > 1 ? static_cast<double>(src_bin) / static_cast<double>(bins - 1) : 0.0;
        const double mix = base_mix * bin_frac;
        for (std::size_t r = 0; r < input.rows; ++r) {
            transported(r, c) = (1.0 - mix) * input(r, c) + mix * delayed(r, c);
        }
    }
    effective_input = std::move(transported);
    return true;
}

void BioRuntime::record_linear_activity(const Tensor* weight_param,
                                        const Tensor& pre_input,
                                        const Tensor& linear_output) {
    if (!cfg_.active_parameter_updates || !weight_param) return;
    const auto it = param_to_state_.find(weight_param);
    if (it == param_to_state_.end()) return;
    ParamState& state = states_[it->second];
    if (!state.bio_edge_eligible) return;

    bool source_rows = state.source_is_rows;
    if (pre_input.cols == state.param->rows && linear_output.cols == state.param->cols) {
        source_rows = true;
    } else if (pre_input.cols == state.param->cols && linear_output.cols == state.param->rows) {
        source_rows = false;
    } else {
        return;
    }
    if (source_rows != state.source_is_rows) {
        state.source_is_rows = source_rows;
        refresh_source_delay_bins(state);
    }

    state.pre_activity_mean.assign(pre_input.cols, 0.0);
    state.post_activity_mean.assign(linear_output.cols, 0.0);
    if (pre_input.rows == 0 || linear_output.rows == 0) return;

    const double inv_pre_n = 1.0 / static_cast<double>(pre_input.rows);
    const double inv_post_n = 1.0 / static_cast<double>(linear_output.rows);
    for (std::size_t c = 0; c < pre_input.cols; ++c) {
        double s = 0.0;
        for (std::size_t r = 0; r < pre_input.rows; ++r) s += pre_input(r, c);
        state.pre_activity_mean[c] = s * inv_pre_n;
    }
    for (std::size_t c = 0; c < linear_output.cols; ++c) {
        double s = 0.0;
        for (std::size_t r = 0; r < linear_output.rows; ++r) s += linear_output(r, c);
        state.post_activity_mean[c] = s * inv_post_n;
    }
}

void BioRuntime::before_forward() {
    if (!cfg_.active_parameter_updates) {
        set_active_runtime(nullptr);
        return;
    }
    set_active_runtime(this);
    for (auto& state : states_) {
        if (!state.param || state.mask.data.size() != state.param->data.size()) continue;
        // Keep structural/EI constraints hard-satisfied before forward.
        for (std::size_t i = 0; i < state.param->data.size(); ++i) {
            state.param->data[i] *= state.mask.data[i];
        }
        if (cfg_.enable_ei_sign_constraints) {
            enforce_ei_sign(state);
        }
    }
}

double BioRuntime::compute_modulator(double loss, double reward) const {
    double mod = 0.0;
    if (finite(reward)) {
        mod = reward;
    } else if (finite(loss) && finite(prev_loss_)) {
        mod = prev_loss_ - loss;
    } else if (finite(loss) && !finite(prev_loss_)) {
        mod = -loss;
    }
    return clamp(mod, -cfg_.modulator_clip, cfg_.modulator_clip);
}

void BioRuntime::apply_homeostasis(ParamState& state, Tensor& target) {
    state.last_homeostasis_error = 0.0;
    if (!cfg_.enable_homeostasis ||
        !state.bio_edge_eligible || target.rows == 0 || target.cols == 0) {
        return;
    }

    const std::size_t rows = target.rows;
    const std::size_t cols = target.cols;
    const double target_norm = cfg_.homeostasis_target_norm;
    for (std::size_t r = 0; r < rows; ++r) {
        double row_sq = 0.0;
        for (std::size_t c = 0; c < cols; ++c) {
            const double v = target(r, c);
            row_sq += v * v;
        }
        const double row_norm = std::sqrt(row_sq + 1e-12);
        const double err = std::abs(target_norm - row_norm) / (target_norm + 1e-12);
        state.last_homeostasis_error += err;

        if (cfg_.active_parameter_updates) {
            double scale = 1.0 + cfg_.homeostasis_lr * (target_norm - row_norm) / (target_norm + 1e-12);
            scale = clamp(scale, 0.5, 1.5);
            for (std::size_t c = 0; c < cols; ++c) {
                target(r, c) *= scale;
            }
        }
    }
    state.last_homeostasis_error /= static_cast<double>(rows);
}

void BioRuntime::after_backward(double loss, double reward, double /*lr*/) {
    if (states_.empty()) return;
    const double mod = compute_modulator(loss, reward);

    epoch_batches_++;
    epoch_modulator_sum_ += mod;
    if (finite(reward)) {
        epoch_reward_sum_ += reward;
        epoch_reward_count_++;
    }

    for (auto& state : states_) {
        if (!state.param) continue;
        const bool has_grad = state.grad && state.grad->data.size() == state.param->data.size();

        if (has_grad) {
            state.grad_l2_sum += state.grad->norm();
            state.abs_grad_sum += mean_abs(*state.grad);
            state.grad_obs++;
        }

        // In observational mode, only collect gradient statistics — skip the
        // expensive per-element eligibility/bioelectric/homeostasis loop.
        if (!cfg_.active_parameter_updates) continue;
        if (!state.bio_edge_eligible) {
            state.last_homeostasis_error = 0.0;
            continue;
        }

        Tensor* target = state.param;
        const std::size_t n = target->data.size();
        double plasticity_abs = 0.0;
        const bool have_local_activity =
            (state.source_is_rows &&
             state.pre_activity_mean.size() == state.param->rows &&
             state.post_activity_mean.size() == state.param->cols) ||
            (!state.source_is_rows &&
             state.pre_activity_mean.size() == state.param->cols &&
             state.post_activity_mean.size() == state.param->rows);

        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t row = state.param->cols > 0 ? (i / state.param->cols) : 0;
            const std::size_t col = state.param->cols > 0 ? (i % state.param->cols) : 0;
            const std::size_t pre_idx = state.source_is_rows ? row : col;
            const std::size_t post_idx = state.source_is_rows ? col : row;
            const double pre = (have_local_activity && pre_idx < state.pre_activity_mean.size())
                                   ? state.pre_activity_mean[pre_idx]
                                   : 0.0;
            const double post = (have_local_activity && post_idx < state.post_activity_mean.size())
                                    ? state.post_activity_mean[post_idx]
                                    : 0.0;
            const double hebb = pre * post;

            if (cfg_.enable_three_factor) {
                state.eligibility.data[i] =
                    cfg_.eligibility_decay * state.eligibility.data[i] + hebb;
                double delta = cfg_.three_factor_lr * mod * state.eligibility.data[i] * state.glia_plasticity;
                delta *= state.mask.data[i];
                target->data[i] -= delta;
                plasticity_abs += std::abs(delta);
            }

            if (cfg_.enable_bioelectric) {
                state.bioelectric.data[i] =
                    (1.0 - cfg_.bioelectric_leak) * state.bioelectric.data[i] +
                    cfg_.bioelectric_coupling * hebb;
            }
        }

        if (cfg_.enable_homeostasis) {
            apply_homeostasis(state, *target);
            epoch_homeostasis_error_sum_ += state.last_homeostasis_error;
            epoch_homeostasis_count_++;
        } else {
            state.last_homeostasis_error = 0.0;
        }

        const double homeo = state.last_homeostasis_error;
        state.glia_gain = clamp(state.glia_gain + 0.02 * mod - 0.01 * homeo, 0.5, 1.5);
        state.glia_plasticity = clamp(state.glia_plasticity + 0.015 * mod - 0.01 * homeo, 0.1, 2.0);
        state.glia_prune_pressure = clamp(0.35 * homeo + 0.15 * (1.0 - std::abs(mod)), 0.0, 1.0);
        state.glia_support = clamp(1.0 - 0.25 * homeo, 0.3, 1.2);

        epoch_energy_ += plasticity_abs;
        epoch_plasticity_abs_sum_ += plasticity_abs;
        epoch_plasticity_count_ += n;
    }

    set_active_runtime(nullptr);
    if (finite(loss)) prev_loss_ = loss;
}

void BioRuntime::after_optimizer_step() {
    set_active_runtime(nullptr);
    if (!cfg_.active_parameter_updates) return;
    for (auto& state : states_) {
        if (!state.param || state.mask.data.size() != state.param->data.size()) continue;

        // Keep structural mask and E/I constraints hard-satisfied in base params.
        for (std::size_t i = 0; i < state.param->data.size(); ++i) {
            state.param->data[i] *= state.mask.data[i];
        }
        if (cfg_.enable_ei_sign_constraints) {
            enforce_ei_sign(state);
        }
    }
}

void BioRuntime::apply_structural_plasticity(ParamState& state) {
    state.last_pruned = 0;
    state.last_grown = 0;
    if (!cfg_.enable_structural_plasticity || !cfg_.active_parameter_updates ||
        !state.bio_edge_eligible || !state.param || state.param->data.empty()) return;

    struct EdgeScore {
        std::size_t idx = 0;
        double score = 0.0;
    };

    std::vector<std::size_t> candidates;
    const std::size_t total_edges = state.param->data.size();
    const std::size_t stride = std::max<std::size_t>(1, cfg_.wiring_cost_sample_stride);
    candidates.reserve((total_edges + stride - 1) / stride);
    for (std::size_t i = 0; i < total_edges; i += stride) candidates.push_back(i);
    if (cfg_.structural_sample_cap > 0 && candidates.size() > cfg_.structural_sample_cap) {
        std::shuffle(candidates.begin(), candidates.end(), rng_);
        candidates.resize(cfg_.structural_sample_cap);
    }

    std::vector<EdgeScore> active;
    std::vector<EdgeScore> inactive_before;
    active.reserve(candidates.size());
    inactive_before.reserve(candidates.size());

    for (std::size_t i : candidates) {
        const double dist = edge_distance(state, i);
        const double weight_mag = std::abs(state.param->data[i]);
        const double elig = std::abs(state.eligibility.data[i]);
        const double importance = (weight_mag + 0.1 * elig) / (1.0 + cfg_.wiring_distance_cost * dist);
        if (state.mask.data[i] > 0.5) {
            active.push_back({i, importance});
        } else {
            const double grow_score = (1.0 + elig) / (1.0 + dist);
            inactive_before.push_back({i, grow_score});
        }
    }

    if (active.empty()) return;

    const std::size_t prune_n = std::min(
        static_cast<std::size_t>(std::floor(cfg_.prune_fraction * static_cast<double>(active.size()))),
        active.size()
    );
    if (prune_n == 0) return;

    if (prune_n < active.size()) {
        std::nth_element(
            active.begin(),
            active.begin() + static_cast<long>(prune_n),
            active.end(),
            [](const EdgeScore& a, const EdgeScore& b) { return a.score < b.score; }
        );
    } else {
        std::sort(active.begin(), active.end(), [](const EdgeScore& a, const EdgeScore& b) {
            return a.score < b.score;
        });
    }
    for (std::size_t i = 0; i < prune_n; ++i) {
        const std::size_t idx = active[i].idx;
        state.mask.data[idx] = 0.0;
        state.param->data[idx] = 0.0;
        state.myelin.data[idx] = 0.0;
        state.delay.data[idx] = 1.0;
    }
    state.last_pruned = prune_n;

    const std::size_t grow_n = std::min(prune_n, inactive_before.size());
    if (grow_n == 0) return;

    if (grow_n < inactive_before.size()) {
        std::nth_element(
            inactive_before.begin(),
            inactive_before.begin() + static_cast<long>(grow_n),
            inactive_before.end(),
            [](const EdgeScore& a, const EdgeScore& b) { return a.score > b.score; }
        );
    } else {
        std::sort(inactive_before.begin(), inactive_before.end(), [](const EdgeScore& a, const EdgeScore& b) {
            return a.score > b.score;
        });
    }
    std::normal_distribution<double> init_dist(0.0, cfg_.growth_noise_std);
    for (std::size_t i = 0; i < grow_n; ++i) {
        const std::size_t idx = inactive_before[i].idx;
        state.mask.data[idx] = 1.0;
        state.param->data[idx] = init_dist(rng_);
    }
    state.last_grown = grow_n;
    refresh_source_delay_bins(state);
}

void BioRuntime::allocate_myelin_and_delay(ParamState& state) {
    if (!cfg_.enable_myelination || !state.bio_edge_eligible || !state.param || state.param->data.empty()) {
        return;
    }

    struct EdgeScore {
        std::size_t idx = 0;
        double score = 0.0;
    };
    const std::size_t stride = std::max<std::size_t>(1, cfg_.wiring_cost_sample_stride);
    std::vector<EdgeScore> active;
    active.reserve((state.param->data.size() + stride - 1) / stride);
    for (std::size_t i = 0; i < state.param->data.size(); i += stride) {
        if (state.mask.data[i] <= 0.5) continue;
        const double dist = edge_distance(state, i);
        const double score = std::abs(state.param->data[i]) * dist;
        active.push_back({i, score});
    }
    if (active.empty()) return;

    const std::size_t myelin_n = std::min(
        static_cast<std::size_t>(std::floor(cfg_.myelin_budget_fraction * static_cast<double>(active.size()))),
        active.size()
    );
    std::sort(active.begin(), active.end(), [](const EdgeScore& a, const EdgeScore& b) {
        return a.score > b.score;
    });

    for (std::size_t i = 0; i < active.size(); ++i) {
        const std::size_t idx = active[i].idx;
        if (i < myelin_n) {
            state.myelin.data[idx] = 1.0;
        } else {
            state.myelin.data[idx] *= 0.95;
        }
    }

    for (std::size_t i = 0; i < state.param->data.size(); ++i) {
        if (state.mask.data[i] <= 0.5) {
            state.delay.data[i] = 1.0;
            continue;
        }
        const double dist = edge_distance(state, i);
        const double velocity =
            cfg_.conduction_base_speed + cfg_.conduction_myelin_boost * state.myelin.data[i];
        // Discretize physical path length into manageable simulation delay bins.
        state.delay.data[i] = 1.0 + std::log1p(dist) / (velocity + 1e-12);
    }
    refresh_source_delay_bins(state);
}

