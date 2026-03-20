thread_local BioRuntime* g_active_runtime = nullptr;

std::string precise(double v) {
    if (!std::isfinite(v)) return "nan";
    std::ostringstream oss;
    oss << std::setprecision(12) << v;
    return oss.str();
}

BioRuntime* active_runtime() { return g_active_runtime; }
void set_active_runtime(BioRuntime* runtime) { g_active_runtime = runtime; }

BioRuntime::BioRuntime(const std::string& model_family,
                       const std::string& model_variant,
                       int seed,
                       BioConfig config)
    : model_family_(model_family),
      model_variant_(model_variant),
      cfg_(config),
      rng_(seed) {}

bool BioRuntime::finite(double v) {
    return std::isfinite(v);
}

double BioRuntime::clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

std::size_t BioRuntime::grid_side(std::size_t n) {
    if (n == 0) return 1;
    return static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(n))));
}

std::vector<BioRuntime::Vec3> BioRuntime::init_positions(std::size_t n, double z_offset) {
    std::vector<Vec3> out(n);
    const std::size_t side = grid_side(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t x = i % side;
        const std::size_t y = i / side;
        out[i] = Vec3{
            static_cast<double>(x),
            static_cast<double>(y),
            z_offset,
        };
    }
    return out;
}

double BioRuntime::mean_abs(const Tensor& t) {
    if (t.data.empty()) return 0.0;
    double s = 0.0;
    for (double v : t.data) s += std::abs(v);
    return s / static_cast<double>(t.data.size());
}

double BioRuntime::tensor_density01(const Tensor& t) {
    if (t.data.empty()) return 0.0;
    std::size_t nnz = 0;
    for (double v : t.data) {
        if (v > 0.5) nnz++;
    }
    return static_cast<double>(nnz) / static_cast<double>(t.data.size());
}

double BioRuntime::edge_distance(const ParamState& state, std::size_t flat_idx) {
    if (!state.param || state.param->cols == 0 || state.param->rows == 0) return 1.0;
    const std::size_t rows = state.param->rows;
    const std::size_t cols = state.param->cols;
    const std::size_t r = flat_idx / cols;
    const std::size_t c = flat_idx % cols;
    if (r >= rows || c >= cols) return 1.0;
    const Vec3& a = state.src_pos[r];
    const Vec3& b = state.dst_pos[c];
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool BioRuntime::is_ei_constraint_eligible_tensor(const Tensor* t) {
    // Bias-like tensors (1xN / Nx1) should not be treated as synaptic projections.
    return t && t->rows > 1 && t->cols > 1;
}

void BioRuntime::enforce_ei_sign(ParamState& state) const {
    if (!state.ei_constraint_eligible || !state.param || state.param->cols == 0 || state.param->rows == 0) return;
    const std::size_t cols = state.param->cols;
    const std::size_t rows = state.param->rows;
    const bool by_dst = cfg_.ei_constraint_axis == EiConstraintAxis::Destination;
    const double alpha = clamp(cfg_.ei_projection_strength, 0.0, 1.0);
    if (alpha <= 0.0) return;
    const double keep = 1.0 - alpha;
    for (std::size_t idx = 0; idx < state.param->data.size(); ++idx) {
        const std::size_t row = idx / cols;
        const std::size_t col = idx % cols;
        bool excitatory = true;
        if (by_dst) {
            if (col < state.dst_type_exc.size()) excitatory = state.dst_type_exc[col] != 0;
        } else {
            if (row < rows && row < state.src_type_exc.size()) excitatory = state.src_type_exc[row] != 0;
        }
        double& w = state.param->data[idx];
        // Project gradually into the sign-constrained cone to avoid abrupt
        // early-training collapse from hard sign flips.
        if (excitatory) {
            if (w < 0.0) {
                w = (alpha >= 1.0) ? 0.0 : (keep * w);
            }
        } else {
            if (w > 0.0) {
                w = (alpha >= 1.0) ? 0.0 : (keep * w);
            }
        }
    }
}

void BioRuntime::refresh_source_delay_bins(ParamState& state) {
    if (!state.bio_edge_eligible || !state.param || state.param->rows == 0 || state.param->cols == 0) {
        state.src_delay_bin.clear();
        return;
    }

    const std::size_t src_dim = state.source_is_rows ? state.param->rows : state.param->cols;
    const std::size_t dst_dim = state.source_is_rows ? state.param->cols : state.param->rows;
    const std::size_t bins = static_cast<std::size_t>(std::max(1, cfg_.conduction_delay_bins));
    state.src_delay_bin.assign(src_dim, 0);

    for (std::size_t s = 0; s < src_dim; ++s) {
        double wsum = 0.0;
        double dsum = 0.0;
        for (std::size_t d = 0; d < dst_dim; ++d) {
            const std::size_t row = state.source_is_rows ? s : d;
            const std::size_t col = state.source_is_rows ? d : s;
            const std::size_t idx = row * state.param->cols + col;
            if (state.mask.data[idx] <= 0.5) continue;
            const double w = std::abs(state.param->data[idx]) + 1e-9;
            wsum += w;
            dsum += w * state.delay.data[idx];
        }
        const double mean_delay = wsum > 0.0 ? (dsum / wsum) : 1.0;
        const std::size_t b = std::min<std::size_t>(
            bins - 1,
            static_cast<std::size_t>(std::floor(std::max(0.0, mean_delay - 1.0)))
        );
        state.src_delay_bin[s] = b;
    }
}

void BioRuntime::attach(const std::vector<Tensor*>& params,
                        const std::vector<Tensor*>& grads,
                        const std::vector<std::string>& names) {
    states_.clear();
    param_to_state_.clear();
    states_.reserve(params.size());

    std::size_t total_synaptic = 0;
    for (const auto* p : params) {
        if (is_ei_constraint_eligible_tensor(p)) total_synaptic++;
    }

    std::size_t synaptic_seen = 0;
    for (std::size_t i = 0; i < params.size(); ++i) {
        if (!params[i]) continue;
        ParamState state;
        state.param = params[i];
        state.grad = (i < grads.size() ? grads[i] : nullptr);
        state.name = (i < names.size() && !names[i].empty()) ? names[i] : ("param_" + std::to_string(i));

        state.mask = Tensor::ones(state.param->rows, state.param->cols);
        state.eligibility = Tensor::zeros(state.param->rows, state.param->cols);
        state.myelin = Tensor::zeros(state.param->rows, state.param->cols);
        state.delay = Tensor::ones(state.param->rows, state.param->cols);
        state.bioelectric = Tensor::zeros(state.param->rows, state.param->cols);

        state.src_pos = init_positions(state.param->rows, static_cast<double>(i) * 2.0);
        state.dst_pos = init_positions(state.param->cols, static_cast<double>(i) * 2.0 + 1.0);

        const bool synaptic = is_ei_constraint_eligible_tensor(state.param);
        state.bio_edge_eligible = synaptic;
        const std::size_t synaptic_index = synaptic ? synaptic_seen : 0;
        const bool is_input_projection = synaptic && synaptic_index == 0;
        const bool is_output_projection =
            synaptic && total_synaptic > 0 && synaptic_index + 1 == total_synaptic;
        if (synaptic) synaptic_seen++;
        state.ei_constraint_eligible =
            cfg_.enable_ei_sign_constraints &&
            synaptic &&
            (!cfg_.ei_skip_input_layer || !is_input_projection) &&
            (!cfg_.ei_skip_output_layer || !is_output_projection);

        if (state.ei_constraint_eligible) {
            state.src_type_exc.resize(state.param->rows, 1);
            state.dst_type_exc.resize(state.param->cols, 1);
            const double p_exc = clamp(cfg_.ei_excitatory_probability, 0.0, 1.0);
            auto assign_types = [this, p_exc](std::vector<std::uint8_t>& types) {
                std::fill(types.begin(), types.end(), static_cast<std::uint8_t>(0));
                if (types.empty()) return;
                std::size_t n_exc = static_cast<std::size_t>(std::llround(p_exc * static_cast<double>(types.size())));
                n_exc = std::min(n_exc, types.size());
                for (std::size_t i = 0; i < n_exc; ++i) {
                    types[i] = 1;
                }
                std::shuffle(types.begin(), types.end(), rng_);
            };
            assign_types(state.src_type_exc);
            assign_types(state.dst_type_exc);
        }

        refresh_source_delay_bins(state);
        states_.push_back(std::move(state));
        param_to_state_[params[i]] = states_.size() - 1;
    }
}
