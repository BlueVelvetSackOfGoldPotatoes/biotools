BioEpochSnapshot BioRuntime::end_epoch(int epoch) {
    set_active_runtime(nullptr);
    BioEpochSnapshot snapshot;
    snapshot.epoch = epoch;
    snapshot.params_attached = states_.size();

    std::size_t total_n = 0;
    double total_mask = 0.0;
    double total_myelin = 0.0;
    double total_delay = 0.0;
    double total_bioelectric = 0.0;
    double total_sign_violation = 0.0;

    for (auto& state : states_) {
        if (!state.param) continue;

        // In active mode, run structural plasticity and myelination.
        // In observational mode, skip these expensive operations.
        if (cfg_.active_parameter_updates) {
            if (cfg_.enable_structural_plasticity && cfg_.rewire_every_epochs > 0 &&
                epoch % cfg_.rewire_every_epochs == 0) {
                apply_structural_plasticity(state);
            } else {
                state.last_pruned = 0;
                state.last_grown = 0;
            }
            if (cfg_.myelin_every_epochs <= 1 || (epoch % cfg_.myelin_every_epochs == 0)) {
                allocate_myelin_and_delay(state);
            }
        } else {
            state.last_pruned = 0;
            state.last_grown = 0;
        }

        BioLayerSnapshot layer;
        layer.layer_name = state.name;
        layer.weight_l2 = state.param->norm();
        layer.grad_l2 = state.grad_obs ? (state.grad_l2_sum / static_cast<double>(state.grad_obs)) : 0.0;
        layer.eligibility_l2 = state.eligibility.norm();
        layer.mask_density = tensor_density01(state.mask);
        layer.mean_abs_weight = mean_abs(*state.param);
        layer.mean_abs_grad = state.grad_obs ? (state.abs_grad_sum / static_cast<double>(state.grad_obs)) : 0.0;
        layer.myelin_fraction = tensor_density01(state.myelin);
        layer.mean_delay = state.delay.mean();
        layer.bioelectric_mean = mean_abs(state.bioelectric);
        layer.homeostasis_error = state.last_homeostasis_error;
        layer.glia_gain = state.glia_gain;
        layer.glia_plasticity = state.glia_plasticity;
        if (state.ei_constraint_eligible && (!state.dst_type_exc.empty() || !state.src_type_exc.empty())) {
            std::size_t exc = 0;
            const bool by_dst = cfg_.ei_constraint_axis == EiConstraintAxis::Destination;
            const auto& types = by_dst ? state.dst_type_exc : state.src_type_exc;
            for (std::uint8_t t : types) {
                if (t != 0) exc++;
            }
            layer.excitatory_fraction =
                types.empty() ? 0.0 : static_cast<double>(exc) / static_cast<double>(types.size());
            layer.inhibitory_fraction = 1.0 - layer.excitatory_fraction;
        }
        if (state.ei_constraint_eligible && state.param->cols > 0 && state.param->rows > 0) {
            const std::size_t cols = state.param->cols;
            const bool by_dst = cfg_.ei_constraint_axis == EiConstraintAxis::Destination;
            std::size_t active = 0;
            std::size_t violations = 0;
            for (std::size_t idx = 0; idx < state.param->data.size(); ++idx) {
                if (state.mask.data[idx] <= 0.5) continue;
                const std::size_t row = idx / cols;
                const std::size_t col = idx % cols;
                bool excitatory = true;
                if (by_dst) {
                    excitatory = col < state.dst_type_exc.size() && state.dst_type_exc[col] != 0;
                } else {
                    excitatory = row < state.src_type_exc.size() && state.src_type_exc[row] != 0;
                }
                const double w = state.param->data[idx];
                active++;
                if ((excitatory && w < 0.0) || (!excitatory && w > 0.0)) {
                    violations++;
                }
            }
            layer.sign_violation_fraction =
                active ? (static_cast<double>(violations) / static_cast<double>(active)) : 0.0;
        }
        layer.pruned_edges = state.last_pruned;
        layer.grown_edges = state.last_grown;
        snapshot.layers.push_back(layer);

        const std::size_t n = state.param->data.size();
        total_n += n;
        total_mask += layer.mask_density * static_cast<double>(n);
        total_myelin += layer.myelin_fraction * static_cast<double>(n);
        total_delay += layer.mean_delay * static_cast<double>(n);
        total_bioelectric += layer.bioelectric_mean * static_cast<double>(n);
        total_sign_violation += layer.sign_violation_fraction * static_cast<double>(n);
        snapshot.total_pruned += layer.pruned_edges;
        snapshot.total_grown += layer.grown_edges;

        // Wiring cost: O(n) per-element loop — skip in observational mode.
        if (cfg_.active_parameter_updates) {
            const std::size_t stride = std::max<std::size_t>(1, cfg_.wiring_cost_sample_stride);
            for (std::size_t i = 0; i < n; i += stride) {
                epoch_wiring_cost_ += static_cast<double>(stride) * state.mask.data[i] * edge_distance(state, i);
            }
        }
    }

    snapshot.energy_used = epoch_energy_;
    snapshot.wiring_cost = epoch_wiring_cost_;
    snapshot.mask_density = total_n ? (total_mask / static_cast<double>(total_n)) : 1.0;
    snapshot.myelin_fraction = total_n ? (total_myelin / static_cast<double>(total_n)) : 0.0;
    snapshot.mean_delay = total_n ? (total_delay / static_cast<double>(total_n)) : 1.0;
    snapshot.bioelectric_mean = total_n ? (total_bioelectric / static_cast<double>(total_n)) : 0.0;
    snapshot.plasticity_update_mean_abs =
        epoch_plasticity_count_ ? (epoch_plasticity_abs_sum_ / static_cast<double>(epoch_plasticity_count_)) : 0.0;
    snapshot.homeostasis_error =
        epoch_homeostasis_count_ ? (epoch_homeostasis_error_sum_ / static_cast<double>(epoch_homeostasis_count_)) : 0.0;
    snapshot.modulator_mean = epoch_batches_ ? (epoch_modulator_sum_ / static_cast<double>(epoch_batches_)) : 0.0;
    snapshot.sign_violation_fraction =
        total_n ? (total_sign_violation / static_cast<double>(total_n)) : 0.0;
    snapshot.reward_mean =
        epoch_reward_count_ ? (epoch_reward_sum_ / static_cast<double>(epoch_reward_count_))
                            : std::numeric_limits<double>::quiet_NaN();
    return snapshot;
}

void BioRuntime::log_epoch(RunLogger& logger, int epoch, const BioEpochSnapshot& snapshot) const {
    const std::string family = model_family_.empty() ? "unknown" : model_family_;
    const std::string epoch_path = "model_specific/" + family + "/bio_epoch_dynamics.csv";
    const std::string layer_path = "model_specific/" + family + "/bio_layer_dynamics.csv";

    logger.append_csv_row(
        epoch_path,
        {"run_id",
         "epoch",
         "model_variant",
         "energy_used",
         "wiring_cost",
         "mask_density",
         "myelin_fraction",
         "mean_delay",
         "bioelectric_mean",
         "plasticity_update_mean_abs",
         "homeostasis_error",
         "modulator_mean",
         "sign_violation_fraction",
         "reward_mean",
         "total_pruned",
         "total_grown",
         "params_attached"},
        {logger.run_id(),
         std::to_string(epoch),
         model_variant_,
         precise(snapshot.energy_used),
         precise(snapshot.wiring_cost),
         precise(snapshot.mask_density),
         precise(snapshot.myelin_fraction),
         precise(snapshot.mean_delay),
         precise(snapshot.bioelectric_mean),
         precise(snapshot.plasticity_update_mean_abs),
         precise(snapshot.homeostasis_error),
         precise(snapshot.modulator_mean),
         precise(snapshot.sign_violation_fraction),
         precise(snapshot.reward_mean),
         std::to_string(snapshot.total_pruned),
         std::to_string(snapshot.total_grown),
         std::to_string(snapshot.params_attached)}
    );

    if (snapshot.layers.empty()) {
        logger.append_csv_row(
            layer_path,
            {"run_id",
             "epoch",
             "layer",
             "weight_l2",
             "grad_l2",
             "eligibility_l2",
             "mask_density",
             "mean_abs_weight",
             "mean_abs_grad",
             "myelin_fraction",
             "mean_delay",
             "bioelectric_mean",
             "homeostasis_error",
             "glia_gain",
             "glia_plasticity",
             "excitatory_fraction",
             "inhibitory_fraction",
             "sign_violation_fraction",
             "pruned_edges",
             "grown_edges"},
            {logger.run_id(),
             std::to_string(epoch),
             "none",
             "0",
             "0",
             "0",
             "1",
             "0",
             "0",
             "0",
             "1",
             "0",
             "0",
             "1",
             "1",
             "0",
             "0",
             "0",
             "0",
             "0"}
        );
    }

    for (const auto& layer : snapshot.layers) {
        logger.append_csv_row(
            layer_path,
            {"run_id",
             "epoch",
             "layer",
             "weight_l2",
             "grad_l2",
             "eligibility_l2",
             "mask_density",
             "mean_abs_weight",
             "mean_abs_grad",
             "myelin_fraction",
             "mean_delay",
             "bioelectric_mean",
             "homeostasis_error",
             "glia_gain",
             "glia_plasticity",
             "excitatory_fraction",
             "inhibitory_fraction",
             "sign_violation_fraction",
             "pruned_edges",
             "grown_edges"},
            {logger.run_id(),
             std::to_string(epoch),
             layer.layer_name,
             precise(layer.weight_l2),
             precise(layer.grad_l2),
             precise(layer.eligibility_l2),
             precise(layer.mask_density),
             precise(layer.mean_abs_weight),
             precise(layer.mean_abs_grad),
             precise(layer.myelin_fraction),
             precise(layer.mean_delay),
             precise(layer.bioelectric_mean),
             precise(layer.homeostasis_error),
             precise(layer.glia_gain),
             precise(layer.glia_plasticity),
             precise(layer.excitatory_fraction),
             precise(layer.inhibitory_fraction),
             precise(layer.sign_violation_fraction),
             std::to_string(layer.pruned_edges),
             std::to_string(layer.grown_edges)}
        );
    }
}

