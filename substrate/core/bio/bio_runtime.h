#pragma once

#include "core/io/run_logger.h"
#include "core/tensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace bio {

class BioRuntime;
BioRuntime* active_runtime();
void set_active_runtime(BioRuntime* runtime);

enum class EiConstraintAxis {
    Source = 0,
    Destination = 1
};

struct BioConfig {
    // Safety default: keep biology hooks observational unless explicitly enabled.
    bool active_parameter_updates = false;

    bool enable_structural_plasticity = true;
    bool enable_homeostasis = true;
    bool enable_three_factor = true;
    bool enable_myelination = true;
    bool enable_bioelectric = true;
    bool enable_ei_sign_constraints = true;
    EiConstraintAxis ei_constraint_axis = EiConstraintAxis::Source;
    // Source-axis E/I should not be imposed on raw input projection by default.
    bool ei_skip_input_layer = true;
    // Leave final readout unconstrained by default to avoid class-logit collapse.
    bool ei_skip_output_layer = true;
    // 1.0 = hard projection; <1.0 = soft projection toward sign-consistent weights.
    double ei_projection_strength = 0.25;
    // Probability of an excitatory unit when initializing E/I cell types.
    double ei_excitatory_probability = 0.9;

    int rewire_every_epochs = 1;
    int myelin_every_epochs = 1;
    double prune_fraction = 0.02;
    double growth_noise_std = 0.01;
    std::size_t structural_sample_cap = 0;

    double eligibility_decay = 0.98;
    double three_factor_lr = 5e-5;
    double modulator_clip = 1.0;

    double homeostasis_target_norm = 1.0;
    double homeostasis_lr = 5e-3;

    double myelin_budget_fraction = 0.15;
    double conduction_base_speed = 1.0;
    double conduction_myelin_boost = 2.5;
    int conduction_delay_bins = 4;
    double conduction_delay_mix = 0.35;

    double bioelectric_leak = 0.1;
    double bioelectric_coupling = 0.1;

    double wiring_distance_cost = 0.04;
    std::size_t wiring_cost_sample_stride = 1;
};

struct BioLayerSnapshot {
    std::string layer_name;
    double weight_l2 = 0.0;
    double grad_l2 = 0.0;
    double eligibility_l2 = 0.0;
    double mask_density = 1.0;
    double mean_abs_weight = 0.0;
    double mean_abs_grad = 0.0;
    double myelin_fraction = 0.0;
    double mean_delay = 1.0;
    double bioelectric_mean = 0.0;
    double homeostasis_error = 0.0;
    double glia_gain = 1.0;
    double glia_plasticity = 1.0;
    double excitatory_fraction = 0.0;
    double inhibitory_fraction = 0.0;
    double sign_violation_fraction = 0.0;
    std::size_t pruned_edges = 0;
    std::size_t grown_edges = 0;
};

struct BioEpochSnapshot {
    int epoch = 0;
    std::size_t params_attached = 0;
    double energy_used = 0.0;
    double wiring_cost = 0.0;
    double mask_density = 1.0;
    double myelin_fraction = 0.0;
    double mean_delay = 1.0;
    double bioelectric_mean = 0.0;
    double plasticity_update_mean_abs = 0.0;
    double homeostasis_error = 0.0;
    double modulator_mean = 0.0;
    double reward_mean = std::numeric_limits<double>::quiet_NaN();
    double sign_violation_fraction = 0.0;
    std::size_t total_pruned = 0;
    std::size_t total_grown = 0;
    std::vector<BioLayerSnapshot> layers;
};

class BioRuntime {
public:
    BioRuntime(const std::string& model_family,
               const std::string& model_variant,
               int seed = 42,
               BioConfig config = {});

    void attach(const std::vector<Tensor*>& params,
                const std::vector<Tensor*>& grads,
                const std::vector<std::string>& names = {});

    void begin_epoch(int epoch);
    void before_forward();
    void after_backward(double loss,
                        double reward = std::numeric_limits<double>::quiet_NaN(),
                        double lr = 0.0);
    void after_optimizer_step();

    BioEpochSnapshot end_epoch(int epoch);
    void log_epoch(RunLogger& logger, int epoch, const BioEpochSnapshot& snapshot) const;

    // Causal linear transport hook: turns delayed/myelinated edges into
    // delayed input channels before matmul.
    bool apply_linear_input_transport(const Tensor* weight_param, const Tensor& input, Tensor& effective_input);
    // Captures local activities for true local-plasticity updates.
    void record_linear_activity(const Tensor* weight_param, const Tensor& pre_input, const Tensor& linear_output);

    bool enabled() const { return !states_.empty(); }
    bool attached() const { return !states_.empty(); }

private:
    struct Vec3 {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
    };

    struct ParamState {
        std::string name;
        Tensor* param = nullptr;
        Tensor* grad = nullptr;
        bool bio_edge_eligible = false;
        bool ei_constraint_eligible = false;

        Tensor mask;
        Tensor eligibility;
        Tensor myelin;
        Tensor delay;
        Tensor bioelectric;
        std::vector<Tensor> input_history;
        std::vector<std::size_t> src_delay_bin;
        std::vector<double> pre_activity_mean;
        std::vector<double> post_activity_mean;
        bool source_is_rows = true;

        std::vector<Vec3> src_pos;
        std::vector<Vec3> dst_pos;
        std::vector<std::uint8_t> src_type_exc;
        std::vector<std::uint8_t> dst_type_exc;

        double glia_gain = 1.0;
        double glia_plasticity = 1.0;
        double glia_prune_pressure = 0.0;
        double glia_support = 1.0;
        double last_homeostasis_error = 0.0;
        std::size_t last_pruned = 0;
        std::size_t last_grown = 0;
        double grad_l2_sum = 0.0;
        double abs_grad_sum = 0.0;
        std::size_t grad_obs = 0;
    };

    std::string model_family_;
    std::string model_variant_;
    BioConfig cfg_;
    std::mt19937 rng_;
    std::vector<ParamState> states_;
    std::unordered_map<const Tensor*, std::size_t> param_to_state_;

    double prev_loss_ = std::numeric_limits<double>::quiet_NaN();
    int epoch_counter_ = 0;

    double epoch_energy_ = 0.0;
    double epoch_wiring_cost_ = 0.0;
    double epoch_modulator_sum_ = 0.0;
    double epoch_reward_sum_ = 0.0;
    std::size_t epoch_reward_count_ = 0;
    double epoch_plasticity_abs_sum_ = 0.0;
    std::size_t epoch_plasticity_count_ = 0;
    double epoch_homeostasis_error_sum_ = 0.0;
    std::size_t epoch_homeostasis_count_ = 0;
    std::size_t epoch_batches_ = 0;

    static bool finite(double v);
    static double clamp(double v, double lo, double hi);
    static std::size_t grid_side(std::size_t n);
    static std::vector<Vec3> init_positions(std::size_t n, double z_offset);
    static double mean_abs(const Tensor& t);
    static double tensor_density01(const Tensor& t);

    static double edge_distance(const ParamState& state, std::size_t flat_idx);
    static bool is_ei_constraint_eligible_tensor(const Tensor* t);
    void enforce_ei_sign(ParamState& state) const;
    void refresh_source_delay_bins(ParamState& state);

    double compute_modulator(double loss, double reward) const;
    void apply_homeostasis(ParamState& state, Tensor& target);
    void apply_structural_plasticity(ParamState& state);
    void allocate_myelin_and_delay(ParamState& state);
};

} // namespace bio
