#pragma once

#include "models/cells/src/cell_module.h"

#include <cstddef>
#include <random>
#include <string>
#include <vector>

namespace cells {

struct CellCircuitConfig {
    std::size_t input_dim = 784;
    std::size_t retina_dim = 196;
    std::size_t v1_dim = 256;
    std::size_t assoc_dim = 128;
    std::size_t decision_dim = 10;

    double dt = 0.02;
    double lr = 0.005;
    double eligibility_decay = 0.95;
    double weight_decay = 1e-4;
    double homeostasis_hmax = 2.0;
    double weight_clip = 3.0;
    double hidden_lr_scale = 0.2;
    int settle_steps = 2;
    bool reset_state_per_sample = true;

    unsigned int seed = 42;
};

struct CellCircuitStepInfo {
    int pred = 0;
    double loss = 0.0;
    double reward = 0.0;
    double confidence = 0.0;
    double mean_energy = 0.0;
    double mean_homeostasis = 0.0;
};

class CellCircuit {
public:
    explicit CellCircuit(const CellCircuitConfig& cfg = CellCircuitConfig{});

    void reset_state();
    std::size_t parameter_count() const;

    CellCircuitStepInfo train_one(const std::vector<double>& x, int label);
    CellCircuitStepInfo evaluate_one(const std::vector<double>& x, int label);
    std::vector<double> predict_proba(const std::vector<double>& x);

    const CellCircuitConfig& config() const { return cfg_; }

private:
    std::vector<double> forward_hidden(const std::vector<double>& x,
                                       std::vector<double>& retina_out,
                                       std::vector<double>& v1_out,
                                       std::vector<double>& assoc_out,
                                       std::vector<double>& decision_out,
                                       bool inject_noise);

    static std::vector<double> matvec(const std::vector<double>& x, const std::vector<double>& w,
                                      std::size_t in_dim, std::size_t out_dim);
    static std::vector<double> softmax(const std::vector<double>& z);
    static double cross_entropy(const std::vector<double>& p, int label);

    void update_eligibility(const std::vector<double>& pre, const std::vector<double>& post,
                            std::vector<double>& elig, std::size_t in_dim, std::size_t out_dim);

    void apply_three_factor_update(const std::vector<double>& post_h,
                                   double reward,
                                   std::vector<double>& w,
                                   std::vector<double>& elig,
                                   std::size_t in_dim,
                                   std::size_t out_dim);

    double mean_energy() const;
    double mean_homeostasis() const;

    CellCircuitConfig cfg_;
    std::mt19937 rng_;

    CellModuleBank retina_;
    CellModuleBank v1_;
    CellModuleBank assoc_;
    CellModuleBank decision_;

    // Dense row-major matrices: [in][out].
    std::vector<double> w_in_retina_;
    std::vector<double> w_retina_v1_;
    std::vector<double> w_v1_assoc_;
    std::vector<double> w_assoc_decision_;

    // Eligibility traces aligned with corresponding weights.
    std::vector<double> e_in_retina_;
    std::vector<double> e_retina_v1_;
    std::vector<double> e_v1_assoc_;
    std::vector<double> e_assoc_decision_;
};

} // namespace cells
