#pragma once

#include "models/cells/src/cell_tissue2d.h"

#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

namespace cells {

struct DevelopmentalEdge {
    std::size_t src_id = 0;
    std::size_t dst_id = 0;
    double weight = 0.0;
    double eligibility = 0.0;
    double contact_time = 0.0;
};

struct DevelopmentalTissueConfig {
    Tissue2DParams tissue_params;

    std::size_t readout_classes = 10;
    std::size_t seed_grid_x = 14;
    std::size_t seed_grid_y = 14;
    std::size_t gene_dim = 8;
    std::size_t max_edges = 6000;

    double edge_lr = 0.010;
    double readout_lr = 0.035;
    double eligibility_decay = 0.94;
    double weight_decay = 1e-4;
    double max_edge_weight = 2.0;
    double contact_radius = 0.18;
    double contact_time_threshold = 0.05;
    double compatibility_threshold = 0.12;
    double prune_below = 1e-3;
    double novelty_gain = 0.25;
    double teacher_gain = 0.75;
    double attention_gain = 0.55;

    int free_settle_steps = 2;
    int taught_settle_steps = 2;
    bool internal_teacher = false;
    double internal_teacher_blend = 0.5;
    unsigned int seed = 42;
};

struct DevelopmentalStepInfo {
    int pred = 0;
    double loss = 0.0;
    double confidence = 0.0;
    double mean_energy = 0.0;
    double mean_homeostasis = 0.0;
    std::size_t edge_count = 0;
    std::size_t active_cells = 0;
};

class DevelopmentalMNISTField final : public Field2D {
public:
    DevelopmentalMNISTField();

    void set_image(const std::vector<double>& image);
    void set_teacher_label(int label, bool enabled);
    void set_novelty(double novelty);

    FieldSample2D sample(const Vec2& p) const override;
    Vec2 clamp_to_domain(const Vec2& p) const override;

private:
    double pixel_at(double x, double y) const;
    void update_center_of_mass();

    std::vector<double> image_;
    Vec2 center_of_mass_;
    double novelty_ = 0.0;
    int teacher_label_ = -1;
    bool teacher_enabled_ = false;
};

class DevelopmentalTissueClassifier {
public:
    explicit DevelopmentalTissueClassifier(const DevelopmentalTissueConfig& cfg = DevelopmentalTissueConfig{});

    void reset_state();
    void reset_dynamics();
    void set_internal_teacher(bool enabled);
    bool internal_teacher_enabled() const;
    std::size_t parameter_count() const;
    std::size_t edge_count() const;

    DevelopmentalStepInfo train_one(const std::vector<double>& image, int label);
    DevelopmentalStepInfo evaluate_one(const std::vector<double>& image, int label);
    std::vector<double> predict_proba(const std::vector<double>& image);
    void apply_damage(double edge_fraction, unsigned int seed);

    const DevelopmentalTissueConfig& config() const { return cfg_; }
    const std::vector<DevelopmentalEdge>& edges() const { return edges_; }

private:
    struct NodeState {
        std::size_t cell_id = 0;
        double activation = 0.0;
        double prev_activation = 0.0;
        double homeostasis_error = 0.0;
        double attention = 0.0;
        double teacher = 0.0;
        double novelty = 0.0;
        std::vector<double> readout_w;
        std::vector<double> readout_elig;
    };

    void initialize_sheet();
    void sync_nodes();
    void update_contact_graph();
    void settle(const DevelopmentalMNISTField& field, int steps, bool taught_phase);
    std::vector<double> compute_logits() const;
    std::vector<double> softmax(const std::vector<double>& z) const;
    double cross_entropy(const std::vector<double>& p, int label) const;
    double mean_energy() const;
    double mean_homeostasis() const;
    std::uint64_t pair_key(std::size_t a, std::size_t b) const;
    bool edge_exists(std::size_t src_id, std::size_t dst_id) const;

    DevelopmentalTissueConfig cfg_;
    Genome2D genome_;
    CellSheet2D sheet_;
    DevelopmentalMNISTField field_;
    std::mt19937 rng_;

    std::vector<NodeState> nodes_;
    std::unordered_map<std::size_t, std::size_t> id_to_index_;
    std::vector<DevelopmentalEdge> edges_;
    std::unordered_map<std::uint64_t, double> contact_timers_;
    int previous_label_ = -1;
};

} // namespace cells
