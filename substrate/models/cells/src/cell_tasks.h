#pragma once

#include "models/cells/src/cell_models.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cells {

enum class CellTaskKind {
    Chemotaxis,
    DistributionShift,
    DamageRecovery,
};

struct CellTaskConfig {
    CellTaskKind kind = CellTaskKind::Chemotaxis;
    CellInit init;
    std::size_t steps = 1200;
    double dt = 0.005;
    unsigned int seed = 42;

    double target_radius = 2.0;
    int shift_every = 250;
    int damage_step = 600;
    double damage_fraction = 0.18;
    bool damage_vertices = true;
    bool damage_particles = true;
};

struct CellTaskObservation {
    std::size_t step = 0;
    Vec3 centroid;
    Vec3 target;
    Vec3 target_direction;
    double distance_to_target = 0.0;
    double energy = 0.0;
    double homeostasis_score = 0.0;
    double volume_drift_pct = 0.0;
    double area_drift_pct = 0.0;
    double centroid_speed = 0.0;
    double target_salience = 1.0;
};

struct CellTaskTraceRow {
    std::size_t step = 0;
    Vec3 centroid;
    Vec3 target;
    double reward = 0.0;
    double distance_to_target = 0.0;
    double energy = 0.0;
    double homeostasis_score = 0.0;
    double volume_drift_pct = 0.0;
    double area_drift_pct = 0.0;
    double centroid_speed = 0.0;
};

struct CellTaskMetrics {
    std::string task_name;
    std::string controller_name;
    std::size_t controller_parameters = 0;

    double cumulative_reward = 0.0;
    double mean_distance = 0.0;
    double final_distance = 0.0;
    double mean_energy = 0.0;
    double mean_homeostasis = 0.0;
    double max_volume_drift_pct = 0.0;
    double max_area_drift_pct = 0.0;
    int shift_count = 0;
    int recovery_steps = -1;
};

struct CellTaskEpisode {
    CellTaskMetrics metrics;
    std::vector<CellTaskTraceRow> trace;
};

class CellTaskController {
public:
    virtual ~CellTaskController() = default;

    virtual std::string name() const = 0;
    virtual std::size_t parameter_count() const = 0;
    virtual void reset(unsigned int seed) = 0;
    virtual CellControlInput act(const CellTaskObservation& obs) = 0;
};

class HomeostaticCellController final : public CellTaskController {
public:
    HomeostaticCellController();

    std::string name() const override;
    std::size_t parameter_count() const override;
    void reset(unsigned int seed) override;
    CellControlInput act(const CellTaskObservation& obs) override;

private:
    double fast_error_ = 0.0;
    double slow_error_ = 0.0;
    double repair_pressure_ = 0.0;
    double momentum_ = 0.0;
    unsigned int seed_ = 0;
};

class RecurrentBaselineController final : public CellTaskController {
public:
    explicit RecurrentBaselineController(std::size_t hidden_dim = 4);

    std::string name() const override;
    std::size_t parameter_count() const override;
    void reset(unsigned int seed) override;
    CellControlInput act(const CellTaskObservation& obs) override;

private:
    std::vector<double> observe(const CellTaskObservation& obs) const;

    std::size_t hidden_dim_ = 4;
    std::vector<double> hidden_;
    std::vector<double> w_in_;
    std::vector<double> w_rec_;
    std::vector<double> b_h_;
    std::vector<double> w_out_;
    std::vector<double> b_out_;
};

class GruBaselineController final : public CellTaskController {
public:
    explicit GruBaselineController(std::size_t hidden_dim = 4);

    std::string name() const override;
    std::size_t parameter_count() const override;
    void reset(unsigned int seed) override;
    CellControlInput act(const CellTaskObservation& obs) override;

private:
    std::size_t hidden_dim_ = 4;
    std::vector<double> hidden_;

    std::vector<double> w_in_z_;
    std::vector<double> w_in_r_;
    std::vector<double> w_in_n_;
    std::vector<double> w_h_z_;
    std::vector<double> w_h_r_;
    std::vector<double> w_h_n_;
    std::vector<double> b_z_;
    std::vector<double> b_r_;
    std::vector<double> b_n_;

    std::vector<double> w_out_;
    std::vector<double> b_out_;
};

class LstmBaselineController final : public CellTaskController {
public:
    explicit LstmBaselineController(std::size_t hidden_dim = 4);

    std::string name() const override;
    std::size_t parameter_count() const override;
    void reset(unsigned int seed) override;
    CellControlInput act(const CellTaskObservation& obs) override;

private:
    std::size_t hidden_dim_ = 4;
    std::vector<double> hidden_;
    std::vector<double> cell_;

    std::vector<double> w_in_;
    std::vector<double> w_h_;
    std::vector<double> b_;

    std::vector<double> w_out_;
    std::vector<double> b_out_;
};

class TransformerBaselineController final : public CellTaskController {
public:
    TransformerBaselineController(std::size_t model_dim = 12, std::size_t window = 6);

    std::string name() const override;
    std::size_t parameter_count() const override;
    void reset(unsigned int seed) override;
    CellControlInput act(const CellTaskObservation& obs) override;

private:
    std::size_t model_dim_ = 12;
    std::size_t window_ = 6;

    std::vector<std::vector<double>> tokens_;

    std::vector<double> proj_in_;
    std::vector<double> q_w_;
    std::vector<double> k_w_;
    std::vector<double> v_w_;
    std::vector<double> ff1_;
    std::vector<double> ff2_;

    std::vector<double> out_w_;
    std::vector<double> out_b_;
};

std::string cell_task_kind_name(CellTaskKind kind);

CellTaskEpisode run_cell_task_episode(const CellTaskConfig& cfg,
                                      CellTaskController& controller);

void write_cell_task_csv(const std::string& path,
                         const CellTaskEpisode& episode);

} // namespace cells
