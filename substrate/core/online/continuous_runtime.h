#pragma once

#include "core/data/dataloader.h"
#include "core/losses/losses.h"
#include "core/optim/optimizer.h"
#include "core/online/trainable_model.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

struct ContinuousConfig {
    int cycles = 12;
    int active_steps_per_cycle = 16;
    int sleep_steps_per_cycle = 6;

    std::size_t active_batch_size = 64;
    std::size_t sleep_batch_size = 48;

    double sleep_lr_scale = 0.25;
    double sleep_replay_ratio = 0.85;
    double sleep_noise_std = 0.0125;

    std::size_t replay_capacity = 20000;
    std::size_t replay_seed_samples = 2048;

    std::size_t eval_subset = 2000;
    double grad_clip_norm = 5.0;

    double target_accuracy = 0.95;
    bool stop_on_target = false;
    int min_cycles_before_stop = 3;
};

struct PhaseMetrics {
    std::string phase;
    int steps = 0;
    std::size_t samples = 0;
    double loss = std::numeric_limits<double>::quiet_NaN();
    double accuracy = std::numeric_limits<double>::quiet_NaN();
    double lr_scale = 1.0;
    double replay_fraction = 0.0;
    double phase_time_ms = 0.0;
    double grad_norm_mean = std::numeric_limits<double>::quiet_NaN();
    double grad_norm_max = std::numeric_limits<double>::quiet_NaN();
    double param_norm_mean = std::numeric_limits<double>::quiet_NaN();
    double param_norm_max = std::numeric_limits<double>::quiet_NaN();
};

struct CycleMetrics {
    int cycle = 0;
    PhaseMetrics active;
    PhaseMetrics sleep;
    std::size_t replay_size = 0;
    std::size_t cumulative_samples = 0;
    double cycle_time_ms = 0.0;
    bool target_reached_this_cycle = false;
};

struct ContinuousHooks {
    std::function<void(int cycle)> on_cycle_begin;
    std::function<void()> before_forward;
    std::function<void(double loss, double lr)> after_backward;
    std::function<void()> after_optimizer_step;
    std::function<void(int cycle)> on_cycle_end;
};

class ContinuousReplayBuffer {
public:
    explicit ContinuousReplayBuffer(std::size_t capacity = 0);

    void clear();
    void push(std::size_t idx);
    void push_many(const std::vector<std::size_t>& indices);
    void sample(std::size_t n, std::mt19937& rng, std::vector<std::size_t>& out) const;

    std::size_t size() const { return size_; }
    std::size_t capacity() const { return capacity_; }

private:
    std::vector<std::size_t> data_;
    std::size_t capacity_ = 0;
    std::size_t size_ = 0;
    std::size_t next_ = 0;
};

class ContinuousRuntime {
public:
    explicit ContinuousRuntime(ContinuousConfig config = {}, int seed = 42);

    void reset();

    void seed_replay(std::size_t dataset_size);

    CycleMetrics run_cycle(TrainableModel& model,
                           Adam& optimizer,
                           CrossEntropyLoss& loss_fn,
                           const MNISTData& train,
                           int cycle,
                           const ContinuousHooks& hooks = {});

    std::size_t replay_size() const { return replay_.size(); }
    std::size_t cumulative_samples() const { return cumulative_samples_; }
    const ContinuousConfig& config() const { return cfg_; }

private:
    ContinuousConfig cfg_;
    std::mt19937 rng_;
    ContinuousReplayBuffer replay_;
    std::size_t cumulative_samples_ = 0;

    void sample_uniform_indices(std::size_t dataset_size, std::size_t n, std::vector<std::size_t>& out);
    void add_noise_inplace(Tensor& x, double stddev);

    PhaseMetrics run_phase(TrainableModel& model,
                           Adam& optimizer,
                           CrossEntropyLoss& loss_fn,
                           const MNISTData& train,
                           const std::string& phase,
                           int steps,
                           std::size_t batch_size,
                           double lr_scale,
                           double replay_fraction,
                           double input_noise_std,
                           const ContinuousHooks& hooks);
};
