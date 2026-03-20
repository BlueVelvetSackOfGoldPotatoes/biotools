#include "core/online/continuous_runtime.h"

#include "core/losses/losses.h"
#include "core/metrics/metrics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {

std::pair<double, double> norm_stats(const std::vector<Tensor*>& ts) {
    if (ts.empty()) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan};
    }
    double sum = 0.0;
    double mx = 0.0;
    for (const auto* t : ts) {
        const double n = t->norm();
        sum += n;
        mx = std::max(mx, n);
    }
    return {sum / static_cast<double>(ts.size()), mx};
}

} // namespace

ContinuousReplayBuffer::ContinuousReplayBuffer(std::size_t capacity) : capacity_(capacity) {
    if (capacity_ > 0) data_.resize(capacity_);
}

void ContinuousReplayBuffer::clear() {
    size_ = 0;
    next_ = 0;
}

void ContinuousReplayBuffer::push(std::size_t idx) {
    if (capacity_ == 0) return;
    data_[next_] = idx;
    next_ = (next_ + 1) % capacity_;
    size_ = std::min(capacity_, size_ + 1);
}

void ContinuousReplayBuffer::push_many(const std::vector<std::size_t>& indices) {
    for (const auto idx : indices) push(idx);
}

// Samples with replacement — standard for experience replay buffers.
void ContinuousReplayBuffer::sample(std::size_t n, std::mt19937& rng, std::vector<std::size_t>& out) const {
    out.clear();
    if (size_ == 0 || n == 0) return;
    out.resize(n);
    std::uniform_int_distribution<std::size_t> dist(0, size_ - 1);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = data_[dist(rng)];
    }
}

ContinuousRuntime::ContinuousRuntime(ContinuousConfig config, int seed)
    : cfg_(config), rng_(seed), replay_(config.replay_capacity) {}

void ContinuousRuntime::reset() {
    cumulative_samples_ = 0;
    replay_.clear();
}

void ContinuousRuntime::seed_replay(std::size_t dataset_size) {
    if (dataset_size == 0 || cfg_.replay_seed_samples == 0) return;
    std::vector<std::size_t> seed;
    sample_uniform_indices(dataset_size, std::min(cfg_.replay_seed_samples, dataset_size), seed);
    replay_.push_many(seed);
}

void ContinuousRuntime::sample_uniform_indices(std::size_t dataset_size,
                                               std::size_t n,
                                               std::vector<std::size_t>& out) {
    out.clear();
    if (dataset_size == 0 || n == 0) return;
    out.resize(n);
    std::uniform_int_distribution<std::size_t> dist(0, dataset_size - 1);
    for (std::size_t i = 0; i < n; ++i) out[i] = dist(rng_);
}

void ContinuousRuntime::add_noise_inplace(Tensor& x, double stddev) {
    if (stddev <= 0.0 || x.data.empty()) return;
    std::normal_distribution<double> noise(0.0, stddev);
    for (double& v : x.data) {
        v = std::max(0.0, std::min(1.0, v + noise(rng_)));
    }
}

PhaseMetrics ContinuousRuntime::run_phase(TrainableModel& model,
                                          Adam& optimizer,
                                          CrossEntropyLoss& loss_fn,
                                          const MNISTData& train,
                                          const std::string& phase,
                                          int steps,
                                          std::size_t batch_size,
                                          double lr_scale,
                                          double replay_fraction,
                                          double input_noise_std,
                                          const ContinuousHooks& hooks) {
    PhaseMetrics pm;
    pm.phase = phase;
    pm.steps = std::max(0, steps);
    pm.lr_scale = lr_scale;

    if (pm.steps <= 0 || batch_size == 0 || train.images.rows == 0) {
        pm.samples = 0;
        pm.replay_fraction = 0.0;
        return pm;
    }

    auto phase_start = std::chrono::high_resolution_clock::now();

    double loss_sum = 0.0;
    double acc_sum = 0.0;
    double grad_mean_sum = 0.0;
    double grad_max = 0.0;
    double param_mean_sum = 0.0;
    double param_max = 0.0;
    std::size_t replay_picks = 0;
    std::size_t total_picks = 0;

    std::vector<std::size_t> idx_replay;
    std::vector<std::size_t> idx_fresh;
    std::vector<std::size_t> merged_idx;
    merged_idx.reserve(batch_size);

    for (int step = 0; step < pm.steps; ++step) {
        const std::size_t replay_n = std::min<std::size_t>(
            batch_size,
            static_cast<std::size_t>(std::llround(batch_size * replay_fraction))
        );
        const std::size_t fresh_n = batch_size - replay_n;

        replay_.sample(replay_n, rng_, idx_replay);
        sample_uniform_indices(train.images.rows, fresh_n, idx_fresh);

        merged_idx.clear();
        merged_idx.insert(merged_idx.end(), idx_replay.begin(), idx_replay.end());
        merged_idx.insert(merged_idx.end(), idx_fresh.begin(), idx_fresh.end());
        if (merged_idx.size() < batch_size) {
            std::vector<std::size_t> extra;
            sample_uniform_indices(train.images.rows, batch_size - merged_idx.size(), extra);
            merged_idx.insert(merged_idx.end(), extra.begin(), extra.end());
        }
        std::shuffle(merged_idx.begin(), merged_idx.end(), rng_);

        Tensor bx = extract_batch(train.images, merged_idx, 0, batch_size);
        Tensor by = extract_batch(train.one_hot, merged_idx, 0, batch_size);
        add_noise_inplace(bx, input_noise_std);

        // Only add newly sampled data to replay.
        // Feeding replay samples back into replay each step causes rapid
        // self-reinforcement and diversity collapse.
        replay_.push_many(idx_fresh);
        replay_picks += idx_replay.size();
        total_picks += merged_idx.size();
        cumulative_samples_ += merged_idx.size();

        model.zero_grad();
        if (hooks.before_forward) hooks.before_forward();

        Tensor logits = model.forward(bx);
        Tensor probs = LossFunctions::softmax(logits);
        const double loss = loss_fn.forward(probs, by);
        const double acc = Metrics::accuracy_from_tensor(probs, by);
        loss_sum += loss;
        acc_sum += acc;

        Tensor grad = loss_fn.backward(probs, by);
        model.backward(grad);
        if (hooks.after_backward) hooks.after_backward(loss, optimizer.lr() * lr_scale);

        auto params = model.parameters();
        auto grads = model.gradients();
        auto [gmean, gmax] = norm_stats(grads);
        auto [pmean, pmax] = norm_stats(params);
        grad_mean_sum += gmean;
        grad_max = std::max(grad_max, gmax);
        param_mean_sum += pmean;
        param_max = std::max(param_max, pmax);

        optimizer.step(params, grads, lr_scale, cfg_.grad_clip_norm);
        Adam::zero_grad(grads);
        if (hooks.after_optimizer_step) hooks.after_optimizer_step();
    }

    auto phase_end = std::chrono::high_resolution_clock::now();
    pm.phase_time_ms =
        std::chrono::duration<double, std::milli>(phase_end - phase_start).count();
    pm.samples = static_cast<std::size_t>(pm.steps) * batch_size;
    pm.loss = loss_sum / static_cast<double>(pm.steps);
    pm.accuracy = acc_sum / static_cast<double>(pm.steps);
    pm.grad_norm_mean = grad_mean_sum / static_cast<double>(pm.steps);
    pm.grad_norm_max = grad_max;
    pm.param_norm_mean = param_mean_sum / static_cast<double>(pm.steps);
    pm.param_norm_max = param_max;
    pm.replay_fraction = total_picks ? static_cast<double>(replay_picks) / static_cast<double>(total_picks) : 0.0;
    return pm;
}

CycleMetrics ContinuousRuntime::run_cycle(TrainableModel& model,
                                          Adam& optimizer,
                                          CrossEntropyLoss& loss_fn,
                                          const MNISTData& train,
                                          int cycle,
                                          const ContinuousHooks& hooks) {
    if (train.images.rows == 0 || train.one_hot.rows == 0) {
        throw std::runtime_error("ContinuousRuntime::run_cycle empty training data");
    }

    if (hooks.on_cycle_begin) hooks.on_cycle_begin(cycle);

    auto cycle_start = std::chrono::high_resolution_clock::now();
    model.train();

    CycleMetrics cm;
    cm.cycle = cycle;
    cm.active = run_phase(
        model,
        optimizer,
        loss_fn,
        train,
        "active",
        cfg_.active_steps_per_cycle,
        cfg_.active_batch_size,
        1.0,
        0.0,
        0.0,
        hooks
    );
    cm.sleep = run_phase(
        model,
        optimizer,
        loss_fn,
        train,
        "sleep",
        cfg_.sleep_steps_per_cycle,
        cfg_.sleep_batch_size,
        cfg_.sleep_lr_scale,
        cfg_.sleep_replay_ratio,
        cfg_.sleep_noise_std,
        hooks
    );
    cm.replay_size = replay_.size();
    cm.cumulative_samples = cumulative_samples_;

    auto cycle_end = std::chrono::high_resolution_clock::now();
    cm.cycle_time_ms =
        std::chrono::duration<double, std::milli>(cycle_end - cycle_start).count();

    if (hooks.on_cycle_end) hooks.on_cycle_end(cycle);
    return cm;
}
