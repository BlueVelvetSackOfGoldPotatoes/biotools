#include "models/cells/src/cell_tasks.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>

namespace cells {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kObsDim = 11;
constexpr std::size_t kOutDim = 8;

inline double clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

inline double sigmoid(double x) {
    if (x >= 0.0) {
        const double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(x);
    return z / (1.0 + z);
}

std::vector<double> init_normal(std::size_t n, unsigned int seed, double scale) {
    std::vector<double> out(n, 0.0);
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, scale);
    for (double& v : out) v = nd(rng);
    return out;
}

std::vector<double> matvec(const std::vector<double>& x,
                           const std::vector<double>& w,
                           std::size_t in_dim,
                           std::size_t out_dim) {
    std::vector<double> y(out_dim, 0.0);
    for (std::size_t i = 0; i < in_dim; ++i) {
        const double xi = (i < x.size()) ? x[i] : 0.0;
        const std::size_t row = i * out_dim;
        for (std::size_t o = 0; o < out_dim; ++o) {
            y[o] += xi * w[row + o];
        }
    }
    return y;
}

double dot_vec(const std::vector<double>& a, const std::vector<double>& b) {
    const std::size_t n = std::min(a.size(), b.size());
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

std::vector<double> encode_observation(const CellTaskObservation& obs) {
    std::vector<double> x(kObsDim, 0.0);
    x[0] = obs.centroid.x;
    x[1] = obs.centroid.y;
    x[2] = obs.centroid.z;
    x[3] = obs.target.x - obs.centroid.x;
    x[4] = obs.target.y - obs.centroid.y;
    x[5] = obs.target.z - obs.centroid.z;
    x[6] = obs.energy;
    x[7] = obs.homeostasis_score;
    x[8] = obs.volume_drift_pct * 0.1;
    x[9] = obs.area_drift_pct * 0.1;
    x[10] = obs.centroid_speed;
    return x;
}

CellControlInput decode_control(const std::vector<double>& out,
                                const CellTaskObservation& obs,
                                double body_mag,
                                double chemo_mag) {
    CellControlInput u;
    const double ox = (out.size() > 0) ? out[0] : 0.0;
    const double oy = (out.size() > 1) ? out[1] : 0.0;
    const double oz = (out.size() > 2) ? out[2] : 0.0;
    const double cx = (out.size() > 3) ? out[3] : 0.0;
    const double cy = (out.size() > 4) ? out[4] : 0.0;
    const double cz = (out.size() > 5) ? out[5] : 0.0;
    const double af = (out.size() > 6) ? out[6] : 0.0;
    const double fr = (out.size() > 7) ? out[7] : 0.0;

    u.body_force = Vec3(std::tanh(ox), std::tanh(oy), std::tanh(oz)) * body_mag;
    u.chemo_gradient = Vec3(std::tanh(cx), std::tanh(cy), std::tanh(cz)) * chemo_mag;
    u.active_force_scale = 0.2 + 1.8 * sigmoid(af);
    u.substrate_friction_scale = 0.1 + 1.9 * sigmoid(fr);
    const Vec3 hint = u.chemo_gradient + obs.target_direction * 0.35 + u.body_force * 6.0;
    u.polarity_hint = (norm2(hint) > 1e-12) ? normalized(hint) : obs.target_direction;
    u.polarity_blend = 0.5;
    return u;
}

Vec3 centroid_of(const std::vector<Vec3>& x) {
    if (x.empty()) return Vec3(0.0, 0.0, 0.0);
    Vec3 c(0.0, 0.0, 0.0);
    for (const auto& p : x) c += p;
    c /= static_cast<double>(x.size());
    return c;
}

Vec3 planar_direction(std::mt19937& rng) {
    std::uniform_real_distribution<double> angle(0.0, 2.0 * kPi);
    const double a = angle(rng);
    return Vec3(std::cos(a), std::sin(a), 0.0);
}

CellTaskObservation make_observation(std::size_t step,
                                     const CellSim& sim,
                                     const Vec3& target,
                                     double target_salience) {
    CellTaskObservation obs;
    obs.step = step;
    obs.centroid = centroid_of(sim.state().membrane_positions);
    obs.target = target;
    obs.target_direction = normalized(target - obs.centroid);
    obs.distance_to_target = norm(target - obs.centroid);
    obs.energy = sim.diagnostics().energy;
    obs.homeostasis_score = sim.diagnostics().homeostasis_score;
    obs.volume_drift_pct = sim.diagnostics().volume_drift_pct;
    obs.area_drift_pct = sim.diagnostics().area_drift_pct;
    obs.centroid_speed = sim.diagnostics().centroid_speed;
    obs.target_salience = target_salience;
    return obs;
}

Vec3 target_for_step(const CellTaskConfig& cfg,
                     std::mt19937& rng,
                     std::size_t step,
                     int& shift_count,
                     Vec3& current_target) {
    if (step == 0) {
        current_target = Vec3(cfg.target_radius, 0.0, 0.0);
        if (cfg.kind == CellTaskKind::DistributionShift) {
            current_target = planar_direction(rng) * cfg.target_radius;
        }
        return current_target;
    }

    if (cfg.kind == CellTaskKind::DistributionShift &&
        cfg.shift_every > 0 &&
        (static_cast<int>(step) % cfg.shift_every) == 0) {
        current_target = planar_direction(rng) * cfg.target_radius;
        shift_count++;
    }
    return current_target;
}

} // namespace

HomeostaticCellController::HomeostaticCellController() = default;

std::string HomeostaticCellController::name() const {
    return "cell_homeostatic";
}

std::size_t HomeostaticCellController::parameter_count() const {
    return 40;
}

void HomeostaticCellController::reset(unsigned int seed) {
    seed_ = seed;
    fast_error_ = 0.0;
    slow_error_ = 0.0;
    repair_pressure_ = 0.0;
    momentum_ = 0.0;
}

CellControlInput HomeostaticCellController::act(const CellTaskObservation& obs) {
    const double homeo = 1.0 / (1.0 + obs.homeostasis_score);
    const double integrity_cost = 0.01 * (obs.volume_drift_pct + obs.area_drift_pct);
    const double distance_err = clamp(obs.distance_to_target / 3.0, 0.0, 1.5);

    fast_error_ = 0.62 * fast_error_ + 0.38 * distance_err;
    slow_error_ = 0.95 * slow_error_ + 0.05 * distance_err;
    repair_pressure_ = 0.90 * repair_pressure_ + 0.10 * (integrity_cost + (1.0 - homeo));
    momentum_ = 0.80 * momentum_ + 0.20 * obs.centroid_speed;

    const double drive = clamp(0.55 + 0.9 * fast_error_ + 0.45 * slow_error_ - 0.25 * momentum_, 0.1, 2.0);
    const double friction = clamp(1.15 + 0.75 * repair_pressure_ - 0.45 * obs.target_salience, 0.15, 2.2);
    const double body_scale = 0.012 + 0.018 * obs.target_salience + 0.012 * fast_error_;

    CellControlInput u;
    u.chemo_gradient = obs.target_direction * (0.30 + 0.85 * drive);
    u.body_force = obs.target_direction * body_scale + Vec3(0.0, 0.0, 0.0025 * (0.25 - obs.centroid.z));
    u.polarity_hint = obs.target_direction;
    u.polarity_blend = clamp(0.35 + 0.45 * obs.target_salience, 0.0, 1.0);
    u.active_force_scale = drive;
    u.substrate_friction_scale = friction;
    return u;
}

RecurrentBaselineController::RecurrentBaselineController(std::size_t hidden_dim)
    : hidden_dim_(std::max<std::size_t>(2, hidden_dim)) {}

std::string RecurrentBaselineController::name() const {
    return "rnn_baseline";
}

std::size_t RecurrentBaselineController::parameter_count() const {
    return kObsDim * hidden_dim_ + hidden_dim_ * hidden_dim_ + hidden_dim_ +
           hidden_dim_ * kOutDim + kOutDim;
}

void RecurrentBaselineController::reset(unsigned int seed) {
    hidden_.assign(hidden_dim_, 0.0);
    w_in_ = init_normal(kObsDim * hidden_dim_, seed + 11U, 0.28);
    w_rec_ = init_normal(hidden_dim_ * hidden_dim_, seed + 17U, 0.22);
    b_h_ = init_normal(hidden_dim_, seed + 23U, 0.05);
    w_out_ = init_normal(hidden_dim_ * kOutDim, seed + 29U, 0.20);
    b_out_ = init_normal(kOutDim, seed + 31U, 0.05);
}

std::vector<double> RecurrentBaselineController::observe(const CellTaskObservation& obs) const {
    return encode_observation(obs);
}

CellControlInput RecurrentBaselineController::act(const CellTaskObservation& obs) {
    const std::vector<double> x = observe(obs);
    std::vector<double> next(hidden_dim_, 0.0);
    for (std::size_t h = 0; h < hidden_dim_; ++h) {
        double pre = b_h_[h];
        for (std::size_t i = 0; i < kObsDim; ++i) {
            pre += x[i] * w_in_[i * hidden_dim_ + h];
        }
        for (std::size_t hp = 0; hp < hidden_dim_; ++hp) {
            pre += hidden_[hp] * w_rec_[hp * hidden_dim_ + h];
        }
        next[h] = std::tanh(pre);
    }
    hidden_.swap(next);

    std::vector<double> out = matvec(hidden_, w_out_, hidden_dim_, kOutDim);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] += b_out_[i];
    return decode_control(out, obs, 0.022, 0.95);
}

GruBaselineController::GruBaselineController(std::size_t hidden_dim)
    : hidden_dim_(std::max<std::size_t>(2, hidden_dim)) {}

std::string GruBaselineController::name() const {
    return "gru_baseline";
}

std::size_t GruBaselineController::parameter_count() const {
    return 3U * (kObsDim * hidden_dim_ + hidden_dim_ * hidden_dim_ + hidden_dim_) +
           hidden_dim_ * kOutDim + kOutDim;
}

void GruBaselineController::reset(unsigned int seed) {
    hidden_.assign(hidden_dim_, 0.0);

    w_in_z_ = init_normal(kObsDim * hidden_dim_, seed + 101U, 0.24);
    w_in_r_ = init_normal(kObsDim * hidden_dim_, seed + 103U, 0.24);
    w_in_n_ = init_normal(kObsDim * hidden_dim_, seed + 107U, 0.24);

    w_h_z_ = init_normal(hidden_dim_ * hidden_dim_, seed + 109U, 0.20);
    w_h_r_ = init_normal(hidden_dim_ * hidden_dim_, seed + 113U, 0.20);
    w_h_n_ = init_normal(hidden_dim_ * hidden_dim_, seed + 127U, 0.20);

    b_z_ = init_normal(hidden_dim_, seed + 131U, 0.05);
    b_r_ = init_normal(hidden_dim_, seed + 137U, 0.05);
    b_n_ = init_normal(hidden_dim_, seed + 139U, 0.05);

    w_out_ = init_normal(hidden_dim_ * kOutDim, seed + 149U, 0.18);
    b_out_ = init_normal(kOutDim, seed + 151U, 0.05);
}

CellControlInput GruBaselineController::act(const CellTaskObservation& obs) {
    const std::vector<double> x = encode_observation(obs);
    std::vector<double> next(hidden_dim_, 0.0);

    for (std::size_t h = 0; h < hidden_dim_; ++h) {
        double z_pre = b_z_[h];
        double r_pre = b_r_[h];
        double n_pre = b_n_[h];

        for (std::size_t i = 0; i < kObsDim; ++i) {
            z_pre += x[i] * w_in_z_[i * hidden_dim_ + h];
            r_pre += x[i] * w_in_r_[i * hidden_dim_ + h];
            n_pre += x[i] * w_in_n_[i * hidden_dim_ + h];
        }
        for (std::size_t j = 0; j < hidden_dim_; ++j) {
            z_pre += hidden_[j] * w_h_z_[j * hidden_dim_ + h];
            r_pre += hidden_[j] * w_h_r_[j * hidden_dim_ + h];
        }

        const double z = sigmoid(z_pre);
        const double r = sigmoid(r_pre);

        for (std::size_t j = 0; j < hidden_dim_; ++j) {
            n_pre += (r * hidden_[j]) * w_h_n_[j * hidden_dim_ + h];
        }

        const double n = std::tanh(n_pre);
        next[h] = (1.0 - z) * n + z * hidden_[h];
    }

    hidden_.swap(next);
    std::vector<double> out = matvec(hidden_, w_out_, hidden_dim_, kOutDim);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] += b_out_[i];
    return decode_control(out, obs, 0.024, 0.98);
}

LstmBaselineController::LstmBaselineController(std::size_t hidden_dim)
    : hidden_dim_(std::max<std::size_t>(2, hidden_dim)) {}

std::string LstmBaselineController::name() const {
    return "lstm_baseline";
}

std::size_t LstmBaselineController::parameter_count() const {
    const std::size_t gates = 4U * hidden_dim_;
    return kObsDim * gates + hidden_dim_ * gates + gates + hidden_dim_ * kOutDim + kOutDim;
}

void LstmBaselineController::reset(unsigned int seed) {
    hidden_.assign(hidden_dim_, 0.0);
    cell_.assign(hidden_dim_, 0.0);

    const std::size_t gates = 4U * hidden_dim_;
    w_in_ = init_normal(kObsDim * gates, seed + 211U, 0.22);
    w_h_ = init_normal(hidden_dim_ * gates, seed + 223U, 0.20);
    b_ = init_normal(gates, seed + 227U, 0.05);

    w_out_ = init_normal(hidden_dim_ * kOutDim, seed + 229U, 0.17);
    b_out_ = init_normal(kOutDim, seed + 233U, 0.05);
}

CellControlInput LstmBaselineController::act(const CellTaskObservation& obs) {
    const std::vector<double> x = encode_observation(obs);
    std::vector<double> next_h(hidden_dim_, 0.0);
    std::vector<double> next_c(hidden_dim_, 0.0);
    const std::size_t gates = 4U * hidden_dim_;

    for (std::size_t h = 0; h < hidden_dim_; ++h) {
        double pi = b_[h];
        double pf = b_[hidden_dim_ + h];
        double pg = b_[2U * hidden_dim_ + h];
        double po = b_[3U * hidden_dim_ + h];

        for (std::size_t i = 0; i < kObsDim; ++i) {
            const std::size_t row = i * gates;
            pi += x[i] * w_in_[row + h];
            pf += x[i] * w_in_[row + hidden_dim_ + h];
            pg += x[i] * w_in_[row + 2U * hidden_dim_ + h];
            po += x[i] * w_in_[row + 3U * hidden_dim_ + h];
        }

        for (std::size_t j = 0; j < hidden_dim_; ++j) {
            const std::size_t row = j * gates;
            pi += hidden_[j] * w_h_[row + h];
            pf += hidden_[j] * w_h_[row + hidden_dim_ + h];
            pg += hidden_[j] * w_h_[row + 2U * hidden_dim_ + h];
            po += hidden_[j] * w_h_[row + 3U * hidden_dim_ + h];
        }

        const double i_gate = sigmoid(pi);
        const double f_gate = sigmoid(pf);
        const double g_gate = std::tanh(pg);
        const double o_gate = sigmoid(po);

        next_c[h] = f_gate * cell_[h] + i_gate * g_gate;
        next_h[h] = o_gate * std::tanh(next_c[h]);
    }

    hidden_.swap(next_h);
    cell_.swap(next_c);

    std::vector<double> out = matvec(hidden_, w_out_, hidden_dim_, kOutDim);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] += b_out_[i];
    return decode_control(out, obs, 0.022, 0.90);
}

TransformerBaselineController::TransformerBaselineController(std::size_t model_dim, std::size_t window)
    : model_dim_(std::max<std::size_t>(4, model_dim)),
      window_(std::max<std::size_t>(2, window)) {}

std::string TransformerBaselineController::name() const {
    return "transformer_baseline";
}

std::size_t TransformerBaselineController::parameter_count() const {
    return kObsDim * model_dim_ +
           5U * model_dim_ * model_dim_ +
           model_dim_ * kOutDim + kOutDim;
}

void TransformerBaselineController::reset(unsigned int seed) {
    tokens_.clear();
    proj_in_ = init_normal(kObsDim * model_dim_, seed + 301U, 0.20);
    q_w_ = init_normal(model_dim_ * model_dim_, seed + 307U, 0.16);
    k_w_ = init_normal(model_dim_ * model_dim_, seed + 311U, 0.16);
    v_w_ = init_normal(model_dim_ * model_dim_, seed + 313U, 0.16);
    ff1_ = init_normal(model_dim_ * model_dim_, seed + 317U, 0.14);
    ff2_ = init_normal(model_dim_ * model_dim_, seed + 331U, 0.14);
    out_w_ = init_normal(model_dim_ * kOutDim, seed + 337U, 0.16);
    out_b_ = init_normal(kOutDim, seed + 347U, 0.05);
}

CellControlInput TransformerBaselineController::act(const CellTaskObservation& obs) {
    const std::vector<double> x = encode_observation(obs);
    std::vector<double> token = matvec(x, proj_in_, kObsDim, model_dim_);
    for (double& v : token) v = std::tanh(v);

    tokens_.push_back(token);
    if (tokens_.size() > window_) {
        tokens_.erase(tokens_.begin());
    }

    const std::vector<double> query = matvec(tokens_.back(), q_w_, model_dim_, model_dim_);

    std::vector<double> scores(tokens_.size(), 0.0);
    std::vector<std::vector<double>> values(tokens_.size(), std::vector<double>(model_dim_, 0.0));
    const double inv_sqrt_d = 1.0 / std::sqrt(static_cast<double>(model_dim_));
    double max_score = -std::numeric_limits<double>::infinity();
    for (std::size_t t = 0; t < tokens_.size(); ++t) {
        const std::vector<double> key = matvec(tokens_[t], k_w_, model_dim_, model_dim_);
        values[t] = matvec(tokens_[t], v_w_, model_dim_, model_dim_);
        scores[t] = dot_vec(query, key) * inv_sqrt_d;
        max_score = std::max(max_score, scores[t]);
    }

    double sum = 0.0;
    for (double& s : scores) {
        s = std::exp(s - max_score);
        sum += s;
    }
    if (sum <= 0.0) sum = 1.0;
    for (double& s : scores) s /= sum;

    std::vector<double> context(model_dim_, 0.0);
    for (std::size_t t = 0; t < tokens_.size(); ++t) {
        for (std::size_t d = 0; d < model_dim_; ++d) {
            context[d] += scores[t] * values[t][d];
        }
    }

    std::vector<double> ff = matvec(context, ff1_, model_dim_, model_dim_);
    for (std::size_t d = 0; d < model_dim_; ++d) {
        ff[d] = std::tanh(ff[d] + context[d]);
    }
    std::vector<double> head = matvec(ff, ff2_, model_dim_, model_dim_);
    for (std::size_t d = 0; d < model_dim_; ++d) {
        head[d] = std::tanh(head[d] + context[d]);
    }

    std::vector<double> out = matvec(head, out_w_, model_dim_, kOutDim);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] += out_b_[i];
    return decode_control(out, obs, 0.020, 0.85);
}

std::string cell_task_kind_name(CellTaskKind kind) {
    switch (kind) {
        case CellTaskKind::Chemotaxis: return "chemotaxis";
        case CellTaskKind::DistributionShift: return "distribution_shift";
        case CellTaskKind::DamageRecovery: return "damage_recovery";
    }
    return "unknown";
}

CellTaskEpisode run_cell_task_episode(const CellTaskConfig& cfg,
                                      CellTaskController& controller) {
    CellTaskEpisode episode;
    episode.trace.reserve(cfg.steps);

    CellInit init = cfg.init;
    init.seed = cfg.seed;
    if (init.params.dt <= 0.0) init.params.dt = cfg.dt;
    CellSim sim(init);

    controller.reset(cfg.seed + 101U);
    episode.metrics.task_name = cell_task_kind_name(cfg.kind);
    episode.metrics.controller_name = controller.name();
    episode.metrics.controller_parameters = controller.parameter_count();

    std::mt19937 rng(cfg.seed + 303U);
    Vec3 target(0.0, 0.0, 0.0);
    int shift_count = 0;
    double mean_distance = 0.0;
    double mean_energy = 0.0;
    double mean_homeo = 0.0;
    bool damage_injected = false;

    CellTaskObservation prev_obs = make_observation(0, sim, target_for_step(cfg, rng, 0, shift_count, target), 1.0);
    double prev_distance = prev_obs.distance_to_target;

    for (std::size_t step = 0; step < cfg.steps; ++step) {
        const Vec3 current_target = target_for_step(cfg, rng, step, shift_count, target);
        const double salience = (cfg.kind == CellTaskKind::DistributionShift && cfg.shift_every > 0 &&
                                 (static_cast<int>(step) % cfg.shift_every) == 0)
                                    ? 1.5
                                    : 1.0;
        CellTaskObservation obs = make_observation(step, sim, current_target, salience);
        if (step > 0) prev_distance = prev_obs.distance_to_target;

        if (!damage_injected && cfg.kind == CellTaskKind::DamageRecovery && static_cast<int>(step) == cfg.damage_step) {
            if (cfg.damage_vertices) {
                sim.damage_random_vertices(cfg.damage_fraction, cfg.seed + 707U);
            }
            if (cfg.damage_particles) {
                sim.damage_random_particles(cfg.damage_fraction, cfg.seed + 809U);
            }
            damage_injected = true;
            obs = make_observation(step, sim, current_target, salience);
        }

        const CellControlInput u = controller.act(obs);
        sim.step(u, cfg.dt);

        CellTaskObservation next_obs = make_observation(step + 1, sim, current_target, salience);
        const double progress = prev_distance - next_obs.distance_to_target;
        const double integrity_penalty = 0.004 * (next_obs.volume_drift_pct + next_obs.area_drift_pct);
        const double reward = progress + 0.015 * next_obs.energy - 0.010 * next_obs.homeostasis_score - integrity_penalty;

        CellTaskTraceRow row;
        row.step = step;
        row.centroid = next_obs.centroid;
        row.target = current_target;
        row.reward = reward;
        row.distance_to_target = next_obs.distance_to_target;
        row.energy = next_obs.energy;
        row.homeostasis_score = next_obs.homeostasis_score;
        row.volume_drift_pct = next_obs.volume_drift_pct;
        row.area_drift_pct = next_obs.area_drift_pct;
        row.centroid_speed = next_obs.centroid_speed;
        episode.trace.push_back(row);

        episode.metrics.cumulative_reward += reward;
        mean_distance += next_obs.distance_to_target;
        mean_energy += next_obs.energy;
        mean_homeo += next_obs.homeostasis_score;
        episode.metrics.max_volume_drift_pct = std::max(episode.metrics.max_volume_drift_pct, next_obs.volume_drift_pct);
        episode.metrics.max_area_drift_pct = std::max(episode.metrics.max_area_drift_pct, next_obs.area_drift_pct);

        if (damage_injected && episode.metrics.recovery_steps < 0) {
            const bool recovered =
                next_obs.volume_drift_pct <= sim.params().homeostasis_tol_volume_pct &&
                next_obs.area_drift_pct <= sim.params().homeostasis_tol_area_pct &&
                next_obs.homeostasis_score <= 0.25;
            if (recovered) {
                episode.metrics.recovery_steps = static_cast<int>(step) - cfg.damage_step;
            }
        }

        prev_obs = next_obs;
    }

    const double denom = (cfg.steps == 0) ? 1.0 : static_cast<double>(cfg.steps);
    episode.metrics.mean_distance = mean_distance / denom;
    episode.metrics.mean_energy = mean_energy / denom;
    episode.metrics.mean_homeostasis = mean_homeo / denom;
    episode.metrics.final_distance = episode.trace.empty() ? 0.0 : episode.trace.back().distance_to_target;
    episode.metrics.shift_count = shift_count;
    return episode;
}

void write_cell_task_csv(const std::string& path,
                         const CellTaskEpisode& episode) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return;

    out << "step,centroid_x,centroid_y,centroid_z,target_x,target_y,target_z,reward,distance_to_target,energy,homeostasis_score,volume_drift_pct,area_drift_pct,centroid_speed\n";
    for (const auto& row : episode.trace) {
        out << row.step << ','
            << row.centroid.x << ','
            << row.centroid.y << ','
            << row.centroid.z << ','
            << row.target.x << ','
            << row.target.y << ','
            << row.target.z << ','
            << row.reward << ','
            << row.distance_to_target << ','
            << row.energy << ','
            << row.homeostasis_score << ','
            << row.volume_drift_pct << ','
            << row.area_drift_pct << ','
            << row.centroid_speed << '\n';
    }
}

} // namespace cells
