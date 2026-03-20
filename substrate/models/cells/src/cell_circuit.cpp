#include "models/cells/src/cell_circuit.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <numeric>

namespace cells {
namespace {

inline double clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

inline double randn(std::mt19937& rng, double sigma) {
    if (sigma <= 0.0) return 0.0;
    std::normal_distribution<double> nd(0.0, sigma);
    return nd(rng);
}

inline std::vector<double> module_homeostasis(const CellModuleBank& bank) {
    std::vector<double> h(bank.size(), 0.0);
    for (std::size_t i = 0; i < bank.size(); ++i) {
        h[i] = bank.module(i).state().H;
    }
    return h;
}

inline CellModuleParams layer_params(double noise_scale) {
    CellModuleParams p;
    p.noise_scale = noise_scale;
    p.energy_replenish = 0.02;
    p.energy_cost = 0.015;
    p.activity_target = 0.18;
    p.activity_tol = 0.12;
    return p;
}

} // namespace

CellCircuit::CellCircuit(const CellCircuitConfig& cfg)
    : cfg_(cfg),
      rng_(cfg.seed),
      retina_(cfg.retina_dim, layer_params(0.01), cfg.seed + 1),
      v1_(cfg.v1_dim, layer_params(0.015), cfg.seed + 2),
      assoc_(cfg.assoc_dim, layer_params(0.02), cfg.seed + 3),
      decision_(cfg.decision_dim, layer_params(0.01), cfg.seed + 4) {
    std::normal_distribution<double> nd(0.0, 0.05);

    auto init_w = [&](std::vector<double>& w, std::size_t n) {
        w.resize(n);
        for (double& v : w) v = nd(rng_);
    };

    init_w(w_in_retina_, cfg_.input_dim * cfg_.retina_dim);
    init_w(w_retina_v1_, cfg_.retina_dim * cfg_.v1_dim);
    init_w(w_v1_assoc_, cfg_.v1_dim * cfg_.assoc_dim);
    init_w(w_assoc_decision_, cfg_.assoc_dim * cfg_.decision_dim);

    e_in_retina_.assign(w_in_retina_.size(), 0.0);
    e_retina_v1_.assign(w_retina_v1_.size(), 0.0);
    e_v1_assoc_.assign(w_v1_assoc_.size(), 0.0);
    e_assoc_decision_.assign(w_assoc_decision_.size(), 0.0);
}

void CellCircuit::reset_state() {
    retina_.reset();
    v1_.reset();
    assoc_.reset();
    decision_.reset();
}

std::size_t CellCircuit::parameter_count() const {
    return w_in_retina_.size() + w_retina_v1_.size() + w_v1_assoc_.size() +
           w_assoc_decision_.size();
}

std::vector<double> CellCircuit::matvec(const std::vector<double>& x, const std::vector<double>& w,
                                        std::size_t in_dim, std::size_t out_dim) {
    std::vector<double> y(out_dim, 0.0);
    for (std::size_t i = 0; i < in_dim; ++i) {
        const double xi = (i < x.size()) ? x[i] : 0.0;
        const std::size_t row = i * out_dim;
        for (std::size_t j = 0; j < out_dim; ++j) {
            y[j] += xi * w[row + j];
        }
    }
    return y;
}

std::vector<double> CellCircuit::softmax(const std::vector<double>& z) {
    if (z.empty()) return {};
    const double m = *std::max_element(z.begin(), z.end());
    std::vector<double> e(z.size(), 0.0);
    double s = 0.0;
    for (std::size_t i = 0; i < z.size(); ++i) {
        e[i] = std::exp(z[i] - m);
        s += e[i];
    }
    if (s <= 0.0) return std::vector<double>(z.size(), 1.0 / static_cast<double>(z.size()));
    for (double& v : e) v /= s;
    return e;
}

double CellCircuit::cross_entropy(const std::vector<double>& p, int label) {
    if (p.empty()) return 0.0;
    const int l = static_cast<int>(clamp(static_cast<double>(label), 0.0, static_cast<double>(p.size() - 1)));
    return -std::log(std::max(1e-12, p[static_cast<std::size_t>(l)]));
}

std::vector<double> CellCircuit::forward_hidden(const std::vector<double>& x,
                                                std::vector<double>& retina_out,
                                                std::vector<double>& v1_out,
                                                std::vector<double>& assoc_out,
                                                std::vector<double>& decision_out,
                                                bool inject_noise) {
    const int settle_steps = std::max(1, cfg_.settle_steps);
    for (int s = 0; s < settle_steps; ++s) {
        std::vector<double> z0 = matvec(x, w_in_retina_, cfg_.input_dim, cfg_.retina_dim);
        if (inject_noise) for (double& v : z0) v += randn(rng_, 0.005);
        retina_out = retina_.step(z0, cfg_.dt);

        std::vector<double> z1 = matvec(retina_out, w_retina_v1_, cfg_.retina_dim, cfg_.v1_dim);
        if (inject_noise) for (double& v : z1) v += randn(rng_, 0.005);
        v1_out = v1_.step(z1, cfg_.dt);

        std::vector<double> z2 = matvec(v1_out, w_v1_assoc_, cfg_.v1_dim, cfg_.assoc_dim);
        if (inject_noise) for (double& v : z2) v += randn(rng_, 0.005);
        assoc_out = assoc_.step(z2, cfg_.dt);

        std::vector<double> z3 = matvec(assoc_out, w_assoc_decision_, cfg_.assoc_dim, cfg_.decision_dim);
        if (inject_noise) for (double& v : z3) v += randn(rng_, 0.002);
        decision_out = decision_.step(z3, cfg_.dt);
    }

    return softmax(decision_out);
}

void CellCircuit::update_eligibility(const std::vector<double>& pre, const std::vector<double>& post,
                                     std::vector<double>& elig, std::size_t in_dim, std::size_t out_dim) {
    const double decay = clamp(cfg_.eligibility_decay, 0.0, 1.0);
    for (std::size_t i = 0; i < in_dim; ++i) {
        const double pi = (i < pre.size()) ? pre[i] : 0.0;
        const std::size_t row = i * out_dim;
        for (std::size_t j = 0; j < out_dim; ++j) {
            const double pj = (j < post.size()) ? post[j] : 0.0;
            elig[row + j] = decay * elig[row + j] + pi * pj;
        }
    }
}

void CellCircuit::apply_three_factor_update(const std::vector<double>& post_h,
                                            double reward,
                                            std::vector<double>& w,
                                            std::vector<double>& elig,
                                            std::size_t in_dim,
                                            std::size_t out_dim) {
    const double w_clip = std::max(0.25, cfg_.weight_clip);
    for (std::size_t i = 0; i < in_dim; ++i) {
        const std::size_t row = i * out_dim;
        for (std::size_t j = 0; j < out_dim; ++j) {
            const double h = (j < post_h.size()) ? post_h[j] : 0.0;
            const double gate = 1.0 / (1.0 + std::max(0.0, h) / std::max(1e-8, cfg_.homeostasis_hmax));
            const double mod = reward;
            const double signal = elig[row + j];
            const double dw = cfg_.lr * mod * signal * gate - cfg_.weight_decay * w[row + j];
            w[row + j] += dw;
            w[row + j] = clamp(w[row + j], -w_clip, w_clip);
        }
    }
}

CellCircuitStepInfo CellCircuit::train_one(const std::vector<double>& x, int label) {
    if (cfg_.reset_state_per_sample) {
        reset_state();
    }

    std::vector<double> retina_out;
    std::vector<double> v1_out;
    std::vector<double> assoc_out;
    std::vector<double> decision_out;
    const std::vector<double> p = forward_hidden(x, retina_out, v1_out, assoc_out, decision_out, true);

    int pred = 0;
    double best = p.empty() ? 0.0 : p[0];
    for (std::size_t j = 1; j < p.size(); ++j) {
        if (p[j] > best) {
            best = p[j];
            pred = static_cast<int>(j);
        }
    }

    const double loss = cross_entropy(p, label);
    const double target_prob = (label >= 0 && static_cast<std::size_t>(label) < p.size()) ? p[static_cast<std::size_t>(label)] : 0.0;
    const double reward = (pred == label) ? (1.0 - target_prob) : -std::max(0.1, target_prob);

    update_eligibility(x, retina_out, e_in_retina_, cfg_.input_dim, cfg_.retina_dim);
    update_eligibility(retina_out, v1_out, e_retina_v1_, cfg_.retina_dim, cfg_.v1_dim);
    update_eligibility(v1_out, assoc_out, e_v1_assoc_, cfg_.v1_dim, cfg_.assoc_dim);
    update_eligibility(assoc_out, decision_out, e_assoc_decision_, cfg_.assoc_dim, cfg_.decision_dim);

    const double hidden_reward = reward * clamp(cfg_.hidden_lr_scale, 0.0, 1.0);

    apply_three_factor_update(module_homeostasis(retina_), hidden_reward,
                              w_in_retina_, e_in_retina_, cfg_.input_dim, cfg_.retina_dim);
    apply_three_factor_update(module_homeostasis(v1_), hidden_reward,
                              w_retina_v1_, e_retina_v1_, cfg_.retina_dim, cfg_.v1_dim);
    apply_three_factor_update(module_homeostasis(assoc_), hidden_reward,
                              w_v1_assoc_, e_v1_assoc_, cfg_.v1_dim, cfg_.assoc_dim);
    apply_three_factor_update(module_homeostasis(decision_), reward,
                              w_assoc_decision_, e_assoc_decision_, cfg_.assoc_dim, cfg_.decision_dim);

    CellCircuitStepInfo info;
    info.pred = pred;
    info.loss = loss;
    info.reward = reward;
    info.confidence = best;
    info.mean_energy = mean_energy();
    info.mean_homeostasis = mean_homeostasis();
    return info;
}

CellCircuitStepInfo CellCircuit::evaluate_one(const std::vector<double>& x, int label) {
    if (cfg_.reset_state_per_sample) {
        reset_state();
    }

    std::vector<double> retina_out;
    std::vector<double> v1_out;
    std::vector<double> assoc_out;
    std::vector<double> decision_out;
    const std::vector<double> p = forward_hidden(x, retina_out, v1_out, assoc_out, decision_out, false);

    int pred = 0;
    double best = p.empty() ? 0.0 : p[0];
    for (std::size_t j = 1; j < p.size(); ++j) {
        if (p[j] > best) {
            best = p[j];
            pred = static_cast<int>(j);
        }
    }

    CellCircuitStepInfo info;
    info.pred = pred;
    info.loss = cross_entropy(p, label);
    info.reward = (pred == label) ? 1.0 : -1.0;
    info.confidence = best;
    info.mean_energy = mean_energy();
    info.mean_homeostasis = mean_homeostasis();
    return info;
}

std::vector<double> CellCircuit::predict_proba(const std::vector<double>& x) {
    std::vector<double> r, v, a, d;
    return forward_hidden(x, r, v, a, d, false);
}

double CellCircuit::mean_energy() const {
    const double n = static_cast<double>(retina_.size() + v1_.size() + assoc_.size() + decision_.size());
    if (n <= 0.0) return 0.0;
    return (retina_.mean_energy() * retina_.size() +
            v1_.mean_energy() * v1_.size() +
            assoc_.mean_energy() * assoc_.size() +
            decision_.mean_energy() * decision_.size()) / n;
}

double CellCircuit::mean_homeostasis() const {
    const double n = static_cast<double>(retina_.size() + v1_.size() + assoc_.size() + decision_.size());
    if (n <= 0.0) return 0.0;
    return (retina_.mean_homeostasis() * retina_.size() +
            v1_.mean_homeostasis() * v1_.size() +
            assoc_.mean_homeostasis() * assoc_.size() +
            decision_.mean_homeostasis() * decision_.size()) / n;
}

} // namespace cells
