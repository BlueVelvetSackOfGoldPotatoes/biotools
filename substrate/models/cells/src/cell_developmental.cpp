#include "models/cells/src/cell_developmental.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>

namespace cells {
namespace {

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

inline double sqr(double x) { return x * x; }

Vec2 teacher_anchor(int label, std::size_t classes) {
    if (classes <= 1) return Vec2(0.0, 0.82);
    const double t = static_cast<double>(std::max(0, label)) / static_cast<double>(classes - 1);
    return Vec2(-0.9 + 1.8 * t, 0.82);
}

std::vector<double> one_hot(std::size_t classes, int label) {
    std::vector<double> t(classes, 0.0);
    if (classes > 0) {
        const int y = std::max(0, std::min(static_cast<int>(classes - 1), label));
        t[static_cast<std::size_t>(y)] = 1.0;
    }
    return t;
}

} // namespace

DevelopmentalMNISTField::DevelopmentalMNISTField()
    : image_(28U * 28U, 0.0), center_of_mass_(0.0, 0.0) {}

void DevelopmentalMNISTField::set_image(const std::vector<double>& image) {
    image_.assign(28U * 28U, 0.0);
    const std::size_t n = std::min<std::size_t>(image_.size(), image.size());
    for (std::size_t i = 0; i < n; ++i) image_[i] = clamp(image[i], 0.0, 1.0);
    update_center_of_mass();
}

void DevelopmentalMNISTField::set_teacher_label(int label, bool enabled) {
    teacher_label_ = label;
    teacher_enabled_ = enabled;
}

void DevelopmentalMNISTField::set_novelty(double novelty) {
    novelty_ = std::max(0.0, novelty);
}

void DevelopmentalMNISTField::update_center_of_mass() {
    double mx = 0.0;
    double my = 0.0;
    double total = 0.0;
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            const double v = image_[static_cast<std::size_t>(y * 28 + x)];
            total += v;
            mx += v * (-1.0 + 2.0 * (static_cast<double>(x) + 0.5) / 28.0);
            my += v * (-1.0 + 2.0 * (static_cast<double>(y) + 0.5) / 28.0);
        }
    }
    if (total <= 1e-9) {
        center_of_mass_ = Vec2(0.0, 0.0);
    } else {
        center_of_mass_ = Vec2(mx / total, my / total);
    }
}

double DevelopmentalMNISTField::pixel_at(double x, double y) const {
    const double u = clamp(0.5 * (x + 1.0), 0.0, 0.999999);
    const double v = clamp(0.5 * (y + 1.0), 0.0, 0.999999);
    const int ix = std::max(0, std::min(27, static_cast<int>(u * 28.0)));
    const int iy = std::max(0, std::min(27, static_cast<int>(v * 28.0)));
    return image_[static_cast<std::size_t>(iy * 28 + ix)];
}

FieldSample2D DevelopmentalMNISTField::sample(const Vec2& p) const {
    FieldSample2D s;
    const double dx = std::max(std::abs(p.x) - 1.0, 0.0);
    const double dy = std::max(std::abs(p.y) - 1.0, 0.0);
    const double outside = std::sqrt(dx * dx + dy * dy);
    const double inside = std::min(1.0 - std::max(std::abs(p.x), std::abs(p.y)), 0.0);
    s.sdf = outside - inside;
    s.grad_sdf = normalized(Vec2(p.x, p.y));

    const double pix = pixel_at(p.x, p.y);
    s.morphogen = pix;
    s.nutrient = 0.15 + 0.85 * pix;
    s.region_identity = clamp(0.5 * (p.x + 1.0), 0.0, 1.0);
    s.attention_gate = clamp(0.2 + 0.8 * pix, 0.0, 1.0);
    s.novelty = clamp(novelty_ * (0.25 + 0.75 * (1.0 - pix)), 0.0, 1.5);

    Vec2 guide = normalized(center_of_mass_ - p);
    if (teacher_enabled_ && teacher_label_ >= 0) {
        const Vec2 anchor = teacher_anchor(teacher_label_, 10);
        const Vec2 d = p - anchor;
        s.teacher_signal = std::exp(-(dot(d, d) / 0.08));
        guide = normalized(guide * 0.6 + normalized(anchor - p) * 0.4);
    }
    s.guidance = guide;
    return s;
}

Vec2 DevelopmentalMNISTField::clamp_to_domain(const Vec2& p) const {
    return Vec2(clamp(p.x, -1.05, 1.05), clamp(p.y, -1.05, 1.05));
}

DevelopmentalTissueClassifier::DevelopmentalTissueClassifier(const DevelopmentalTissueConfig& cfg)
    : cfg_(cfg),
      genome_(Genome2D::random(std::max<std::size_t>(2, cfg.gene_dim), cfg.seed + 1U)),
      sheet_(genome_, cfg.tissue_params, cfg.seed + 2U),
      rng_(cfg.seed + 3U) {
    cfg_.tissue_params.max_cells = std::max<std::size_t>(cfg_.tissue_params.max_cells,
                                                         cfg_.seed_grid_x * cfg_.seed_grid_y + 32U);
    cfg_.tissue_params.max_age = 1e9;
    cfg_.tissue_params.apoptosis_energy = -1.0;
    genome_.division_energy = 10.0;
    genome_.mutation_std = 0.0;
    initialize_sheet();
}

void DevelopmentalTissueClassifier::initialize_sheet() {
    sheet_ = CellSheet2D(genome_, cfg_.tissue_params, cfg_.seed + 2U);
    sheet_.seed_grid(cfg_.seed_grid_x, cfg_.seed_grid_y, Vec2(-0.92, -0.92), Vec2(0.92, 0.92));
    edges_.clear();
    contact_timers_.clear();
    previous_label_ = -1;
    sync_nodes();
}

void DevelopmentalTissueClassifier::reset_state() {
    initialize_sheet();
}

void DevelopmentalTissueClassifier::reset_dynamics() {
    sheet_ = CellSheet2D(genome_, cfg_.tissue_params, cfg_.seed + 2U);
    sheet_.seed_grid(cfg_.seed_grid_x, cfg_.seed_grid_y, Vec2(-0.92, -0.92), Vec2(0.92, 0.92));
    contact_timers_.clear();
    sync_nodes();
    for (auto& n : nodes_) {
        n.activation = 0.0;
        n.prev_activation = 0.0;
        n.homeostasis_error = 0.0;
        n.attention = 0.0;
        n.teacher = 0.0;
        n.novelty = 0.0;
    }
}

void DevelopmentalTissueClassifier::set_internal_teacher(bool enabled) {
    cfg_.internal_teacher = enabled;
}

bool DevelopmentalTissueClassifier::internal_teacher_enabled() const {
    return cfg_.internal_teacher;
}

std::size_t DevelopmentalTissueClassifier::parameter_count() const {
    return edge_count() + nodes_.size() * cfg_.readout_classes;
}

std::size_t DevelopmentalTissueClassifier::edge_count() const {
    return edges_.size();
}

std::uint64_t DevelopmentalTissueClassifier::pair_key(std::size_t a, std::size_t b) const {
    const std::uint64_t x = static_cast<std::uint64_t>(std::min(a, b));
    const std::uint64_t y = static_cast<std::uint64_t>(std::max(a, b));
    return (x << 32U) | y;
}

bool DevelopmentalTissueClassifier::edge_exists(std::size_t src_id, std::size_t dst_id) const {
    for (const auto& e : edges_) {
        if (e.src_id == src_id && e.dst_id == dst_id) return true;
    }
    return false;
}

void DevelopmentalTissueClassifier::sync_nodes() {
    std::unordered_map<std::size_t, NodeState> old;
    old.reserve(nodes_.size());
    for (const auto& n : nodes_) old.emplace(n.cell_id, n);

    nodes_.clear();
    id_to_index_.clear();
    const auto& cells = sheet_.cells();
    nodes_.reserve(cells.size());
    for (const auto& c : cells) {
        NodeState n;
        const auto it = old.find(c.id);
        if (it != old.end()) {
            n = it->second;
        } else {
            n.cell_id = c.id;
            n.readout_w.assign(cfg_.readout_classes, 0.0);
            n.readout_elig.assign(cfg_.readout_classes, 0.0);
            std::normal_distribution<double> nd(0.0, 0.02);
            for (double& w : n.readout_w) w = nd(rng_);
        }
        n.cell_id = c.id;
        if (n.readout_w.size() != cfg_.readout_classes) {
            n.readout_w.assign(cfg_.readout_classes, 0.0);
            n.readout_elig.assign(cfg_.readout_classes, 0.0);
        }
        id_to_index_[c.id] = nodes_.size();
        nodes_.push_back(n);
    }

    std::unordered_set<std::size_t> alive_ids;
    alive_ids.reserve(nodes_.size());
    for (const auto& c : cells) alive_ids.insert(c.id);
    edges_.erase(std::remove_if(edges_.begin(), edges_.end(), [&](const DevelopmentalEdge& e) {
        return alive_ids.find(e.src_id) == alive_ids.end() || alive_ids.find(e.dst_id) == alive_ids.end();
    }), edges_.end());
}

void DevelopmentalTissueClassifier::update_contact_graph() {
    const auto& cells = sheet_.cells();
    std::unordered_set<std::uint64_t> touched;
    touched.reserve(cells.size() * 2U);

    for (std::size_t i = 0; i < cells.size(); ++i) {
        for (std::size_t j = i + 1; j < cells.size(); ++j) {
            const Vec2 d = cells[j].position - cells[i].position;
            if (norm2(d) > cfg_.contact_radius * cfg_.contact_radius) continue;

            const std::uint64_t key = pair_key(cells[i].id, cells[j].id);
            touched.insert(key);
            double& timer = contact_timers_[key];
            timer += cfg_.tissue_params.dt;
            if (timer < cfg_.contact_time_threshold) continue;

            double compat = 0.0;
            const std::size_t gdim = std::min(cells[i].genes.size(), cells[j].genes.size());
            for (std::size_t g = 0; g < gdim; ++g) compat += cells[i].genes[g] * cells[j].genes[g];
            compat /= std::max<std::size_t>(1, gdim);
            compat = 0.5 + 0.5 * compat;
            compat += 0.25 * (1.0 - std::abs(cells[i].adhesion_signal - cells[j].adhesion_signal));

            if (compat < cfg_.compatibility_threshold || edges_.size() + 2U > cfg_.max_edges) {
                continue;
            }

            if (!edge_exists(cells[i].id, cells[j].id)) {
                edges_.push_back(DevelopmentalEdge{cells[i].id, cells[j].id, 0.05 + 0.10 * compat, 0.0, timer});
            }
            if (!edge_exists(cells[j].id, cells[i].id)) {
                edges_.push_back(DevelopmentalEdge{cells[j].id, cells[i].id, 0.05 + 0.10 * compat, 0.0, timer});
            }
        }
    }

    for (auto it = contact_timers_.begin(); it != contact_timers_.end();) {
        if (touched.find(it->first) == touched.end()) {
            it->second *= 0.85;
            if (it->second < 1e-6) {
                it = contact_timers_.erase(it);
                continue;
            }
        }
        ++it;
    }
}

void DevelopmentalTissueClassifier::settle(const DevelopmentalMNISTField& field,
                                           int steps,
                                           bool taught_phase) {
    for (int iter = 0; iter < std::max(1, steps); ++iter) {
        sheet_.step(field);
        sync_nodes();
        update_contact_graph();

        std::vector<double> incoming(nodes_.size(), 0.0);
        for (const auto& e : edges_) {
            const auto sit = id_to_index_.find(e.src_id);
            const auto dit = id_to_index_.find(e.dst_id);
            if (sit == id_to_index_.end() || dit == id_to_index_.end()) continue;
            incoming[dit->second] += e.weight * nodes_[sit->second].prev_activation;
        }

        const auto& cells = sheet_.cells();
        for (std::size_t i = 0; i < cells.size() && i < nodes_.size(); ++i) {
            const FieldSample2D s = field.sample(cells[i].position);
            NodeState& n = nodes_[i];
            n.attention = s.attention_gate;
            n.teacher = s.teacher_signal;
            n.novelty = s.novelty;
            const double local_drive =
                1.10 * s.morphogen +
                0.35 * s.nutrient +
                cfg_.attention_gain * s.attention_gate +
                cfg_.novelty_gain * s.novelty +
                0.20 * s.region_identity +
                0.25 * cells[i].motility_signal +
                0.20 * cells[i].adhesion_signal +
                0.45 * incoming[i] +
                (taught_phase ? cfg_.teacher_gain * s.teacher_signal : 0.0);
            n.activation = std::tanh(0.45 * n.prev_activation + local_drive);
            n.homeostasis_error = std::abs(n.activation - n.prev_activation);
        }

        for (auto& e : edges_) {
            const auto sit = id_to_index_.find(e.src_id);
            const auto dit = id_to_index_.find(e.dst_id);
            if (sit == id_to_index_.end() || dit == id_to_index_.end()) continue;
            const NodeState& src = nodes_[sit->second];
            const NodeState& dst = nodes_[dit->second];
            const double mod = 0.3 + dst.attention + (taught_phase ? dst.teacher : 0.0);
            e.contact_time = contact_timers_[pair_key(e.src_id, e.dst_id)];
            e.eligibility = cfg_.eligibility_decay * e.eligibility + src.prev_activation * dst.activation * mod;
            if (taught_phase) {
                e.weight += cfg_.edge_lr * e.eligibility - cfg_.weight_decay * e.weight;
                e.weight = clamp(e.weight, -cfg_.max_edge_weight, cfg_.max_edge_weight);
            }
        }

        edges_.erase(std::remove_if(edges_.begin(), edges_.end(), [&](const DevelopmentalEdge& e) {
            return std::abs(e.weight) < cfg_.prune_below;
        }), edges_.end());

        for (auto& n : nodes_) {
            n.prev_activation = n.activation;
        }
    }
}

std::vector<double> DevelopmentalTissueClassifier::compute_logits() const {
    std::vector<double> logits(cfg_.readout_classes, 0.0);
    if (nodes_.empty()) return logits;
    for (const auto& n : nodes_) {
        for (std::size_t c = 0; c < cfg_.readout_classes; ++c) {
            logits[c] += n.activation * n.readout_w[c];
        }
    }
    const double inv = 1.0 / static_cast<double>(nodes_.size());
    for (double& v : logits) v *= inv;
    return logits;
}

std::vector<double> DevelopmentalTissueClassifier::softmax(const std::vector<double>& z) const {
    if (z.empty()) return {};
    const double m = *std::max_element(z.begin(), z.end());
    std::vector<double> out(z.size(), 0.0);
    double sum = 0.0;
    for (std::size_t i = 0; i < z.size(); ++i) {
        out[i] = std::exp(z[i] - m);
        sum += out[i];
    }
    if (sum <= 0.0) return std::vector<double>(z.size(), 1.0 / static_cast<double>(z.size()));
    for (double& v : out) v /= sum;
    return out;
}

double DevelopmentalTissueClassifier::cross_entropy(const std::vector<double>& p, int label) const {
    if (p.empty()) return 0.0;
    const int y = std::max(0, std::min(static_cast<int>(p.size() - 1), label));
    return -std::log(std::max(1e-12, p[static_cast<std::size_t>(y)]));
}

double DevelopmentalTissueClassifier::mean_energy() const {
    const auto& cells = sheet_.cells();
    if (cells.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& c : cells) sum += c.energy;
    return sum / static_cast<double>(cells.size());
}

double DevelopmentalTissueClassifier::mean_homeostasis() const {
    if (nodes_.empty()) return 0.0;
    double sum = 0.0;
    for (const auto& n : nodes_) sum += n.homeostasis_error;
    return sum / static_cast<double>(nodes_.size());
}

DevelopmentalStepInfo DevelopmentalTissueClassifier::train_one(const std::vector<double>& image, int label) {
    field_.set_image(image);
    const double novelty = (previous_label_ < 0 || previous_label_ == label) ? 0.0 : 1.0;
    field_.set_novelty(novelty);
    field_.set_teacher_label(label, false);

    settle(field_, cfg_.free_settle_steps, false);
    const std::vector<double> free_logits = compute_logits();
    const std::vector<double> free_p = softmax(free_logits);

    int pred = 0;
    double conf = free_p.empty() ? 0.0 : free_p[0];
    for (std::size_t i = 1; i < free_p.size(); ++i) {
        if (free_p[i] > conf) {
            conf = free_p[i];
            pred = static_cast<int>(i);
        }
    }

    std::vector<double> target = one_hot(cfg_.readout_classes, label);
    if (cfg_.internal_teacher && !free_p.empty()) {
        const double blend = clamp(cfg_.internal_teacher_blend, 0.0, 1.0);
        for (std::size_t i = 0; i < target.size(); ++i) {
            target[i] = (1.0 - blend) * target[i] + blend * free_p[i];
        }
    }

    field_.set_teacher_label(label, true);
    settle(field_, cfg_.taught_settle_steps, true);

    for (auto& n : nodes_) {
        for (std::size_t c = 0; c < cfg_.readout_classes; ++c) {
            const double err = target[c] - (c < free_p.size() ? free_p[c] : 0.0);
            n.readout_elig[c] = cfg_.eligibility_decay * n.readout_elig[c] + n.activation * err * (0.4 + n.attention + n.teacher);
            n.readout_w[c] += cfg_.readout_lr * n.readout_elig[c] - cfg_.weight_decay * n.readout_w[c];
            n.readout_w[c] = clamp(n.readout_w[c], -cfg_.max_edge_weight, cfg_.max_edge_weight);
        }
    }

    previous_label_ = label;

    DevelopmentalStepInfo out;
    out.pred = pred;
    out.loss = cross_entropy(free_p, label);
    out.confidence = conf;
    out.mean_energy = mean_energy();
    out.mean_homeostasis = mean_homeostasis();
    out.edge_count = edges_.size();
    out.active_cells = nodes_.size();
    return out;
}

DevelopmentalStepInfo DevelopmentalTissueClassifier::evaluate_one(const std::vector<double>& image, int label) {
    field_.set_image(image);
    field_.set_novelty(0.0);
    field_.set_teacher_label(label, false);
    settle(field_, cfg_.free_settle_steps, false);
    const std::vector<double> p = softmax(compute_logits());

    DevelopmentalStepInfo out;
    if (!p.empty()) {
        out.pred = static_cast<int>(std::distance(p.begin(), std::max_element(p.begin(), p.end())));
        out.confidence = *std::max_element(p.begin(), p.end());
    }
    out.loss = cross_entropy(p, label);
    out.mean_energy = mean_energy();
    out.mean_homeostasis = mean_homeostasis();
    out.edge_count = edges_.size();
    out.active_cells = nodes_.size();
    return out;
}

std::vector<double> DevelopmentalTissueClassifier::predict_proba(const std::vector<double>& image) {
    field_.set_image(image);
    field_.set_novelty(0.0);
    field_.set_teacher_label(0, false);
    settle(field_, cfg_.free_settle_steps, false);
    return softmax(compute_logits());
}

void DevelopmentalTissueClassifier::apply_damage(double edge_fraction, unsigned int seed) {
    if (edges_.empty()) return;
    const double frac = clamp(edge_fraction, 0.0, 1.0);
    const std::size_t drop = static_cast<std::size_t>(std::floor(frac * static_cast<double>(edges_.size())));
    if (drop == 0) return;

    std::mt19937 rng(seed);
    std::vector<std::size_t> idx(edges_.size(), 0);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    for (std::size_t i = 0; i < drop; ++i) {
        edges_[idx[i]].weight = 0.0;
    }
    edges_.erase(std::remove_if(edges_.begin(), edges_.end(), [&](const DevelopmentalEdge& e) {
        return std::abs(e.weight) < cfg_.prune_below;
    }), edges_.end());
}

} // namespace cells
