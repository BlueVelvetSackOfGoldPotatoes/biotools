#include "models/cells/src/cell_tissue2d.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>

namespace cells {
namespace {

constexpr double kPi = 3.14159265358979323846;

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

inline int next_prime(int v) {
    auto is_prime = [](int n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if ((n % 2) == 0 || (n % 3) == 0) return false;
        for (int i = 5; i * i <= n; i += 6) {
            if ((n % i) == 0 || (n % (i + 2)) == 0) return false;
        }
        return true;
    };
    int n = std::max(3, v | 1);
    while (!is_prime(n)) {
        n += 2;
    }
    return n;
}

} // namespace

double dot(const Vec2& a, const Vec2& b) {
    return a.x * b.x + a.y * b.y;
}

double norm2(const Vec2& v) {
    return dot(v, v);
}

double norm(const Vec2& v) {
    return std::sqrt(norm2(v));
}

Vec2 normalized(const Vec2& v) {
    const double n = norm(v);
    if (n < 1e-12) return Vec2(0.0, 0.0);
    return v / n;
}

Genome2D Genome2D::random(std::size_t gene_dim, unsigned int seed) {
    Genome2D g;
    g.gene_dim = std::max<std::size_t>(2, gene_dim);
    g.regulatory_w.resize(g.gene_dim * g.gene_dim, 0.0);
    g.bias.resize(g.gene_dim, 0.0);
    g.env_w.resize(kTissueEnvFeatureCount * g.gene_dim, 0.0);
    g.growth_w.resize(g.gene_dim, 0.0);
    g.motility_w.resize(g.gene_dim, 0.0);
    g.adhesion_w.resize(g.gene_dim, 0.0);

    std::mt19937 rng(seed);
    std::normal_distribution<double> wn(0.0, 0.35);
    for (double& v : g.regulatory_w) v = wn(rng);
    for (double& v : g.bias) v = wn(rng) * 0.25;
    for (double& v : g.env_w) v = wn(rng) * 0.5;
    for (double& v : g.growth_w) v = wn(rng);
    for (double& v : g.motility_w) v = wn(rng);
    for (double& v : g.adhesion_w) v = wn(rng);
    return g;
}

void Genome2D::mutate(std::mt19937& rng, double scale) {
    const double s = std::max(0.0, mutation_std * scale);
    std::normal_distribution<double> nd(0.0, s);
    for (double& v : regulatory_w) v += nd(rng);
    for (double& v : bias) v += nd(rng) * 0.25;
    for (double& v : env_w) v += nd(rng);
    for (double& v : growth_w) v += nd(rng);
    for (double& v : motility_w) v += nd(rng);
    for (double& v : adhesion_w) v += nd(rng);
    division_energy = clamp(division_energy + nd(rng) * 0.25, 0.7, 2.0);
}

LeafField2D::LeafField2D() : cfg_(Config{}) {}

LeafField2D::LeafField2D(const Config& cfg) : cfg_(cfg) {}

double LeafField2D::half_width_for_normalized_x(double xn) const {
    const double x = clamp(xn, -1.0, 1.0);
    const double taper = std::pow(std::max(0.0, 1.0 - std::abs(x)), cfg_.taper_power);
    const double serr = 1.0 + cfg_.serration * std::sin(cfg_.serration_freq * kPi * x);
    return std::max(0.01, cfg_.max_half_width * taper * serr);
}

double LeafField2D::signed_distance(const Vec2& p) const {
    const Vec2 q = p - cfg_.center;
    const double xn = q.x / std::max(1e-8, cfg_.half_length);
    const double hw = half_width_for_normalized_x(xn);
    const double dy = std::abs(q.y) - hw;
    const double dx = std::abs(xn) - 1.0;
    // Approximate smooth SDF for a bounded tapered shape.
    const double ox = std::max(0.0, dx);
    const double oy = std::max(0.0, dy);
    const double outside = std::sqrt(ox * ox + oy * oy);
    const double inside = std::min(std::max(dx, dy), 0.0);
    return outside + inside;
}

Vec2 LeafField2D::signed_distance_gradient(const Vec2& p) const {
    // Numerical gradient is robust for this implicit shape.
    const double e = 1e-4;
    const double s0 = signed_distance(p);
    const double sx = signed_distance(Vec2(p.x + e, p.y));
    const double sy = signed_distance(Vec2(p.x, p.y + e));
    Vec2 g((sx - s0) / e, (sy - s0) / e);
    const double n = norm(g);
    if (n < 1e-9) return Vec2(0.0, 0.0);
    return g / n;
}

FieldSample2D LeafField2D::sample(const Vec2& p) const {
    FieldSample2D s;
    s.sdf = signed_distance(p);
    s.grad_sdf = signed_distance_gradient(p);

    const double xn = (p.x - cfg_.center.x) / std::max(1e-8, cfg_.half_length);
    s.morphogen = clamp(0.5 * (xn + 1.0), 0.0, 1.0);
    // Inside leaf and near centerline gives highest nutrient.
    const double centerline = std::exp(-4.0 * std::abs(p.y - cfg_.center.y));
    s.nutrient = clamp((s.sdf <= 0.0 ? 1.0 : std::exp(-2.5 * s.sdf)) * centerline, 0.0, 1.0);
    s.region_identity = clamp(0.5 * (xn + 1.0), 0.0, 1.0);
    s.attention_gate = s.nutrient;
    s.teacher_signal = 0.0;
    s.novelty = std::max(0.0, s.sdf);
    s.guidance = normalized(-1.0 * s.grad_sdf);
    return s;
}

Vec2 LeafField2D::clamp_to_domain(const Vec2& p) const {
    const double hx = cfg_.half_length + cfg_.domain_pad;
    const double hy = cfg_.max_half_width + cfg_.domain_pad;
    return Vec2(
        clamp(p.x, cfg_.center.x - hx, cfg_.center.x + hx),
        clamp(p.y, cfg_.center.y - hy, cfg_.center.y + hy));
}

CellSheet2D::CellSheet2D(const Genome2D& genome, const Tissue2DParams& params, unsigned int seed)
    : genome_(genome), params_(params), rng_(seed) {}

void CellSheet2D::seed_disc(std::size_t count, double radius) {
    cells_.clear();
    next_id_ = 1;

    if (count == 0) {
        refresh_stats(nullptr);
        return;
    }

    const std::size_t n = std::max<std::size_t>(1, count);
    const std::size_t rings = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(n))));

    for (std::size_t r = 0; r < rings && cells_.size() < n; ++r) {
        const double rr = radius * (static_cast<double>(r) + 1.0) / static_cast<double>(rings);
        const std::size_t points = std::max<std::size_t>(6, 6 * (r + 1));
        for (std::size_t i = 0; i < points && cells_.size() < n; ++i) {
            const double a = (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(points);
            Cell2D c;
            c.id = next_id_++;
            c.parent_id = c.id;
            c.position = Vec2(rr * std::cos(a), rr * std::sin(a));
            c.velocity = Vec2(0.0, 0.0);
            c.genes.assign(genome_.gene_dim, 0.0);
            c.radius = params_.base_radius;
            c.energy = 1.0;
            c.age = 0.0;
            c.alive = true;
            cells_.push_back(c);
        }
    }
    refresh_stats(nullptr);
}

void CellSheet2D::seed_grid(std::size_t nx,
                            std::size_t ny,
                            const Vec2& min_corner,
                            const Vec2& max_corner) {
    cells_.clear();
    next_id_ = 1;

    const std::size_t gx = std::max<std::size_t>(1, nx);
    const std::size_t gy = std::max<std::size_t>(1, ny);
    for (std::size_t iy = 0; iy < gy; ++iy) {
        const double ty = (gy == 1) ? 0.5 : static_cast<double>(iy) / static_cast<double>(gy - 1);
        const double y = min_corner.y + (max_corner.y - min_corner.y) * ty;
        for (std::size_t ix = 0; ix < gx; ++ix) {
            const double tx = (gx == 1) ? 0.5 : static_cast<double>(ix) / static_cast<double>(gx - 1);
            const double x = min_corner.x + (max_corner.x - min_corner.x) * tx;
            Cell2D c;
            c.id = next_id_++;
            c.parent_id = c.id;
            c.position = Vec2(x, y);
            c.velocity = Vec2(0.0, 0.0);
            c.genes.assign(genome_.gene_dim, 0.0);
            c.radius = params_.base_radius;
            c.energy = 1.0;
            c.age = 0.0;
            c.alive = true;
            cells_.push_back(c);
        }
    }
    refresh_stats(nullptr);
}

void CellSheet2D::run(const Field2D& field, std::size_t steps) {
    for (std::size_t i = 0; i < steps; ++i) {
        step(field);
    }
}

long long CellSheet2D::hash_key(int gx, int gy) {
    return (static_cast<long long>(gx) << 32) ^ static_cast<unsigned int>(gy);
}

void CellSheet2D::rebuild_spatial_index(double cell_size) {
    bucket_cell_size_ = std::max(1e-6, cell_size);
    bucket_stride_ = next_prime(static_cast<int>(std::max<std::size_t>(97, cells_.size() * 2 + 1)));
    buckets_.assign(static_cast<std::size_t>(bucket_stride_), {});
    cached_cell_coords_.assign(cells_.size(), SpatialCell{});
    cached_cell_keys_.assign(cells_.size(), 0);

    for (std::size_t i = 0; i < cells_.size(); ++i) {
        if (!cells_[i].alive) continue;
        const int gx = static_cast<int>(std::floor(cells_[i].position.x / bucket_cell_size_));
        const int gy = static_cast<int>(std::floor(cells_[i].position.y / bucket_cell_size_));
        const long long key = hash_key(gx, gy);
        const std::size_t b = static_cast<std::size_t>((key % bucket_stride_ + bucket_stride_) % bucket_stride_);
        cached_cell_coords_[i] = SpatialCell{gx, gy};
        cached_cell_keys_[i] = key;
        buckets_[b].push_back(i);
    }
}

std::vector<std::size_t> CellSheet2D::query_neighbors(const Vec2& p, double radius, std::size_t skip_id) const {
    std::vector<std::size_t> out;
    if (buckets_.empty()) return out;

    const double r = std::max(0.0, radius);
    const double r2 = r * r;
    const int gx0 = static_cast<int>(std::floor(p.x / bucket_cell_size_));
    const int gy0 = static_cast<int>(std::floor(p.y / bucket_cell_size_));
    const int rg = std::max(1, static_cast<int>(std::ceil(r / bucket_cell_size_)));

    for (int dx = -rg; dx <= rg; ++dx) {
        for (int dy = -rg; dy <= rg; ++dy) {
            const int gx = gx0 + dx;
            const int gy = gy0 + dy;
            const long long key = hash_key(gx, gy);
            const std::size_t b = static_cast<std::size_t>((key % bucket_stride_ + bucket_stride_) % bucket_stride_);
            for (std::size_t idx : buckets_[b]) {
                if (idx == skip_id) continue;
                if (!cells_[idx].alive) continue;
                if (cached_cell_keys_[idx] != key) continue;
                const Vec2 d = cells_[idx].position - p;
                if (norm2(d) <= r2) out.push_back(idx);
            }
        }
    }
    return out;
}

double CellSheet2D::local_density(std::size_t idx) const {
    if (idx >= cells_.size() || !cells_[idx].alive) return 0.0;
    const auto neigh = query_neighbors(cells_[idx].position, params_.local_density_radius, idx);
    const double area = kPi * params_.local_density_radius * params_.local_density_radius;
    return area > 1e-12 ? static_cast<double>(neigh.size()) / area : 0.0;
}

void CellSheet2D::update_gene_program(Cell2D& c, const FieldSample2D& s, double density, double dt) {
    if (c.genes.size() != genome_.gene_dim) {
        c.genes.assign(genome_.gene_dim, 0.0);
    }
    const std::size_t gdim = genome_.gene_dim;
    std::vector<double> next(gdim, 0.0);

    const double features[kTissueEnvFeatureCount] = {
        s.morphogen,
        s.nutrient,
        std::max(0.0, -s.sdf),
        density,
        s.region_identity,
        s.attention_gate,
        s.teacher_signal,
        s.novelty
    };

    for (std::size_t j = 0; j < gdim; ++j) {
        double v = genome_.bias[j];
        for (std::size_t k = 0; k < gdim; ++k) {
            v += c.genes[k] * genome_.regulatory_w[k * gdim + j];
        }
        for (std::size_t f = 0; f < kTissueEnvFeatureCount; ++f) {
            v += features[f] * genome_.env_w[f * gdim + j];
        }
        next[j] = std::tanh(v);
    }

    const double blend = clamp(5.0 * dt, 0.0, 1.0);
    for (std::size_t j = 0; j < gdim; ++j) {
        c.genes[j] = (1.0 - blend) * c.genes[j] + blend * next[j];
    }

    double growth = 0.0;
    double motility = 0.0;
    double adhesion = 0.0;
    for (std::size_t j = 0; j < gdim; ++j) {
        growth += c.genes[j] * genome_.growth_w[j];
        motility += c.genes[j] * genome_.motility_w[j];
        adhesion += c.genes[j] * genome_.adhesion_w[j];
    }
    c.growth_signal = sigmoid(growth);
    c.motility_signal = sigmoid(motility);
    c.adhesion_signal = sigmoid(adhesion);

    const double target_radius = params_.base_radius + params_.growth_radius_gain * c.growth_signal;
    c.radius += (target_radius - c.radius) * clamp(4.0 * dt, 0.0, 1.0);
    c.radius = clamp(c.radius, params_.min_radius, params_.max_radius);

    const double inside_gain = (s.sdf <= 0.0) ? params_.energy_gain : -params_.energy_loss_outside * std::min(1.0, s.sdf);
    const double cost = params_.metabolic_cost * (0.6 + 0.7 * c.growth_signal + 0.4 * std::abs(c.motility_signal));
    c.energy += dt * (inside_gain - cost);
    c.energy = clamp(c.energy, 0.0, 2.0);
    c.age += dt;
}

void CellSheet2D::apply_interactions(const Field2D& field, double dt) {
    (void)field;
    const double r = params_.interaction_radius;
    for (std::size_t i = 0; i < cells_.size(); ++i) {
        if (!cells_[i].alive) continue;
        const auto neigh = query_neighbors(cells_[i].position, r, i);
        for (std::size_t j : neigh) {
            if (j <= i) continue;
            if (!cells_[j].alive) continue;

            Vec2 d = cells_[j].position - cells_[i].position;
            const double dist = std::max(1e-9, norm(d));
            const Vec2 n = d / dist;
            const double desired = (cells_[i].radius + cells_[j].radius) * 1.1;

            if (dist < desired) {
                const double push = params_.repulsion_k * (desired - dist);
                cells_[i].velocity -= n * (push * dt);
                cells_[j].velocity += n * (push * dt);
            } else if (dist < r) {
                const double adhes = params_.adhesion_k * 0.5 * (cells_[i].adhesion_signal + cells_[j].adhesion_signal);
                const double pull = adhes * (dist - desired);
                cells_[i].velocity += n * (pull * dt);
                cells_[j].velocity -= n * (pull * dt);
            }
        }
    }
}

void CellSheet2D::integrate(const Field2D& field, double dt) {
    for (Cell2D& c : cells_) {
        if (!c.alive) continue;
        const FieldSample2D s = field.sample(c.position);
        const Vec2 inward = normalized(-1.0 * s.grad_sdf);
        const Vec2 axial((s.morphogen >= 0.5) ? 1.0 : -1.0, 0.0);
        // Always keep a restorative component toward the leaf manifold;
        // add a smaller axial drift for lamina elongation.
        Vec2 guide = inward * (0.55 + 0.50 * std::max(0.0, s.sdf)) +
                     axial * 0.20 +
                     s.guidance * (0.25 + 0.50 * s.attention_gate);
        guide = normalized(guide);

        c.velocity += guide * (0.20 * c.motility_signal * dt);
        c.velocity *= std::exp(-params_.drag * dt);

        const double sp = norm(c.velocity);
        if (sp > params_.max_speed) {
            c.velocity *= params_.max_speed / std::max(1e-9, sp);
        }

        c.position += c.velocity * dt;
        const Vec2 clamped = field.clamp_to_domain(c.position);
        if (std::abs(clamped.x - c.position.x) > 1e-12 || std::abs(clamped.y - c.position.y) > 1e-12) {
            c.position = clamped;
            c.velocity *= 0.3;
        }
    }
}

void CellSheet2D::divide_and_prune() {
    std::vector<Cell2D> survivors;
    survivors.reserve(cells_.size());
    std::vector<Cell2D> daughters;
    daughters.reserve(cells_.size() / 2 + 1);

    std::normal_distribution<double> mut(0.0, genome_.mutation_std);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::uniform_real_distribution<double> ang(0.0, 2.0 * kPi);

    for (std::size_t i = 0; i < cells_.size(); ++i) {
        Cell2D c = cells_[i];
        if (!c.alive) continue;

        const bool die = (c.energy < params_.apoptosis_energy) || (c.age > params_.max_age);
        if (die) {
            deaths_total_++;
            continue;
        }

        const double density = local_density(i);
        const bool under_capacity = (survivors.size() + daughters.size()) < params_.max_cells;
        const bool divide = under_capacity &&
                            (density < params_.density_for_division) &&
                            (c.energy > genome_.division_energy);

        if (divide) {
            const double a = ang(rng_);
            const Vec2 jitter = Vec2(std::cos(a), std::sin(a)) * params_.division_jitter;

            Cell2D d = c;
            d.id = next_id_++;
            d.parent_id = c.id;
            d.position = c.position + jitter;
            d.velocity = c.velocity - jitter * 0.5;
            d.age = 0.0;
            d.energy = c.energy * 0.5;
            d.radius = clamp(c.radius * 0.95, params_.min_radius, params_.max_radius);
            for (double& g : d.genes) {
                g += mut(rng_) * (0.5 + 0.5 * uni(rng_));
            }

            c.position -= jitter * 0.5;
            c.velocity += jitter * 0.5;
            c.energy *= 0.5;
            c.age = 0.0;

            daughters.push_back(d);
            divisions_total_++;
        }

        survivors.push_back(c);
    }

    survivors.insert(survivors.end(), daughters.begin(), daughters.end());
    if (survivors.size() > params_.max_cells) {
        survivors.resize(params_.max_cells);
    }
    cells_.swap(survivors);
}

void CellSheet2D::refresh_stats(const LeafField2D* leaf_eval) {
    Tissue2DStats s;
    s.step = step_counter_;
    s.alive_cells = cells_.size();
    s.divisions = divisions_total_;
    s.deaths = deaths_total_;

    if (cells_.empty()) {
        last_stats_ = s;
        return;
    }

    double energy_sum = 0.0;
    double abs_sdf_sum = 0.0;
    double inside_count = 0.0;
    Vec2 centroid(0.0, 0.0);

    for (const Cell2D& c : cells_) {
        energy_sum += c.energy;
        centroid += c.position;
        if (leaf_eval != nullptr) {
            const double sdf = leaf_eval->signed_distance(c.position);
            abs_sdf_sum += std::abs(sdf);
            if (sdf <= 0.0) inside_count += 1.0;
        }
    }

    s.mean_energy = energy_sum / static_cast<double>(cells_.size());
    s.centroid = centroid / static_cast<double>(cells_.size());

    if (leaf_eval != nullptr) {
        s.mean_abs_sdf = abs_sdf_sum / static_cast<double>(cells_.size());
        s.inside_fraction = inside_count / static_cast<double>(cells_.size());
        s.iou_to_leaf = estimate_iou_to_leaf(*leaf_eval, 160);
    }
    last_stats_ = s;
}

double CellSheet2D::estimate_iou_to_leaf(const LeafField2D& leaf, int grid_res) const {
    const int n = std::max(32, grid_res);
    const auto cfg = leaf.config();
    const double hx = cfg.half_length + cfg.domain_pad;
    const double hy = cfg.max_half_width + cfg.domain_pad;

    std::vector<unsigned char> pred(static_cast<std::size_t>(n * n), 0);
    std::vector<unsigned char> target(static_cast<std::size_t>(n * n), 0);

    auto idx = [n](int ix, int iy) { return static_cast<std::size_t>(iy * n + ix); };

    for (int iy = 0; iy < n; ++iy) {
        const double y = cfg.center.y - hy + (2.0 * hy) * (static_cast<double>(iy) + 0.5) / static_cast<double>(n);
        for (int ix = 0; ix < n; ++ix) {
            const double x = cfg.center.x - hx + (2.0 * hx) * (static_cast<double>(ix) + 0.5) / static_cast<double>(n);
            target[idx(ix, iy)] = (leaf.signed_distance(Vec2(x, y)) <= 0.0) ? 1 : 0;
        }
    }

    const double gx = (2.0 * hx) / static_cast<double>(n);
    const double gy = (2.0 * hy) / static_cast<double>(n);
    for (const Cell2D& c : cells_) {
        const int ix0 = std::max(0, static_cast<int>(std::floor((c.position.x - c.radius - (cfg.center.x - hx)) / gx)));
        const int ix1 = std::min(n - 1, static_cast<int>(std::ceil((c.position.x + c.radius - (cfg.center.x - hx)) / gx)));
        const int iy0 = std::max(0, static_cast<int>(std::floor((c.position.y - c.radius - (cfg.center.y - hy)) / gy)));
        const int iy1 = std::min(n - 1, static_cast<int>(std::ceil((c.position.y + c.radius - (cfg.center.y - hy)) / gy)));
        for (int iy = iy0; iy <= iy1; ++iy) {
            const double y = cfg.center.y - hy + (iy + 0.5) * gy;
            for (int ix = ix0; ix <= ix1; ++ix) {
                const double x = cfg.center.x - hx + (ix + 0.5) * gx;
                const Vec2 d = Vec2(x, y) - c.position;
                if (norm2(d) <= c.radius * c.radius) {
                    pred[idx(ix, iy)] = 1;
                }
            }
        }
    }

    double inter = 0.0;
    double uni = 0.0;
    for (std::size_t i = 0; i < pred.size(); ++i) {
        if (pred[i] && target[i]) inter += 1.0;
        if (pred[i] || target[i]) uni += 1.0;
    }
    if (uni <= 0.0) return 0.0;
    return inter / uni;
}

void CellSheet2D::step(const Field2D& field) {
    const int substeps = std::max(1, params_.substeps);
    const double dt = params_.dt / static_cast<double>(substeps);

    for (int ss = 0; ss < substeps; ++ss) {
        rebuild_spatial_index(std::max(params_.interaction_radius, params_.local_density_radius));
        for (std::size_t i = 0; i < cells_.size(); ++i) {
            if (!cells_[i].alive) continue;
            const FieldSample2D sample = field.sample(cells_[i].position);
            const double density = local_density(i);
            update_gene_program(cells_[i], sample, density, dt);
        }
        apply_interactions(field, dt);
        integrate(field, dt);
        rebuild_spatial_index(std::max(params_.interaction_radius, params_.local_density_radius));
        divide_and_prune();
        step_counter_++;
    }
    refresh_stats(dynamic_cast<const LeafField2D*>(&field));
}

Tissue2DStats CellSheet2D::compute_stats(const LeafField2D* leaf_eval, int grid_res) const {
    Tissue2DStats out = last_stats_;
    if (leaf_eval != nullptr) {
        out.iou_to_leaf = estimate_iou_to_leaf(*leaf_eval, grid_res);
    }
    return out;
}

void CellSheet2D::export_cells_csv(const std::string& path) const {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return;

    out << "id,parent_id,x,y,vx,vy,radius,energy,age";
    for (std::size_t j = 0; j < genome_.gene_dim; ++j) out << ",g" << j;
    out << '\n';

    for (const Cell2D& c : cells_) {
        out << c.id << ','
            << c.parent_id << ','
            << c.position.x << ','
            << c.position.y << ','
            << c.velocity.x << ','
            << c.velocity.y << ','
            << c.radius << ','
            << c.energy << ','
            << c.age;
        for (double g : c.genes) out << ',' << g;
        out << '\n';
    }
}

void CellSheet2D::export_genome_csv(const std::string& path) const {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return;

    out << "section,index,value\n";
    for (std::size_t i = 0; i < genome_.regulatory_w.size(); ++i) {
        out << "regulatory_w," << i << "," << genome_.regulatory_w[i] << '\n';
    }
    for (std::size_t i = 0; i < genome_.bias.size(); ++i) {
        out << "bias," << i << "," << genome_.bias[i] << '\n';
    }
    for (std::size_t i = 0; i < genome_.env_w.size(); ++i) {
        out << "env_w," << i << "," << genome_.env_w[i] << '\n';
    }
    for (std::size_t i = 0; i < genome_.growth_w.size(); ++i) {
        out << "growth_w," << i << "," << genome_.growth_w[i] << '\n';
    }
    for (std::size_t i = 0; i < genome_.motility_w.size(); ++i) {
        out << "motility_w," << i << "," << genome_.motility_w[i] << '\n';
    }
    for (std::size_t i = 0; i < genome_.adhesion_w.size(); ++i) {
        out << "adhesion_w," << i << "," << genome_.adhesion_w[i] << '\n';
    }
    out << "scalar,division_energy," << genome_.division_energy << '\n';
    out << "scalar,mutation_std," << genome_.mutation_std << '\n';
}

} // namespace cells
