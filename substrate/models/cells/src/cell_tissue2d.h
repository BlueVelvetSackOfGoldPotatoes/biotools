#pragma once

#include <cstddef>
#include <random>
#include <string>
#include <vector>

namespace cells {

static constexpr std::size_t kTissueEnvFeatureCount = 8;

struct Vec2 {
    double x = 0.0;
    double y = 0.0;

    Vec2() = default;
    Vec2(double x_, double y_) : x(x_), y(y_) {}

    Vec2& operator+=(const Vec2& o) {
        x += o.x;
        y += o.y;
        return *this;
    }
    Vec2& operator-=(const Vec2& o) {
        x -= o.x;
        y -= o.y;
        return *this;
    }
    Vec2& operator*=(double s) {
        x *= s;
        y *= s;
        return *this;
    }
    Vec2& operator/=(double s) {
        x /= s;
        y /= s;
        return *this;
    }
};

inline Vec2 operator+(Vec2 a, const Vec2& b) { return a += b; }
inline Vec2 operator-(Vec2 a, const Vec2& b) { return a -= b; }
inline Vec2 operator*(Vec2 a, double s) { return a *= s; }
inline Vec2 operator*(double s, Vec2 a) { return a *= s; }
inline Vec2 operator/(Vec2 a, double s) { return a /= s; }

double dot(const Vec2& a, const Vec2& b);
double norm2(const Vec2& v);
double norm(const Vec2& v);
Vec2 normalized(const Vec2& v);

struct Genome2D {
    std::size_t gene_dim = 8;

    // Row-major [in_gene][out_gene].
    std::vector<double> regulatory_w;
    std::vector<double> bias;

    // Environment projection: [feature][gene], features =
    // morphogen, nutrient, negative_sdf, local_density,
    // region_identity, attention_gate, teacher_signal, novelty.
    std::vector<double> env_w;

    // Readouts from gene state.
    std::vector<double> growth_w;
    std::vector<double> motility_w;
    std::vector<double> adhesion_w;

    double division_energy = 1.15;
    double mutation_std = 0.015;

    static Genome2D random(std::size_t gene_dim, unsigned int seed);
    void mutate(std::mt19937& rng, double scale = 1.0);
};

struct Cell2D {
    std::size_t id = 0;
    std::size_t parent_id = 0;

    Vec2 position;
    Vec2 velocity;

    std::vector<double> genes;

    double radius = 0.01;
    double energy = 1.0;
    double age = 0.0;
    bool alive = true;

    double growth_signal = 0.0;
    double motility_signal = 0.0;
    double adhesion_signal = 0.0;
};

struct Tissue2DParams {
    double dt = 0.02;
    int substeps = 1;

    double drag = 2.5;
    double interaction_radius = 0.06;
    double local_density_radius = 0.12;
    double repulsion_k = 10.0;
    double adhesion_k = 2.0;
    double max_speed = 0.25;

    double base_radius = 0.010;
    double min_radius = 0.006;
    double max_radius = 0.018;
    double growth_radius_gain = 0.006;
    double division_jitter = 0.012;
    double density_for_division = 1000.0;

    double energy_gain = 0.16;
    double energy_loss_outside = 0.06;
    double metabolic_cost = 0.02;
    double apoptosis_energy = 0.02;
    double max_age = 100.0;

    std::size_t max_cells = 4000;
};

struct FieldSample2D {
    // Signed distance to target boundary: negative means inside.
    double sdf = 0.0;
    Vec2 grad_sdf;

    double morphogen = 0.0;
    double nutrient = 0.0;
    double region_identity = 0.0;
    double attention_gate = 0.0;
    double teacher_signal = 0.0;
    double novelty = 0.0;
    Vec2 guidance;
};

class Field2D {
public:
    virtual ~Field2D() = default;
    virtual FieldSample2D sample(const Vec2& p) const = 0;
    virtual Vec2 clamp_to_domain(const Vec2& p) const = 0;
};

class LeafField2D final : public Field2D {
public:
    struct Config {
        Vec2 center{0.0, 0.0};
        double half_length = 1.0;
        double max_half_width = 0.55;
        double taper_power = 1.7;
        double serration = 0.03;
        double serration_freq = 8.0;
        double domain_pad = 0.25;
    };

    LeafField2D();
    explicit LeafField2D(const Config& cfg);

    FieldSample2D sample(const Vec2& p) const override;
    Vec2 clamp_to_domain(const Vec2& p) const override;

    double signed_distance(const Vec2& p) const;
    Vec2 signed_distance_gradient(const Vec2& p) const;
    double half_width_for_normalized_x(double xn) const;

    const Config& config() const { return cfg_; }

private:
    Config cfg_;
};

struct Tissue2DStats {
    std::size_t step = 0;
    std::size_t alive_cells = 0;
    std::size_t divisions = 0;
    std::size_t deaths = 0;

    double mean_energy = 0.0;
    double mean_abs_sdf = 0.0;
    double inside_fraction = 0.0;
    double iou_to_leaf = 0.0;

    Vec2 centroid;
};

class CellSheet2D {
public:
    CellSheet2D(const Genome2D& genome, const Tissue2DParams& params, unsigned int seed = 42);

    void seed_disc(std::size_t count, double radius);
    void seed_grid(std::size_t nx, std::size_t ny, const Vec2& min_corner, const Vec2& max_corner);
    void step(const Field2D& field);
    void run(const Field2D& field, std::size_t steps);

    Tissue2DStats compute_stats(const LeafField2D* leaf_eval = nullptr, int grid_res = 160) const;

    void export_cells_csv(const std::string& path) const;
    void export_genome_csv(const std::string& path) const;

    const Genome2D& genome() const { return genome_; }
    const Tissue2DParams& params() const { return params_; }
    Tissue2DParams& params() { return params_; }
    const std::vector<Cell2D>& cells() const { return cells_; }
    const Tissue2DStats& last_stats() const { return last_stats_; }

private:
    struct SpatialCell {
        int gx = 0;
        int gy = 0;
    };

    void rebuild_spatial_index(double cell_size);
    std::vector<std::size_t> query_neighbors(const Vec2& p, double radius, std::size_t skip_id) const;
    double local_density(std::size_t idx) const;

    void update_gene_program(Cell2D& c, const FieldSample2D& s, double density, double dt);
    void apply_interactions(const Field2D& field, double dt);
    void integrate(const Field2D& field, double dt);
    void divide_and_prune();
    void refresh_stats(const LeafField2D* leaf_eval);
    double estimate_iou_to_leaf(const LeafField2D& leaf, int grid_res) const;

    static long long hash_key(int gx, int gy);

    Genome2D genome_;
    Tissue2DParams params_;
    std::mt19937 rng_;

    std::vector<Cell2D> cells_;
    std::size_t next_id_ = 1;

    std::vector<SpatialCell> cached_cell_coords_;
    std::vector<long long> cached_cell_keys_;
    std::vector<std::vector<std::size_t>> buckets_;
    int bucket_stride_ = 1;
    double bucket_cell_size_ = 1.0;

    std::size_t step_counter_ = 0;
    std::size_t divisions_total_ = 0;
    std::size_t deaths_total_ = 0;
    Tissue2DStats last_stats_;
};

} // namespace cells
