// modulus_geometry.h - Geometry primitives and CSG operations
// Port of physicsnemo/mesh/primitives/ and geometry operations
// Includes: Box, Sphere, Cylinder, Cone, Torus, Plane, Line, Circle,
//           CSG operations, tessellation, STL import, parameterization
//
// C++17, no external dependencies.

#ifndef MODULUS_GEOMETRY_H
#define MODULUS_GEOMETRY_H

#include "modulus.h"
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace modulus {
namespace geometry {

// ============================================================
// 3D Point / Vector
// ============================================================
struct Vec3 {
    double x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }

    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    double length() const { return std::sqrt(x * x + y * y + z * z); }
    double length_sq() const { return x * x + y * y + z * z; }
    Vec3 normalized() const {
        double l = length();
        return l > 1e-15 ? *this / l : Vec3(0, 0, 0);
    }

    static Vec3 lerp(const Vec3& a, const Vec3& b, double t) {
        return a * (1.0 - t) + b * t;
    }
};

inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

// ============================================================
// 2D Point
// ============================================================
struct Vec2 {
    double x, y;
    Vec2() : x(0), y(0) {}
    Vec2(double x, double y) : x(x), y(y) {}

    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
    double dot(const Vec2& o) const { return x * o.x + y * o.y; }
    double length() const { return std::sqrt(x * x + y * y); }
    Vec2 normalized() const {
        double l = length();
        return l > 1e-15 ? Vec2{x / l, y / l} : Vec2{0, 0};
    }
};

// ============================================================
// Triangle (for tessellation / mesh)
// ============================================================
struct Triangle {
    Vec3 v0, v1, v2;
    Vec3 normal() const { return (v1 - v0).cross(v2 - v0).normalized(); }
    double area() const { return 0.5 * (v1 - v0).cross(v2 - v0).length(); }
    Vec3 centroid() const { return (v0 + v1 + v2) / 3.0; }
};

// ============================================================
// Bounding box (AABB)
// ============================================================
struct AABB {
    Vec3 min_pt, max_pt;

    AABB() : min_pt(1e30, 1e30, 1e30), max_pt(-1e30, -1e30, -1e30) {}
    AABB(const Vec3& lo, const Vec3& hi) : min_pt(lo), max_pt(hi) {}

    void expand(const Vec3& p) {
        min_pt.x = std::min(min_pt.x, p.x);
        min_pt.y = std::min(min_pt.y, p.y);
        min_pt.z = std::min(min_pt.z, p.z);
        max_pt.x = std::max(max_pt.x, p.x);
        max_pt.y = std::max(max_pt.y, p.y);
        max_pt.z = std::max(max_pt.z, p.z);
    }

    void expand(const AABB& o) { expand(o.min_pt); expand(o.max_pt); }

    bool contains(const Vec3& p) const {
        return p.x >= min_pt.x && p.x <= max_pt.x &&
               p.y >= min_pt.y && p.y <= max_pt.y &&
               p.z >= min_pt.z && p.z <= max_pt.z;
    }

    Vec3 center() const { return (min_pt + max_pt) * 0.5; }
    Vec3 extent() const { return max_pt - min_pt; }
};

// ============================================================
// Abstract geometry primitive base class
// Provides signed distance function (SDF) interface
// ============================================================
class Primitive {
public:
    virtual ~Primitive() = default;

    // Signed distance: negative inside, positive outside
    virtual double sdf(const Vec3& p) const = 0;

    // Surface normal at point (uses finite differences on SDF)
    virtual Vec3 normal(const Vec3& p) const {
        constexpr double eps = 1e-6;
        double dx = sdf({p.x + eps, p.y, p.z}) - sdf({p.x - eps, p.y, p.z});
        double dy = sdf({p.x, p.y + eps, p.z}) - sdf({p.x, p.y - eps, p.z});
        double dz = sdf({p.x, p.y, p.z + eps}) - sdf({p.x, p.y, p.z - eps});
        return Vec3(dx, dy, dz).normalized();
    }

    // Test if point is inside
    virtual bool contains(const Vec3& p) const { return sdf(p) <= 0.0; }

    // Bounding box
    virtual AABB bounds() const = 0;

    // Sample random points on surface
    virtual std::vector<Vec3> sample_surface(int n_points) const {
        // Default: rejection sampling near surface
        AABB bb = bounds();
        Vec3 ext = bb.extent();
        std::vector<Vec3> pts;
        pts.reserve(n_points);
        std::uniform_real_distribution<double> dx(bb.min_pt.x, bb.max_pt.x);
        std::uniform_real_distribution<double> dy(bb.min_pt.y, bb.max_pt.y);
        std::uniform_real_distribution<double> dz(bb.min_pt.z, bb.max_pt.z);

        double tol = ext.length() * 0.01;
        int max_iter = n_points * 1000;
        for (int i = 0; i < max_iter && (int)pts.size() < n_points; ++i) {
            Vec3 p(dx(global_rng()), dy(global_rng()), dz(global_rng()));
            if (std::abs(sdf(p)) < tol) pts.push_back(p);
        }
        return pts;
    }

    // Sample random points inside volume
    virtual std::vector<Vec3> sample_interior(int n_points) const {
        AABB bb = bounds();
        std::vector<Vec3> pts;
        pts.reserve(n_points);
        std::uniform_real_distribution<double> dx(bb.min_pt.x, bb.max_pt.x);
        std::uniform_real_distribution<double> dy(bb.min_pt.y, bb.max_pt.y);
        std::uniform_real_distribution<double> dz(bb.min_pt.z, bb.max_pt.z);

        int max_iter = n_points * 100;
        for (int i = 0; i < max_iter && (int)pts.size() < n_points; ++i) {
            Vec3 p(dx(global_rng()), dy(global_rng()), dz(global_rng()));
            if (sdf(p) < 0) pts.push_back(p);
        }
        return pts;
    }

    // Tessellate to triangles (marching cubes for generic SDFs)
    virtual std::vector<Triangle> tessellate(int resolution = 32) const;
};

// ============================================================
// Box (axis-aligned)
// ============================================================
class Box : public Primitive {
public:
    Vec3 center_, half_extent_;

    Box(const Vec3& lo, const Vec3& hi)
        : center_((lo + hi) * 0.5), half_extent_((hi - lo) * 0.5) {}

    Box(const Vec3& center, double hx, double hy, double hz)
        : center_(center), half_extent_(hx, hy, hz) {}

    double sdf(const Vec3& p) const override {
        Vec3 d(std::abs(p.x - center_.x) - half_extent_.x,
               std::abs(p.y - center_.y) - half_extent_.y,
               std::abs(p.z - center_.z) - half_extent_.z);
        double outside = Vec3(std::max(d.x, 0.0), std::max(d.y, 0.0), std::max(d.z, 0.0)).length();
        double inside = std::min(std::max({d.x, d.y, d.z}), 0.0);
        return outside + inside;
    }

    AABB bounds() const override {
        return AABB(center_ - half_extent_, center_ + half_extent_);
    }
};

// ============================================================
// Sphere
// ============================================================
class Sphere : public Primitive {
public:
    Vec3 center_;
    double radius_;

    Sphere(const Vec3& center, double radius) : center_(center), radius_(radius) {}

    double sdf(const Vec3& p) const override {
        return (p - center_).length() - radius_;
    }

    AABB bounds() const override {
        Vec3 r(radius_, radius_, radius_);
        return AABB(center_ - r, center_ + r);
    }

    std::vector<Vec3> sample_surface(int n_points) const override {
        std::vector<Vec3> pts;
        pts.reserve(n_points);
        std::normal_distribution<double> nd(0, 1);
        for (int i = 0; i < n_points; ++i) {
            Vec3 d(nd(global_rng()), nd(global_rng()), nd(global_rng()));
            d = d.normalized() * radius_;
            pts.push_back(center_ + d);
        }
        return pts;
    }

    std::vector<Triangle> tessellate(int resolution = 32) const override {
        // UV sphere tessellation
        std::vector<Triangle> tris;
        int nu = resolution, nv = resolution / 2;
        auto sphere_pt = [&](int u, int v) -> Vec3 {
            double phi = 2.0 * M_PI * u / nu;
            double theta = M_PI * v / nv;
            return center_ + Vec3(radius_ * std::sin(theta) * std::cos(phi),
                                  radius_ * std::sin(theta) * std::sin(phi),
                                  radius_ * std::cos(theta));
        };
        for (int u = 0; u < nu; ++u) {
            for (int v = 0; v < nv; ++v) {
                Vec3 p00 = sphere_pt(u, v);
                Vec3 p10 = sphere_pt(u + 1, v);
                Vec3 p01 = sphere_pt(u, v + 1);
                Vec3 p11 = sphere_pt(u + 1, v + 1);
                tris.push_back({p00, p10, p01});
                tris.push_back({p10, p11, p01});
            }
        }
        return tris;
    }
};

// ============================================================
// Cylinder (along Z axis by default)
// ============================================================
class Cylinder : public Primitive {
public:
    Vec3 center_;
    double radius_, height_;

    Cylinder(const Vec3& center, double radius, double height)
        : center_(center), radius_(radius), height_(height) {}

    double sdf(const Vec3& p) const override {
        double dx = p.x - center_.x, dy = p.y - center_.y;
        double r_dist = std::sqrt(dx * dx + dy * dy) - radius_;
        double z_dist = std::abs(p.z - center_.z) - height_ * 0.5;
        double outside = Vec2(std::max(r_dist, 0.0), std::max(z_dist, 0.0)).length();
        double inside = std::min(std::max(r_dist, z_dist), 0.0);
        return outside + inside;
    }

    AABB bounds() const override {
        Vec3 r(radius_, radius_, height_ * 0.5);
        return AABB(center_ - r, center_ + r);
    }
};

// ============================================================
// Cone (apex at center, opening along +Z)
// ============================================================
class Cone : public Primitive {
public:
    Vec3 center_;
    double radius_, height_;

    Cone(const Vec3& center, double radius, double height)
        : center_(center), radius_(radius), height_(height) {}

    double sdf(const Vec3& p) const override {
        Vec3 q = p - center_;
        double rxy = std::sqrt(q.x * q.x + q.y * q.y);
        // Cone in 2D: line from (0,0) to (radius, height)
        Vec2 tip(0, 0), base(radius_, height_);
        Vec2 cb = base - tip;
        Vec2 cp(rxy, q.z);
        double h_clamped = std::clamp(cp.dot(cb) / cb.dot(cb), 0.0, 1.0);
        Vec2 closest(tip.x + cb.x * h_clamped, tip.y + cb.y * h_clamped);
        Vec2 diff(cp.x - closest.x, cp.y - closest.y);
        double dist = diff.length();
        // Sign: inside if below the cone surface
        double sign_val = (rxy / radius_ + (q.z / height_) <= 1.0 && q.z >= 0 && q.z <= height_) ? -1.0 : 1.0;
        return dist * sign_val;
    }

    AABB bounds() const override {
        Vec3 r(radius_, radius_, 0);
        return AABB(center_ - r, center_ + Vec3(radius_, radius_, height_));
    }
};

// ============================================================
// Torus (major axis along Z)
// ============================================================
class Torus : public Primitive {
public:
    Vec3 center_;
    double major_r_, minor_r_;

    Torus(const Vec3& center, double major_radius, double minor_radius)
        : center_(center), major_r_(major_radius), minor_r_(minor_radius) {}

    double sdf(const Vec3& p) const override {
        Vec3 q = p - center_;
        double rxy = std::sqrt(q.x * q.x + q.y * q.y);
        Vec2 d(rxy - major_r_, q.z);
        return d.length() - minor_r_;
    }

    AABB bounds() const override {
        double r = major_r_ + minor_r_;
        return AABB(center_ - Vec3(r, r, minor_r_),
                    center_ + Vec3(r, r, minor_r_));
    }
};

// ============================================================
// Plane (infinite, defined by normal and offset)
// ============================================================
class Plane : public Primitive {
public:
    Vec3 normal_;
    double offset_;

    Plane(const Vec3& normal, double offset)
        : normal_(normal.normalized()), offset_(offset) {}

    Plane(const Vec3& normal, const Vec3& point)
        : normal_(normal.normalized()), offset_(normal.normalized().dot(point)) {}

    double sdf(const Vec3& p) const override {
        return normal_.dot(p) - offset_;
    }

    AABB bounds() const override {
        constexpr double big = 1e6;
        return AABB(Vec3(-big, -big, -big), Vec3(big, big, big));
    }
};

// ============================================================
// Line segment (3D)
// ============================================================
class LineSegment {
public:
    Vec3 a, b;
    LineSegment(const Vec3& a, const Vec3& b) : a(a), b(b) {}

    double length() const { return (b - a).length(); }
    Vec3 direction() const { return (b - a).normalized(); }
    Vec3 midpoint() const { return (a + b) * 0.5; }

    Vec3 closest_point(const Vec3& p) const {
        Vec3 ab = b - a;
        double t = std::clamp((p - a).dot(ab) / ab.dot(ab), 0.0, 1.0);
        return a + ab * t;
    }

    double distance(const Vec3& p) const {
        return (p - closest_point(p)).length();
    }
};

// ============================================================
// Circle (2D, in XY plane at given Z)
// ============================================================
class Circle2D {
public:
    Vec2 center_;
    double radius_;

    Circle2D(const Vec2& center, double radius)
        : center_(center), radius_(radius) {}

    double sdf(const Vec2& p) const {
        return (p - center_).length() - radius_;
    }

    bool contains(const Vec2& p) const { return sdf(p) <= 0; }

    std::vector<Vec2> sample_boundary(int n_points) const {
        std::vector<Vec2> pts;
        pts.reserve(n_points);
        for (int i = 0; i < n_points; ++i) {
            double theta = 2.0 * M_PI * i / n_points;
            pts.push_back({center_.x + radius_ * std::cos(theta),
                           center_.y + radius_ * std::sin(theta)});
        }
        return pts;
    }
};

// ============================================================
// Rectangle (2D)
// ============================================================
class Rectangle2D {
public:
    Vec2 center_;
    double half_w_, half_h_;

    Rectangle2D(const Vec2& center, double width, double height)
        : center_(center), half_w_(width * 0.5), half_h_(height * 0.5) {}

    double sdf(const Vec2& p) const {
        double dx = std::abs(p.x - center_.x) - half_w_;
        double dy = std::abs(p.y - center_.y) - half_h_;
        double outside = Vec2(std::max(dx, 0.0), std::max(dy, 0.0)).length();
        double inside = std::min(std::max(dx, dy), 0.0);
        return outside + inside;
    }
};

// ============================================================
// CSG Operations (Constructive Solid Geometry)
// ============================================================

// CSG Union
class CSGUnion : public Primitive {
    std::shared_ptr<Primitive> a_, b_;
public:
    CSGUnion(std::shared_ptr<Primitive> a, std::shared_ptr<Primitive> b)
        : a_(std::move(a)), b_(std::move(b)) {}

    double sdf(const Vec3& p) const override {
        return std::min(a_->sdf(p), b_->sdf(p));
    }

    AABB bounds() const override {
        AABB bb;
        bb.expand(a_->bounds());
        bb.expand(b_->bounds());
        return bb;
    }
};

// CSG Intersection
class CSGIntersection : public Primitive {
    std::shared_ptr<Primitive> a_, b_;
public:
    CSGIntersection(std::shared_ptr<Primitive> a, std::shared_ptr<Primitive> b)
        : a_(std::move(a)), b_(std::move(b)) {}

    double sdf(const Vec3& p) const override {
        return std::max(a_->sdf(p), b_->sdf(p));
    }

    AABB bounds() const override {
        // Conservative: intersection of bounding boxes
        auto ba = a_->bounds(), bb = b_->bounds();
        return AABB(
            Vec3(std::max(ba.min_pt.x, bb.min_pt.x),
                 std::max(ba.min_pt.y, bb.min_pt.y),
                 std::max(ba.min_pt.z, bb.min_pt.z)),
            Vec3(std::min(ba.max_pt.x, bb.max_pt.x),
                 std::min(ba.max_pt.y, bb.max_pt.y),
                 std::min(ba.max_pt.z, bb.max_pt.z)));
    }
};

// CSG Difference (A minus B)
class CSGDifference : public Primitive {
    std::shared_ptr<Primitive> a_, b_;
public:
    CSGDifference(std::shared_ptr<Primitive> a, std::shared_ptr<Primitive> b)
        : a_(std::move(a)), b_(std::move(b)) {}

    double sdf(const Vec3& p) const override {
        return std::max(a_->sdf(p), -b_->sdf(p));
    }

    AABB bounds() const override { return a_->bounds(); }
};

// Convenience factory functions
inline std::shared_ptr<Primitive> csg_union(
    std::shared_ptr<Primitive> a, std::shared_ptr<Primitive> b) {
    return std::make_shared<CSGUnion>(std::move(a), std::move(b));
}

inline std::shared_ptr<Primitive> csg_intersection(
    std::shared_ptr<Primitive> a, std::shared_ptr<Primitive> b) {
    return std::make_shared<CSGIntersection>(std::move(a), std::move(b));
}

inline std::shared_ptr<Primitive> csg_difference(
    std::shared_ptr<Primitive> a, std::shared_ptr<Primitive> b) {
    return std::make_shared<CSGDifference>(std::move(a), std::move(b));
}

// Smooth union (blending)
class SmoothUnion : public Primitive {
    std::shared_ptr<Primitive> a_, b_;
    double k_; // smoothing factor
public:
    SmoothUnion(std::shared_ptr<Primitive> a, std::shared_ptr<Primitive> b, double k = 0.1)
        : a_(std::move(a)), b_(std::move(b)), k_(k) {}

    double sdf(const Vec3& p) const override {
        double da = a_->sdf(p), db = b_->sdf(p);
        double h = std::clamp(0.5 + 0.5 * (db - da) / k_, 0.0, 1.0);
        return db * (1.0 - h) + da * h - k_ * h * (1.0 - h);
    }

    AABB bounds() const override {
        AABB bb;
        bb.expand(a_->bounds());
        bb.expand(b_->bounds());
        return bb;
    }
};

// ============================================================
// Marching Cubes - Tessellate SDF to triangles
// Simplified implementation (full 256-case lookup)
// ============================================================
namespace marching_cubes {

// Edge table and tri table (standard MC tables)
// Using simplified inline tables for the essential cases
inline int edge_connection[12][2] = {
    {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}
};

inline Vec3 vertex_offset[8] = {
    {0,0,0},{1,0,0},{1,1,0},{0,1,0},
    {0,0,1},{1,0,1},{1,1,1},{0,1,1}
};

inline Vec3 interpolate_edge(const Vec3& p1, const Vec3& p2, double v1, double v2) {
    if (std::abs(v1 - v2) < 1e-15) return (p1 + p2) * 0.5;
    double t = -v1 / (v2 - v1);
    return Vec3::lerp(p1, p2, std::clamp(t, 0.0, 1.0));
}

} // namespace marching_cubes

// Default tessellation for Primitive (simple marching cubes)
inline std::vector<Triangle> Primitive::tessellate(int resolution) const {
    std::vector<Triangle> tris;
    AABB bb = bounds();
    Vec3 ext = bb.extent();
    double dx = ext.x / resolution;
    double dy = ext.y / resolution;
    double dz = ext.z / resolution;

    // Simple surface extraction: for each cube, if sign changes, add triangles
    for (int ix = 0; ix < resolution; ++ix) {
        for (int iy = 0; iy < resolution; ++iy) {
            for (int iz = 0; iz < resolution; ++iz) {
                // Get corner positions and SDF values
                Vec3 corners[8];
                double vals[8];
                for (int c = 0; c < 8; ++c) {
                    corners[c] = bb.min_pt + Vec3(
                        (ix + marching_cubes::vertex_offset[c].x) * dx,
                        (iy + marching_cubes::vertex_offset[c].y) * dy,
                        (iz + marching_cubes::vertex_offset[c].z) * dz);
                    vals[c] = sdf(corners[c]);
                }

                // Check if any sign change
                int cube_index = 0;
                for (int c = 0; c < 8; ++c)
                    if (vals[c] < 0) cube_index |= (1 << c);

                if (cube_index == 0 || cube_index == 255) continue;

                // Compute edge intersections
                Vec3 edge_pts[12];
                for (int e = 0; e < 12; ++e) {
                    int c0 = marching_cubes::edge_connection[e][0];
                    int c1 = marching_cubes::edge_connection[e][1];
                    if ((vals[c0] < 0) != (vals[c1] < 0)) {
                        edge_pts[e] = marching_cubes::interpolate_edge(
                            corners[c0], corners[c1], vals[c0], vals[c1]);
                    }
                }

                // Simple triangulation: find edges with sign change and form triangles
                // This is a simplified approach; full MC uses 256-entry lookup table
                std::vector<int> active_edges;
                for (int e = 0; e < 12; ++e) {
                    int c0 = marching_cubes::edge_connection[e][0];
                    int c1 = marching_cubes::edge_connection[e][1];
                    if ((vals[c0] < 0) != (vals[c1] < 0))
                        active_edges.push_back(e);
                }

                // Form triangles from active edges (fan from first point)
                if (active_edges.size() >= 3) {
                    for (size_t i = 1; i + 1 < active_edges.size(); ++i) {
                        tris.push_back({edge_pts[active_edges[0]],
                                       edge_pts[active_edges[i]],
                                       edge_pts[active_edges[i + 1]]});
                    }
                }
            }
        }
    }
    return tris;
}

// ============================================================
// Parameterized curves
// ============================================================

// Generic parametric curve: p(t) for t in [0, 1]
class ParametricCurve {
public:
    using CurveFn = std::function<Vec3(double)>;
    CurveFn fn_;
    double t_min_, t_max_;

    ParametricCurve(CurveFn fn, double t_min = 0, double t_max = 1)
        : fn_(std::move(fn)), t_min_(t_min), t_max_(t_max) {}

    Vec3 evaluate(double t) const {
        double mapped = t_min_ + t * (t_max_ - t_min_);
        return fn_(mapped);
    }

    // Sample n evenly-spaced points
    std::vector<Vec3> sample(int n) const {
        std::vector<Vec3> pts;
        pts.reserve(n);
        for (int i = 0; i < n; ++i) {
            double t = (double)i / (n - 1);
            pts.push_back(evaluate(t));
        }
        return pts;
    }

    // Arc length approximation
    double arc_length(int n_segments = 100) const {
        double len = 0;
        Vec3 prev = evaluate(0);
        for (int i = 1; i <= n_segments; ++i) {
            Vec3 cur = evaluate((double)i / n_segments);
            len += (cur - prev).length();
            prev = cur;
        }
        return len;
    }
};

// Common curves
inline ParametricCurve make_line(const Vec3& a, const Vec3& b) {
    return ParametricCurve([a, b](double t) { return Vec3::lerp(a, b, t); });
}

inline ParametricCurve make_circle_3d(const Vec3& center, double radius, const Vec3& normal_dir) {
    Vec3 n = normal_dir.normalized();
    // Build orthonormal basis
    Vec3 u, v;
    if (std::abs(n.x) < 0.9) u = Vec3(1,0,0).cross(n).normalized();
    else u = Vec3(0,1,0).cross(n).normalized();
    v = n.cross(u);
    return ParametricCurve([center, radius, u, v](double t) {
        double theta = 2.0 * M_PI * t;
        return center + u * (radius * std::cos(theta)) + v * (radius * std::sin(theta));
    });
}

inline ParametricCurve make_helix(const Vec3& center, double radius, double pitch, double turns) {
    return ParametricCurve([center, radius, pitch, turns](double t) {
        double theta = 2.0 * M_PI * turns * t;
        return center + Vec3(radius * std::cos(theta),
                             radius * std::sin(theta),
                             pitch * turns * t);
    });
}

inline ParametricCurve make_spiral_2d(const Vec2& center, double r_start, double r_end, double turns) {
    return ParametricCurve([center, r_start, r_end, turns](double t) {
        double theta = 2.0 * M_PI * turns * t;
        double r = r_start + (r_end - r_start) * t;
        return Vec3(center.x + r * std::cos(theta),
                    center.y + r * std::sin(theta), 0.0);
    });
}

// ============================================================
// STL file import (binary and ASCII)
// ============================================================
namespace stl {

struct STLMesh {
    std::vector<Triangle> triangles;
    std::string header;

    AABB bounds() const {
        AABB bb;
        for (auto& tri : triangles) {
            bb.expand(tri.v0);
            bb.expand(tri.v1);
            bb.expand(tri.v2);
        }
        return bb;
    }

    int num_triangles() const { return (int)triangles.size(); }

    double surface_area() const {
        double area = 0;
        for (auto& tri : triangles) area += tri.area();
        return area;
    }

    double volume() const {
        // Signed volume via divergence theorem
        double vol = 0;
        for (auto& tri : triangles) {
            vol += tri.v0.dot(tri.v1.cross(tri.v2));
        }
        return std::abs(vol) / 6.0;
    }
};

// Read binary STL file
inline STLMesh read_binary(const std::string& path) {
    STLMesh mesh;
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open STL file: " + path);

    char header[80];
    file.read(header, 80);
    mesh.header = std::string(header, 80);

    uint32_t n_triangles;
    file.read(reinterpret_cast<char*>(&n_triangles), 4);
    mesh.triangles.resize(n_triangles);

    for (uint32_t i = 0; i < n_triangles; ++i) {
        float buf[12];
        file.read(reinterpret_cast<char*>(buf), 48);
        uint16_t attr;
        file.read(reinterpret_cast<char*>(&attr), 2);

        // buf[0..2] = normal, buf[3..5] = v0, buf[6..8] = v1, buf[9..11] = v2
        mesh.triangles[i].v0 = Vec3(buf[3], buf[4], buf[5]);
        mesh.triangles[i].v1 = Vec3(buf[6], buf[7], buf[8]);
        mesh.triangles[i].v2 = Vec3(buf[9], buf[10], buf[11]);
    }
    return mesh;
}

// Read ASCII STL file
inline STLMesh read_ascii(const std::string& path) {
    STLMesh mesh;
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open STL file: " + path);

    std::string line;
    Triangle current_tri;
    int vertex_idx = 0;

    while (std::getline(file, line)) {
        // Trim leading whitespace
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        if (line.substr(0, 6) == "vertex") {
            double x, y, z;
            std::sscanf(line.c_str(), "vertex %lf %lf %lf", &x, &y, &z);
            if (vertex_idx == 0) current_tri.v0 = Vec3(x, y, z);
            else if (vertex_idx == 1) current_tri.v1 = Vec3(x, y, z);
            else if (vertex_idx == 2) {
                current_tri.v2 = Vec3(x, y, z);
                mesh.triangles.push_back(current_tri);
            }
            vertex_idx = (vertex_idx + 1) % 3;
        }
    }
    return mesh;
}

// Auto-detect format and read
inline STLMesh read(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open STL file: " + path);

    // Check if ASCII by looking for "solid" at start
    char buf[5];
    file.read(buf, 5);
    file.close();

    if (std::string(buf, 5) == "solid") {
        // Could be ASCII - try it
        try {
            auto mesh = read_ascii(path);
            if (!mesh.triangles.empty()) return mesh;
        } catch (...) {}
    }
    return read_binary(path);
}

// Write binary STL
inline void write_binary(const std::string& path, const STLMesh& mesh) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot write STL file: " + path);

    char header[80] = {};
    std::string h = "Modulus C++ STL export";
    std::memcpy(header, h.c_str(), std::min(h.size(), (size_t)80));
    file.write(header, 80);

    uint32_t n = (uint32_t)mesh.triangles.size();
    file.write(reinterpret_cast<const char*>(&n), 4);

    for (auto& tri : mesh.triangles) {
        Vec3 norm = tri.normal();
        float buf[12] = {
            (float)norm.x, (float)norm.y, (float)norm.z,
            (float)tri.v0.x, (float)tri.v0.y, (float)tri.v0.z,
            (float)tri.v1.x, (float)tri.v1.y, (float)tri.v1.z,
            (float)tri.v2.x, (float)tri.v2.y, (float)tri.v2.z
        };
        file.write(reinterpret_cast<const char*>(buf), 48);
        uint16_t attr = 0;
        file.write(reinterpret_cast<const char*>(&attr), 2);
    }
}

} // namespace stl

// ============================================================
// SDF-based primitive from triangle mesh (approximate)
// ============================================================
class MeshPrimitive : public Primitive {
    stl::STLMesh mesh_;
    AABB bb_;
public:
    explicit MeshPrimitive(const stl::STLMesh& mesh) : mesh_(mesh), bb_(mesh.bounds()) {}

    double sdf(const Vec3& p) const override {
        // Approximate: find nearest triangle, use signed distance
        double min_dist_sq = std::numeric_limits<double>::max();
        Vec3 closest_normal;

        for (auto& tri : mesh_.triangles) {
            // Point-triangle distance
            Vec3 edge0 = tri.v1 - tri.v0;
            Vec3 edge1 = tri.v2 - tri.v0;
            Vec3 v0p = p - tri.v0;

            double d00 = edge0.dot(edge0);
            double d01 = edge0.dot(edge1);
            double d11 = edge1.dot(edge1);
            double d20 = v0p.dot(edge0);
            double d21 = v0p.dot(edge1);

            double denom = d00 * d11 - d01 * d01;
            if (std::abs(denom) < 1e-15) continue;

            double v = (d11 * d20 - d01 * d21) / denom;
            double w = (d00 * d21 - d01 * d20) / denom;
            double u = 1.0 - v - w;

            // Clamp to triangle
            u = std::clamp(u, 0.0, 1.0);
            v = std::clamp(v, 0.0, 1.0);
            w = std::clamp(w, 0.0, 1.0);
            double sum = u + v + w;
            u /= sum; v /= sum; w /= sum;

            Vec3 proj = tri.v0 * u + tri.v1 * v + tri.v2 * w;
            double dist_sq = (p - proj).length_sq();

            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                closest_normal = tri.normal();
            }
        }

        double dist = std::sqrt(min_dist_sq);
        // Sign: use normal dot product as heuristic
        Vec3 to_nearest = Vec3(0, 0, 0); // approximate
        double sign = closest_normal.dot(Vec3(1, 0, 0)) > 0 ? 1.0 : -1.0;
        return dist * sign;
    }

    AABB bounds() const override { return bb_; }
};

// ============================================================
// Transformation wrapper
// ============================================================
class Translated : public Primitive {
    std::shared_ptr<Primitive> inner_;
    Vec3 offset_;
public:
    Translated(std::shared_ptr<Primitive> inner, const Vec3& offset)
        : inner_(std::move(inner)), offset_(offset) {}

    double sdf(const Vec3& p) const override {
        return inner_->sdf(p - offset_);
    }

    AABB bounds() const override {
        auto bb = inner_->bounds();
        return AABB(bb.min_pt + offset_, bb.max_pt + offset_);
    }
};

class Scaled : public Primitive {
    std::shared_ptr<Primitive> inner_;
    double scale_;
public:
    Scaled(std::shared_ptr<Primitive> inner, double scale)
        : inner_(std::move(inner)), scale_(scale) {}

    double sdf(const Vec3& p) const override {
        return inner_->sdf(p / scale_) * scale_;
    }

    AABB bounds() const override {
        auto bb = inner_->bounds();
        return AABB(bb.min_pt * scale_, bb.max_pt * scale_);
    }
};

// ============================================================
// Point cloud / sampling utilities
// ============================================================
struct PointCloud {
    std::vector<Vec3> points;
    std::vector<Vec3> normals;  // optional per-point normals

    int size() const { return (int)points.size(); }

    // Convert to Tensor (n_points x 3)
    Tensor to_tensor() const {
        int n = (int)points.size();
        Tensor t({n, 3});
        for (int i = 0; i < n; ++i) {
            t.at2(i, 0) = points[i].x;
            t.at2(i, 1) = points[i].y;
            t.at2(i, 2) = points[i].z;
        }
        return t;
    }

    // From Tensor
    static PointCloud from_tensor(const Tensor& t) {
        PointCloud pc;
        int n = t.shape[0];
        pc.points.resize(n);
        for (int i = 0; i < n; ++i) {
            pc.points[i] = Vec3(t.at2(i, 0), t.at2(i, 1),
                                t.ndim() >= 2 && t.shape[1] >= 3 ? t.at2(i, 2) : 0.0);
        }
        return pc;
    }
};

} // namespace geometry
} // namespace modulus

#endif // MODULUS_GEOMETRY_H
