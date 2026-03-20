// deepxde_geometry.h - Extended geometry types for DeepXDE C++ port
// Ports: Disk, Ellipse, Polygon, Triangle, Cuboid, Sphere, Hypercube, Hypersphere,
//        PointCloud, CSGUnion, CSGDifference, CSGIntersection
#ifndef DEEPXDE_GEOMETRY_H
#define DEEPXDE_GEOMETRY_H

#include "deepxde.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <set>

namespace deepxde {

// ============================================================================
// Hypercube (N-dimensional box)
// Ports deepxde.geometry.geometry_nd.Hypercube
// ============================================================================
class Hypercube : public Geometry {
public:
    std::vector<double> xmin, xmax, side_length;
    double volume;

    Hypercube(const std::vector<double>& lo, const std::vector<double>& hi)
        : Geometry(static_cast<int>(lo.size()), lo, hi, 0.0)
    {
        if (lo.size() != hi.size())
            throw std::invalid_argument("Hypercube: xmin/xmax dimension mismatch");
        xmin = lo;
        xmax = hi;
        side_length.resize(dim);
        double diag2 = 0.0;
        volume = 1.0;
        for (int i = 0; i < dim; ++i) {
            if (xmin[i] >= xmax[i])
                throw std::invalid_argument("Hypercube: xmin >= xmax");
            side_length[i] = xmax[i] - xmin[i];
            volume *= side_length[i];
            diag2 += side_length[i] * side_length[i];
        }
        diam = std::sqrt(diag2);
    }

    bool inside(const std::vector<double>& x) const override {
        for (int i = 0; i < dim; ++i)
            if (x[i] < xmin[i] - 1e-12 || x[i] > xmax[i] + 1e-12) return false;
        return true;
    }

    bool on_boundary(const std::vector<double>& x) const override {
        if (!inside(x)) return false;
        for (int i = 0; i < dim; ++i)
            if (std::fabs(x[i] - xmin[i]) < 1e-12 || std::fabs(x[i] - xmax[i]) < 1e-12)
                return true;
        return false;
    }

    std::vector<double> boundary_normal(const std::vector<double>& x) const override {
        std::vector<double> n(dim, 0.0);
        for (int i = 0; i < dim; ++i) {
            if (std::fabs(x[i] - xmin[i]) < 1e-12) n[i] = -1.0;
            else if (std::fabs(x[i] - xmax[i]) < 1e-12) n[i] = 1.0;
        }
        double len = 0;
        for (int i = 0; i < dim; ++i) len += n[i] * n[i];
        len = std::sqrt(len);
        if (len > 0) for (int i = 0; i < dim; ++i) n[i] /= len;
        return n;
    }

    Matrix random_points(int n) const override {
        Matrix m(n, dim);
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < dim; ++d) {
                std::uniform_real_distribution<double> dist(xmin[d], xmax[d]);
                m(i, d) = dist(global_rng());
            }
        return m;
    }

    Matrix uniform_points(int n, bool boundary = true) const override {
        double dx = std::pow(volume / n, 1.0 / dim);
        std::vector<int> ni(dim);
        int total = 1;
        for (int d = 0; d < dim; ++d) {
            ni[d] = std::max(1, static_cast<int>(std::ceil(side_length[d] / dx)));
            total *= ni[d];
        }
        Matrix m(total, dim);
        std::vector<int> idx(dim, 0);
        for (int pt = 0; pt < total; ++pt) {
            for (int d = 0; d < dim; ++d) {
                if (boundary)
                    m(pt, d) = xmin[d] + side_length[d] * idx[d] / std::max(ni[d] - 1, 1);
                else
                    m(pt, d) = xmin[d] + side_length[d] * (idx[d] + 1.0) / (ni[d] + 1.0);
            }
            for (int d = dim - 1; d >= 0; --d) {
                idx[d]++;
                if (idx[d] < ni[d]) break;
                idx[d] = 0;
            }
        }
        return m;
    }

    Matrix random_boundary_points(int n) const override {
        Matrix m(n, dim);
        for (int i = 0; i < n; ++i) {
            for (int d = 0; d < dim; ++d) {
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                m(i, d) = dist(global_rng());
            }
            int rand_dim = global_rng()() % dim;
            m(i, rand_dim) = (m(i, rand_dim) > 0.5) ? 1.0 : 0.0;
            for (int d = 0; d < dim; ++d)
                m(i, d) = xmin[d] + side_length[d] * m(i, d);
        }
        return m;
    }

    Matrix uniform_boundary_points(int n) const override {
        return random_boundary_points(n);
    }

    std::vector<double> periodic_point(const std::vector<double>& x, int component) const {
        std::vector<double> y = x;
        if (std::fabs(y[component] - xmin[component]) < 1e-12)
            y[component] = xmax[component];
        else if (std::fabs(y[component] - xmax[component]) < 1e-12)
            y[component] = xmin[component];
        return y;
    }
};

// ============================================================================
// Hypersphere (N-dimensional sphere)
// ============================================================================
class Hypersphere : public Geometry {
public:
    std::vector<double> center;
    double radius;

    Hypersphere(const std::vector<double>& c, double r)
        : Geometry(static_cast<int>(c.size()),
                   [&]() { auto lo = c; for (auto& v : lo) v -= r; return lo; }(),
                   [&]() { auto hi = c; for (auto& v : hi) v += r; return hi; }(),
                   2 * r),
          center(c), radius(r) {}

    bool inside(const std::vector<double>& x) const override {
        double d2 = 0;
        for (int i = 0; i < dim; ++i) d2 += (x[i] - center[i]) * (x[i] - center[i]);
        return std::sqrt(d2) <= radius + 1e-12;
    }

    bool on_boundary(const std::vector<double>& x) const override {
        double d = 0;
        for (int i = 0; i < dim; ++i) d += (x[i] - center[i]) * (x[i] - center[i]);
        return std::fabs(std::sqrt(d) - radius) < 1e-12;
    }

    std::vector<double> boundary_normal(const std::vector<double>& x) const override {
        std::vector<double> n(dim);
        double len = 0;
        for (int i = 0; i < dim; ++i) { n[i] = x[i] - center[i]; len += n[i] * n[i]; }
        len = std::sqrt(len);
        if (len > 0) for (int i = 0; i < dim; ++i) n[i] /= len;
        return n;
    }

    Matrix random_points(int n) const override {
        Matrix m(n, dim);
        int filled = 0;
        while (filled < n) {
            std::vector<double> pt(dim);
            for (int d = 0; d < dim; ++d) {
                std::uniform_real_distribution<double> dist(-radius, radius);
                pt[d] = dist(global_rng());
            }
            double r2 = 0;
            for (int d = 0; d < dim; ++d) r2 += pt[d] * pt[d];
            if (r2 <= radius * radius) {
                for (int d = 0; d < dim; ++d) m(filled, d) = center[d] + pt[d];
                filled++;
            }
        }
        return m;
    }

    Matrix uniform_points(int n, bool = true) const override { return random_points(n); }

    Matrix random_boundary_points(int n) const override {
        Matrix m(n, dim);
        std::normal_distribution<double> norm(0.0, 1.0);
        for (int i = 0; i < n; ++i) {
            double len = 0;
            for (int d = 0; d < dim; ++d) { m(i, d) = norm(global_rng()); len += m(i, d) * m(i, d); }
            len = std::sqrt(len);
            for (int d = 0; d < dim; ++d) m(i, d) = center[d] + radius * m(i, d) / len;
        }
        return m;
    }

    Matrix uniform_boundary_points(int n) const override { return random_boundary_points(n); }
};

// ============================================================================
// Disk (2D circle)
// ============================================================================
class Disk : public Hypersphere {
public:
    Disk(const std::vector<double>& c, double r) : Hypersphere(c, r) {
        if (c.size() != 2) throw std::invalid_argument("Disk center must be 2D");
    }

    Matrix random_points(int n) const override {
        Matrix m(n, 2);
        std::uniform_real_distribution<double> d01(0.0, 1.0);
        for (int i = 0; i < n; ++i) {
            double r = std::sqrt(d01(global_rng())) * radius;
            double th = 2.0 * M_PI * d01(global_rng());
            m(i, 0) = center[0] + r * std::cos(th);
            m(i, 1) = center[1] + r * std::sin(th);
        }
        return m;
    }

    Matrix uniform_boundary_points(int n) const override {
        Matrix m(n, 2);
        for (int i = 0; i < n; ++i) {
            double th = 2.0 * M_PI * i / n;
            m(i, 0) = center[0] + radius * std::cos(th);
            m(i, 1) = center[1] + radius * std::sin(th);
        }
        return m;
    }

    Matrix random_boundary_points(int n) const override {
        Matrix m(n, 2);
        std::uniform_real_distribution<double> d01(0.0, 1.0);
        for (int i = 0; i < n; ++i) {
            double th = 2.0 * M_PI * d01(global_rng());
            m(i, 0) = center[0] + radius * std::cos(th);
            m(i, 1) = center[1] + radius * std::sin(th);
        }
        return m;
    }
};

// ============================================================================
// Sphere (3D)
// ============================================================================
class Sphere : public Hypersphere {
public:
    Sphere(const std::vector<double>& c, double r) : Hypersphere(c, r) {
        if (c.size() != 3) throw std::invalid_argument("Sphere center must be 3D");
    }
};

// ============================================================================
// Cuboid (3D box)
// ============================================================================
class Cuboid : public Hypercube {
public:
    double surface_area;
    Cuboid(const std::vector<double>& lo, const std::vector<double>& hi) : Hypercube(lo, hi) {
        if (dim != 3) throw std::invalid_argument("Cuboid must be 3D");
        double dx = side_length[0], dy = side_length[1], dz = side_length[2];
        surface_area = 2.0 * (dx * dy + dy * dz + dz * dx);
    }
};

// ============================================================================
// Ellipse (2D)
// ============================================================================
class Ellipse : public Geometry {
public:
    std::vector<double> center;
    double semimajor, semiminor, angle;

    Ellipse(const std::vector<double>& ctr, double a, double b, double ang = 0.0)
        : Geometry(2, {ctr[0] - a, ctr[1] - b}, {ctr[0] + a, ctr[1] + b},
                   2.0 * std::sqrt(std::max(0.0, a * a - b * b))),
          center(ctr), semimajor(a), semiminor(b), angle(ang) {}

    bool inside(const std::vector<double>& x) const override {
        double ca = std::cos(-angle), sa = std::sin(-angle);
        double lx = (x[0] - center[0]) * ca - (x[1] - center[1]) * sa;
        double ly = (x[0] - center[0]) * sa + (x[1] - center[1]) * ca;
        return (lx * lx) / (semimajor * semimajor) + (ly * ly) / (semiminor * semiminor) <= 1.0 + 1e-12;
    }

    bool on_boundary(const std::vector<double>& x) const override {
        double ca = std::cos(-angle), sa = std::sin(-angle);
        double lx = (x[0] - center[0]) * ca - (x[1] - center[1]) * sa;
        double ly = (x[0] - center[0]) * sa + (x[1] - center[1]) * ca;
        return std::fabs((lx * lx) / (semimajor * semimajor) + (ly * ly) / (semiminor * semiminor) - 1.0) < 1e-10;
    }

    std::vector<double> boundary_normal(const std::vector<double>& x) const override {
        double ca = std::cos(-angle), sa = std::sin(-angle);
        double lx = (x[0] - center[0]) * ca - (x[1] - center[1]) * sa;
        double ly = (x[0] - center[0]) * sa + (x[1] - center[1]) * ca;
        double nx = 2.0 * lx / (semimajor * semimajor);
        double ny = 2.0 * ly / (semiminor * semiminor);
        double ca2 = std::cos(angle), sa2 = std::sin(angle);
        double gnx = nx * ca2 - ny * sa2;
        double gny = nx * sa2 + ny * ca2;
        double len = std::hypot(gnx, gny);
        if (len > 0) { gnx /= len; gny /= len; }
        return {gnx, gny};
    }

    Matrix random_points(int n) const override {
        Matrix m(n, 2);
        std::uniform_real_distribution<double> d01(0.0, 1.0);
        for (int i = 0; i < n; ++i) {
            double r = std::sqrt(d01(global_rng()));
            double th = 2.0 * M_PI * d01(global_rng());
            double x = semimajor * std::cos(th) * r;
            double y = semiminor * std::sin(th) * r;
            double ca2 = std::cos(angle), sa2 = std::sin(angle);
            m(i, 0) = center[0] + x * ca2 - y * sa2;
            m(i, 1) = center[1] + x * sa2 + y * ca2;
        }
        return m;
    }
    Matrix uniform_points(int n, bool = true) const override { return random_points(n); }

    Matrix random_boundary_points(int n) const override {
        Matrix m(n, 2);
        std::uniform_real_distribution<double> d01(0.0, 1.0);
        for (int i = 0; i < n; ++i) {
            double th = 2.0 * M_PI * d01(global_rng());
            double x = semimajor * std::cos(th);
            double y = semiminor * std::sin(th);
            double ca2 = std::cos(angle), sa2 = std::sin(angle);
            m(i, 0) = center[0] + x * ca2 - y * sa2;
            m(i, 1) = center[1] + x * sa2 + y * ca2;
        }
        return m;
    }
    Matrix uniform_boundary_points(int n) const override {
        Matrix m(n, 2);
        for (int i = 0; i < n; ++i) {
            double th = 2.0 * M_PI * i / n;
            double x = semimajor * std::cos(th);
            double y = semiminor * std::sin(th);
            double ca2 = std::cos(angle), sa2 = std::sin(angle);
            m(i, 0) = center[0] + x * ca2 - y * sa2;
            m(i, 1) = center[1] + x * sa2 + y * ca2;
        }
        return m;
    }
};

// ============================================================================
// Triangle (2D)
// ============================================================================
class Triangle : public Geometry {
public:
    std::vector<double> v0, v1, v2;

    Triangle(const std::vector<double>& a, const std::vector<double>& b,
             const std::vector<double>& c)
        : Geometry(2,
                   {std::min({a[0], b[0], c[0]}), std::min({a[1], b[1], c[1]})},
                   {std::max({a[0], b[0], c[0]}), std::max({a[1], b[1], c[1]})},
                   std::max({std::hypot(b[0]-a[0],b[1]-a[1]),
                             std::hypot(c[0]-b[0],c[1]-b[1]),
                             std::hypot(a[0]-c[0],a[1]-c[1])})),
          v0(a), v1(b), v2(c) {}

    bool inside(const std::vector<double>& x) const override {
        double d1 = xsign(x, v0, v1), d2 = xsign(x, v1, v2), d3 = xsign(x, v2, v0);
        bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
        bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
        return !(has_neg && has_pos);
    }

    bool on_boundary(const std::vector<double>& x) const override {
        return pt_on_seg(x, v0, v1) || pt_on_seg(x, v1, v2) || pt_on_seg(x, v2, v0);
    }

    std::vector<double> boundary_normal(const std::vector<double>& x) const override {
        struct E { std::vector<double> a, b; };
        E edges[3] = {{v0, v1}, {v1, v2}, {v2, v0}};
        double best = 1e30; int bi = 0;
        for (int e = 0; e < 3; ++e) {
            double d = dseg(x, edges[e].a, edges[e].b);
            if (d < best) { best = d; bi = e; }
        }
        double ex = edges[bi].b[0] - edges[bi].a[0], ey = edges[bi].b[1] - edges[bi].a[1];
        double nx = ey, ny = -ex;
        double len = std::hypot(nx, ny);
        if (len > 0) { nx /= len; ny /= len; }
        double cx = (v0[0]+v1[0]+v2[0])/3, cy = (v0[1]+v1[1]+v2[1])/3;
        double mx = (edges[bi].a[0]+edges[bi].b[0])/2, my = (edges[bi].a[1]+edges[bi].b[1])/2;
        if ((mx-cx)*nx + (my-cy)*ny < 0) { nx = -nx; ny = -ny; }
        return {nx, ny};
    }

    Matrix random_points(int n) const override {
        Matrix m(n, 2);
        std::uniform_real_distribution<double> d01(0.0, 1.0);
        for (int i = 0; i < n; ++i) {
            double r1 = std::sqrt(d01(global_rng())), r2 = d01(global_rng());
            double u = 1-r1, v = r1*(1-r2), w = r1*r2;
            m(i, 0) = u*v0[0]+v*v1[0]+w*v2[0];
            m(i, 1) = u*v0[1]+v*v1[1]+w*v2[1];
        }
        return m;
    }
    Matrix uniform_points(int n, bool = true) const override { return random_points(n); }

    Matrix random_boundary_points(int n) const override {
        double l0 = std::hypot(v1[0]-v0[0],v1[1]-v0[1]);
        double l1 = std::hypot(v2[0]-v1[0],v2[1]-v1[1]);
        double l2 = std::hypot(v0[0]-v2[0],v0[1]-v2[1]);
        double perim = l0+l1+l2;
        Matrix m(n, 2);
        std::uniform_real_distribution<double> dist(0.0, perim);
        for (int i = 0; i < n; ++i) {
            double u = dist(global_rng());
            if (u < l0) { double t=u/l0; m(i,0)=v0[0]+t*(v1[0]-v0[0]); m(i,1)=v0[1]+t*(v1[1]-v0[1]); }
            else if (u < l0+l1) { double t=(u-l0)/l1; m(i,0)=v1[0]+t*(v2[0]-v1[0]); m(i,1)=v1[1]+t*(v2[1]-v1[1]); }
            else { double t=(u-l0-l1)/l2; m(i,0)=v2[0]+t*(v0[0]-v2[0]); m(i,1)=v2[1]+t*(v0[1]-v2[1]); }
        }
        return m;
    }
    Matrix uniform_boundary_points(int n) const override { return random_boundary_points(n); }

private:
    static double xsign(const std::vector<double>& p, const std::vector<double>& a, const std::vector<double>& b) {
        return (p[0]-b[0])*(a[1]-b[1]) - (a[0]-b[0])*(p[1]-b[1]);
    }
    static bool pt_on_seg(const std::vector<double>& p, const std::vector<double>& a, const std::vector<double>& b) {
        double cross = (p[0]-a[0])*(b[1]-a[1]) - (p[1]-a[1])*(b[0]-a[0]);
        if (std::fabs(cross) > 1e-10) return false;
        double dot = (p[0]-a[0])*(b[0]-a[0]) + (p[1]-a[1])*(b[1]-a[1]);
        double l2 = (b[0]-a[0])*(b[0]-a[0]) + (b[1]-a[1])*(b[1]-a[1]);
        return dot >= -1e-12 && dot <= l2+1e-12;
    }
    static double dseg(const std::vector<double>& p, const std::vector<double>& a, const std::vector<double>& b) {
        double dx=b[0]-a[0], dy=b[1]-a[1], l2=dx*dx+dy*dy;
        if (l2 < 1e-30) return std::hypot(p[0]-a[0], p[1]-a[1]);
        double t = std::clamp(((p[0]-a[0])*dx+(p[1]-a[1])*dy)/l2, 0.0, 1.0);
        return std::hypot(p[0]-(a[0]+t*dx), p[1]-(a[1]+t*dy));
    }
};

// ============================================================================
// Polygon (2D)
// ============================================================================
class Polygon : public Geometry {
public:
    std::vector<std::vector<double>> vertices;
    int n_vertices;
    double perimeter;

    Polygon(const std::vector<std::vector<double>>& verts)
        : Geometry(2,
                   [&]() { double x=1e30,y=1e30; for(auto&v:verts){x=std::min(x,v[0]);y=std::min(y,v[1]);} return std::vector<double>{x,y}; }(),
                   [&]() { double x=-1e30,y=-1e30; for(auto&v:verts){x=std::max(x,v[0]);y=std::max(y,v[1]);} return std::vector<double>{x,y}; }(),
                   [&]() { double d=0; for(size_t i=0;i<verts.size();++i) for(size_t j=i+1;j<verts.size();++j) d=std::max(d,std::hypot(verts[i][0]-verts[j][0],verts[i][1]-verts[j][1])); return d; }()),
          vertices(verts), n_vertices(static_cast<int>(verts.size()))
    {
        perimeter = 0;
        for (int i = 0; i < n_vertices; ++i) {
            int j = (i+1) % n_vertices;
            perimeter += std::hypot(vertices[j][0]-vertices[i][0], vertices[j][1]-vertices[i][1]);
        }
    }

    bool inside(const std::vector<double>& x) const override {
        int crossings = 0;
        for (int i = 0; i < n_vertices; ++i) {
            int j = (i+1) % n_vertices;
            double yi = vertices[i][1], yj = vertices[j][1];
            if ((yi <= x[1] && yj > x[1]) || (yj <= x[1] && yi > x[1])) {
                double t = (x[1]-yi)/(yj-yi);
                if (x[0] < vertices[i][0] + t*(vertices[j][0]-vertices[i][0])) crossings++;
            }
        }
        return (crossings % 2 == 1) || on_boundary(x);
    }

    bool on_boundary(const std::vector<double>& x) const override {
        for (int i = 0; i < n_vertices; ++i) {
            int j = (i+1) % n_vertices;
            double cross = (x[0]-vertices[i][0])*(vertices[j][1]-vertices[i][1])
                         - (x[1]-vertices[i][1])*(vertices[j][0]-vertices[i][0]);
            if (std::fabs(cross) > 1e-10) continue;
            double dot = (x[0]-vertices[i][0])*(vertices[j][0]-vertices[i][0])
                       + (x[1]-vertices[i][1])*(vertices[j][1]-vertices[i][1]);
            double l2 = std::pow(vertices[j][0]-vertices[i][0],2) + std::pow(vertices[j][1]-vertices[i][1],2);
            if (dot >= -1e-12 && dot <= l2+1e-12) return true;
        }
        return false;
    }

    std::vector<double> boundary_normal(const std::vector<double>& x) const override {
        double best = 1e30; int bi = 0;
        for (int i = 0; i < n_vertices; ++i) {
            int j = (i+1)%n_vertices;
            double d = dseg(x, vertices[i], vertices[j]);
            if (d < best) { best=d; bi=i; }
        }
        int j = (bi+1) % n_vertices;
        double ex=vertices[j][0]-vertices[bi][0], ey=vertices[j][1]-vertices[bi][1];
        double nx=ey, ny=-ex, len=std::hypot(nx,ny);
        if (len>0) { nx/=len; ny/=len; }
        double cx=0,cy=0; for(auto&v:vertices){cx+=v[0];cy+=v[1];} cx/=n_vertices;cy/=n_vertices;
        double mx=(vertices[bi][0]+vertices[j][0])/2, my=(vertices[bi][1]+vertices[j][1])/2;
        if ((mx-cx)*nx+(my-cy)*ny < 0) {nx=-nx;ny=-ny;}
        return {nx, ny};
    }

    Matrix random_points(int n) const override {
        Matrix m(n, 2); int filled=0;
        while (filled < n) {
            std::uniform_real_distribution<double> dx(bbox_lo[0],bbox_hi[0]), dy(bbox_lo[1],bbox_hi[1]);
            std::vector<double> pt = {dx(global_rng()), dy(global_rng())};
            if (inside(pt)) { m(filled,0)=pt[0]; m(filled,1)=pt[1]; filled++; }
        }
        return m;
    }
    Matrix uniform_points(int n, bool = true) const override { return random_points(n); }

    Matrix random_boundary_points(int n) const override {
        Matrix m(n, 2);
        std::uniform_real_distribution<double> dist(0.0, perimeter);
        for (int i = 0; i < n; ++i) {
            double u = dist(global_rng()), cum = 0;
            for (int e = 0; e < n_vertices; ++e) {
                int ne = (e+1) % n_vertices;
                double elen = std::hypot(vertices[ne][0]-vertices[e][0], vertices[ne][1]-vertices[e][1]);
                if (cum+elen >= u) {
                    double t = (u-cum)/elen;
                    m(i,0)=vertices[e][0]+t*(vertices[ne][0]-vertices[e][0]);
                    m(i,1)=vertices[e][1]+t*(vertices[ne][1]-vertices[e][1]);
                    break;
                }
                cum += elen;
            }
        }
        return m;
    }
    Matrix uniform_boundary_points(int n) const override { return random_boundary_points(n); }

private:
    static double dseg(const std::vector<double>& p, const std::vector<double>& a, const std::vector<double>& b) {
        double dx=b[0]-a[0],dy=b[1]-a[1],l2=dx*dx+dy*dy;
        if (l2<1e-30) return std::hypot(p[0]-a[0],p[1]-a[1]);
        double t=std::clamp(((p[0]-a[0])*dx+(p[1]-a[1])*dy)/l2,0.0,1.0);
        return std::hypot(p[0]-(a[0]+t*dx),p[1]-(a[1]+t*dy));
    }
};

// ============================================================================
// PointCloud
// ============================================================================
class PointCloud : public Geometry {
public:
    Matrix points;
    Matrix bdy_points;
    Matrix bdy_normals;
    bool has_boundary, has_normals;

    PointCloud(const Matrix& pts, const Matrix& bp = Matrix(), const Matrix& bn = Matrix())
        : Geometry(pts.cols,
                   [&]() { std::vector<double> lo(pts.cols,1e30); for(int i=0;i<pts.rows;++i) for(int j=0;j<pts.cols;++j) lo[j]=std::min(lo[j],pts(i,j)); return lo; }(),
                   [&]() { std::vector<double> hi(pts.cols,-1e30); for(int i=0;i<pts.rows;++i) for(int j=0;j<pts.cols;++j) hi[j]=std::max(hi[j],pts(i,j)); return hi; }(),
                   1e30),
          points(pts), bdy_points(bp), bdy_normals(bn),
          has_boundary(bp.rows > 0), has_normals(bn.rows > 0) {}

    bool inside(const std::vector<double>& x) const override {
        for (int i = 0; i < points.rows; ++i) {
            bool m = true;
            for (int d = 0; d < dim; ++d) if (std::fabs(x[d]-points(i,d))>1e-10) { m=false; break; }
            if (m) return true;
        }
        return false;
    }
    bool on_boundary(const std::vector<double>& x) const override {
        if (!has_boundary) return false;
        for (int i = 0; i < bdy_points.rows; ++i) {
            bool m = true;
            for (int d = 0; d < dim; ++d) if (std::fabs(x[d]-bdy_points(i,d))>1e-10) { m=false; break; }
            if (m) return true;
        }
        return false;
    }
    std::vector<double> boundary_normal(const std::vector<double>& x) const override {
        if (!has_normals) throw std::runtime_error("PointCloud: no normals");
        double best=1e30; int idx=0;
        for (int i=0; i<bdy_points.rows; ++i) {
            double d2=0; for(int d=0;d<dim;++d) d2+=(x[d]-bdy_points(i,d))*(x[d]-bdy_points(i,d));
            if (d2<best) { best=d2; idx=i; }
        }
        return bdy_normals.row(idx);
    }
    Matrix random_points(int n) const override {
        Matrix m(n,dim);
        for(int i=0;i<n;++i) { int idx=global_rng()()%points.rows; m.set_row(i,points.row(idx)); }
        return m;
    }
    Matrix uniform_points(int n, bool = true) const override { return random_points(n); }
    Matrix random_boundary_points(int n) const override {
        if (!has_boundary) throw std::runtime_error("PointCloud: no boundary");
        Matrix m(n,dim);
        for(int i=0;i<n;++i) { int idx=global_rng()()%bdy_points.rows; m.set_row(i,bdy_points.row(idx)); }
        return m;
    }
    Matrix uniform_boundary_points(int n) const override { return random_boundary_points(n); }
};

// ============================================================================
// CSG Operations
// ============================================================================
class CSGUnion : public Geometry {
public:
    std::shared_ptr<Geometry> geom1, geom2;
    CSGUnion(std::shared_ptr<Geometry> g1, std::shared_ptr<Geometry> g2)
        : Geometry(g1->dim,
                   [&](){std::vector<double> lo(g1->dim); for(int i=0;i<g1->dim;++i) lo[i]=std::min(g1->bbox_lo[i],g2->bbox_lo[i]); return lo;}(),
                   [&](){std::vector<double> hi(g1->dim); for(int i=0;i<g1->dim;++i) hi[i]=std::max(g1->bbox_hi[i],g2->bbox_hi[i]); return hi;}(),
                   g1->diam+g2->diam), geom1(g1), geom2(g2) {}
    bool inside(const std::vector<double>& x) const override { return geom1->inside(x)||geom2->inside(x); }
    bool on_boundary(const std::vector<double>& x) const override {
        return (geom1->on_boundary(x)&&!geom2->inside(x))||(geom2->on_boundary(x)&&!geom1->inside(x));
    }
    std::vector<double> boundary_normal(const std::vector<double>& x) const override {
        if (geom1->on_boundary(x)&&!geom2->inside(x)) return geom1->boundary_normal(x);
        return geom2->boundary_normal(x);
    }
    Matrix random_points(int n) const override {
        Matrix m(n,dim); int f=0;
        while(f<n) { std::vector<double> pt(dim); for(int d=0;d<dim;++d){std::uniform_real_distribution<double> dist(bbox_lo[d],bbox_hi[d]); pt[d]=dist(global_rng());} if(inside(pt)){m.set_row(f++,pt);} }
        return m;
    }
    Matrix uniform_points(int n, bool = true) const override { return random_points(n); }
    Matrix random_boundary_points(int n) const override {
        Matrix m(n,dim); int f=0;
        while(f<n) {
            auto p1=geom1->random_boundary_points(n); for(int i=0;i<p1.rows&&f<n;++i){auto r=p1.row(i); if(!geom2->inside(r)) m.set_row(f++,r);}
            auto p2=geom2->random_boundary_points(n); for(int i=0;i<p2.rows&&f<n;++i){auto r=p2.row(i); if(!geom1->inside(r)) m.set_row(f++,r);}
        }
        return m;
    }
    Matrix uniform_boundary_points(int n) const override { return random_boundary_points(n); }
};

class CSGDifference : public Geometry {
public:
    std::shared_ptr<Geometry> geom1, geom2;
    CSGDifference(std::shared_ptr<Geometry> g1, std::shared_ptr<Geometry> g2)
        : Geometry(g1->dim, g1->bbox_lo, g1->bbox_hi, g1->diam), geom1(g1), geom2(g2) {}
    bool inside(const std::vector<double>& x) const override { return geom1->inside(x)&&!geom2->inside(x); }
    bool on_boundary(const std::vector<double>& x) const override {
        return (geom1->on_boundary(x)&&!geom2->inside(x))||(geom1->inside(x)&&geom2->on_boundary(x));
    }
    std::vector<double> boundary_normal(const std::vector<double>& x) const override {
        if (geom1->on_boundary(x)&&!geom2->inside(x)) return geom1->boundary_normal(x);
        auto n=geom2->boundary_normal(x); for(auto&v:n) v=-v; return n;
    }
    Matrix random_points(int n) const override {
        Matrix m(n,dim); int f=0;
        while(f<n) { auto pts=geom1->random_points(n); for(int i=0;i<pts.rows&&f<n;++i){auto r=pts.row(i); if(!geom2->inside(r)) m.set_row(f++,r);} }
        return m;
    }
    Matrix uniform_points(int n, bool = true) const override { return random_points(n); }
    Matrix random_boundary_points(int n) const override {
        Matrix m(n,dim); int f=0;
        while(f<n) {
            auto p1=geom1->random_boundary_points(n); for(int i=0;i<p1.rows&&f<n;++i){auto r=p1.row(i); if(!geom2->inside(r)) m.set_row(f++,r);}
            auto p2=geom2->random_boundary_points(n); for(int i=0;i<p2.rows&&f<n;++i){auto r=p2.row(i); if(geom1->inside(r)) m.set_row(f++,r);}
        }
        return m;
    }
    Matrix uniform_boundary_points(int n) const override { return random_boundary_points(n); }
};

class CSGIntersection : public Geometry {
public:
    std::shared_ptr<Geometry> geom1, geom2;
    CSGIntersection(std::shared_ptr<Geometry> g1, std::shared_ptr<Geometry> g2)
        : Geometry(g1->dim,
                   [&](){std::vector<double> lo(g1->dim); for(int i=0;i<g1->dim;++i) lo[i]=std::max(g1->bbox_lo[i],g2->bbox_lo[i]); return lo;}(),
                   [&](){std::vector<double> hi(g1->dim); for(int i=0;i<g1->dim;++i) hi[i]=std::min(g1->bbox_hi[i],g2->bbox_hi[i]); return hi;}(),
                   std::min(g1->diam,g2->diam)), geom1(g1), geom2(g2) {}
    bool inside(const std::vector<double>& x) const override { return geom1->inside(x)&&geom2->inside(x); }
    bool on_boundary(const std::vector<double>& x) const override {
        return (geom1->on_boundary(x)&&geom2->inside(x))||(geom1->inside(x)&&geom2->on_boundary(x));
    }
    std::vector<double> boundary_normal(const std::vector<double>& x) const override {
        if (geom1->on_boundary(x)&&geom2->inside(x)) return geom1->boundary_normal(x);
        return geom2->boundary_normal(x);
    }
    Matrix random_points(int n) const override {
        Matrix m(n,dim); int f=0;
        while(f<n) { auto pts=geom1->random_points(n); for(int i=0;i<pts.rows&&f<n;++i){auto r=pts.row(i); if(geom2->inside(r)) m.set_row(f++,r);} }
        return m;
    }
    Matrix uniform_points(int n, bool = true) const override { return random_points(n); }
    Matrix random_boundary_points(int n) const override {
        Matrix m(n,dim); int f=0;
        while(f<n) {
            auto p1=geom1->random_boundary_points(n); for(int i=0;i<p1.rows&&f<n;++i){auto r=p1.row(i); if(geom2->inside(r)) m.set_row(f++,r);}
            auto p2=geom2->random_boundary_points(n); for(int i=0;i<p2.rows&&f<n;++i){auto r=p2.row(i); if(geom1->inside(r)) m.set_row(f++,r);}
        }
        return m;
    }
    Matrix uniform_boundary_points(int n) const override { return random_boundary_points(n); }
};

} // namespace deepxde
#endif // DEEPXDE_GEOMETRY_H
