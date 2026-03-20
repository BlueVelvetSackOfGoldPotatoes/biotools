// BETSE C++ Port - Utility Functions
// Library utilities, YAML config parsing helpers, math helpers,
// geometry utilities, and file I/O.
// Ported from: betse/util/*, betse/lib/*, betse/science/sim_toolbox.py,
//              betse/science/math/mathunit.py, betse/science/cells.py,
//              betse/util/math/geometry/*
#pragma once

#include "betse_types.h"
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <functional>

namespace betse {

// ============================================================================
// Unit Conversion Utilities (from math/mathunit.py)
// ============================================================================
namespace units {

constexpr double MILLI  = 1.0e-3;
constexpr double MICRO  = 1.0e-6;
constexpr double NANO   = 1.0e-9;
constexpr double PICO   = 1.0e-12;
constexpr double FEMTO  = 1.0e-15;
constexpr double DEBYE  = 3.336e-30;  // 1 Debye in C*m

inline double mV_to_V(double mV) { return mV * 1e-3; }
inline double V_to_mV(double V) { return V * 1e3; }
inline double um_to_m(double um) { return um * 1e-6; }
inline double m_to_um(double m) { return m * 1e6; }
inline double mM_to_M(double mM) { return mM * 1e-3; }
inline double M_to_mM(double M) { return M * 1e3; }
inline double nm_to_m(double nm) { return nm * 1e-9; }

// pH <-> H+ concentration [mM]
inline double pH_to_H_mM(double pH) { return std::pow(10.0, -pH) * 1e3; }
inline double H_mM_to_pH(double H_mM) {
    return (H_mM > 0) ? -std::log10(H_mM * 1e-3) : 7.0;
}

// Temperature
inline double celsius_to_kelvin(double C) { return C + 273.15; }
inline double kelvin_to_celsius(double K) { return K - 273.15; }

} // namespace units

// ============================================================================
// Geometry Utilities (from util/math/geometry/)
// ============================================================================
namespace geometry {

// Point-in-polygon test (ray casting algorithm)
inline bool point_in_polygon(Vec2 p, const std::vector<Vec2>& polygon) {
    int n = (int)polygon.size();
    if (n < 3) return false;
    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        if (((polygon[i].y > p.y) != (polygon[j].y > p.y)) &&
            (p.x < (polygon[j].x - polygon[i].x) *
             (p.y - polygon[i].y) / (polygon[j].y - polygon[i].y) +
             polygon[i].x)) {
            inside = !inside;
        }
    }
    return inside;
}

// Polygon area (shoelace formula)
inline double polygon_area(const std::vector<Vec2>& polygon) {
    int n = (int)polygon.size();
    double area = 0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        area += polygon[i].x * polygon[j].y;
        area -= polygon[j].x * polygon[i].y;
    }
    return std::abs(area) / 2.0;
}

// Polygon centroid
inline Vec2 polygon_centroid(const std::vector<Vec2>& polygon) {
    int n = (int)polygon.size();
    if (n == 0) return {0, 0};
    double A = polygon_area(polygon);
    if (A < 1e-30) {
        Vec2 c{0, 0};
        for (auto& p : polygon) { c.x += p.x; c.y += p.y; }
        return c / (double)n;
    }
    double cx = 0, cy = 0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        double f = polygon[i].x * polygon[j].y - polygon[j].x * polygon[i].y;
        cx += (polygon[i].x + polygon[j].x) * f;
        cy += (polygon[i].y + polygon[j].y) * f;
    }
    return {cx / (6.0 * A), cy / (6.0 * A)};
}

// Convex hull (Graham scan)
inline std::vector<Vec2> convex_hull(std::vector<Vec2> points) {
    int n = (int)points.size();
    if (n < 3) return points;

    // Find bottom-most (then left-most) point
    int bot = 0;
    for (int i = 1; i < n; i++) {
        if (points[i].y < points[bot].y ||
            (points[i].y == points[bot].y && points[i].x < points[bot].x))
            bot = i;
    }
    std::swap(points[0], points[bot]);
    Vec2 pivot = points[0];

    // Sort by polar angle
    std::sort(points.begin() + 1, points.end(),
              [&](const Vec2& a, const Vec2& b) {
        double cross = (a - pivot).cross(b - pivot);
        if (std::abs(cross) < 1e-30)
            return (a - pivot).norm2() < (b - pivot).norm2();
        return cross > 0;
    });

    std::vector<Vec2> hull;
    for (auto& p : points) {
        while (hull.size() > 1) {
            Vec2 a = hull[hull.size() - 2];
            Vec2 b = hull[hull.size() - 1];
            if ((b - a).cross(p - a) <= 0)
                hull.pop_back();
            else break;
        }
        hull.push_back(p);
    }
    return hull;
}

// Distance from point to line segment
inline double point_to_segment_dist(Vec2 p, Vec2 a, Vec2 b) {
    Vec2 ab = b - a;
    double t = ab.dot(p - a) / (ab.norm2() + 1e-30);
    t = std::clamp(t, 0.0, 1.0);
    Vec2 proj = a + ab * t;
    return (p - proj).norm();
}

// Circle-circle intersection test
inline bool circles_intersect(Vec2 c1, double r1, Vec2 c2, double r2) {
    return (c1 - c2).norm() < (r1 + r2);
}

} // namespace geometry

// ============================================================================
// Interpolation Utilities (from util/math/mathinterp.py)
// ============================================================================
namespace interp {

// Linear interpolation
inline double lerp(double a, double b, double t) {
    return a + (b - a) * t;
}

// Bilinear interpolation on a grid
inline double bilinear(const std::vector<double>& f, int nx, int ny,
                        double x, double y,
                        double xmin, double xmax,
                        double ymin, double ymax) {
    double fx = (x - xmin) / (xmax - xmin) * (nx - 1);
    double fy = (y - ymin) / (ymax - ymin) * (ny - 1);
    int ix = std::clamp((int)fx, 0, nx - 2);
    int iy = std::clamp((int)fy, 0, ny - 2);
    double sx = fx - ix;
    double sy = fy - iy;
    double f00 = f[iy * nx + ix];
    double f10 = f[iy * nx + ix + 1];
    double f01 = f[(iy + 1) * nx + ix];
    double f11 = f[(iy + 1) * nx + ix + 1];
    return lerp(lerp(f00, f10, sx), lerp(f01, f11, sx), sy);
}

// Nearest-neighbor interpolation from cell centres to a point
inline double nearest_cell(Vec2 point, const std::vector<Vec2>& centers,
                            const std::vector<double>& values) {
    int best = 0;
    double best_dist = 1e30;
    for (size_t i = 0; i < centers.size(); i++) {
        double d = (point - centers[i]).norm2();
        if (d < best_dist) { best_dist = d; best = (int)i; }
    }
    return (best < (int)values.size()) ? values[best] : 0.0;
}

// Inverse distance weighting interpolation
inline double idw(Vec2 point, const std::vector<Vec2>& centers,
                   const std::vector<double>& values, double power = 2.0) {
    double sum_w = 0, sum_wv = 0;
    for (size_t i = 0; i < centers.size(); i++) {
        double d = (point - centers[i]).norm();
        if (d < 1e-30) return values[i];
        double w = 1.0 / std::pow(d, power);
        sum_w += w;
        sum_wv += w * values[i];
    }
    return (sum_w > 0) ? sum_wv / sum_w : 0.0;
}

} // namespace interp

// ============================================================================
// Signal Processing Utilities (from sim_toolbox.py)
// ============================================================================
namespace signal {

// Pulse function: smooth rectangular pulse with tanh ramps
inline double pulse(double t, double t_start, double t_stop, double rate) {
    double ramp_up = 0.5 * (1.0 + std::tanh(rate * (t - t_start)));
    double ramp_down = 0.5 * (1.0 + std::tanh(rate * (t_stop - t)));
    return ramp_up * ramp_down;
}

// Step function: smooth Heaviside step
inline double step(double t, double t_on, double rate = 100.0) {
    return 0.5 * (1.0 + std::tanh(rate * (t - t_on)));
}

// Ramp function
inline double ramp(double t, double t_start, double t_end) {
    if (t <= t_start) return 0.0;
    if (t >= t_end) return 1.0;
    return (t - t_start) / (t_end - t_start);
}

// Periodic sin^2 function
inline double periodic_sin2(double t, double freq, double phase = 0.0) {
    double s = std::sin(M_PI * freq * t + phase);
    return s * s;
}

// Chirp (frequency sweep)
inline double chirp(double t, double f0, double f_rate) {
    double s = std::sin(M_PI * (f_rate * t + f0) * t);
    return s * s;
}

// Gaussian pulse
inline double gaussian_pulse(double t, double center, double sigma) {
    return std::exp(-0.5 * std::pow((t - center) / sigma, 2));
}

} // namespace signal

// ============================================================================
// Voronoi Cell Generation (from cells.py)
// Generates cell cluster with Voronoi tessellation
// ============================================================================
struct VoronoiGenerator {

    // Generate random cell centres on a hexagonal or square lattice
    // with disorder applied
    static std::vector<Vec2> generate_centres(
        int num_cells, double tissue_radius,
        CellLatticeType lattice, double disorder,
        std::mt19937& rng)
    {
        std::vector<Vec2> centres;
        double spacing = tissue_radius * 2.0 / std::sqrt((double)num_cells);

        if (lattice == CellLatticeType::HEX) {
            // Hexagonal lattice
            int nx = (int)(2.0 * tissue_radius / spacing) + 1;
            int ny = (int)(2.0 * tissue_radius / (spacing * std::sqrt(3.0) / 2.0)) + 1;
            for (int j = 0; j < ny; j++) {
                double y = -tissue_radius + j * spacing * std::sqrt(3.0) / 2.0;
                double x_offset = (j % 2 == 0) ? 0.0 : spacing / 2.0;
                for (int i = 0; i < nx; i++) {
                    double x = -tissue_radius + i * spacing + x_offset;
                    Vec2 p{x, y};
                    if (p.norm() <= tissue_radius)
                        centres.push_back(p);
                }
            }
        } else {
            // Square lattice
            int n_side = (int)(2.0 * tissue_radius / spacing) + 1;
            for (int j = 0; j < n_side; j++) {
                double y = -tissue_radius + j * spacing;
                for (int i = 0; i < n_side; i++) {
                    double x = -tissue_radius + i * spacing;
                    Vec2 p{x, y};
                    if (p.norm() <= tissue_radius)
                        centres.push_back(p);
                }
            }
        }

        // Apply disorder
        if (disorder > 0) {
            std::normal_distribution<double> noise(0.0, disorder * spacing);
            for (auto& p : centres) {
                p.x += noise(rng);
                p.y += noise(rng);
            }
            // Remove any that escaped tissue boundary
            centres.erase(
                std::remove_if(centres.begin(), centres.end(),
                    [tissue_radius](const Vec2& p) {
                        return p.norm() > tissue_radius;
                    }),
                centres.end());
        }

        // Truncate or pad to desired count
        if ((int)centres.size() > num_cells) {
            std::shuffle(centres.begin(), centres.end(), rng);
            centres.resize(num_cells);
        }

        return centres;
    }

    // Lloyd's relaxation for more uniform cell distribution
    static void lloyd_relax(std::vector<Vec2>& centres, double tissue_radius,
                             int iterations = 5) {
        for (int iter = 0; iter < iterations; iter++) {
            int n = (int)centres.size();
            std::vector<Vec2> new_centres(n, {0, 0});
            std::vector<int> counts(n, 0);

            // Sample grid points and assign to nearest centre
            int grid_n = 100;
            double step = 2.0 * tissue_radius / grid_n;
            for (int j = 0; j < grid_n; j++) {
                double y = -tissue_radius + (j + 0.5) * step;
                for (int i = 0; i < grid_n; i++) {
                    double x = -tissue_radius + (i + 0.5) * step;
                    Vec2 p{x, y};
                    if (p.norm() > tissue_radius) continue;

                    // Find nearest centre
                    int nearest = 0;
                    double best = 1e30;
                    for (int c = 0; c < n; c++) {
                        double d = (p - centres[c]).norm2();
                        if (d < best) { best = d; nearest = c; }
                    }
                    new_centres[nearest] += p;
                    counts[nearest]++;
                }
            }

            for (int c = 0; c < n; c++) {
                if (counts[c] > 0) {
                    centres[c] = new_centres[c] / (double)counts[c];
                    // Keep inside boundary
                    if (centres[c].norm() > tissue_radius * 0.95) {
                        centres[c] = centres[c].normalized() * tissue_radius * 0.95;
                    }
                }
            }
        }
    }
};

// ============================================================================
// File I/O Utilities
// ============================================================================
namespace fileio {

// Read CSV file into vectors
inline std::map<std::string, std::vector<double>> read_csv(
    const std::string& filename)
{
    std::map<std::string, std::vector<double>> data;
    std::ifstream file(filename);
    if (!file) return data;

    std::string header_line;
    std::getline(file, header_line);

    // Parse headers
    std::vector<std::string> headers;
    std::istringstream hss(header_line);
    std::string h;
    while (std::getline(hss, h, ',')) {
        // Trim whitespace
        h.erase(0, h.find_first_not_of(" \t\r\n"));
        h.erase(h.find_last_not_of(" \t\r\n") + 1);
        headers.push_back(h);
        data[h] = {};
    }

    // Parse data rows
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream lss(line);
        std::string val;
        int col = 0;
        while (std::getline(lss, val, ',') && col < (int)headers.size()) {
            try {
                data[headers[col]].push_back(std::stod(val));
            } catch (...) {
                data[headers[col]].push_back(0.0);
            }
            col++;
        }
    }
    return data;
}

// Write a 2D array to CSV
inline void write_array_csv(const std::string& filename,
                              const std::vector<std::vector<double>>& data,
                              const std::vector<std::string>& headers) {
    std::ofstream f(filename);
    if (!f) return;
    for (size_t i = 0; i < headers.size(); i++) {
        if (i > 0) f << ",";
        f << headers[i];
    }
    f << "\n";

    if (data.empty()) return;
    int nrows = (int)data[0].size();
    for (int r = 0; r < nrows; r++) {
        for (size_t c = 0; c < data.size(); c++) {
            if (c > 0) f << ",";
            f << (r < (int)data[c].size() ? data[c][r] : 0.0);
        }
        f << "\n";
    }
}

} // namespace fileio

// ============================================================================
// Statistics Utilities
// ============================================================================
namespace stats {

inline double min(const std::vector<double>& v) {
    return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end());
}

inline double max(const std::vector<double>& v) {
    return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end());
}

inline double variance(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    double m = vecutil::mean(v);
    double var = 0;
    for (double x : v) var += (x - m) * (x - m);
    return var / (v.size() - 1);
}

inline double stddev(const std::vector<double>& v) {
    return std::sqrt(variance(v));
}

inline double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    int n = (int)v.size();
    return (n % 2 == 0) ? (v[n/2 - 1] + v[n/2]) / 2.0 : v[n/2];
}

// Percentile (0-100)
inline double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    int lo = (int)idx;
    int hi = std::min(lo + 1, (int)v.size() - 1);
    return v[lo] + (idx - lo) * (v[hi] - v[lo]);
}

} // namespace stats

// ============================================================================
// Random Utilities
// ============================================================================
namespace random {

// Generate N random points in a circle
inline std::vector<Vec2> random_in_circle(int n, double radius,
                                           std::mt19937& rng) {
    std::uniform_real_distribution<double> angle(0, 2 * M_PI);
    std::uniform_real_distribution<double> r(0, 1);
    std::vector<Vec2> pts(n);
    for (int i = 0; i < n; i++) {
        double a = angle(rng);
        double rad = radius * std::sqrt(r(rng));
        pts[i] = {rad * std::cos(a), rad * std::sin(a)};
    }
    return pts;
}

// Generate N random points in a rectangle
inline std::vector<Vec2> random_in_rect(int n, double xmin, double xmax,
                                         double ymin, double ymax,
                                         std::mt19937& rng) {
    std::uniform_real_distribution<double> dx(xmin, xmax);
    std::uniform_real_distribution<double> dy(ymin, ymax);
    std::vector<Vec2> pts(n);
    for (int i = 0; i < n; i++)
        pts[i] = {dx(rng), dy(rng)};
    return pts;
}

} // namespace random

} // namespace betse
