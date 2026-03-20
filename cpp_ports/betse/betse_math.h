// BETSE C++ Port - Mathematical Utilities
// Finite difference operators, Gaussian filter, Helmholtz-Hodge decomposition,
// gradient, divergence, Laplacian solvers, wave equation, modulator functions.
// Ported from: betse/science/math/finitediff.py, toolbox.py, modulate.py, waves.py
#pragma once

#include "betse_types.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

namespace betse {

// ============================================================================
// 2D Grid for finite difference operations
// Mirrors finitediff.py FiniteDiffSolver
// ============================================================================
struct Grid2D {
    int nx = 0, ny = 0;
    double delta = 1.0;
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0;
    std::vector<double> x, y;

    Grid2D() = default;

    void make_grid(int n, double xmin_, double xmax_, double ymin_, double ymax_) {
        nx = n; ny = n;
        xmin = xmin_; xmax = xmax_; ymin = ymin_; ymax = ymax_;
        x.resize(nx); y.resize(ny);
        for (int i = 0; i < nx; i++) x[i] = xmin + (xmax - xmin) * i / std::max(nx - 1, 1);
        for (int j = 0; j < ny; j++) y[j] = ymin + (ymax - ymin) * j / std::max(ny - 1, 1);
        delta = (nx > 1) ? (x[1] - x[0]) : 1.0;
    }

    void cell_grid(double grid_delta, double xmin_, double xmax_, double ymin_, double ymax_) {
        xmin = xmin_; xmax = xmax_; ymin = ymin_; ymax = ymax_;
        delta = grid_delta;
        nx = (int)((xmax - xmin) / grid_delta);
        ny = (int)((ymax - ymin) / grid_delta);
        if (nx < 1) nx = 1;
        if (ny < 1) ny = 1;
        x.resize(nx); y.resize(ny);
        for (int i = 0; i < nx; i++) x[i] = xmin + (i + 0.5) * grid_delta;
        for (int j = 0; j < ny; j++) y[j] = ymin + (j + 0.5) * grid_delta;
    }

    int idx(int i, int j) const { return j * nx + i; }
    int size() const { return nx * ny; }
};

// ============================================================================
// Finite Difference Operators (from finitediff.py)
// ============================================================================
namespace fd {

inline void gradient(const std::vector<double>& f, int nx, int ny, double dx,
                     std::vector<double>& gx, std::vector<double>& gy) {
    gx.assign(nx * ny, 0.0);
    gy.assign(nx * ny, 0.0);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int k = j * nx + i;
            if (i > 0 && i < nx - 1)
                gx[k] = (f[k + 1] - f[k - 1]) / (2.0 * dx);
            else if (i == 0 && nx > 1)
                gx[k] = (f[k + 1] - f[k]) / dx;
            else if (i == nx - 1 && nx > 1)
                gx[k] = (f[k] - f[k - 1]) / dx;

            if (j > 0 && j < ny - 1)
                gy[k] = (f[k + nx] - f[k - nx]) / (2.0 * dx);
            else if (j == 0 && ny > 1)
                gy[k] = (f[k + nx] - f[k]) / dx;
            else if (j == ny - 1 && ny > 1)
                gy[k] = (f[k] - f[k - nx]) / dx;
        }
    }
}

inline std::vector<double> divergence(const std::vector<double>& fx,
                                       const std::vector<double>& fy,
                                       int nx, int ny, double dx, double dy) {
    std::vector<double> div(nx * ny, 0.0);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int k = j * nx + i;
            double dfx = 0, dfy = 0;
            if (i > 0 && i < nx - 1)      dfx = (fx[k+1] - fx[k-1]) / (2.0*dx);
            else if (i == 0 && nx > 1)     dfx = (fx[k+1] - fx[k]) / dx;
            else if (i == nx-1 && nx > 1)  dfx = (fx[k] - fx[k-1]) / dx;
            if (j > 0 && j < ny - 1)      dfy = (fy[k+nx] - fy[k-nx]) / (2.0*dy);
            else if (j == 0 && ny > 1)     dfy = (fy[k+nx] - fy[k]) / dy;
            else if (j == ny-1 && ny > 1)  dfy = (fy[k] - fy[k-nx]) / dy;
            div[k] = dfx + dfy;
        }
    }
    return div;
}

inline std::vector<double> laplacian(const std::vector<double>& f,
                                      int nx, int ny, double dx) {
    std::vector<double> lap(nx * ny, 0.0);
    double dx2 = dx * dx;
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int k = j * nx + i;
            double fk = f[k];
            double fxp = (i < nx-1) ? f[k+1] : fk;
            double fxm = (i > 0) ? f[k-1] : fk;
            double fyp = (j < ny-1) ? f[k+nx] : fk;
            double fym = (j > 0) ? f[k-nx] : fk;
            lap[k] = (fxp + fxm + fyp + fym - 4.0*fk) / dx2;
        }
    }
    return lap;
}

inline std::vector<double> gaussian_filter(const std::vector<double>& f,
                                            int nx, int ny, double sigma) {
    if (sigma <= 0.0 || nx <= 2 || ny <= 2) return f;
    int r = std::max(1, (int)(sigma + 0.5));
    std::vector<double> buf = f;
    std::vector<double> tmp(nx * ny);
    for (int pass = 0; pass < 3; pass++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double sum = 0; int cnt = 0;
                for (int di = -r; di <= r; di++) {
                    int ii = std::clamp(i + di, 0, nx - 1);
                    sum += buf[j * nx + ii]; cnt++;
                }
                tmp[j * nx + i] = sum / cnt;
            }
        }
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double sum = 0; int cnt = 0;
                for (int dj = -r; dj <= r; dj++) {
                    int jj = std::clamp(j + dj, 0, ny - 1);
                    sum += tmp[jj * nx + i]; cnt++;
                }
                buf[j * nx + i] = sum / cnt;
            }
        }
    }
    return buf;
}

inline std::vector<double> integrator(const std::vector<double>& f,
                                       int nx, int ny, double dx, double sharp = 0.5) {
    auto lap = laplacian(f, nx, ny, dx);
    double alpha = (1.0 - sharp) * dx * dx * 0.25;
    std::vector<double> result(f.size());
    for (size_t i = 0; i < f.size(); i++)
        result[i] = f[i] + alpha * lap[i];
    return result;
}

inline std::vector<double> curl_2d(const std::vector<double>& fx,
                                    const std::vector<double>& fy,
                                    int nx, int ny, double dx) {
    std::vector<double> gfyx, gfyy, gfxx, gfxy;
    gradient(fy, nx, ny, dx, gfyx, gfyy);
    gradient(fx, nx, ny, dx, gfxx, gfxy);
    std::vector<double> omega(nx * ny);
    for (int i = 0; i < nx * ny; i++)
        omega[i] = gfyx[i] - gfxy[i];
    return omega;
}

} // namespace fd

// ============================================================================
// Poisson solvers
// ============================================================================
inline std::vector<double> solve_poisson_jacobi(const std::vector<double>& rhs,
                                                  int nx, int ny, double dx,
                                                  int max_iter = 500, double tol = 1e-6) {
    std::vector<double> u(nx * ny, 0.0);
    std::vector<double> u_new(nx * ny, 0.0);
    double dx2 = dx * dx;
    for (int iter = 0; iter < max_iter; iter++) {
        double max_diff = 0.0;
        for (int j = 1; j < ny - 1; j++) {
            for (int i = 1; i < nx - 1; i++) {
                int k = j * nx + i;
                u_new[k] = 0.25 * (u[k-1] + u[k+1] + u[k-nx] + u[k+nx] - dx2 * rhs[k]);
                max_diff = std::max(max_diff, std::abs(u_new[k] - u[k]));
            }
        }
        std::swap(u, u_new);
        if (max_diff < tol) break;
    }
    return u;
}

inline std::vector<double> solve_graph_poisson(
    const std::vector<double>& rhs, int nc,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<std::vector<double>>& neighbor_dists,
    int max_iter = 300, double tol = 1e-6)
{
    std::vector<double> u(nc, 0.0), u_new(nc, 0.0);
    for (int iter = 0; iter < max_iter; iter++) {
        double max_diff = 0.0;
        for (int c = 0; c < nc; c++) {
            if (neighbors[c].empty()) { u_new[c] = 0.0; continue; }
            double sum_w = 0, sum_wu = 0;
            for (size_t k = 0; k < neighbors[c].size(); k++) {
                double w = 1.0 / (neighbor_dists[c][k] * neighbor_dists[c][k] + 1e-30);
                sum_w += w;
                sum_wu += w * u[neighbors[c][k]];
            }
            u_new[c] = (sum_wu - rhs[c]) / (sum_w + 1e-30);
            max_diff = std::max(max_diff, std::abs(u_new[c] - u[c]));
        }
        std::swap(u, u_new);
        if (max_diff < tol) break;
    }
    return u;
}

// ============================================================================
// Helmholtz-Hodge Decomposition (sim_toolbox.py HH_Decomp)
// ============================================================================
struct HHDecomp {
    std::vector<double> phi;
    std::vector<double> Fx_rot, Fy_rot;
    std::vector<double> Fx_irrot, Fy_irrot;
};

inline HHDecomp helmholtz_hodge_2d(const std::vector<double>& fx,
                                    const std::vector<double>& fy,
                                    int nx, int ny, double dx) {
    HHDecomp result;
    int n = nx * ny;
    auto div_f = fd::divergence(fx, fy, nx, ny, dx, dx);
    result.phi = solve_poisson_jacobi(div_f, nx, ny, dx, 500, 1e-7);
    fd::gradient(result.phi, nx, ny, dx, result.Fx_irrot, result.Fy_irrot);
    result.Fx_rot.resize(n);
    result.Fy_rot.resize(n);
    for (int i = 0; i < n; i++) {
        result.Fx_rot[i] = fx[i] - result.Fx_irrot[i];
        result.Fy_rot[i] = fy[i] - result.Fy_irrot[i];
    }
    return result;
}

struct HHDecompCells {
    std::vector<double> ux_rot, uy_rot;
    std::vector<double> ux_irrot, uy_irrot;
    std::vector<double> phi;
};

inline HHDecompCells helmholtz_hodge_cells(
    const std::vector<double>& ux, const std::vector<double>& uy, int nc,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<std::vector<double>>& neighbor_dists,
    const std::vector<Vec2>& cell_centers, bool /*rot_only*/ = true)
{
    HHDecompCells result;
    std::vector<double> div_u(nc, 0.0);
    for (int c = 0; c < nc; c++) {
        for (size_t k = 0; k < neighbors[c].size(); k++) {
            int nb = neighbors[c][k];
            Vec2 d = cell_centers[nb] - cell_centers[c];
            double dist = d.norm();
            if (dist < 1e-30) continue;
            double un = (ux[nb] - ux[c]) * d.x / dist + (uy[nb] - uy[c]) * d.y / dist;
            div_u[c] += un / dist;
        }
        if (!neighbors[c].empty()) div_u[c] /= (double)neighbors[c].size();
    }
    result.phi = solve_graph_poisson(div_u, nc, neighbors, neighbor_dists);
    result.ux_irrot.assign(nc, 0.0);
    result.uy_irrot.assign(nc, 0.0);
    for (int c = 0; c < nc; c++) {
        for (size_t k = 0; k < neighbors[c].size(); k++) {
            int nb = neighbors[c][k];
            Vec2 d = cell_centers[nb] - cell_centers[c];
            double dist = d.norm();
            if (dist < 1e-30) continue;
            double dphi = (result.phi[nb] - result.phi[c]) / dist;
            result.ux_irrot[c] += dphi * d.x / dist;
            result.uy_irrot[c] += dphi * d.y / dist;
        }
        if (!neighbors[c].empty()) {
            result.ux_irrot[c] /= (double)neighbors[c].size();
            result.uy_irrot[c] /= (double)neighbors[c].size();
        }
    }
    result.ux_rot.resize(nc);
    result.uy_rot.resize(nc);
    for (int c = 0; c < nc; c++) {
        result.ux_rot[c] = ux[c] - result.ux_irrot[c];
        result.uy_rot[c] = uy[c] - result.uy_irrot[c];
    }
    return result;
}

inline void smooth_flux(std::vector<double>& fx, std::vector<double>& fy,
                         int nx, int ny, double dx) {
    auto hh = helmholtz_hodge_2d(fx, fy, nx, ny, dx);
    fx = hh.Fx_rot;
    fy = hh.Fy_rot;
}

inline std::vector<double> single_cell_div_free(
    const std::vector<double>& flux, int nm,
    const std::vector<int>& mem_to_cell,
    const std::vector<double>& mem_sa, int nc)
{
    std::vector<double> div_sum(nc, 0.0), sa_sum(nc, 0.0);
    for (int m = 0; m < nm; m++) {
        int c = mem_to_cell[m];
        div_sum[c] += flux[m] * mem_sa[m];
        sa_sum[c] += mem_sa[m];
    }
    std::vector<double> flux_df(nm);
    for (int m = 0; m < nm; m++) {
        int c = mem_to_cell[m];
        double mean_flux = (sa_sum[c] > 0) ? div_sum[c] / sa_sum[c] : 0.0;
        flux_df[m] = flux[m] - mean_flux;
    }
    return flux_df;
}

// ============================================================================
// Modulator functions (modulate.py)
// ============================================================================
namespace modulate {

inline std::vector<double> gradient_x(const std::vector<Vec2>& points,
                                       double x_offset = 0.0, double slope = 1.0,
                                       double z_offset = 0.0, double exponent = 2.0) {
    int n = (int)points.size();
    if (n == 0) return {};
    double xmin = points[0].x, xmax = points[0].x;
    for (auto& p : points) { xmin = std::min(xmin, p.x); xmax = std::max(xmax, p.x); }
    double range = xmax - xmin;
    if (range < 1e-30) range = 1.0;
    double x_mid = 0.5 + x_offset;
    std::vector<double> result(n);
    for (int i = 0; i < n; i++) {
        double x_norm = (points[i].x - xmin) / range;
        double ratio = std::pow(x_norm / (x_mid + 1e-30), exponent);
        result[i] = (ratio / (1.0 + ratio)) * slope + z_offset;
    }
    return result;
}

inline std::vector<double> gradient_y(const std::vector<Vec2>& points,
                                       double y_offset = 0.0, double slope = 1.0,
                                       double z_offset = 0.0, double exponent = 2.0) {
    int n = (int)points.size();
    if (n == 0) return {};
    double ymin = points[0].y, ymax = points[0].y;
    for (auto& p : points) { ymin = std::min(ymin, p.y); ymax = std::max(ymax, p.y); }
    double range = ymax - ymin;
    if (range < 1e-30) range = 1.0;
    double y_mid = 0.5 + y_offset;
    std::vector<double> result(n);
    for (int i = 0; i < n; i++) {
        double y_norm = (points[i].y - ymin) / range;
        double ratio = std::pow(y_norm / (y_mid + 1e-30), exponent);
        result[i] = (ratio / (1.0 + ratio)) * slope + z_offset;
    }
    return result;
}

inline std::vector<double> gradient_r(const std::vector<Vec2>& points,
                                       Vec2 center = {0, 0},
                                       double slope = 1.0, double z_offset = 0.0) {
    int n = (int)points.size();
    if (n == 0) return {};
    double rmax = 0;
    std::vector<double> r(n);
    for (int i = 0; i < n; i++) {
        r[i] = (points[i] - center).norm();
        rmax = std::max(rmax, r[i]);
    }
    if (rmax < 1e-30) rmax = 1.0;
    std::vector<double> result(n);
    for (int i = 0; i < n; i++)
        result[i] = (r[i] / rmax) * slope + z_offset;
    return result;
}

inline double periodic(double t, double frequency = 1.0, double phase = 0.0) {
    double val = std::sin(t * M_PI * frequency + phase);
    return val * val;
}

inline double f_sweep(double t, double f_start, double f_slope) {
    double val = std::sin(M_PI * (f_slope * t + f_start) * t);
    return val * val;
}

} // namespace modulate

// ============================================================================
// Wave Equation Solver (math/waves.py WaveSolver)
// FDTD for 2D damped, forced wave equation: u_tt + gamma*u_t = c^2*Lap(u) + F
// ============================================================================
struct WaveSolver {
    Grid2D grid;
    double c = 1.0;
    double gamma = 1e-3;
    double dt = 0.01;
    std::vector<double> U_curr, U_prev, F_field;
    enum BoundaryType { FIXED, REFLECTIVE, OPEN };
    BoundaryType boundary = FIXED;
    std::function<double(double)> forcing_func = [](double t) {
        return std::cos(2.0 * M_PI * 0.75 * t);
    };

    void init(int grid_n, double xmin, double xmax, double ymin, double ymax,
              double wave_speed, double damping, double time_step) {
        c = wave_speed; gamma = damping; dt = time_step;
        grid.make_grid(grid_n, xmin, xmax, ymin, ymax);
        int n = grid.size();
        U_curr.assign(n, 0.0); U_prev.assign(n, 0.0);
        F_field.resize(n);
        double cx = (xmax + xmin) / 2.0, cy = (ymax + ymin) / 2.0;
        for (int j = 0; j < grid.ny; j++)
            for (int i = 0; i < grid.nx; i++) {
                double x = grid.x[i], y = grid.y[j];
                F_field[j * grid.nx + i] = std::exp(-10.0 * ((x-cx)*(x-cx) + (y-cy)*(y-cy)));
            }
        double K = (dt * c) / grid.delta;
        if (K > 0.5) dt = 0.45 * grid.delta / c;
        double ff = forcing_func(0.0);
        double coeff = (dt * dt) / (2.0 - dt * gamma);
        for (int i = 0; i < n; i++) U_curr[i] = coeff * F_field[i] * ff;
    }

    void step(double t) {
        int nx = grid.nx, ny = grid.ny;
        double dx = grid.delta;
        double K2 = (dt / dx) * (dt / dx);
        double ff = forcing_func(t);
        double inv_gam = 1.0 / (1.0 + gamma);
        std::vector<double> U_next(nx * ny, 0.0);
        for (int j = 1; j < ny - 1; j++)
            for (int i = 1; i < nx - 1; i++) {
                int k = j * nx + i;
                double lap = c * (U_curr[k+1] + U_curr[k-1] + U_curr[k+nx] + U_curr[k-nx] - 4.0*U_curr[k]);
                U_next[k] = inv_gam * (gamma * U_curr[k] + K2 * lap +
                    2.0 * U_curr[k] - U_prev[k] + dt * dt * F_field[k] * ff);
            }
        if (boundary == REFLECTIVE) {
            for (int j = 1; j < ny - 1; j++) {
                U_next[j*nx] = U_next[j*nx + 1];
                U_next[j*nx + nx - 1] = U_next[j*nx + nx - 2];
            }
            for (int i = 0; i < nx; i++) {
                U_next[i] = U_next[nx + i];
                U_next[(ny-1)*nx + i] = U_next[(ny-2)*nx + i];
            }
        } else if (boundary == OPEN) {
            double abc = ((dt * c / dx) - 1.0) / ((dt * c / dx) + 1.0);
            for (int j = 1; j < ny - 1; j++) {
                U_next[j*nx] = U_curr[j*nx+1] + abc * (U_next[j*nx+1] - U_curr[j*nx]);
                U_next[j*nx+nx-1] = U_curr[j*nx+nx-2] + abc * (U_next[j*nx+nx-2] - U_curr[j*nx+nx-1]);
            }
            for (int i = 1; i < nx - 1; i++) {
                U_next[i] = U_curr[nx+i] + abc * (U_next[nx+i] - U_curr[i]);
                U_next[(ny-1)*nx+i] = U_curr[(ny-2)*nx+i] + abc * (U_next[(ny-2)*nx+i] - U_curr[(ny-1)*nx+i]);
            }
        }
        U_prev = U_curr;
        U_curr = U_next;
    }
};

// ============================================================================
// Utility functions from sim_toolbox.py
// ============================================================================
inline void bicarbonate_buffer(double cCO2, double cHCO3,
                                double& cH_out, double& pH_out) {
    if (cCO2 < 1e-30) cCO2 = 1e-30;
    if (cHCO3 < 1e-30) cHCO3 = 1e-30;
    pH_out = 6.1 + std::log10(cHCO3 / cCO2);
    cH_out = std::pow(10.0, -pH_out) * 1e3;
}

inline double ghk_voltage(const std::vector<double>& Dm_cation,
                           const std::vector<double>& c_cation_in,
                           const std::vector<double>& c_cation_out,
                           const std::vector<double>& Dm_anion,
                           const std::vector<double>& c_anion_in,
                           const std::vector<double>& c_anion_out,
                           double T, double tm) {
    double sum_cat_out = 0, sum_cat_in = 0, sum_an_out = 0, sum_an_in = 0;
    for (size_t i = 0; i < Dm_cation.size(); i++) {
        double P = Dm_cation[i] / tm;
        sum_cat_out += P * c_cation_out[i];
        sum_cat_in += P * c_cation_in[i];
    }
    for (size_t i = 0; i < Dm_anion.size(); i++) {
        double P = Dm_anion[i] / tm;
        sum_an_in += P * c_anion_in[i];
        sum_an_out += P * c_anion_out[i];
    }
    double num = sum_cat_out + sum_an_in;
    double den = sum_cat_in + sum_an_out;
    if (den < 1e-30) den = 1e-30;
    return (R_GAS * T / F_FARADAY) * std::log(num / den);
}

} // namespace betse
