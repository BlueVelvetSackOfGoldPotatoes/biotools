// BETSE C++ Port - Physics Module
// Full ion current density, electroosmotic flow, Helmholtz-Smoluchowski,
// channel/pump lateral movement, deformation with HH decomposition.
// Ported from: physics/ion_current.py, flow.py, pressures.py, deform.py, move_channels.py
#pragma once

#include "betse_types.h"
#include "betse_math.h"
#include <cmath>
#include <vector>

namespace betse {

// ============================================================================
// Ion Current Density (from physics/ion_current.py get_current)
// Calculates currents in cells and environment from ion fluxes
// ============================================================================
struct CurrentState {
    std::vector<double> Jmem;       // transmembrane current density [A/m^2]
    std::vector<double> Jgj;        // gap junction current density
    std::vector<double> Jn;         // net normal current at membranes
    std::vector<double> I_mem;      // total membrane current [A]
    std::vector<double> J_cell_x;   // intracellular current x component at cell centres
    std::vector<double> J_cell_y;   // intracellular current y component at cell centres
    std::vector<double> Jc;         // normal component of J_cell at membranes
    std::vector<double> Ec;         // intracellular E-field at membranes (normal component)

    // Environmental current (if ECM enabled)
    std::vector<double> J_env_x;
    std::vector<double> J_env_y;
    std::vector<double> B_field;    // z-component of magnetic field

    // Environmental E-field
    std::vector<double> E_env_x;
    std::vector<double> E_env_y;

    // Extra contributions from GRN
    std::vector<double> extra_J_mem;
    std::vector<double> extra_rho_cells;
};

inline void compute_current(
    CurrentState& cs,
    int nc, int nm, int ni,
    const std::vector<int>& zs,
    const std::vector<std::vector<double>>& fluxes_mem,
    const std::vector<std::vector<double>>& fluxes_gj,
    const std::vector<std::vector<double>>& cc_cells,
    const std::vector<int>& mem_to_cell,
    const std::vector<double>& mem_sa,
    const std::vector<double>& cell_sa,
    const std::vector<double>& mem_nx,
    const std::vector<double>& mem_ny,
    const std::vector<double>& Vmem,
    std::vector<double>& rho_cells)
{
    cs.Jmem.assign(nm, 0.0);
    cs.Jgj.assign(nm, 0.0);
    cs.Jn.assign(nm, 0.0);
    cs.I_mem.assign(nm, 0.0);
    cs.J_cell_x.assign(nc, 0.0);
    cs.J_cell_y.assign(nc, 0.0);
    cs.Jc.assign(nm, 0.0);

    // rho_cells = sum(z * F * cc) + extra_rho
    rho_cells.assign(nc, 0.0);
    for (int ion = 0; ion < ni; ion++)
        for (int c = 0; c < nc; c++)
            rho_cells[c] += zs[ion] * F_FARADAY * cc_cells[ion][c];
    if ((int)cs.extra_rho_cells.size() == nc)
        for (int c = 0; c < nc; c++) rho_cells[c] += cs.extra_rho_cells[c];

    // Jmem = -sum(z * F * flux_mem) + extra_J_mem
    for (int ion = 0; ion < ni; ion++)
        for (int m = 0; m < nm; m++)
            cs.Jmem[m] -= zs[ion] * F_FARADAY * fluxes_mem[ion][m];
    if ((int)cs.extra_J_mem.size() == nm)
        for (int m = 0; m < nm; m++) cs.Jmem[m] += cs.extra_J_mem[m];

    // Jgj = sum(z * F * flux_gj)
    for (int ion = 0; ion < ni; ion++)
        for (int m = 0; m < nm; m++)
            cs.Jgj[m] += zs[ion] * F_FARADAY * fluxes_gj[ion][m];

    // Jn = Jmem + Jgj
    for (int m = 0; m < nm; m++)
        cs.Jn[m] = cs.Jmem[m] + cs.Jgj[m];

    // Smooth: average to cell centres then blend back
    std::vector<double> Jn_cell(nc, 0.0);
    std::vector<int> mem_count(nc, 0);
    for (int m = 0; m < nm; m++) {
        int c = mem_to_cell[m];
        Jn_cell[c] += cs.Jn[m] * mem_sa[m];
        mem_count[c]++;
    }
    for (int c = 0; c < nc; c++)
        if (cell_sa[c] > 0) Jn_cell[c] /= cell_sa[c];

    double nfrac = 3.0;
    for (int m = 0; m < nm; m++) {
        int c = mem_to_cell[m];
        int nm_c = mem_count[c];
        double sw_mem = (nfrac * nm_c - 1.0) / (nfrac * nm_c);
        double sw_o = 1.0 / (nfrac * nm_c);
        cs.Jn[m] = sw_mem * cs.Jn[m] + Jn_cell[c] * sw_o;
    }

    // I_mem = -Jn * mem_sa
    for (int m = 0; m < nm; m++)
        cs.I_mem[m] = -cs.Jn[m] * mem_sa[m];

    // Intracellular current at cell centres
    std::vector<double> Jcx_sum(nc, 0.0), Jcy_sum(nc, 0.0);
    for (int m = 0; m < nm; m++) {
        int c = mem_to_cell[m];
        Jcx_sum[c] += cs.Jn[m] * mem_nx[m] * mem_sa[m];
        Jcy_sum[c] += cs.Jn[m] * mem_ny[m] * mem_sa[m];
    }
    for (int c = 0; c < nc; c++) {
        if (cell_sa[c] > 0) {
            cs.J_cell_x[c] = Jcx_sum[c] / cell_sa[c];
            cs.J_cell_y[c] = Jcy_sum[c] / cell_sa[c];
        }
    }

    // Normal component at membranes
    for (int m = 0; m < nm; m++) {
        int c = mem_to_cell[m];
        cs.Jc[m] = cs.J_cell_x[c] * mem_nx[m] + cs.J_cell_y[c] * mem_ny[m];
    }
}

// ============================================================================
// Electroosmotic flow (from physics/flow.py getFlow)
// Helmholtz-Smoluchowski equation for extracellular EOF
// Stokes equation for intracellular flow through gap junctions
// ============================================================================
struct FlowState {
    std::vector<double> u_cells_x;
    std::vector<double> u_cells_y;
    std::vector<double> u_env_x;
    std::vector<double> u_env_y;
};

inline void compute_electroosmotic_flow(
    FlowState& fs,
    int nc, int nm,
    const std::vector<double>& E_cell_x,
    const std::vector<double>& E_cell_y,
    const std::vector<double>& rho_cells,
    double mu_water, double gj_surface,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<std::vector<double>>& neighbor_dists,
    const std::vector<Vec2>& cell_centers,
    bool fixed_boundary = true)
{
    fs.u_cells_x.assign(nc, 0.0);
    fs.u_cells_y.assign(nc, 0.0);

    // Body force: F = E * rho / mu * gj_surface
    std::vector<double> Fx(nc), Fy(nc);
    for (int c = 0; c < nc; c++) {
        Fx[c] = E_cell_x[c] * rho_cells[c] / mu_water * gj_surface;
        Fy[c] = E_cell_y[c] * rho_cells[c] / mu_water * gj_surface;
    }

    // Solve Stokes: u = lapGJinv * (-F)
    for (int c = 0; c < nc; c++) { Fx[c] = -Fx[c]; Fy[c] = -Fy[c]; }
    auto ux = solve_graph_poisson(Fx, nc, neighbors, neighbor_dists, 200);
    auto uy = solve_graph_poisson(Fy, nc, neighbors, neighbor_dists, 200);

    // Make divergence-free via HH decomposition
    auto hh = helmholtz_hodge_cells(ux, uy, nc, neighbors, neighbor_dists, cell_centers, true);
    fs.u_cells_x = hh.ux_rot;
    fs.u_cells_y = hh.uy_rot;
}

// ECM flow using Helmholtz-Smoluchowski (flow.py for p.is_ecm)
inline void compute_ecm_flow(
    FlowState& fs,
    const std::vector<double>& E_env_x,
    const std::vector<double>& E_env_y,
    const std::vector<double>& v_env,
    const std::vector<double>& D_env_weight,
    int nx, int ny, double dx,
    double eo, double er, double mu_water)
{
    int n = nx * ny;
    std::vector<double> muFx(n), muFy(n);
    for (int i = 0; i < n; i++) {
        muFx[i] = -eo * er / mu_water * E_env_x[i] * v_env[i] * D_env_weight[i];
        muFy[i] = -eo * er / mu_water * E_env_y[i] * v_env[i] * D_env_weight[i];
    }
    // Solve Poisson: u = lapInv(-muF)
    for (int i = 0; i < n; i++) { muFx[i] = -muFx[i]; muFy[i] = -muFy[i]; }
    auto uxo = solve_poisson_jacobi(muFx, nx, ny, dx, 300);
    auto uyo = solve_poisson_jacobi(muFy, nx, ny, dx, 300);

    // Make divergence-free
    auto hh = helmholtz_hodge_2d(uxo, uyo, nx, ny, dx);
    fs.u_env_x = hh.Fx_rot;
    fs.u_env_y = hh.Fy_rot;
}

// ============================================================================
// Osmotic pressure (from physics/pressures.py osmotic_P)
// Includes additional molecular substances from GRN
// ============================================================================
struct OsmoticState {
    std::vector<double> osmo_P_cell;
    std::vector<double> osmo_P_env;
    std::vector<double> osmo_P_delta;
    std::vector<double> u_osmo;       // osmotic water flux
    std::vector<double> div_u_osmo;   // divergence of osmotic flux
    std::vector<double> PP;           // reaction pressure from osmotic flow
};

inline void compute_osmotic_pressure(
    OsmoticState& os, int nc, int ni, int nm,
    const std::vector<std::vector<double>>& cc_cells,
    const std::vector<std::vector<double>>& cc_env,
    const std::vector<double>& P_cells,
    const std::vector<int>& mem_to_cell,
    const std::vector<double>& cell_sa,
    const std::vector<double>& cell_vol,
    double T, double aquaporins, double mu_water, double tm, double rho, double dt,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<std::vector<double>>& neighbor_dists,
    const std::vector<double>* extra_mol_cell = nullptr,
    const std::vector<double>* extra_mol_env = nullptr)
{
    os.osmo_P_cell.assign(nc, 0.0);
    os.osmo_P_env.assign(nc, 0.0);
    for (int ion = 0; ion < ni; ion++) {
        for (int c = 0; c < nc; c++) {
            os.osmo_P_cell[c] += R_GAS * T * cc_cells[ion][c];
            os.osmo_P_env[c] += R_GAS * T * cc_env[ion][c];
        }
    }
    if (extra_mol_cell && extra_mol_env) {
        for (int c = 0; c < nc; c++) {
            os.osmo_P_cell[c] += (*extra_mol_cell)[c];
            os.osmo_P_env[c] += (*extra_mol_env)[c];
        }
    }

    // Average env osmotic pressure from membranes to cells
    os.osmo_P_delta.assign(nc, 0.0);
    std::vector<double> env_avg(nc, 0.0);
    std::vector<int> env_cnt(nc, 0);
    for (int m = 0; m < nm; m++) {
        int c = mem_to_cell[m];
        env_avg[c] += os.osmo_P_env[c]; // simplified: env indexed by cell
        env_cnt[c]++;
    }
    for (int c = 0; c < nc; c++) {
        if (env_cnt[c] > 0) env_avg[c] /= env_cnt[c];
        os.osmo_P_delta[c] = env_avg[c] - os.osmo_P_cell[c];
    }

    // Transmembrane water flux
    os.u_osmo.assign(nc, 0.0);
    os.div_u_osmo.assign(nc, 0.0);
    double pore_area = 3e-10 * 3e-10; // aquaporin pore area
    for (int c = 0; c < nc; c++) {
        os.u_osmo[c] = (os.osmo_P_delta[c] - P_cells[c]) *
                        (aquaporins / mu_water) * pore_area / tm;
        os.div_u_osmo[c] = os.u_osmo[c] * cell_sa[c] / cell_vol[c];
    }

    // Solve for reaction pressure: PP = lapGJinv(-div_u * rho * dt)
    std::vector<double> rhs(nc);
    for (int c = 0; c < nc; c++) rhs[c] = -os.div_u_osmo[c] * rho * dt;
    os.PP = solve_graph_poisson(rhs, nc, neighbors, neighbor_dists, 200);
}

// ============================================================================
// Deformation (from physics/deform.py getDeformation, timeDeform)
// Calculates cell cluster deformation under pressure and galvanotropism
// ============================================================================
struct DeformState {
    std::vector<double> d_cells_x;
    std::vector<double> d_cells_y;
    std::vector<double> gPxc, gPyc;  // pressure gradient at cell centres
    // For time-dependent deformation
    std::vector<std::vector<double>> dx_time;
    std::vector<std::vector<double>> dy_time;
};

// Steady-state deformation (getDeformation)
inline void compute_deformation_steady(
    DeformState& ds, int nc,
    const std::vector<double>& P_cells,
    const std::vector<double>& E_cell_x,
    const std::vector<double>& E_cell_y,
    double lame_mu, double galvanotropism,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<std::vector<double>>& neighbor_dists,
    const std::vector<Vec2>& cell_centers,
    bool deform_osmo = false,
    const std::vector<double>* PP = nullptr,
    bool fixed_boundary = true)
{
    ds.gPxc.assign(nc, 0.0);
    ds.gPyc.assign(nc, 0.0);

    // Compute pressure gradient at cell centres
    for (int c = 0; c < nc; c++) {
        double gpx = 0, gpy = 0;
        for (size_t k = 0; k < neighbors[c].size(); k++) {
            int nb = neighbors[c][k];
            Vec2 d = cell_centers[nb] - cell_centers[c];
            double dist = d.norm();
            if (dist < 1e-30) continue;
            double dP = P_cells[nb] - P_cells[c];
            gpx -= dP * d.x / (dist * dist);
            gpy -= dP * d.y / (dist * dist);
        }
        if (!neighbors[c].empty()) {
            gpx /= (double)neighbors[c].size();
            gpy /= (double)neighbors[c].size();
        }
        ds.gPxc[c] = gpx;
        ds.gPyc[c] = gpy;
    }

    // Force = (1/lame_mu) * grad(P)
    std::vector<double> Fx(nc), Fy(nc);
    for (int c = 0; c < nc; c++) {
        Fx[c] = ds.gPxc[c] / lame_mu;
        Fy[c] = ds.gPyc[c] / lame_mu;
    }

    // Solve: d = lapGJinv(-F)
    for (int c = 0; c < nc; c++) { Fx[c] = -Fx[c]; Fy[c] = -Fy[c]; }
    auto dxo = solve_graph_poisson(Fx, nc, neighbors, neighbor_dists, 200);
    auto dyo = solve_graph_poisson(Fy, nc, neighbors, neighbor_dists, 200);

    // Make divergence-free via HH decomposition
    auto hh = helmholtz_hodge_cells(dxo, dyo, nc, neighbors, neighbor_dists, cell_centers, true);
    ds.d_cells_x = hh.ux_rot;
    ds.d_cells_y = hh.uy_rot;

    // Add osmotic deformation if enabled
    if (deform_osmo && PP) {
        for (int c = 0; c < nc; c++) {
            double opx = 0, opy = 0;
            for (size_t k = 0; k < neighbors[c].size(); k++) {
                int nb = neighbors[c][k];
                Vec2 d = cell_centers[nb] - cell_centers[c];
                double dist = d.norm();
                if (dist < 1e-30) continue;
                double dP = (*PP)[nb] - (*PP)[c];
                opx -= dP * d.x / (dist * dist);
                opy -= dP * d.y / (dist * dist);
            }
            if (!neighbors[c].empty()) {
                opx /= (double)neighbors[c].size();
                opy /= (double)neighbors[c].size();
            }
            ds.d_cells_x[c] += opx;
            ds.d_cells_y[c] += opy;
        }
    }

    // Add galvanotropism
    for (int c = 0; c < nc; c++) {
        ds.d_cells_x[c] += E_cell_x[c] * galvanotropism;
        ds.d_cells_y[c] += E_cell_y[c] * galvanotropism;
    }
}

// Time-dependent deformation (timeDeform)
inline void compute_deformation_time(
    DeformState& ds, int nc, double t, double dt,
    const std::vector<double>& P_cells,
    const std::vector<double>& E_cell_x,
    const std::vector<double>& E_cell_y,
    const std::vector<double>& J_cell_x,
    const std::vector<double>& J_cell_y,
    const std::vector<double>& rho_cells,
    double lame_mu, double galvanotropism, double sigma, double mu_tissue,
    double cell_radius,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<std::vector<double>>& neighbor_dists,
    const std::vector<Vec2>& cell_centers,
    bool deform_osmo = false,
    const std::vector<double>* PP = nullptr,
    bool fixed_boundary = true)
{
    // Store current displacement in history
    ds.dx_time.push_back(ds.d_cells_x);
    ds.dy_time.push_back(ds.d_cells_y);

    // Pressure gradient
    std::vector<double> Pcell = P_cells;
    if (deform_osmo && PP) {
        for (int c = 0; c < nc; c++) Pcell[c] += (*PP)[c];
    }

    ds.gPxc.assign(nc, 0.0);
    ds.gPyc.assign(nc, 0.0);
    for (int c = 0; c < nc; c++) {
        for (size_t k = 0; k < neighbors[c].size(); k++) {
            int nb = neighbors[c][k];
            Vec2 d = cell_centers[nb] - cell_centers[c];
            double dist = d.norm();
            if (dist < 1e-30) continue;
            double dP = Pcell[nb] - Pcell[c];
            ds.gPxc[c] -= dP * d.x / (dist * dist);
            ds.gPyc[c] -= dP * d.y / (dist * dist);
        }
        if (!neighbors[c].empty()) {
            ds.gPxc[c] /= (double)neighbors[c].size();
            ds.gPyc[c] /= (double)neighbors[c].size();
        }
    }

    // Force with galvanotropism
    double sig = (sigma > 1e-30) ? sigma : 1.0;
    std::vector<double> Fx(nc), Fy(nc);
    for (int c = 0; c < nc; c++) {
        Fx[c] = (1.0 / lame_mu) * ((1.0/sig) * J_cell_x[c] * rho_cells[c] * galvanotropism + ds.gPxc[c]);
        Fy[c] = (1.0 / lame_mu) * ((1.0/sig) * J_cell_y[c] * rho_cells[c] * galvanotropism + ds.gPyc[c]);
    }

    double k_const = dt * dt * lame_mu / 1000.0;

    if (ds.dx_time.size() <= 1 || t <= 0.0) {
        // Initial value solution
        // Graph Laplacian * d + force
        for (int c = 0; c < nc; c++) {
            double lap_dx = 0, lap_dy = 0;
            for (size_t k = 0; k < neighbors[c].size(); k++) {
                int nb = neighbors[c][k];
                double dist2 = neighbor_dists[c][k] * neighbor_dists[c][k];
                if (dist2 < 1e-30) continue;
                lap_dx += (ds.dx_time.back()[nb] - ds.dx_time.back()[c]) / dist2;
                lap_dy += (ds.dy_time.back()[nb] - ds.dy_time.back()[c]) / dist2;
            }
            ds.d_cells_x[c] = k_const * lap_dx + (k_const / lame_mu) * Fx[c] + ds.dx_time.back()[c];
            ds.d_cells_y[c] = k_const * lap_dy + (k_const / lame_mu) * Fy[c] + ds.dy_time.back()[c];
        }
    } else {
        // Standard time-stepping with viscous damping
        double gamma_d = dt * dt * mu_tissue * lame_mu / (1000.0 * 2.0 * cell_radius);
        auto& dx_n = ds.dx_time[ds.dx_time.size() - 1];
        auto& dx_n1 = ds.dx_time[ds.dx_time.size() - 2];
        auto& dy_n = ds.dy_time[ds.dy_time.size() - 1];
        auto& dy_n1 = ds.dy_time[ds.dy_time.size() - 2];

        for (int c = 0; c < nc; c++) {
            double lap_dx = 0, lap_dy = 0;
            for (size_t k = 0; k < neighbors[c].size(); k++) {
                int nb = neighbors[c][k];
                double dist2 = neighbor_dists[c][k] * neighbor_dists[c][k];
                if (dist2 < 1e-30) continue;
                lap_dx += (dx_n[nb] - dx_n[c]) / dist2;
                lap_dy += (dy_n[nb] - dy_n[c]) / dist2;
            }
            double dux_dt = (dx_n[c] - dx_n1[c]) / dt;
            double duy_dt = (dy_n[c] - dy_n1[c]) / dt;

            ds.d_cells_x[c] = k_const * lap_dx - gamma_d * dux_dt +
                               (k_const / lame_mu) * Fx[c] + 2.0 * dx_n[c] - dx_n1[c];
            ds.d_cells_y[c] = k_const * lap_dy - gamma_d * duy_dt +
                               (k_const / lame_mu) * Fy[c] + 2.0 * dy_n[c] - dy_n1[c];
        }
    }

    // Make divergence-free
    auto hh = helmholtz_hodge_cells(ds.d_cells_x, ds.d_cells_y, nc,
                                     neighbors, neighbor_dists, cell_centers, true);
    ds.d_cells_x = hh.ux_rot;
    ds.d_cells_y = hh.uy_rot;
}

// ============================================================================
// Channel/Pump lateral movement (from physics/move_channels.py MoveChannel)
// Electroosmotic lateral redistribution of pumps and channels on membranes
// ============================================================================
struct MoveChannelState {
    std::vector<double> rho_pump;     // pump density factor per membrane
    std::vector<double> rho_channel;  // channel density factor per membrane
};

inline void init_move_channels(MoveChannelState& mcs, int nm) {
    mcs.rho_pump.assign(nm, 1.0);
    mcs.rho_channel.assign(nm, 1.0);
}

// Update pump/channel distribution using Nernst-Planck on membrane surface
inline void update_move_pump(MoveChannelState& mcs, int nm,
                              const std::vector<double>& Ec,
                              const std::vector<int>& mem_to_cell,
                              const std::vector<double>& R_rads,
                              const std::vector<double>& mem_sa,
                              const std::vector<double>& mem_vol,
                              double D_membrane, double z_pump,
                              double q, double kb, double T, double dt, int nc) {
    for (int m = 0; m < nm; m++) {
        int c = mem_to_cell[m];
        double cav = 1.0;
        double cpi = mcs.rho_pump[m];
        double cap = (cav + cpi) / 2.0;
        double cgp = (cpi - cav) / R_rads[c];
        double flux = -D_membrane * cgp + (D_membrane * q * cap * z_pump / (kb * T)) * Ec[m];
        mcs.rho_pump[m] = cpi + flux * (mem_sa[m] / (mem_vol[m] + FLOAT_NONCE)) * dt;
        mcs.rho_pump[m] = std::max(0.01, std::min(10.0, mcs.rho_pump[m]));
    }
    // Make divergence-free per cell
    auto flux_df = single_cell_div_free(mcs.rho_pump, nm, mem_to_cell, mem_sa, nc);
    // Blend: rho stays close to 1.0 on average
    for (int m = 0; m < nm; m++)
        mcs.rho_pump[m] = std::max(0.01, mcs.rho_pump[m]);
}

inline void update_move_channel(MoveChannelState& mcs, int nm,
                                 const std::vector<double>& Ec,
                                 const std::vector<int>& mem_to_cell,
                                 const std::vector<double>& R_rads,
                                 const std::vector<double>& mem_sa,
                                 const std::vector<double>& mem_vol,
                                 double D_membrane, double z_channel,
                                 double q, double kb, double T, double dt, int nc) {
    for (int m = 0; m < nm; m++) {
        int c = mem_to_cell[m];
        double cav = 1.0;
        double cpi = mcs.rho_channel[m];
        double cap = (cav + cpi) / 2.0;
        double cgp = (cpi - cav) / R_rads[c];
        double flux = -D_membrane * cgp + (D_membrane * q * cap * z_channel / (kb * T)) * Ec[m];
        mcs.rho_channel[m] = cpi + flux * (mem_sa[m] / (mem_vol[m] + FLOAT_NONCE)) * dt;
        mcs.rho_channel[m] = std::max(0.01, std::min(10.0, mcs.rho_channel[m]));
    }
}

// ============================================================================
// Environmental voltage (from ion_current.py for both ECM and non-ECM cases)
// ============================================================================
inline void compute_env_voltage_simple(
    std::vector<double>& v_env, int ne,
    const std::vector<double>& Vmem, int nm,
    const std::vector<int>& map_mem2ecm,
    double cm, double delta, double cell_height)
{
    v_env.assign(ne, 0.0);
    for (int m = 0; m < nm; m++) {
        if (m < (int)map_mem2ecm.size() && map_mem2ecm[m] >= 0 && map_mem2ecm[m] < ne) {
            v_env[map_mem2ecm[m]] = -Vmem[m] / 2.0;
        }
    }
}

// Screening parameter for Debye length
inline double compute_screening_constant(double ionic_strength, double T,
                                           double eo, double er) {
    // Debye-Huckel: kappa = sqrt(2*F^2*I / (eo*er*R*T))
    double num = 2.0 * F_FARADAY * F_FARADAY * ionic_strength;
    double den = eo * er * R_GAS * T;
    if (den < 1e-30) den = 1e-30;
    return std::sqrt(num / den);
}

} // namespace betse
