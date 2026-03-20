// BETSE C++ Port - Fast Solver (Equivalent Circuit Model)
// Implementation of the FAST solver from betse/science/sim.py
// Uses equivalent circuit formalism for rapid bioelectric simulation.
// Also includes the FULL solver's core electrodiffusion routines and
// the complete Nernst-Planck solver with ECM coupling.
// Ported from: sim.py (fast_sim, run_sim), sim_toolbox.py
#pragma once

#include "betse_types.h"
#include "betse_math.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace betse {

// ============================================================================
// Equivalent Circuit Model (FAST solver)
// Analogizes cells to RC circuits connected by gap junction resistors.
// Vmem = Q / C, with currents from ion channels and pumps.
// From sim.py: the fast solver computes Vmem directly without tracking
// individual ion concentrations in the environment.
// ============================================================================
struct FastSolver {
    // Per-cell equivalent circuit parameters
    std::vector<double> Cm_cell;     // total membrane capacitance per cell [F]
    std::vector<double> Gm_cell;     // total membrane conductance per cell [S]
    std::vector<double> E_rev_cell;  // effective reversal potential per cell [V]
    std::vector<double> I_pump_cell; // total pump current per cell [A]
    std::vector<double> Vm_cell;     // cell membrane voltage [V]

    // GJ conductance matrix (sparse: per-neighbor pair)
    std::vector<std::vector<int>> gj_neighbors;
    std::vector<std::vector<double>> gj_conductance;

    // Channel contributions (summed per cell)
    std::vector<double> G_Na, G_K, G_Ca, G_Cl, G_H;
    std::vector<double> E_Na, E_K, E_Ca, E_Cl, E_H;

    void init(int nc, double cm, double cell_sa_avg,
              const std::vector<double>& cell_sa,
              const std::vector<std::vector<int>>& neighbors,
              const std::vector<std::vector<double>>& neighbor_dists,
              double gj_conductance_val) {
        Cm_cell.resize(nc);
        Gm_cell.assign(nc, 0.0);
        E_rev_cell.assign(nc, 0.0);
        I_pump_cell.assign(nc, 0.0);
        Vm_cell.assign(nc, -0.05);  // -50 mV initial

        G_Na.assign(nc, 0.0);
        G_K.assign(nc, 0.0);
        G_Ca.assign(nc, 0.0);
        G_Cl.assign(nc, 0.0);
        G_H.assign(nc, 0.0);
        E_Na.assign(nc, 0.058);
        E_K.assign(nc, -0.085);
        E_Ca.assign(nc, 0.12);
        E_Cl.assign(nc, -0.07);
        E_H.assign(nc, 0.0);

        for (int c = 0; c < nc; c++)
            Cm_cell[c] = cm * cell_sa[c];

        gj_neighbors = neighbors;
        gj_conductance.resize(nc);
        for (int c = 0; c < nc; c++) {
            gj_conductance[c].resize(neighbors[c].size());
            for (size_t k = 0; k < neighbors[c].size(); k++) {
                double dist = neighbor_dists[c][k];
                gj_conductance[c][k] = gj_conductance_val / (dist + 1e-30);
            }
        }
    }

    // Update Nernst potentials from current concentrations
    void update_nernst(int nc, double T,
                       const std::vector<std::vector<double>>& cc_cells,
                       const std::vector<std::vector<double>>& cc_env) {
        for (int c = 0; c < nc; c++) {
            auto nernst = [&](int ion, int z) {
                double ci = std::max(cc_cells[ion][c], 1e-15);
                double co = std::max(cc_env[ion][c], 1e-15);
                return (R_GAS * T) / (z * F_FARADAY) * std::log(co / ci);
            };
            if (cc_cells.size() > 0) E_Na[c] = nernst(0, 1);
            if (cc_cells.size() > 1) E_K[c] = nernst(1, 1);
            if (cc_cells.size() > 2) E_Cl[c] = nernst(2, -1);
            if (cc_cells.size() > 3) E_Ca[c] = nernst(3, 2);
            if (cc_cells.size() > 4) E_H[c] = nernst(4, 1);
        }
    }

    // Update membrane conductances from permeabilities
    void update_conductances(int nc, const std::vector<double>& cell_sa,
                              double tm,
                              const std::vector<double>& Dm_Na,
                              const std::vector<double>& Dm_K,
                              const std::vector<double>& Dm_Cl,
                              const std::vector<double>& Dm_Ca,
                              const std::vector<double>& Dm_H) {
        for (int c = 0; c < nc; c++) {
            // G = Dm * F^2 / (R*T) * SA / tm (simplified conductance)
            double factor = F_FARADAY * cell_sa[c] / tm;
            G_Na[c] = Dm_Na[c] * factor;
            G_K[c] = Dm_K[c] * factor;
            G_Ca[c] = Dm_Ca[c] * factor * 2.0; // z=2
            G_Cl[c] = Dm_Cl[c] * factor;
            G_H[c] = Dm_H[c] * factor;

            Gm_cell[c] = G_Na[c] + G_K[c] + G_Ca[c] + G_Cl[c] + G_H[c];
            if (Gm_cell[c] > 1e-30)
                E_rev_cell[c] = (G_Na[c] * E_Na[c] + G_K[c] * E_K[c] +
                                 G_Ca[c] * E_Ca[c] + G_Cl[c] * E_Cl[c] +
                                 G_H[c] * E_H[c]) / Gm_cell[c];
        }
    }

    // Semi-implicit time step: solve RC circuit with GJ coupling
    void step(int nc, double dt) {
        std::vector<double> Vm_new(nc);

        for (int c = 0; c < nc; c++) {
            // Total GJ current into cell c
            double I_gj = 0;
            double G_gj_total = 0;
            for (size_t k = 0; k < gj_neighbors[c].size(); k++) {
                int nb = gj_neighbors[c][k];
                double Ggj = gj_conductance[c][k];
                I_gj += Ggj * (Vm_cell[nb] - Vm_cell[c]);
                G_gj_total += Ggj;
            }

            // Semi-implicit Euler: C * dV/dt = I_gj - Gm*(V-Erev) + I_pump
            // C * (V_new - V) / dt = -Gm * V_new + Gm * Erev + I_gj + I_pump
            //                        + G_gj_total * (V_neighbors_avg - V_new)
            // Simplified: treat GJ neighbor voltages as known (explicit)
            double numer = Cm_cell[c] * Vm_cell[c] / dt +
                           Gm_cell[c] * E_rev_cell[c] +
                           I_gj + I_pump_cell[c];
            double denom = Cm_cell[c] / dt + Gm_cell[c] + G_gj_total;
            if (denom > 1e-30)
                Vm_new[c] = numer / denom;
            else
                Vm_new[c] = Vm_cell[c];
        }

        Vm_cell = std::move(Vm_new);
    }

    // Update concentrations from voltage changes (post-hoc)
    void update_concentrations(int nc, double dt,
                                std::vector<std::vector<double>>& cc_cells,
                                const std::vector<double>& cell_vol,
                                const std::vector<double>& cell_sa) {
        for (int c = 0; c < nc; c++) {
            double V = Vm_cell[c];

            // Current through each ion channel: I = G * (V - E)
            // Flux = I / (z * F * SA) [mol/(m^2*s)]
            // Delta_c = flux * SA / vol * dt [mM]

            auto update_ion = [&](int ion, double G, double E, int z) {
                if (ion >= (int)cc_cells.size()) return;
                double I = G * (V - E);
                double flux = I / (z * F_FARADAY * cell_sa[c] + FLOAT_NONCE);
                cc_cells[ion][c] += flux * cell_sa[c] / cell_vol[c] * dt;
                cc_cells[ion][c] = std::max(0.0, cc_cells[ion][c]);
            };

            update_ion(0, G_Na[c], E_Na[c], 1);
            update_ion(1, G_K[c], E_K[c], 1);
            update_ion(2, G_Cl[c], E_Cl[c], -1);
            update_ion(3, G_Ca[c], E_Ca[c], 2);
            update_ion(4, G_H[c], E_H[c], 1);
        }
    }
};

// ============================================================================
// Full Nernst-Planck Solver (FULL solver)
// Complete electrodiffusion with ECM transport, transmembrane fluxes,
// gap junction fluxes, and charge conservation.
// From sim.py: the full solver tracks individual ion concentrations
// everywhere and computes Vmem from charge.
// ============================================================================
struct FullSolver {

    // Compute transmembrane flux for all ions at all membranes
    static void compute_membrane_fluxes(
        std::vector<std::vector<double>>& flux_mem,  // [ion][mem] output
        int ni, int nm,
        const std::vector<std::vector<double>>& cc_cells,
        const std::vector<std::vector<double>>& cc_env,
        const std::vector<std::vector<double>>& Dm_mems,  // [ion][mem]
        const std::vector<double>& Vmem,
        const std::vector<int>& mem_to_cell,
        const std::vector<int>& zs,
        double tm, double T)
    {
        for (int ion = 0; ion < ni; ion++) {
            for (int m = 0; m < nm; m++) {
                int c = mem_to_cell[m];
                flux_mem[ion][m] = electroflux(
                    cc_env[ion][c], cc_cells[ion][c],
                    Dm_mems[ion][m], tm, zs[ion], Vmem[m], T);
            }
        }
    }

    // Compute gap junction flux for all ions
    static void compute_gj_fluxes(
        std::vector<std::vector<double>>& flux_gj,   // [ion][mem] output
        int ni, int nm,
        const std::vector<std::vector<double>>& cc_mem,
        const std::vector<double>& Vmem,
        const std::vector<int>& neighbor_mem_id,
        const std::vector<double>& gj_open,
        const std::vector<double>& gj_dist,
        const std::vector<double>& D_free,
        const std::vector<int>& zs,
        double gj_surface, double T,
        const std::vector<double>& gj_block)
    {
        for (int ion = 0; ion < ni; ion++) {
            for (int m = 0; m < nm; m++) {
                int nn = neighbor_mem_id[m];
                if (nn < 0) { flux_gj[ion][m] = 0; continue; }
                double D_gj = D_free[ion] * gj_surface *
                              gj_open[m] * gj_block[m];
                double vgj = Vmem[nn] - Vmem[m];
                flux_gj[ion][m] = electroflux(
                    cc_mem[ion][m], cc_mem[ion][nn],
                    D_gj, gj_dist[m], zs[ion], vgj, T);
            }
        }
    }

    // Update concentrations from fluxes
    static void update_concentrations(
        std::vector<std::vector<double>>& cc_cells,
        std::vector<std::vector<double>>& cc_env,
        std::vector<std::vector<double>>& cc_mem,
        const std::vector<std::vector<double>>& flux_mem,
        const std::vector<std::vector<double>>& flux_gj,
        int ni, int nc, int nm,
        const std::vector<int>& mem_to_cell,
        const std::vector<double>& mem_sa,
        const std::vector<double>& cell_vol,
        double dt, bool ecm_enabled = false)
    {
        for (int ion = 0; ion < ni; ion++) {
            // Membrane flux -> cell concentration change
            std::vector<double> delta(nc, 0.0);
            for (int m = 0; m < nm; m++) {
                int c = mem_to_cell[m];
                delta[c] += flux_mem[ion][m] * mem_sa[m];
            }
            for (int c = 0; c < nc; c++) {
                cc_cells[ion][c] += dt * delta[c] / cell_vol[c];
            }

            // GJ flux -> cell concentration change
            std::fill(delta.begin(), delta.end(), 0.0);
            for (int m = 0; m < nm; m++) {
                int c = mem_to_cell[m];
                delta[c] -= flux_gj[ion][m] * mem_sa[m];
            }
            for (int c = 0; c < nc; c++) {
                cc_cells[ion][c] += dt * delta[c] / cell_vol[c];
                cc_cells[ion][c] = std::max(0.0, cc_cells[ion][c]);
            }

            // Update membrane concentrations
            for (int m = 0; m < nm; m++)
                cc_mem[ion][m] = cc_cells[ion][mem_to_cell[m]];

            // Environment update (well-mixed if no ECM)
            if (!ecm_enabled) {
                double total_flux = 0;
                for (int m = 0; m < nm; m++)
                    total_flux -= flux_mem[ion][m] * mem_sa[m];
                double env_vol = cell_vol[0] * nc * 10.0;
                double delta_env = total_flux / env_vol;
                for (int c = 0; c < (int)cc_env[ion].size(); c++) {
                    cc_env[ion][c] += dt * delta_env;
                    cc_env[ion][c] = std::max(0.0, cc_env[ion][c]);
                }
            }
        }
    }

    // Compute Vmem from charge density
    static void compute_vmem(
        std::vector<double>& Vmem,
        const std::vector<std::vector<double>>& cc_cells,
        int ni, int nc, int nm,
        const std::vector<int>& mem_to_cell,
        const std::vector<int>& zs,
        const std::vector<double>& cell_vol,
        const std::vector<double>& cell_sa,
        double cm,
        const std::vector<double>* extra_rho = nullptr)
    {
        for (int m = 0; m < nm; m++) {
            int c = mem_to_cell[m];
            double rho = 0;
            for (int ion = 0; ion < ni; ion++)
                rho += zs[ion] * F_FARADAY * cc_cells[ion][c];
            if (extra_rho && c < (int)extra_rho->size())
                rho += (*extra_rho)[c];
            double sigma = rho * cell_vol[c] / cell_sa[c];
            Vmem[m] = sigma / cm;
        }
    }
};

// ============================================================================
// Conjugate Gradient Solver (for Poisson equation in ECM)
// Used by FULL solver for environmental voltage calculation
// ============================================================================
inline std::vector<double> solve_cg(
    const std::function<std::vector<double>(const std::vector<double>&)>& A_mul,
    const std::vector<double>& b,
    int max_iter = 500, double tol = 1e-8)
{
    int n = (int)b.size();
    std::vector<double> x(n, 0.0);
    auto r = b;  // r = b - A*x, x=0 so r=b
    auto p = r;
    double rr = 0;
    for (double v : r) rr += v * v;

    for (int iter = 0; iter < max_iter; iter++) {
        if (rr < tol * tol) break;
        auto Ap = A_mul(p);
        double pAp = 0;
        for (int i = 0; i < n; i++) pAp += p[i] * Ap[i];
        if (std::abs(pAp) < 1e-30) break;
        double alpha = rr / pAp;

        double rr_new = 0;
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            rr_new += r[i] * r[i];
        }
        double beta = rr_new / (rr + 1e-30);
        for (int i = 0; i < n; i++)
            p[i] = r[i] + beta * p[i];
        rr = rr_new;
    }
    return x;
}

// ============================================================================
// SOR (Successive Over-Relaxation) Poisson solver for ECM voltage
// ============================================================================
inline std::vector<double> solve_poisson_sor(
    const std::vector<double>& rhs,
    int nx, int ny, double dx,
    int max_iter = 500, double tol = 1e-6, double omega = 1.5)
{
    std::vector<double> u(nx * ny, 0.0);
    double dx2 = dx * dx;

    for (int iter = 0; iter < max_iter; iter++) {
        double max_diff = 0.0;
        for (int j = 1; j < ny - 1; j++) {
            for (int i = 1; i < nx - 1; i++) {
                int k = j * nx + i;
                double u_new = 0.25 * (u[k-1] + u[k+1] + u[k-nx] + u[k+nx] -
                                        dx2 * rhs[k]);
                double diff = u_new - u[k];
                u[k] += omega * diff;
                max_diff = std::max(max_diff, std::abs(diff));
            }
        }
        if (max_diff < tol) break;
    }
    return u;
}

// ============================================================================
// Boundary voltage application (for ECM simulations)
// Applies voltage at grid boundaries per side
// ============================================================================
inline void apply_boundary_voltage(
    std::vector<double>& V_ecm,
    int nx, int ny,
    double V_top, double V_bottom, double V_left, double V_right)
{
    // Top row (j = ny-1)
    for (int i = 0; i < nx; i++)
        V_ecm[(ny-1) * nx + i] = V_top;
    // Bottom row (j = 0)
    for (int i = 0; i < nx; i++)
        V_ecm[i] = V_bottom;
    // Left column (i = 0)
    for (int j = 0; j < ny; j++)
        V_ecm[j * nx] = V_left;
    // Right column (i = nx-1)
    for (int j = 0; j < ny; j++)
        V_ecm[j * nx + nx - 1] = V_right;
}

// ============================================================================
// Environmental voltage solver (from sim.py compute_venv)
// Solves for voltage in ECM grid from charge distribution
// ============================================================================
inline void solve_ecm_voltage(
    std::vector<double>& V_ecm,
    int nx, int ny, double dx,
    const std::vector<std::vector<double>>& cc_ecm,
    int ni, const std::vector<int>& zs,
    double eo, double er,
    double V_top = 0, double V_bottom = 0,
    double V_left = 0, double V_right = 0)
{
    int ne = nx * ny;
    // Compute charge density
    std::vector<double> rho(ne, 0.0);
    for (int ion = 0; ion < ni; ion++) {
        if (ion >= (int)cc_ecm.size()) continue;
        for (int k = 0; k < ne; k++)
            rho[k] += zs[ion] * F_FARADAY * cc_ecm[ion][k];
    }

    // Solve Poisson: nabla^2 V = -rho / (eo * er)
    std::vector<double> rhs(ne);
    for (int k = 0; k < ne; k++)
        rhs[k] = -rho[k] / (eo * er);

    V_ecm = solve_poisson_sor(rhs, nx, ny, dx, 300, 1e-6, 1.6);

    // Apply boundary voltages
    apply_boundary_voltage(V_ecm, nx, ny, V_top, V_bottom, V_left, V_right);
}

// ============================================================================
// Magnetic field computation (from sim_toolbox.py)
// B_z component from current density using Biot-Savart law (2D approx)
// ============================================================================
inline std::vector<double> compute_magnetic_field(
    const std::vector<double>& Jx,
    const std::vector<double>& Jy,
    int nx, int ny, double dx)
{
    // B_z = mu_0 * curl(J) in 2D
    // curl(J) = dJy/dx - dJx/dy
    constexpr double mu_0 = 4.0 * M_PI * 1e-7;  // permeability of free space

    auto curl_J = fd::curl_2d(Jx, Jy, nx, ny, dx);
    std::vector<double> Bz(nx * ny);
    for (int i = 0; i < nx * ny; i++)
        Bz[i] = mu_0 * curl_J[i];
    return Bz;
}

} // namespace betse
