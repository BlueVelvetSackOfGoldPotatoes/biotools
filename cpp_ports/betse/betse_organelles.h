// BETSE C++ Port - Organelles Module
// Complete reimplementation of all organelle dynamics:
//   - Endoplasmic reticulum calcium stores and IP3 receptor dynamics
//   - Mitochondrial membrane potential, ETC, and calcium uniporter
//   - Nuclear transport and pore dynamics
//   - Microtubule alignment and electrophoretic forces
// Ported from: organelles/endo_retic.py, mitochondria.py, microtubules.py
#pragma once

#include "betse_types.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

namespace betse {

// ============================================================================
// IP3 Receptor Model (from endo_retic.py - De Young-Bhalla 1998 style)
// Models Ca2+-induced Ca2+ release (CICR) gated by IP3 and Ca2+
// ============================================================================
struct IP3Receptor {
    // IP3R state variables per cell
    std::vector<double> x_act;   // activation by IP3 (fraction open)
    std::vector<double> x_inh;   // inactivation by Ca2+ (fraction not inhibited)
    std::vector<double> P_open;  // open probability

    // Parameters
    double d1 = 0.13e-3;    // IP3 dissociation constant [mM]
    double d2 = 1.049e-3;   // Ca2+ inhibition dissociation constant [mM]
    double d3 = 0.9434e-3;  // IP3 binding with Ca2+ inhibited [mM]
    double d5 = 0.08234e-3; // Ca2+ activation dissociation constant [mM]
    double a2 = 0.2;        // Ca2+ inhibition binding rate [1/(mM*s)]

    void init(int nc) {
        x_act.assign(nc, 0.0);
        x_inh.assign(nc, 1.0);
        P_open.assign(nc, 0.0);
    }

    void update(int nc, const std::vector<double>& ca_cell,
                const std::vector<double>& ip3_cell, double dt) {
        for (int c = 0; c < nc; c++) {
            double ca = std::max(ca_cell[c], 1e-15);
            double ip3 = std::max(ip3_cell[c], 1e-15);

            // IP3 activation
            double m_inf = ip3 / (ip3 + d1);
            // Ca2+ activation
            double n_inf = ca / (ca + d5);

            // Ca2+ inactivation (slow)
            double h_inf = d2 / (d2 + ca);
            double tau_h = 1.0 / (a2 * (d2 + ca));

            // Semi-implicit update of inactivation
            x_inh[c] = (tau_h * x_inh[c] + dt * h_inf) / (tau_h + dt);

            // Open probability: m^3 * n^3 * h^3
            double m3 = m_inf * m_inf * m_inf;
            double n3 = n_inf * n_inf * n_inf;
            double h3 = x_inh[c] * x_inh[c] * x_inh[c];
            P_open[c] = m3 * n3 * h3;
        }
    }
};

// ============================================================================
// Extended Endoplasmic Reticulum (from organelles/endo_retic.py)
// Full ER model with IP3R, RyR, SERCA, and leak channels
// ============================================================================
struct EndoReticulumFull {
    // Per-cell state
    std::vector<double> er_vol;     // ER volume [m^3]
    std::vector<double> er_sa;      // ER surface area [m^2]
    std::vector<double> Ver;        // ER transmembrane voltage [V]
    std::vector<double> Q;          // ER charge [C]

    // Per-ion, per-cell ER concentrations
    std::vector<std::vector<double>> cc_er;      // [ion][cell]
    std::vector<std::vector<double>> Dm_er;      // [ion][cell] ER membrane permeability
    std::vector<std::vector<double>> Dm_er_base; // base (non-gated) permeability

    // Channel/receptor models
    IP3Receptor ip3r;
    std::vector<double> ryr_open;   // RyR open probability per cell

    // Ryanodine receptor parameters
    double ryr_Km_ca = 0.001;       // Ca half-activation for RyR [mM]
    double ryr_n = 3;               // Hill coefficient for RyR
    double ryr_max_perm = 1.0e-14;  // Max RyR permeability

    // Leak channel
    double er_leak_Dm = 1.0e-19;

    // SERCA pump
    double serca_max_rate = 1.0e-5;
    double serca_Km_Ca = 0.0005;
    double serca_Km_ATP = 0.5;

    // IP3R parameters
    double ip3r_max_perm = 1.0e-14;

    void init(int nc, int ni, double vol_frac, double sa_frac,
              const std::vector<double>& cell_vol,
              const std::vector<double>& cell_sa,
              const std::vector<std::vector<double>>& cc_cells,
              double cm, double init_Ca, int iCa) {
        er_vol.resize(nc);
        er_sa.resize(nc);
        Ver.assign(nc, 0.0);
        Q.assign(nc, 0.0);
        cc_er.resize(ni, std::vector<double>(nc));
        Dm_er.resize(ni, std::vector<double>(nc, er_leak_Dm));
        Dm_er_base.resize(ni, std::vector<double>(nc, er_leak_Dm));
        ryr_open.assign(nc, 0.0);

        for (int c = 0; c < nc; c++) {
            er_vol[c] = vol_frac * cell_vol[c];
            er_sa[c] = sa_frac * cell_sa[c];
        }
        for (int ion = 0; ion < ni; ion++)
            for (int c = 0; c < nc; c++)
                cc_er[ion][c] = cc_cells[ion][c];
        for (int c = 0; c < nc; c++)
            cc_er[iCa][c] = init_Ca;

        ip3r.init(nc);
    }

    // Update ER voltage from charge
    void update_voltage(int nc, int ni, double cm,
                        const std::vector<int>& zs) {
        for (int c = 0; c < nc; c++) {
            double q = 0;
            for (int ion = 0; ion < ni; ion++)
                q += zs[ion] * F_FARADAY * cc_er[ion][c];
            Q[c] = q * er_vol[c];
            double cap = er_sa[c] * cm;
            Ver[c] = (cap > 0) ? Q[c] / cap : 0.0;
        }
    }

    // Update all ER channel permeabilities
    void update_channels(int nc, int iCa,
                         const std::vector<std::vector<double>>& cc_cells,
                         double act_Km_Ca, double act_n_Ca,
                         double inh_Km_Ca, double inh_n_Ca,
                         const std::vector<double>* ip3_cells = nullptr,
                         double dt = 1e-4) {
        int ni = (int)Dm_er.size();

        for (int c = 0; c < nc; c++) {
            double ca = std::max(cc_cells[iCa][c], 1e-15);

            // Ca-activated, Ca-inhibited channel (standard BETSE model)
            double ca_act = std::pow(ca / act_Km_Ca, act_n_Ca);
            double ca_inh = std::pow(ca / inh_Km_Ca, inh_n_Ca);
            double Dm_cicr = ip3r_max_perm * (ca_act / (1.0 + ca_act)) *
                             (1.0 / (1.0 + ca_inh));

            // IP3R gating (if IP3 is available)
            double Dm_ip3r = 0.0;
            if (ip3_cells && !ip3_cells->empty()) {
                // Use full IP3R model
                ip3r.update(nc, cc_cells[iCa], *ip3_cells, dt);
                Dm_ip3r = ip3r_max_perm * ip3r.P_open[c];
            }

            // Ryanodine receptor (RyR) - Ca-induced Ca release
            double ryr_act = std::pow(ca / ryr_Km_ca, ryr_n);
            ryr_open[c] = ryr_act / (1.0 + ryr_act);
            double Dm_ryr = ryr_max_perm * ryr_open[c];

            // Total Ca permeability
            Dm_er[iCa][c] = Dm_er_base[iCa][c] + Dm_cicr + Dm_ip3r + Dm_ryr;
        }

        // Other ions: just use base permeability
        for (int ion = 0; ion < ni; ion++) {
            if (ion == iCa) continue;
            for (int c = 0; c < nc; c++)
                Dm_er[ion][c] = Dm_er_base[ion][c];
        }
    }

    // SERCA pump flux (pumps Ca from cytoplasm into ER)
    double serca_flux(double ca_cell, double ca_er, double Ver_c,
                      double T, double deltaGATP, double cATP,
                      double cADP, double cPi) const {
        ca_cell = std::max(ca_cell, 1e-15);
        ca_er = std::max(ca_er, 1e-15);
        double Qnum = cADP * cPi * ca_cell;
        double Qden = cATP * ca_er;
        if (std::abs(Qden) < 1e-30) Qden = 1e-30;
        double Q = Qnum / Qden;
        double exp_arg = -deltaGATP / (R_GAS * T) -
                         2.0 * (F_FARADAY * Ver_c) / (R_GAS * T);
        exp_arg = std::clamp(exp_arg, -500.0, 500.0);
        double Keq = std::exp(exp_arg);
        if (std::abs(Keq) < 1e-30) Keq = 1e-30;
        double numo = (ca_cell / serca_Km_Ca) * (cATP / serca_Km_ATP);
        double deno = (1.0 + ca_cell / serca_Km_Ca) * (1.0 + cATP / serca_Km_ATP);
        double fwd = numo / deno;
        double result = serca_max_rate * fwd * (1.0 - Q / Keq);
        return std::isfinite(result) ? result : 0.0;
    }

    // Full ER timestep
    void step(int nc, int ni, int iCa,
              std::vector<std::vector<double>>& cc_cells,
              const std::vector<double>& cell_vol,
              const std::vector<int>& zs,
              double T, double cm, double tm, double dt,
              double deltaGATP, double cATP, double cADP, double cPi,
              double act_Km_Ca, double act_n_Ca,
              double inh_Km_Ca, double inh_n_Ca,
              const std::vector<double>* ip3_cells = nullptr) {
        // Update channel permeabilities
        update_channels(nc, iCa, cc_cells, act_Km_Ca, act_n_Ca,
                        inh_Km_Ca, inh_n_Ca, ip3_cells, dt);

        // SERCA pump
        for (int c = 0; c < nc; c++) {
            double f = serca_flux(cc_cells[iCa][c], cc_er[iCa][c],
                                  Ver[c], T, deltaGATP, cATP, cADP, cPi);
            // Limit flux
            double max_out = cc_cells[iCa][c] * cell_vol[c] /
                             (er_sa[c] * dt + FLOAT_NONCE);
            double max_in = cc_er[iCa][c] * er_vol[c] /
                            (er_sa[c] * dt + FLOAT_NONCE);
            if (f > 0) f = std::min(f, max_out * 0.5);
            else       f = std::max(f, -max_in * 0.5);

            cc_cells[iCa][c] -= f * (er_sa[c] / cell_vol[c]) * dt;
            cc_er[iCa][c] += f * (er_sa[c] / er_vol[c]) * dt;
            cc_cells[iCa][c] = std::max(0.0, cc_cells[iCa][c]);
            cc_er[iCa][c] = std::max(0.0, cc_er[iCa][c]);
        }

        // Electrodiffusion for all ions
        for (int ion = 0; ion < ni; ion++) {
            int z = zs[ion];
            for (int c = 0; c < nc; c++) {
                double cA = std::max(cc_cells[ion][c], 0.0);
                double cB = std::max(cc_er[ion][c], 0.0);
                double f = electroflux(cA, cB, Dm_er[ion][c], tm, z, Ver[c], T);
                if (!std::isfinite(f)) f = 0.0;
                double max_out = cA * cell_vol[c] /
                                 (er_sa[c] * dt + FLOAT_NONCE);
                double max_in = cB * er_vol[c] /
                                (er_sa[c] * dt + FLOAT_NONCE);
                if (f > 0) f = std::min(f, max_out * 0.5);
                else       f = std::max(f, -max_in * 0.5);
                cc_cells[ion][c] -= f * (er_sa[c] / cell_vol[c]) * dt;
                cc_er[ion][c] += f * (er_sa[c] / er_vol[c]) * dt;
                cc_cells[ion][c] = std::max(0.0, cc_cells[ion][c]);
                cc_er[ion][c] = std::max(0.0, cc_er[ion][c]);
            }
        }

        // Update ER voltage
        update_voltage(nc, ni, cm, zs);
    }
};

// ============================================================================
// Extended Mitochondria (from organelles/mitochondria.py)
// Full mitochondrial model with ETC, membrane potential, Ca uniporter
// ============================================================================
struct MitochondriaFull {
    std::vector<double> mit_vol;    // [m^3]
    std::vector<double> mit_sa;     // [m^2]
    std::vector<double> Vmit;       // mitochondrial membrane voltage [V]
    std::vector<double> Q;          // charge [C]
    std::vector<std::vector<double>> cc_mit;   // [ion][cell]
    std::vector<std::vector<double>> Dm_mit;   // [ion][cell]

    // ETC (electron transport chain) parameters
    double extra_rho = 0.0;         // extra fixed charge in mito matrix
    double proton_pump_rate = 1.0e-7;  // ETC proton pump rate [mol/(m^2*s)]

    // Calcium uniporter
    double mcu_max = 1.0e-16;      // max Ca uniporter permeability [m/s]
    double mcu_Km_Ca = 0.001;      // Ca half-activation [mM]
    double mcu_n = 2;              // Hill coefficient

    // NCLX (Na/Ca exchanger in mito)
    double nclx_rate = 1.0e-7;     // NCLX exchange rate
    double nclx_Km_Ca = 0.001;
    double nclx_Km_Na = 10.0;

    void init(int nc, int ni, double vol_frac, double sa_frac,
              const std::vector<double>& cell_vol,
              const std::vector<double>& cell_sa,
              const std::vector<std::vector<double>>& cc_cells) {
        mit_vol.resize(nc);
        mit_sa.resize(nc);
        Vmit.assign(nc, -0.18);  // typical mito membrane potential ~ -180 mV
        Q.assign(nc, 0.0);
        cc_mit.resize(ni, std::vector<double>(nc));
        Dm_mit.resize(ni, std::vector<double>(nc, 1.0e-19));

        for (int c = 0; c < nc; c++) {
            mit_vol[c] = vol_frac * cell_vol[c];
            mit_sa[c] = sa_frac * cell_sa[c];
        }
        for (int ion = 0; ion < ni; ion++)
            for (int c = 0; c < nc; c++)
                cc_mit[ion][c] = cc_cells[ion][c];
    }

    void update_voltage(int nc, int ni, double cm,
                        const std::vector<int>& zs) {
        for (int c = 0; c < nc; c++) {
            double q = 0;
            for (int ion = 0; ion < ni; ion++)
                q += zs[ion] * F_FARADAY * cc_mit[ion][c] * mit_vol[c];
            Q[c] = q + extra_rho * mit_vol[c];
            double cap = mit_sa[c] * cm;
            Vmit[c] = (cap > 0) ? Q[c] / cap : 0.0;
        }
    }

    // Mitochondrial calcium uniporter (MCU)
    double mcu_flux(double ca_cell, double ca_mit, double Vmit_c, double T) const {
        double ca_act = std::pow(ca_cell / mcu_Km_Ca, mcu_n);
        double P_open = ca_act / (1.0 + ca_act);
        return electroflux(ca_cell, ca_mit, mcu_max * P_open,
                           1e-8, 2, Vmit_c, T);
    }

    // ETC proton pumping (simplified)
    double etc_proton_pump(double pH_mit, double Vmit_c, double T) const {
        // Simplified: pump rate depends on driving force
        double driving = 1.0 / (1.0 + std::exp(10.0 * (Vmit_c + 0.2)));
        return proton_pump_rate * driving;
    }

    // Full mitochondria timestep
    void step(int nc, int ni, int iCa, int iH,
              std::vector<std::vector<double>>& cc_cells,
              const std::vector<double>& cell_vol,
              const std::vector<int>& zs,
              double T, double cm, double tm, double dt) {
        // MCU: Ca uptake into mitochondria
        for (int c = 0; c < nc; c++) {
            double f_mcu = mcu_flux(cc_cells[iCa][c], cc_mit[iCa][c],
                                    Vmit[c], T);
            if (!std::isfinite(f_mcu)) f_mcu = 0.0;
            double max_out = cc_cells[iCa][c] * cell_vol[c] /
                             (mit_sa[c] * dt + FLOAT_NONCE);
            if (f_mcu > 0) f_mcu = std::min(f_mcu, max_out * 0.5);

            cc_cells[iCa][c] -= f_mcu * (mit_sa[c] / cell_vol[c]) * dt;
            cc_mit[iCa][c] += f_mcu * (mit_sa[c] / mit_vol[c]) * dt;
            cc_cells[iCa][c] = std::max(0.0, cc_cells[iCa][c]);
            cc_mit[iCa][c] = std::max(0.0, cc_mit[iCa][c]);
        }

        // ETC proton pumping (H+ out of matrix)
        if (iH >= 0 && iH < ni) {
            for (int c = 0; c < nc; c++) {
                double f_etc = etc_proton_pump(cc_mit[iH][c], Vmit[c], T);
                cc_mit[iH][c] -= f_etc * (mit_sa[c] / mit_vol[c]) * dt;
                cc_cells[iH][c] += f_etc * (mit_sa[c] / cell_vol[c]) * dt;
                cc_mit[iH][c] = std::max(0.0, cc_mit[iH][c]);
            }
        }

        // General electrodiffusion for all ions
        for (int ion = 0; ion < ni; ion++) {
            if (ion == iCa) continue; // MCU handles Ca
            int z = zs[ion];
            for (int c = 0; c < nc; c++) {
                double cA = std::max(cc_cells[ion][c], 0.0);
                double cB = std::max(cc_mit[ion][c], 0.0);
                double f = electroflux(cA, cB, Dm_mit[ion][c], tm, z,
                                       Vmit[c], T);
                if (!std::isfinite(f)) f = 0.0;
                double max_c = cA * cell_vol[c] /
                               (mit_sa[c] * dt + FLOAT_NONCE);
                double max_m = cB * mit_vol[c] /
                               (mit_sa[c] * dt + FLOAT_NONCE);
                if (f > 0) f = std::min(f, max_c * 0.5);
                else       f = std::max(f, -max_m * 0.5);
                cc_cells[ion][c] -= f * (mit_sa[c] / cell_vol[c]) * dt;
                cc_mit[ion][c] += f * (mit_sa[c] / mit_vol[c]) * dt;
                cc_cells[ion][c] = std::max(0.0, cc_cells[ion][c]);
                cc_mit[ion][c] = std::max(0.0, cc_mit[ion][c]);
            }
        }

        update_voltage(nc, ni, cm, zs);
    }
};

// ============================================================================
// Nuclear Transport (simplified nuclear pore complex model)
// Passive diffusion through nuclear pores for signaling molecules
// ============================================================================
struct NuclearTransport {
    std::vector<double> nuc_vol;     // nuclear volume per cell [m^3]
    std::vector<double> nuc_sa;      // nuclear surface area [m^2]
    double npc_perm = 1.0e-17;       // nuclear pore complex permeability [m/s]
    int npc_count = 2000;            // average NPCs per nucleus
    double npc_radius = 5.0e-9;      // NPC channel radius [m]

    // Per-molecule nuclear concentrations
    std::map<std::string, std::vector<double>> cc_nuc;

    void init(int nc, double vol_frac,
              const std::vector<double>& cell_vol) {
        nuc_vol.resize(nc);
        nuc_sa.resize(nc);
        for (int c = 0; c < nc; c++) {
            nuc_vol[c] = vol_frac * cell_vol[c];
            // Approximate nuclear SA from volume (sphere)
            double r = std::cbrt(3.0 * nuc_vol[c] / (4.0 * M_PI));
            nuc_sa[c] = 4.0 * M_PI * r * r;
        }
    }

    // Transport a molecule between cytoplasm and nucleus
    void transport_molecule(const std::string& mol_name,
                            std::vector<double>& c_cyto,
                            int nc, double Dm_nuc, int z,
                            double Vmem_avg, double T, double dt) {
        if (cc_nuc.find(mol_name) == cc_nuc.end())
            cc_nuc[mol_name].assign(nc, 0.0);

        auto& c_nuc = cc_nuc[mol_name];
        for (int c = 0; c < nc; c++) {
            // Simple diffusion through nuclear pores
            double flux = Dm_nuc * npc_count * M_PI * npc_radius * npc_radius *
                          (c_cyto[c] - c_nuc[c]) / (npc_radius);
            c_cyto[c] -= flux * dt / (nuc_vol[c] + FLOAT_NONCE);
            c_nuc[c] += flux * dt / (nuc_vol[c] + FLOAT_NONCE);
            c_cyto[c] = std::max(0.0, c_cyto[c]);
            c_nuc[c] = std::max(0.0, c_nuc[c]);
        }
    }
};

// ============================================================================
// Extended Microtubules (from organelles/microtubules.py)
// Full model with electrophoretic drift, rotational diffusion, and
// alignment to electric fields
// ============================================================================
struct MicrotubulesFull {
    std::vector<double> mt_theta;    // orientation angle per membrane
    std::vector<double> mtubes_x;    // x-component of MT direction
    std::vector<double> mtubes_y;    // y-component of MT direction
    std::vector<double> mtdf;        // MT density function
    std::vector<double> umtn;        // normal component at membranes
    std::vector<double> uxmt;        // x at cell centres
    std::vector<double> uymt;        // y at cell centres
    std::vector<double> L;           // length per membrane
    std::vector<double> charge_mt;   // charge per MT
    std::vector<double> drag_r;      // rotational drag coefficient
    std::vector<double> Dr;          // rotational diffusion coefficient
    std::vector<double> P_dipole;    // dipole moment per MT
    double mt_density = 1.0;

    void init(int nm, int nc, double mt_radius,
              const std::vector<double>& R_rads,
              double mean_R,
              double T, double length_charge, double tubulin_dipole,
              double cytoplasm_viscosity) {
        mt_theta.assign(nm, 0.0);
        mtubes_x.assign(nm, 0.0);
        mtubes_y.assign(nm, 0.0);
        mtdf.assign(nm, 0.0);
        umtn.assign(nm, 0.0);
        uxmt.assign(nc, 0.0);
        uymt.assign(nc, 0.0);
        L.resize(nm);
        charge_mt.resize(nm);
        drag_r.resize(nm);
        Dr.resize(nm);
        P_dipole.resize(nm);

        for (int m = 0; m < nm; m++) {
            int ci = m % nc;
            L[m] = mt_radius * (R_rads[ci] / mean_R);
            charge_mt[m] = (length_charge * Q_ELECTRON * L[m]) / 1.0e-6;

            // Tubulin count and dipole
            double tubulin_N = (222.0 / 1.0e-6) * L[m];
            P_dipole[m] = tubulin_dipole * 3.336e-30 * tubulin_N; // Debye -> C*m

            // Drag (Broersma 1960)
            double ratio = L[m] / mt_radius;
            double ln_r = std::log(ratio);
            double v = 1.0 / ln_r;
            double g_rad = -0.446 - 0.2 * v - 16.0 * v * v +
                           63.0 * v * v * v - 62.0 * v * v * v * v;
            double C_rad = (1.0 / 3.0) * M_PI /
                           (std::log(L[m] / (2.0 * mt_radius)) + g_rad);
            drag_r[m] = C_rad * cytoplasm_viscosity * L[m] * L[m] * L[m];
            Dr[m] = (K_BOLTZMANN * T) / drag_r[m];
        }
    }

    void update(int nm, int nc,
                const std::vector<double>& E_cell_x,
                const std::vector<double>& E_cell_y,
                const std::vector<int>& mem_to_cell,
                const std::vector<double>& mem_nx,
                const std::vector<double>& mem_ny,
                const std::vector<double>& mem_sa,
                const std::vector<double>& cell_sa,
                double dt, double dilate_dt, double T,
                bool tethered, std::mt19937& rng) {
        if (dilate_dt <= 0) return;

        for (int m = 0; m < nm; m++) {
            int c = mem_to_cell[m];
            double Ex = E_cell_x[c];
            double Ey = E_cell_y[c];
            double ui = std::cos(mt_theta[m]) * L[m];
            double vi = std::sin(mt_theta[m]) * L[m];
            double q = charge_mt[m];

            // Torque from electric field on charged microtubule
            double torque;
            if (tethered) {
                torque = q * ui * Ey - q * vi * Ex;
            } else {
                torque = q * ui * Ex + q * vi * Ey;
            }

            // Add dipole alignment torque
            torque += P_dipole[m] * (Ex * std::sin(mt_theta[m]) -
                                     Ey * std::cos(mt_theta[m]));

            double flux_theta = torque / drag_r[m];

            // Rotational Brownian noise
            double stdev = std::sqrt(2.0 * dt * dilate_dt *
                                     K_BOLTZMANN * T / drag_r[m] *
                                     L[m] * L[m]);
            std::normal_distribution<double> noise(0.0, stdev);

            mt_theta[m] += flux_theta * dt * dilate_dt + noise(rng);

            // Normalize angle
            while (mt_theta[m] > M_PI) mt_theta[m] -= 2.0 * M_PI;
            while (mt_theta[m] < -M_PI) mt_theta[m] += 2.0 * M_PI;

            mtubes_x[m] = std::cos(mt_theta[m]) * mt_density;
            mtubes_y[m] = std::sin(mt_theta[m]) * mt_density;
        }

        // Average to cell centres
        std::fill(uxmt.begin(), uxmt.end(), 0.0);
        std::fill(uymt.begin(), uymt.end(), 0.0);
        std::vector<int> count(nc, 0);
        for (int m = 0; m < nm; m++) {
            int c = mem_to_cell[m];
            uxmt[c] += mtubes_x[m];
            uymt[c] += mtubes_y[m];
            count[c]++;
        }
        for (int c = 0; c < nc; c++) {
            if (count[c] > 0) {
                uxmt[c] /= count[c];
                uymt[c] /= count[c];
            }
        }

        // Normal component at membranes
        for (int m = 0; m < nm; m++) {
            int c = mem_to_cell[m];
            umtn[m] = uxmt[c] * mem_nx[m] + uymt[c] * mem_ny[m];
        }

        // Density function (|MT| at membranes)
        for (int m = 0; m < nm; m++) {
            mtdf[m] = std::sqrt(mtubes_x[m] * mtubes_x[m] +
                                mtubes_y[m] * mtubes_y[m]);
        }
    }
};

} // namespace betse
