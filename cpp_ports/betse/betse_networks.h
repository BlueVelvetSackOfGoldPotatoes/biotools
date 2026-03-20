// BETSE C++ Port - Networks Module
// Full GRN with channels, transporters, reactions, molecule transport,
// generic active pumps, facilitated transporters, and molecule movers.
// Ported from: chemistry/networks.py, molecules.py, gene.py, sim_toolbox.py
#pragma once

#include "betse_types.h"
#include "betse_channels.h"
#include "betse_math.h"
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include <functional>

namespace betse {

// ============================================================================
// Extended Molecule (from chemistry/networks.py Molecule class)
// Full molecule with environment transport, GJ diffusion, boundary conditions
// ============================================================================
struct MoleculeExt {
    std::string name;
    std::vector<double> c_cells;     // concentration in cells [mM]
    std::vector<double> cc_at_mem;   // concentration at membranes
    std::vector<double> c_env;       // concentration in environment
    std::vector<double> c_mit;       // concentration in mitochondria
    std::vector<double> flux_intra;  // intracellular flux storage
    double c_bound = 0.0;           // boundary concentration
    double D_free = 1.0e-10;        // free diffusion [m^2/s]
    double D_mem = 0.0;             // membrane permeability [m/s]
    double D_gj = 1.0e-12;          // gap junction diffusion
    double Ftj = 1.0;               // tight junction diffusion factor
    double decay_rate = 0.0;        // first-order decay [1/s]
    double growth_rate = 0.0;       // zero-order production [mM/s]
    int z = 0;                      // charge
    double initial_cell = 0.0;
    double initial_env = 0.0;
    double initial_mit = 0.0;
    bool use_time_dilation = false;
    double time_dilation_factor = 1.0;
    bool mit_enabled = false;
    bool transmem = false;          // has transmembrane transport?
    double mu_mem = 0.0;            // electrophoretic mobility in membrane

    // Modulator: spatial modulation pattern (lambda returning scale factor)
    std::vector<double> init_asym;  // initial asymmetry pattern
};

// ============================================================================
// Reaction types (from chemistry/networks.py)
// ============================================================================
struct ReactionExt {
    std::string name;
    std::vector<std::string> reactants;
    std::vector<std::string> products;
    std::vector<double> reactant_stoich;
    std::vector<double> product_stoich;
    double rate_const = 0.0;
    double Km = 1.0;
    int reaction_zone = 0;  // 0=cell, 1=env, 2=membrane, 3=mitochondria
    double max_rate = 1e10; // upper bound on rate
    bool reversible = false;
    double Keq = 1.0;       // equilibrium constant for reversible reactions
};

// ============================================================================
// Transporter (from chemistry/networks.py)
// Active transport pump or facilitated transporter for molecules
// ============================================================================
struct Transporter {
    std::string name;
    std::string substrate;       // molecule name being transported
    double alpha_max = 1.0e-8;  // maximum rate [mol/(m^2*s)]
    double Km = 1.0;            // Michaelis constant [mM]
    double Km_ATP = 0.5;        // ATP half-saturation
    bool pump_into_cell = false; // direction
    bool uses_ATP = true;        // active (ATP) vs facilitated
    double Keq = 1.0;           // equilibrium constant (for facilitated)
    int n = 1;                   // Hill coefficient
    int z = 0;                   // charge of transported species
    std::vector<double> rho;    // spatial density modulator
};

// ============================================================================
// Network Channel (from chemistry/networks.py)
// Voltage-gated channel defined as part of the GRN
// ============================================================================
struct NetworkChannel {
    std::string name;
    ChannelType type;
    double max_Dm = 1.0e-15;
    ChannelState channel_state;
    std::vector<int> target_mems;
    // Modulator: e.g., by a GRN molecule
    std::string modulator_substance;
    double modulator_Km = 1.0;
    int modulator_n = 1;
    bool modulator_activates = true;
};

// ============================================================================
// Modulator entry (from chemistry/networks.py)
// Allows substances to modulate membrane permeability
// ============================================================================
struct MembraneModulator {
    std::string name;
    std::string substance;    // which molecule modulates
    std::string target_ion;   // which ion Dm to affect (Na, K, Ca, etc.)
    double Km = 1.0;
    int n = 1;
    double max_effect = 1.0;  // maximum fold-change in Dm
    bool activates = true;    // true = increase Dm, false = decrease
};

// ============================================================================
// Master of Networks (from chemistry/networks.py MasterOfNetworks)
// Full gene regulatory network with molecules, reactions, transporters,
// channels, and modulators
// ============================================================================
struct MasterOfNetworks {
    std::map<std::string, MoleculeExt> molecules;
    std::vector<ReactionExt> reactions;
    std::vector<ReactionExt> reactions_env;   // environment-only reactions
    std::vector<ReactionExt> reactions_mit;   // mitochondrial reactions
    std::vector<Transporter> transporters;
    std::vector<NetworkChannel> channels;
    std::vector<MembraneModulator> modulators;
    bool enabled = false;

    // Extra charge contributions from network molecules
    std::vector<double> extra_rho_cells;
    std::vector<double> extra_J_mem;

    void init(int nc, int nm, int ne) {
        for (auto& [name, mol] : molecules) {
            mol.c_cells.assign(nc, mol.initial_cell);
            if (!mol.init_asym.empty() && (int)mol.init_asym.size() == nc)
                for (int c = 0; c < nc; c++) mol.c_cells[c] *= mol.init_asym[c];
            mol.c_env.assign(ne, mol.initial_env);
            mol.cc_at_mem.assign(nm, mol.initial_cell);
            mol.flux_intra.assign(nm, 0.0);
            if (mol.mit_enabled)
                mol.c_mit.assign(nc, mol.initial_mit);
        }
        extra_rho_cells.assign(nc, 0.0);
        extra_J_mem.assign(nm, 0.0);
    }

    // ========================================================================
    // Molecule transport across membrane (GHK flux)
    // From sim_toolbox.py molecule_mover
    // ========================================================================
    void move_molecule(MoleculeExt& mol, int nc, int nm, double dt,
                        const std::vector<int>& mem_to_cell,
                        const std::vector<double>& mem_sa,
                        const std::vector<double>& cell_vol,
                        const std::vector<double>& Vmem,
                        double T, double tm,
                        const std::vector<double>& gjopen,
                        const std::vector<int>& neighbor_mem_id,
                        const std::vector<double>& gj_len,
                        double gj_block = 1.0) {
        // 1. Transmembrane electrodiffusion (Goldman flux)
        if (mol.D_mem > 0.0) {
            for (int m = 0; m < nm; m++) {
                int c = mem_to_cell[m];
                double cenv = mol.c_env[c]; // simplified: env by cell index
                double cmem = mol.c_cells[c];
                double flux = electroflux(cenv, cmem, mol.D_mem, tm,
                                          mol.z, Vmem[m], T);
                // Update concentrations
                double delta_cell = flux * mem_sa[m] / cell_vol[c];
                mol.c_cells[c] += delta_cell * dt * mol.time_dilation_factor;
            }
            for (int c = 0; c < nc; c++)
                mol.c_cells[c] = std::max(0.0, mol.c_cells[c]);

            // Update env (simplified: well-mixed)
            double total_flux = 0;
            for (int m = 0; m < nm; m++) {
                int c = mem_to_cell[m];
                double cenv = mol.c_env[c];
                double cmem = mol.c_cells[c];
                total_flux -= electroflux(cenv, cmem, mol.D_mem, tm,
                                          mol.z, Vmem[m], T) * mem_sa[m];
            }
            double env_vol = cell_vol[0] * nc * 10.0;
            double delta_env = total_flux / env_vol;
            for (int c = 0; c < nc; c++) {
                mol.c_env[c] += delta_env * dt * mol.time_dilation_factor;
                mol.c_env[c] = std::max(0.0, mol.c_env[c]);
            }
        }

        // 2. Gap junction transport
        if (mol.D_gj > 0.0) {
            for (int m = 0; m < nm; m++) {
                int nn = neighbor_mem_id[m];
                if (nn < 0) continue;
                int c1 = mem_to_cell[m];
                int c2 = mem_to_cell[nn];
                double fgj = electroflux(mol.c_cells[c1], mol.c_cells[c2],
                                          mol.D_gj * gj_block * gjopen[m],
                                          gj_len[m], mol.z,
                                          Vmem[nn] - Vmem[m], T);
                mol.c_cells[c1] -= fgj * mem_sa[m] / cell_vol[c1] * dt * mol.time_dilation_factor;
                mol.c_cells[c1] = std::max(0.0, mol.c_cells[c1]);
            }
        }

        // 3. Update membrane concentrations
        for (int m = 0; m < nm; m++)
            mol.cc_at_mem[m] = mol.c_cells[mem_to_cell[m]];
    }

    // ========================================================================
    // Generic active pump (sim_toolbox.py molecule_pump)
    // ========================================================================
    void run_pump(Transporter& tp, MoleculeExt& mol, int nc, int nm, double dt,
                   const std::vector<int>& mem_to_cell,
                   const std::vector<double>& mem_sa,
                   const std::vector<double>& cell_vol,
                   const std::vector<double>& Vmem,
                   double T, double tm,
                   double deltaGATP, double cATP, double cADP, double cPi) {
        for (int m = 0; m < nm; m++) {
            int c = mem_to_cell[m];
            double cX_cell = mol.c_cells[c];
            double cX_env = mol.c_env[c];
            double f_X = 0.0;
            double rho = (m < (int)tp.rho.size()) ? tp.rho[m] : 1.0;

            if (tp.uses_ATP) {
                // ATP-driven pump
                double Qnum, Qden;
                if (tp.pump_into_cell) {
                    Qnum = cADP * cPi * cX_cell;
                    Qden = cATP * (cX_env + 1e-30);
                } else {
                    Qnum = cADP * cPi * cX_env;
                    Qden = cATP * (cX_cell + 1e-30);
                }
                if (std::abs(Qden) < 1e-30) Qden = 1e-30;
                double Q = Qnum / Qden;
                double exp_arg = -(deltaGATP / (R_GAS * T));
                if (!tp.pump_into_cell)
                    exp_arg += (tp.z * F_FARADAY * Vmem[m]) / (R_GAS * T);
                else
                    exp_arg -= (tp.z * F_FARADAY * Vmem[m]) / (R_GAS * T);
                exp_arg = std::clamp(exp_arg, -500.0, 500.0);
                double Keq = std::exp(exp_arg);
                if (std::abs(Keq) < 1e-30) Keq = 1e-30;

                double numo_E, deno_E;
                if (tp.pump_into_cell) {
                    numo_E = std::pow(cX_env / tp.Km, tp.n) * (cATP / tp.Km_ATP);
                    deno_E = (1.0 + std::pow(cX_env / tp.Km, tp.n)) * (1.0 + cATP / tp.Km_ATP);
                    f_X = rho * tp.alpha_max * (numo_E / deno_E) * (1.0 - Q / Keq);
                } else {
                    numo_E = std::pow(cX_cell / tp.Km, tp.n) * (cATP / tp.Km_ATP);
                    deno_E = (1.0 + std::pow(cX_cell / tp.Km, tp.n)) * (1.0 + cATP / tp.Km_ATP);
                    f_X = -rho * tp.alpha_max * (numo_E / deno_E) * (1.0 - Q / Keq);
                }
            } else {
                // Facilitated transporter (no ATP)
                double Qnum, Qden;
                if (tp.pump_into_cell) {
                    Qnum = cX_cell;
                    Qden = cX_env + 1e-30;
                } else {
                    Qnum = cX_env;
                    Qden = cX_cell + 1e-30;
                }
                double Q = Qnum / Qden;
                double Keq_mod = tp.Keq;
                if (tp.z != 0)
                    Keq_mod *= std::exp(tp.z * F_FARADAY * Vmem[m] / (R_GAS * T));

                double numo_E, deno_E;
                if (tp.pump_into_cell) {
                    numo_E = cX_env / tp.Km;
                    deno_E = 1.0 + cX_env / tp.Km;
                    f_X = rho * tp.alpha_max * (numo_E / deno_E) * (1.0 - Q / Keq_mod);
                } else {
                    numo_E = cX_cell / tp.Km;
                    deno_E = 1.0 + cX_cell / tp.Km;
                    f_X = -rho * tp.alpha_max * (numo_E / deno_E) * (1.0 - Q / Keq_mod);
                }
            }

            // Apply flux
            mol.c_cells[c] += f_X * mem_sa[m] / cell_vol[c] * dt;
            mol.c_cells[c] = std::max(0.0, mol.c_cells[c]);
        }
    }

    // ========================================================================
    // Reaction step (cell reactions)
    // ========================================================================
    void step_reactions(double dt, int nc) {
        for (auto& rxn : reactions) {
            for (int c = 0; c < nc; c++) {
                // Compute rate
                double rate = rxn.rate_const;
                for (size_t r = 0; r < rxn.reactants.size(); r++) {
                    auto it = molecules.find(rxn.reactants[r]);
                    if (it != molecules.end()) {
                        double conc = it->second.c_cells[c];
                        if (rxn.reactants.size() == 1) {
                            rate *= conc / (rxn.Km + conc + 1e-30);
                        } else {
                            rate *= std::pow(std::max(0.0, conc), rxn.reactant_stoich[r]);
                        }
                    }
                }
                rate = std::min(rate, rxn.max_rate);

                // Apply stoichiometry
                for (size_t r = 0; r < rxn.reactants.size(); r++) {
                    auto it = molecules.find(rxn.reactants[r]);
                    if (it != molecules.end()) {
                        it->second.c_cells[c] -= rxn.reactant_stoich[r] * rate * dt;
                        it->second.c_cells[c] = std::max(0.0, it->second.c_cells[c]);
                    }
                }
                for (size_t p = 0; p < rxn.products.size(); p++) {
                    auto it = molecules.find(rxn.products[p]);
                    if (it != molecules.end()) {
                        it->second.c_cells[c] += rxn.product_stoich[p] * rate * dt;
                    }
                }
            }
        }
    }

    // ========================================================================
    // Environment reactions
    // ========================================================================
    void step_reactions_env(double dt, int ne) {
        for (auto& rxn : reactions_env) {
            for (int e = 0; e < ne; e++) {
                double rate = rxn.rate_const;
                for (size_t r = 0; r < rxn.reactants.size(); r++) {
                    auto it = molecules.find(rxn.reactants[r]);
                    if (it != molecules.end()) {
                        double conc = (e < (int)it->second.c_env.size()) ? it->second.c_env[e] : 0.0;
                        rate *= conc / (rxn.Km + conc + 1e-30);
                    }
                }
                rate = std::min(rate, rxn.max_rate);

                for (size_t r = 0; r < rxn.reactants.size(); r++) {
                    auto it = molecules.find(rxn.reactants[r]);
                    if (it != molecules.end() && e < (int)it->second.c_env.size()) {
                        it->second.c_env[e] -= rxn.reactant_stoich[r] * rate * dt;
                        it->second.c_env[e] = std::max(0.0, it->second.c_env[e]);
                    }
                }
                for (size_t p = 0; p < rxn.products.size(); p++) {
                    auto it = molecules.find(rxn.products[p]);
                    if (it != molecules.end() && e < (int)it->second.c_env.size()) {
                        it->second.c_env[e] += rxn.product_stoich[p] * rate * dt;
                    }
                }
            }
        }
    }

    // ========================================================================
    // Growth and decay (for all molecules)
    // ========================================================================
    void step_growth_decay(double dt, int nc) {
        for (auto& [name, mol] : molecules) {
            for (int c = 0; c < nc; c++) {
                mol.c_cells[c] += (mol.growth_rate - mol.decay_rate * mol.c_cells[c]) * dt;
                mol.c_cells[c] = std::max(0.0, mol.c_cells[c]);
            }
        }
    }

    // ========================================================================
    // Apply membrane modulators
    // Returns per-ion, per-membrane Dm modification factors
    // ========================================================================
    void apply_modulators(int nm, int ni,
                           const std::vector<int>& mem_to_cell,
                           std::vector<std::vector<double>>& Dm_cells) {
        for (auto& mod : modulators) {
            auto it = molecules.find(mod.substance);
            if (it == molecules.end()) continue;

            int ion_id = -1;
            if (mod.target_ion == "Na") ion_id = 0;
            else if (mod.target_ion == "K") ion_id = 1;
            else if (mod.target_ion == "Cl") ion_id = 2;
            else if (mod.target_ion == "Ca") ion_id = 3;
            else if (mod.target_ion == "H") ion_id = 4;
            if (ion_id < 0 || ion_id >= ni) continue;

            for (int m = 0; m < nm; m++) {
                int c = mem_to_cell[m];
                double conc = it->second.c_cells[c];
                double hill = std::pow(conc / mod.Km, mod.n);
                double frac = hill / (1.0 + hill);
                double factor;
                if (mod.activates)
                    factor = 1.0 + mod.max_effect * frac;
                else
                    factor = 1.0 / (1.0 + mod.max_effect * frac);
                Dm_cells[ion_id][m] *= factor;
            }
        }
    }

    // ========================================================================
    // Compute extra charge from network molecules (for Vmem)
    // ========================================================================
    void compute_extra_charge(int nc) {
        extra_rho_cells.assign(nc, 0.0);
        for (auto& [name, mol] : molecules) {
            if (mol.z != 0) {
                for (int c = 0; c < nc; c++)
                    extra_rho_cells[c] += mol.z * F_FARADAY * mol.c_cells[c];
            }
        }
    }

    // ========================================================================
    // Full network step
    // ========================================================================
    void step(double dt, int nc, int nm, int ne,
              const std::vector<int>& mem_to_cell,
              const std::vector<double>& mem_sa,
              const std::vector<double>& cell_vol,
              const std::vector<double>& Vmem,
              double T, double tm,
              const std::vector<double>& gjopen,
              const std::vector<int>& neighbor_mem_id,
              const std::vector<double>& gj_len,
              double gj_block,
              double deltaGATP, double cATP, double cADP, double cPi) {
        if (!enabled) return;

        // Growth and decay
        step_growth_decay(dt, nc);

        // Cell reactions
        step_reactions(dt, nc);

        // Environment reactions
        step_reactions_env(dt, ne);

        // Transport molecules across membrane and through GJ
        for (auto& [name, mol] : molecules) {
            move_molecule(mol, nc, nm, dt, mem_to_cell, mem_sa, cell_vol,
                         Vmem, T, tm, gjopen, neighbor_mem_id, gj_len, gj_block);
        }

        // Active pumps/transporters
        for (auto& tp : transporters) {
            auto it = molecules.find(tp.substrate);
            if (it != molecules.end()) {
                run_pump(tp, it->second, nc, nm, dt, mem_to_cell, mem_sa,
                         cell_vol, Vmem, T, tm, deltaGATP, cATP, cADP, cPi);
            }
        }

        // Network channels are stepped in the main sim loop with regular channels

        // Update extra charge
        compute_extra_charge(nc);
    }
};

// ============================================================================
// V-ATPase proton pump (from networks.py)
// Pumps H+ out of cells, generates charge separation
// ============================================================================
inline double pump_VATP(double cHi, double cHo, double Vm, double T,
                         double alpha_V = 1.0e-6, double Km_H = 1.0e-4,
                         double deltaGATP = -54000.0,
                         double cATP = 1.5, double cADP = 0.01, double cPi = 0.5) {
    cHi = std::max(cHi, 1e-15);
    cHo = std::max(cHo, 1e-15);
    double Qnum = cADP * cPi * cHo;
    double Qden = cATP * cHi;
    if (std::abs(Qden) < 1e-30) Qden = 1e-30;
    double Q = Qnum / Qden;
    double exp_arg = -(deltaGATP / (R_GAS * T) - F_FARADAY * Vm / (R_GAS * T));
    exp_arg = std::clamp(exp_arg, -500.0, 500.0);
    double Keq = std::exp(exp_arg);
    if (std::abs(Keq) < 1e-30) Keq = 1e-30;
    double numo_E = (cHi / Km_H) * (cATP / 0.5);
    double deno_E = (1.0 + cHi / Km_H) * (1.0 + cATP / 0.5);
    double fwd = numo_E / deno_E;
    return -alpha_V * fwd * (1.0 - Q / Keq);
}

// ============================================================================
// H-K-ATPase pump (gastric proton pump)
// Exchanges H+ out for K+ in
// ============================================================================
inline void pump_HKATP(double cHi, double cKo, double cHo, double cKi,
                        double Vm, double T,
                        double alpha_HK, double Km_H, double Km_K,
                        double deltaGATP, double cATP, double cADP, double cPi,
                        double& f_H, double& f_K) {
    double Qnum = cADP * cPi * cHo * cKi;
    double Qden = cATP * cHi * cKo;
    if (std::abs(Qden) < 1e-30) Qden = 1e-30;
    double Q = Qnum / Qden;
    double exp_arg = -(deltaGATP / (R_GAS * T));
    exp_arg = std::clamp(exp_arg, -500.0, 500.0);
    double Keq = std::exp(exp_arg);
    double numo_E = (cHi / Km_H) * (cKo / Km_K) * (cATP / 0.5);
    double deno_E = (1.0 + cHi / Km_H) * (1.0 + cKo / Km_K) * (1.0 + cATP / 0.5);
    double fwd = numo_E / deno_E;
    f_H = -alpha_HK * fwd * (1.0 - Q / Keq);
    f_K = -f_H;
}

// ============================================================================
// Na-Ca exchanger (NCX) - secondary active transport
// 3 Na+ in : 1 Ca2+ out (electrogenic)
// ============================================================================
inline void pump_NCX(double cNai, double cNao, double cCai, double cCao,
                      double Vm, double T, double alpha_NCX,
                      double& f_Na, double& f_Ca) {
    double VF_RT = F_FARADAY * Vm / (R_GAS * T);
    // Simplified NCX model
    double f_fwd = std::pow(cNao / 87.5, 3) * (cCai / 0.0036) * std::exp(0.35 * VF_RT);
    double f_rev = std::pow(cNai / 87.5, 3) * (cCao / 0.0036) * std::exp(-0.65 * VF_RT);
    double denom = 1.0 + 0.1 * std::exp(-0.65 * VF_RT);
    double I_NCX = alpha_NCX * (f_fwd - f_rev) / denom;
    f_Na = 3.0 * I_NCX;  // 3 Na in
    f_Ca = -I_NCX;        // 1 Ca out
}

// ============================================================================
// NKCC co-transporter (Na-K-2Cl cotransport)
// 1 Na+ : 1 K+ : 2 Cl- into cell (electroneutral)
// ============================================================================
inline void pump_NKCC(double cNai, double cNao, double cKi, double cKo,
                       double cCli, double cClo, double alpha_NKCC,
                       double& f_Na, double& f_K, double& f_Cl) {
    // Driving force: (Na_o * K_o * Cl_o^2) - (Na_i * K_i * Cl_i^2)
    double driving = cNao * cKo * cClo * cClo - cNai * cKi * cCli * cCli;
    double flux = alpha_NKCC * driving;
    f_Na = flux;
    f_K = flux;
    f_Cl = 2.0 * flux;
}

// ============================================================================
// KCC co-transporter (K-Cl cotransport)
// 1 K+ : 1 Cl- out of cell (electroneutral)
// ============================================================================
inline void pump_KCC(double cKi, double cKo, double cCli, double cClo,
                      double alpha_KCC, double& f_K, double& f_Cl) {
    double driving = cKi * cCli - cKo * cClo;
    double flux = -alpha_KCC * driving;
    f_K = flux;
    f_Cl = flux;
}

} // namespace betse
