// BETSE C++ Port - Bioelectric Tissue Simulation Engine
// Comprehensive reimplementation of Python BETSE
// Original: https://github.com/betsee/betse
#pragma once

#include "betse_types.h"
#include "betse_channels.h"
#include "betse_math.h"
#include "betse_physics.h"
#include "betse_networks.h"
#include "betse_phase.h"
#include "betse_enums.h"
#include "betse_config.h"
#include "betse_tissue.h"
#include "betse_organelles.h"
#include "betse_export.h"
#include "betse_solver.h"
#include "betse_util.h"

namespace betse {

// ============================================================================
// Configuration (from betse/science/parameters.py)
// ============================================================================
struct SimConfig {
    // Time
    double dt              = 1.0e-4;   // time step [s]
    double total_time      = 1.0;      // total sim duration [s]
    int    sample_rate     = 100;       // save every N steps

    // Temperature
    double T               = 310.0;    // Kelvin

    // Membrane
    double cm              = 0.01;     // membrane capacitance [F/m^2]
    double tm              = 1.0e-8;   // membrane thickness [m]

    // Cell geometry
    double cell_radius     = 5.0e-6;   // typical cell radius [m]
    double cell_height     = 10.0e-6;  // tissue thickness [m]

    // Gap junctions
    double gj_surface      = 1.0e-12;  // gap junction area [m^2]
    double gj_len          = 1.0e-6;   // gap junction length [m]
    double gj_vthresh      = -40.0;    // gating voltage threshold [mV]
    double gj_min          = 0.1;      // minimum conductance fraction
    double gj_vgrad        = 10.0;     // voltage gradient for simplified gating

    // ECM
    bool   ecm_enabled     = false;
    double ecm_delta       = 2.0e-6;   // ECM grid spacing [m]

    // Cell polarizability
    double cell_polarizability = 0.0;

    // Initial concentrations [mol/m^3 = mM]
    struct IonConc {
        double cell;     // intracellular
        double env;      // extracellular
        double D_free;   // free diffusion coefficient [m^2/s]
        double D_mem;    // membrane permeability [m/s]
    };
    IonConc ions[6] = {
        {  12.0, 145.0, 1.33e-9, 1.0e-18 },  // Na
        { 140.0,   5.0, 1.96e-9, 1.0e-16 },  // K
        {   4.0, 110.0, 2.03e-9, 1.0e-18 },  // Cl
        {  0.0001, 2.0, 0.79e-9, 1.0e-20 },  // Ca
        {  6.3e-5, 4.0e-5, 9.31e-9, 1.0e-18 }, // H
        { 135.0,  10.0, 0.0,     0.0     },  // P (proteins)
    };

    // Na-K-ATPase pump (from sim_toolbox.py pumpNaKATP)
    bool   pump_enabled    = true;
    double alpha_NaK       = 1.0e-5;   // pump rate [mol/(m^2*s)]
    double deltaGATP       = -54000.0;  // ATP hydrolysis free energy [J/mol]
    double cATP            = 1.5;       // ATP concentration [mM]
    double cADP            = 0.01;      // ADP concentration [mM]
    double cPi             = 0.5;       // Pi concentration [mM]
    double KmNK_Na         = 12.0;      // Na half-sat [mM]
    double KmNK_K          = 1.0;       // K half-sat [mM]
    double KmNK_ATP        = 0.5;       // ATP half-sat [mM]

    // Ca-ATPase pump (from sim_toolbox.py pumpCaATP)
    bool   ca_dyn_enabled  = false;
    double alpha_Ca        = 1.0e-6;    // Ca pump max rate [mol/(m^2*s)]
    double KmCa_Ca         = 0.0005;    // Ca half-sat [mM]
    double KmCa_ATP        = 0.5;       // ATP half-sat for Ca pump

    // Endoplasmic reticulum (from organelles/endo_retic.py)
    bool   er_enabled      = false;
    double er_vol_frac     = 0.1;       // ER volume as fraction of cell vol
    double er_sa_frac      = 1.0;       // ER surface area as fraction of cell SA
    double serca_max       = 1.0e-5;    // SERCA pump max rate
    double max_er          = 1.0e-14;   // max ER Ca channel permeability
    double act_Km_Ca       = 0.0005;    // Ca activation Km
    double act_n_Ca        = 2.0;       // Ca activation Hill coefficient
    double inh_Km_Ca       = 0.001;     // Ca inhibition Km
    double inh_n_Ca        = 4.0;       // Ca inhibition Hill coefficient
    double act_Km_IP3      = 0.0005;    // IP3 activation Km
    double act_n_IP3       = 2.0;       // IP3 activation Hill coefficient
    double er_init_Ca      = 0.2;       // initial ER Ca [mM]

    // Mitochondria (from organelles/mitochondria.py)
    bool   mito_enabled    = false;
    double mito_vol_frac   = 0.5;       // mito volume fraction
    double mito_sa_frac    = 0.5;       // mito SA fraction

    // Noise
    bool   noise_enabled   = false;
    double noise_level     = 0.001;

    // Tissue generation
    int    num_cells       = 100;
    double tissue_radius   = 100.0e-6; // [m]

    // Physics (from physics/)
    bool   deform_enabled  = false;
    double lame_mu         = 1.0e3;    // Lame shear modulus [Pa]
    double galvanotropism  = 0.0;      // galvanotropism coupling
    bool   deform_osmo     = false;
    bool   fixed_cluster_bound = true;

    bool   flow_enabled    = false;
    double mu_water        = 1.0e-3;   // dynamic viscosity of water [Pa*s]
    double eo              = 8.854e-12; // vacuum permittivity
    double er              = 80.0;     // relative permittivity of water
    double rho             = 1000.0;   // water density [kg/m^3]
    double aquaporins      = 1.0e-2;   // aquaporin fraction

    // Osmotic pressure
    bool   osmo_enabled    = false;

    // Microtubules (from organelles/microtubules.py)
    bool   mtube_enabled       = false;
    double mt_radius           = 12.0e-9; // microtubule radius [m]
    double tubulin_dipole      = 1740.0;  // tubulin dimer dipole [Debye]
    double tubulin_polar       = 20.0;    // tubulin polarisability [A^3]
    double length_charge       = 40.0;    // charge per um of length
    double cytoplasm_viscocity = 0.02;    // cytoplasm viscosity [Pa*s]
    double dilate_mtube_dt     = 0.0;     // time dilation for mtube dynamics
    bool   tethered_tubule     = false;

    // Gene regulatory network
    bool   grn_enabled     = false;
    bool   molecules_enabled = false;

    // Voltage-gated channel configuration
    struct ChannelConfig {
        ChannelType type;
        double max_Dm;     // max membrane diffusion [m/s]
        // Region targeting: -1 = all cells
        int target_profile_id = -1;
    };
    std::vector<ChannelConfig> channels;

    // Tissue profile (heterogeneity) configuration
    struct TissueProfile {
        int id = 0;
        std::string name = "default";
        TissueProfileShape shape = TissueProfileShape::ALL;
        Vec2 center = {0, 0};
        double radius = 0;
        double width = 0, height = 0;
        // Per-ion permeability multipliers
        double Dm_multipliers[6] = {1, 1, 1, 1, 1, 1};
    };
    std::vector<TissueProfile> tissue_profiles;

    // Cutting / wounding events
    struct CutEvent {
        double time = 0;
        Vec2 center = {0, 0};
        double radius = 0;
    };
    std::vector<CutEvent> cut_events;
    double wound_close_factor = 0.01;

    // Voltage clamp / scheduled interventions
    struct VoltageClamp {
        double start_time = 0;
        double end_time = 0;
        double voltage = 0;  // [V]
        int target_profile_id = -1;
    };
    std::vector<VoltageClamp> voltage_clamps;

    // ECM boundary voltages [V] indexed by BoundarySide: TOP=0, BOTTOM=1, LEFT=2, RIGHT=3
    double bound_V[4] = {0, 0, 0, 0};

    // GJ blocking events
    struct GJBlockEvent {
        double time = 0;
        Vec2 center = {0, 0};
        double radius = 0;
        double block_factor = 0.0; // 0 = fully blocked, 1 = unblocked
    };
    std::vector<GJBlockEvent> gj_block_events;

    // Boundary voltage events (time-dependent)
    struct BoundaryVoltageEvent {
        double start_time = 0;
        double end_time = 0;
        double voltage = 0;   // [V]
        int side = 0;         // BoundarySide index
    };
    std::vector<BoundaryVoltageEvent> boundary_voltage_events;

    // Multi-phase time settings
    double init_total_time = 10.0;  // init phase duration [s]
    double init_dt = 1.0e-3;        // init phase time step
    int    init_sample_rate = 100;
};

// ============================================================================
// Membrane Structure
// ============================================================================
struct Membrane {
    int    id;
    int    cell_id;
    Vec2   midpoint;
    Vec2   normal;
    Vec2   tangent;
    double length;
    double surface_area;
    int    neighbor_mem_id = -1;
    double gj_dist = 0;
};

// ============================================================================
// Cell Structure
// ============================================================================
struct Cell {
    int    id;
    Vec2   center;
    double radius;
    double volume;
    double surface_area;
    std::vector<Vec2> vertices;
    std::vector<int>  membrane_ids;
    std::vector<int>  neighbor_ids;
    int    tissue_profile_id = 0;
};

// ============================================================================
// Gap Junction Gating (Harris et al. 1983)
// ============================================================================
struct GapJunctionGate {
    double gjopen   = 1.0;
    double lamb     = 0.0013;
    double A1       = 0.077;
    double A2       = 0.14;
    double Ao       = 0.217;

    void update(double vgj_mV, double dt_ms, double gj_min) {
        double alpha = lamb * std::exp(-A1 * (std::abs(vgj_mV) - 0.0));
        double beta_raw = lamb * std::exp(A2 * (std::abs(vgj_mV) - 0.0));
        double beta = beta_raw / (1.0 + 50.0 * beta_raw);
        gjopen = (gjopen + dt_ms * (alpha + beta * gj_min)) /
                 (1.0 + alpha * dt_ms + beta * dt_ms);
        gjopen = std::clamp(gjopen, gj_min, 1.0);
    }
};

// ============================================================================
// Tissue Mesh - Voronoi-based cell layout
// ============================================================================
class TissueMesh {
public:
    std::vector<Cell>     cells;
    std::vector<Membrane> membranes;
    std::vector<int>      mem_to_cell;

    // M_sum_mems: for each cell c, sum membrane values
    void sum_mems_to_cells(const std::vector<double>& mem_vals,
                           std::vector<double>& cell_vals) const {
        std::fill(cell_vals.begin(), cell_vals.end(), 0.0);
        for (size_t m = 0; m < membranes.size(); m++) {
            cell_vals[mem_to_cell[m]] += mem_vals[m];
        }
    }

    // Cell average: average membrane values over each cell
    void avg_mems_to_cells(const std::vector<double>& mem_vals,
                           std::vector<double>& cell_vals) const {
        std::vector<int> count(cells.size(), 0);
        std::fill(cell_vals.begin(), cell_vals.end(), 0.0);
        for (size_t m = 0; m < membranes.size(); m++) {
            int c = mem_to_cell[m];
            cell_vals[c] += mem_vals[m];
            count[c]++;
        }
        for (size_t c = 0; c < cells.size(); c++) {
            if (count[c] > 0) cell_vals[c] /= count[c];
        }
    }

    void generate(int num_cells, double tissue_radius, double cell_height, std::mt19937& rng) {
        cells.resize(num_cells);
        std::uniform_real_distribution<double> dist(-tissue_radius, tissue_radius);
        std::vector<Vec2> centers;
        double min_spacing = tissue_radius * 2.0 / std::sqrt((double)num_cells) * 0.8;

        for (int i = 0; i < num_cells; i++) {
            Vec2 p;
            bool ok = false;
            for (int attempt = 0; attempt < 1000 && !ok; attempt++) {
                p = {dist(rng), dist(rng)};
                if (p.norm() > tissue_radius) continue;
                ok = true;
                for (auto& c : centers) {
                    if ((p - c).norm() < min_spacing) { ok = false; break; }
                }
            }
            if (!ok) { p = {dist(rng), dist(rng)}; }
            centers.push_back(p);
        }

        double avg_radius = tissue_radius / std::sqrt((double)num_cells);
        int next_mem_id = 0;

        for (int i = 0; i < num_cells; i++) {
            cells[i].id = i;
            cells[i].center = centers[i];
            cells[i].radius = avg_radius;
            cells[i].volume = M_PI * avg_radius * avg_radius * cell_height;
            cells[i].surface_area = 0;

            for (int j = 0; j < num_cells; j++) {
                if (i == j) continue;
                if ((centers[i] - centers[j]).norm() < 3.0 * avg_radius)
                    cells[i].neighbor_ids.push_back(j);
            }

            int n_mems = std::max((int)cells[i].neighbor_ids.size(), 4);
            double angle_step = 2.0 * M_PI / n_mems;

            for (int k = 0; k < n_mems; k++) {
                Membrane mem;
                mem.id = next_mem_id++;
                mem.cell_id = i;
                double angle = k * angle_step;
                mem.normal = {std::cos(angle), std::sin(angle)};
                mem.tangent = {-std::sin(angle), std::cos(angle)};
                mem.midpoint = centers[i] + mem.normal * avg_radius;
                mem.length = avg_radius * angle_step;
                mem.surface_area = mem.length * cell_height;
                cells[i].surface_area += mem.surface_area;
                cells[i].membrane_ids.push_back(mem.id);
                membranes.push_back(mem);
            }
        }

        mem_to_cell.resize(membranes.size());
        for (auto& m : membranes) mem_to_cell[m.id] = m.cell_id;

        for (auto& mem : membranes) {
            double best_dist = 1e30;
            int best_id = -1;
            for (int nb : cells[mem.cell_id].neighbor_ids) {
                for (int mid : cells[nb].membrane_ids) {
                    double d = (membranes[mid].midpoint - mem.midpoint).norm();
                    if (d < best_dist) { best_dist = d; best_id = mid; }
                }
            }
            mem.neighbor_mem_id = best_id;
            mem.gj_dist = best_dist;
        }
    }

    // Remove cells (for cutting events)
    void remove_cells(const std::vector<int>& cell_ids_to_remove) {
        if (cell_ids_to_remove.empty()) return;
        std::vector<bool> remove_cell(cells.size(), false);
        for (int id : cell_ids_to_remove) {
            if (id >= 0 && id < (int)cells.size()) remove_cell[id] = true;
        }
        // Mark membranes for removal
        std::vector<bool> remove_mem(membranes.size(), false);
        for (size_t m = 0; m < membranes.size(); m++) {
            if (remove_cell[mem_to_cell[m]]) remove_mem[m] = true;
        }
        // Build new cell list and index map
        std::vector<int> old_to_new(cells.size(), -1);
        std::vector<Cell> new_cells;
        for (size_t i = 0; i < cells.size(); i++) {
            if (!remove_cell[i]) {
                old_to_new[i] = (int)new_cells.size();
                new_cells.push_back(cells[i]);
                new_cells.back().id = (int)new_cells.size() - 1;
            }
        }
        // Build new membrane list
        std::vector<int> old_mem_to_new(membranes.size(), -1);
        std::vector<Membrane> new_mems;
        for (size_t m = 0; m < membranes.size(); m++) {
            if (!remove_mem[m]) {
                old_mem_to_new[m] = (int)new_mems.size();
                Membrane nm = membranes[m];
                nm.id = (int)new_mems.size();
                nm.cell_id = old_to_new[nm.cell_id];
                new_mems.push_back(nm);
            }
        }
        // Remap neighbor/membrane references
        for (auto& c : new_cells) {
            std::vector<int> new_nb;
            for (int nb : c.neighbor_ids) {
                if (old_to_new[nb] >= 0) new_nb.push_back(old_to_new[nb]);
            }
            c.neighbor_ids = new_nb;
            std::vector<int> new_mi;
            for (int mi : c.membrane_ids) {
                if (old_mem_to_new[mi] >= 0) new_mi.push_back(old_mem_to_new[mi]);
            }
            c.membrane_ids = new_mi;
        }
        for (auto& m : new_mems) {
            if (m.neighbor_mem_id >= 0 && old_mem_to_new[m.neighbor_mem_id] >= 0)
                m.neighbor_mem_id = old_mem_to_new[m.neighbor_mem_id];
            else
                m.neighbor_mem_id = -1;
        }
        cells = std::move(new_cells);
        membranes = std::move(new_mems);
        mem_to_cell.resize(membranes.size());
        for (auto& m : membranes) mem_to_cell[m.id] = m.cell_id;
    }
};

// ============================================================================
// ECM Grid
// ============================================================================
struct ECMGrid {
    int nx = 0, ny = 0;
    double delta = 0;
    std::vector<double> X, Y;
    std::vector<int> mem_to_ecm;

    void generate(double tissue_radius, double spacing) {
        delta = spacing;
        nx = (int)(2.0 * tissue_radius / spacing) + 1;
        ny = nx;
        X.resize(nx * ny);
        Y.resize(nx * ny);
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                X[j * nx + i] = -tissue_radius + i * spacing;
                Y[j * nx + i] = -tissue_radius + j * spacing;
            }
    }

    void map_membranes(const std::vector<Membrane>& mems) {
        mem_to_ecm.resize(mems.size());
        for (size_t m = 0; m < mems.size(); m++) {
            double best = 1e30;
            int best_i = 0;
            for (int k = 0; k < nx * ny; k++) {
                double dx = mems[m].midpoint.x - X[k];
                double dy = mems[m].midpoint.y - Y[k];
                double d = dx * dx + dy * dy;
                if (d < best) { best = d; best_i = k; }
            }
            mem_to_ecm[m] = best_i;
        }
    }

    double laplacian(const std::vector<double>& f, int i, int j) const {
        int idx = j * nx + i;
        double lap = -4.0 * f[idx];
        lap += (i > 0) ? f[idx - 1] : f[idx];
        lap += (i < nx - 1) ? f[idx + 1] : f[idx];
        lap += (j > 0) ? f[idx - nx] : f[idx];
        lap += (j < ny - 1) ? f[idx + nx] : f[idx];
        return lap / (delta * delta);
    }

    Vec2 gradient(const std::vector<double>& f, int i, int j) const {
        int idx = j * nx + i;
        double gx = 0, gy = 0;
        if (i > 0 && i < nx - 1)
            gx = (f[idx + 1] - f[idx - 1]) / (2.0 * delta);
        else if (i == 0)
            gx = (f[idx + 1] - f[idx]) / delta;
        else
            gx = (f[idx] - f[idx - 1]) / delta;
        if (j > 0 && j < ny - 1)
            gy = (f[idx + nx] - f[idx - nx]) / (2.0 * delta);
        else if (j == 0)
            gy = (f[idx + nx] - f[idx]) / delta;
        else
            gy = (f[idx] - f[idx - nx]) / delta;
        return {gx, gy};
    }

    // Diffuse a quantity on the ECM grid (single step)
    void diffuse(std::vector<double>& f, double D, double dt) const {
        std::vector<double> f_new(f.size());
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                f_new[j * nx + i] = f[j * nx + i] + D * dt * laplacian(f, i, j);
            }
        }
        f = std::move(f_new);
    }
};

// ============================================================================
// Endoplasmic Reticulum (from organelles/endo_retic.py)
// ============================================================================
struct EndoReticulum {
    std::vector<double> er_vol;    // ER volume per cell
    std::vector<double> er_sa;     // ER surface area per cell
    std::vector<double> Ver;       // ER transmembrane voltage per cell
    std::vector<double> Q;         // total charge in ER per cell
    std::vector<std::vector<double>> cc_er; // [ion][cell] ER concentrations
    std::vector<std::vector<double>> Dm_er; // [ion][cell] ER membrane permeability
    std::vector<std::vector<double>> Dm_er_base;
    std::vector<std::vector<double>> Dm_channels;

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
        Dm_er.resize(ni, std::vector<double>(nc, 1.0e-18));
        Dm_er_base.resize(ni, std::vector<double>(nc, 1.0e-18));
        Dm_channels.resize(ni, std::vector<double>(nc, 0.0));

        for (int c = 0; c < nc; c++) {
            er_vol[c] = vol_frac * cell_vol[c];
            er_sa[c] = sa_frac * cell_sa[c];
        }
        // Copy cell concentrations to ER, set Ca to initial value
        for (int ion = 0; ion < ni; ion++) {
            for (int c = 0; c < nc; c++) {
                cc_er[ion][c] = cc_cells[ion][c];
            }
        }
        for (int c = 0; c < nc; c++) {
            cc_er[iCa][c] = init_Ca;
        }
    }

    void get_voltage(int nc, int ni, double cm,
                     const std::vector<int>& zs) {
        for (int c = 0; c < nc; c++) {
            double q = 0;
            for (int ion = 0; ion < ni; ion++) {
                q += zs[ion] * F_FARADAY * cc_er[ion][c];
            }
            Q[c] = q * er_vol[c];
            double cap = er_sa[c] * cm;
            Ver[c] = (cap > 0) ? Q[c] / cap : 0.0;
        }
    }

    void update_channels(int nc, int iCa,
                         const std::vector<std::vector<double>>& cc_cells,
                         double max_er_perm,
                         double act_Km_Ca, double act_n_Ca,
                         double inh_Km_Ca, double inh_n_Ca) {
        for (int c = 0; c < nc; c++) {
            double ca = cc_cells[iCa][c];
            double ca_act = std::pow(ca / act_Km_Ca, act_n_Ca);
            double ca_inh = std::pow(ca / inh_Km_Ca, inh_n_Ca);
            double Dm_mod = (ca_act / (1.0 + ca_act)) * (1.0 / (1.0 + ca_inh));
            Dm_channels[iCa][c] = max_er_perm * Dm_mod;
        }
        int ni = (int)Dm_er.size();
        for (int ion = 0; ion < ni; ion++) {
            for (int c = 0; c < nc; c++) {
                Dm_er[ion][c] = Dm_er_base[ion][c] + Dm_channels[ion][c];
            }
        }
    }
};

// ============================================================================
// Mitochondria (from organelles/mitochondria.py)
// ============================================================================
struct Mitochondria {
    std::vector<double> mit_vol;
    std::vector<double> mit_sa;
    std::vector<double> Vmit;
    std::vector<double> Q;
    std::vector<std::vector<double>> cc_mit;
    std::vector<std::vector<double>> Dm_mit;
    double extra_rho = 0.0;

    void init(int nc, int ni,
              double vol_frac, double sa_frac,
              const std::vector<double>& cell_vol,
              const std::vector<double>& cell_sa,
              const std::vector<std::vector<double>>& cc_cells) {
        mit_vol.resize(nc);
        mit_sa.resize(nc);
        Vmit.assign(nc, 0.0);
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

    void get_voltage(int nc, int ni, double cm,
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
};

// ============================================================================
// Microtubules (from organelles/microtubules.py)
// ============================================================================
struct Microtubules {
    std::vector<double> mt_theta;   // angle per membrane
    std::vector<double> mtubes_x;   // normalized x component
    std::vector<double> mtubes_y;   // normalized y component
    std::vector<double> mtdf;       // density function
    std::vector<double> umtn;       // normal component at membranes
    std::vector<double> uxmt;       // x at cell centres
    std::vector<double> uymt;       // y at cell centres
    std::vector<double> L;          // length per membrane
    std::vector<double> charge_mtube;
    std::vector<double> drag_r;
    std::vector<double> Dr;
    double mt_density = 1.0;

    void init(int nm, int nc, double mt_radius,
              const std::vector<double>& R_rads,
              double mean_R,
              const SimConfig& cfg) {
        mt_theta.resize(nm, 0.0);
        mtubes_x.resize(nm, 0.0);
        mtubes_y.resize(nm, 0.0);
        mtdf.resize(nm, 0.0);
        umtn.resize(nm, 0.0);
        uxmt.assign(nc, 0.0);
        uymt.assign(nc, 0.0);
        L.resize(nm);
        charge_mtube.resize(nm);
        drag_r.resize(nm);
        Dr.resize(nm);

        for (int m = 0; m < nm; m++) {
            // Scale L by local cell radius
            int ci = m % nc; // rough mapping
            L[m] = mt_radius * (R_rads[ci] / mean_R);
            double tubulin_N = (222.0 / 1.0e-6) * L[m];
            charge_mtube[m] = (cfg.length_charge * Q_ELECTRON * L[m]) / 1.0e-6;

            // Drag coefficients (Broersma/Hunt 1994)
            double v = 1.0 / std::log(L[m] / mt_radius);
            double g_rad = -0.446 - 0.2 * v - 16.0 * v * v + 63.0 * v * v * v - 62.0 * v * v * v * v;
            double C_rad = (1.0 / 3.0) * M_PI / (std::log(L[m] / (2.0 * mt_radius)) + g_rad);
            drag_r[m] = C_rad * cfg.cytoplasm_viscocity * L[m] * L[m] * L[m];
            Dr[m] = (K_BOLTZMANN * cfg.T) / drag_r[m];
        }
    }

    void update(int nm, const std::vector<double>& E_cell_x,
                const std::vector<double>& E_cell_y,
                const std::vector<int>& mem_to_cell,
                double dt, const SimConfig& cfg, std::mt19937& rng) {
        if (cfg.dilate_mtube_dt <= 0) return;

        for (int m = 0; m < nm; m++) {
            int c = mem_to_cell[m];
            double Ex = E_cell_x[c];
            double Ey = E_cell_y[c];
            double ui = std::cos(mt_theta[m]) * L[m];
            double vi = std::sin(mt_theta[m]) * L[m];
            double q = charge_mtube[m];

            double torque;
            if (cfg.tethered_tubule) {
                torque = q * ui * Ey - q * vi * Ex;
            } else {
                torque = q * ui * Ex + q * vi * Ey;
            }

            double flux_theta = torque / drag_r[m];
            double stdev = std::sqrt(2.0 * dt * cfg.dilate_mtube_dt *
                                      K_BOLTZMANN * cfg.T / drag_r[m] * L[m] * L[m]);
            std::normal_distribution<double> noise(0.0, stdev);

            mt_theta[m] += flux_theta * dt * cfg.dilate_mtube_dt + noise(rng);
            mtubes_x[m] = std::cos(mt_theta[m]) * mt_density;
            mtubes_y[m] = std::sin(mt_theta[m]) * mt_density;
        }
    }
};

// ============================================================================
// Molecule / GRN substance (from chemistry/networks.py)
// ============================================================================
struct Molecule {
    std::string name;
    std::vector<double> c_cells;  // concentration in cells [mM]
    std::vector<double> c_env;    // concentration in environment
    double D_free = 1.0e-10;      // free diffusion [m^2/s]
    double D_mem = 0.0;           // membrane permeability [m/s]
    double decay_rate = 0.0;      // first-order decay [1/s]
    double growth_rate = 0.0;     // zero-order production [mM/s]
    int z = 0;                    // charge
    double initial_cell = 0.0;
    double initial_env = 0.0;
};

// ============================================================================
// Reaction (for GRN, from chemistry/networks.py)
// ============================================================================
struct Reaction {
    std::string name;
    std::vector<std::string> reactants;
    std::vector<std::string> products;
    std::vector<double> reactant_stoich;
    std::vector<double> product_stoich;
    double rate_const = 0.0;    // forward rate constant
    double Km = 1.0;            // Michaelis constant
    int reaction_zone = 0;      // 0=cell, 1=env, 2=membrane
};

// ============================================================================
// Gene Regulatory Network (simplified from chemistry/networks.py)
// ============================================================================
struct GRNetwork {
    std::map<std::string, Molecule> molecules;
    std::vector<Reaction> reactions;
    bool enabled = false;

    void init(int nc, int ne) {
        for (auto& [name, mol] : molecules) {
            mol.c_cells.assign(nc, mol.initial_cell);
            mol.c_env.assign(ne, mol.initial_env);
        }
    }

    // Modulator definition: GRN substance modulating membrane Dm for an ion
    struct Modulator {
        std::string substance;
        std::string target_ion;  // "Na", "K", "Cl", "Ca", "H"
        double Km = 1.0;
        int n = 1;
        bool activates = true;
        double max_effect = 1.0;
    };
    std::vector<Modulator> modulators;

    // Simplified step: growth, decay, reactions (intracellular only)
    void step(double dt, int nc) {
        if (!enabled) return;
        // Growth and decay
        for (auto& [name, mol] : molecules) {
            for (int c = 0; c < nc; c++) {
                mol.c_cells[c] += (mol.growth_rate - mol.decay_rate * mol.c_cells[c]) * dt;
                mol.c_cells[c] = std::max(0.0, mol.c_cells[c]);
            }
        }
        // Reactions (mass action / Michaelis-Menten)
        for (auto& rxn : reactions) {
            if (rxn.reaction_zone != 0) continue; // cell reactions only
            for (int c = 0; c < nc; c++) {
                double rate = rxn.rate_const;
                for (size_t r = 0; r < rxn.reactants.size(); r++) {
                    auto it = molecules.find(rxn.reactants[r]);
                    if (it != molecules.end()) {
                        double conc = it->second.c_cells[c];
                        if (rxn.reactants.size() == 1) {
                            rate *= conc / (rxn.Km + conc);
                        } else {
                            rate *= std::pow(conc, rxn.reactant_stoich[r]);
                        }
                    }
                }
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

    // Full step with transport (membrane + GJ diffusion of GRN molecules)
    void step(double dt, int nc, int nm, int ne,
              const std::vector<int>& mem_to_cell,
              const std::vector<double>& mem_sa,
              const std::vector<double>& cell_vol,
              const std::vector<double>& Vmem,
              double T, double tm,
              const std::vector<double>& gjopen,
              const std::vector<int>& neighbor_mem_id,
              const std::vector<double>& gj_len,
              double /*gj_block_global*/,
              double /*deltaGATP*/, double /*cATP*/,
              double /*cADP*/, double /*cPi*/) {
        if (!enabled) return;

        // Intracellular reactions
        step(dt, nc);

        // Transmembrane transport of GRN molecules (electrodiffusion)
        for (auto& [name, mol] : molecules) {
            if (mol.D_mem <= 0) continue;
            for (int m = 0; m < nm; m++) {
                int c = mem_to_cell[m];
                double flux = electroflux(
                    mol.c_env.empty() ? 0.0 : mol.c_env[c < (int)mol.c_env.size() ? c : 0],
                    mol.c_cells[c],
                    mol.D_mem, tm, mol.z, Vmem[m], T);
                double dC = flux * mem_sa[m] / cell_vol[c] * dt;
                mol.c_cells[c] += dC;
                mol.c_cells[c] = std::max(0.0, mol.c_cells[c]);
            }
        }

        // GJ transport of GRN molecules
        for (auto& [name, mol] : molecules) {
            if (mol.D_free <= 0) continue;
            for (int m = 0; m < nm; m++) {
                int nn = neighbor_mem_id[m];
                if (nn < 0) continue;
                int c1 = mem_to_cell[m];
                int c2 = mem_to_cell[nn];
                double D_gj = mol.D_free * gjopen[m];
                double flux = electroflux(
                    mol.c_cells[c1], mol.c_cells[c2],
                    D_gj, gj_len[m], mol.z,
                    Vmem[nn] - Vmem[m], T);
                double dC = flux * mem_sa[m] / cell_vol[c1] * dt;
                mol.c_cells[c1] += dC;
                mol.c_cells[c1] = std::max(0.0, mol.c_cells[c1]);
            }
        }
    }

    // Apply modulators: update Dm_cells[ion][membrane] factors from GRN substances
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
};

// ============================================================================
// Simulation State
// ============================================================================
struct SimState {
    int num_cells;
    int num_mems;
    int num_ions;

    // Per-ion, per-cell concentrations [mol/m^3]
    std::vector<std::vector<double>> cc_cells;   // [ion][cell]
    std::vector<std::vector<double>> cc_env;     // [ion][cell or ecm_point]

    // Per-ion, per-membrane concentrations
    std::vector<std::vector<double>> cc_mem;     // [ion][membrane]

    // Membrane voltage [V]
    std::vector<double> Vmem;

    // Charge densities
    std::vector<double> rho_cells;

    // Gap junction gating
    std::vector<GapJunctionGate> gj_gates;

    // Per-ion fluxes
    std::vector<std::vector<double>> flux_mem;   // [ion][membrane] transmembrane
    std::vector<std::vector<double>> flux_gj;    // [ion][membrane] gap junction

    // Pump fluxes
    std::vector<double> pump_Na;
    std::vector<double> pump_K;

    // Electric field (cell-centred)
    std::vector<double> E_cell_x;
    std::vector<double> E_cell_y;

    // Osmotic pressure
    std::vector<double> osmo_P_cell;
    std::vector<double> osmo_P_env;
    std::vector<double> P_cells;     // hydrostatic pressure

    // Deformation
    std::vector<double> d_cells_x;
    std::vector<double> d_cells_y;

    // Flow velocity
    std::vector<double> u_cells_x;
    std::vector<double> u_cells_y;

    // Time tracking
    double time = 0.0;
    int    step = 0;

    // ECM state
    std::vector<std::vector<double>> cc_ecm;
    std::vector<double> V_ecm;

    // Wound tracking
    std::vector<bool> wound_flags;           // per-cell: true if wounded
    std::vector<double> wound_channel_factor; // per-membrane: TRP channel scaling

    // Gap junction blocking (per-membrane, 1.0 = fully open, 0.0 = fully blocked)
    std::vector<double> gj_block;

    // Per-ion per-membrane Dm modulation factors from GRN modulators
    std::vector<std::vector<double>> Dm_mod;  // [ion][membrane]

    // Simulation phase tracking
    enum class Phase { SEED, INIT, SIM };
    Phase phase = Phase::SEED;
};

// ============================================================================
// Simulation Snapshot
// ============================================================================
struct SimSnapshot {
    double time;
    std::vector<double> Vmem;
    std::vector<std::vector<double>> cc_cells;
    std::vector<double> rho_cells;
};

// ============================================================================

// Nernst-Planck flux in 2D
inline Vec2 nernst_planck_flux(double D, int z, double c,
                               Vec2 dc, Vec2 dv, Vec2 u,
                               double T, double mu = 0.0) {
    double alpha = (D * z * Q_ELECTRON) / (K_BOLTZMANN * T);
    double fx = -D * dc.x - alpha * dv.x * c + u.x * c - mu * c * dv.x;
    double fy = -D * dc.y - alpha * dv.y * c + u.y * c - mu * c * dv.y;
    return {fx, fy};
}

// Na-K-ATPase pump (full thermodynamic model from sim_toolbox.py)
inline void pump_NaKATP_full(double cNai, double cKo, double cNao, double cKi,
                              double Vm, double T, const SimConfig& p,
                              double& f_Na, double& f_K) {
    double Qnum = (p.cADP * 1e-3) * (p.cPi * 1e-3) *
                  std::pow(cNao * 1e-3, 3) * std::pow(cKi * 1e-3, 2);
    double Qden = (p.cATP * 1e-3) *
                  std::pow(cNai * 1e-3, 3) * std::pow(cKo * 1e-3, 2);
    if (std::abs(Qden) < 1e-30) Qden = 1e-30;
    double Q = Qnum / Qden;
    double nk_exp_arg = -(p.deltaGATP / (R_GAS * T) - (F_FARADAY * Vm) / (R_GAS * T));
    nk_exp_arg = std::clamp(nk_exp_arg, -500.0, 500.0);
    double Keq = std::exp(nk_exp_arg);
    double numo_E = std::pow(cNai / p.KmNK_Na, 3) *
                    std::pow(cKo / p.KmNK_K, 2) *
                    (p.cATP / p.KmNK_ATP);
    double deno_E = (1.0 + std::pow(cNai / p.KmNK_Na, 3)) *
                    (1.0 + std::pow(cKo / p.KmNK_K, 2)) *
                    (1.0 + p.cATP / p.KmNK_ATP);
    double fwd = numo_E / deno_E;
    f_Na = -3.0 * p.alpha_NaK * fwd * (1.0 - Q / Keq);
    f_K = -(2.0 / 3.0) * f_Na;
}

// Ca-ATPase pump (from sim_toolbox.py pumpCaATP)
inline double pump_CaATP(double cCai, double cCao, double Vm, double T,
                          const SimConfig& p) {
    cCai = std::max(cCai, 1e-15);
    cCao = std::max(cCao, 1e-15);
    double Qnum = p.cADP * p.cPi * cCao;
    double Qden = p.cATP * cCai;
    if (std::abs(Qden) < 1e-30) Qden = 1e-30;
    double Q = Qnum / Qden;
    double exp_arg = -(p.deltaGATP / (R_GAS * T) - 2.0 * (F_FARADAY * Vm) / (R_GAS * T));
    exp_arg = std::clamp(exp_arg, -500.0, 500.0);
    double Keq = std::exp(exp_arg);
    if (std::abs(Keq) < 1e-30) Keq = 1e-30;
    double numo_E = (cCai / p.KmCa_Ca) * (p.cATP / p.KmCa_ATP);
    double deno_E = (1.0 + cCai / p.KmCa_Ca) * (1.0 + p.cATP / p.KmCa_ATP);
    double fwd = numo_E / deno_E;
    double result = -p.alpha_Ca * fwd * (1.0 - Q / Keq);
    if (!std::isfinite(result)) result = 0.0;
    return result;
}

// SERCA pump (from sim_toolbox.py pumpCaER)
inline double pump_SERCA(double cCa_er, double cCa_cell, double Ver,
                          double T, const SimConfig& p) {
    cCa_er = std::max(cCa_er, 1e-15);
    cCa_cell = std::max(cCa_cell, 1e-15);
    double Qnum = p.cADP * p.cPi * cCa_cell;
    double Qden = p.cATP * cCa_er;
    if (std::abs(Qden) < 1e-30) Qden = 1e-30;
    double Q = Qnum / Qden;
    double exp_arg = -p.deltaGATP / (R_GAS * T) - 2.0 * (F_FARADAY * Ver) / (R_GAS * T);
    exp_arg = std::clamp(exp_arg, -500.0, 500.0);
    double Keq = std::exp(exp_arg);
    if (std::abs(Keq) < 1e-30) Keq = 1e-30;
    double numo_E = (cCa_er / p.KmCa_Ca) * (p.cATP / p.KmCa_ATP);
    double deno_E = (1.0 + cCa_er / p.KmCa_Ca) * (1.0 + p.cATP / p.KmCa_ATP);
    double fwd = numo_E / deno_E;
    double result = p.serca_max * fwd * (1.0 - Q / Keq);
    if (!std::isfinite(result)) result = 0.0;
    return result;
}

// ============================================================================
// Main Simulator Class
// ============================================================================
class Simulator {
public:
    SimConfig   config;
    TissueMesh  mesh;
    ECMGrid     ecm;
    SimState    state;
    std::vector<SimSnapshot> history;
    std::mt19937 rng;

    // Subsystems
    std::vector<ActiveChannel> active_channels;
    EndoReticulum er;
    Mitochondria  mito;
    Microtubules  mtubes;
    GRNetwork     grn;

    Simulator() : rng(42) {}
    explicit Simulator(const SimConfig& cfg) : config(cfg), rng(42) {}

    // ========================================================================
    // Initialize
    // ========================================================================
    void init() {
        mesh.generate(config.num_cells, config.tissue_radius, config.cell_height, rng);
        int nc = (int)mesh.cells.size();
        int nm = (int)mesh.membranes.size();
        int ni = static_cast<int>(Ion::COUNT);

        if (config.ecm_enabled) {
            ecm.generate(config.tissue_radius, config.ecm_delta);
            ecm.map_membranes(mesh.membranes);
        }

        state.num_cells = nc;
        state.num_mems  = nm;
        state.num_ions  = ni;
        state.time = 0.0;
        state.step = 0;

        // Allocate arrays
        state.cc_cells.resize(ni, std::vector<double>(nc));
        state.cc_env.resize(ni, std::vector<double>(nc));
        state.cc_mem.resize(ni, std::vector<double>(nm));
        state.Vmem.resize(nm, 0.0);
        state.rho_cells.resize(nc, 0.0);
        state.gj_gates.resize(nm);
        state.flux_mem.resize(ni, std::vector<double>(nm, 0.0));
        state.flux_gj.resize(ni, std::vector<double>(nm, 0.0));
        state.pump_Na.resize(nm, 0.0);
        state.pump_K.resize(nm, 0.0);
        state.E_cell_x.assign(nc, 0.0);
        state.E_cell_y.assign(nc, 0.0);
        state.osmo_P_cell.assign(nc, 0.0);
        state.osmo_P_env.assign(nc, 0.0);
        state.P_cells.assign(nc, 0.0);
        state.d_cells_x.assign(nc, 0.0);
        state.d_cells_y.assign(nc, 0.0);
        state.u_cells_x.assign(nc, 0.0);
        state.u_cells_y.assign(nc, 0.0);

        // Wound tracking
        state.wound_flags.assign(nc, false);
        state.wound_channel_factor.assign(nm, 0.0);

        // GJ blocking (default: all open)
        state.gj_block.assign(nm, 1.0);

        // GRN Dm modulation (default: no modulation)
        state.Dm_mod.resize(ni, std::vector<double>(nm, 1.0));

        if (config.ecm_enabled) {
            int ne = ecm.nx * ecm.ny;
            state.cc_ecm.resize(ni, std::vector<double>(ne));
            state.V_ecm.resize(ne, 0.0);
        }

        // Set initial concentrations
        for (int ion = 0; ion < ni; ion++) {
            for (int c = 0; c < nc; c++) {
                state.cc_cells[ion][c] = config.ions[ion].cell;
                state.cc_env[ion][c]   = config.ions[ion].env;
            }
            for (int m = 0; m < nm; m++) {
                state.cc_mem[ion][m] = config.ions[ion].cell;
            }
            if (config.ecm_enabled) {
                for (int e = 0; e < ecm.nx * ecm.ny; e++)
                    state.cc_ecm[ion][e] = config.ions[ion].env;
            }
        }

        // Apply tissue profiles
        for (auto& prof : config.tissue_profiles) {
            for (auto& cell : mesh.cells) {
                bool in_profile = false;
                switch (prof.shape) {
                    case TissueProfileShape::ALL:
                        in_profile = true;
                        break;
                    case TissueProfileShape::CIRCULAR:
                        in_profile = (cell.center - prof.center).norm() < prof.radius;
                        break;
                    case TissueProfileShape::RECTANGULAR:
                        in_profile = std::abs(cell.center.x - prof.center.x) < prof.width / 2 &&
                                     std::abs(cell.center.y - prof.center.y) < prof.height / 2;
                        break;
                    default: break;
                }
                if (in_profile) {
                    cell.tissue_profile_id = prof.id;
                }
            }
        }

        // Initialize voltage-gated channels
        for (auto& ch_cfg : config.channels) {
            ActiveChannel ach;
            ach.type = ch_cfg.type;
            ach.max_Dm = ch_cfg.max_Dm;

            // Determine target membranes
            if (ch_cfg.target_profile_id < 0) {
                ach.target_mems.resize(nm);
                std::iota(ach.target_mems.begin(), ach.target_mems.end(), 0);
            } else {
                for (int m = 0; m < nm; m++) {
                    int c = mesh.mem_to_cell[m];
                    if (mesh.cells[c].tissue_profile_id == ch_cfg.target_profile_id) {
                        ach.target_mems.push_back(m);
                    }
                }
            }

            // Build Vm in mV for channel init
            std::vector<double> Vm_mV(ach.target_mems.size());
            for (size_t i = 0; i < ach.target_mems.size(); i++) {
                Vm_mV[i] = state.Vmem[ach.target_mems[i]] * 1000.0;
            }

            channel_init(ach.state, ach.type, Vm_mV);
            active_channels.push_back(std::move(ach));
        }

        // Initialize organelles
        if (config.er_enabled) {
            std::vector<double> cell_vol(nc), cell_sa(nc);
            for (int c = 0; c < nc; c++) {
                cell_vol[c] = mesh.cells[c].volume;
                cell_sa[c] = mesh.cells[c].surface_area;
            }
            er.init(nc, ni, config.er_vol_frac, config.er_sa_frac,
                    cell_vol, cell_sa, state.cc_cells, config.cm,
                    config.er_init_Ca, ion_index(Ion::Ca));
        }

        if (config.mito_enabled) {
            std::vector<double> cell_vol(nc), cell_sa(nc);
            for (int c = 0; c < nc; c++) {
                cell_vol[c] = mesh.cells[c].volume;
                cell_sa[c] = mesh.cells[c].surface_area;
            }
            mito.init(nc, ni, config.mito_vol_frac, config.mito_sa_frac,
                      cell_vol, cell_sa, state.cc_cells);
        }

        if (config.mtube_enabled) {
            std::vector<double> R_rads(nc);
            double mean_R = 0;
            for (int c = 0; c < nc; c++) {
                R_rads[c] = mesh.cells[c].radius;
                mean_R += R_rads[c];
            }
            mean_R /= nc;
            mtubes.init(nm, nc, config.mt_radius, R_rads, mean_R, config);
        }

        if (config.grn_enabled) {
            grn.enabled = true;
            grn.init(nc, nc); // simplified: env = nc points
        }

        update_charge();
        update_voltage();
    }

    // ========================================================================
    // Charge density calculation
    // ========================================================================
    void update_charge() {
        int nc = state.num_cells;
        int ni = state.num_ions;
        for (int c = 0; c < nc; c++) {
            double rho = 0.0;
            for (int ion = 0; ion < ni; ion++)
                rho += ion_valence(static_cast<Ion>(ion)) * F_FARADAY * state.cc_cells[ion][c];
            state.rho_cells[c] = rho;
        }
    }

    // ========================================================================
    // Voltage calculation
    // ========================================================================
    void update_voltage() {
        int nm = state.num_mems;
        for (int m = 0; m < nm; m++) {
            int c = mesh.mem_to_cell[m];
            double sigma = state.rho_cells[c] * mesh.cells[c].volume / mesh.cells[c].surface_area;
            state.Vmem[m] = sigma / config.cm;
        }
    }

    // ========================================================================
    // Electric field calculation (at cell centres)
    // ========================================================================
    void update_electric_field() {
        int nc = state.num_cells;
        // Compute E from voltage gradient between neighbours
        for (int c = 0; c < nc; c++) {
            double Ex = 0, Ey = 0;
            int count = 0;
            for (int nb : mesh.cells[c].neighbor_ids) {
                Vec2 d = mesh.cells[nb].center - mesh.cells[c].center;
                double dist = d.norm();
                if (dist < 1e-30) continue;
                // Average Vmem for each cell
                double Vc = 0, Vnb = 0;
                int nc_mems = 0, nnb_mems = 0;
                for (int mi : mesh.cells[c].membrane_ids) {
                    Vc += state.Vmem[mi]; nc_mems++;
                }
                for (int mi : mesh.cells[nb].membrane_ids) {
                    Vnb += state.Vmem[mi]; nnb_mems++;
                }
                if (nc_mems > 0) Vc /= nc_mems;
                if (nnb_mems > 0) Vnb /= nnb_mems;

                double dV = Vnb - Vc;
                Ex -= dV * d.x / (dist * dist);
                Ey -= dV * d.y / (dist * dist);
                count++;
            }
            if (count > 0) {
                state.E_cell_x[c] = Ex / count;
                state.E_cell_y[c] = Ey / count;
            }
        }
    }

    // ========================================================================
    // Gap junction flux
    // ========================================================================
    void update_gap_junctions() {
        int nm = state.num_mems;
        double dt_ms = config.dt * 1000.0;

        for (int m = 0; m < nm; m++) {
            int nn = mesh.membranes[m].neighbor_mem_id;
            if (nn < 0) continue;
            double vgj = state.Vmem[nn] - state.Vmem[m];
            state.gj_gates[m].update(vgj * 1000.0, dt_ms, config.gj_min);

            // Apply GJ blocking factor (from tissue events or GRN)
            double gj_eff = state.gj_gates[m].gjopen * state.gj_block[m];

            for (int ion = 0; ion < state.num_ions; ion++) {
                double D_gj = config.ions[ion].D_free * config.gj_surface * gj_eff;
                state.flux_gj[ion][m] = electroflux(
                    state.cc_mem[ion][m], state.cc_mem[ion][nn],
                    D_gj, mesh.membranes[m].gj_dist,
                    ion_valence(static_cast<Ion>(ion)), vgj, config.T);
            }
        }
    }

    // ========================================================================
    // Transmembrane flux (Goldman)
    // ========================================================================
    void update_membrane_flux() {
        int nm = state.num_mems;
        for (int ion = 0; ion < state.num_ions; ion++) {
            double Dm_base = config.ions[ion].D_mem;
            if (Dm_base <= 0) continue;
            int z = ion_valence(static_cast<Ion>(ion));
            for (int m = 0; m < nm; m++) {
                int c = mesh.mem_to_cell[m];
                // Apply tissue profile Dm multiplier
                double Dm = Dm_base;
                if (mesh.cells[c].tissue_profile_id >= 0 &&
                    mesh.cells[c].tissue_profile_id < (int)config.tissue_profiles.size()) {
                    Dm *= config.tissue_profiles[mesh.cells[c].tissue_profile_id].Dm_multipliers[ion];
                }
                // Apply GRN modulator factor
                Dm *= state.Dm_mod[ion][m];
                // Apply wound channel factor (TRP channels increase permeability)
                if (state.wound_channel_factor[m] > 0.0) {
                    Dm *= (1.0 + state.wound_channel_factor[m]);
                }
                state.flux_mem[ion][m] = electroflux(
                    state.cc_env[ion][c], state.cc_cells[ion][c],
                    Dm, config.tm, z, state.Vmem[m], config.T);
            }
        }
    }

    // ========================================================================
    // Voltage-gated channel fluxes
    // ========================================================================
    void update_channels() {
        for (auto& ach : active_channels) {
            // Build Vm_mV for target membranes
            std::vector<double> Vm_mV(ach.target_mems.size());
            for (size_t i = 0; i < ach.target_mems.size(); i++) {
                Vm_mV[i] = state.Vmem[ach.target_mems[i]] * 1000.0;
            }

            // Step the channel model
            channel_step(ach.state, ach.type, Vm_mV, config.dt, state.num_mems);

            // Apply channel flux for each permeable ion
            for (size_t ion_idx = 0; ion_idx < ach.state.ions.size(); ion_idx++) {
                // Map ion name to Ion enum
                int ion_id = -1;
                const std::string& ion_name_str = ach.state.ions[ion_idx];
                if (ion_name_str == "Na") ion_id = ion_index(Ion::Na);
                else if (ion_name_str == "K") ion_id = ion_index(Ion::K);
                else if (ion_name_str == "Cl") ion_id = ion_index(Ion::Cl);
                else if (ion_name_str == "Ca") ion_id = ion_index(Ion::Ca);
                else if (ion_name_str == "H") ion_id = ion_index(Ion::H);
                if (ion_id < 0) continue;

                double rel_p = ach.state.rel_perm[ion_idx];
                int z = ion_valence(static_cast<Ion>(ion_id));

                for (size_t ti = 0; ti < ach.target_mems.size(); ti++) {
                    int m = ach.target_mems[ti];
                    int c = mesh.mem_to_cell[m];
                    double P_open = (ti < ach.state.P.size()) ? ach.state.P[ti] : 0.0;
                    double Dchan = ach.max_Dm * P_open * rel_p * ach.state.modulator;

                    double flux = electroflux(
                        state.cc_env[ion_id][c], state.cc_cells[ion_id][c],
                        Dchan, config.tm, z, state.Vmem[m], config.T);

                    state.flux_mem[ion_id][m] += flux;
                }
            }
        }
    }

    // ========================================================================
    // Na-K-ATPase pump
    // ========================================================================
    void update_pumps() {
        if (!config.pump_enabled) return;
        int nm = state.num_mems;
        int iNa = ion_index(Ion::Na);
        int iK  = ion_index(Ion::K);

        for (int m = 0; m < nm; m++) {
            int c = mesh.mem_to_cell[m];
            double f_Na, f_K;
            pump_NaKATP_full(
                state.cc_cells[iNa][c], state.cc_env[iK][c],
                state.cc_env[iNa][c], state.cc_cells[iK][c],
                state.Vmem[m], config.T, config,
                f_Na, f_K);
            state.pump_Na[m] = f_Na;
            state.pump_K[m] = f_K;
            state.flux_mem[iNa][m] += f_Na;
            state.flux_mem[iK][m]  += f_K;
        }
    }

    // ========================================================================
    // Ca2+ dynamics (PMCA pump)
    // ========================================================================
    void update_calcium() {
        if (!config.ca_dyn_enabled) return;
        int iCa = ion_index(Ion::Ca);
        int nm = state.num_mems;
        for (int m = 0; m < nm; m++) {
            int c = mesh.mem_to_cell[m];
            double f_Ca = pump_CaATP(
                state.cc_cells[iCa][c], state.cc_env[iCa][c],
                state.Vmem[m], config.T, config);
            state.flux_mem[iCa][m] += f_Ca;
        }
    }

    // ========================================================================
    // Endoplasmic reticulum update
    // ========================================================================
    void update_er() {
        if (!config.er_enabled) return;
        int nc = state.num_cells;
        int ni = state.num_ions;
        int iCa = ion_index(Ion::Ca);

        // Update ER calcium channels
        er.update_channels(nc, iCa, state.cc_cells,
                          config.max_er, config.act_Km_Ca, config.act_n_Ca,
                          config.inh_Km_Ca, config.inh_n_Ca);

        // SERCA pump
        for (int c = 0; c < nc; c++) {
            double f_SERCA = pump_SERCA(er.cc_er[iCa][c], state.cc_cells[iCa][c],
                                         er.Ver[c], config.T, config);
            if (!std::isfinite(f_SERCA)) f_SERCA = 0.0;
            // Limit SERCA flux to prevent driving concentrations negative
            double ca_cell = std::max(state.cc_cells[iCa][c], 0.0);
            double ca_er = std::max(er.cc_er[iCa][c], 0.0);
            double max_serca_out = ca_cell * mesh.cells[c].volume / (er.er_sa[c] * config.dt + FLOAT_NONCE);
            double max_serca_in = ca_er * er.er_vol[c] / (er.er_sa[c] * config.dt + FLOAT_NONCE);
            if (f_SERCA > 0) f_SERCA = std::min(f_SERCA, max_serca_out * 0.5);
            else              f_SERCA = std::max(f_SERCA, -max_serca_in * 0.5);
            state.cc_cells[iCa][c] -= f_SERCA * (er.er_sa[c] / mesh.cells[c].volume) * config.dt;
            er.cc_er[iCa][c] += f_SERCA * (er.er_sa[c] / er.er_vol[c]) * config.dt;
            state.cc_cells[iCa][c] = std::max(0.0, state.cc_cells[iCa][c]);
            er.cc_er[iCa][c] = std::max(0.0, er.cc_er[iCa][c]);
        }

        // Electrodiffusion across ER membrane for all ions
        for (int ion = 0; ion < ni; ion++) {
            int z = ion_valence(static_cast<Ion>(ion));
            for (int c = 0; c < nc; c++) {
                double cA = std::max(state.cc_cells[ion][c], 0.0);
                double cB = std::max(er.cc_er[ion][c], 0.0);
                double f = electroflux(cA, cB, er.Dm_er[ion][c], config.tm, z, er.Ver[c], config.T);
                if (!std::isfinite(f)) f = 0.0;
                // Limit flux to prevent driving concentrations negative
                double max_out_cell = cA * mesh.cells[c].volume / (er.er_sa[c] * config.dt + FLOAT_NONCE);
                double max_out_er = cB * er.er_vol[c] / (er.er_sa[c] * config.dt + FLOAT_NONCE);
                if (f > 0) f = std::min(f, max_out_cell * 0.5);     // cell -> ER
                else       f = std::max(f, -max_out_er * 0.5);      // ER -> cell
                state.cc_cells[ion][c] -= f * (er.er_sa[c] / mesh.cells[c].volume) * config.dt;
                er.cc_er[ion][c] += f * (er.er_sa[c] / er.er_vol[c]) * config.dt;
                state.cc_cells[ion][c] = std::max(0.0, state.cc_cells[ion][c]);
                er.cc_er[ion][c] = std::max(0.0, er.cc_er[ion][c]);
            }
        }

        // Update ER voltage
        std::vector<int> zs(ni);
        for (int i = 0; i < ni; i++) zs[i] = ion_valence(static_cast<Ion>(i));
        er.get_voltage(nc, ni, config.cm, zs);
    }

    // ========================================================================
    // Mitochondria update
    // ========================================================================
    void update_mito() {
        if (!config.mito_enabled) return;
        int nc = state.num_cells;
        int ni = state.num_ions;

        for (int ion = 0; ion < ni; ion++) {
            int z = ion_valence(static_cast<Ion>(ion));
            for (int c = 0; c < nc; c++) {
                double cA = std::max(state.cc_cells[ion][c], 0.0);
                double cB = std::max(mito.cc_mit[ion][c], 0.0);
                double f = electroflux(cA, cB, mito.Dm_mit[ion][c], config.tm, z, mito.Vmit[c], config.T);
                if (!std::isfinite(f)) f = 0.0;
                double max_out_cell = cA * mesh.cells[c].volume / (mito.mit_sa[c] * config.dt + FLOAT_NONCE);
                double max_out_mit = cB * mito.mit_vol[c] / (mito.mit_sa[c] * config.dt + FLOAT_NONCE);
                if (f > 0) f = std::min(f, max_out_cell * 0.5);
                else       f = std::max(f, -max_out_mit * 0.5);
                state.cc_cells[ion][c] -= f * (mito.mit_sa[c] / mesh.cells[c].volume) * config.dt;
                mito.cc_mit[ion][c] += f * (mito.mit_sa[c] / mito.mit_vol[c]) * config.dt;
                state.cc_cells[ion][c] = std::max(0.0, state.cc_cells[ion][c]);
                mito.cc_mit[ion][c] = std::max(0.0, mito.cc_mit[ion][c]);
            }
        }

        std::vector<int> zs(ni);
        for (int i = 0; i < ni; i++) zs[i] = ion_valence(static_cast<Ion>(i));
        mito.get_voltage(nc, ni, config.cm, zs);
    }

    // ========================================================================
    // Osmotic pressure (from physics/pressures.py)
    // ========================================================================
    void update_osmotic_pressure() {
        if (!config.osmo_enabled) return;
        int nc = state.num_cells;
        for (int c = 0; c < nc; c++) {
            double P_cell = 0, P_env = 0;
            for (int ion = 0; ion < state.num_ions; ion++) {
                P_cell += R_GAS * config.T * state.cc_cells[ion][c];
                P_env += R_GAS * config.T * state.cc_env[ion][c];
            }
            state.osmo_P_cell[c] = P_cell;
            state.osmo_P_env[c] = P_env;
            // Transmembrane water flux due to osmotic pressure
            double delta_P = P_env - P_cell - state.P_cells[c];
            double u_osmo = delta_P * (config.aquaporins / config.mu_water) *
                           std::pow(3.0e-10, 2) / config.tm;
            double div_u = u_osmo * mesh.cells[c].surface_area / mesh.cells[c].volume;
            state.P_cells[c] += -div_u * config.rho * config.dt;
        }
    }

    // ========================================================================
    // Deformation (from physics/deform.py)
    // ========================================================================
    void update_deformation() {
        if (!config.deform_enabled) return;
        int nc = state.num_cells;
        // Compute pressure gradient between neighbors
        for (int c = 0; c < nc; c++) {
            double gPx = 0, gPy = 0;
            int count = 0;
            for (int nb : mesh.cells[c].neighbor_ids) {
                Vec2 d = mesh.cells[nb].center - mesh.cells[c].center;
                double dist = d.norm();
                if (dist < 1e-30) continue;
                double dP = state.P_cells[nb] - state.P_cells[c];
                gPx -= dP * d.x / (dist * dist);
                gPy -= dP * d.y / (dist * dist);
                count++;
            }
            if (count > 0) { gPx /= count; gPy /= count; }
            // Linear elasticity: displacement proportional to pressure gradient
            state.d_cells_x[c] = gPx / config.lame_mu +
                                  state.E_cell_x[c] * config.galvanotropism;
            state.d_cells_y[c] = gPy / config.lame_mu +
                                  state.E_cell_y[c] * config.galvanotropism;
        }
    }

    // ========================================================================
    // Electroosmotic flow (from physics/flow.py)
    // ========================================================================
    void update_flow() {
        if (!config.flow_enabled) return;
        int nc = state.num_cells;
        for (int c = 0; c < nc; c++) {
            double Fx = state.E_cell_x[c] * state.rho_cells[c] / config.mu_water * config.gj_surface;
            double Fy = state.E_cell_y[c] * state.rho_cells[c] / config.mu_water * config.gj_surface;
            // Simplified: flow proportional to force
            state.u_cells_x[c] = Fx * config.dt;
            state.u_cells_y[c] = Fy * config.dt;
        }
    }

    // ========================================================================
    // Concentration updates
    // ========================================================================
    void update_concentrations() {
        int nc = state.num_cells;
        double dt = config.dt;

        for (int ion = 0; ion < state.num_ions; ion++) {
            std::vector<double> delta(nc, 0.0);
            for (int m = 0; m < state.num_mems; m++) {
                int c = mesh.mem_to_cell[m];
                delta[c] += state.flux_mem[ion][m] * mesh.membranes[m].surface_area;
            }
            for (int c = 0; c < nc; c++)
                state.cc_cells[ion][c] += dt * delta[c] / mesh.cells[c].volume;

            std::fill(delta.begin(), delta.end(), 0.0);
            for (int m = 0; m < state.num_mems; m++) {
                int c = mesh.mem_to_cell[m];
                delta[c] -= state.flux_gj[ion][m] * mesh.membranes[m].surface_area;
            }
            for (int c = 0; c < nc; c++)
                state.cc_cells[ion][c] += dt * delta[c] / mesh.cells[c].volume;

            for (int c = 0; c < nc; c++)
                state.cc_cells[ion][c] = std::max(0.0, state.cc_cells[ion][c]);

            for (int m = 0; m < state.num_mems; m++)
                state.cc_mem[ion][m] = state.cc_cells[ion][mesh.mem_to_cell[m]];

            if (!config.ecm_enabled) {
                double total_flux = 0;
                for (int m = 0; m < state.num_mems; m++)
                    total_flux -= state.flux_mem[ion][m] * mesh.membranes[m].surface_area;
                double env_vol = mesh.cells[0].volume * nc * 10.0;
                double delta_env = total_flux / env_vol;
                for (int c = 0; c < nc; c++) {
                    state.cc_env[ion][c] += dt * delta_env;
                    state.cc_env[ion][c] = std::max(0.0, state.cc_env[ion][c]);
                }
            } else {
                // ECM diffusion
                ecm.diffuse(state.cc_ecm[ion], config.ions[ion].D_free, dt);
            }
        }
    }

    // ========================================================================
    // ECM voltage calculation
    // ========================================================================
    void update_ecm_voltage() {
        if (!config.ecm_enabled) return;
        int nx = ecm.nx;
        int ny = ecm.ny;
        int ne = nx * ny;
        for (int e = 0; e < ne; e++) {
            double rho = 0;
            for (int ion = 0; ion < state.num_ions; ion++) {
                rho += ion_valence(static_cast<Ion>(ion)) * F_FARADAY * state.cc_ecm[ion][e];
            }
            state.V_ecm[e] = rho / (config.cm / config.tm);
        }

        // Apply boundary voltage conditions (from tissue events)
        // bound_V[0]=TOP, [1]=BOTTOM, [2]=LEFT, [3]=RIGHT
        // Check if any boundary voltages are set (non-zero)
        bool has_bc = false;
        for (int i = 0; i < 4; i++)
            if (std::abs(config.bound_V[i]) > 1e-15) has_bc = true;
        if (!has_bc) return;

        // Top boundary (last row: y = ny-1)
        if (std::abs(config.bound_V[0]) > 1e-15) {
            for (int x = 0; x < nx; x++)
                state.V_ecm[(ny - 1) * nx + x] = config.bound_V[0];
        }
        // Bottom boundary (first row: y = 0)
        if (std::abs(config.bound_V[1]) > 1e-15) {
            for (int x = 0; x < nx; x++)
                state.V_ecm[x] = config.bound_V[1];
        }
        // Left boundary (first column: x = 0)
        if (std::abs(config.bound_V[2]) > 1e-15) {
            for (int y = 0; y < ny; y++)
                state.V_ecm[y * nx] = config.bound_V[2];
        }
        // Right boundary (last column: x = nx-1)
        if (std::abs(config.bound_V[3]) > 1e-15) {
            for (int y = 0; y < ny; y++)
                state.V_ecm[y * nx + (nx - 1)] = config.bound_V[3];
        }
    }

    // ========================================================================
    // Process cutting events (tissue damage)
    // ========================================================================
    void process_cuts() {
        for (auto& cut : config.cut_events) {
            if (std::abs(state.time - cut.time) < config.dt * 0.5) {
                std::vector<int> to_remove;
                for (auto& cell : mesh.cells) {
                    if ((cell.center - cut.center).norm() < cut.radius) {
                        to_remove.push_back(cell.id);
                    }
                }
                if (!to_remove.empty()) {
                    std::cout << "BETSE: Cutting " << to_remove.size()
                              << " cells at t=" << state.time << "s\n";
                    // Remove from mesh
                    mesh.remove_cells(to_remove);

                    // Reinitialize state for new cell count
                    int new_nc = (int)mesh.cells.size();
                    int new_nm = (int)mesh.membranes.size();
                    state.num_cells = new_nc;
                    state.num_mems = new_nm;

                    // Rebuild arrays (simplified: keep existing values for surviving cells)
                    for (int ion = 0; ion < state.num_ions; ion++) {
                        state.cc_cells[ion].resize(new_nc);
                        state.cc_env[ion].resize(new_nc);
                        state.cc_mem[ion].resize(new_nm);
                        state.flux_mem[ion].resize(new_nm, 0.0);
                        state.flux_gj[ion].resize(new_nm, 0.0);
                    }
                    state.Vmem.resize(new_nm, 0.0);
                    state.rho_cells.resize(new_nc, 0.0);
                    state.gj_gates.resize(new_nm);
                    state.pump_Na.resize(new_nm, 0.0);
                    state.pump_K.resize(new_nm, 0.0);
                    state.E_cell_x.resize(new_nc, 0.0);
                    state.E_cell_y.resize(new_nc, 0.0);
                    state.P_cells.resize(new_nc, 0.0);
                    state.d_cells_x.resize(new_nc, 0.0);
                    state.d_cells_y.resize(new_nc, 0.0);
                    state.u_cells_x.resize(new_nc, 0.0);
                    state.u_cells_y.resize(new_nc, 0.0);
                    state.wound_flags.resize(new_nc, false);
                    state.wound_channel_factor.resize(new_nm, 0.0);
                    state.gj_block.resize(new_nm, 1.0);
                    for (int ion = 0; ion < state.num_ions; ion++)
                        state.Dm_mod[ion].resize(new_nm, 1.0);

                    // Mark surviving cells near wound as wounded (TRP channel activation)
                    for (int c = 0; c < new_nc; c++) {
                        double dist = (mesh.cells[c].center - cut.center).norm();
                        if (dist < cut.radius * 2.0) {
                            state.wound_flags[c] = true;
                        }
                    }
                }
            }
        }
    }

    // ========================================================================
    // Update wound channel factors (TRP channels open near wounds, then decay)
    // ========================================================================
    void update_wound_channels() {
        int nm = state.num_mems;
        for (int m = 0; m < nm; m++) {
            int c = mesh.mem_to_cell[m];
            if (state.wound_flags[c]) {
                // Open wound channels: approach max factor of 2.0 with rate 0.1/s
                state.wound_channel_factor[m] = std::max(0.0,
                    state.wound_channel_factor[m] +
                    (2.0 - state.wound_channel_factor[m]) * 0.1 * config.dt);
            }
            // All wound channels decay over time (wound closure)
            state.wound_channel_factor[m] *= (1.0 - config.wound_close_factor * config.dt);
            if (state.wound_channel_factor[m] < 1e-12)
                state.wound_channel_factor[m] = 0.0;
        }
    }

    // ========================================================================
    // Apply voltage clamps
    // ========================================================================
    void apply_voltage_clamps() {
        for (auto& vc : config.voltage_clamps) {
            if (state.time >= vc.start_time && state.time <= vc.end_time) {
                for (int m = 0; m < state.num_mems; m++) {
                    int c = mesh.mem_to_cell[m];
                    if (vc.target_profile_id < 0 ||
                        mesh.cells[c].tissue_profile_id == vc.target_profile_id) {
                        state.Vmem[m] = vc.voltage;
                    }
                }
            }
        }
    }

    // ========================================================================
    // Add noise
    // ========================================================================
    void add_noise() {
        if (!config.noise_enabled) return;
        std::normal_distribution<double> noise(0.0, config.noise_level);
        for (int m = 0; m < state.num_mems; m++)
            state.Vmem[m] += noise(rng);
    }

    // ========================================================================
    // Stability check
    // ========================================================================
    bool check_stability() const {
        for (int m = 0; m < state.num_mems; m++) {
            if (std::isnan(state.Vmem[m]) || std::isinf(state.Vmem[m]))
                return false;
        }
        return true;
    }

    // ========================================================================
    // Single time step
    // ========================================================================
    bool step() {
        // Clear fluxes
        for (int ion = 0; ion < state.num_ions; ion++) {
            std::fill(state.flux_mem[ion].begin(), state.flux_mem[ion].end(), 0.0);
            std::fill(state.flux_gj[ion].begin(),  state.flux_gj[ion].end(),  0.0);
        }

        // 1. Process cutting events (wound channels activated here)
        process_cuts();

        // 1b. Process GJ block events
        for (auto& gev : config.gj_block_events) {
            if (std::abs(state.time - gev.time) < config.dt * 0.5) {
                for (int m = 0; m < state.num_mems; m++) {
                    Vec2 mp = mesh.membranes[m].midpoint;
                    if ((mp - gev.center).norm() < gev.radius)
                        state.gj_block[m] = gev.block_factor;
                }
            }
        }

        // 1c. Process boundary voltage events
        for (auto& bev : config.boundary_voltage_events) {
            if (state.time >= bev.start_time && state.time <= bev.end_time)
                config.bound_V[bev.side] = bev.voltage;
        }

        // 1d. Update wound channels (TRP activation/decay)
        update_wound_channels();

        // 2. Apply voltage clamps
        apply_voltage_clamps();

        // 3. Pumps
        update_pumps();

        // 4. Transmembrane flux (Goldman)
        update_membrane_flux();

        // 5. Voltage-gated channels
        update_channels();

        // 6. Gap junction flux
        update_gap_junctions();

        // 7. Ca2+ dynamics
        update_calcium();

        // 8. Update concentrations
        update_concentrations();

        // 9. Recompute charge
        update_charge();

        // 10. Recompute voltage
        update_voltage();

        // 11. Electric field
        update_electric_field();

        // 12. Endoplasmic reticulum
        update_er();

        // 13. Mitochondria
        update_mito();

        // 14. Osmotic pressure
        update_osmotic_pressure();

        // 15. Deformation
        update_deformation();

        // 16. Electroosmotic flow
        update_flow();

        // 17. Microtubules
        if (config.mtube_enabled) {
            mtubes.update(state.num_mems, state.E_cell_x, state.E_cell_y,
                          mesh.mem_to_cell, config.dt, config, rng);
        }

        // 18. Gene regulatory network
        if (config.grn_enabled) {
            int ne = config.ecm_enabled ? ecm.nx * ecm.ny : 0;
            // Build membrane SA and cell volume arrays
            std::vector<double> mem_sa(state.num_mems);
            std::vector<double> cell_vol(state.num_cells);
            for (int m = 0; m < state.num_mems; m++)
                mem_sa[m] = mesh.membranes[m].surface_area;
            for (int c = 0; c < state.num_cells; c++)
                cell_vol[c] = mesh.cells[c].volume;

            // Build GJ open array (gating * blocking)
            std::vector<double> gj_open(state.num_mems);
            std::vector<int> neighbor_ids(state.num_mems);
            std::vector<double> gj_lens(state.num_mems);
            for (int m = 0; m < state.num_mems; m++) {
                gj_open[m] = state.gj_gates[m].gjopen * state.gj_block[m];
                neighbor_ids[m] = mesh.membranes[m].neighbor_mem_id;
                gj_lens[m] = mesh.membranes[m].gj_dist;
            }

            grn.step(config.dt, state.num_cells, state.num_mems, ne,
                     mesh.mem_to_cell, mem_sa, cell_vol,
                     state.Vmem, config.T, config.tm,
                     gj_open, neighbor_ids, gj_lens, 1.0,
                     config.deltaGATP, config.cATP, config.cADP, config.cPi);

            // Apply GRN modulators to update Dm_mod
            for (int ion = 0; ion < state.num_ions; ion++)
                std::fill(state.Dm_mod[ion].begin(), state.Dm_mod[ion].end(), 1.0);
            grn.apply_modulators(state.num_mems, state.num_ions,
                                 mesh.mem_to_cell, state.Dm_mod);
        }

        // 19. ECM voltage
        update_ecm_voltage();

        // 20. Noise
        add_noise();

        // 21. Advance time
        state.time += config.dt;
        state.step++;

        return check_stability();
    }

    // ========================================================================
    // Run full simulation
    // ========================================================================
    // Run INIT phase: bring system to steady-state with relaxed parameters
    bool run_init() {
        state.phase = SimState::Phase::INIT;
        double saved_dt = config.dt;
        double saved_time = config.total_time;
        int saved_sr = config.sample_rate;

        // Use init-phase settings (larger dt, shorter duration, no events)
        config.dt = config.init_dt;
        config.total_time = config.init_total_time;
        config.sample_rate = config.init_sample_rate;

        int total_steps = (int)(config.total_time / config.dt);
        std::cout << "BETSE: INIT phase (" << total_steps << " steps, dt="
                  << config.dt << "s)...\n";

        for (int s = 0; s < total_steps; s++) {
            if (!step()) {
                std::cerr << "BETSE: NaN during INIT at step " << s << ". Aborting.\n";
                config.dt = saved_dt;
                config.total_time = saved_time;
                config.sample_rate = saved_sr;
                return false;
            }
        }

        // Restore sim-phase settings
        config.dt = saved_dt;
        config.total_time = saved_time;
        config.sample_rate = saved_sr;
        state.time = 0.0;
        state.step = 0;

        std::cout << "BETSE: INIT complete. Vmem range: ["
                  << *std::min_element(state.Vmem.begin(), state.Vmem.end()) * 1000.0
                  << ", "
                  << *std::max_element(state.Vmem.begin(), state.Vmem.end()) * 1000.0
                  << "] mV\n";
        return true;
    }

    // Run SIM phase: full simulation with all events and recording
    bool run_sim() {
        state.phase = SimState::Phase::SIM;
        int total_steps = (int)(config.total_time / config.dt);
        history.clear();
        save_snapshot();

        std::cout << "BETSE: SIM phase (" << total_steps << " steps, dt="
                  << config.dt << "s)...\n";

        for (int s = 0; s < total_steps; s++) {
            if (!step()) {
                std::cerr << "BETSE: NaN at step " << s << " (t=" << state.time << "s). Aborting.\n";
                return false;
            }
            if (s % config.sample_rate == 0) save_snapshot();
        }

        std::cout << "BETSE: SIM complete. " << total_steps << " steps, "
                  << history.size() << " snapshots.\n";
        return true;
    }

    // Full pipeline: SEED -> INIT -> SIM
    bool run() {
        // SEED phase: mesh generation and initial state (already done in init())
        state.phase = SimState::Phase::SEED;
        std::cout << "BETSE: SEED complete. " << state.num_cells << " cells, "
                  << state.num_mems << " membranes.\n";

        // INIT phase: bring to steady state
        if (!run_init()) return false;

        // SIM phase: full simulation
        return run_sim();
    }

    void save_snapshot() {
        SimSnapshot snap;
        snap.time = state.time;
        snap.Vmem = state.Vmem;
        snap.cc_cells = state.cc_cells;
        snap.rho_cells = state.rho_cells;
        history.push_back(std::move(snap));
    }

    double avg_Vmem() const {
        if (state.Vmem.empty()) return 0;
        double sum = 0;
        for (double v : state.Vmem) sum += v;
        return sum / state.Vmem.size();
    }

    // ========================================================================
    // CSV output
    // ========================================================================
    void write_csv(const std::string& filename) const {
        std::ofstream f(filename);
        if (!f) { std::cerr << "Cannot open " << filename << "\n"; return; }
        f << "time,avg_Vmem_mV";
        for (int ion = 0; ion < state.num_ions; ion++)
            f << ",avg_" << ion_name(static_cast<Ion>(ion)) << "_mM";
        f << "\n";
        for (auto& snap : history) {
            double avg_v = 0;
            for (double v : snap.Vmem) avg_v += v;
            avg_v = (avg_v / snap.Vmem.size()) * 1000.0;
            f << snap.time << "," << avg_v;
            for (int ion = 0; ion < state.num_ions; ion++) {
                double avg_c = 0;
                for (double c : snap.cc_cells[ion]) avg_c += c;
                avg_c /= snap.cc_cells[ion].size();
                f << "," << avg_c;
            }
            f << "\n";
        }
    }
};

} // namespace betse
