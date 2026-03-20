// BETSE C++ Port - Phase Management and Data Export
// Phase callbacks, time-series storage, scheduled events,
// tissue handler, data export, full electrodiffusion solvers.
// Ported from: phase/phasecls.py, phasecallbacks.py, tissue/tishandler.py,
//              tissue/event/*, pipe/export/*
#pragma once

#include "betse_types.h"
#include "betse_math.h"
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include <fstream>
#include <sstream>

namespace betse {

// ============================================================================
// Phase Callbacks (from phase/phasecallbacks.py)
// User-definable callbacks invoked at specific simulation events
// ============================================================================
struct PhaseCallbacks {
    std::function<void(double time, int step)> on_step;
    std::function<void(double time, int step)> on_sample;
    std::function<void()> on_init_complete;
    std::function<void()> on_sim_complete;
    std::function<bool(double time)> should_abort;  // return true to abort

    PhaseCallbacks() {
        on_step = [](double, int) {};
        on_sample = [](double, int) {};
        on_init_complete = []() {};
        on_sim_complete = []() {};
        should_abort = [](double) { return false; };
    }
};

// ============================================================================
// Extended Simulation Snapshot (full time-series data storage)
// Mirrors sim.py time-series attributes
// ============================================================================
struct ExtendedSnapshot {
    double time = 0;

    // Voltage data
    std::vector<double> Vmem;         // per-membrane Vmem
    std::vector<double> vm_ave;       // Vmem averaged to cell centres
    std::vector<double> v_env;        // environmental voltage

    // Concentration data
    std::vector<std::vector<double>> cc_cells;  // [ion][cell]
    std::vector<std::vector<double>> cc_env;    // [ion][env]

    // Charge density
    std::vector<double> rho_cells;

    // Current density
    std::vector<double> J_cell_x;
    std::vector<double> J_cell_y;
    std::vector<double> J_env_x;
    std::vector<double> J_env_y;

    // Electric field
    std::vector<double> E_gj_x;
    std::vector<double> E_gj_y;
    std::vector<double> E_env_x;
    std::vector<double> E_env_y;

    // Pressure
    std::vector<double> P_cells;

    // Deformation
    std::vector<double> d_cells_x;
    std::vector<double> d_cells_y;

    // Flow
    std::vector<double> u_cells_x;
    std::vector<double> u_cells_y;

    // Pump rates
    std::vector<double> pump_Na;
    std::vector<double> pump_K;

    // GRN molecule concentrations (name -> cells values)
    std::map<std::string, std::vector<double>> mol_cells;
};

// ============================================================================
// Time Series Recorder (collects ExtendedSnapshots over time)
// ============================================================================
struct TimeSeriesRecorder {
    std::vector<ExtendedSnapshot> snapshots;
    std::vector<double> time_points;

    void clear() {
        snapshots.clear();
        time_points.clear();
    }

    void add_snapshot(const ExtendedSnapshot& snap) {
        snapshots.push_back(snap);
        time_points.push_back(snap.time);
    }

    int num_snapshots() const { return (int)snapshots.size(); }

    // Get time series of average Vmem
    std::vector<double> avg_vmem_series() const {
        std::vector<double> result;
        for (auto& snap : snapshots) {
            double avg = 0;
            if (!snap.Vmem.empty()) {
                for (double v : snap.Vmem) avg += v;
                avg /= snap.Vmem.size();
            }
            result.push_back(avg);
        }
        return result;
    }

    // Get time series of a single cell's Vmem
    std::vector<double> cell_vmem_series(int cell_id) const {
        std::vector<double> result;
        for (auto& snap : snapshots) {
            if (cell_id < (int)snap.vm_ave.size())
                result.push_back(snap.vm_ave[cell_id]);
            else
                result.push_back(0.0);
        }
        return result;
    }

    // Get time series of average ion concentration
    std::vector<double> avg_ion_series(int ion) const {
        std::vector<double> result;
        for (auto& snap : snapshots) {
            double avg = 0;
            if (ion < (int)snap.cc_cells.size() && !snap.cc_cells[ion].empty()) {
                for (double c : snap.cc_cells[ion]) avg += c;
                avg /= snap.cc_cells[ion].size();
            }
            result.push_back(avg);
        }
        return result;
    }
};

// ============================================================================
// Scheduled Events (from tissue/event/)
// Generalised event system for dynamic parameter changes
// ============================================================================
enum class EventType {
    VOLTAGE_CLAMP,          // Set Vmem to a value
    CURRENT_INJECTION,      // Apply current to cells
    CONCENTRATION_CHANGE,   // Modify ion concentrations
    PERMEABILITY_CHANGE,    // Modify membrane Dm
    TEMPERATURE_CHANGE,     // Modify temperature
    GLOBAL_INTERVENTION,    // Apply to all cells
    BOUNDARY_VOLTAGE,       // Apply voltage at boundary (for ECM)
    WOUND_CHANNEL_ACTIVATION, // Activate wound channels
    CUSTOM                  // User-defined callback
};

struct ScheduledEvent {
    EventType type = EventType::VOLTAGE_CLAMP;
    double start_time = 0;
    double end_time = 0;
    double value = 0;
    int target_profile_id = -1;    // -1 = all cells
    int target_ion = -1;           // for concentration changes
    std::string boundary_side;     // "N","S","E","W" for boundary events

    // For modulated events
    bool use_modulation = false;
    double modulation_freq = 0;
    double modulation_phase = 0;

    // For custom events
    std::function<void(double time)> custom_func;

    bool is_active(double time) const {
        return time >= start_time && time <= end_time;
    }

    double get_value(double time) const {
        if (!use_modulation) return value;
        double mod = std::sin(time * M_PI * modulation_freq + modulation_phase);
        return value * mod * mod;
    }
};

// ============================================================================
// Tissue Handler (from tissue/tishandler.py)
// Manages tissue profiles, cutting/wounding, boundary flags
// ============================================================================
struct TissueHandler {
    // Boundary cell/membrane flags
    std::vector<bool> bflags_cells;    // true if cell is on boundary
    std::vector<bool> bflags_mems;     // true if membrane is on boundary
    std::vector<int> boundary_cells;   // indices of boundary cells
    std::vector<int> boundary_mems;    // indices of boundary membranes

    // Wound state
    std::vector<bool> wound_flags;     // true if cell is wounded
    std::vector<double> wound_channel_factor; // wound channel scaling per membrane

    void init(int nc, int nm, const std::vector<Vec2>& cell_centers,
              double tissue_radius) {
        bflags_cells.assign(nc, false);
        bflags_mems.assign(nm, false);
        wound_flags.assign(nc, false);
        wound_channel_factor.assign(nm, 0.0);
        boundary_cells.clear();
        boundary_mems.clear();

        // Mark boundary cells (near edge of tissue)
        double boundary_thresh = tissue_radius * 0.85;
        for (int c = 0; c < nc; c++) {
            if (cell_centers[c].norm() > boundary_thresh) {
                bflags_cells[c] = true;
                boundary_cells.push_back(c);
            }
        }
    }

    // Apply wound to cells in a region
    void apply_wound(const std::vector<Vec2>& cell_centers, Vec2 center,
                      double radius, double wound_factor = 2.0) {
        int nc = (int)cell_centers.size();
        for (int c = 0; c < nc; c++) {
            if ((cell_centers[c] - center).norm() < radius) {
                wound_flags[c] = true;
            }
        }
    }

    // Update wound channel factor (decays over time)
    void update_wound_channels(int nm, const std::vector<int>& mem_to_cell,
                                double wound_close_rate, double dt) {
        for (int m = 0; m < nm; m++) {
            int c = mem_to_cell[m];
            if (wound_flags[c]) {
                wound_channel_factor[m] = std::max(0.0,
                    wound_channel_factor[m] + (2.0 - wound_channel_factor[m]) * 0.1 * dt);
            }
            wound_channel_factor[m] *= (1.0 - wound_close_rate * dt);
        }
    }
};

// ============================================================================
// Full CSV Exporter (extended from betse.h write_csv)
// Exports comprehensive time-series data
// ============================================================================
struct CSVExporter {
    static void write_vmem_csv(const std::string& filename,
                                const TimeSeriesRecorder& recorder) {
        std::ofstream f(filename);
        if (!f) return;
        int n_cells = 0;
        if (!recorder.snapshots.empty() && !recorder.snapshots[0].vm_ave.empty())
            n_cells = (int)recorder.snapshots[0].vm_ave.size();

        f << "time_s,avg_Vmem_mV";
        for (int c = 0; c < std::min(n_cells, 10); c++)
            f << ",cell_" << c << "_Vmem_mV";
        f << "\n";

        for (auto& snap : recorder.snapshots) {
            double avg = 0;
            for (double v : snap.vm_ave) avg += v;
            if (!snap.vm_ave.empty()) avg /= snap.vm_ave.size();
            f << snap.time << "," << avg * 1000.0;
            for (int c = 0; c < std::min(n_cells, 10); c++) {
                f << "," << (c < (int)snap.vm_ave.size() ? snap.vm_ave[c] * 1000.0 : 0.0);
            }
            f << "\n";
        }
    }

    static void write_ions_csv(const std::string& filename,
                                const TimeSeriesRecorder& recorder) {
        std::ofstream f(filename);
        if (!f) return;
        f << "time_s";
        int ni = 0;
        if (!recorder.snapshots.empty())
            ni = (int)recorder.snapshots[0].cc_cells.size();
        for (int ion = 0; ion < ni; ion++)
            f << ",avg_" << ion_name(static_cast<Ion>(ion)) << "_mM";
        f << "\n";

        for (auto& snap : recorder.snapshots) {
            f << snap.time;
            for (int ion = 0; ion < ni; ion++) {
                double avg = 0;
                if (ion < (int)snap.cc_cells.size())
                    for (double c : snap.cc_cells[ion]) avg += c;
                if (ion < (int)snap.cc_cells.size() && !snap.cc_cells[ion].empty())
                    avg /= snap.cc_cells[ion].size();
                f << "," << avg;
            }
            f << "\n";
        }
    }

    static void write_molecules_csv(const std::string& filename,
                                     const TimeSeriesRecorder& recorder) {
        std::ofstream f(filename);
        if (!f) return;
        // Collect molecule names from first snapshot
        if (recorder.snapshots.empty()) return;
        auto& first = recorder.snapshots[0];
        std::vector<std::string> mol_names;
        for (auto& [name, _] : first.mol_cells) mol_names.push_back(name);

        f << "time_s";
        for (auto& name : mol_names) f << ",avg_" << name << "_mM";
        f << "\n";

        for (auto& snap : recorder.snapshots) {
            f << snap.time;
            for (auto& name : mol_names) {
                double avg = 0;
                auto it = snap.mol_cells.find(name);
                if (it != snap.mol_cells.end() && !it->second.empty()) {
                    for (double c : it->second) avg += c;
                    avg /= it->second.size();
                }
                f << "," << avg;
            }
            f << "\n";
        }
    }

    static void write_full_csv(const std::string& filename,
                                const TimeSeriesRecorder& recorder) {
        std::ofstream f(filename);
        if (!f) return;
        int ni = 0;
        if (!recorder.snapshots.empty())
            ni = (int)recorder.snapshots[0].cc_cells.size();

        f << "time_s,avg_Vmem_mV";
        for (int ion = 0; ion < ni; ion++)
            f << ",avg_" << ion_name(static_cast<Ion>(ion)) << "_mM";
        f << ",avg_P_Pa,avg_dx_m,avg_dy_m";
        f << "\n";

        for (auto& snap : recorder.snapshots) {
            double avg_v = 0;
            for (double v : snap.vm_ave) avg_v += v;
            if (!snap.vm_ave.empty()) avg_v /= snap.vm_ave.size();
            f << snap.time << "," << avg_v * 1000.0;

            for (int ion = 0; ion < ni; ion++) {
                double avg = 0;
                if (ion < (int)snap.cc_cells.size())
                    for (double c : snap.cc_cells[ion]) avg += c;
                if (ion < (int)snap.cc_cells.size() && !snap.cc_cells[ion].empty())
                    avg /= snap.cc_cells[ion].size();
                f << "," << avg;
            }

            double avg_p = 0, avg_dx = 0, avg_dy = 0;
            if (!snap.P_cells.empty()) {
                for (double p : snap.P_cells) avg_p += p;
                avg_p /= snap.P_cells.size();
            }
            if (!snap.d_cells_x.empty()) {
                for (double d : snap.d_cells_x) avg_dx += d;
                avg_dx /= snap.d_cells_x.size();
            }
            if (!snap.d_cells_y.empty()) {
                for (double d : snap.d_cells_y) avg_dy += d;
                avg_dy /= snap.d_cells_y.size();
            }
            f << "," << avg_p << "," << avg_dx << "," << avg_dy;
            f << "\n";
        }
    }
};

// ============================================================================
// Electrodiffusion solver modes (from sim.py)
// ============================================================================
enum class SolverType {
    BASIC,     // Simple Goldman flux (current implementation)
    FULL,      // Full Nernst-Planck with ECM
    FAST       // Simplified fast solver
};

// ============================================================================
// Full electrodiffusion solver for ECM (from sim.py update_ecm)
// Solves Nernst-Planck transport in the extracellular space
// ============================================================================
inline void solve_ecm_transport(
    int ni, int nx, int ny, double dx, double dt,
    std::vector<std::vector<double>>& cc_ecm,
    const std::vector<double>& E_env_x,
    const std::vector<double>& E_env_y,
    const std::vector<double>& u_env_x,
    const std::vector<double>& u_env_y,
    const std::vector<double>& D_env,
    const std::vector<int>& zs,
    const std::vector<double>& D_env_weight,
    double T,
    const std::vector<double>& c_env_bound,
    double sharpness = 1.0)
{
    int ne = nx * ny;
    for (int ion = 0; ion < ni; ion++) {
        auto& cc = cc_ecm[ion];
        if (cc.empty() || (int)cc.size() != ne) continue;

        // Set boundary conditions
        for (int j = 0; j < ny; j++) {
            cc[j * nx] = c_env_bound[ion];
            cc[j * nx + nx - 1] = c_env_bound[ion];
        }
        for (int i = 0; i < nx; i++) {
            cc[i] = c_env_bound[ion];
            cc[(ny-1) * nx + i] = c_env_bound[ion];
        }

        // Compute concentration gradient
        std::vector<double> gcx, gcy;
        fd::gradient(cc, nx, ny, dx, gcx, gcy);

        // Nernst-Planck flux
        double alpha = (D_env[ion] * zs[ion] * Q_ELECTRON) / (K_BOLTZMANN * T);
        std::vector<double> fx(ne), fy(ne);
        for (int k = 0; k < ne; k++) {
            double Dw = D_env[ion] * D_env_weight[k];
            double a = alpha * D_env_weight[k] / (D_env[ion] + 1e-30) * Dw;
            fx[k] = -Dw * gcx[k] - a * E_env_x[k] * cc[k];
            fy[k] = -Dw * gcy[k] - a * E_env_y[k] * cc[k];
            // Add advection
            if (!u_env_x.empty()) {
                fx[k] += u_env_x[k] * cc[k];
                fy[k] += u_env_y[k] * cc[k];
            }
        }

        // Divergence
        auto div_f = fd::divergence(fx, fy, nx, ny, dx, dx);

        // Update concentration
        for (int k = 0; k < ne; k++) {
            cc[k] -= div_f[k] * dt;
            cc[k] = std::max(0.0, cc[k]);
        }

        // Smooth if needed
        if (sharpness < 1.0)
            cc = fd::integrator(cc, nx, ny, dx, sharpness);
    }
}

// ============================================================================
// Concentration update helper (from sim_toolbox.py update_Co)
// Updates cell and env concentrations from membrane flux
// ============================================================================
inline void update_concentrations_from_flux(
    std::vector<double>& cc_cells,     // [cell]
    std::vector<double>& cc_env,       // [cell or ecm]
    const std::vector<double>& flux,   // [membrane] (+ = into cell)
    int nc, int nm,
    const std::vector<int>& mem_to_cell,
    const std::vector<double>& mem_sa,
    const std::vector<double>& cell_vol,
    double dt, double env_vol_factor = 10.0)
{
    // Cell update
    std::vector<double> delta(nc, 0.0);
    for (int m = 0; m < nm; m++) {
        int c = mem_to_cell[m];
        delta[c] += flux[m] * mem_sa[m];
    }
    for (int c = 0; c < nc; c++) {
        cc_cells[c] += dt * delta[c] / cell_vol[c];
        cc_cells[c] = std::max(0.0, cc_cells[c]);
    }

    // Environment update (simplified well-mixed)
    double total_flux = 0;
    for (int m = 0; m < nm; m++)
        total_flux -= flux[m] * mem_sa[m];
    double env_vol = cell_vol[0] * nc * env_vol_factor;
    double delta_env = total_flux / env_vol;
    for (int c = 0; c < (int)cc_env.size(); c++) {
        cc_env[c] += dt * delta_env;
        cc_env[c] = std::max(0.0, cc_env[c]);
    }
}

// ============================================================================
// Nernst potential calculator
// ============================================================================
inline double nernst_potential(double c_in, double c_out, int z, double T) {
    if (c_in < 1e-30) c_in = 1e-30;
    if (c_out < 1e-30) c_out = 1e-30;
    return (R_GAS * T) / (z * F_FARADAY) * std::log(c_out / c_in);
}

// ============================================================================
// Simulation phase runner (from phase/phasecls.py SimPhase)
// Manages seed, init, and sim phases with proper ordering
// ============================================================================
struct SimPhaseRunner {
    SimPhaseKind current_phase = SimPhaseKind::SEED;
    PhaseCallbacks callbacks;
    TimeSeriesRecorder recorder;
    std::vector<ScheduledEvent> events;
    TissueHandler tissue;

    double progress = 0.0;  // 0.0 to 1.0

    // Check if phase transition is needed
    bool is_seed() const { return current_phase == SimPhaseKind::SEED; }
    bool is_init() const { return current_phase == SimPhaseKind::INIT; }
    bool is_sim() const { return current_phase == SimPhaseKind::SIM; }

    void advance_phase() {
        if (current_phase == SimPhaseKind::SEED)
            current_phase = SimPhaseKind::INIT;
        else if (current_phase == SimPhaseKind::INIT)
            current_phase = SimPhaseKind::SIM;
    }

    // Process scheduled events at current time
    void process_events(double time, int nc, int nm,
                         const std::vector<int>& mem_to_cell,
                         const std::vector<int>& cell_profile_ids,
                         std::vector<double>& Vmem,
                         std::vector<std::vector<double>>& cc_cells,
                         std::vector<std::vector<double>>& Dm_cells_vec,
                         double& T) {
        for (auto& ev : events) {
            if (!ev.is_active(time)) continue;
            double val = ev.get_value(time);

            switch (ev.type) {
                case EventType::VOLTAGE_CLAMP:
                    for (int m = 0; m < nm; m++) {
                        int c = mem_to_cell[m];
                        if (ev.target_profile_id < 0 || cell_profile_ids[c] == ev.target_profile_id)
                            Vmem[m] = val;
                    }
                    break;

                case EventType::CONCENTRATION_CHANGE:
                    if (ev.target_ion >= 0 && ev.target_ion < (int)cc_cells.size()) {
                        for (int c = 0; c < nc; c++) {
                            if (ev.target_profile_id < 0 || cell_profile_ids[c] == ev.target_profile_id)
                                cc_cells[ev.target_ion][c] = val;
                        }
                    }
                    break;

                case EventType::PERMEABILITY_CHANGE:
                    if (ev.target_ion >= 0 && ev.target_ion < (int)Dm_cells_vec.size()) {
                        for (int m = 0; m < nm; m++) {
                            int c = mem_to_cell[m];
                            if (ev.target_profile_id < 0 || cell_profile_ids[c] == ev.target_profile_id)
                                Dm_cells_vec[ev.target_ion][m] *= val;
                        }
                    }
                    break;

                case EventType::TEMPERATURE_CHANGE:
                    T = val;
                    break;

                case EventType::BOUNDARY_VOLTAGE:
                    // Applied at ECM boundaries (handled separately)
                    break;

                case EventType::CUSTOM:
                    if (ev.custom_func) ev.custom_func(time);
                    break;

                default:
                    break;
            }
        }
    }
};

} // namespace betse
