// BETSE C++ Port - Data Export and Visualization Module
// Complete data pipeline, CSV export, data extraction, and
// visualization data structures.
// Ported from: science/pipe/export/*, science/visual/*, science/pipe/piperun.py,
//              science/visual/visabc.py, science/visual/layer/*
#pragma once

#include "betse_types.h"
#include "betse_enums.h"
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <functional>

namespace betse {

// ============================================================================
// Data Snapshot (extended from betse_phase.h)
// Single time point of all simulation data
// ============================================================================
struct DataSnapshot {
    double time = 0;
    int step = 0;

    // Voltage
    std::vector<double> Vmem;
    std::vector<double> vm_ave;       // Vmem averaged to cells
    std::vector<double> v_env;

    // Concentrations [ion][cell or env]
    std::vector<std::vector<double>> cc_cells;
    std::vector<std::vector<double>> cc_env;

    // Charge
    std::vector<double> rho_cells;

    // Current
    std::vector<double> J_cell_x, J_cell_y;
    std::vector<double> J_env_x, J_env_y;
    std::vector<double> J_mem;
    std::vector<double> J_gj;

    // Electric field
    std::vector<double> E_cell_x, E_cell_y;
    std::vector<double> E_env_x, E_env_y;

    // Pressure
    std::vector<double> P_cells;

    // Deformation
    std::vector<double> d_cells_x, d_cells_y;

    // Flow
    std::vector<double> u_cells_x, u_cells_y;
    std::vector<double> u_env_x, u_env_y;

    // Pump rates
    std::vector<double> pump_Na, pump_K, pump_Ca;

    // GRN molecule concentrations
    std::map<std::string, std::vector<double>> mol_cells;
    std::map<std::string, std::vector<double>> mol_env;

    // Channel open probabilities
    std::map<std::string, std::vector<double>> channel_P_open;

    // ER
    std::vector<double> er_Ca;
    std::vector<double> er_V;

    // Mitochondria
    std::vector<double> mito_V;
    std::vector<double> mito_Ca;

    // Magnetic field
    std::vector<double> B_field;

    // Microtubule angles
    std::vector<double> mt_theta;
};

// ============================================================================
// Time Series Database (collects DataSnapshots)
// ============================================================================
struct TimeSeriesDB {
    std::vector<DataSnapshot> snapshots;
    std::vector<double> time_points;

    void clear() {
        snapshots.clear();
        time_points.clear();
    }

    void add(const DataSnapshot& snap) {
        snapshots.push_back(snap);
        time_points.push_back(snap.time);
    }

    int size() const { return (int)snapshots.size(); }

    // Extract a scalar time series for a cell
    std::vector<double> cell_vmem_series(int cell_id) const {
        std::vector<double> r;
        for (auto& s : snapshots) {
            if (cell_id < (int)s.vm_ave.size())
                r.push_back(s.vm_ave[cell_id]);
            else r.push_back(0.0);
        }
        return r;
    }

    // Average Vmem time series
    std::vector<double> avg_vmem_series() const {
        std::vector<double> r;
        for (auto& s : snapshots) {
            double avg = 0;
            if (!s.vm_ave.empty()) {
                for (double v : s.vm_ave) avg += v;
                avg /= s.vm_ave.size();
            }
            r.push_back(avg);
        }
        return r;
    }

    // Average ion concentration series
    std::vector<double> avg_ion_series(int ion) const {
        std::vector<double> r;
        for (auto& s : snapshots) {
            double avg = 0;
            if (ion < (int)s.cc_cells.size() && !s.cc_cells[ion].empty()) {
                for (double c : s.cc_cells[ion]) avg += c;
                avg /= s.cc_cells[ion].size();
            }
            r.push_back(avg);
        }
        return r;
    }

    // Average molecule concentration series
    std::vector<double> avg_molecule_series(const std::string& name) const {
        std::vector<double> r;
        for (auto& s : snapshots) {
            double avg = 0;
            auto it = s.mol_cells.find(name);
            if (it != s.mol_cells.end() && !it->second.empty()) {
                for (double c : it->second) avg += c;
                avg /= it->second.size();
            }
            r.push_back(avg);
        }
        return r;
    }

    // Min/max of a channel across all snapshots
    void vmem_range(double& vmin, double& vmax) const {
        vmin = 1e30; vmax = -1e30;
        for (auto& s : snapshots) {
            for (double v : s.vm_ave) {
                vmin = std::min(vmin, v);
                vmax = std::max(vmax, v);
            }
        }
    }
};

// ============================================================================
// Export Pipeline Item (from pipe/export/pipeexps.py)
// Defines what to export and how
// ============================================================================
struct ExportPipelineItem {
    std::string name;
    SimExportType export_type = SimExportType::CSV;
    SimDataChannel channel = SimDataChannel::VMEM;
    std::string molecule_name;  // for MOLECULE channel
    bool enabled = true;

    // CSV-specific
    std::string csv_filename;

    // Plot-specific
    std::string plot_filename;
    std::string colormap = "RdBu_r";
    bool auto_scale = true;
    double vmin = -70.0, vmax = 10.0;

    // Animation-specific
    int fps = 15;
    std::string anim_filename;
    std::string codec = "ffv1";
};

// ============================================================================
// CSV Writer (from pipe/export/pipeexpcsv.py)
// Comprehensive CSV export for all data types
// ============================================================================
struct CSVWriter {

    // Write generic time series CSV
    static void write_time_series(const std::string& filename,
                                   const std::vector<double>& times,
                                   const std::vector<std::string>& headers,
                                   const std::vector<std::vector<double>>& columns) {
        std::ofstream f(filename);
        if (!f) return;
        f << std::setprecision(8);
        f << "time_s";
        for (auto& h : headers) f << "," << h;
        f << "\n";
        for (size_t t = 0; t < times.size(); t++) {
            f << times[t];
            for (size_t c = 0; c < columns.size(); c++) {
                f << "," << (t < columns[c].size() ? columns[c][t] : 0.0);
            }
            f << "\n";
        }
    }

    // Write Vmem CSV
    static void write_vmem(const std::string& filename,
                            const TimeSeriesDB& db,
                            int max_cells = 10) {
        std::ofstream f(filename);
        if (!f) return;
        f << std::setprecision(8);
        int nc = 0;
        if (!db.snapshots.empty() && !db.snapshots[0].vm_ave.empty())
            nc = (int)db.snapshots[0].vm_ave.size();
        int show = std::min(nc, max_cells);

        f << "time_s,avg_Vmem_mV";
        for (int c = 0; c < show; c++) f << ",cell_" << c << "_Vmem_mV";
        f << "\n";

        for (auto& s : db.snapshots) {
            double avg = 0;
            for (double v : s.vm_ave) avg += v;
            if (!s.vm_ave.empty()) avg /= s.vm_ave.size();
            f << s.time << "," << avg * 1000.0;
            for (int c = 0; c < show; c++) {
                double v = (c < (int)s.vm_ave.size()) ? s.vm_ave[c] * 1000.0 : 0.0;
                f << "," << v;
            }
            f << "\n";
        }
    }

    // Write ion concentrations CSV
    static void write_ions(const std::string& filename,
                            const TimeSeriesDB& db) {
        std::ofstream f(filename);
        if (!f) return;
        f << std::setprecision(8);
        int ni = 0;
        if (!db.snapshots.empty())
            ni = (int)db.snapshots[0].cc_cells.size();

        f << "time_s";
        const char* ion_names[] = {"Na", "K", "Cl", "Ca", "H", "P"};
        for (int ion = 0; ion < ni && ion < 6; ion++)
            f << ",avg_" << ion_names[ion] << "_mM";
        f << "\n";

        for (auto& s : db.snapshots) {
            f << s.time;
            for (int ion = 0; ion < ni && ion < 6; ion++) {
                double avg = 0;
                if (ion < (int)s.cc_cells.size() && !s.cc_cells[ion].empty()) {
                    for (double c : s.cc_cells[ion]) avg += c;
                    avg /= s.cc_cells[ion].size();
                }
                f << "," << avg;
            }
            f << "\n";
        }
    }

    // Write molecule concentrations CSV
    static void write_molecules(const std::string& filename,
                                 const TimeSeriesDB& db) {
        std::ofstream f(filename);
        if (!f || db.snapshots.empty()) return;
        f << std::setprecision(8);

        std::vector<std::string> names;
        for (auto& [name, _] : db.snapshots[0].mol_cells)
            names.push_back(name);

        f << "time_s";
        for (auto& n : names) f << ",avg_" << n << "_mM";
        f << "\n";

        for (auto& s : db.snapshots) {
            f << s.time;
            for (auto& n : names) {
                double avg = 0;
                auto it = s.mol_cells.find(n);
                if (it != s.mol_cells.end() && !it->second.empty()) {
                    for (double c : it->second) avg += c;
                    avg /= it->second.size();
                }
                f << "," << avg;
            }
            f << "\n";
        }
    }

    // Write current density CSV
    static void write_current(const std::string& filename,
                               const TimeSeriesDB& db) {
        std::ofstream f(filename);
        if (!f) return;
        f << std::setprecision(8);
        f << "time_s,avg_Jx_A_m2,avg_Jy_A_m2,avg_Jmag_A_m2\n";

        for (auto& s : db.snapshots) {
            double avg_jx = 0, avg_jy = 0;
            int nc = (int)s.J_cell_x.size();
            for (int c = 0; c < nc; c++) {
                avg_jx += s.J_cell_x[c];
                avg_jy += s.J_cell_y[c];
            }
            if (nc > 0) { avg_jx /= nc; avg_jy /= nc; }
            double mag = std::sqrt(avg_jx * avg_jx + avg_jy * avg_jy);
            f << s.time << "," << avg_jx << "," << avg_jy << "," << mag << "\n";
        }
    }

    // Write pressure CSV
    static void write_pressure(const std::string& filename,
                                const TimeSeriesDB& db) {
        std::ofstream f(filename);
        if (!f) return;
        f << std::setprecision(8);
        f << "time_s,avg_P_Pa,min_P_Pa,max_P_Pa\n";

        for (auto& s : db.snapshots) {
            double avg = 0, min_p = 1e30, max_p = -1e30;
            int nc = (int)s.P_cells.size();
            for (int c = 0; c < nc; c++) {
                avg += s.P_cells[c];
                min_p = std::min(min_p, s.P_cells[c]);
                max_p = std::max(max_p, s.P_cells[c]);
            }
            if (nc > 0) avg /= nc;
            f << s.time << "," << avg << "," << min_p << "," << max_p << "\n";
        }
    }

    // Write comprehensive data CSV (all major quantities)
    static void write_full(const std::string& filename,
                            const TimeSeriesDB& db) {
        std::ofstream f(filename);
        if (!f) return;
        f << std::setprecision(8);
        int ni = 0;
        if (!db.snapshots.empty())
            ni = (int)db.snapshots[0].cc_cells.size();

        f << "time_s,avg_Vmem_mV";
        const char* ion_names[] = {"Na", "K", "Cl", "Ca", "H", "P"};
        for (int ion = 0; ion < ni && ion < 6; ion++)
            f << ",avg_" << ion_names[ion] << "_mM";
        f << ",avg_P_Pa,avg_Jx,avg_Jy,avg_dx,avg_dy";
        f << "\n";

        for (auto& s : db.snapshots) {
            // Vmem
            double avg_v = 0;
            for (double v : s.vm_ave) avg_v += v;
            if (!s.vm_ave.empty()) avg_v /= s.vm_ave.size();
            f << s.time << "," << avg_v * 1000.0;

            // Ions
            for (int ion = 0; ion < ni && ion < 6; ion++) {
                double avg = 0;
                if (ion < (int)s.cc_cells.size() && !s.cc_cells[ion].empty()) {
                    for (double c : s.cc_cells[ion]) avg += c;
                    avg /= s.cc_cells[ion].size();
                }
                f << "," << avg;
            }

            // Pressure, current, deformation
            auto avg_of = [](const std::vector<double>& v) {
                if (v.empty()) return 0.0;
                double s = 0; for (double x : v) s += x; return s / v.size();
            };
            f << "," << avg_of(s.P_cells);
            f << "," << avg_of(s.J_cell_x) << "," << avg_of(s.J_cell_y);
            f << "," << avg_of(s.d_cells_x) << "," << avg_of(s.d_cells_y);
            f << "\n";
        }
    }

    // Write spatial snapshot at a given time index
    static void write_spatial_snapshot(const std::string& filename,
                                       const DataSnapshot& snap,
                                       const std::vector<Vec2>& cell_centers) {
        std::ofstream f(filename);
        if (!f) return;
        f << std::setprecision(8);
        int nc = (int)cell_centers.size();
        f << "cell_id,x_m,y_m,Vmem_mV";
        int ni = (int)snap.cc_cells.size();
        const char* ion_names[] = {"Na", "K", "Cl", "Ca", "H", "P"};
        for (int ion = 0; ion < ni && ion < 6; ion++)
            f << "," << ion_names[ion] << "_mM";
        f << ",P_Pa\n";

        for (int c = 0; c < nc; c++) {
            double vm = (c < (int)snap.vm_ave.size()) ? snap.vm_ave[c] * 1000.0 : 0.0;
            f << c << "," << cell_centers[c].x << "," << cell_centers[c].y
              << "," << vm;
            for (int ion = 0; ion < ni && ion < 6; ion++) {
                double cc = (ion < (int)snap.cc_cells.size() &&
                             c < (int)snap.cc_cells[ion].size())
                            ? snap.cc_cells[ion][c] : 0.0;
                f << "," << cc;
            }
            double p = (c < (int)snap.P_cells.size()) ? snap.P_cells[c] : 0.0;
            f << "," << p;
            f << "\n";
        }
    }
};

// ============================================================================
// Export Pipeline Runner (from pipe/piperun.py)
// Runs all configured exports after simulation
// ============================================================================
struct ExportPipeline {
    std::vector<ExportPipelineItem> items;
    std::string output_dir = "RESULTS";

    void run(const TimeSeriesDB& db,
             const std::vector<Vec2>& cell_centers) const {
        for (auto& item : items) {
            if (!item.enabled) continue;

            switch (item.export_type) {
                case SimExportType::CSV: {
                    std::string path = output_dir + "/" + item.csv_filename;
                    switch (item.channel) {
                        case SimDataChannel::VMEM:
                            CSVWriter::write_vmem(path, db);
                            break;
                        case SimDataChannel::CC_NA:
                        case SimDataChannel::CC_K:
                        case SimDataChannel::CC_CL:
                        case SimDataChannel::CC_CA:
                        case SimDataChannel::CC_H:
                            CSVWriter::write_ions(path, db);
                            break;
                        case SimDataChannel::MOLECULE:
                            CSVWriter::write_molecules(path, db);
                            break;
                        case SimDataChannel::J_CELL:
                            CSVWriter::write_current(path, db);
                            break;
                        case SimDataChannel::P_CELLS:
                            CSVWriter::write_pressure(path, db);
                            break;
                        default:
                            CSVWriter::write_full(path, db);
                            break;
                    }
                    break;
                }
                case SimExportType::PLOT:
                case SimExportType::ANIM:
                    // Plot/animation export would require a plotting backend
                    // (matplotlib equivalent). Store data for external processing.
                    break;
            }
        }
    }
};

// ============================================================================
// Visualization Data Extractor
// Extracts data arrays from snapshots for external visualization
// ============================================================================
struct VisDataExtractor {

    // Extract scalar field for cells at a given snapshot
    static std::vector<double> extract_cell_scalar(
            const DataSnapshot& snap, SimDataChannel ch, int ion_idx = 0) {
        switch (ch) {
            case SimDataChannel::VMEM:
                return snap.vm_ave;
            case SimDataChannel::CC_NA:
                return (0 < (int)snap.cc_cells.size()) ? snap.cc_cells[0] : std::vector<double>{};
            case SimDataChannel::CC_K:
                return (1 < (int)snap.cc_cells.size()) ? snap.cc_cells[1] : std::vector<double>{};
            case SimDataChannel::CC_CL:
                return (2 < (int)snap.cc_cells.size()) ? snap.cc_cells[2] : std::vector<double>{};
            case SimDataChannel::CC_CA:
                return (3 < (int)snap.cc_cells.size()) ? snap.cc_cells[3] : std::vector<double>{};
            case SimDataChannel::CC_H:
                return (4 < (int)snap.cc_cells.size()) ? snap.cc_cells[4] : std::vector<double>{};
            case SimDataChannel::RHO_CELLS:
                return snap.rho_cells;
            case SimDataChannel::P_CELLS:
                return snap.P_cells;
            case SimDataChannel::ER_CA:
                return snap.er_Ca;
            case SimDataChannel::ER_V:
                return snap.er_V;
            case SimDataChannel::MITO_V:
                return snap.mito_V;
            case SimDataChannel::B_FIELD:
                return snap.B_field;
            case SimDataChannel::PH_CELL: {
                // pH = -log10([H+]) where [H+] is in mM -> convert to M
                std::vector<double> pH;
                if (4 < (int)snap.cc_cells.size()) {
                    pH.resize(snap.cc_cells[4].size());
                    for (size_t i = 0; i < pH.size(); i++) {
                        double h = snap.cc_cells[4][i] * 1e-3; // mM to M
                        pH[i] = (h > 0) ? -std::log10(h) : 7.0;
                    }
                }
                return pH;
            }
            default:
                return {};
        }
    }

    // Extract vector field for cells
    static void extract_cell_vector(
            const DataSnapshot& snap, SimDataChannel ch,
            std::vector<double>& vx, std::vector<double>& vy) {
        switch (ch) {
            case SimDataChannel::J_CELL:
                vx = snap.J_cell_x; vy = snap.J_cell_y; break;
            case SimDataChannel::E_CELL:
                vx = snap.E_cell_x; vy = snap.E_cell_y; break;
            case SimDataChannel::DEFORM:
                vx = snap.d_cells_x; vy = snap.d_cells_y; break;
            case SimDataChannel::FLOW_CELLS:
                vx = snap.u_cells_x; vy = snap.u_cells_y; break;
            default:
                vx.clear(); vy.clear(); break;
        }
    }

    // Compute magnitude of a vector field
    static std::vector<double> vector_magnitude(
            const std::vector<double>& vx, const std::vector<double>& vy) {
        int n = (int)std::min(vx.size(), vy.size());
        std::vector<double> mag(n);
        for (int i = 0; i < n; i++)
            mag[i] = std::sqrt(vx[i] * vx[i] + vy[i] * vy[i]);
        return mag;
    }
};

} // namespace betse
