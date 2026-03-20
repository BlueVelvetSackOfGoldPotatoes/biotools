// BETSE C++ Port - Tissue Management Module
// Tissue profiles, heterogeneity, tissue picking, cutting/wounding,
// targeted interventions, tissue growth, and channelpedia reference data.
// Ported from: tissue/tisprofile.py, tissue/tishandler.py,
//              tissue/event/tisevecut.py, tissue/event/tisevevolt.py,
//              tissue/event/tiseveabc.py, tissue/picker/tispickcls.py,
//              tissue/channels_o.py
#pragma once

#include "betse_types.h"
#include "betse_enums.h"
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <functional>

namespace betse {

// ============================================================================
// Tissue Picker (from tissue/picker/tispickcls.py TissuePickerABC)
// Selects a subset of cells for a tissue profile
// ============================================================================
struct TissuePicker {
    CellsPickerType type = CellsPickerType::ALL;

    // For CIRCULAR/RECTANGULAR (spatial)
    Vec2 center = {0, 0};
    double radius = 0;
    double width = 0, height = 0;

    // For INDICES
    std::vector<int> cell_indices;

    // For PERCENT
    double percent = 100.0;

    // For IMAGE (bitmap path - not used in C++ port but stored)
    std::string image_path;

    // For COLOR (SVG color match)
    std::string color_hex;

    // Pick cells from a cell cluster
    std::vector<int> pick(const std::vector<Vec2>& cell_centers,
                          int num_cells, std::mt19937& rng) const {
        std::vector<int> result;
        switch (type) {
            case CellsPickerType::ALL:
                result.resize(num_cells);
                for (int i = 0; i < num_cells; i++) result[i] = i;
                break;

            case CellsPickerType::INDICES:
                result = cell_indices;
                break;

            case CellsPickerType::PERCENT: {
                std::vector<int> all(num_cells);
                for (int i = 0; i < num_cells; i++) all[i] = i;
                std::shuffle(all.begin(), all.end(), rng);
                int count = std::max(1, (int)(num_cells * percent / 100.0));
                result.assign(all.begin(), all.begin() + count);
                std::sort(result.begin(), result.end());
                break;
            }

            default:
                // For IMAGE and COLOR, fall back to spatial selection
                // using center/radius if provided, else ALL
                if (radius > 0) {
                    for (int i = 0; i < num_cells; i++) {
                        if ((cell_centers[i] - center).norm() < radius)
                            result.push_back(i);
                    }
                } else {
                    result.resize(num_cells);
                    for (int i = 0; i < num_cells; i++) result[i] = i;
                }
                break;
        }
        return result;
    }
};

// ============================================================================
// Tissue Profile (from tissue/tisprofile.py TissueProfile)
// Full tissue region with per-ion membrane diffusion constants,
// gap junction settings, and spatial heterogeneity
// ============================================================================
struct TissueProfileFull {
    int id = 0;
    std::string name = "default";
    int z_order = 1;
    TissuePicker picker;

    // Gap junction configuration
    GJConnectivityType gj_connectivity = GJConnectivityType::NORMAL;
    double gj_surface_override = -1.0;  // -1 = use global default

    // Per-ion membrane diffusion constants [m/s]
    double Dm_Na  = 1.0e-18;
    double Dm_K   = 1.0e-16;
    double Dm_Cl  = 1.0e-18;
    double Dm_Ca  = 1.0e-20;
    double Dm_H   = 1.0e-18;
    double Dm_M   = 1.0e-18;
    double Dm_P   = 0.0;

    // Get Dm for an ion by index
    double get_Dm(int ion_idx) const {
        switch (ion_idx) {
            case 0: return Dm_Na;
            case 1: return Dm_K;
            case 2: return Dm_Cl;
            case 3: return Dm_Ca;
            case 4: return Dm_H;
            case 5: return Dm_P;
            default: return 0.0;
        }
    }

    // Resolved cell indices (populated after picking)
    std::vector<int> cell_indices;
    std::vector<int> mem_indices;  // membrane indices belonging to this profile
};

// ============================================================================
// Tissue Event Base (from tissue/event/tiseveabc.py SimEventABC)
// ============================================================================
struct TissueEvent {
    enum class Type {
        CUTTING,            // Remove cells
        VOLTAGE_PULSE,      // Apply voltage to boundary
        CURRENT_INJECTION,  // Inject current into cells
        PRESSURE_PULSE,     // Apply mechanical pressure
        GLOBAL_INTERVENTION,// Modify global parameters
        WOUND,              // Wound formation
        BLOCK_GJ,           // Block gap junctions in a region
        UNBLOCK_GJ,         // Restore gap junctions
        DM_CHANGE,          // Change membrane permeability
        CONCENTRATION_SET   // Set ion concentration
    };

    Type type = Type::CUTTING;

    // Timing (from tiseveabc.py SimEventSpikeABC / SimEventPulseABC)
    double time_step = 0;               // For spike events (single moment)
    double start_time = 0;              // For pulse events
    double stop_time = 0;
    double rate = 1.0;                  // Rate of change for pulse

    // Spatial targeting
    TissuePicker picker;
    int target_profile_id = -1;         // -1 = all

    // Value
    double value = 0;
    int target_ion = -1;                // For concentration/Dm changes

    // Voltage pulse (from tisevevolt.py)
    BoundarySide positive_boundary = BoundarySide::TOP;
    BoundarySide negative_boundary = BoundarySide::BOTTOM;

    // Cutting (from tisevecut.py)
    Vec2 cut_center = {0, 0};
    double cut_radius = 0;

    // Custom callback
    std::function<void(double time)> custom_func;

    bool is_active(double time) const {
        if (type == Type::CUTTING || type == Type::WOUND)
            return std::abs(time - time_step) < 1e-10;
        return time >= start_time && time <= stop_time;
    }

    // Compute pulse effector (from toolbox.py pulse function)
    double pulse_effector(double time) const {
        if (time < start_time || time > stop_time) return 0.0;
        double t_mid = (start_time + stop_time) / 2.0;
        double width = (stop_time - start_time) / 2.0;
        if (width < 1e-30) return value;
        double x = (time - t_mid) / width;
        // Smooth pulse using tanh ramp
        double ramp_up = 0.5 * (1.0 + std::tanh(rate * (time - start_time)));
        double ramp_down = 0.5 * (1.0 + std::tanh(rate * (stop_time - time)));
        return value * ramp_up * ramp_down;
    }
};

// ============================================================================
// Tissue Heterogeneity Manager (from tissue/tishandler.py TissueHandler)
// Extended version that manages full tissue profiles, cutting, wounding
// ============================================================================
struct TissueManager {
    std::vector<TissueProfileFull> profiles;
    std::vector<TissueEvent> events;

    // Per-cell profile assignment
    std::vector<int> cell_profile_ids;

    // Per-membrane profile-based Dm values [ion][membrane]
    std::vector<std::vector<double>> Dm_profile;

    // Boundary cell/membrane flags
    std::vector<bool> bflags_cells;
    std::vector<bool> bflags_mems;
    std::vector<int> boundary_cells;
    std::vector<int> boundary_mems;

    // Wound state
    std::vector<bool> wound_flags;
    std::vector<double> wound_channel_factor;

    // GJ block state per membrane (1.0 = fully open, 0.0 = blocked)
    std::vector<double> gj_block;

    // Boundary voltage (for ECM simulations)
    // Indexed by BoundarySide: T=0, B=1, L=2, R=3
    double bound_V[4] = {0, 0, 0, 0};

    void init(int nc, int nm, int ni,
              const std::vector<Vec2>& cell_centers,
              const std::vector<int>& mem_to_cell,
              double tissue_radius, std::mt19937& rng) {
        cell_profile_ids.assign(nc, 0);
        bflags_cells.assign(nc, false);
        bflags_mems.assign(nm, false);
        wound_flags.assign(nc, false);
        wound_channel_factor.assign(nm, 0.0);
        gj_block.assign(nm, 1.0);

        // Mark boundary cells
        double boundary_thresh = tissue_radius * 0.85;
        for (int c = 0; c < nc; c++) {
            if (cell_centers[c].norm() > boundary_thresh) {
                bflags_cells[c] = true;
                boundary_cells.push_back(c);
            }
        }
        for (int m = 0; m < nm; m++) {
            if (bflags_cells[mem_to_cell[m]]) {
                bflags_mems[m] = true;
                boundary_mems.push_back(m);
            }
        }

        // Assign cells to profiles
        for (auto& prof : profiles) {
            prof.cell_indices = prof.picker.pick(cell_centers, nc, rng);
            for (int c : prof.cell_indices) {
                if (c >= 0 && c < nc)
                    cell_profile_ids[c] = prof.id;
            }
            // Resolve membrane indices
            prof.mem_indices.clear();
            for (int m = 0; m < nm; m++) {
                if (cell_profile_ids[mem_to_cell[m]] == prof.id)
                    prof.mem_indices.push_back(m);
            }
        }

        // Initialize per-ion, per-membrane Dm from profiles
        Dm_profile.resize(ni, std::vector<double>(nm));
        for (int m = 0; m < nm; m++) {
            int c = mem_to_cell[m];
            int pid = cell_profile_ids[c];
            // Find profile
            const TissueProfileFull* prof = nullptr;
            for (const auto& p : profiles) {
                if (p.id == pid) { prof = &p; break; }
            }
            for (int ion = 0; ion < ni; ion++) {
                Dm_profile[ion][m] = prof ? prof->get_Dm(ion) : 1.0e-18;
            }
        }
    }

    // Apply wound to cells in a region
    void apply_wound(const std::vector<Vec2>& cell_centers,
                     Vec2 center, double radius,
                     double wound_factor = 2.0) {
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
                    wound_channel_factor[m] +
                    (2.0 - wound_channel_factor[m]) * 0.1 * dt);
            }
            wound_channel_factor[m] *= (1.0 - wound_close_rate * dt);
        }
    }

    // Block GJs in a region
    void block_gj_region(const std::vector<Vec2>& mem_midpoints,
                          int nm, Vec2 center, double radius,
                          double block_factor = 0.0) {
        for (int m = 0; m < nm; m++) {
            if ((mem_midpoints[m] - center).norm() < radius) {
                gj_block[m] = block_factor;
            }
        }
    }

    // Unblock all GJs
    void unblock_all_gj(int nm) {
        gj_block.assign(nm, 1.0);
    }

    // Get cells to cut at the current time
    std::vector<int> get_cells_to_cut(double time,
                                       const std::vector<Vec2>& cell_centers,
                                       int nc, double dt) const {
        std::vector<int> to_remove;
        for (const auto& ev : events) {
            if (ev.type == TissueEvent::Type::CUTTING &&
                std::abs(time - ev.time_step) < dt * 0.5) {
                for (int c = 0; c < nc; c++) {
                    if ((cell_centers[c] - ev.cut_center).norm() < ev.cut_radius)
                        to_remove.push_back(c);
                }
            }
        }
        return to_remove;
    }

    // Apply boundary voltage events (for ECM)
    void apply_boundary_voltage_events(double time) {
        for (const auto& ev : events) {
            if (ev.type == TissueEvent::Type::VOLTAGE_PULSE && ev.is_active(time)) {
                double v = ev.pulse_effector(time);
                bound_V[static_cast<int>(ev.positive_boundary)] = v;
                bound_V[static_cast<int>(ev.negative_boundary)] = -v;
            }
        }
    }

    // Process Dm change events
    void apply_dm_change_events(double time, int nm,
                                 const std::vector<int>& mem_to_cell) {
        for (const auto& ev : events) {
            if (ev.type == TissueEvent::Type::DM_CHANGE && ev.is_active(time)) {
                double factor = ev.pulse_effector(time) / (ev.value + 1e-30);
                for (int m = 0; m < nm; m++) {
                    int c = mem_to_cell[m];
                    bool target = (ev.target_profile_id < 0 ||
                                   cell_profile_ids[c] == ev.target_profile_id);
                    if (target && ev.target_ion >= 0 &&
                        ev.target_ion < (int)Dm_profile.size()) {
                        Dm_profile[ev.target_ion][m] *= factor;
                    }
                }
            }
        }
    }
};

// ============================================================================
// Tissue Growth Model (from sim.py growth/proliferation code)
// Simplified tissue growth through cell division
// ============================================================================
struct TissueGrowth {
    bool enabled = false;
    double growth_rate = 0.0;           // cells per second
    double max_cells = 10000;
    double min_division_volume = 0.0;   // minimum volume to divide
    double division_asymmetry = 0.5;    // 0.5 = symmetric

    // Cell age tracking
    std::vector<double> cell_age;
    double age_threshold = 100.0;       // seconds before eligible to divide

    void init(int nc) {
        cell_age.assign(nc, 0.0);
    }

    void update_ages(int nc, double dt) {
        for (int c = 0; c < nc; c++)
            cell_age[c] += dt;
    }

    // Get cells eligible for division
    std::vector<int> get_dividing_cells(int nc, double dt,
                                         const std::vector<double>& cell_vol,
                                         std::mt19937& rng) const {
        if (!enabled) return {};
        std::vector<int> dividers;
        double prob = growth_rate * dt;
        std::uniform_real_distribution<double> dice(0.0, 1.0);
        for (int c = 0; c < nc && nc + (int)dividers.size() < (int)max_cells; c++) {
            if (cell_age[c] >= age_threshold &&
                cell_vol[c] >= min_division_volume &&
                dice(rng) < prob) {
                dividers.push_back(c);
            }
        }
        return dividers;
    }
};

// ============================================================================
// Ion Profile Presets (from parameters.py ion profile definitions)
// Preset concentration profiles for different organism types
// ============================================================================
struct IonProfilePreset {
    double Na_cell, Na_env;
    double K_cell, K_env;
    double Cl_cell, Cl_env;
    double Ca_cell, Ca_env;
    double H_cell, H_env;
    double P_cell, P_env;
};

inline IonProfilePreset get_ion_profile(IonProfileType type) {
    switch (type) {
        case IonProfileType::BASIC:
            return {12.0, 145.0,  140.0, 5.0,  0.0, 0.0,
                    0.0, 0.0,     0.0, 0.0,    135.0, 10.0};
        case IonProfileType::BASIC_CA:
            return {12.0, 145.0,  140.0, 5.0,  0.0, 0.0,
                    1e-4, 2.0,    0.0, 0.0,    135.0, 10.0};
        case IonProfileType::MAMMAL:
            return {12.0, 145.0,  140.0, 5.0,  4.0, 110.0,
                    1e-4, 2.0,    6.3e-5, 4e-5, 135.0, 10.0};
        case IonProfileType::AMPHIBIAN:
            return {10.0, 110.0,  120.0, 2.5,  2.5, 80.0,
                    1e-4, 1.9,    6.3e-5, 4e-5, 110.0, 5.0};
        case IonProfileType::CUSTOM:
        default:
            return {12.0, 145.0,  140.0, 5.0,  4.0, 110.0,
                    1e-4, 2.0,    6.3e-5, 4e-5, 135.0, 10.0};
    }
}

// ============================================================================
// Channelpedia Reference Data (from tissue/channelpedia.py)
// Reference kinetic parameters for channel models
// ============================================================================
struct ChannelRef {
    std::string name;
    std::string gene;
    std::string ion;
    int mpower, hpower;
    double vrev;        // reversal potential [mV]
    std::string reference;
};

inline std::vector<ChannelRef> get_channelpedia_database() {
    return {
        {"Nav1.2", "SCN2A", "Na", 3, 1, 50.0, "Hammil 1991"},
        {"Nav1.3", "SCN3A", "Na", 3, 1, 50.0, "Cummins 2001"},
        {"Nav1.6", "SCN8A", "Na", 1, 0, 50.0, "Smith 1998"},
        {"NavRat1", "SCN1A", "Na", 3, 1, 50.0, "Huguenard 1988"},
        {"NavRat2", "SCN2A", "Na", 3, 1, 50.0, "McCormick 1992"},
        {"Kv1.1", "KCNA1", "K", 1, 2, -65.0, "Bhatt 2002"},
        {"Kv1.2", "KCNA2", "K", 1, 1, -65.0, "Bhatt 2002"},
        {"Kv1.3", "KCNA3", "K", 1, 1, -65.0, "Bhatt 2002"},
        {"Kv1.4", "KCNA4", "K", 1, 1, -65.0, "Bhatt 2002"},
        {"Kv1.5", "KCNA5", "K", 1, 1, -65.0, "Bhatt 2002"},
        {"Kv1.6", "KCNA6", "K", 1, 1, -65.0, "Bhatt 2002"},
        {"Kv2.1", "KCNB1", "K", 1, 1, -65.0, "Bhatt 2002"},
        {"Kv2.2", "KCNB2", "K", 1, 1, -65.0, "Bhatt 2002"},
        {"Kv3.1", "KCNC1", "K", 1, 0, -65.0, "Bhatt 2002"},
        {"Kv3.2", "KCNC2", "K", 2, 0, -65.0, "Bhatt 2002"},
        {"Kv3.3", "KCNC3", "K", 2, 1, 82.0, "Rooma 2008"},
        {"Kv3.4", "KCNC4", "K", 1, 1, -65.0, "Bhatt 2002"},
        {"Kir2.1", "KCNJ2", "K", 1, 2, -70.6, "Bhatt 2002"},
        {"Cav1.2", "CACNA1C", "Ca", 2, 1, 131.0, "Carlin 2000"},
        {"Cav1.3", "CACNA1D", "Ca", 2, 1, 113.0, "Avery 1996"},
        {"Cav2.1", "CACNA1A", "Ca", 1, 0, 135.0, "Miyasho 2001"},
        {"Cav2.2", "CACNA1B", "Ca", 2, 1, 135.0, "Huang 1998"},
        {"Cav2.3", "CACNA1E", "Ca", 1, 1, 30.0, "Miyasho 2001"},
        {"Cav3.1", "CACNA1G", "Ca", 1, 1, 30.0, "Traboulsie 2007"},
        {"Cav3.3", "CACNA1I", "Ca", 1, 1, 30.0, "Traboulsie 2007"},
        {"HCN1", "HCN1", "Na/K/Ca", 1, 0, -45.0, "Altomare 2003"},
        {"HCN2", "HCN2", "Na/K/Ca", 1, 0, -45.0, "Altomare 2003"},
        {"HCN4", "HCN4", "Na/K/Ca", 1, 0, -45.0, "Altomare 2003"},
    };
}

} // namespace betse
