// BETSE C++ Port - Full Configuration / Parameters System
// Complete reimplementation of betse/science/parameters.py Parameters class,
// betse/science/config/model/conftis.py, betse/science/config/grn/confgrn.py,
// and all export configuration from betse/science/config/export/
#pragma once

#include "betse_types.h"
#include "betse_enums.h"
#include "betse_tissue.h"
#include <string>
#include <vector>
#include <map>
#include <cmath>

namespace betse {

// ============================================================================
// Export Configuration (from config/export/)
// ============================================================================

// Single CSV export item (from confexpcsv.py)
struct CSVExportItem {
    std::string name;
    bool enabled = false;
    SimDataChannel channel = SimDataChannel::VMEM;
};

// Single plot export item (from confexpvisplot.py)
struct PlotExportItem {
    std::string name;
    bool enabled = false;
    SimDataChannel channel = SimDataChannel::VMEM;
    std::string colormap = "RdBu_r";
    bool show_colorbar = true;
    double vmin = -70.0;  // data range minimum (for manual scaling)
    double vmax = 10.0;
    bool auto_scale = true;
};

// Single animation export item (from confexpvisanim.py)
struct AnimExportItem {
    std::string name;
    bool enabled = false;
    SimDataChannel channel = SimDataChannel::VMEM;
    std::string colormap = "RdBu_r";
    bool show_colorbar = true;
    double vmin = -70.0;
    double vmax = 10.0;
    bool auto_scale = true;
    int fps = 15;
    std::string codec = "ffv1";
    std::string format = "mkv";
};

// All exports configuration (from config/export/)
struct ExportConfig {
    // CSV exports
    std::vector<CSVExportItem> csv_exports;
    bool csv_enabled = true;

    // Plot exports
    std::vector<PlotExportItem> plot_exports;
    bool plots_enabled = true;

    // Animation exports
    std::vector<AnimExportItem> anim_exports;
    bool anims_enabled = false;

    // Visual settings (from confexpvisual.py)
    bool show_cell_indices = false;
    int single_cell_index = 0;

    // Colormap names
    std::string colormap_diverging = "RdBu_r";
    std::string colormap_sequential = "YlOrRd";
    std::string colormap_gj = "plasma";
    std::string colormap_grn = "viridis";

    // Image settings
    int image_dpi = 300;
    std::string image_format = "png";

    // Output directory paths
    std::string init_export_dirname = "RESULTS/init";
    std::string sim_export_dirname = "RESULTS/sim";
    std::string grn_export_dirname = "RESULTS/grn";
};

// ============================================================================
// GRN Configuration (from config/grn/confgrn.py)
// ============================================================================
struct GRNMoleculeConfig {
    std::string name;
    double Dm = 0.0;               // membrane diffusion [m/s]
    double Do = 1e-9;              // free diffusion [m^2/s]
    double Dgj = 1e-14;            // gap junction diffusion [m^2/s]
    double Ftj = 1.0;              // tight junction factor
    double c_cell = 0.0;           // initial cell concentration [mM]
    double c_env = 0.0;            // initial env concentration [mM]
    double c_mit = 0.0;            // initial mito concentration [mM]
    int z = 0;                     // charge
    double decay_rate = 0.0;       // first-order decay [1/s]
    double growth_rate = 0.0;      // zero-order production [mM/s]
    double growth_profiles_factor = 1.0;
    bool use_time_dilation = false;
    double time_dilation_factor = 1.0;
    bool mit_enabled = false;
    bool transmem = false;
    double mu_mem = 0.0;           // electrophoretic mobility in membrane
    std::string spatial_modulator; // name of modulator function
    double c_bound = 0.0;         // boundary concentration
    // Channel-specific
    bool is_channel_gating = false;
    std::string channel_type_name;
    double max_Dm_channel = 1e-15;
};

struct GRNReactionConfig {
    std::string name;
    std::vector<std::string> reactants;
    std::vector<std::string> products;
    std::vector<double> reactant_stoich;
    std::vector<double> product_stoich;
    double rate_const = 0.0;
    double Km = 1.0;
    int reaction_zone = 0;   // 0=cell, 1=env, 2=mem, 3=mito
    double max_rate = 1e10;
    bool reversible = false;
    double Keq = 1.0;
};

struct GRNTransporterConfig {
    std::string name;
    std::string substrate;
    double alpha_max = 1e-8;
    double Km = 1.0;
    double Km_ATP = 0.5;
    bool pump_into_cell = false;
    bool uses_ATP = true;
    double Keq = 1.0;
    int n = 1;      // Hill coefficient
    int z = 0;
};

struct GRNModulatorConfig {
    std::string name;
    std::string substance;
    std::string target_ion;
    double Km = 1.0;
    int n = 1;
    double max_effect = 1.0;
    bool activates = true;
};

struct GRNConfig {
    bool enabled = false;
    GrnUnpicklePhaseType unpickle_phase = GrnUnpicklePhaseType::INIT;
    std::vector<GRNMoleculeConfig> molecules;
    std::vector<GRNReactionConfig> reactions;
    std::vector<GRNTransporterConfig> transporters;
    std::vector<GRNModulatorConfig> modulators;
};

// ============================================================================
// Tissue Profile Configuration (from config/model/conftis.py)
// ============================================================================
struct TissueProfileConfig {
    std::string name = "default";
    CellsPickerType picker_type = CellsPickerType::ALL;
    Vec2 center = {0, 0};
    double radius = 0;
    double width = 0, height = 0;
    std::vector<int> indices;
    double percent = 100.0;
    std::string image_path;
    std::string color;

    // Per-ion membrane diffusion [m/s]
    double Dm_Na  = 1.0e-18;
    double Dm_K   = 1.0e-16;
    double Dm_Cl  = 1.0e-18;
    double Dm_Ca  = 1.0e-20;
    double Dm_H   = 1.0e-18;
    double Dm_M   = 1.0e-18;
    double Dm_P   = 0.0;

    // GJ
    bool is_gj_insular = false;
    double gj_surface = -1.0;  // -1 = use global

    int z_order = 1;
};

struct CutEventConfig {
    double time = 0;
    Vec2 center = {0, 0};
    double radius = 0;
    CellsPickerType picker_type = CellsPickerType::ALL;
    std::string image_path;
};

// ============================================================================
// Full Simulation Configuration (from parameters.py Parameters)
// All parameters from the YAML configuration file
// ============================================================================
struct FullSimConfig {
    // --- Solver ---
    SolverType solver_type = SolverType::FULL;

    // --- Time ---
    double dt               = 1.0e-4;      // time step [s]
    double init_time        = 10.0;         // init phase duration [s]
    double sim_time         = 1.0;          // sim phase duration [s]
    int    sample_rate      = 100;          // save every N steps
    double init_time_step   = 1.0e-3;      // init time step [s]
    double sim_time_step    = 1.0e-4;      // sim time step [s]

    // --- Temperature ---
    double T                = 310.0;        // Kelvin

    // --- Cell Geometry ---
    CellLatticeType lattice_type = CellLatticeType::HEX;
    double cell_radius      = 5.0e-6;      // [m]
    double cell_height      = 10.0e-6;     // tissue thickness [m]
    int    num_cells        = 100;
    double tissue_radius    = 100.0e-6;    // [m]
    double lattice_disorder = 0.4;         // amount of lattice randomization

    // --- Membrane ---
    double cm               = 0.01;        // membrane capacitance [F/m^2]
    double tm               = 1.0e-8;      // membrane thickness [m]

    // --- Gap Junctions ---
    double gj_surface       = 1.0e-12;     // [m^2]
    double gj_len           = 1.0e-6;      // [m]
    double gj_vthresh       = 0.0;         // gating threshold [mV]
    double gj_min           = 0.1;         // minimum conductance fraction
    double gj_vgrad         = 10.0;        // voltage gradient for gating
    bool   gj_voltage_gating = true;       // enable voltage-dependent GJ gating
    double gj_default_block = 1.0;         // default block factor

    // --- Ion Profile ---
    IonProfileType ion_profile = IonProfileType::MAMMAL;

    // --- Ion Concentrations (6 ions) ---
    struct IonParams {
        double cell;      // intracellular [mM]
        double env;       // extracellular [mM]
        double D_free;    // free diffusion coefficient [m^2/s]
        double Dm;        // membrane permeability [m/s]
    };
    IonParams ions[6] = {
        {  12.0, 145.0, 1.33e-9, 1.0e-18 },   // Na
        { 140.0,   5.0, 1.96e-9, 1.0e-16 },   // K
        {   4.0, 110.0, 2.03e-9, 1.0e-18 },   // Cl
        { 0.0001, 2.0,  0.79e-9, 1.0e-20 },   // Ca
        { 6.3e-5, 4.0e-5, 9.31e-9, 1.0e-18 }, // H
        { 135.0,  10.0, 0.0,     0.0     },    // P
    };

    // --- Na-K-ATPase Pump ---
    bool   pump_enabled     = true;
    double alpha_NaK        = 1.0e-5;      // [mol/(m^2*s)]
    double deltaGATP        = -54000.0;     // [J/mol]
    double cATP             = 1.5;          // [mM]
    double cADP             = 0.01;         // [mM]
    double cPi              = 0.5;          // [mM]
    double KmNK_Na          = 12.0;         // [mM]
    double KmNK_K           = 1.0;          // [mM]
    double KmNK_ATP         = 0.5;          // [mM]

    // --- Ca-ATPase Pump ---
    bool   ca_dyn_enabled   = false;
    double alpha_Ca         = 1.0e-6;
    double KmCa_Ca          = 0.0005;
    double KmCa_ATP         = 0.5;

    // --- ECM ---
    bool   ecm_enabled      = false;
    double ecm_delta        = 2.0e-6;
    double sharpness        = 1.0;

    // --- Noise ---
    bool   noise_enabled    = false;
    double noise_level      = 0.001;

    // --- Deformation ---
    bool   deform_enabled   = false;
    DeformSolverType deform_type = DeformSolverType::STEADY_STATE;
    double lame_mu          = 1.0e3;
    double galvanotropism   = 0.0;
    bool   deform_osmo      = false;
    bool   fixed_cluster_bound = true;
    double mu_tissue        = 0.01;         // tissue viscosity

    // --- Electroosmotic Flow ---
    bool   flow_enabled     = false;
    double mu_water         = 1.0e-3;       // [Pa*s]
    double eo               = 8.854e-12;
    double er               = 80.0;
    double rho_water        = 1000.0;

    // --- Osmotic Pressure ---
    bool   osmo_enabled     = false;
    double aquaporins       = 1.0e-2;

    // --- ER ---
    bool   er_enabled       = false;
    double er_vol_frac      = 0.1;
    double er_sa_frac       = 1.0;
    double serca_max        = 1.0e-5;
    double max_er           = 1.0e-14;
    double act_Km_Ca        = 0.0005;
    double act_n_Ca         = 2.0;
    double inh_Km_Ca        = 0.001;
    double inh_n_Ca         = 4.0;
    double act_Km_IP3       = 0.0005;
    double act_n_IP3        = 2.0;
    double er_init_Ca       = 0.2;

    // --- Mitochondria ---
    bool   mito_enabled     = false;
    double mito_vol_frac    = 0.5;
    double mito_sa_frac     = 0.5;

    // --- Microtubules ---
    bool   mtube_enabled    = false;
    double mt_radius        = 12.0e-9;
    double tubulin_dipole   = 1740.0;
    double tubulin_polar    = 20.0;
    double length_charge    = 40.0;
    double cytoplasm_viscosity = 0.02;
    double dilate_mtube_dt  = 0.0;
    bool   tethered_tubule  = false;

    // --- Cell Polarizability ---
    double cell_polarizability = 0.0;

    // --- Channels ---
    struct ChannelEntry {
        ChannelType type;
        double max_Dm = 1.0e-15;
        int target_profile_id = -1;
    };
    std::vector<ChannelEntry> channels;

    // --- Tissue Profiles ---
    std::vector<TissueProfileConfig> tissue_profiles;

    // --- Cut Events ---
    std::vector<CutEventConfig> cut_events;
    double wound_close_factor = 0.01;

    // --- External Voltage Events ---
    struct VoltageEventConfig {
        double start_time = 0;
        double stop_time = 0;
        double rate = 1.0;
        double peak_voltage = 0;
        std::string positive_boundary = "T";
        std::string negative_boundary = "B";
    };
    std::vector<VoltageEventConfig> voltage_events;

    // --- Targeted Interventions (per-profile modifications) ---
    struct InterventionConfig {
        double start_time = 0;
        double stop_time = 0;
        double rate = 1.0;
        int target_profile_id = -1;
        int target_ion = -1;
        double Dm_factor = 1.0;
    };
    std::vector<InterventionConfig> interventions;

    // --- GRN ---
    GRNConfig grn;

    // --- Export ---
    ExportConfig exports;

    // --- Convenience: apply ion profile preset ---
    void apply_ion_profile() {
        auto preset = get_ion_profile(ion_profile);
        ions[0].cell = preset.Na_cell; ions[0].env = preset.Na_env;
        ions[1].cell = preset.K_cell;  ions[1].env = preset.K_env;
        ions[2].cell = preset.Cl_cell; ions[2].env = preset.Cl_env;
        ions[3].cell = preset.Ca_cell; ions[3].env = preset.Ca_env;
        ions[4].cell = preset.H_cell;  ions[4].env = preset.H_env;
        ions[5].cell = preset.P_cell;  ions[5].env = preset.P_env;
    }
};

} // namespace betse
