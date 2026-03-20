// BETSE C++ Port - Complete Enumerations
// All enumerations from betse/science/enum/, betse/science/config/,
// betse/science/pipe/export/pipeexpenum.py, and related modules.
// Ported from: enumconf.py, enumphase.py, enumion.py, pipeexpenum.py
#pragma once

#include <string>
#include <stdexcept>

namespace betse {

// ============================================================================
// Cell Lattice Type (from enumconf.py CellLatticeType)
// ============================================================================
enum class CellLatticeType {
    HEX,        // Hexagonal base cell lattice
    SQUARE      // Rectilinear (square) base cell lattice
};

inline const char* cell_lattice_name(CellLatticeType t) {
    switch (t) {
        case CellLatticeType::HEX: return "hex";
        case CellLatticeType::SQUARE: return "square";
    }
    return "unknown";
}

// ============================================================================
// Cells Picker Type (from enumconf.py CellsPickerType)
// How cells are selected for tissue profiles
// ============================================================================
enum class CellsPickerType {
    ALL,        // All cells unconditionally
    COLOR,      // SVG color-based selection
    IMAGE,      // Raster image-based selection (black pixels)
    INDICES,    // Explicit cell index list
    PERCENT     // Random percentage of cells
};

inline const char* cells_picker_name(CellsPickerType t) {
    switch (t) {
        case CellsPickerType::ALL: return "all";
        case CellsPickerType::COLOR: return "color";
        case CellsPickerType::IMAGE: return "image";
        case CellsPickerType::INDICES: return "indices";
        case CellsPickerType::PERCENT: return "percent";
    }
    return "unknown";
}

// ============================================================================
// GRN Unpickle Phase Type (from enumconf.py GrnUnpicklePhaseType)
// ============================================================================
enum class GrnUnpicklePhaseType {
    SEED,   // Run GRN on seed-phase results
    INIT,   // Run GRN on init-phase results
    SIM     // Run GRN on sim-phase results
};

// ============================================================================
// Ion Profile Type (from enumconf.py IonProfileType)
// Predefined sets of ions and their concentrations
// ============================================================================
enum class IonProfileType {
    BASIC,      // Na+, K+, M-, P-
    BASIC_CA,   // + Ca2+
    MAMMAL,     // + Ca2+, Cl-, H+ (amniotic concentrations)
    AMPHIBIAN,  // + Ca2+, Cl-, H+ (aquatic concentrations)
    CUSTOM      // User-defined
};

inline const char* ion_profile_name(IonProfileType t) {
    switch (t) {
        case IonProfileType::BASIC: return "basic";
        case IonProfileType::BASIC_CA: return "basic_Ca";
        case IonProfileType::MAMMAL: return "mammal";
        case IonProfileType::AMPHIBIAN: return "amphibian";
        case IonProfileType::CUSTOM: return "custom";
    }
    return "unknown";
}

// ============================================================================
// Simulation Export Type (from pipeexpenum.py SimExportType)
// ============================================================================
enum class SimExportType {
    ANIM,   // Animated video export
    CSV,    // Comma-separated value file
    PLOT    // Static plot image
};

inline const char* export_type_name(SimExportType t) {
    switch (t) {
        case SimExportType::ANIM: return "anim";
        case SimExportType::CSV: return "csv";
        case SimExportType::PLOT: return "plot";
    }
    return "unknown";
}

// ============================================================================
// Boundary Side (for voltage events applied at ECM boundaries)
// From tissue/event/tisevevolt.py
// ============================================================================
enum class BoundarySide {
    TOP,
    BOTTOM,
    LEFT,
    RIGHT
};

inline const char* boundary_side_name(BoundarySide s) {
    switch (s) {
        case BoundarySide::TOP: return "T";
        case BoundarySide::BOTTOM: return "B";
        case BoundarySide::LEFT: return "L";
        case BoundarySide::RIGHT: return "R";
    }
    return "?";
}

inline BoundarySide boundary_side_from_string(const std::string& s) {
    if (s == "T" || s == "top" || s == "Top") return BoundarySide::TOP;
    if (s == "B" || s == "bottom" || s == "Bottom") return BoundarySide::BOTTOM;
    if (s == "L" || s == "left" || s == "Left") return BoundarySide::LEFT;
    if (s == "R" || s == "right" || s == "Right") return BoundarySide::RIGHT;
    throw std::invalid_argument("Unknown boundary side: " + s);
}

// ============================================================================
// Colormap Type (from parameters.py colormap settings)
// ============================================================================
enum class ColormapType {
    DIVERGING,      // Zero-centered data (e.g., Vmem)
    SEQUENTIAL,     // Zero-based monotonic data (e.g., currents)
    GAP_JUNCTION,   // GJ-specific
    GRN             // Gene regulatory network
};

// ============================================================================
// Deformation solver type (from parameters.py)
// ============================================================================
enum class DeformSolverType {
    STEADY_STATE,   // Equilibrium deformation (getDeformation)
    TIME_DEPENDENT  // Dynamic deformation (timeDeform)
};

// ============================================================================
// Gap junction connectivity type
// From tissue/tisprofile.py
// ============================================================================
enum class GJConnectivityType {
    NORMAL,     // GJs connect to all neighboring cells
    INSULAR     // GJs only connect within same tissue profile
};

// ============================================================================
// Electrodiffusion model type (from parameters.py)
// ============================================================================
enum class ElectroDiffusionType {
    FULL_NERNST_PLANCK,     // Full NP with electric field
    GOLDMAN_HODGKIN_KATZ,   // GHK simplified
    EQUIVALENT_CIRCUIT      // Fast solver approximation
};

// ============================================================================
// Osmotic water model type
// ============================================================================
enum class OsmoticModelType {
    NONE,           // No osmotic effects
    BASIC,          // Simple transmembrane osmotic flux
    FULL            // Full with aquaporin density, pressure feedback
};

// ============================================================================
// Visual layer type (from visual/layer/)
// ============================================================================
enum class VisualLayerType {
    CELLS_SCALAR,       // Scalar field on cells (e.g., Vmem)
    CELLS_VECTOR,       // Vector field on cells (e.g., current)
    ECM_SCALAR,         // Scalar field on ECM grid
    ECM_VECTOR,         // Vector field on ECM grid
    STREAM_FIELD,       // Streamline plot of vector field
    QUIVER_FIELD,       // Quiver (arrow) plot of vector field
    TEXT_OVERLAY,       // Text annotations
    CELL_BOUNDARY,      // Cell boundary visualization
    GAP_JUNCTION,       // GJ connections visualization
    DEFORMATION,        // Displacement arrows
    MEMBRANE            // Membrane-level scalar field
};

// ============================================================================
// Simulation data channel (what quantity to visualize/export)
// From pipe/export/pipeexps.py and visual modules
// ============================================================================
enum class SimDataChannel {
    // Voltage
    VMEM, VMEM_SINGLE_CELL, V_ENV,
    // Concentrations
    CC_NA, CC_K, CC_CL, CC_CA, CC_H, CC_M, CC_P,
    CC_ENV_NA, CC_ENV_K, CC_ENV_CL, CC_ENV_CA,
    // Charge
    RHO_CELLS, RHO_ENV,
    // Current
    J_CELL, J_MEM, J_ENV, J_GJ,
    // Electric field
    E_CELL, E_ENV,
    // Pressure
    P_CELLS, P_OSMO,
    // Deformation
    DEFORM,
    // Flow
    FLOW_CELLS, FLOW_ENV,
    // Pumps
    PUMP_NA_K, PUMP_CA,
    // Channel
    CHANNEL_P_OPEN,
    // GRN molecules
    MOLECULE,
    // Organelles
    ER_CA, ER_V, MITO_V,
    // Magnetic field
    B_FIELD,
    // Microtubules
    MTUBE_ANGLE,
    // pH
    PH_CELL, PH_ENV
};

inline const char* sim_data_channel_name(SimDataChannel ch) {
    switch (ch) {
        case SimDataChannel::VMEM: return "Vmem";
        case SimDataChannel::VMEM_SINGLE_CELL: return "Vmem_single";
        case SimDataChannel::V_ENV: return "V_env";
        case SimDataChannel::CC_NA: return "cc_Na";
        case SimDataChannel::CC_K: return "cc_K";
        case SimDataChannel::CC_CL: return "cc_Cl";
        case SimDataChannel::CC_CA: return "cc_Ca";
        case SimDataChannel::CC_H: return "cc_H";
        case SimDataChannel::CC_M: return "cc_M";
        case SimDataChannel::CC_P: return "cc_P";
        case SimDataChannel::CC_ENV_NA: return "cc_env_Na";
        case SimDataChannel::CC_ENV_K: return "cc_env_K";
        case SimDataChannel::CC_ENV_CL: return "cc_env_Cl";
        case SimDataChannel::CC_ENV_CA: return "cc_env_Ca";
        case SimDataChannel::RHO_CELLS: return "rho_cells";
        case SimDataChannel::RHO_ENV: return "rho_env";
        case SimDataChannel::J_CELL: return "J_cell";
        case SimDataChannel::J_MEM: return "J_mem";
        case SimDataChannel::J_ENV: return "J_env";
        case SimDataChannel::J_GJ: return "J_gj";
        case SimDataChannel::E_CELL: return "E_cell";
        case SimDataChannel::E_ENV: return "E_env";
        case SimDataChannel::P_CELLS: return "P_cells";
        case SimDataChannel::P_OSMO: return "P_osmo";
        case SimDataChannel::DEFORM: return "deform";
        case SimDataChannel::FLOW_CELLS: return "flow_cells";
        case SimDataChannel::FLOW_ENV: return "flow_env";
        case SimDataChannel::PUMP_NA_K: return "pump_NaK";
        case SimDataChannel::PUMP_CA: return "pump_Ca";
        case SimDataChannel::CHANNEL_P_OPEN: return "channel_P_open";
        case SimDataChannel::MOLECULE: return "molecule";
        case SimDataChannel::ER_CA: return "ER_Ca";
        case SimDataChannel::ER_V: return "ER_V";
        case SimDataChannel::MITO_V: return "mito_V";
        case SimDataChannel::B_FIELD: return "B_field";
        case SimDataChannel::MTUBE_ANGLE: return "mtube_angle";
        case SimDataChannel::PH_CELL: return "pH_cell";
        case SimDataChannel::PH_ENV: return "pH_env";
    }
    return "unknown";
}

} // namespace betse
