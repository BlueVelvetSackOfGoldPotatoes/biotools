// cellblender.h - C++ port of CellBlender core functionality
// Ported from the CellBlender Blender addon for MCell cellular simulations.
// This port covers: MDL generation, data model serialization, molecule/reaction
// management, geometry processing, simulation control, parameter sweeps, and
// results parsing/analysis. Blender-specific UI code is not ported.
//
// Original CellBlender is licensed under GPL v2+.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace cellblender {

// ============================================================================
// Forward declarations
// ============================================================================
struct DataModel;
struct MoleculeSpecies;
struct Reaction;
struct ReleaseSite;
struct SurfaceClass;
struct SurfaceClassProperty;
struct ModSurfaceRegion;
struct ReleasePattern;
struct ReactionOutput;
struct Partition;
struct GeometryObject;
struct SurfaceRegion;

// ============================================================================
// Vec3 - basic 3D vector
// ============================================================================
struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(double s) const { return {x*s, y*s, z*s}; }
    double dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x};
    }
    double length() const { return std::sqrt(x*x + y*y + z*z); }
    Vec3 normalized() const {
        double l = length();
        return l > 0 ? Vec3{x/l, y/l, z/l} : Vec3{0,0,0};
    }
};

// ============================================================================
// Triangle face (indices into a vertex list)
// ============================================================================
struct Face {
    int v0 = 0, v1 = 0, v2 = 0;
    Face() = default;
    Face(int a, int b, int c) : v0(a), v1(b), v2(c) {}
};

// ============================================================================
// ParameterSpace - expression-based parameter system
// Mirrors CellBlender's ParameterSpace class from ParameterSpace.py
// ============================================================================
class ParameterSpace {
public:
    ParameterSpace() = default;

    int define(const std::string& name, const std::string& expr) {
        auto it = name_to_id_.find(name);
        if (it != name_to_id_.end()) {
            set_expr(it->second, expr);
            return it->second;
        }
        int id = next_id_++;
        name_to_id_[name] = id;
        id_to_name_[id] = name;
        id_to_expr_[id] = expr;
        id_to_value_[id] = evaluate_expr(expr);
        return id;
    }

    void set_expr(int id, const std::string& expr) {
        id_to_expr_[id] = expr;
        id_to_value_[id] = evaluate_expr(expr);
    }

    std::string get_expr(int id) const {
        auto it = id_to_expr_.find(id);
        return (it != id_to_expr_.end()) ? it->second : "";
    }

    double get_value(int id) const {
        auto it = id_to_value_.find(id);
        return (it != id_to_value_.end()) ? it->second : 0.0;
    }

    double get_value(const std::string& name) const {
        auto it = name_to_id_.find(name);
        if (it == name_to_id_.end()) return 0.0;
        return get_value(it->second);
    }

    std::string get_name(int id) const {
        auto it = id_to_name_.find(id);
        return (it != id_to_name_.end()) ? it->second : "";
    }

    int get_id(const std::string& name) const {
        auto it = name_to_id_.find(name);
        return (it != name_to_id_.end()) ? it->second : -1;
    }

    bool has(const std::string& name) const {
        return name_to_id_.count(name) > 0;
    }

    bool remove(const std::string& name) {
        auto it = name_to_id_.find(name);
        if (it == name_to_id_.end()) return false;
        int id = it->second;
        name_to_id_.erase(it);
        id_to_name_.erase(id);
        id_to_expr_.erase(id);
        id_to_value_.erase(id);
        return true;
    }

    std::vector<std::string> get_all_names() const {
        std::vector<std::string> names;
        names.reserve(name_to_id_.size());
        for (auto& [n, _] : name_to_id_) names.push_back(n);
        return names;
    }

    void eval_all() {
        for (auto& [id, expr] : id_to_expr_) {
            id_to_value_[id] = evaluate_expr(expr);
        }
    }

    std::string get_as_string_or_value(int id, bool as_expression) const {
        if (as_expression) return get_expr(id);
        std::ostringstream oss;
        oss << std::setprecision(15) << get_value(id);
        return oss.str();
    }

    std::string get_as_string_or_value(const std::string& name, bool as_expression) const {
        int id = get_id(name);
        if (id < 0) return name;
        return get_as_string_or_value(id, as_expression);
    }

    size_t num_parameters() const { return name_to_id_.size(); }

    void clear() {
        name_to_id_.clear();
        id_to_name_.clear();
        id_to_expr_.clear();
        id_to_value_.clear();
        next_id_ = 1;
    }

private:
    double evaluate_expr(const std::string& expr) const {
        if (expr.empty()) return 0.0;
        try {
            size_t pos = 0;
            double val = std::stod(expr, &pos);
            if (pos == expr.size()) return val;
        } catch (...) {}
        auto it = name_to_id_.find(expr);
        if (it != name_to_id_.end()) {
            auto vit = id_to_value_.find(it->second);
            if (vit != id_to_value_.end()) return vit->second;
        }
        return 0.0;
    }

    std::map<std::string, int> name_to_id_;
    std::map<int, std::string> id_to_name_;
    std::map<int, std::string> id_to_expr_;
    std::map<int, double>      id_to_value_;
    int next_id_ = 1;
};

// ============================================================================
// Surface Region definition
// ============================================================================
struct SurfaceRegion {
    std::string name;
    std::vector<int> face_indices;
};

// ============================================================================
// Geometry Object - triangulated mesh for MCell
// ============================================================================
struct GeometryObject {
    std::string name;
    std::vector<Vec3> vertices;
    std::vector<Face> faces;
    std::vector<SurfaceRegion> regions;
    Vec3 location{0, 0, 0};
    std::string parent_object;
    std::string membrane_name;
    bool dynamic = false;

    int num_vertices() const { return static_cast<int>(vertices.size()); }
    int num_faces() const { return static_cast<int>(faces.size()); }
    int num_edges() const {
        std::set<std::pair<int,int>> edge_set;
        for (auto& f : faces) {
            auto add_edge = [&](int a, int b) {
                edge_set.insert({std::min(a,b), std::max(a,b)});
            };
            add_edge(f.v0, f.v1);
            add_edge(f.v1, f.v2);
            add_edge(f.v2, f.v0);
        }
        return static_cast<int>(edge_set.size());
    }

    double surface_area() const {
        double area = 0;
        for (auto& f : faces) {
            Vec3 e1 = vertices[f.v1] - vertices[f.v0];
            Vec3 e2 = vertices[f.v2] - vertices[f.v0];
            area += e1.cross(e2).length() * 0.5;
        }
        return area;
    }

    double signed_volume() const {
        double vol = 0;
        for (auto& f : faces) {
            const Vec3& a = vertices[f.v0];
            const Vec3& b = vertices[f.v1];
            const Vec3& c = vertices[f.v2];
            vol += a.dot(b.cross(c)) / 6.0;
        }
        return vol;
    }

    int euler_characteristic() const {
        return num_vertices() - num_edges() + num_faces();
    }

    int genus() const {
        int chi = euler_characteristic();
        return 1 - chi / 2;
    }

    bool is_watertight() const {
        std::map<std::pair<int,int>, int> edge_count;
        for (auto& f : faces) {
            auto add = [&](int a, int b) {
                edge_count[{std::min(a,b), std::max(a,b)}]++;
            };
            add(f.v0, f.v1);
            add(f.v1, f.v2);
            add(f.v2, f.v0);
        }
        for (auto& [e, cnt] : edge_count) {
            if (cnt != 2) return false;
        }
        return true;
    }
};

// ============================================================================
// Molecule Component (for BioNetGen structured molecules)
// ============================================================================
struct MoleculeComponent {
    std::string name;
    std::vector<std::string> states;
    bool is_key = false;
    Vec3 location{0, 0, 0};
    Vec3 rotation_axis{0, 0, 0};
    double rotation_angle = 0;
    int rot_index = -1;
};

// ============================================================================
// Molecule Species Definition
// ============================================================================
struct MoleculeSpecies {
    std::string name;
    std::string description;
    enum Type { VOLUME_3D, SURFACE_2D } mol_type = VOLUME_3D;
    std::string diffusion_constant = "0";
    bool target_only = false;
    std::string custom_time_step;
    std::string custom_space_step;
    std::string maximum_step_length;
    bool export_viz = false;
    std::string bngl_label;
    std::string spatial_structure = "None";
    std::vector<MoleculeComponent> components;
};

// ============================================================================
// Reaction Definition
// ============================================================================
struct Reaction {
    std::string name;
    std::string description;
    std::string rxn_name;
    std::string reactants;
    std::string products;
    enum Type { IRREVERSIBLE, REVERSIBLE } rxn_type = IRREVERSIBLE;
    std::string fwd_rate = "0";
    std::string bkwd_rate;
    bool variable_rate_switch = false;
    std::string variable_rate;
    std::string variable_rate_text;
    bool variable_rate_valid = false;

    void update_name() {
        std::string arrow = (rxn_type == REVERSIBLE) ? "<->" : "->";
        name = reactants + " " + arrow + " " + products;
    }
};

// ============================================================================
// Surface Class Property
// ============================================================================
struct SurfaceClassProperty {
    enum AffectedMols { ALL_MOLECULES, ALL_VOLUME_MOLECULES, ALL_SURFACE_MOLECULES, SINGLE };
    AffectedMols affected_mols = ALL_MOLECULES;
    std::string molecule;
    enum Orient { TOP_FRONT, BOTTOM_BACK, IGNORE };
    Orient orient = IGNORE;
    enum ClassType { ABSORPTIVE, TRANSPARENT, REFLECTIVE, CLAMP_CONCENTRATION };
    ClassType class_type = TRANSPARENT;
    std::string clamp_value = "0";

    std::string orient_str() const {
        switch (orient) {
            case TOP_FRONT: return "'";
            case BOTTOM_BACK: return ",";
            case IGNORE: return ";";
        }
        return ";";
    }

    std::string class_type_str() const {
        switch (class_type) {
            case ABSORPTIVE: return "ABSORPTIVE";
            case TRANSPARENT: return "TRANSPARENT";
            case REFLECTIVE: return "REFLECTIVE";
            case CLAMP_CONCENTRATION: return "CLAMP_CONCENTRATION";
        }
        return "TRANSPARENT";
    }

    std::string affected_mols_str() const {
        switch (affected_mols) {
            case ALL_MOLECULES: return "ALL_MOLECULES";
            case ALL_VOLUME_MOLECULES: return "ALL_VOLUME_MOLECULES";
            case ALL_SURFACE_MOLECULES: return "ALL_SURFACE_MOLECULES";
            case SINGLE: return "SINGLE";
        }
        return "ALL_MOLECULES";
    }
};

// ============================================================================
// Surface Class
// ============================================================================
struct SurfaceClass {
    std::string name = "Surface_Class";
    std::string description;
    std::vector<SurfaceClassProperty> properties;
};

// ============================================================================
// Modify Surface Region
// ============================================================================
struct ModSurfaceRegion {
    std::string name;
    std::string object_name;
    std::string region_name;
    std::string surf_class_name;
};

// ============================================================================
// Release Pattern
// ============================================================================
struct ReleasePattern {
    std::string name;
    std::string description;
    std::string delay = "0";
    std::string release_interval;
    std::string train_duration;
    std::string train_interval = "0";
    std::string number_of_trains = "1";
};

// ============================================================================
// Release Site
// ============================================================================
struct ReleaseSite {
    std::string name = "Release_Site";
    std::string description;
    std::string molecule;
    enum Shape { CUBIC, SPHERICAL, SPHERICAL_SHELL, LIST, OBJECT } shape = SPHERICAL;
    enum Orient { TOP_FRONT, TOP_BACK, MIXED } orient = TOP_FRONT;
    std::string object_expr;
    std::string location_x = "0", location_y = "0", location_z = "0";
    std::string diameter = "0";
    std::string probability = "1";
    enum QuantityType { NUMBER_TO_RELEASE, GAUSSIAN_RELEASE_NUMBER, DENSITY };
    QuantityType quantity_type = NUMBER_TO_RELEASE;
    std::string quantity;
    std::string stddev = "0";
    std::string pattern;
    std::vector<Vec3> points_list;

    std::string shape_str() const {
        switch (shape) {
            case CUBIC: return "CUBIC";
            case SPHERICAL: return "SPHERICAL";
            case SPHERICAL_SHELL: return "SPHERICAL_SHELL";
            case LIST: return "LIST";
            case OBJECT: return "OBJECT";
        }
        return "SPHERICAL";
    }

    std::string orient_str() const {
        switch (orient) {
            case TOP_FRONT: return "'";
            case TOP_BACK: return ",";
            case MIXED: return ";";
        }
        return "'";
    }
};

// ============================================================================
// Reaction Data Output
// ============================================================================
struct ReactionOutput {
    std::string name;
    enum RxnOrMol { MOLECULE, REACTION, MDL_STRING, FILE_OUTPUT } rxn_or_mol = MOLECULE;
    std::string molecule_name;
    std::string reaction_name;
    std::string object_name;
    std::string region_name;
    enum CountLocation { WORLD, OBJECT_LOC, REGION_LOC } count_location = WORLD;
    std::string mdl_string;
    std::string mdl_file_prefix;
    std::string data_file_name;
    bool plotting_enabled = true;
};

// ============================================================================
// Partition System
// ============================================================================
struct Partition {
    bool include = false;
    double x_start = -1, x_end = 1, x_step = 0.05;
    double y_start = -1, y_end = 1, y_step = 0.05;
    double z_start = -1, z_end = 1, z_step = 0.05;
};

// ============================================================================
// Viz Output Settings
// ============================================================================
struct VizOutput {
    bool export_all = true;
    bool all_iterations = true;
    std::string start = "0";
    std::string end_val = "1";
    std::string step = "1";
};

// ============================================================================
// Simulation Initialization Parameters
// ============================================================================
struct InitializationParams {
    std::string iterations = "1000";
    std::string time_step = "1e-6";
    std::string vacancy_search_distance = "10";
    std::string time_step_max;
    std::string space_step;
    std::string interaction_radius;
    std::string radial_directions;
    std::string radial_subdivisions;
    std::string surface_grid_density = "10000";
    bool accurate_3d_reactions = true;
    bool center_molecules_grid = false;
    std::string microscopic_reversibility = "ON";
    bool export_all_ascii = false;

    std::string all_notifications = "INDIVIDUAL";
    std::string probability_report = "ON";
    double probability_report_threshold = 0;
    std::string diffusion_constant_report = "BRIEF";
    bool file_output_report = false;
    bool final_summary = true;
    bool iteration_report = true;
    bool partition_location_report = false;
    bool varying_probability_report = true;
    bool progress_report = true;
    bool release_event_report = true;
    bool molecule_collision_report = false;

    std::string all_warnings = "INDIVIDUAL";
    std::string degenerate_polygons = "WARNING";
    std::string negative_diffusion_constant = "WARNING";
    std::string missing_surface_orientation = "ERROR";
    std::string negative_reaction_rate = "WARNING";
    std::string useless_volume_orientation = "WARNING";
    std::string high_reaction_probability = "IGNORED";
    std::string lifetime_too_short = "WARNING";
    double lifetime_threshold = 50;
    std::string missed_reactions = "WARNING";
    double missed_reaction_threshold = 0.001;

    Partition partitions;
};

// ============================================================================
// Simulation Control
// ============================================================================
struct SimulationControl {
    int start_seed = 1;
    int end_seed = 1;
    std::string export_format = "mcell_mdl_modular";
    std::string mcell_binary = "mcell";
};

// ============================================================================
// DataModel - top-level CellBlender project representation
// ============================================================================
struct DataModel {
    ParameterSpace parameters;
    std::vector<MoleculeSpecies> molecules;
    std::vector<Reaction> reactions;
    std::vector<ReleaseSite> release_sites;
    std::vector<ReleasePattern> release_patterns;
    std::vector<SurfaceClass> surface_classes;
    std::vector<ModSurfaceRegion> mod_surf_regions;
    std::vector<GeometryObject> geometry_objects;
    std::vector<ReactionOutput> reaction_outputs;

    InitializationParams initialization;
    VizOutput viz_output;
    SimulationControl sim_control;
    std::string scene_name = "Scene";
};

// ============================================================================
// MDLWriter - generates MCell MDL files from a DataModel
// ============================================================================
class MDLWriter {
public:
    explicit MDLWriter(const DataModel& dm) : dm_(dm) {}

    void write_modular(const std::filesystem::path& dir) const {
        namespace fs = std::filesystem;
        fs::create_directories(dir);
        std::string base = dm_.scene_name;

        std::ofstream main_f(dir / (base + ".main.mdl"));
        if (!main_f) throw std::runtime_error("Cannot open main MDL file");

        {
            std::ofstream f(dir / (base + ".parameters.mdl"));
            if (write_parameters(f))
                main_f << "INCLUDE_FILE = \"" << base << ".parameters.mdl\"\n\n";
        }
        {
            std::ofstream f(dir / (base + ".initialization.mdl"));
            write_initialization(f);
            main_f << "INCLUDE_FILE = \"" << base << ".initialization.mdl\"\n\n";
        }
        {
            std::ofstream f(dir / (base + ".molecules.mdl"));
            if (write_molecules(f))
                main_f << "INCLUDE_FILE = \"" << base << ".molecules.mdl\"\n\n";
        }
        {
            std::ofstream f(dir / (base + ".surface_classes.mdl"));
            if (write_surface_classes(f))
                main_f << "INCLUDE_FILE = \"" << base << ".surface_classes.mdl\"\n\n";
        }
        {
            std::ofstream f(dir / (base + ".reactions.mdl"));
            if (write_reactions(f))
                main_f << "INCLUDE_FILE = \"" << base << ".reactions.mdl\"\n\n";
        }
        {
            std::ofstream f(dir / (base + ".geometry.mdl"));
            if (write_geometry(f))
                main_f << "INCLUDE_FILE = \"" << base << ".geometry.mdl\"\n\n";
        }
        {
            std::ofstream f(dir / (base + ".mod_surf_regions.mdl"));
            if (write_mod_surf_regions(f))
                main_f << "INCLUDE_FILE = \"" << base << ".mod_surf_regions.mdl\"\n\n";
        }
        {
            std::ofstream f(dir / (base + ".release_patterns.mdl"));
            if (write_release_patterns(f))
                main_f << "INCLUDE_FILE = \"" << base << ".release_patterns.mdl\"\n\n";
        }

        write_instantiation(main_f);
        main_f << "sprintf(seed,\"%05g\",SEED)\n\n";

        {
            std::ofstream f(dir / (base + ".viz_output.mdl"));
            if (write_viz_output(f))
                main_f << "INCLUDE_FILE = \"" << base << ".viz_output.mdl\"\n\n";
        }
        {
            std::ofstream f(dir / (base + ".rxn_output.mdl"));
            if (write_rxn_output(f))
                main_f << "INCLUDE_FILE = \"" << base << ".rxn_output.mdl\"\n\n";
        }
    }

    void write_single(const std::filesystem::path& filepath) const {
        std::ofstream f(filepath);
        if (!f) throw std::runtime_error("Cannot open MDL file: " + filepath.string());
        write_parameters(f);
        write_initialization(f);
        write_molecules(f);
        write_surface_classes(f);
        write_reactions(f);
        write_geometry(f);
        write_mod_surf_regions(f);
        write_release_patterns(f);
        write_instantiation(f);
        f << "sprintf(seed,\"%05g\",SEED)\n\n";
        write_viz_output(f);
        write_rxn_output(f);
    }

    bool write_parameters(std::ostream& f) const {
        auto names = dm_.parameters.get_all_names();
        if (names.empty()) return false;
        for (auto& n : names) {
            int id = dm_.parameters.get_id(n);
            std::string expr = dm_.parameters.get_expr(id);
            if (!expr.empty()) f << n << " = " << expr << "\n";
        }
        f << "\n";
        return true;
    }

    bool write_molecules(std::ostream& f) const {
        if (dm_.molecules.empty()) return false;
        f << "DEFINE_MOLECULES\n{\n";
        for (auto& m : dm_.molecules) {
            f << "  " << m.name << "\n  {\n";
            if (m.mol_type == MoleculeSpecies::SURFACE_2D)
                f << "    DIFFUSION_CONSTANT_2D = " << m.diffusion_constant << "\n";
            else
                f << "    DIFFUSION_CONSTANT_3D = " << m.diffusion_constant << "\n";
            if (!m.custom_time_step.empty())
                f << "    CUSTOM_TIME_STEP = " << m.custom_time_step << "\n";
            if (!m.custom_space_step.empty())
                f << "    CUSTOM_SPACE_STEP = " << m.custom_space_step << "\n";
            if (!m.maximum_step_length.empty())
                f << "    MAXIMUM_STEP_LENGTH = " << m.maximum_step_length << "\n";
            if (m.target_only)
                f << "    TARGET_ONLY\n";
            f << "  }\n";
        }
        f << "}\n\n";
        return true;
    }

    bool write_reactions(std::ostream& f) const {
        if (dm_.reactions.empty()) return false;
        f << "DEFINE_REACTIONS\n{\n";
        for (auto& r : dm_.reactions) {
            f << "  " << r.name << " ";
            if (r.rxn_type == Reaction::IRREVERSIBLE) {
                if (r.variable_rate_switch && r.variable_rate_valid)
                    f << "[\"" << r.variable_rate << "\"]";
                else
                    f << "[" << r.fwd_rate << "]";
            } else {
                f << "[>" << r.fwd_rate << ", <" << r.bkwd_rate << "]";
            }
            if (!r.rxn_name.empty()) f << " : " << r.rxn_name;
            f << "\n";
        }
        f << "}\n\n";
        return true;
    }

    bool write_surface_classes(std::ostream& f) const {
        if (dm_.surface_classes.empty()) return false;
        f << "DEFINE_SURFACE_CLASSES\n{\n";
        for (auto& sc : dm_.surface_classes) {
            f << "  " << sc.name << "\n  {\n";
            for (auto& prop : sc.properties) {
                std::string mol_str;
                if (prop.affected_mols == SurfaceClassProperty::SINGLE)
                    mol_str = prop.molecule + prop.orient_str();
                else
                    mol_str = prop.affected_mols_str() + prop.orient_str();
                f << "    " << prop.class_type_str() << " = " << mol_str;
                if (prop.class_type == SurfaceClassProperty::CLAMP_CONCENTRATION)
                    f << " = " << prop.clamp_value;
                f << "\n";
            }
            f << "  }\n";
        }
        f << "}\n\n";
        return true;
    }

    bool write_geometry(std::ostream& f) const {
        if (dm_.geometry_objects.empty()) return false;
        for (auto& obj : dm_.geometry_objects) {
            f << obj.name << " POLYGON_LIST\n{\n";
            f << "  VERTEX_LIST\n  {\n";
            for (auto& v : obj.vertices) {
                f << "    [ " << std::setprecision(15) << v.x + obj.location.x
                  << ", " << v.y + obj.location.y
                  << ", " << v.z + obj.location.z << " ]\n";
            }
            f << "  }\n";
            f << "  ELEMENT_CONNECTIONS\n  {\n";
            for (auto& face : obj.faces) {
                f << "    [ " << face.v0 << ", " << face.v1 << ", " << face.v2 << " ]\n";
            }
            f << "  }\n";
            if (!obj.regions.empty()) {
                f << "  DEFINE_SURFACE_REGIONS\n  {\n";
                for (auto& reg : obj.regions) {
                    f << "    " << reg.name << "\n    {\n";
                    f << "      ELEMENT_LIST = [";
                    for (size_t i = 0; i < reg.face_indices.size(); ++i) {
                        if (i > 0) f << ", ";
                        f << reg.face_indices[i];
                    }
                    f << "]\n    }\n";
                }
                f << "  }\n";
            }
            f << "}\n\n";
        }
        return true;
    }

    void write_initialization(std::ostream& f) const {
        const auto& init = dm_.initialization;
        f << "ITERATIONS = " << init.iterations << "\n";
        f << "TIME_STEP = " << init.time_step << "\n";
        if (!init.vacancy_search_distance.empty())
            f << "VACANCY_SEARCH_DISTANCE = " << init.vacancy_search_distance << "\n";
        else
            f << "VACANCY_SEARCH_DISTANCE = 10\n";
        f << "\n";
        if (!init.time_step_max.empty()) f << "TIME_STEP_MAX = " << init.time_step_max << "\n";
        if (!init.space_step.empty()) f << "SPACE_STEP = " << init.space_step << "\n";
        if (!init.interaction_radius.empty()) f << "INTERACTION_RADIUS = " << init.interaction_radius << "\n";
        if (!init.radial_directions.empty()) f << "RADIAL_DIRECTIONS = " << init.radial_directions << "\n";
        if (!init.radial_subdivisions.empty()) f << "RADIAL_SUBDIVISIONS = " << init.radial_subdivisions << "\n";
        f << "SURFACE_GRID_DENSITY = " << init.surface_grid_density << "\n";
        f << "ACCURATE_3D_REACTIONS = " << (init.accurate_3d_reactions ? "TRUE" : "FALSE") << "\n";
        f << "CENTER_MOLECULES_ON_GRID = " << (init.center_molecules_grid ? "TRUE" : "FALSE") << "\n";
        f << "MICROSCOPIC_REVERSIBILITY = " << init.microscopic_reversibility << "\n\n";

        f << "NOTIFICATIONS\n{\n";
        if (init.all_notifications == "INDIVIDUAL") {
            if (init.probability_report == "THRESHOLD")
                f << "   PROBABILITY_REPORT_THRESHOLD = " << std::setprecision(15) << init.probability_report_threshold << "\n";
            else
                f << "   PROBABILITY_REPORT = " << init.probability_report << "\n";
            f << "   DIFFUSION_CONSTANT_REPORT = " << init.diffusion_constant_report << "\n";
            f << "   FILE_OUTPUT_REPORT = " << (init.file_output_report ? "ON" : "OFF") << "\n";
            f << "   FINAL_SUMMARY = " << (init.final_summary ? "ON" : "OFF") << "\n";
            f << "   ITERATION_REPORT = " << (init.iteration_report ? "ON" : "OFF") << "\n";
            f << "   PARTITION_LOCATION_REPORT = " << (init.partition_location_report ? "ON" : "OFF") << "\n";
            f << "   VARYING_PROBABILITY_REPORT = " << (init.varying_probability_report ? "ON" : "OFF") << "\n";
            f << "   PROGRESS_REPORT = " << (init.progress_report ? "ON" : "OFF") << "\n";
            f << "   RELEASE_EVENT_REPORT = " << (init.release_event_report ? "ON" : "OFF") << "\n";
            f << "   MOLECULE_COLLISION_REPORT = " << (init.molecule_collision_report ? "ON" : "OFF") << "\n";
        } else {
            f << "   ALL_NOTIFICATIONS = " << init.all_notifications << "\n";
        }
        f << "}\n\n";

        f << "WARNINGS\n{\n";
        if (init.all_warnings == "INDIVIDUAL") {
            f << "   DEGENERATE_POLYGONS = " << init.degenerate_polygons << "\n";
            f << "   NEGATIVE_DIFFUSION_CONSTANT = " << init.negative_diffusion_constant << "\n";
            f << "   MISSING_SURFACE_ORIENTATION = " << init.missing_surface_orientation << "\n";
            f << "   NEGATIVE_REACTION_RATE = " << init.negative_reaction_rate << "\n";
            f << "   USELESS_VOLUME_ORIENTATION = " << init.useless_volume_orientation << "\n";
            f << "   HIGH_REACTION_PROBABILITY = " << init.high_reaction_probability << "\n";
            f << "   LIFETIME_TOO_SHORT = " << init.lifetime_too_short << "\n";
            if (init.lifetime_too_short == "WARNING")
                f << "   LIFETIME_THRESHOLD = " << init.lifetime_threshold << "\n";
            f << "   MISSED_REACTIONS = " << init.missed_reactions << "\n";
            if (init.missed_reactions == "WARNING")
                f << "   MISSED_REACTION_THRESHOLD = " << std::setprecision(15) << init.missed_reaction_threshold << "\n";
        } else {
            f << "   ALL_WARNINGS = " << init.all_warnings << "\n";
        }
        f << "}\n\n";

        if (init.partitions.include) {
            f << "PARTITION_X = [[" << std::setprecision(15) << init.partitions.x_start
              << " TO " << init.partitions.x_end << " STEP " << init.partitions.x_step << "]]\n";
            f << "PARTITION_Y = [[" << init.partitions.y_start
              << " TO " << init.partitions.y_end << " STEP " << init.partitions.y_step << "]]\n";
            f << "PARTITION_Z = [[" << init.partitions.z_start
              << " TO " << init.partitions.z_end << " STEP " << init.partitions.z_step << "]]\n\n";
        }
    }

    void write_instantiation(std::ostream& f) const {
        bool has_objects = !dm_.geometry_objects.empty();
        bool has_releases = !dm_.release_sites.empty();
        if (!has_objects && !has_releases) return;

        f << "INSTANTIATE " << dm_.scene_name << " OBJECT\n{\n";
        for (auto& obj : dm_.geometry_objects)
            f << "  " << obj.name << " OBJECT " << obj.name << " {}\n";

        for (auto& rel : dm_.release_sites) {
            f << "  " << rel.name << " RELEASE_SITE\n  {\n";
            if (rel.shape == ReleaseSite::LIST || rel.shape == ReleaseSite::CUBIC ||
                rel.shape == ReleaseSite::SPHERICAL || rel.shape == ReleaseSite::SPHERICAL_SHELL) {
                f << "   SHAPE = " << rel.shape_str() << "\n";
                if (rel.shape != ReleaseSite::LIST)
                    f << "   LOCATION = [" << rel.location_x << ", " << rel.location_y << ", " << rel.location_z << "]\n";
                f << "   SITE_DIAMETER = " << rel.diameter << "\n";
            }
            if (rel.shape == ReleaseSite::OBJECT)
                f << "   SHAPE = " << dm_.scene_name << "." << rel.object_expr << "\n";

            std::string mol_spec = rel.molecule;
            for (auto& m : dm_.molecules) {
                if (m.name == rel.molecule && m.mol_type == MoleculeSpecies::SURFACE_2D) {
                    mol_spec = rel.molecule + rel.orient_str();
                    break;
                }
            }

            if (rel.shape == ReleaseSite::LIST) {
                f << "   MOLECULE_POSITIONS\n   {\n";
                for (auto& p : rel.points_list)
                    f << "     " << mol_spec << " [" << std::setprecision(15) << p.x << ", " << p.y << ", " << p.z << "]\n";
                f << "   }\n";
            } else {
                f << "   MOLECULE = " << mol_spec << "\n";
                if (rel.quantity_type == ReleaseSite::NUMBER_TO_RELEASE) {
                    f << "   NUMBER_TO_RELEASE = " << rel.quantity << "\n";
                } else if (rel.quantity_type == ReleaseSite::GAUSSIAN_RELEASE_NUMBER) {
                    f << "   GAUSSIAN_RELEASE_NUMBER\n   {\n";
                    f << "        MEAN_NUMBER = " << rel.quantity << "\n";
                    f << "        STANDARD_DEVIATION = " << rel.stddev << "\n";
                    f << "   }\n";
                } else if (rel.quantity_type == ReleaseSite::DENSITY) {
                    bool is_2d = false;
                    for (auto& m : dm_.molecules) {
                        if (m.name == rel.molecule && m.mol_type == MoleculeSpecies::SURFACE_2D) {
                            is_2d = true; break;
                        }
                    }
                    f << (is_2d ? "   DENSITY = " : "   CONCENTRATION = ") << rel.quantity << "\n";
                }
            }
            f << "   RELEASE_PROBABILITY = " << rel.probability << "\n";
            if (!rel.pattern.empty()) f << "   RELEASE_PATTERN = " << rel.pattern << "\n";
            f << "  }\n";
        }
        f << "}\n\n";
    }

    bool write_release_patterns(std::ostream& f) const {
        if (dm_.release_patterns.empty()) return false;
        for (auto& rp : dm_.release_patterns) {
            f << "DEFINE_RELEASE_PATTERN " << rp.name << "\n{\n";
            f << "  DELAY = " << rp.delay << "\n";
            if (!rp.release_interval.empty()) f << "  RELEASE_INTERVAL = " << rp.release_interval << "\n";
            if (!rp.train_duration.empty()) f << "  TRAIN_DURATION = " << rp.train_duration << "\n";
            if (!rp.train_interval.empty()) f << "  TRAIN_INTERVAL = " << rp.train_interval << "\n";
            f << "  NUMBER_OF_TRAINS = " << rp.number_of_trains << "\n";
            f << "}\n\n";
        }
        return true;
    }

    bool write_mod_surf_regions(std::ostream& f) const {
        if (dm_.mod_surf_regions.empty()) return false;
        f << "MODIFY_SURFACE_REGIONS\n{\n";
        for (auto& msr : dm_.mod_surf_regions) {
            f << "  " << msr.object_name << "[" << msr.region_name << "]\n  {\n";
            f << "    SURFACE_CLASS = " << msr.surf_class_name << "\n  }\n";
        }
        f << "}\n\n";
        return true;
    }

    bool write_viz_output(std::ostream& f) const {
        std::string mol_list_str;
        if (dm_.viz_output.export_all) {
            mol_list_str = "ALL_MOLECULES";
        } else {
            for (auto& m : dm_.molecules) {
                if (m.export_viz) {
                    if (!mol_list_str.empty()) mol_list_str += " ";
                    mol_list_str += m.name;
                }
            }
        }
        if (mol_list_str.empty()) return false;
        f << "VIZ_OUTPUT\n{\n";
        f << (dm_.initialization.export_all_ascii ? "  MODE = ASCII\n" : "  MODE = CELLBLENDER\n");
        f << "  FILENAME = \"./viz_data/seed_\" & seed & \"/" << dm_.scene_name << "\"\n";
        f << "  MOLECULES\n  {\n";
        f << "    NAME_LIST {" << mol_list_str << "}\n";
        if (dm_.viz_output.all_iterations)
            f << "    ITERATION_NUMBERS {ALL_DATA @ ALL_ITERATIONS}\n";
        else
            f << "    ITERATION_NUMBERS {ALL_DATA @ [[" << dm_.viz_output.start
              << " TO " << dm_.viz_output.end_val << " STEP " << dm_.viz_output.step << "]]}\n";
        f << "  }\n}\n\n";
        return true;
    }

    bool write_rxn_output(std::ostream& f) const {
        if (dm_.reaction_outputs.empty()) return false;
        f << "REACTION_DATA_OUTPUT\n{\n";
        f << "  STEP = " << dm_.initialization.time_step << "\n";
        for (auto& ro : dm_.reaction_outputs) {
            std::string count_expr, file_name;
            if (ro.rxn_or_mol == ReactionOutput::MOLECULE) {
                if (ro.count_location == ReactionOutput::WORLD) {
                    count_expr = "COUNT[" + ro.molecule_name + ",WORLD]";
                    file_name = ro.molecule_name + ".World.dat";
                } else if (ro.count_location == ReactionOutput::OBJECT_LOC) {
                    count_expr = "COUNT[" + ro.molecule_name + "," + dm_.scene_name + "." + ro.object_name + "]";
                    file_name = ro.molecule_name + "." + ro.object_name + ".dat";
                } else {
                    count_expr = "COUNT[" + ro.molecule_name + "," + dm_.scene_name + "." + ro.object_name + "[" + ro.region_name + "]]";
                    file_name = ro.molecule_name + "." + ro.object_name + "." + ro.region_name + ".dat";
                }
            } else if (ro.rxn_or_mol == ReactionOutput::REACTION) {
                count_expr = "COUNT[" + ro.reaction_name + ",WORLD]";
                file_name = ro.reaction_name + ".World.dat";
            } else if (ro.rxn_or_mol == ReactionOutput::MDL_STRING) {
                count_expr = ro.mdl_string;
                file_name = ro.mdl_file_prefix + "_MDLString.dat";
            }
            if (!count_expr.empty() && !file_name.empty())
                f << "  { " << count_expr << " } => \"./react_data/seed_\" & seed & \"/" << file_name << "\"\n";
        }
        f << "}\n\n";
        return true;
    }

private:
    const DataModel& dm_;
};

// ============================================================================
// MDL Parser - reads MCell MDL files back into a DataModel
// ============================================================================
class MDLParser {
public:
    DataModel parse(const std::filesystem::path& filepath) {
        DataModel dm;
        std::ifstream f(filepath);
        if (!f) throw std::runtime_error("Cannot open MDL file: " + filepath.string());
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        pos_ = 0;
        src_ = content;
        base_dir_ = filepath.parent_path();

        while (pos_ < src_.size()) {
            skip_ws();
            if (pos_ >= src_.size()) break;
            std::string token = read_token();
            if (token.empty()) break;

            if (token == "INCLUDE_FILE") {
                skip_ws(); expect('='); skip_ws();
                std::string inc = read_quoted_string();
                auto inc_path = base_dir_ / inc;
                if (std::filesystem::exists(inc_path)) {
                    MDLParser sub;
                    DataModel sub_dm = sub.parse(inc_path);
                    merge(dm, sub_dm);
                }
            } else if (token == "ITERATIONS") {
                skip_ws(); expect('='); dm.initialization.iterations = read_value_string();
            } else if (token == "TIME_STEP") {
                skip_ws(); expect('='); dm.initialization.time_step = read_value_string();
            } else if (token == "VACANCY_SEARCH_DISTANCE") {
                skip_ws(); expect('='); dm.initialization.vacancy_search_distance = read_value_string();
            } else if (token == "DEFINE_MOLECULES") {
                parse_molecules(dm);
            } else if (token == "DEFINE_REACTIONS") {
                parse_reactions(dm);
            } else if (token == "PARTITION_X") {
                dm.initialization.partitions.include = true;
                parse_partition(dm.initialization.partitions.x_start, dm.initialization.partitions.x_end, dm.initialization.partitions.x_step);
            } else if (token == "PARTITION_Y") {
                dm.initialization.partitions.include = true;
                parse_partition(dm.initialization.partitions.y_start, dm.initialization.partitions.y_end, dm.initialization.partitions.y_step);
            } else if (token == "PARTITION_Z") {
                dm.initialization.partitions.include = true;
                parse_partition(dm.initialization.partitions.z_start, dm.initialization.partitions.z_end, dm.initialization.partitions.z_step);
            } else {
                skip_ws();
                if (pos_ < src_.size()) {
                    std::string next = peek_token();
                    if (next == "POLYGON_LIST") {
                        read_token();
                        parse_polygon_list(dm, token);
                    }
                }
            }
        }
        return dm;
    }

private:
    std::string src_;
    size_t pos_ = 0;
    std::filesystem::path base_dir_;

    void skip_ws() {
        while (pos_ < src_.size()) {
            if (std::isspace((unsigned char)src_[pos_])) { pos_++; }
            else if (pos_+1 < src_.size() && src_[pos_]=='/' && src_[pos_+1]=='*') {
                pos_ += 2;
                while (pos_+1 < src_.size() && !(src_[pos_]=='*' && src_[pos_+1]=='/')) pos_++;
                if (pos_+1 < src_.size()) pos_ += 2;
            } else break;
        }
    }

    void expect(char c) {
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == c) pos_++;
    }

    std::string read_token() {
        skip_ws();
        if (pos_ >= src_.size()) return "";
        if (src_[pos_]=='{' || src_[pos_]=='}' || src_[pos_]=='[' || src_[pos_]==']')
            return std::string(1, src_[pos_++]);
        size_t start = pos_;
        while (pos_ < src_.size() && !std::isspace((unsigned char)src_[pos_])
               && src_[pos_]!='=' && src_[pos_]!='{' && src_[pos_]!='}'
               && src_[pos_]!='[' && src_[pos_]!=']' && src_[pos_]!=',')
            pos_++;
        return src_.substr(start, pos_ - start);
    }

    std::string peek_token() { size_t s = pos_; auto t = read_token(); pos_ = s; return t; }

    std::string read_quoted_string() {
        skip_ws();
        if (pos_ >= src_.size() || src_[pos_] != '"') return read_token();
        pos_++;
        size_t start = pos_;
        while (pos_ < src_.size() && src_[pos_] != '"') pos_++;
        std::string r = src_.substr(start, pos_ - start);
        if (pos_ < src_.size()) pos_++;
        return r;
    }

    std::string read_value_string() {
        skip_ws();
        size_t start = pos_;
        while (pos_ < src_.size() && src_[pos_] != '\n' && src_[pos_] != '{' && src_[pos_] != '}') pos_++;
        std::string val = src_.substr(start, pos_ - start);
        auto end = val.find_last_not_of(" \t\r");
        if (end != std::string::npos) val = val.substr(0, end + 1);
        return val;
    }

    void skip_block() { int d = 1; while (pos_ < src_.size() && d > 0) { if (src_[pos_]=='{') d++; else if (src_[pos_]=='}') d--; pos_++; } }

    void parse_molecules(DataModel& dm) {
        skip_ws(); expect('{');
        while (true) {
            skip_ws();
            if (pos_ >= src_.size() || src_[pos_] == '}') { if (pos_ < src_.size()) pos_++; break; }
            std::string mol_name = read_token();
            if (mol_name.empty() || mol_name == "}") break;
            MoleculeSpecies mol; mol.name = mol_name;
            skip_ws(); expect('{');
            while (true) {
                skip_ws();
                if (pos_ >= src_.size() || src_[pos_]=='}') { if (pos_ < src_.size()) pos_++; break; }
                std::string key = read_token();
                if (key == "}") break;
                if (key == "DIFFUSION_CONSTANT_3D") { expect('='); mol.diffusion_constant = read_value_string(); mol.mol_type = MoleculeSpecies::VOLUME_3D; }
                else if (key == "DIFFUSION_CONSTANT_2D") { expect('='); mol.diffusion_constant = read_value_string(); mol.mol_type = MoleculeSpecies::SURFACE_2D; }
                else if (key == "TARGET_ONLY") { mol.target_only = true; }
                else if (key == "CUSTOM_TIME_STEP") { expect('='); mol.custom_time_step = read_value_string(); }
                else if (key == "CUSTOM_SPACE_STEP") { expect('='); mol.custom_space_step = read_value_string(); }
            }
            dm.molecules.push_back(std::move(mol));
        }
    }

    void parse_reactions(DataModel& dm) {
        skip_ws(); expect('{');
        int depth = 1; size_t bs = pos_;
        while (pos_ < src_.size() && depth > 0) { if (src_[pos_]=='{') depth++; else if (src_[pos_]=='}') depth--; if (depth > 0) pos_++; }
        std::string block = src_.substr(bs, pos_ - bs);
        if (pos_ < src_.size()) pos_++;
        std::istringstream ss(block); std::string line;
        while (std::getline(ss, line)) {
            auto fs = line.find_first_not_of(" \t");
            if (fs == std::string::npos) continue;
            line = line.substr(fs);
            if (line.empty() || line[0] == '/' || line[0] == '#') continue;
            auto bs2 = line.find('['); auto be = line.rfind(']');
            if (bs2 == std::string::npos || be == std::string::npos) continue;
            Reaction rxn;
            rxn.name = line.substr(0, bs2);
            while (!rxn.name.empty() && rxn.name.back() == ' ') rxn.name.pop_back();
            auto dap = rxn.name.find(" <-> ");
            auto ap = rxn.name.find(" -> ");
            if (dap != std::string::npos) {
                rxn.reactants = rxn.name.substr(0, dap); rxn.products = rxn.name.substr(dap + 5); rxn.rxn_type = Reaction::REVERSIBLE;
            } else if (ap != std::string::npos) {
                rxn.reactants = rxn.name.substr(0, ap); rxn.products = rxn.name.substr(ap + 4); rxn.rxn_type = Reaction::IRREVERSIBLE;
            }
            std::string rate = line.substr(bs2 + 1, be - bs2 - 1);
            if (rxn.rxn_type == Reaction::REVERSIBLE) {
                auto c = rate.find(',');
                if (c != std::string::npos) {
                    std::string fw = rate.substr(0,c), bk = rate.substr(c+1);
                    if (!fw.empty() && fw[0]=='>') fw = fw.substr(1);
                    auto lt = bk.find('<'); if (lt!=std::string::npos) bk = bk.substr(lt+1);
                    auto trim = [](std::string& s) { while(!s.empty()&&s[0]==' ')s=s.substr(1); while(!s.empty()&&s.back()==' ')s.pop_back(); };
                    trim(fw); trim(bk);
                    rxn.fwd_rate = fw; rxn.bkwd_rate = bk;
                }
            } else {
                auto trim = [](std::string& s) { while(!s.empty()&&s[0]==' ')s=s.substr(1); while(!s.empty()&&s.back()==' ')s.pop_back(); };
                trim(rate); rxn.fwd_rate = rate;
            }
            auto cp = line.find(':', be);
            if (cp != std::string::npos) {
                rxn.rxn_name = line.substr(cp+1);
                auto trim = [](std::string& s) { while(!s.empty()&&s[0]==' ')s=s.substr(1); while(!s.empty()&&s.back()==' ')s.pop_back(); };
                trim(rxn.rxn_name);
            }
            dm.reactions.push_back(std::move(rxn));
        }
    }

    void parse_partition(double& start, double& end, double& step) {
        skip_ws(); expect('=');
        std::string val_str = read_value_string();
        std::regex re(R"(\[\[\s*([-\d.eE+]+)\s+TO\s+([-\d.eE+]+)\s+STEP\s+([-\d.eE+]+)\s*\]\])");
        std::smatch m;
        if (std::regex_search(val_str, m, re)) {
            try { start = std::stod(m[1].str()); end = std::stod(m[2].str()); step = std::stod(m[3].str()); } catch (...) {}
        }
    }

    void parse_polygon_list(DataModel& dm, const std::string& obj_name) {
        GeometryObject obj; obj.name = obj_name;
        skip_ws(); expect('{');
        while (true) {
            skip_ws();
            if (pos_ >= src_.size() || src_[pos_]=='}') { if (pos_ < src_.size()) pos_++; break; }
            std::string key = read_token();
            if (key == "}") break;
            if (key == "VERTEX_LIST") {
                skip_ws(); expect('{');
                while (true) {
                    skip_ws();
                    if (pos_ >= src_.size() || src_[pos_]=='}') { pos_++; break; }
                    expect('[');
                    double x = std::stod(read_token()); expect(',');
                    double y = std::stod(read_token()); expect(',');
                    double z = std::stod(read_token());
                    skip_ws(); if (pos_ < src_.size() && src_[pos_]==']') pos_++;
                    obj.vertices.push_back({x,y,z});
                }
            } else if (key == "ELEMENT_CONNECTIONS") {
                skip_ws(); expect('{');
                while (true) {
                    skip_ws();
                    if (pos_ >= src_.size() || src_[pos_]=='}') { pos_++; break; }
                    expect('[');
                    int v0 = std::stoi(read_token()); expect(',');
                    int v1 = std::stoi(read_token()); expect(',');
                    int v2 = std::stoi(read_token());
                    skip_ws(); if (pos_ < src_.size() && src_[pos_]==']') pos_++;
                    obj.faces.push_back({v0,v1,v2});
                }
            } else if (key == "DEFINE_SURFACE_REGIONS") {
                skip_ws(); expect('{');
                while (true) {
                    skip_ws();
                    if (pos_ >= src_.size() || src_[pos_]=='}') { pos_++; break; }
                    std::string rn = read_token();
                    if (rn == "}") break;
                    SurfaceRegion reg; reg.name = rn;
                    skip_ws(); expect('{');
                    while (true) {
                        skip_ws();
                        if (pos_ >= src_.size() || src_[pos_]=='}') { pos_++; break; }
                        std::string rk = read_token();
                        if (rk == "}") break;
                        if (rk == "ELEMENT_LIST") {
                            expect('='); skip_ws(); expect('[');
                            while (true) {
                                skip_ws();
                                if (pos_ >= src_.size() || src_[pos_]==']') { pos_++; break; }
                                std::string num = read_token();
                                if (num == "]") break;
                                if (!num.empty() && num.back()==',') num.pop_back();
                                if (!num.empty()) { try { reg.face_indices.push_back(std::stoi(num)); } catch (...) {} }
                                skip_ws(); if (pos_ < src_.size() && src_[pos_]==',') pos_++;
                            }
                        }
                    }
                    obj.regions.push_back(std::move(reg));
                }
            } else if (key == "TRANSLATE") {
                expect('='); skip_ws(); expect('[');
                obj.location.x = std::stod(read_token()); expect(',');
                obj.location.y = std::stod(read_token()); expect(',');
                obj.location.z = std::stod(read_token());
                skip_ws(); if (pos_ < src_.size() && src_[pos_]==']') pos_++;
            }
        }
        dm.geometry_objects.push_back(std::move(obj));
    }

    void merge(DataModel& dst, const DataModel& src) {
        for (auto& m : src.molecules) dst.molecules.push_back(m);
        for (auto& r : src.reactions) dst.reactions.push_back(r);
        for (auto& g : src.geometry_objects) dst.geometry_objects.push_back(g);
        for (auto& sc : src.surface_classes) dst.surface_classes.push_back(sc);
        if (src.initialization.partitions.include) dst.initialization.partitions = src.initialization.partitions;
        if (!src.initialization.iterations.empty() && src.initialization.iterations != "1000")
            dst.initialization.iterations = src.initialization.iterations;
        if (!src.initialization.time_step.empty() && src.initialization.time_step != "1e-6")
            dst.initialization.time_step = src.initialization.time_step;
    }
};

// ============================================================================
// Data Model JSON Serialization
// ============================================================================
class DataModelIO {
public:
    static void save_json(const DataModel& dm, const std::filesystem::path& filepath) {
        std::ofstream f(filepath);
        if (!f) throw std::runtime_error("Cannot open file: " + filepath.string());
        f << "{\n  \"mcell\": {\n";
        f << "    \"parameter_system\": {\n      \"model_parameters\": [\n";
        auto names = dm.parameters.get_all_names();
        for (size_t i = 0; i < names.size(); ++i) {
            int id = dm.parameters.get_id(names[i]);
            f << "        {\"par_name\": \"" << names[i] << "\", \"par_expression\": \"" << dm.parameters.get_expr(id) << "\"}";
            if (i + 1 < names.size()) f << ",";
            f << "\n";
        }
        f << "      ]\n    },\n";
        f << "    \"initialization\": {\n      \"iterations\": \"" << dm.initialization.iterations
          << "\",\n      \"time_step\": \"" << dm.initialization.time_step
          << "\",\n      \"vacancy_search_distance\": \"" << dm.initialization.vacancy_search_distance << "\"\n    },\n";
        f << "    \"define_molecules\": {\n      \"molecule_list\": [\n";
        for (size_t i = 0; i < dm.molecules.size(); ++i) {
            auto& m = dm.molecules[i];
            f << "        {\"mol_name\": \"" << m.name << "\", \"mol_type\": \""
              << (m.mol_type == MoleculeSpecies::SURFACE_2D ? "2D" : "3D")
              << "\", \"diffusion_constant\": \"" << m.diffusion_constant
              << "\", \"target_only\": " << (m.target_only ? "true" : "false")
              << ", \"export_viz\": " << (m.export_viz ? "true" : "false") << "}";
            if (i + 1 < dm.molecules.size()) f << ",";
            f << "\n";
        }
        f << "      ]\n    },\n";
        f << "    \"define_reactions\": {\n      \"reaction_list\": [\n";
        for (size_t i = 0; i < dm.reactions.size(); ++i) {
            auto& r = dm.reactions[i];
            f << "        {\"name\": \"" << r.name << "\", \"reactants\": \"" << r.reactants
              << "\", \"products\": \"" << r.products << "\", \"rxn_type\": \""
              << (r.rxn_type == Reaction::REVERSIBLE ? "reversible" : "irreversible")
              << "\", \"fwd_rate\": \"" << r.fwd_rate << "\", \"bkwd_rate\": \"" << r.bkwd_rate << "\"}";
            if (i + 1 < dm.reactions.size()) f << ",";
            f << "\n";
        }
        f << "      ]\n    },\n";
        f << "    \"geometrical_objects\": {\n      \"object_list\": [\n";
        for (size_t i = 0; i < dm.geometry_objects.size(); ++i) {
            auto& g = dm.geometry_objects[i];
            f << "        {\"name\": \"" << g.name << "\", \"vertex_list\": [";
            for (size_t j = 0; j < g.vertices.size(); ++j) {
                f << "[" << std::setprecision(15) << g.vertices[j].x << "," << g.vertices[j].y << "," << g.vertices[j].z << "]";
                if (j + 1 < g.vertices.size()) f << ",";
            }
            f << "], \"element_connections\": [";
            for (size_t j = 0; j < g.faces.size(); ++j) {
                f << "[" << g.faces[j].v0 << "," << g.faces[j].v1 << "," << g.faces[j].v2 << "]";
                if (j + 1 < g.faces.size()) f << ",";
            }
            f << "]}";
            if (i + 1 < dm.geometry_objects.size()) f << ",";
            f << "\n";
        }
        f << "      ]\n    }\n  }\n}\n";
    }
};

// ============================================================================
// Results Parser - parses MCell reaction output data files
// ============================================================================
struct TimeSeriesData {
    std::string label;
    std::vector<double> times;
    std::vector<double> values;
};

class ResultsParser {
public:
    static TimeSeriesData parse_dat_file(const std::filesystem::path& filepath) {
        TimeSeriesData ts;
        ts.label = filepath.stem().string();
        std::ifstream f(filepath);
        if (!f) return ts;
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            double t, v;
            if (iss >> t >> v) { ts.times.push_back(t); ts.values.push_back(v); }
        }
        return ts;
    }

    static std::vector<TimeSeriesData> parse_seed_dir(const std::filesystem::path& dir) {
        std::vector<TimeSeriesData> results;
        namespace fs = std::filesystem;
        if (!fs::exists(dir)) return results;
        for (auto& entry : fs::directory_iterator(dir))
            if (entry.path().extension() == ".dat")
                results.push_back(parse_dat_file(entry.path()));
        return results;
    }

    struct AggregatedTimeSeries {
        std::string label;
        std::vector<double> times;
        std::vector<double> means;
        std::vector<double> stddevs;
        int num_seeds = 0;
    };

    static AggregatedTimeSeries aggregate_seeds(const std::filesystem::path& react_data_dir,
        const std::string& file_name, int start_seed, int end_seed) {
        AggregatedTimeSeries agg;
        agg.label = std::filesystem::path(file_name).stem().string();
        std::vector<std::vector<double>> all_values;
        for (int seed = start_seed; seed <= end_seed; ++seed) {
            char sd[64]; std::snprintf(sd, sizeof(sd), "seed_%05d", seed);
            auto ts = parse_dat_file(react_data_dir / sd / file_name);
            if (ts.times.empty()) continue;
            if (agg.times.empty()) agg.times = ts.times;
            all_values.push_back(std::move(ts.values));
        }
        agg.num_seeds = static_cast<int>(all_values.size());
        if (agg.num_seeds == 0) return agg;
        size_t n = agg.times.size();
        agg.means.resize(n, 0); agg.stddevs.resize(n, 0);
        for (size_t i = 0; i < n; ++i) {
            double sum = 0; int count = 0;
            for (auto& vals : all_values) { if (i < vals.size()) { sum += vals[i]; count++; } }
            double mean = count > 0 ? sum / count : 0;
            agg.means[i] = mean;
            double var = 0;
            for (auto& vals : all_values) { if (i < vals.size()) { double d = vals[i] - mean; var += d*d; } }
            agg.stddevs[i] = count > 1 ? std::sqrt(var / (count - 1)) : 0;
        }
        return agg;
    }
};

// ============================================================================
// Parameter Sweep Engine
// ============================================================================
struct SweepParameter {
    std::string name;
    std::vector<double> values;
};

class ParameterSweepEngine {
public:
    void add_parameter(const std::string& name, const std::vector<double>& values) {
        params_.push_back({name, values});
    }

    size_t total_runs() const {
        if (params_.empty()) return 1;
        size_t total = 1;
        for (auto& p : params_) total *= p.values.size();
        return total;
    }

    std::map<std::string, double> get_point(size_t run_index) const {
        std::map<std::string, double> point;
        size_t idx = run_index;
        for (int i = static_cast<int>(params_.size()) - 1; i >= 0; --i) {
            size_t n = params_[i].values.size();
            point[params_[i].name] = params_[i].values[idx % n];
            idx /= n;
        }
        return point;
    }

    std::string get_run_path(size_t run_index) const {
        std::string path;
        size_t idx = run_index;
        for (int i = static_cast<int>(params_.size()) - 1; i >= 0; --i) {
            size_t n = params_[i].values.size();
            std::string seg = params_[i].name + "_index_" + std::to_string(idx % n);
            idx /= n;
            path = path.empty() ? seg : seg + "/" + path;
        }
        return path;
    }

    std::string get_label(size_t run_index) const {
        auto point = get_point(run_index);
        std::string label;
        for (auto& [name, val] : point) {
            if (!label.empty()) label += ",";
            std::ostringstream oss; oss << name << "=" << val;
            label += oss.str();
        }
        return label;
    }

    void write_data_layout(const std::filesystem::path& dir, int start_seed, int end_seed) const {
        namespace fs = std::filesystem;
        fs::create_directories(dir);
        std::ofstream f(dir / "data_layout.json");
        f << "{\n  \"version\": 2,\n  \"data_layout\": [\n";
        f << "    [\"/DIR\", [\"output_data\"]],\n";
        for (auto& p : params_) {
            f << "    [\"" << p.name << "\", [";
            for (size_t i = 0; i < p.values.size(); ++i) { if (i > 0) f << ", "; f << p.values[i]; }
            f << "]],\n";
        }
        f << "    [\"/FILE_TYPE\", [\"react_data\"]],\n";
        f << "    [\"/SEED\", [";
        for (int s = start_seed; s <= end_seed; ++s) { if (s > start_seed) f << ", "; f << s; }
        f << "]]\n  ]\n}\n";
    }

    std::vector<std::pair<std::string, DataModel>> generate_models(const DataModel& base_dm) const {
        std::vector<std::pair<std::string, DataModel>> models;
        size_t total = total_runs();
        for (size_t i = 0; i < total; ++i) {
            auto point = get_point(i);
            DataModel run_dm = base_dm;
            for (auto& [name, val] : point) {
                std::ostringstream oss; oss << std::setprecision(15) << val;
                run_dm.parameters.define(name, oss.str());
            }
            run_dm.parameters.eval_all();
            models.push_back({get_run_path(i), std::move(run_dm)});
        }
        return models;
    }

    const std::vector<SweepParameter>& parameters() const { return params_; }

private:
    std::vector<SweepParameter> params_;
};

// ============================================================================
// SimulationRunner - manages MCell simulation execution
// ============================================================================
class SimulationRunner {
public:
    struct RunConfig {
        std::string mcell_binary = "mcell";
        std::filesystem::path project_dir;
        std::string base_name = "Scene";
        int start_seed = 1;
        int end_seed = 1;
    };

    static void prepare(const DataModel& dm, const RunConfig& config) {
        namespace fs = std::filesystem;
        fs::path output_dir = config.project_dir / "output_data";
        fs::create_directories(output_dir);
        fs::create_directories(output_dir / "react_data");
        fs::create_directories(output_dir / "viz_data");
        MDLWriter writer(dm);
        writer.write_modular(output_dir);
        write_data_layout(config.project_dir, config.start_seed, config.end_seed);
    }

    static std::vector<std::vector<std::string>> generate_commands(const RunConfig& config) {
        std::vector<std::vector<std::string>> commands;
        namespace fs = std::filesystem;
        fs::path output_dir = config.project_dir / "output_data";
        std::string mdl_path = (output_dir / (config.base_name + ".main.mdl")).string();
        for (int seed = config.start_seed; seed <= config.end_seed; ++seed)
            commands.push_back({config.mcell_binary, "-seed", std::to_string(seed), mdl_path});
        return commands;
    }

    static void write_data_layout(const std::filesystem::path& dir, int start_seed, int end_seed) {
        namespace fs = std::filesystem;
        fs::create_directories(dir);
        std::ofstream f(dir / "data_layout.json");
        f << "{\n  \"version\": 2,\n  \"data_layout\": [\n";
        f << "    [\"/DIR\", [\"output_data\"]],\n";
        f << "    [\"/FILE_TYPE\", [\"react_data\"]],\n";
        f << "    [\"/SEED\", [";
        for (int s = start_seed; s <= end_seed; ++s) { if (s > start_seed) f << ", "; f << s; }
        f << "]]\n  ]\n}\n";
    }
};

// ============================================================================
// Geometry Utilities
// ============================================================================
namespace geometry {
    inline GeometryObject make_cube(const std::string& name, double half_size = 1.0) {
        GeometryObject obj; obj.name = name;
        double s = half_size;
        obj.vertices = {{-s,-s,-s},{s,-s,-s},{s,s,-s},{-s,s,-s},{-s,-s,s},{s,-s,s},{s,s,s},{-s,s,s}};
        obj.faces = {{0,1,2},{0,2,3},{4,6,5},{4,7,6},{0,4,5},{0,5,1},{2,6,7},{2,7,3},{0,3,7},{0,7,4},{1,5,6},{1,6,2}};
        return obj;
    }

    inline GeometryObject make_icosphere(const std::string& name, double radius = 1.0, int subdivisions = 1) {
        GeometryObject obj; obj.name = name;
        const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
        auto add_v = [&](double x, double y, double z) -> int {
            double l = std::sqrt(x*x+y*y+z*z);
            obj.vertices.push_back({x/l*radius, y/l*radius, z/l*radius});
            return static_cast<int>(obj.vertices.size()) - 1;
        };
        add_v(-1,phi,0); add_v(1,phi,0); add_v(-1,-phi,0); add_v(1,-phi,0);
        add_v(0,-1,phi); add_v(0,1,phi); add_v(0,-1,-phi); add_v(0,1,-phi);
        add_v(phi,0,-1); add_v(phi,0,1); add_v(-phi,0,-1); add_v(-phi,0,1);
        obj.faces = {{0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},{1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
                     {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},{4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}};
        for (int s = 0; s < subdivisions; ++s) {
            std::map<std::pair<int,int>,int> cache;
            std::vector<Face> nf;
            auto mid = [&](int v1, int v2) -> int {
                auto k = std::make_pair(std::min(v1,v2),std::max(v1,v2));
                auto it = cache.find(k); if (it != cache.end()) return it->second;
                Vec3 m = (obj.vertices[v1] + obj.vertices[v2]) * 0.5;
                int idx = add_v(m.x, m.y, m.z); cache[k] = idx; return idx;
            };
            for (auto& fc : obj.faces) {
                int a = mid(fc.v0,fc.v1), b = mid(fc.v1,fc.v2), c = mid(fc.v2,fc.v0);
                nf.push_back({fc.v0,a,c}); nf.push_back({fc.v1,b,a}); nf.push_back({fc.v2,c,b}); nf.push_back({a,b,c});
            }
            obj.faces = std::move(nf);
        }
        return obj;
    }

    inline std::pair<Vec3,Vec3> bounding_box(const GeometryObject& obj) {
        if (obj.vertices.empty()) return {{0,0,0},{0,0,0}};
        Vec3 mn = obj.vertices[0], mx = obj.vertices[0];
        for (auto& v : obj.vertices) {
            mn.x = std::min(mn.x,v.x); mn.y = std::min(mn.y,v.y); mn.z = std::min(mn.z,v.z);
            mx.x = std::max(mx.x,v.x); mx.y = std::max(mx.y,v.y); mx.z = std::max(mx.z,v.z);
        }
        return {mn, mx};
    }

    inline Partition auto_partitions(const std::vector<GeometryObject>& objects, double step = 0.05) {
        Partition p;
        if (objects.empty()) return p;
        p.include = true;
        Vec3 gmin{1e30,1e30,1e30}, gmax{-1e30,-1e30,-1e30};
        for (auto& obj : objects) {
            for (auto& v : obj.vertices) {
                Vec3 gv{v.x+obj.location.x, v.y+obj.location.y, v.z+obj.location.z};
                gmin.x = std::min(gmin.x,gv.x); gmin.y = std::min(gmin.y,gv.y); gmin.z = std::min(gmin.z,gv.z);
                gmax.x = std::max(gmax.x,gv.x); gmax.y = std::max(gmax.y,gv.y); gmax.z = std::max(gmax.z,gv.z);
            }
        }
        p.x_start = gmin.x; p.x_end = gmax.x; p.x_step = step;
        p.y_start = gmin.y; p.y_end = gmax.y; p.y_step = step;
        p.z_start = gmin.z; p.z_end = gmax.z; p.z_step = step;
        return p;
    }
} // namespace geometry

} // namespace cellblender
