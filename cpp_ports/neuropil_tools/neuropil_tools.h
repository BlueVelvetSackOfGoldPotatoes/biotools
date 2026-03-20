// neuropil_tools.h
// C++17 port of the Python neuropil_tools package
// Original: https://github.com/mcellteam/neuropil_tools
// A neuropil analysis toolkit for MCell / Reconstruct / Blender workflows.
//
// GPL v2+ (see original Python sources)

#ifndef NEUROPIL_TOOLS_H
#define NEUROPIL_TOOLS_H

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
#include <numeric>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace neuropil {

// ============================================================================
// Forward declarations
// ============================================================================
struct Vec3;
struct Triangle;
struct Mesh;
struct NamedRegion;
struct NeuropilObject;
struct ContourPoint;
struct ContourTrace;
struct ContactPattern;
struct SpinePSDData;
struct ConnectivityRecord;

// ============================================================================
// Vec3 -- basic 3-component vector
// ============================================================================
struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(double s) { x *= s; y *= s; z *= s; return *this; }

    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    double length() const { return std::sqrt(x * x + y * y + z * z); }
    double length_sq() const { return x * x + y * y + z * z; }
    Vec3 normalized() const {
        double l = length();
        return (l > 1e-15) ? (*this / l) : Vec3{0, 0, 0};
    }

    bool operator==(const Vec3& o) const {
        return x == o.x && y == o.y && z == o.z;
    }
    bool operator!=(const Vec3& o) const { return !(*this == o); }

    double& operator[](int i) { return (&x)[i]; }
    double operator[](int i) const { return (&x)[i]; }
};

inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

inline std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}

// ============================================================================
// Triangle -- indices into a vertex array (0-based)
// ============================================================================
struct Triangle {
    uint32_t v0 = 0, v1 = 0, v2 = 0;
    Triangle() = default;
    Triangle(uint32_t a, uint32_t b, uint32_t c) : v0(a), v1(b), v2(c) {}
};

// ============================================================================
// Triangle area helper
// ============================================================================
inline double triangle_area(const Vec3& a, const Vec3& b, const Vec3& c) {
    return 0.5 * (b - a).cross(c - a).length();
}

// Signed volume contribution of a triangle (for closed meshes)
// Uses the divergence theorem: V = (1/6) * sum_i (v0 . (v1 x v2))
inline double signed_triangle_volume(const Vec3& a, const Vec3& b, const Vec3& c) {
    return a.dot(b.cross(c)) / 6.0;
}

// ============================================================================
// NamedRegion -- a named subset of faces on a mesh (ports MCell region concept)
// ============================================================================
struct NamedRegion {
    std::string name;
    std::set<uint32_t> face_indices; // indices into Mesh::triangles
};

// ============================================================================
// Mesh -- triangle mesh with optional named regions
// ============================================================================
struct Mesh {
    std::vector<Vec3> vertices;
    std::vector<Triangle> triangles;
    std::vector<NamedRegion> regions;

    // ---- Geometric queries ------------------------------------------------

    double surface_area() const {
        double a = 0.0;
        for (auto& t : triangles)
            a += triangle_area(vertices[t.v0], vertices[t.v1], vertices[t.v2]);
        return a;
    }

    double signed_volume() const {
        double v = 0.0;
        for (auto& t : triangles)
            v += signed_triangle_volume(vertices[t.v0], vertices[t.v1], vertices[t.v2]);
        return v;
    }

    double volume() const { return std::abs(signed_volume()); }

    double region_area(const std::string& region_name) const {
        for (auto& r : regions)
            if (r.name == region_name) return region_area(r);
        return 0.0;
    }

    double region_area(const NamedRegion& r) const {
        double a = 0.0;
        for (uint32_t fi : r.face_indices) {
            auto& t = triangles[fi];
            a += triangle_area(vertices[t.v0], vertices[t.v1], vertices[t.v2]);
        }
        return a;
    }

    Vec3 region_centroid(const std::string& region_name) const {
        for (auto& r : regions)
            if (r.name == region_name) return region_centroid(r);
        return {};
    }

    Vec3 region_centroid(const NamedRegion& r) const {
        std::set<uint32_t> verts;
        for (uint32_t fi : r.face_indices) {
            auto& t = triangles[fi];
            verts.insert(t.v0);
            verts.insert(t.v1);
            verts.insert(t.v2);
        }
        Vec3 sum{};
        for (uint32_t vi : verts) sum += vertices[vi];
        if (!verts.empty()) sum = sum / static_cast<double>(verts.size());
        return sum;
    }

    double region_signed_volume(const NamedRegion& r) const {
        double v = 0.0;
        for (uint32_t fi : r.face_indices) {
            auto& t = triangles[fi];
            v += signed_triangle_volume(vertices[t.v0], vertices[t.v1], vertices[t.v2]);
        }
        return v;
    }

    Mesh extract_region(const NamedRegion& r) const {
        Mesh sub;
        std::map<uint32_t, uint32_t> old_to_new;
        for (uint32_t fi : r.face_indices) {
            auto& t = triangles[fi];
            for (uint32_t vi : {t.v0, t.v1, t.v2}) {
                if (old_to_new.find(vi) == old_to_new.end()) {
                    uint32_t ni = static_cast<uint32_t>(sub.vertices.size());
                    old_to_new[vi] = ni;
                    sub.vertices.push_back(vertices[vi]);
                }
            }
            sub.triangles.emplace_back(
                old_to_new[t.v0], old_to_new[t.v1], old_to_new[t.v2]);
        }
        return sub;
    }

    Mesh extract_region(const std::string& name) const {
        for (auto& r : regions)
            if (r.name == name) return extract_region(r);
        return {};
    }

    std::pair<Vec3, Vec3> bounding_box() const {
        if (vertices.empty()) return {{}, {}};
        Vec3 mn = vertices[0], mx = vertices[0];
        for (auto& v : vertices) {
            mn.x = std::min(mn.x, v.x); mn.y = std::min(mn.y, v.y); mn.z = std::min(mn.z, v.z);
            mx.x = std::max(mx.x, v.x); mx.y = std::max(mx.y, v.y); mx.z = std::max(mx.z, v.z);
        }
        return {mn, mx};
    }

    Vec3 centroid() const {
        Vec3 c{};
        for (auto& v : vertices) c += v;
        if (!vertices.empty()) c = c / static_cast<double>(vertices.size());
        return c;
    }

    const NamedRegion* find_region(const std::string& name) const {
        for (auto& r : regions)
            if (r.name == name) return &r;
        return nullptr;
    }

    NamedRegion* find_region(const std::string& name) {
        for (auto& r : regions)
            if (r.name == name) return &r;
        return nullptr;
    }

    // ---- Connectivity / topology ------------------------------------------

    std::vector<std::set<uint32_t>> vertex_adjacency() const {
        std::vector<std::set<uint32_t>> adj(vertices.size());
        for (auto& t : triangles) {
            adj[t.v0].insert(t.v1); adj[t.v0].insert(t.v2);
            adj[t.v1].insert(t.v0); adj[t.v1].insert(t.v2);
            adj[t.v2].insert(t.v0); adj[t.v2].insert(t.v1);
        }
        return adj;
    }

    int count_components() const {
        if (vertices.empty()) return 0;
        auto adj = vertex_adjacency();
        std::vector<bool> visited(vertices.size(), false);
        int n_comp = 0;
        for (size_t seed = 0; seed < vertices.size(); ++seed) {
            if (visited[seed]) continue;
            ++n_comp;
            std::vector<uint32_t> stack;
            stack.push_back(static_cast<uint32_t>(seed));
            visited[seed] = true;
            while (!stack.empty()) {
                uint32_t cur = stack.back(); stack.pop_back();
                for (uint32_t nb : adj[cur]) {
                    if (!visited[nb]) {
                        visited[nb] = true;
                        stack.push_back(nb);
                    }
                }
            }
        }
        return n_comp;
    }

    std::vector<std::vector<uint32_t>> connected_components() const {
        std::vector<std::vector<uint32_t>> comps;
        if (vertices.empty()) return comps;
        auto adj = vertex_adjacency();
        std::vector<bool> visited(vertices.size(), false);
        for (size_t seed = 0; seed < vertices.size(); ++seed) {
            if (visited[seed]) continue;
            comps.emplace_back();
            auto& comp = comps.back();
            std::vector<uint32_t> stack;
            stack.push_back(static_cast<uint32_t>(seed));
            visited[seed] = true;
            while (!stack.empty()) {
                uint32_t cur = stack.back(); stack.pop_back();
                comp.push_back(cur);
                for (uint32_t nb : adj[cur]) {
                    if (!visited[nb]) {
                        visited[nb] = true;
                        stack.push_back(nb);
                    }
                }
            }
        }
        return comps;
    }

    // ---- Mesh cleaning ----------------------------------------------------

    void remove_degenerate_triangles(double eps = 1e-15) {
        std::vector<Triangle> clean;
        clean.reserve(triangles.size());
        for (auto& t : triangles) {
            double a = triangle_area(vertices[t.v0], vertices[t.v1], vertices[t.v2]);
            if (a > eps) clean.push_back(t);
        }
        triangles = std::move(clean);
    }

    void compact_vertices() {
        std::vector<bool> used(vertices.size(), false);
        for (auto& t : triangles) {
            used[t.v0] = true;
            used[t.v1] = true;
            used[t.v2] = true;
        }
        std::vector<uint32_t> remap(vertices.size(), UINT32_MAX);
        std::vector<Vec3> new_verts;
        for (size_t i = 0; i < vertices.size(); ++i) {
            if (used[i]) {
                remap[i] = static_cast<uint32_t>(new_verts.size());
                new_verts.push_back(vertices[i]);
            }
        }
        for (auto& t : triangles) {
            t.v0 = remap[t.v0];
            t.v1 = remap[t.v1];
            t.v2 = remap[t.v2];
        }
        vertices = std::move(new_verts);
    }

    void laplacian_smooth(int n_iter = 1, double factor = 0.5) {
        auto adj = vertex_adjacency();
        for (int it = 0; it < n_iter; ++it) {
            std::vector<Vec3> new_verts(vertices.size());
            for (size_t i = 0; i < vertices.size(); ++i) {
                if (adj[i].empty()) {
                    new_verts[i] = vertices[i];
                    continue;
                }
                Vec3 avg{};
                for (uint32_t nb : adj[i]) avg += vertices[nb];
                avg = avg / static_cast<double>(adj[i].size());
                new_verts[i] = vertices[i] * (1.0 - factor) + avg * factor;
            }
            vertices = std::move(new_verts);
        }
    }

    void scale(double sx, double sy, double sz) {
        for (auto& v : vertices) { v.x *= sx; v.y *= sy; v.z *= sz; }
    }
    void scale(double s) { scale(s, s, s); }

    void translate(const Vec3& t) {
        for (auto& v : vertices) v += t;
    }

    void merge(const Mesh& other) {
        uint32_t v_offset = static_cast<uint32_t>(vertices.size());
        uint32_t t_offset = static_cast<uint32_t>(triangles.size());
        vertices.insert(vertices.end(), other.vertices.begin(), other.vertices.end());
        for (auto& t : other.triangles)
            triangles.emplace_back(t.v0 + v_offset, t.v1 + v_offset, t.v2 + v_offset);
        for (auto& r : other.regions) {
            NamedRegion nr;
            nr.name = r.name;
            for (uint32_t fi : r.face_indices) nr.face_indices.insert(fi + t_offset);
            regions.push_back(std::move(nr));
        }
    }

    void flip_normals() {
        for (auto& t : triangles) std::swap(t.v1, t.v2);
    }

    std::vector<Vec3> face_normals() const {
        std::vector<Vec3> norms(triangles.size());
        for (size_t i = 0; i < triangles.size(); ++i) {
            auto& t = triangles[i];
            norms[i] = (vertices[t.v1] - vertices[t.v0])
                            .cross(vertices[t.v2] - vertices[t.v0])
                            .normalized();
        }
        return norms;
    }

    std::vector<Vec3> vertex_normals() const {
        std::vector<Vec3> vnorms(vertices.size());
        for (auto& t : triangles) {
            Vec3 n = (vertices[t.v1] - vertices[t.v0]).cross(vertices[t.v2] - vertices[t.v0]);
            vnorms[t.v0] += n;
            vnorms[t.v1] += n;
            vnorms[t.v2] += n;
        }
        for (auto& n : vnorms) n = n.normalized();
        return vnorms;
    }
};

// ============================================================================
// NeuropilObject -- named mesh object with processor/spine metadata
// ============================================================================
struct NeuropilObject {
    std::string name;
    Mesh mesh;

    bool smoothed = false;
    bool newton = false;
    bool multi_synaptic = false;
    bool multi_component = false;
    bool genus_issue = false;
    bool generated = false;

    std::vector<std::string> contact_pattern_matches;
};

// ============================================================================
// ContactPattern -- naming convention for synapse-related contact regions
// ============================================================================
struct ContactPattern {
    std::string name;
    std::string base_name_1_pattern;
    std::string base_name_2_pattern;
    std::string contact_name_pattern;

    std::string base_name_1_regex;
    std::string base_name_2_regex;
    std::string contact_name_regex;

    static std::string pattern_to_regex(const std::string& pat) {
        std::string r;
        for (char c : pat) {
            if (c == '#') r += "[0-9]";
            else if (c == '*') r += ".*";
            else if (c == '.' || c == '(' || c == ')' || c == '[' || c == ']'
                     || c == '{' || c == '}' || c == '+' || c == '?'
                     || c == '^' || c == '$' || c == '|' || c == '\\') {
                r += '\\'; r += c;
            }
            else r += c;
        }
        return r;
    }

    void compile() {
        base_name_1_regex = pattern_to_regex(base_name_1_pattern);
        base_name_2_regex = pattern_to_regex(base_name_2_pattern);
        contact_name_regex = pattern_to_regex(contact_name_pattern);
        name = base_name_1_pattern + contact_name_pattern + base_name_2_pattern;
    }

    bool matches_region(const std::string& region_name) const {
        std::string full_regex = base_name_1_regex + contact_name_regex + base_name_2_regex;
        try {
            std::regex re(full_regex);
            return std::regex_match(region_name, re);
        } catch (...) {
            return false;
        }
    }

    bool matches_base_name(const std::string& obj_name) const {
        try {
            return std::regex_match(obj_name, std::regex(base_name_1_regex))
                || std::regex_match(obj_name, std::regex(base_name_2_regex));
        } catch (...) {
            return false;
        }
    }
};

// ============================================================================
// SpinePSDData -- morphometric properties of a PSD / spine / bouton
// ============================================================================
enum class ContactType { PLAIN, PROTRUSION, VARICOSITY };
enum class Ensheathment { DISTANT, PARTIAL, FULL };

struct SpinePSDData {
    std::string name;
    bool char_postsynaptic = true;
    ContactType contact_type = ContactType::PLAIN;
    Ensheathment ensheathment = Ensheathment::DISTANT;

    std::string head_name;
    std::string spine_name;
    std::string neck_name;

    double volume_head = 0.0;
    double volume_spine = 0.0;
    double volume_neck = 0.0;

    double area_head = 0.0;
    double area_spine = 0.0;
    double area_neck = 0.0;
    double area_neck_cross_section_abt = 0.0;
    double area_neck_cross_section_lbt = 0.0;
    double area_psd_az = 0.0;

    double diameter_neck_max = 0.0;
    double diameter_neck_min = 0.0;
    double diameter_neck_lbt = 0.0;
    double length_neck = 0.0;
    double length_neck_lbt = 0.0;

    double diameter_head_max = 0.0;
    double diameter_head_min = 0.0;
    double diameter_head_lbt = 0.0;
    double length_head = 0.0;
    double length_head_lbt = 0.0;

    Vec3 psd_az_location;
    Vec3 neck_top_location;
    Vec3 neck_base_location;

    bool char_mito = false;
    bool exclude = false;

    void init_from_mesh(const Mesh& mesh, const std::string& region_name) {
        name = region_name;
        contact_type = ContactType::PLAIN;
        char_postsynaptic = true;
        auto* reg = mesh.find_region(region_name);
        if (reg) {
            area_psd_az = mesh.region_area(*reg);
            psd_az_location = mesh.region_centroid(*reg);
        }
    }

    void compute_volumes(const Mesh& mesh, const std::string& base_sy_name) {
        std::string head_reg = base_sy_name + "_sph";
        auto* rh = mesh.find_region(head_reg);
        if (rh) {
            head_name = head_reg;
            Mesh sub = mesh.extract_region(*rh);
            volume_head = sub.volume();
            area_head = sub.surface_area();
        }

        std::string spine_reg = base_sy_name + "_sp";
        auto* rs = mesh.find_region(spine_reg);
        if (rs) {
            spine_name = spine_reg;
            Mesh sub = mesh.extract_region(*rs);
            volume_spine = sub.volume();
            area_spine = sub.surface_area();
        }

        std::string neck_reg = base_sy_name + "_spn";
        auto* rn = mesh.find_region(neck_reg);
        if (rn) {
            neck_name = neck_reg;
            Mesh sub = mesh.extract_region(*rn);
            volume_neck = sub.volume();
            area_neck = sub.surface_area();
        }

        if (volume_spine > 0.0 && volume_head > 0.0 && volume_neck == 0.0)
            volume_neck = volume_spine - volume_head;
    }

    void compute_areas(const Mesh& mesh) {
        auto* reg = mesh.find_region(name);
        if (reg) area_psd_az = mesh.region_area(*reg);
        if (!head_name.empty()) {
            auto* rh = mesh.find_region(head_name);
            if (rh) area_head = mesh.region_area(*rh);
        }
        if (!spine_name.empty()) {
            auto* rs = mesh.find_region(spine_name);
            if (rs) area_spine = mesh.region_area(*rs);
        }
        if (!neck_name.empty()) {
            auto* rn = mesh.find_region(neck_name);
            if (rn) area_neck = mesh.region_area(*rn);
        }
    }
};

// ============================================================================
// ConnectivityRecord -- pre-to-post / post-to-pre connection data
// ============================================================================
struct ConnectivityRecord {
    std::string axon_name;
    std::string synapse_region;
    std::string dendrite_name;
};

// ============================================================================
// Connectivity analysis
// ============================================================================
class ConnectivityAnalyzer {
public:
    std::map<std::string, NeuropilObject> objects;

    std::vector<std::string> find_dendrites(const std::string& pattern = "d[0-9]{3}") const {
        std::regex re(pattern);
        std::vector<std::string> result;
        for (auto& [name, obj] : objects)
            if (std::regex_match(name, re)) result.push_back(name);
        std::sort(result.begin(), result.end());
        return result;
    }

    std::vector<std::string> find_axons(const std::string& pattern = "a[0-9]{3}") const {
        std::regex re(pattern);
        std::vector<std::string> result;
        for (auto& [name, obj] : objects)
            if (std::regex_match(name, re)) result.push_back(name);
        std::sort(result.begin(), result.end());
        return result;
    }

    std::vector<std::string> find_synapse_regions(const NeuropilObject& obj) const {
        std::vector<std::string> sy_list;
        for (auto& r : obj.mesh.regions)
            if (r.name.find("sy") != std::string::npos)
                sy_list.push_back(r.name);
        return sy_list;
    }

    std::string get_dendrite_from_synapse(const std::string& sy_name,
                                          const std::string& dend_pattern = "d[0-9]{3}") const {
        std::regex re(dend_pattern);
        std::smatch m;
        if (std::regex_search(sy_name, m, re)) return m[0].str();
        return {};
    }

    std::string get_axon_for_synapse(const std::string& sy_name,
                                     const std::string& axon_pattern = "a[0-9]{3}") const {
        std::regex re(axon_pattern);
        for (auto& [name, obj] : objects) {
            if (!std::regex_match(name, re)) continue;
            for (auto& r : obj.mesh.regions)
                if (r.name == sy_name) return name;
        }
        return {};
    }

    std::vector<ConnectivityRecord> output_pre_to_post() const {
        std::vector<ConnectivityRecord> records;
        auto axons = find_axons();
        for (auto& axon_name : axons) {
            auto it = objects.find(axon_name);
            if (it == objects.end()) continue;
            auto sy_list = find_synapse_regions(it->second);
            for (auto& sy : sy_list) {
                ConnectivityRecord cr;
                cr.axon_name = axon_name;
                cr.synapse_region = sy;
                cr.dendrite_name = get_dendrite_from_synapse(sy);
                records.push_back(cr);
            }
        }
        return records;
    }

    std::vector<ConnectivityRecord> output_post_to_pre() const {
        std::vector<ConnectivityRecord> records;
        auto dends = find_dendrites();
        for (auto& dend_name : dends) {
            auto it = objects.find(dend_name);
            if (it == objects.end()) continue;
            auto sy_list = find_synapse_regions(it->second);
            for (auto& sy : sy_list) {
                ConnectivityRecord cr;
                cr.dendrite_name = dend_name;
                cr.synapse_region = sy;
                cr.axon_name = get_axon_for_synapse(sy);
                records.push_back(cr);
            }
        }
        return records;
    }

    void write_pre_to_post(const std::string& filename) const {
        auto records = output_pre_to_post();
        std::ofstream f(filename);
        for (auto& r : records)
            f << r.axon_name << " " << r.synapse_region << " " << r.dendrite_name << "\n";
    }

    void write_post_to_pre(const std::string& filename) const {
        auto records = output_post_to_pre();
        std::ofstream f(filename);
        for (auto& r : records)
            f << r.dendrite_name << " " << r.synapse_region << " " << r.axon_name << "\n";
    }
};

// ============================================================================
// DiameterCalculator -- computes diameter metrics for mesh regions
// ============================================================================
class DiameterCalculator {
public:
    static double max_diameter(const Mesh& mesh, const NamedRegion& region) {
        std::set<uint32_t> vert_set;
        for (uint32_t fi : region.face_indices) {
            auto& t = mesh.triangles[fi];
            vert_set.insert(t.v0);
            vert_set.insert(t.v1);
            vert_set.insert(t.v2);
        }
        std::vector<uint32_t> verts(vert_set.begin(), vert_set.end());

        double max_d = 0.0;
        for (size_t i = 0; i < verts.size(); ++i)
            for (size_t j = i + 1; j < verts.size(); ++j) {
                double d = (mesh.vertices[verts[i]] - mesh.vertices[verts[j]]).length();
                max_d = std::max(max_d, d);
            }
        return max_d;
    }

    static double min_diameter(const Mesh& mesh, const NamedRegion& region) {
        std::set<uint32_t> vert_set;
        for (uint32_t fi : region.face_indices) {
            auto& t = mesh.triangles[fi];
            vert_set.insert(t.v0);
            vert_set.insert(t.v1);
            vert_set.insert(t.v2);
        }
        if (vert_set.size() < 2) return 0.0;

        double extents[3] = {0, 0, 0};
        for (int ax = 0; ax < 3; ++ax) {
            double mn = 1e30, mx = -1e30;
            for (uint32_t vi : vert_set) {
                double val = mesh.vertices[vi][ax];
                mn = std::min(mn, val);
                mx = std::max(mx, val);
            }
            extents[ax] = mx - mn;
        }
        return *std::min_element(extents, extents + 3);
    }

    struct DiameterResult {
        double max_diameter = 0.0;
        double min_diameter = 0.0;
    };

    static DiameterResult compute(const Mesh& mesh, const NamedRegion& region) {
        DiameterResult r;
        r.max_diameter = max_diameter(mesh, region);
        r.min_diameter = min_diameter(mesh, region);
        return r;
    }

    static DiameterResult compute(const Mesh& mesh, const std::string& region_name) {
        auto* reg = mesh.find_region(region_name);
        if (!reg) return {};
        return compute(mesh, *reg);
    }
};

// ============================================================================
// Spine Head Analysis -- morphometric analysis of spines, heads, necks
// ============================================================================
class SpineHeadAnalyzer {
public:
    std::string varicosity_label = "axb";
    std::string protrusion_label = "sp";
    std::string head_label = "sph";
    std::string neck_label = "spn";

    std::string spine_namestruct;
    std::string psd_namestruct;

    std::map<std::string, SpinePSDData> psd_data;

    void init_all_psds(const Mesh& mesh, const std::vector<ContactPattern>& patterns) {
        for (auto& region : mesh.regions) {
            for (auto& pat : patterns) {
                if (pat.matches_region(region.name)) {
                    SpinePSDData d;
                    d.init_from_mesh(mesh, region.name);
                    psd_data[region.name] = std::move(d);
                    break;
                }
            }
        }
    }

    void compute_psd_volumes(const Mesh& mesh, const std::string& psd_name) {
        auto it = psd_data.find(psd_name);
        if (it == psd_data.end()) return;
        it->second.compute_volumes(mesh, psd_name);
    }

    void compute_psd_diameters(const Mesh& mesh, const std::string& psd_name) {
        auto it = psd_data.find(psd_name);
        if (it == psd_data.end()) return;
        auto& d = it->second;
        if (!d.neck_name.empty()) {
            auto dr = DiameterCalculator::compute(mesh, d.neck_name);
            d.diameter_neck_max = dr.max_diameter;
            d.diameter_neck_min = dr.min_diameter;
        }
        if (!d.head_name.empty()) {
            auto dr = DiameterCalculator::compute(mesh, d.head_name);
            d.diameter_head_max = dr.max_diameter;
            d.diameter_head_min = dr.min_diameter;
        }
    }

    void recompute_all(const Mesh& mesh) {
        for (auto& [name, d] : psd_data) {
            d.compute_areas(mesh);
            d.compute_volumes(mesh, name);
            if (!d.neck_name.empty()) {
                auto dr_n = DiameterCalculator::compute(mesh, d.neck_name);
                d.diameter_neck_max = dr_n.max_diameter;
                d.diameter_neck_min = dr_n.min_diameter;
            }
            if (!d.head_name.empty()) {
                auto dr_h = DiameterCalculator::compute(mesh, d.head_name);
                d.diameter_head_max = dr_h.max_diameter;
                d.diameter_head_min = dr_h.min_diameter;
            }
        }
    }

    void write_report(const std::string& filename) const {
        std::ofstream f(filename);
        f << "# sy pre_post head_vol spine_vol neck_vol computed_area region_area "
             "dia_head_max dia_head_min dia_neck_max dia_neck_min\n";
        for (auto& [name, d] : psd_data) {
            std::string pp = d.char_postsynaptic ? "post" : "pre";
            f << name << " " << pp
              << " " << d.volume_head
              << " " << d.volume_spine
              << " " << d.volume_neck
              << " " << d.area_head
              << " " << d.area_psd_az
              << " " << d.diameter_head_max
              << " " << d.diameter_head_min
              << " " << d.diameter_neck_max
              << " " << d.diameter_neck_min
              << "\n";
        }
    }
};

// ============================================================================
// ContourPoint / ContourTrace -- for Reconstruct .ser contour data
// ============================================================================
struct ContourPoint {
    double x = 0.0, y = 0.0;
};

struct ContourTrace {
    std::string name;
    int section = 0;
    std::vector<ContourPoint> points;
};

// ============================================================================
// ReconstructSeries -- parsing Reconstruct .ser / trace files
// ============================================================================
class ReconstructSeries {
public:
    std::string filepath;
    std::string series_prefix;
    int min_section = 0;
    int max_section = 0;
    double section_thickness = 0.05;

    std::vector<std::string> contour_names;
    std::vector<std::string> include_list;

    bool read_series(const std::string& ser_filepath) {
        filepath = ser_filepath;
        namespace fs = std::filesystem;
        fs::path ser_path(ser_filepath);
        series_prefix = ser_path.stem().string();
        fs::path ser_dir = ser_path.parent_path();

        std::ifstream sf(ser_filepath);
        if (!sf.is_open()) {
            std::cerr << "Cannot open .ser file: " << ser_filepath << "\n";
            return false;
        }
        std::string ser_data((std::istreambuf_iterator<char>(sf)),
                              std::istreambuf_iterator<char>());
        sf.close();

        {
            std::regex re("defaultThickness=\"([^\"]*)\"");
            std::smatch m;
            if (std::regex_search(ser_data, m, re))
                section_thickness = std::stod(m[1].str());
        }

        std::vector<int> trace_nums;
        std::string prefix_with_dot = series_prefix + ".";
        for (auto& entry : fs::directory_iterator(ser_dir)) {
            if (!entry.is_regular_file()) continue;
            std::string fname = entry.path().filename().string();
            if (fname.size() > prefix_with_dot.size()
                && fname.substr(0, prefix_with_dot.size()) == prefix_with_dot) {
                std::string suffix = fname.substr(prefix_with_dot.size());
                bool all_digit = !suffix.empty()
                    && std::all_of(suffix.begin(), suffix.end(), ::isdigit);
                if (all_digit)
                    trace_nums.push_back(std::stoi(suffix));
            }
        }
        std::sort(trace_nums.begin(), trace_nums.end());

        if (trace_nums.empty()) {
            std::cerr << "No trace files found for series: " << series_prefix << "\n";
            return false;
        }

        find_longest_contiguous_run(trace_nums);

        std::regex contour_re("Contour name=\"([^\"]*)\"");
        std::set<std::string> all_names_set;

        for (int i = min_section; i <= max_section; ++i) {
            fs::path trace_path = ser_dir / (series_prefix + "." + std::to_string(i));
            std::ifstream tf(trace_path);
            if (!tf.is_open()) continue;
            std::string trace_data((std::istreambuf_iterator<char>(tf)),
                                    std::istreambuf_iterator<char>());
            tf.close();

            auto begin = std::sregex_iterator(trace_data.begin(), trace_data.end(), contour_re);
            auto end = std::sregex_iterator();
            for (auto it = begin; it != end; ++it)
                all_names_set.insert((*it)[1].str());
        }

        contour_names.assign(all_names_set.begin(), all_names_set.end());
        std::sort(contour_names.begin(), contour_names.end());
        return true;
    }

    void include_contour(const std::string& name) {
        if (std::find(include_list.begin(), include_list.end(), name) == include_list.end())
            include_list.push_back(name);
    }

    void include_filtered(const std::string& filter_pattern) {
        try {
            std::regex re(filter_pattern);
            for (auto& name : contour_names)
                if (std::regex_search(name, re))
                    include_contour(name);
        } catch (...) {}
    }

    void remove_contour(const std::string& name) {
        include_list.erase(
            std::remove(include_list.begin(), include_list.end(), name),
            include_list.end());
    }

    void clear_include_list() { include_list.clear(); }

    void print_summary(std::ostream& os = std::cout) const {
        os << "Series: " << series_prefix << "\n"
           << "Section range: " << min_section << " - " << max_section << "\n"
           << "Section thickness: " << section_thickness << "\n"
           << "Contour names (" << contour_names.size() << "):\n";
        for (auto& n : contour_names) os << "  " << n << "\n";
    }

private:
    void find_longest_contiguous_run(const std::vector<int>& nums) {
        if (nums.empty()) return;
        int best_start = 0, best_len = 1;
        int cur_start = 0, cur_len = 1;
        for (size_t i = 1; i < nums.size(); ++i) {
            if (nums[i] == nums[i - 1] + 1) {
                ++cur_len;
            } else {
                if (cur_len > best_len) {
                    best_start = cur_start;
                    best_len = cur_len;
                }
                cur_start = static_cast<int>(i);
                cur_len = 1;
            }
        }
        if (cur_len > best_len) {
            best_start = cur_start;
            best_len = cur_len;
        }
        min_section = nums[best_start];
        max_section = nums[best_start + best_len - 1];
    }
};

// ============================================================================
// OBJ File I/O
// ============================================================================
namespace obj_io {

inline Mesh read_obj(const std::string& filename) {
    Mesh mesh;
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Cannot open OBJ file: " << filename << "\n";
        return mesh;
    }

    std::string line;
    while (std::getline(f, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "v") {
            Vec3 v;
            iss >> v.x >> v.y >> v.z;
            mesh.vertices.push_back(v);
        } else if (token == "f") {
            std::vector<uint32_t> face_verts;
            std::string vert_str;
            while (iss >> vert_str) {
                size_t slash = vert_str.find('/');
                std::string vi_str = (slash != std::string::npos)
                                         ? vert_str.substr(0, slash)
                                         : vert_str;
                int vi = std::stoi(vi_str);
                if (vi < 0) vi = static_cast<int>(mesh.vertices.size()) + vi + 1;
                face_verts.push_back(static_cast<uint32_t>(vi - 1));
            }
            for (size_t i = 2; i < face_verts.size(); ++i)
                mesh.triangles.emplace_back(face_verts[0], face_verts[i - 1], face_verts[i]);
        }
    }
    return mesh;
}

inline std::vector<Mesh> read_multiple_objs(const std::string& directory,
                                             double scale_val = 1.0) {
    namespace fs = std::filesystem;
    std::vector<Mesh> meshes;
    std::vector<fs::path> obj_files;

    for (auto& entry : fs::directory_iterator(directory))
        if (entry.is_regular_file() && entry.path().extension() == ".obj")
            obj_files.push_back(entry.path());
    std::sort(obj_files.begin(), obj_files.end());

    for (auto& p : obj_files) {
        Mesh m = read_obj(p.string());
        if (scale_val != 1.0) m.scale(scale_val);
        meshes.push_back(std::move(m));
    }
    return meshes;
}

inline void write_obj(const std::string& filename, const Mesh& mesh) {
    std::ofstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Cannot open OBJ file for writing: " << filename << "\n";
        return;
    }

    f << "# OBJ file generated by neuropil_tools C++ port\n";
    f << "# Vertices: " << mesh.vertices.size()
      << "  Faces: " << mesh.triangles.size() << "\n\n";

    f << std::fixed << std::setprecision(8);
    for (auto& v : mesh.vertices)
        f << "v " << v.x << " " << v.y << " " << v.z << "\n";
    f << "\n";

    for (auto& t : mesh.triangles)
        f << "f " << (t.v0 + 1) << " " << (t.v1 + 1) << " " << (t.v2 + 1) << "\n";
}

inline void write_region_obj(const std::string& filename,
                              const Mesh& mesh,
                              const NamedRegion& region) {
    Mesh sub = mesh.extract_region(region);
    write_obj(filename, sub);
}

} // namespace obj_io

// ============================================================================
// MDL File I/O -- MCell MDL geometry format
// ============================================================================
namespace mdl_io {

inline bool insert_mdl_region(const std::string& mdl_geom_file,
                               const std::string& mdl_region_file,
                               std::ostream& out) {
    std::ifstream gf(mdl_geom_file);
    if (!gf.is_open()) {
        std::cerr << "Cannot open MDL geometry file: " << mdl_geom_file << "\n";
        return false;
    }
    std::string geom_data((std::istreambuf_iterator<char>(gf)),
                           std::istreambuf_iterator<char>());
    gf.close();

    size_t last_brace = geom_data.rfind('}');
    if (last_brace == std::string::npos) {
        std::cerr << "No closing brace found in MDL geometry file\n";
        return false;
    }

    out.write(geom_data.data(), static_cast<std::streamsize>(last_brace));

    std::ifstream rf(mdl_region_file);
    if (!rf.is_open()) {
        std::cerr << "Cannot open MDL region file: " << mdl_region_file << "\n";
        return false;
    }
    out << rf.rdbuf();
    rf.close();
    out << "}\n";
    return true;
}

inline bool insert_mdl_region(const std::string& mdl_geom_file,
                               const std::string& mdl_region_file,
                               const std::string& output_file) {
    std::ofstream of(output_file);
    if (!of.is_open()) {
        std::cerr << "Cannot open output file: " << output_file << "\n";
        return false;
    }
    return insert_mdl_region(mdl_geom_file, mdl_region_file, of);
}

inline Mesh read_mdl_mesh(const std::string& filename) {
    Mesh mesh;
    std::ifstream f(filename);
    if (!f.is_open()) return mesh;

    std::string line;
    bool in_vertex_list = false;
    bool in_element_list = false;

    while (std::getline(f, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start);

        if (line.find("VERTEX_LIST") != std::string::npos) {
            in_vertex_list = true; in_element_list = false; continue;
        }
        if (line.find("ELEMENT_CONNECTIONS") != std::string::npos) {
            in_vertex_list = false; in_element_list = true; continue;
        }
        if (line[0] == '}') {
            in_vertex_list = false; in_element_list = false; continue;
        }

        if (in_vertex_list && line[0] == '[') {
            std::replace(line.begin(), line.end(), '[', ' ');
            std::replace(line.begin(), line.end(), ']', ' ');
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream iss(line);
            Vec3 v;
            iss >> v.x >> v.y >> v.z;
            mesh.vertices.push_back(v);
        }
        if (in_element_list && line[0] == '[') {
            std::replace(line.begin(), line.end(), '[', ' ');
            std::replace(line.begin(), line.end(), ']', ' ');
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream iss(line);
            uint32_t a, b, c;
            iss >> a >> b >> c;
            mesh.triangles.emplace_back(a, b, c);
        }
    }
    return mesh;
}

} // namespace mdl_io

// ============================================================================
// OFF I/O (Object File Format)
// ============================================================================
namespace off_io {

inline Mesh read_off(const std::string& filename) {
    Mesh mesh;
    std::ifstream ifs(filename);
    if (!ifs.is_open()) return mesh;

    std::string line;
    // Read header "OFF"
    std::getline(ifs, line);
    while (!line.empty() && line[0] == '#') std::getline(ifs, line); // skip comments
    if (line.substr(0, 3) != "OFF" && line.substr(0, 4) != "COFF") {
        // Try if counts are on the header line
    }
    // Read counts: nVertices nFaces nEdges
    int nv = 0, nf = 0, ne = 0;
    if (line.size() > 3 && std::isdigit(line[3])) {
        // Counts on same line as OFF
        std::istringstream iss(line.substr(3));
        iss >> nv >> nf >> ne;
    } else {
        // Counts on next line
        while (std::getline(ifs, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            iss >> nv >> nf >> ne;
            break;
        }
    }

    // Read vertices
    mesh.vertices.reserve(nv);
    for (int i = 0; i < nv; i++) {
        std::getline(ifs, line);
        while (line.empty() || line[0] == '#') std::getline(ifs, line);
        std::istringstream iss(line);
        double x, y, z;
        iss >> x >> y >> z;
        mesh.vertices.emplace_back(x, y, z);
    }

    // Read faces (triangulate if needed)
    mesh.triangles.reserve(nf);
    for (int i = 0; i < nf; i++) {
        std::getline(ifs, line);
        while (line.empty() || line[0] == '#') std::getline(ifs, line);
        std::istringstream iss(line);
        int count;
        iss >> count;
        std::vector<int> verts(count);
        for (int j = 0; j < count; j++) iss >> verts[j];
        // Fan triangulation for n-gons
        for (int j = 1; j + 1 < count; j++) {
            mesh.triangles.emplace_back(verts[0], verts[j], verts[j + 1]);
        }
    }
    return mesh;
}

inline void write_off(const std::string& filename, const Mesh& mesh) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) return;
    ofs << "OFF\n";
    ofs << mesh.vertices.size() << " " << mesh.triangles.size() << " 0\n";
    for (auto& v : mesh.vertices) {
        ofs << v.x << " " << v.y << " " << v.z << "\n";
    }
    for (auto& t : mesh.triangles) {
        ofs << "3 " << t.v0 << " " << t.v1 << " " << t.v2 << "\n";
    }
}

} // namespace off_io

// ============================================================================
// STL I/O (ASCII and Binary)
// ============================================================================
namespace stl_io {

inline Mesh read_stl(const std::string& filename) {
    Mesh mesh;
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) return mesh;

    // Detect ASCII vs binary: ASCII starts with "solid"
    char header[6] = {};
    ifs.read(header, 5);
    header[5] = '\0';
    ifs.seekg(0);

    if (std::string(header) == "solid") {
        // Try ASCII parsing
        std::string line;
        std::getline(ifs, line); // "solid name"
        std::map<std::tuple<float,float,float>, int> vert_map;

        auto get_or_add = [&](double x, double y, double z) -> int {
            auto key = std::make_tuple((float)x, (float)y, (float)z);
            auto it = vert_map.find(key);
            if (it != vert_map.end()) return it->second;
            int idx = (int)mesh.vertices.size();
            mesh.vertices.emplace_back(x, y, z);
            vert_map[key] = idx;
            return idx;
        };

        int face_verts[3];
        int vi = 0;
        while (std::getline(ifs, line)) {
            std::istringstream iss(line);
            std::string tok;
            iss >> tok;
            if (tok == "vertex") {
                double x, y, z;
                iss >> x >> y >> z;
                face_verts[vi++] = get_or_add(x, y, z);
                if (vi == 3) {
                    mesh.triangles.emplace_back(face_verts[0], face_verts[1], face_verts[2]);
                    vi = 0;
                }
            } else if (tok == "endsolid") {
                break;
            }
        }

        // If no triangles parsed, might be binary with "solid" in header
        if (mesh.triangles.empty()) {
            ifs.clear();
            ifs.seekg(0);
            goto read_binary;
        }
        return mesh;
    }

read_binary:
    {
        // Binary STL: 80-byte header + 4-byte triangle count + triangles
        char bin_header[80];
        ifs.read(bin_header, 80);
        uint32_t num_tri = 0;
        ifs.read(reinterpret_cast<char*>(&num_tri), 4);

        std::map<std::tuple<float,float,float>, int> vert_map;
        auto get_or_add = [&](float x, float y, float z) -> int {
            auto key = std::make_tuple(x, y, z);
            auto it = vert_map.find(key);
            if (it != vert_map.end()) return it->second;
            int idx = (int)mesh.vertices.size();
            mesh.vertices.emplace_back(x, y, z);
            vert_map[key] = idx;
            return idx;
        };

        for (uint32_t i = 0; i < num_tri; i++) {
            float normal[3], v1[3], v2[3], v3[3];
            uint16_t attr;
            ifs.read(reinterpret_cast<char*>(normal), 12);
            ifs.read(reinterpret_cast<char*>(v1), 12);
            ifs.read(reinterpret_cast<char*>(v2), 12);
            ifs.read(reinterpret_cast<char*>(v3), 12);
            ifs.read(reinterpret_cast<char*>(&attr), 2);
            int i0 = get_or_add(v1[0], v1[1], v1[2]);
            int i1 = get_or_add(v2[0], v2[1], v2[2]);
            int i2 = get_or_add(v3[0], v3[1], v3[2]);
            mesh.triangles.emplace_back(i0, i1, i2);
        }
    }
    return mesh;
}

inline void write_stl_ascii(const std::string& filename, const Mesh& mesh,
                             const std::string& solid_name = "mesh") {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) return;
    ofs << "solid " << solid_name << "\n";
    for (auto& t : mesh.triangles) {
        Vec3 a = mesh.vertices[t.v0];
        Vec3 b = mesh.vertices[t.v1];
        Vec3 c = mesh.vertices[t.v2];
        Vec3 n = (b - a).cross(c - a);
        double len = n.length();
        if (len > 1e-15) n = n * (1.0 / len);
        ofs << "  facet normal " << n.x << " " << n.y << " " << n.z << "\n";
        ofs << "    outer loop\n";
        ofs << "      vertex " << a.x << " " << a.y << " " << a.z << "\n";
        ofs << "      vertex " << b.x << " " << b.y << " " << b.z << "\n";
        ofs << "      vertex " << c.x << " " << c.y << " " << c.z << "\n";
        ofs << "    endloop\n";
        ofs << "  endfacet\n";
    }
    ofs << "endsolid " << solid_name << "\n";
}

inline void write_stl_binary(const std::string& filename, const Mesh& mesh) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) return;
    char header[80] = {};
    ofs.write(header, 80);
    uint32_t num_tri = (uint32_t)mesh.triangles.size();
    ofs.write(reinterpret_cast<const char*>(&num_tri), 4);
    for (auto& t : mesh.triangles) {
        Vec3 a = mesh.vertices[t.v0];
        Vec3 b = mesh.vertices[t.v1];
        Vec3 c = mesh.vertices[t.v2];
        Vec3 n = (b - a).cross(c - a);
        double len = n.length();
        if (len > 1e-15) n = n * (1.0 / len);
        float fn[3] = {(float)n.x, (float)n.y, (float)n.z};
        float fv[9] = {(float)a.x, (float)a.y, (float)a.z,
                        (float)b.x, (float)b.y, (float)b.z,
                        (float)c.x, (float)c.y, (float)c.z};
        ofs.write(reinterpret_cast<const char*>(fn), 12);
        ofs.write(reinterpret_cast<const char*>(fv), 36);
        uint16_t attr = 0;
        ofs.write(reinterpret_cast<const char*>(&attr), 2);
    }
}

} // namespace stl_io

// ============================================================================
// Processor Tool -- mesh generation and contact tagging pipeline
// ============================================================================
class ProcessorTool {
public:
    std::string filepath;
    std::string series_prefix;

    std::vector<std::string> contour_list;
    std::vector<std::string> include_list;
    std::vector<ContactPattern> contact_patterns;

    std::map<std::string, NeuropilObject> objects;

    void add_contact_pattern(const std::string& bn1, const std::string& contact,
                             const std::string& bn2) {
        ContactPattern cp;
        cp.base_name_1_pattern = bn1;
        cp.contact_name_pattern = contact;
        cp.base_name_2_pattern = bn2;
        cp.compile();
        contact_patterns.push_back(std::move(cp));
    }

    void remove_contact_pattern(size_t index) {
        if (index < contact_patterns.size())
            contact_patterns.erase(contact_patterns.begin() + static_cast<std::ptrdiff_t>(index));
    }

    void update_all_contact_matches() {
        for (auto& [name, obj] : objects) {
            obj.contact_pattern_matches.clear();
            for (auto& cp : contact_patterns)
                if (cp.matches_base_name(name))
                    obj.contact_pattern_matches.push_back(cp.name);
        }
    }

    void tag_contacts() {
        for (auto& cp : contact_patterns) {
            std::regex bn1_re(cp.base_name_1_regex);
            std::regex bn2_re(cp.base_name_2_regex);

            std::vector<std::string> bn1_objs, bn2_objs;
            for (auto& [name, obj] : objects) {
                if (std::regex_match(name, bn1_re)) bn1_objs.push_back(name);
                if (std::regex_match(name, bn2_re)) bn2_objs.push_back(name);
            }

            for (auto& b1 : bn1_objs) {
                for ([[maybe_unused]] auto& b2 : bn2_objs) {
                    auto& obj1 = objects[b1];
                    for (auto& r : obj1.mesh.regions) {
                        if (cp.matches_region(r.name)) {
                            std::cout << "Contact found: " << r.name
                                      << " on " << b1 << "\n";
                        }
                    }
                }
            }
        }
    }

    void load_obj(const std::string& name, const std::string& obj_file) {
        NeuropilObject nobj;
        nobj.name = name;
        nobj.mesh = obj_io::read_obj(obj_file);
        objects[name] = std::move(nobj);
    }

    void smooth_object(const std::string& name, int iterations = 5, double factor = 0.5) {
        auto it = objects.find(name);
        if (it != objects.end()) {
            it->second.mesh.laplacian_smooth(iterations, factor);
            it->second.smoothed = true;
        }
    }

    void merge_objects(const std::vector<std::string>& names, const std::string& merged_name) {
        NeuropilObject merged;
        merged.name = merged_name;
        for (auto& n : names) {
            auto it = objects.find(n);
            if (it != objects.end())
                merged.mesh.merge(it->second.mesh);
        }
        objects[merged_name] = std::move(merged);
    }

    NeuropilObject* get_object(const std::string& name) {
        auto it = objects.find(name);
        return (it != objects.end()) ? &it->second : nullptr;
    }
    const NeuropilObject* get_object(const std::string& name) const {
        auto it = objects.find(name);
        return (it != objects.end()) ? &it->second : nullptr;
    }
};

// ============================================================================
// Spatial analysis utilities
// ============================================================================
namespace spatial {

inline std::pair<Vec3, Vec3> bounding_box(const std::vector<Vec3>& points) {
    if (points.empty()) return {{}, {}};
    Vec3 mn = points[0], mx = points[0];
    for (auto& p : points) {
        mn.x = std::min(mn.x, p.x); mn.y = std::min(mn.y, p.y); mn.z = std::min(mn.z, p.z);
        mx.x = std::max(mx.x, p.x); mx.y = std::max(mx.y, p.y); mx.z = std::max(mx.z, p.z);
    }
    return {mn, mx};
}

inline double distance(const Vec3& a, const Vec3& b) {
    return (a - b).length();
}

inline Vec3 closest_point_on_triangle(const Vec3& p,
                                       const Vec3& a, const Vec3& b, const Vec3& c) {
    Vec3 ab = b - a, ac = c - a, ap = p - a;
    double d1 = ab.dot(ap), d2 = ac.dot(ap);
    if (d1 <= 0 && d2 <= 0) return a;

    Vec3 bp = p - b;
    double d3 = ab.dot(bp), d4 = ac.dot(bp);
    if (d3 >= 0 && d4 <= d3) return b;

    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        double denom_ab = d1 - d3;
        if (std::abs(denom_ab) < 1e-15) return a;
        double v = d1 / denom_ab;
        return a + ab * v;
    }

    Vec3 cp_v = p - c;
    double d5 = ab.dot(cp_v), d6 = ac.dot(cp_v);
    if (d6 >= 0 && d5 <= d6) return c;

    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        double denom_ac = d2 - d6;
        if (std::abs(denom_ac) < 1e-15) return a;
        double w = d2 / denom_ac;
        return a + ac * w;
    }

    double va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
        double denom_bc = (d4 - d3) + (d5 - d6);
        if (std::abs(denom_bc) < 1e-15) return b;
        double w = (d4 - d3) / denom_bc;
        return b + (c - b) * w;
    }

    double sum_abc = va + vb + vc;
    if (std::abs(sum_abc) < 1e-15) return a; // degenerate triangle fallback
    double denom = 1.0 / sum_abc;
    double v = vb * denom;
    double w = vc * denom;
    return a + ab * v + ac * w;
}

inline double point_to_mesh_distance(const Vec3& point, const Mesh& mesh) {
    double min_dist = 1e30;
    for (auto& t : mesh.triangles) {
        Vec3 cp = closest_point_on_triangle(
            point, mesh.vertices[t.v0], mesh.vertices[t.v1], mesh.vertices[t.v2]);
        double d = (point - cp).length();
        min_dist = std::min(min_dist, d);
    }
    return min_dist;
}

inline double hausdorff_distance_directed(const Mesh& mesh1, const Mesh& mesh2) {
    double max_dist = 0.0;
    for (auto& v : mesh1.vertices) {
        double d = point_to_mesh_distance(v, mesh2);
        max_dist = std::max(max_dist, d);
    }
    return max_dist;
}

inline double hausdorff_distance(const Mesh& mesh1, const Mesh& mesh2) {
    return std::max(hausdorff_distance_directed(mesh1, mesh2),
                    hausdorff_distance_directed(mesh2, mesh1));
}

} // namespace spatial

// ============================================================================
// Visualization data export -- CSV for external plotting
// ============================================================================
namespace export_data {

inline void write_spine_csv(const std::string& filename,
                             const std::map<std::string, SpinePSDData>& data) {
    std::ofstream f(filename);
    f << "name,contact_type,postsynaptic,volume_head,volume_spine,volume_neck,"
         "area_head,area_spine,area_neck,area_psd_az,"
         "dia_head_max,dia_head_min,dia_neck_max,dia_neck_min,"
         "length_neck,length_head,"
         "psd_x,psd_y,psd_z,exclude\n";

    for (auto& [name, d] : data) {
        std::string ct;
        switch (d.contact_type) {
            case ContactType::PLAIN: ct = "plain"; break;
            case ContactType::PROTRUSION: ct = "protrusion"; break;
            case ContactType::VARICOSITY: ct = "varicosity"; break;
        }
        f << name << "," << ct << "," << (d.char_postsynaptic ? "yes" : "no")
          << "," << d.volume_head << "," << d.volume_spine << "," << d.volume_neck
          << "," << d.area_head << "," << d.area_spine << "," << d.area_neck
          << "," << d.area_psd_az
          << "," << d.diameter_head_max << "," << d.diameter_head_min
          << "," << d.diameter_neck_max << "," << d.diameter_neck_min
          << "," << d.length_neck << "," << d.length_head
          << "," << d.psd_az_location.x << "," << d.psd_az_location.y
          << "," << d.psd_az_location.z
          << "," << (d.exclude ? "yes" : "no")
          << "\n";
    }
}

inline void write_connectivity_csv(const std::string& filename,
                                    const std::vector<ConnectivityRecord>& records) {
    std::ofstream f(filename);
    f << "axon,synapse,dendrite\n";
    for (auto& r : records)
        f << r.axon_name << "," << r.synapse_region << "," << r.dendrite_name << "\n";
}

inline void write_mesh_stats_csv(const std::string& filename,
                                  const std::map<std::string, NeuropilObject>& objects) {
    std::ofstream f(filename);
    f << "name,num_vertices,num_faces,surface_area,volume,num_components,num_regions\n";
    for (auto& [name, obj] : objects) {
        auto& m = obj.mesh;
        f << name << "," << m.vertices.size() << "," << m.triangles.size()
          << "," << m.surface_area() << "," << m.volume()
          << "," << m.count_components() << "," << m.regions.size() << "\n";
    }
}

} // namespace export_data

} // namespace neuropil

#endif // NEUROPIL_TOOLS_H
