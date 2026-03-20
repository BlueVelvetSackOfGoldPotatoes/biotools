// main.cpp
// Demo program for the neuropil_tools C++17 port
// Demonstrates all major subsystems: mesh I/O, geometry analysis, diameter
// computation, connectivity analysis, spine head analysis, MDL region
// insertion, Reconstruct series parsing, and data export.

#include "neuropil_tools.h"

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

using namespace neuropil;

// ---------------------------------------------------------------------------
// Helper: create a simple tetrahedron mesh for testing
// ---------------------------------------------------------------------------
static Mesh make_tetrahedron(double scale = 1.0) {
    Mesh m;
    m.vertices = {
        {1 * scale,  1 * scale,  1 * scale},
        {1 * scale, -1 * scale, -1 * scale},
        {-1 * scale, 1 * scale, -1 * scale},
        {-1 * scale,-1 * scale,  1 * scale}
    };
    m.triangles = {
        {0, 1, 2},
        {0, 3, 1},
        {0, 2, 3},
        {1, 3, 2}
    };
    return m;
}

// ---------------------------------------------------------------------------
// Helper: create a unit cube mesh (8 verts, 12 tris)
// ---------------------------------------------------------------------------
static Mesh make_cube(double side = 1.0) {
    Mesh m;
    double h = side / 2.0;
    m.vertices = {
        {-h, -h, -h}, { h, -h, -h}, { h,  h, -h}, {-h,  h, -h},
        {-h, -h,  h}, { h, -h,  h}, { h,  h,  h}, {-h,  h,  h}
    };
    // 6 faces, each split into 2 triangles
    m.triangles = {
        // front
        {0, 1, 2}, {0, 2, 3},
        // back
        {5, 4, 7}, {5, 7, 6},
        // left
        {4, 0, 3}, {4, 3, 7},
        // right
        {1, 5, 6}, {1, 6, 2},
        // bottom
        {4, 5, 1}, {4, 1, 0},
        // top
        {3, 2, 6}, {3, 6, 7}
    };

    // Add a named region covering two faces (the "front" face)
    NamedRegion front_region;
    front_region.name = "front_face";
    front_region.face_indices = {0, 1};
    m.regions.push_back(front_region);

    // Add a synapse-like region for connectivity demo
    NamedRegion sy_region;
    sy_region.name = "d001sy01a002";
    sy_region.face_indices = {2, 3};
    m.regions.push_back(sy_region);

    return m;
}

// ---------------------------------------------------------------------------
// Demo 1: Basic mesh geometry
// ---------------------------------------------------------------------------
static void demo_mesh_geometry() {
    std::cout << "=== Demo 1: Mesh Geometry ===\n\n";

    Mesh tet = make_tetrahedron();
    std::cout << "Tetrahedron:\n";
    std::cout << "  Vertices: " << tet.vertices.size() << "\n";
    std::cout << "  Faces: " << tet.triangles.size() << "\n";
    std::cout << "  Surface area: " << tet.surface_area() << "\n";
    std::cout << "  Volume: " << tet.volume() << "\n";
    std::cout << "  Centroid: " << tet.centroid() << "\n";

    auto [bb_min, bb_max] = tet.bounding_box();
    std::cout << "  Bounding box: " << bb_min << " to " << bb_max << "\n";
    std::cout << "  Components: " << tet.count_components() << "\n";

    Mesh cube = make_cube(2.0);
    std::cout << "\nUnit Cube (side=2):\n";
    std::cout << "  Vertices: " << cube.vertices.size() << "\n";
    std::cout << "  Faces: " << cube.triangles.size() << "\n";
    std::cout << "  Surface area: " << cube.surface_area() << " (expected 24)\n";
    std::cout << "  Volume: " << cube.volume() << " (expected 8)\n";
    std::cout << "  Centroid: " << cube.centroid() << "\n";
    std::cout << "  Components: " << cube.count_components() << "\n";

    // Region queries
    std::cout << "  Region 'front_face' area: " << cube.region_area("front_face")
              << " (expected 4)\n";
    std::cout << "  Region 'front_face' centroid: "
              << cube.region_centroid("front_face") << "\n";

    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Demo 2: Mesh operations (smoothing, merging, cleaning)
// ---------------------------------------------------------------------------
static void demo_mesh_operations() {
    std::cout << "=== Demo 2: Mesh Operations ===\n\n";

    Mesh cube = make_cube(1.0);
    double sa_before = cube.surface_area();
    std::cout << "Cube surface area before smoothing: " << sa_before << "\n";

    cube.laplacian_smooth(3, 0.5);
    double sa_after = cube.surface_area();
    std::cout << "Cube surface area after 3 iterations of Laplacian smoothing: "
              << sa_after << "\n";

    // Merge two meshes
    Mesh tet1 = make_tetrahedron(0.5);
    Mesh tet2 = make_tetrahedron(0.5);
    tet2.translate({3, 0, 0});

    Mesh merged;
    merged.merge(tet1);
    merged.merge(tet2);
    std::cout << "\nMerged two tetrahedra:\n";
    std::cout << "  Vertices: " << merged.vertices.size() << "\n";
    std::cout << "  Faces: " << merged.triangles.size() << "\n";
    std::cout << "  Components: " << merged.count_components() << "\n";

    // Face normals
    Mesh simple_tet = make_tetrahedron();
    auto fnorms = simple_tet.face_normals();
    std::cout << "\nTetrahedron face normals:\n";
    for (size_t i = 0; i < fnorms.size(); ++i) {
        std::cout << "  Face " << i << ": " << fnorms[i]
                  << " (length=" << fnorms[i].length() << ")\n";
    }

    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Demo 3: OBJ I/O
// ---------------------------------------------------------------------------
static void demo_obj_io() {
    std::cout << "=== Demo 3: OBJ File I/O ===\n\n";

    Mesh cube = make_cube(1.0);
    std::string obj_path = "/tmp/neuropil_demo_cube.obj";
    obj_io::write_obj(obj_path, cube);
    std::cout << "Wrote cube to: " << obj_path << "\n";

    Mesh loaded = obj_io::read_obj(obj_path);
    std::cout << "Loaded back:\n";
    std::cout << "  Vertices: " << loaded.vertices.size() << "\n";
    std::cout << "  Faces: " << loaded.triangles.size() << "\n";
    std::cout << "  Surface area: " << loaded.surface_area() << "\n";
    std::cout << "  Volume: " << loaded.volume() << "\n";

    // Clean up
    std::filesystem::remove(obj_path);
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Demo 4: MDL region insertion
// ---------------------------------------------------------------------------
static void demo_mdl_insertion() {
    std::cout << "=== Demo 4: MDL Region Insertion ===\n\n";

    // Create a mock MDL geometry file
    std::string geom_path = "/tmp/neuropil_demo_geom.mdl";
    {
        std::ofstream f(geom_path);
        f << "my_object POLYGON_LIST\n{\n"
          << "  VERTEX_LIST\n  {\n"
          << "    [ 0.0, 0.0, 0.0 ]\n"
          << "    [ 1.0, 0.0, 0.0 ]\n"
          << "    [ 0.0, 1.0, 0.0 ]\n"
          << "  }\n"
          << "  ELEMENT_CONNECTIONS\n  {\n"
          << "    [ 0, 1, 2 ]\n"
          << "  }\n"
          << "}\n";
    }

    // Create a mock MDL region file
    std::string region_path = "/tmp/neuropil_demo_region.mdl";
    {
        std::ofstream f(region_path);
        f << "  DEFINE_SURFACE_REGIONS\n  {\n"
          << "    my_region\n    {\n"
          << "      ELEMENT_LIST = [0]\n"
          << "    }\n"
          << "  }\n";
    }

    // Insert region into geometry
    std::ostringstream combined;
    bool ok = mdl_io::insert_mdl_region(geom_path, region_path, combined);
    std::cout << "MDL region insertion " << (ok ? "succeeded" : "failed") << "\n";
    std::cout << "Combined MDL output (" << combined.str().size() << " chars):\n";
    std::cout << combined.str() << "\n";

    // Also test MDL mesh reading
    Mesh mdl_mesh = mdl_io::read_mdl_mesh(geom_path);
    std::cout << "Read MDL mesh: " << mdl_mesh.vertices.size() << " vertices, "
              << mdl_mesh.triangles.size() << " faces\n";

    std::filesystem::remove(geom_path);
    std::filesystem::remove(region_path);
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Demo 5: Diameter calculation
// ---------------------------------------------------------------------------
static void demo_diameter() {
    std::cout << "=== Demo 5: Diameter Calculation ===\n\n";

    Mesh cube = make_cube(2.0);
    auto* front = cube.find_region("front_face");
    if (front) {
        auto dr = DiameterCalculator::compute(cube, *front);
        std::cout << "Cube 'front_face' region:\n";
        std::cout << "  Max diameter: " << dr.max_diameter << "\n";
        std::cout << "  Min diameter: " << dr.min_diameter << "\n";
    }

    // Compute for whole mesh using all faces as a region
    NamedRegion all_faces;
    all_faces.name = "all";
    for (uint32_t i = 0; i < static_cast<uint32_t>(cube.triangles.size()); ++i)
        all_faces.face_indices.insert(i);
    auto dr_all = DiameterCalculator::compute(cube, all_faces);
    std::cout << "\nCube (all faces):\n";
    std::cout << "  Max diameter: " << dr_all.max_diameter
              << " (expected ~3.46 = 2*sqrt(3))\n";
    std::cout << "  Min diameter: " << dr_all.min_diameter << " (expected 2)\n";

    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Demo 6: Connectivity analysis
// ---------------------------------------------------------------------------
static void demo_connectivity() {
    std::cout << "=== Demo 6: Connectivity Analysis ===\n\n";

    ConnectivityAnalyzer ca;

    // Create mock dendrite and axon objects with synapse regions
    {
        NeuropilObject d001;
        d001.name = "d001";
        d001.mesh = make_cube(1.0);
        NamedRegion sy;
        sy.name = "d001sy01a002";
        sy.face_indices = {0, 1};
        d001.mesh.regions.push_back(sy);
        ca.objects["d001"] = std::move(d001);
    }
    {
        NeuropilObject a002;
        a002.name = "a002";
        a002.mesh = make_cube(1.0);
        NamedRegion sy;
        sy.name = "d001sy01a002";
        sy.face_indices = {2, 3};
        a002.mesh.regions.push_back(sy);
        ca.objects["a002"] = std::move(a002);
    }

    auto dends = ca.find_dendrites();
    std::cout << "Dendrites found: " << dends.size() << "\n";
    for (auto& d : dends) std::cout << "  " << d << "\n";

    auto axons = ca.find_axons();
    std::cout << "Axons found: " << axons.size() << "\n";
    for (auto& a : axons) std::cout << "  " << a << "\n";

    auto pre2post = ca.output_pre_to_post();
    std::cout << "\nPre-to-post connectivity:\n";
    for (auto& r : pre2post)
        std::cout << "  " << r.axon_name << " -> " << r.synapse_region
                  << " -> " << r.dendrite_name << "\n";

    auto post2pre = ca.output_post_to_pre();
    std::cout << "\nPost-to-pre connectivity:\n";
    for (auto& r : post2pre)
        std::cout << "  " << r.dendrite_name << " <- " << r.synapse_region
                  << " <- " << r.axon_name << "\n";

    // Write to files
    std::string pre2post_file = "/tmp/neuropil_demo_pre2post.txt";
    std::string post2pre_file = "/tmp/neuropil_demo_post2pre.txt";
    ca.write_pre_to_post(pre2post_file);
    ca.write_post_to_pre(post2pre_file);
    std::cout << "\nWrote connectivity files to /tmp/\n";
    std::filesystem::remove(pre2post_file);
    std::filesystem::remove(post2pre_file);

    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Demo 7: Contact patterns
// ---------------------------------------------------------------------------
static void demo_contact_patterns() {
    std::cout << "=== Demo 7: Contact Patterns ===\n\n";

    ContactPattern cp;
    cp.base_name_1_pattern = "d###";
    cp.contact_name_pattern = "sy##";
    cp.base_name_2_pattern = "a###";
    cp.compile();

    std::cout << "Pattern: " << cp.name << "\n";
    std::cout << "  BN1 regex: " << cp.base_name_1_regex << "\n";
    std::cout << "  Contact regex: " << cp.contact_name_regex << "\n";
    std::cout << "  BN2 regex: " << cp.base_name_2_regex << "\n";

    std::cout << "\nTesting matches:\n";
    std::vector<std::string> test_names = {
        "d001sy01a002", "d123sy99a456", "d00sy01a002",
        "d001xx01a002", "axon01", "d001"
    };
    for (auto& tn : test_names) {
        std::cout << "  '" << tn << "' matches region: "
                  << (cp.matches_region(tn) ? "yes" : "no") << "\n";
    }

    std::cout << "\nTesting base name matches:\n";
    std::vector<std::string> base_names = {"d001", "a002", "d12", "a1234", "xyz"};
    for (auto& bn : base_names) {
        std::cout << "  '" << bn << "' matches base: "
                  << (cp.matches_base_name(bn) ? "yes" : "no") << "\n";
    }

    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Demo 8: Spine head analysis
// ---------------------------------------------------------------------------
static void demo_spine_analysis() {
    std::cout << "=== Demo 8: Spine Head Analysis ===\n\n";

    // Create a mesh with synapse and spine regions
    Mesh mesh = make_cube(1.0);

    NamedRegion sy_reg;
    sy_reg.name = "d001sy01a002";
    sy_reg.face_indices = {0, 1};
    mesh.regions.push_back(sy_reg);

    NamedRegion sph_reg;
    sph_reg.name = "d001sy01a002_sph";
    sph_reg.face_indices = {2, 3, 4, 5};
    mesh.regions.push_back(sph_reg);

    NamedRegion sp_reg;
    sp_reg.name = "d001sy01a002_sp";
    sp_reg.face_indices = {2, 3, 4, 5, 6, 7, 8, 9};
    mesh.regions.push_back(sp_reg);

    NamedRegion spn_reg;
    spn_reg.name = "d001sy01a002_spn";
    spn_reg.face_indices = {6, 7, 8, 9};
    mesh.regions.push_back(spn_reg);

    // Set up analyzer
    SpineHeadAnalyzer sha;
    ContactPattern cp;
    cp.base_name_1_pattern = "d###";
    cp.contact_name_pattern = "sy##";
    cp.base_name_2_pattern = "a###";
    cp.compile();

    sha.init_all_psds(mesh, {cp});
    std::cout << "Found " << sha.psd_data.size() << " PSD(s)\n";

    for (auto& [name, d] : sha.psd_data) {
        std::cout << "\nPSD: " << name << "\n";
        std::cout << "  PSD/AZ area: " << d.area_psd_az << "\n";
        std::cout << "  PSD location: " << d.psd_az_location << "\n";

        // Compute volumes
        sha.compute_psd_volumes(mesh, name);
        std::cout << "  Head volume: " << d.volume_head << "\n";
        std::cout << "  Spine volume: " << d.volume_spine << "\n";
        std::cout << "  Neck volume: " << d.volume_neck << "\n";

        // Compute diameters
        sha.compute_psd_diameters(mesh, name);
        std::cout << "  Head max diameter: " << d.diameter_head_max << "\n";
        std::cout << "  Neck max diameter: " << d.diameter_neck_max << "\n";
    }

    // Write report
    std::string report_path = "/tmp/neuropil_demo_spine_report.txt";
    sha.write_report(report_path);
    std::cout << "\nWrote spine report to: " << report_path << "\n";
    std::filesystem::remove(report_path);

    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Demo 9: Processor tool
// ---------------------------------------------------------------------------
static void demo_processor_tool() {
    std::cout << "=== Demo 9: Processor Tool ===\n\n";

    ProcessorTool pt;

    // Add contact pattern
    pt.add_contact_pattern("d###", "sy##", "a###");
    std::cout << "Contact patterns: " << pt.contact_patterns.size() << "\n";
    std::cout << "  Pattern: " << pt.contact_patterns[0].name << "\n";

    // Create objects
    {
        NeuropilObject d001;
        d001.name = "d001";
        d001.mesh = make_cube(1.0);
        NamedRegion r;
        r.name = "d001sy01a002";
        r.face_indices = {0, 1};
        d001.mesh.regions.push_back(r);
        pt.objects["d001"] = std::move(d001);
    }
    {
        NeuropilObject a002;
        a002.name = "a002";
        a002.mesh = make_cube(0.5);
        pt.objects["a002"] = std::move(a002);
    }

    // Update contact matches
    pt.update_all_contact_matches();
    for (auto& [name, obj] : pt.objects) {
        std::cout << "Object '" << name << "' contact matches:";
        for (auto& cm : obj.contact_pattern_matches)
            std::cout << " " << cm;
        std::cout << "\n";
    }

    // Smooth an object
    pt.smooth_object("d001", 3, 0.5);
    auto* d001_obj = pt.get_object("d001");
    std::cout << "d001 smoothed: " << (d001_obj && d001_obj->smoothed ? "yes" : "no") << "\n";

    // Merge objects
    pt.merge_objects({"d001", "a002"}, "merged_scene");
    auto* merged = pt.get_object("merged_scene");
    if (merged) {
        std::cout << "Merged object: " << merged->mesh.vertices.size() << " verts, "
                  << merged->mesh.triangles.size() << " faces\n";
    }

    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Demo 10: Spatial analysis
// ---------------------------------------------------------------------------
static void demo_spatial_analysis() {
    std::cout << "=== Demo 10: Spatial Analysis ===\n\n";

    // Point-to-triangle distance
    Vec3 p(0, 0, 2);
    Vec3 a(0, 0, 0), b(1, 0, 0), c(0, 1, 0);
    Vec3 cp = spatial::closest_point_on_triangle(p, a, b, c);
    std::cout << "Closest point on triangle to (0,0,2): " << cp << "\n";
    std::cout << "Distance: " << spatial::distance(p, cp) << "\n";

    // Hausdorff distance between two meshes
    Mesh cube1 = make_cube(1.0);
    Mesh cube2 = make_cube(1.0);
    cube2.translate({0.5, 0, 0});

    double hd = spatial::hausdorff_distance(cube1, cube2);
    std::cout << "\nHausdorff distance between cube and shifted cube: " << hd << "\n";

    // Bounding box of arbitrary points
    std::vector<Vec3> pts = {{1,2,3}, {-1,0,5}, {3,-2,1}};
    auto [mn, mx] = spatial::bounding_box(pts);
    std::cout << "Bounding box of points: " << mn << " to " << mx << "\n";

    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Demo 11: Data export
// ---------------------------------------------------------------------------
static void demo_data_export() {
    std::cout << "=== Demo 11: Data Export ===\n\n";

    // Create some test PSD data
    std::map<std::string, SpinePSDData> psd_map;
    {
        SpinePSDData d;
        d.name = "d001sy01a002";
        d.contact_type = ContactType::PROTRUSION;
        d.char_postsynaptic = true;
        d.volume_head = 0.035;
        d.volume_spine = 0.082;
        d.volume_neck = 0.047;
        d.area_psd_az = 0.12;
        d.diameter_head_max = 0.45;
        d.diameter_head_min = 0.31;
        d.diameter_neck_max = 0.18;
        d.diameter_neck_min = 0.12;
        d.psd_az_location = {1.5, 2.3, 0.8};
        psd_map["d001sy01a002"] = d;
    }
    {
        SpinePSDData d;
        d.name = "d002sy01a003";
        d.contact_type = ContactType::PLAIN;
        d.char_postsynaptic = false;
        d.area_psd_az = 0.08;
        d.psd_az_location = {-0.5, 1.1, 2.2};
        psd_map["d002sy01a003"] = d;
    }

    std::string csv_path = "/tmp/neuropil_demo_spines.csv";
    export_data::write_spine_csv(csv_path, psd_map);
    std::cout << "Wrote spine CSV to: " << csv_path << "\n";

    // Read back and display
    std::ifstream check(csv_path);
    std::string line;
    int n = 0;
    while (std::getline(check, line) && n < 5) {
        std::cout << "  " << line << "\n";
        ++n;
    }
    std::filesystem::remove(csv_path);

    // Connectivity CSV
    std::vector<ConnectivityRecord> records;
    records.push_back({"a001", "d001sy01a001", "d001"});
    records.push_back({"a002", "d001sy02a002", "d001"});
    std::string conn_csv = "/tmp/neuropil_demo_connectivity.csv";
    export_data::write_connectivity_csv(conn_csv, records);
    std::cout << "\nWrote connectivity CSV to: " << conn_csv << "\n";
    std::filesystem::remove(conn_csv);

    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "neuropil_tools C++ port -- demo program\n";
    std::cout << "========================================\n\n";

    demo_mesh_geometry();
    demo_mesh_operations();
    demo_obj_io();
    demo_mdl_insertion();
    demo_diameter();
    demo_connectivity();
    demo_contact_patterns();
    demo_spine_analysis();
    demo_processor_tool();
    demo_spatial_analysis();
    demo_data_export();

    std::cout << "All demos completed successfully.\n";
    return 0;
}
