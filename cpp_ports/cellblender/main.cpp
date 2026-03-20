// main.cpp - Demo program for the CellBlender C++ port
// Demonstrates: model building, MDL generation, data model serialization,
// geometry processing, mesh analysis, parameter sweeps, and MDL round-tripping.

#include "cellblender.h"
#include <iostream>
#include <cassert>
#include <cmath>

namespace cb = cellblender;

// Helper: compare doubles with tolerance
bool approx_eq(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

// ============================================================================
// Demo 1: Build a complete MCell model and export MDL
// ============================================================================
void demo_model_building() {
    std::cout << "=== Demo 1: Model Building and MDL Export ===\n";

    cb::DataModel dm;

    // Define parameters
    dm.parameters.define("dc_A", "1e-6");
    dm.parameters.define("dc_B", "5e-7");
    dm.parameters.define("kf", "1e8");
    dm.parameters.define("n_A", "1000");
    dm.parameters.define("n_B", "500");

    // Define molecules
    {
        cb::MoleculeSpecies mol_a;
        mol_a.name = "A";
        mol_a.mol_type = cb::MoleculeSpecies::VOLUME_3D;
        mol_a.diffusion_constant = "dc_A";
        mol_a.export_viz = true;
        dm.molecules.push_back(mol_a);

        cb::MoleculeSpecies mol_b;
        mol_b.name = "B";
        mol_b.mol_type = cb::MoleculeSpecies::VOLUME_3D;
        mol_b.diffusion_constant = "dc_B";
        mol_b.export_viz = true;
        dm.molecules.push_back(mol_b);

        cb::MoleculeSpecies mol_c;
        mol_c.name = "C";
        mol_c.mol_type = cb::MoleculeSpecies::VOLUME_3D;
        mol_c.diffusion_constant = "1e-6";
        mol_c.export_viz = true;
        dm.molecules.push_back(mol_c);

        cb::MoleculeSpecies mol_s;
        mol_s.name = "S";
        mol_s.mol_type = cb::MoleculeSpecies::SURFACE_2D;
        mol_s.diffusion_constant = "1e-7";
        mol_s.export_viz = true;
        dm.molecules.push_back(mol_s);
    }

    // Define reactions
    {
        cb::Reaction rxn1;
        rxn1.reactants = "A + B";
        rxn1.products = "C";
        rxn1.rxn_type = cb::Reaction::IRREVERSIBLE;
        rxn1.fwd_rate = "kf";
        rxn1.rxn_name = "binding";
        rxn1.update_name();
        dm.reactions.push_back(rxn1);

        cb::Reaction rxn2;
        rxn2.reactants = "C";
        rxn2.products = "A + B";
        rxn2.rxn_type = cb::Reaction::IRREVERSIBLE;
        rxn2.fwd_rate = "1e3";
        rxn2.rxn_name = "unbinding";
        rxn2.update_name();
        dm.reactions.push_back(rxn2);
    }

    // Create geometry: a cube as the simulation volume
    auto cube = cb::geometry::make_cube("Cube", 0.5);
    // Add a surface region covering top faces
    cb::SurfaceRegion top_region;
    top_region.name = "top";
    top_region.face_indices = {2, 3};  // top faces of cube
    cube.regions.push_back(top_region);
    dm.geometry_objects.push_back(cube);

    // Define surface class
    {
        cb::SurfaceClass sc;
        sc.name = "absorb_A";
        cb::SurfaceClassProperty prop;
        prop.affected_mols = cb::SurfaceClassProperty::SINGLE;
        prop.molecule = "A";
        prop.orient = cb::SurfaceClassProperty::TOP_FRONT;
        prop.class_type = cb::SurfaceClassProperty::ABSORPTIVE;
        sc.properties.push_back(prop);
        dm.surface_classes.push_back(sc);
    }

    // Modify surface region
    {
        cb::ModSurfaceRegion msr;
        msr.object_name = "Cube";
        msr.region_name = "top";
        msr.surf_class_name = "absorb_A";
        dm.mod_surf_regions.push_back(msr);
    }

    // Define release sites
    {
        cb::ReleaseSite rel_a;
        rel_a.name = "Release_A";
        rel_a.molecule = "A";
        rel_a.shape = cb::ReleaseSite::OBJECT;
        rel_a.object_expr = "Cube";
        rel_a.quantity_type = cb::ReleaseSite::NUMBER_TO_RELEASE;
        rel_a.quantity = "n_A";
        dm.release_sites.push_back(rel_a);

        cb::ReleaseSite rel_b;
        rel_b.name = "Release_B";
        rel_b.molecule = "B";
        rel_b.shape = cb::ReleaseSite::SPHERICAL;
        rel_b.location_x = "0";
        rel_b.location_y = "0";
        rel_b.location_z = "0";
        rel_b.diameter = "0.5";
        rel_b.quantity_type = cb::ReleaseSite::NUMBER_TO_RELEASE;
        rel_b.quantity = "n_B";
        dm.release_sites.push_back(rel_b);
    }

    // Define reaction outputs
    {
        cb::ReactionOutput ro_a;
        ro_a.molecule_name = "A";
        ro_a.rxn_or_mol = cb::ReactionOutput::MOLECULE;
        ro_a.count_location = cb::ReactionOutput::WORLD;
        dm.reaction_outputs.push_back(ro_a);

        cb::ReactionOutput ro_b;
        ro_b.molecule_name = "B";
        ro_b.rxn_or_mol = cb::ReactionOutput::MOLECULE;
        ro_b.count_location = cb::ReactionOutput::WORLD;
        dm.reaction_outputs.push_back(ro_b);

        cb::ReactionOutput ro_c;
        ro_c.molecule_name = "C";
        ro_c.rxn_or_mol = cb::ReactionOutput::MOLECULE;
        ro_c.count_location = cb::ReactionOutput::WORLD;
        dm.reaction_outputs.push_back(ro_c);
    }

    // Set initialization parameters
    dm.initialization.iterations = "10000";
    dm.initialization.time_step = "1e-6";
    dm.initialization.partitions = cb::geometry::auto_partitions(dm.geometry_objects, 0.1);

    // Export as modular MDL
    namespace fs = std::filesystem;
    fs::path output_dir = fs::temp_directory_path() / "cellblender_demo";
    fs::create_directories(output_dir);

    cb::MDLWriter writer(dm);
    writer.write_modular(output_dir);
    std::cout << "  Modular MDL written to: " << output_dir << "\n";

    // Also write single-file MDL
    fs::path single_mdl = output_dir / "all_in_one.mdl";
    writer.write_single(single_mdl);
    std::cout << "  Single-file MDL written to: " << single_mdl << "\n";

    // Export as JSON data model
    fs::path json_path = output_dir / "model.json";
    cb::DataModelIO::save_json(dm, json_path);
    std::cout << "  JSON data model written to: " << json_path << "\n";

    // Verify the main MDL file exists and has content
    std::ifstream check(output_dir / "Scene.main.mdl");
    std::string content((std::istreambuf_iterator<char>(check)),
                         std::istreambuf_iterator<char>());
    assert(!content.empty());
    assert(content.find("INCLUDE_FILE") != std::string::npos);
    std::cout << "  Main MDL file verified (" << content.size() << " bytes)\n";

    // Verify molecules MDL
    std::ifstream mol_check(output_dir / "Scene.molecules.mdl");
    std::string mol_content((std::istreambuf_iterator<char>(mol_check)),
                             std::istreambuf_iterator<char>());
    assert(mol_content.find("DEFINE_MOLECULES") != std::string::npos);
    assert(mol_content.find("DIFFUSION_CONSTANT_3D = dc_A") != std::string::npos);
    assert(mol_content.find("DIFFUSION_CONSTANT_2D = 1e-7") != std::string::npos);
    std::cout << "  Molecules MDL verified\n";

    // Verify reactions MDL
    std::ifstream rxn_check(output_dir / "Scene.reactions.mdl");
    std::string rxn_content((std::istreambuf_iterator<char>(rxn_check)),
                             std::istreambuf_iterator<char>());
    assert(rxn_content.find("DEFINE_REACTIONS") != std::string::npos);
    assert(rxn_content.find("binding") != std::string::npos);
    std::cout << "  Reactions MDL verified\n";

    std::cout << "  PASSED\n\n";
}

// ============================================================================
// Demo 2: Mesh Analysis (port of cellblender_meshalyzer.py)
// ============================================================================
void demo_mesh_analysis() {
    std::cout << "=== Demo 2: Mesh Analysis ===\n";

    // Create a unit cube
    auto cube = cb::geometry::make_cube("TestCube", 1.0);
    std::cout << "  Cube: " << cube.num_vertices() << " vertices, "
              << cube.num_edges() << " edges, "
              << cube.num_faces() << " faces\n";

    assert(cube.num_vertices() == 8);
    assert(cube.num_faces() == 12);  // 6 faces * 2 triangles each
    assert(cube.num_edges() == 18);  // 12 edges + 6 diagonal edges

    // Euler characteristic for a closed surface: V - E + F = 2
    std::cout << "  Euler characteristic: " << cube.euler_characteristic() << "\n";
    assert(cube.euler_characteristic() == 2);

    // Genus should be 0 for a sphere-topology
    std::cout << "  Genus: " << cube.genus() << "\n";
    assert(cube.genus() == 0);

    // Surface area of a 2x2x2 cube = 6 * 4 = 24
    double area = cube.surface_area();
    std::cout << "  Surface area: " << area << "\n";
    assert(approx_eq(area, 24.0, 0.01));

    // Volume of a 2x2x2 cube = 8
    double vol = cube.signed_volume();
    std::cout << "  Signed volume: " << vol << "\n";
    // Note: sign depends on face orientation; check absolute value
    assert(approx_eq(std::abs(vol), 8.0, 0.01));

    // Watertight check
    std::cout << "  Watertight: " << (cube.is_watertight() ? "yes" : "no") << "\n";
    assert(cube.is_watertight());

    // Bounding box
    auto [bb_min, bb_max] = cb::geometry::bounding_box(cube);
    std::cout << "  Bounding box: [" << bb_min.x << "," << bb_min.y << "," << bb_min.z << "] to ["
              << bb_max.x << "," << bb_max.y << "," << bb_max.z << "]\n";
    assert(approx_eq(bb_min.x, -1.0));
    assert(approx_eq(bb_max.x, 1.0));

    // Test icosphere
    auto sphere = cb::geometry::make_icosphere("TestSphere", 1.0, 2);
    std::cout << "  Icosphere (2 subdivisions): " << sphere.num_vertices() << " vertices, "
              << sphere.num_faces() << " faces\n";
    assert(sphere.num_vertices() > 12);  // More than base icosahedron
    assert(sphere.is_watertight());

    // Surface area should approximate 4*pi for unit sphere
    double sphere_area = sphere.surface_area();
    std::cout << "  Icosphere area: " << sphere_area << " (4*pi = " << 4*M_PI << ")\n";
    // Should be within ~5% of analytical value with 2 subdivisions
    assert(std::abs(sphere_area - 4*M_PI) / (4*M_PI) < 0.05);

    std::cout << "  PASSED\n\n";
}

// ============================================================================
// Demo 3: Parameter Space (port of ParameterSpace.py)
// ============================================================================
void demo_parameter_space() {
    std::cout << "=== Demo 3: Parameter Space ===\n";

    cb::ParameterSpace ps;

    // Define parameters
    int id_a = ps.define("alpha", "0.5");
    int id_b = ps.define("beta", "1.5");
    [[maybe_unused]] int id_c = ps.define("gamma", "2.0");

    assert(ps.num_parameters() == 3);
    assert(approx_eq(ps.get_value(id_a), 0.5));
    assert(approx_eq(ps.get_value("beta"), 1.5));
    assert(ps.get_name(id_c) == "gamma");
    assert(ps.get_id("alpha") == id_a);

    // Update expression
    ps.set_expr(id_a, "0.75");
    assert(approx_eq(ps.get_value(id_a), 0.75));

    // String/value export
    std::string as_expr = ps.get_as_string_or_value(id_b, true);
    assert(as_expr == "1.5");
    std::string as_val = ps.get_as_string_or_value(id_b, false);
    assert(as_val.find("1.5") != std::string::npos);

    // Get all names
    auto names = ps.get_all_names();
    assert(names.size() == 3);

    // Remove parameter
    assert(ps.remove("gamma"));
    assert(ps.num_parameters() == 2);
    assert(!ps.has("gamma"));

    // Clear all
    ps.clear();
    assert(ps.num_parameters() == 0);

    std::cout << "  All parameter space tests passed\n";
    std::cout << "  PASSED\n\n";
}

// ============================================================================
// Demo 4: MDL Round-Trip (Write -> Parse -> Verify)
// ============================================================================
void demo_mdl_roundtrip() {
    std::cout << "=== Demo 4: MDL Round-Trip ===\n";

    // Create a model
    cb::DataModel dm;
    dm.initialization.iterations = "5000";
    dm.initialization.time_step = "1e-5";
    dm.initialization.partitions.include = true;
    dm.initialization.partitions.x_start = -2; dm.initialization.partitions.x_end = 2;
    dm.initialization.partitions.x_step = 0.5;
    dm.initialization.partitions.y_start = -2; dm.initialization.partitions.y_end = 2;
    dm.initialization.partitions.y_step = 0.5;
    dm.initialization.partitions.z_start = -2; dm.initialization.partitions.z_end = 2;
    dm.initialization.partitions.z_step = 0.5;

    cb::MoleculeSpecies mol;
    mol.name = "X";
    mol.mol_type = cb::MoleculeSpecies::VOLUME_3D;
    mol.diffusion_constant = "2e-6";
    mol.export_viz = true;
    dm.molecules.push_back(mol);

    cb::MoleculeSpecies mol_s;
    mol_s.name = "Y";
    mol_s.mol_type = cb::MoleculeSpecies::SURFACE_2D;
    mol_s.diffusion_constant = "1e-7";
    mol_s.target_only = true;
    dm.molecules.push_back(mol_s);

    cb::Reaction rxn;
    rxn.reactants = "X + X";
    rxn.products = "X";
    rxn.rxn_type = cb::Reaction::IRREVERSIBLE;
    rxn.fwd_rate = "1e7";
    rxn.rxn_name = "dimerize";
    rxn.update_name();
    dm.reactions.push_back(rxn);

    auto cube = cb::geometry::make_cube("Box", 1.0);
    dm.geometry_objects.push_back(cube);

    // Write to temp file
    namespace fs = std::filesystem;
    fs::path tmp_dir = fs::temp_directory_path() / "cellblender_roundtrip";
    fs::create_directories(tmp_dir);
    cb::MDLWriter writer(dm);
    writer.write_single(tmp_dir / "test.mdl");

    // Parse it back
    cb::MDLParser parser;
    auto parsed = parser.parse(tmp_dir / "test.mdl");

    // Verify
    assert(parsed.initialization.iterations == "5000");
    assert(parsed.initialization.time_step == "1e-5");
    assert(parsed.molecules.size() == 2);
    assert(parsed.molecules[0].name == "X");
    assert(parsed.molecules[0].diffusion_constant == "2e-6");
    assert(parsed.molecules[1].name == "Y");
    assert(parsed.molecules[1].mol_type == cb::MoleculeSpecies::SURFACE_2D);
    assert(parsed.molecules[1].target_only == true);
    assert(parsed.reactions.size() == 1);
    assert(parsed.reactions[0].fwd_rate == "1e7");
    assert(parsed.reactions[0].rxn_name == "dimerize");
    assert(parsed.geometry_objects.size() == 1);
    assert(parsed.geometry_objects[0].name == "Box");
    assert(parsed.geometry_objects[0].vertices.size() == 8);
    assert(parsed.geometry_objects[0].faces.size() == 12);
    assert(parsed.initialization.partitions.include == true);
    assert(approx_eq(parsed.initialization.partitions.x_start, -2.0));
    assert(approx_eq(parsed.initialization.partitions.x_end, 2.0));

    std::cout << "  MDL round-trip verified successfully\n";
    std::cout << "  PASSED\n\n";
}

// ============================================================================
// Demo 5: Parameter Sweep Engine
// ============================================================================
void demo_parameter_sweep() {
    std::cout << "=== Demo 5: Parameter Sweep ===\n";

    cb::ParameterSweepEngine sweep;
    sweep.add_parameter("dc", {1e-6, 5e-6, 1e-5});
    sweep.add_parameter("rate", {1e7, 1e8});

    assert(sweep.total_runs() == 6);  // 3 * 2

    // Check that all points are generated correctly
    std::set<std::string> labels;
    for (size_t i = 0; i < sweep.total_runs(); ++i) {
        auto point = sweep.get_point(i);
        std::string label = sweep.get_label(i);
        std::string path = sweep.get_run_path(i);
        labels.insert(label);

        std::cout << "  Run " << i << ": " << label << " -> " << path << "\n";

        // Verify point has both parameters
        assert(point.count("dc") == 1);
        assert(point.count("rate") == 1);
    }
    assert(labels.size() == 6);  // All unique

    // Generate models from a base data model
    cb::DataModel base_dm;
    base_dm.parameters.define("dc", "1e-6");
    base_dm.parameters.define("rate", "1e7");

    auto models = sweep.generate_models(base_dm);
    assert(models.size() == 6);

    // First model should have dc=1e-6, rate=1e7
    assert(approx_eq(models[0].second.parameters.get_value("dc"), 1e-6));
    assert(approx_eq(models[0].second.parameters.get_value("rate"), 1e7));

    // Write data layout
    namespace fs = std::filesystem;
    fs::path sweep_dir = fs::temp_directory_path() / "cellblender_sweep";
    sweep.write_data_layout(sweep_dir, 1, 3);

    // Verify data_layout.json was written
    std::ifstream dl_check(sweep_dir / "data_layout.json");
    std::string dl_content((std::istreambuf_iterator<char>(dl_check)),
                            std::istreambuf_iterator<char>());
    assert(dl_content.find("\"version\": 2") != std::string::npos);
    assert(dl_content.find("\"dc\"") != std::string::npos);
    assert(dl_content.find("\"rate\"") != std::string::npos);

    std::cout << "  PASSED\n\n";
}

// ============================================================================
// Demo 6: Simulation Preparation
// ============================================================================
void demo_simulation_prep() {
    std::cout << "=== Demo 6: Simulation Preparation ===\n";

    cb::DataModel dm;

    cb::MoleculeSpecies mol;
    mol.name = "Ligand";
    mol.diffusion_constant = "1e-6";
    mol.export_viz = true;
    dm.molecules.push_back(mol);

    auto sphere = cb::geometry::make_icosphere("Cell", 0.5, 1);
    dm.geometry_objects.push_back(sphere);

    cb::ReleaseSite rel;
    rel.name = "Release_Ligand";
    rel.molecule = "Ligand";
    rel.shape = cb::ReleaseSite::OBJECT;
    rel.object_expr = "Cell";
    rel.quantity_type = cb::ReleaseSite::NUMBER_TO_RELEASE;
    rel.quantity = "500";
    dm.release_sites.push_back(rel);

    cb::ReactionOutput ro;
    ro.molecule_name = "Ligand";
    ro.rxn_or_mol = cb::ReactionOutput::MOLECULE;
    ro.count_location = cb::ReactionOutput::WORLD;
    dm.reaction_outputs.push_back(ro);

    dm.initialization.iterations = "1000";
    dm.initialization.time_step = "1e-6";
    dm.initialization.partitions = cb::geometry::auto_partitions(dm.geometry_objects, 0.1);

    // Prepare simulation
    namespace fs = std::filesystem;
    cb::SimulationRunner::RunConfig config;
    config.project_dir = fs::temp_directory_path() / "cellblender_sim";
    config.start_seed = 1;
    config.end_seed = 3;
    config.mcell_binary = "mcell";

    cb::SimulationRunner::prepare(dm, config);

    // Verify output structure
    assert(fs::exists(config.project_dir / "output_data"));
    assert(fs::exists(config.project_dir / "output_data" / "react_data"));
    assert(fs::exists(config.project_dir / "output_data" / "viz_data"));
    assert(fs::exists(config.project_dir / "output_data" / "Scene.main.mdl"));
    assert(fs::exists(config.project_dir / "data_layout.json"));

    // Generate commands
    auto commands = cb::SimulationRunner::generate_commands(config);
    assert(commands.size() == 3);  // 3 seeds
    for (int i = 0; i < 3; ++i) {
        assert(commands[i][0] == "mcell");
        assert(commands[i][1] == "-seed");
        assert(commands[i][2] == std::to_string(i + 1));
        std::cout << "  Command " << i << ": ";
        for (auto& arg : commands[i]) std::cout << arg << " ";
        std::cout << "\n";
    }

    std::cout << "  PASSED\n\n";
}

// ============================================================================
// Demo 7: Results Parsing
// ============================================================================
void demo_results_parsing() {
    std::cout << "=== Demo 7: Results Parsing ===\n";

    // Create some synthetic result files
    namespace fs = std::filesystem;
    fs::path react_dir = fs::temp_directory_path() / "cellblender_results" / "react_data";

    for (int seed = 1; seed <= 3; ++seed) {
        char sd[64];
        std::snprintf(sd, sizeof(sd), "seed_%05d", seed);
        fs::path seed_dir = react_dir / sd;
        fs::create_directories(seed_dir);

        // Write a synthetic .dat file
        std::ofstream f(seed_dir / "Ligand.World.dat");
        for (int i = 0; i <= 100; ++i) {
            double t = i * 1e-6;
            double v = 500.0 * std::exp(-0.001 * i) + seed * 5;  // Slightly different per seed
            f << std::setprecision(15) << t << " " << v << "\n";
        }
    }

    // Parse a single file
    auto ts = cb::ResultsParser::parse_dat_file(react_dir / "seed_00001" / "Ligand.World.dat");
    assert(!ts.times.empty());
    assert(ts.times.size() == 101);
    assert(ts.label == "Ligand.World");
    std::cout << "  Parsed single file: " << ts.times.size() << " data points\n";
    std::cout << "    First: t=" << ts.times[0] << " v=" << ts.values[0] << "\n";
    std::cout << "    Last:  t=" << ts.times.back() << " v=" << ts.values.back() << "\n";

    // Parse all files in a seed directory
    auto all_ts = cb::ResultsParser::parse_seed_dir(react_dir / "seed_00001");
    assert(!all_ts.empty());
    std::cout << "  Parsed seed directory: " << all_ts.size() << " files\n";

    // Aggregate across seeds
    auto agg = cb::ResultsParser::aggregate_seeds(react_dir, "Ligand.World.dat", 1, 3);
    assert(agg.num_seeds == 3);
    assert(!agg.means.empty());
    assert(!agg.stddevs.empty());
    std::cout << "  Aggregated " << agg.num_seeds << " seeds\n";
    std::cout << "    Mean at t=0: " << agg.means[0] << " +/- " << agg.stddevs[0] << "\n";
    std::cout << "    Mean at t=end: " << agg.means.back() << " +/- " << agg.stddevs.back() << "\n";

    // Verify that stddev is non-zero (since seeds have different values)
    assert(agg.stddevs[0] > 0);

    std::cout << "  PASSED\n\n";
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "CellBlender C++ Port - Demo and Verification Program\n";
    std::cout << "=====================================================\n\n";

    try {
        demo_model_building();
        demo_mesh_analysis();
        demo_parameter_space();
        demo_mdl_roundtrip();
        demo_parameter_sweep();
        demo_simulation_prep();
        demo_results_parsing();

        std::cout << "=====================================================\n";
        std::cout << "All demos passed successfully.\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
