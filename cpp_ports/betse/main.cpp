// BETSE C++ Port - Comprehensive Test/Demo
// Exercises: ion channels, pumps, gap junctions, ER, mitochondria,
//            tissue profiles, cutting events, voltage clamps, and GRN.
#include "betse.h"
#include <chrono>
#include <iomanip>

void test_basic_simulation() {
    using namespace betse;
    std::cout << "=== Test 1: Basic bioelectric simulation ===\n";

    SimConfig cfg;
    cfg.num_cells       = 50;
    cfg.tissue_radius   = 80.0e-6;
    cfg.dt              = 1.0e-4;
    cfg.total_time      = 0.5;
    cfg.sample_rate     = 500;
    cfg.pump_enabled    = true;
    cfg.noise_enabled   = true;
    cfg.noise_level     = 0.0001;

    Simulator sim(cfg);

    auto t0 = std::chrono::high_resolution_clock::now();
    sim.init();
    auto t1 = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "  Init: " << init_ms << " ms\n";
    std::cout << "  Cells: " << sim.mesh.cells.size()
              << ", Membranes: " << sim.mesh.membranes.size() << "\n";
    std::cout << "  Initial Vmem: " << std::fixed << std::setprecision(2)
              << sim.avg_Vmem() * 1000.0 << " mV\n";

    auto t2 = std::chrono::high_resolution_clock::now();
    bool ok = sim.run();
    auto t3 = std::chrono::high_resolution_clock::now();
    double run_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    if (ok) {
        std::cout << "  Completed in " << run_ms << " ms\n";
        std::cout << "  Final Vmem: " << sim.avg_Vmem() * 1000.0 << " mV\n";
        std::cout << "  Snapshots: " << sim.history.size() << "\n";
        sim.write_csv("betse_basic.csv");
        std::cout << "  Output: betse_basic.csv\n";
    } else {
        std::cout << "  FAILED (numerical instability)\n";
    }
}

void test_action_potential() {
    using namespace betse;
    std::cout << "\n=== Test 2: Action potential with Nav1.2 + Kv1.2 ===\n";

    SimConfig cfg;
    cfg.num_cells     = 20;
    cfg.tissue_radius = 50.0e-6;
    cfg.dt            = 1.0e-5;  // smaller dt for AP dynamics
    cfg.total_time    = 0.1;
    cfg.sample_rate   = 100;
    cfg.pump_enabled  = true;
    cfg.ca_dyn_enabled = true;

    // Add Nav1.2 channel
    cfg.channels.push_back({ChannelType::Nav1p2, 5.0e-15, -1});
    // Add Kv1.2 channel
    cfg.channels.push_back({ChannelType::Kv1p2, 3.0e-15, -1});

    // Add a brief voltage clamp to trigger AP
    cfg.voltage_clamps.push_back({0.01, 0.012, -0.020, -1}); // -20mV clamp

    Simulator sim(cfg);
    sim.init();

    std::cout << "  Initial Vmem: " << std::fixed << std::setprecision(2)
              << sim.avg_Vmem() * 1000.0 << " mV\n";
    std::cout << "  Active channels: " << sim.active_channels.size() << "\n";
    for (auto& ch : sim.active_channels) {
        std::cout << "    - " << ch.state.ions[0] << " channel, "
                  << ch.state.m.size() << " gates, maxDm=" << ch.max_Dm << "\n";
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = sim.run();
    auto t1 = std::chrono::high_resolution_clock::now();

    if (ok) {
        std::cout << "  Completed in "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
        std::cout << "  Final Vmem: " << sim.avg_Vmem() * 1000.0 << " mV\n";
        sim.write_csv("betse_ap.csv");
        std::cout << "  Output: betse_ap.csv\n";

        // Report min/max Vmem across all snapshots
        double v_min = 1e30, v_max = -1e30;
        for (auto& snap : sim.history) {
            for (double v : snap.Vmem) {
                v_min = std::min(v_min, v);
                v_max = std::max(v_max, v);
            }
        }
        std::cout << "  Vmem range: [" << v_min * 1000.0 << ", "
                  << v_max * 1000.0 << "] mV\n";
    } else {
        std::cout << "  FAILED\n";
    }
}

void test_channel_models() {
    using namespace betse;
    std::cout << "\n=== Test 3: Channel model verification ===\n";

    // Test all channel types at resting potential (-70 mV)
    struct TestCase {
        ChannelType type;
        const char* name;
    };

    std::vector<TestCase> tests = {
        {ChannelType::Nav1p2, "Nav1.2"},
        {ChannelType::Nav1p3, "Nav1.3"},
        {ChannelType::Nav1p6, "Nav1.6"},
        {ChannelType::NavRat1, "NavRat1"},
        {ChannelType::NavRat2, "NavRat2"},
        {ChannelType::NaLeak, "NaLeak"},
        {ChannelType::Kv1p1, "Kv1.1"},
        {ChannelType::Kv1p2, "Kv1.2"},
        {ChannelType::Kv1p3, "Kv1.3"},
        {ChannelType::Kv3p3, "Kv3.3"},
        {ChannelType::K_Fast, "K_Fast"},
        {ChannelType::KLeak, "KLeak"},
        {ChannelType::Kir2p1, "Kir2.1"},
        {ChannelType::Cav1p3, "Cav1.3"},
        {ChannelType::Cav2p1, "Cav2.1"},
        {ChannelType::Cav3p1, "Cav3.1"},
        {ChannelType::Cav3p3, "Cav3.3"},
        {ChannelType::Cav_G, "Cav_G"},
        {ChannelType::CaLeak, "CaLeak"},
        {ChannelType::HCN1, "HCN1"},
        {ChannelType::HCN2, "HCN2"},
        {ChannelType::HCN4, "HCN4"},
        {ChannelType::HCNLeak, "HCNLeak"},
        {ChannelType::ClLeak, "ClLeak"},
        {ChannelType::CatLeak, "CatLeak"},
        {ChannelType::Kv_ML1, "Kv_ML1"},
        {ChannelType::Nav_ML, "Nav_ML"},
        {ChannelType::Cav_L_ML, "Cav_L_ML"},
        {ChannelType::Kir_ML, "Kir_ML"},
        {ChannelType::TRP, "TRP"},
    };

    std::vector<double> Vm_mV = {-70.0};

    std::cout << std::setw(12) << "Channel" << std::setw(10) << "m_init"
              << std::setw(10) << "h_init" << std::setw(10) << "P_open"
              << std::setw(10) << "vrev" << std::setw(8) << "ions" << "\n";

    for (auto& tc : tests) {
        ChannelState cs;
        channel_init(cs, tc.type, Vm_mV);
        channel_step(cs, tc.type, Vm_mV, 1e-4, 1);

        std::string ion_str;
        for (auto& ion : cs.ions) ion_str += ion + " ";

        std::cout << std::setw(12) << tc.name
                  << std::setw(10) << std::setprecision(4) << cs.m[0]
                  << std::setw(10) << cs.h[0]
                  << std::setw(10) << (cs.P.empty() ? 0.0 : cs.P[0])
                  << std::setw(10) << cs.vrev
                  << "  " << ion_str << "\n";
    }
}

void test_organelles() {
    using namespace betse;
    std::cout << "\n=== Test 4: Simulation with ER and mitochondria ===\n";

    SimConfig cfg;
    cfg.num_cells     = 30;
    cfg.tissue_radius = 60.0e-6;
    cfg.dt            = 1.0e-4;
    cfg.total_time    = 0.2;
    cfg.sample_rate   = 200;
    cfg.pump_enabled  = true;
    cfg.ca_dyn_enabled = true;
    cfg.er_enabled    = true;
    cfg.mito_enabled  = true;

    Simulator sim(cfg);
    sim.init();

    std::cout << "  ER Ca initial: " << sim.er.cc_er[ion_index(Ion::Ca)][0] << " mM\n";
    std::cout << "  Cell Ca initial: " << sim.state.cc_cells[ion_index(Ion::Ca)][0] << " mM\n";

    bool ok = sim.run();
    if (ok) {
        std::cout << "  ER Ca final: " << sim.er.cc_er[ion_index(Ion::Ca)][0] << " mM\n";
        std::cout << "  Cell Ca final: " << sim.state.cc_cells[ion_index(Ion::Ca)][0] << " mM\n";
        std::cout << "  Final Vmem: " << sim.avg_Vmem() * 1000.0 << " mV\n";
        sim.write_csv("betse_organelles.csv");
        std::cout << "  Output: betse_organelles.csv\n";
    } else {
        std::cout << "  FAILED\n";
    }
}

void test_tissue_profiles() {
    using namespace betse;
    std::cout << "\n=== Test 5: Tissue heterogeneity and cutting ===\n";

    SimConfig cfg;
    cfg.num_cells     = 40;
    cfg.tissue_radius = 70.0e-6;
    cfg.dt            = 1.0e-4;
    cfg.total_time    = 0.3;
    cfg.sample_rate   = 300;
    cfg.pump_enabled  = true;

    // Define a circular tissue profile
    SimConfig::TissueProfile prof;
    prof.id = 1;
    prof.name = "excitable_region";
    prof.shape = TissueProfileShape::CIRCULAR;
    prof.center = {0, 0};
    prof.radius = 30.0e-6;
    cfg.tissue_profiles.push_back(prof);

    // Add channels only to the excitable region
    cfg.channels.push_back({ChannelType::Nav_ML, 1.0e-14, 1});
    cfg.channels.push_back({ChannelType::Kv_ML1, 5.0e-15, 1});

    // Schedule a cutting event
    cfg.cut_events.push_back({0.15, {40.0e-6, 0}, 15.0e-6});

    Simulator sim(cfg);
    sim.init();

    int excitable_count = 0;
    for (auto& cell : sim.mesh.cells) {
        if (cell.tissue_profile_id == 1) excitable_count++;
    }
    std::cout << "  Excitable cells: " << excitable_count << " / "
              << sim.mesh.cells.size() << "\n";

    bool ok = sim.run();
    if (ok) {
        std::cout << "  Cells after cut: " << sim.mesh.cells.size() << "\n";
        std::cout << "  Final Vmem: " << sim.avg_Vmem() * 1000.0 << " mV\n";
        sim.write_csv("betse_tissue.csv");
        std::cout << "  Output: betse_tissue.csv\n";
    } else {
        std::cout << "  FAILED\n";
    }
}

void test_grn() {
    using namespace betse;
    std::cout << "\n=== Test 6: Gene regulatory network ===\n";

    SimConfig cfg;
    cfg.num_cells     = 20;
    cfg.tissue_radius = 50.0e-6;
    cfg.dt            = 1.0e-4;
    cfg.total_time    = 0.1;
    cfg.sample_rate   = 100;
    cfg.pump_enabled  = true;
    cfg.grn_enabled   = true;

    Simulator sim(cfg);

    // Add a simple molecule with growth and decay
    Molecule mol;
    mol.name = "morphogen_A";
    mol.initial_cell = 0.1;
    mol.initial_env = 0.0;
    mol.growth_rate = 0.01;   // production rate [mM/s]
    mol.decay_rate = 0.1;     // decay rate [1/s]
    mol.D_free = 1e-10;
    sim.grn.molecules["morphogen_A"] = mol;

    Molecule mol2;
    mol2.name = "morphogen_B";
    mol2.initial_cell = 0.0;
    mol2.initial_env = 0.0;
    mol2.growth_rate = 0.0;
    mol2.decay_rate = 0.05;
    mol2.D_free = 1e-10;
    sim.grn.molecules["morphogen_B"] = mol2;

    // Add a reaction: A -> B
    Reaction rxn;
    rxn.name = "A_to_B";
    rxn.reactants = {"morphogen_A"};
    rxn.products = {"morphogen_B"};
    rxn.reactant_stoich = {1.0};
    rxn.product_stoich = {1.0};
    rxn.rate_const = 0.05;
    rxn.Km = 0.05;
    rxn.reaction_zone = 0;
    sim.grn.reactions.push_back(rxn);

    sim.init();

    std::cout << "  Molecules: " << sim.grn.molecules.size() << "\n";
    std::cout << "  Reactions: " << sim.grn.reactions.size() << "\n";
    std::cout << "  Initial A: " << sim.grn.molecules["morphogen_A"].c_cells[0] << " mM\n";

    bool ok = sim.run();
    if (ok) {
        std::cout << "  Final A: " << sim.grn.molecules["morphogen_A"].c_cells[0] << " mM\n";
        std::cout << "  Final B: " << sim.grn.molecules["morphogen_B"].c_cells[0] << " mM\n";
        std::cout << "  Final Vmem: " << sim.avg_Vmem() * 1000.0 << " mV\n";
    } else {
        std::cout << "  FAILED\n";
    }
}

void test_full_featured() {
    using namespace betse;
    std::cout << "\n=== Test 7: Full-featured simulation ===\n";

    SimConfig cfg;
    cfg.num_cells       = 30;
    cfg.tissue_radius   = 60.0e-6;
    cfg.dt              = 5.0e-5;
    cfg.total_time      = 0.05;
    cfg.sample_rate     = 200;
    cfg.pump_enabled    = true;
    cfg.ca_dyn_enabled  = true;
    cfg.er_enabled      = true;
    cfg.mito_enabled    = true;
    cfg.osmo_enabled    = true;
    cfg.deform_enabled  = true;
    cfg.flow_enabled    = true;
    cfg.noise_enabled   = true;
    cfg.noise_level     = 0.00005;

    // Multiple channel types
    cfg.channels.push_back({ChannelType::Nav1p2, 2.0e-15, -1});
    cfg.channels.push_back({ChannelType::Kv1p5, 1.0e-15, -1});
    cfg.channels.push_back({ChannelType::Cav1p3, 5.0e-16, -1});
    cfg.channels.push_back({ChannelType::HCN2, 3.0e-16, -1});
    cfg.channels.push_back({ChannelType::KLeak, 1.0e-16, -1});

    Simulator sim(cfg);
    sim.init();

    std::cout << "  Cells: " << sim.mesh.cells.size() << "\n";
    std::cout << "  Active channel types: " << sim.active_channels.size() << "\n";
    std::cout << "  ER enabled: " << (cfg.er_enabled ? "yes" : "no") << "\n";
    std::cout << "  Mito enabled: " << (cfg.mito_enabled ? "yes" : "no") << "\n";
    std::cout << "  Osmo enabled: " << (cfg.osmo_enabled ? "yes" : "no") << "\n";
    std::cout << "  Deform enabled: " << (cfg.deform_enabled ? "yes" : "no") << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = sim.run();
    auto t1 = std::chrono::high_resolution_clock::now();

    if (ok) {
        std::cout << "  Completed in "
                  << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";
        std::cout << "  Final Vmem: " << sim.avg_Vmem() * 1000.0 << " mV\n";

        // Report ion concentrations
        std::cout << "  Final concentrations:\n";
        for (int ion = 0; ion < (int)Ion::COUNT; ion++) {
            double avg = 0;
            for (int c = 0; c < sim.state.num_cells; c++)
                avg += sim.state.cc_cells[ion][c];
            avg /= sim.state.num_cells;
            std::cout << "    " << ion_name(static_cast<Ion>(ion)) << ": "
                      << avg << " mM\n";
        }

        sim.write_csv("betse_full.csv");
        std::cout << "  Output: betse_full.csv\n";
    } else {
        std::cout << "  FAILED\n";
    }
}

int main() {
    std::cout << "BETSE C++ Port - Comprehensive Test Suite\n";
    std::cout << "==========================================\n";

    test_basic_simulation();
    test_action_potential();
    test_channel_models();
    test_organelles();
    test_tissue_profiles();
    test_grn();
    test_full_featured();

    std::cout << "\n==========================================\n";
    std::cout << "All tests completed.\n";

    return 0;
}
