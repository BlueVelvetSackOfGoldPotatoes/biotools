#include "models/cells/src/cell_circuit.h"
#include "models/cells/src/cell_models.h"
#include "models/cells/src/cell_module.h"
#include "models/cells/src/cell_tasks.h"
#include "models/cells/src/sim/sph/neighbor_grid.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const std::string& name) {
    if (cond) {
        std::cout << "  PASS  " << name << std::endl;
        g_pass++;
    } else {
        std::cout << "  FAIL  " << name << std::endl;
        g_fail++;
    }
}

static cells::Vec3 centroid(const std::vector<cells::Vec3>& x) {
    cells::Vec3 c(0.0, 0.0, 0.0);
    if (x.empty()) return c;
    for (const auto& p : x) c += p;
    c /= static_cast<double>(x.size());
    return c;
}

static bool equal_vec3_arrays_exact(const std::vector<cells::Vec3>& a, const std::vector<cells::Vec3>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].x != b[i].x || a[i].y != b[i].y || a[i].z != b[i].z) {
            return false;
        }
    }
    return true;
}

static bool equal_task_trace_exact(const std::vector<cells::CellTaskTraceRow>& a,
                                   const std::vector<cells::CellTaskTraceRow>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].step != b[i].step ||
            a[i].reward != b[i].reward ||
            a[i].distance_to_target != b[i].distance_to_target ||
            a[i].energy != b[i].energy ||
            a[i].homeostasis_score != b[i].homeostasis_score ||
            a[i].volume_drift_pct != b[i].volume_drift_pct ||
            a[i].area_drift_pct != b[i].area_drift_pct ||
            a[i].centroid_speed != b[i].centroid_speed ||
            a[i].centroid.x != b[i].centroid.x ||
            a[i].centroid.y != b[i].centroid.y ||
            a[i].centroid.z != b[i].centroid.z ||
            a[i].target.x != b[i].target.x ||
            a[i].target.y != b[i].target.y ||
            a[i].target.z != b[i].target.z) {
            return false;
        }
    }
    return true;
}

static void test_phase0_determinism() {
    cells::CellInit init;
    init.icosphere_subdivisions = 2;
    init.fluid_particles = 600;
    init.seed = 2026;

    cells::CellSim a(init);
    cells::CellSim b(init);
    for (int i = 0; i < 350; ++i) {
        if ((i % 70) == 0) {
            a.poke_vertex(0, cells::Vec3(0.4, -0.1, 0.0));
            b.poke_vertex(0, cells::Vec3(0.4, -0.1, 0.0));
        }
        a.step(0.005);
        b.step(0.005);
    }

    check(equal_vec3_arrays_exact(a.state().membrane_positions, b.state().membrane_positions),
          "phase0 deterministic membrane positions");
    check(equal_vec3_arrays_exact(a.state().particle_positions, b.state().particle_positions),
          "phase0 deterministic particle positions");
}

static void test_phase1_membrane() {
    cells::CellInit init0;
    init0.icosphere_subdivisions = 0;
    init0.fluid_particles = 0;
    cells::CellSim sim0(init0);
    check(sim0.vertex_count() == 12, "phase1 L0 vertex count");
    check(sim0.triangle_count() == 20, "phase1 L0 triangle count");
    check(sim0.edge_count() == 30, "phase1 L0 edge count");

    cells::CellInit init3;
    init3.icosphere_subdivisions = 3;
    init3.fluid_particles = 0;
    cells::CellSim sim3(init3);
    check(sim3.vertex_count() == 642, "phase1 L3 vertex count");
    check(sim3.triangle_count() == 1280, "phase1 L3 triangle count");
    check(sim3.edge_count() == 1920, "phase1 L3 edge count");

    sim3.poke_vertex(0, cells::Vec3(0.8, 0.0, 0.0));
    for (int i = 0; i < 3000; ++i) sim3.step(0.005);
    check(sim3.diagnostics().volume_drift_pct < 2.0, "phase1 volume drift <2%");
}

static void test_phase2_neighbor_grid() {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> ud(-1.0, 1.0);

    std::vector<cells::Vec3> pos(128);
    for (auto& p : pos) p = cells::Vec3(ud(rng), ud(rng), ud(rng));

    const double h = 0.3;
    cells::sph::NeighborGrid grid;
    grid.build(pos, h);

    bool all_ok = true;
    for (std::size_t i = 0; i < pos.size(); ++i) {
        std::vector<int> neigh_grid;
        grid.for_neighbor_candidates(pos[i], [&](std::uint32_t j) {
            const double r = cells::norm(pos[i] - pos[j]);
            if (r <= h) neigh_grid.push_back(static_cast<int>(j));
        });
        std::sort(neigh_grid.begin(), neigh_grid.end());
        neigh_grid.erase(std::unique(neigh_grid.begin(), neigh_grid.end()), neigh_grid.end());

        std::vector<int> neigh_brute;
        for (std::size_t j = 0; j < pos.size(); ++j) {
            const double r = cells::norm(pos[i] - pos[j]);
            if (r <= h) neigh_brute.push_back(static_cast<int>(j));
        }

        if (neigh_grid != neigh_brute) {
            all_ok = false;
            break;
        }
    }
    check(all_ok, "phase2 neighbor grid exact set match");

    cells::sph::NeighborGrid grid2;
    grid2.build(pos, h);
    bool deterministic = true;
    for (std::size_t i = 0; i < pos.size(); ++i) {
        std::vector<int> a;
        std::vector<int> b;
        grid.for_neighbor_candidates(pos[i], [&](std::uint32_t j) {
            if (cells::norm(pos[i] - pos[j]) <= h) a.push_back(static_cast<int>(j));
        });
        grid2.for_neighbor_candidates(pos[i], [&](std::uint32_t j) {
            if (cells::norm(pos[i] - pos[j]) <= h) b.push_back(static_cast<int>(j));
        });
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        if (a != b) {
            deterministic = false;
            break;
        }
    }
    check(deterministic, "phase2 neighbor grid deterministic");
}

static void test_phase3_fluid() {
    cells::CellInit init;
    init.icosphere_subdivisions = 2;
    init.fluid_particles = 900;
    init.seed = 77;
    cells::CellSim sim(init);

    for (int i = 0; i < 500; ++i) {
        sim.step(0.005);
    }
    const auto& d = sim.diagnostics();
    check(std::isfinite(d.fluid_density_error_avg), "phase3 finite density avg");
    check(std::isfinite(d.fluid_density_error_max), "phase3 finite density max");
    check(d.fluid_density_error_avg < 0.6, "phase3 density avg <0.6");
}

static void test_phase4_coupling() {
    cells::CellInit init;
    init.icosphere_subdivisions = 2;
    init.fluid_particles = 700;
    init.seed = 99;
    cells::CellSim sim(init);

    int leak_steps = 0;
    for (int i = 0; i < 800; ++i) {
        if ((i % 100) == 0) sim.poke_vertex(0, cells::Vec3(1.1, 0.0, 0.0));
        sim.step(0.005);
        if (sim.diagnostics().penetration_count > 0) leak_steps++;
    }

    const double leak_rate = static_cast<double>(leak_steps) / 800.0;
    check(leak_rate < 0.5, "phase4 coupling leak rate <50%");
    check(sim.diagnostics().max_penetration_depth < 1.0, "phase4 max penetration depth <1.0");

    cells::CellInit ccd_init;
    ccd_init.icosphere_subdivisions = 2;
    ccd_init.fluid_particles = 900;
    ccd_init.seed = 909;
    ccd_init.params.coupling_enable_ccd = true;
    ccd_init.params.coupling_iterations = 2;
    ccd_init.params.coupling_impulse_cap = 0.4;
    cells::CellSim ccd(ccd_init);

    int leak_sum = 0;
    double max_pen = 0.0;
    for (int i = 0; i < 600; ++i) {
        if ((i % 75) == 0) ccd.poke_vertex(0, cells::Vec3(1.6, 0.0, 0.0));
        ccd.step(0.0125);
        leak_sum += std::max(0, ccd.diagnostics().penetration_count);
        max_pen = std::max(max_pen, ccd.diagnostics().max_penetration_depth);
    }
    const double mean_leak_frac = static_cast<double>(leak_sum) / (600.0 * std::max<std::size_t>(1, ccd.particle_count()));
    check(mean_leak_frac < 0.15, "phase4 ccd mean leak fraction <15%");
    const bool ccd_pen_ok = std::isfinite(max_pen) && max_pen < 0.5;
    if (!ccd_pen_ok) {
        std::cout << "  INFO  phase4 ccd max penetration depth value=" << max_pen << std::endl;
    }
    check(ccd_pen_ok, "phase4 ccd max penetration depth <0.5");
}

static void test_phase5_area_bend() {
    cells::CellInit init;
    init.icosphere_subdivisions = 2;
    init.fluid_particles = 400;
    init.params.enable_bending = true;
    init.params.bend_compliance = 8e-4;
    init.params.area_compliance = 1.5e-4;

    cells::CellSim sim(init);
    sim.poke_vertex(0, cells::Vec3(0.6, 0.0, 0.0));
    for (int i = 0; i < 2000; ++i) sim.step(0.005);

    const auto& d = sim.diagnostics();
    check(d.area_drift_pct < 4.0, "phase5 area drift <4%");
    check(d.volume_drift_pct < 3.0, "phase5 volume drift <3%");
    check(std::isfinite(d.bend_residual), "phase5 bend residual finite");

    cells::CellInit long_init;
    long_init.icosphere_subdivisions = 2;
    long_init.fluid_particles = 1000;
    long_init.seed = 515;
    cells::CellSim long_sim(long_init);

    double max_area = 0.0;
    double max_vol = 0.0;
    for (int i = 0; i < 2500; ++i) {
        if ((i % 500) == 0 && i > 0) long_sim.poke_vertex(0, cells::Vec3(0.8, 0.0, 0.0));
        long_sim.step(0.005);
        max_area = std::max(max_area, long_sim.diagnostics().area_drift_pct);
        max_vol = std::max(max_vol, long_sim.diagnostics().volume_drift_pct);
    }
    check(max_area < 3.5, "phase5 long-run max area drift <3.5%");
    check(max_vol < 2.5, "phase5 long-run max volume drift <2.5%");
}

static void test_phase6_active_behavior() {
    cells::CellInit passive_init;
    passive_init.icosphere_subdivisions = 2;
    passive_init.fluid_particles = 200;
    passive_init.seed = 111;

    cells::CellInit active_init = passive_init;
    active_init.params.enable_active_forces = true;
    active_init.params.active_force = 0.35;
    active_init.params.substrate_friction = 1.0;
    active_init.params.substrate_z = -0.75;

    cells::CellSim passive(passive_init);
    cells::CellSim active(active_init);

    const cells::Vec3 c0p = centroid(passive.state().membrane_positions);
    const cells::Vec3 c0a = centroid(active.state().membrane_positions);

    for (int i = 0; i < 1000; ++i) {
        passive.step(0.005);
        active.step(0.005);
    }

    const double dp = cells::norm(centroid(passive.state().membrane_positions) - c0p);
    const double da = cells::norm(centroid(active.state().membrane_positions) - c0a);

    check(da > dp * 1.1, "phase6 active displacement > passive");
    check(active.diagnostics().homeostasis_score >= 0.0, "phase6 homeostasis score valid");
}

static void test_phase6_control_input_identity() {
    cells::CellInit init;
    init.icosphere_subdivisions = 2;
    init.fluid_particles = 256;
    init.seed = 314;
    init.params.enable_active_forces = true;
    init.params.active_force = 0.22;

    cells::CellSim legacy(init);
    cells::CellSim controlled(init);
    const cells::CellControlInput control;

    for (int i = 0; i < 240; ++i) {
        legacy.step(0.005);
        controlled.step(control, 0.005);
    }

    check(equal_vec3_arrays_exact(legacy.state().membrane_positions, controlled.state().membrane_positions),
          "phase6 control api preserves membrane path");
    check(equal_vec3_arrays_exact(legacy.state().particle_positions, controlled.state().particle_positions),
          "phase6 control api preserves fluid path");
}

static void test_phase6_task_harness_determinism() {
    cells::CellTaskConfig cfg;
    cfg.kind = cells::CellTaskKind::DamageRecovery;
    cfg.steps = 120;
    cfg.dt = 0.005;
    cfg.seed = 77;
    cfg.damage_step = 40;
    cfg.damage_fraction = 0.12;
    cfg.init.icosphere_subdivisions = 1;
    cfg.init.fluid_particles = 220;
    cfg.init.seed = 77;
    cfg.init.params.enable_active_forces = true;
    cfg.init.params.active_force = 0.24;
    cfg.init.params.substrate_friction = 0.8;

    auto check_controller = [&](cells::CellTaskController& a,
                                cells::CellTaskController& b,
                                const std::string& label) {
        const auto ep_a = cells::run_cell_task_episode(cfg, a);
        const auto ep_b = cells::run_cell_task_episode(cfg, b);
        check(ep_a.metrics.cumulative_reward == ep_b.metrics.cumulative_reward,
              "phase6 " + label + " cumulative reward deterministic");
        check(ep_a.metrics.final_distance == ep_b.metrics.final_distance,
              "phase6 " + label + " final distance deterministic");
        check(ep_a.metrics.recovery_steps == ep_b.metrics.recovery_steps,
              "phase6 " + label + " recovery deterministic");
        check(equal_task_trace_exact(ep_a.trace, ep_b.trace),
              "phase6 " + label + " trace deterministic");
    };

    cells::HomeostaticCellController homeo_a;
    cells::HomeostaticCellController homeo_b;
    check_controller(homeo_a, homeo_b, "homeostatic task");

    cells::RecurrentBaselineController rnn_a(4);
    cells::RecurrentBaselineController rnn_b(4);
    check_controller(rnn_a, rnn_b, "rnn baseline task");

    cells::GruBaselineController gru_a(4);
    cells::GruBaselineController gru_b(4);
    check_controller(gru_a, gru_b, "gru baseline task");

    cells::LstmBaselineController lstm_a(4);
    cells::LstmBaselineController lstm_b(4);
    check_controller(lstm_a, lstm_b, "lstm baseline task");

    cells::TransformerBaselineController tr_a(12, 6);
    cells::TransformerBaselineController tr_b(12, 6);
    check_controller(tr_a, tr_b, "transformer baseline task");
}

static void test_phase7_cell_module() {
    cells::CellModuleParams p;
    std::vector<double> x(200), y(200);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<double>(i) / 200.0;
        y[i] = 0.2 + 0.6 * x[i];
    }

    const auto cal = cells::calibrate_module_gain_bias(p, x, y, 0.02, 123);
    check(cal.mse_after <= cal.mse_before, "phase7 calibration improves mse");

    cells::CellModule m(p, 77);
    const double o = m.step(0.5, 0.02);
    check(o >= 0.0 && o <= 1.0, "phase7 module bounded output");
}

static void test_phase8_circuit_learning() {
    cells::CellCircuitConfig cfg;
    cfg.input_dim = 16;
    cfg.retina_dim = 12;
    cfg.v1_dim = 10;
    cfg.assoc_dim = 8;
    cfg.decision_dim = 2;
    cfg.lr = 0.01;
    cfg.dt = 0.03;
    cfg.hidden_lr_scale = 0.0;
    cfg.reset_state_per_sample = false;
    cfg.seed = 5;

    cells::CellCircuit model(cfg);

    std::mt19937 rng(5);
    std::normal_distribution<double> nd(0.0, 1.0);

    std::vector<std::vector<double>> xs(512, std::vector<double>(16, 0.0));
    std::vector<int> ys(512, 0);

    for (std::size_t i = 0; i < xs.size(); ++i) {
        for (double& v : xs[i]) v = nd(rng);
        const double a = std::accumulate(xs[i].begin(), xs[i].begin() + 8, 0.0);
        const double b = std::accumulate(xs[i].begin() + 8, xs[i].end(), 0.0);
        ys[i] = (a > b) ? 1 : 0;
    }

    auto eval_acc = [&]() {
        model.reset_state();
        int correct = 0;
        for (std::size_t i = 0; i < xs.size(); ++i) {
            auto s = model.evaluate_one(xs[i], ys[i]);
            if (s.pred == ys[i]) correct++;
        }
        return static_cast<double>(correct) / static_cast<double>(xs.size());
    };

    const double acc0 = eval_acc();
    for (int epoch = 0; epoch < 8; ++epoch) {
        model.reset_state();
        for (std::size_t i = 0; i < xs.size(); ++i) {
            model.train_one(xs[i], ys[i]);
        }
    }
    const double acc1 = eval_acc();
    check(std::isfinite(acc0) && std::isfinite(acc1), "phase8 circuit finite accuracies");
    check(acc1 > 0.40, "phase8 circuit stays above degenerate floor");
    check(std::abs(acc1 - acc0) < 0.20, "phase8 circuit remains numerically stable");
}

int main() {
    std::cout << "=== Cells Deep Tests ===" << std::endl;

    test_phase0_determinism();
    test_phase1_membrane();
    test_phase2_neighbor_grid();
    test_phase3_fluid();
    test_phase4_coupling();
    test_phase5_area_bend();
    test_phase6_active_behavior();
    test_phase6_control_input_identity();
    test_phase6_task_harness_determinism();
    test_phase7_cell_module();
    test_phase8_circuit_learning();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail == 0 ? 0 : 1;
}
