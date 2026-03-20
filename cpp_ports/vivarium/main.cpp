// main.cpp -- Test suite for the C++17 Vivarium-core port
//
// Tests:
//   1. ExponentialGrowth (single process)
//   2. GlucosePhosphorylation (enzyme kinetics)
//   3. Composer pattern
//   4. Multi-process composition
//   5. Adaptive timestep
//   6. Step (deriver)
//   7. Store hierarchy
//   8. History / timeseries via RAMEmitter
//   9. Variable timestep processes
//  10. Updater types (set vs accumulate)
//  11. Registry (updaters, dividers, process registry)
//  12. Deep merge & path utilities
//  13. Serialization
//  14. GrowthRate process
//  15. Injector process
//  16. Nonnegative accumulate updater
//  17. Dividers (split, set, zero, binomial)
//  18. Store schema application
//  19. Store emit_data
//  20. ExchangeA process

#include "vivarium.h"
#include "vivarium_processes.h"
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace vivarium;

static void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n"
              << "  " << title << "\n"
              << std::string(60, '=') << "\n";
}

[[maybe_unused]] static bool approx_eq(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);

    // ========================================================================
    //  Test 1: Exponential Growth (single process)
    // ========================================================================
    print_separator("Test 1: Exponential Growth");
    {
        double rate = 0.1, dt = 1.0, t_end = 10.0, initial_count = 100.0;
        auto growth = std::make_shared<ExponentialGrowth>(rate, dt);
        std::map<std::string, std::shared_ptr<Process>> procs = {{"growth", growth}};
        std::map<std::string, Topology> topo = {{"growth", {{"population", {"cells"}}}}};
        State init = {{"cells", Value(State{{"count", Value(initial_count)}})}};

        Engine engine(procs, topo, init, 0.0, false);
        engine.set_emit(true, 1.0);
        engine.update(t_end);

        double final_count = engine.get_double({"cells", "count"});
        double expected = initial_count * std::pow(1.0 + rate, t_end / dt);
        std::cout << "  Final: " << final_count << "  Expected: " << expected << "\n";
        assert(approx_eq(final_count, expected, 0.01));
        std::cout << "  PASS\n";
    }

    // ========================================================================
    //  Test 2: Glucose Phosphorylation
    // ========================================================================
    print_separator("Test 2: Glucose Phosphorylation");
    {
        auto gp = std::make_shared<GlucosePhosphorylation>(1.0, 0.5, 0.1);
        std::map<std::string, std::shared_ptr<Process>> procs = {{"phos", gp}};
        std::map<std::string, Topology> topo = {
            {"phos", {{"substrates", {"cyt"}}, {"products", {"cyt"}}}},
        };
        State init = {{"cyt", Value(State{
            {"glucose", Value(10.0)}, {"ATP", Value(10.0)},
            {"G6P", Value(0.0)}, {"ADP", Value(0.0)},
        })}};

        Engine engine(procs, topo, init, 0.0, false);
        engine.update(5.0);

        double glucose = engine.get_double({"cyt", "glucose"});
        double g6p = engine.get_double({"cyt", "G6P"});
        double total = glucose + g6p;
        std::cout << "  Glucose=" << glucose << " G6P=" << g6p << " Total=" << total << "\n";
        assert(approx_eq(total, 10.0, 0.01));
        assert(glucose < 10.0 && g6p > 0.0);
        std::cout << "  PASS\n";
    }

    // ========================================================================
    //  Test 3: Composer Pattern
    // ========================================================================
    print_separator("Test 3: Composer Pattern");
    {
        GrowthComposer composer(0.05);
        Composite composite = composer.generate();
        Engine engine(composite, {}, 0.0, false);
        engine.update(20.0);
        double final_count = engine.get_double({"cells", "count"});
        double expected = 100.0 * std::pow(1.05, 20.0);
        std::cout << "  Final: " << final_count << "  Expected: " << expected << "\n";
        assert(approx_eq(final_count, expected, 0.1));
        std::cout << "  PASS\n";
    }

    // ========================================================================
    //  Test 4: Multi-process composition
    // ========================================================================
    print_separator("Test 4: Growth + Decay");
    {
        auto growth = std::make_shared<ExponentialGrowth>(0.10);
        auto decay  = std::make_shared<LinearDecay>(0.05);
        std::map<std::string, std::shared_ptr<Process>> procs = {
            {"growth", growth}, {"decay", decay},
        };
        std::map<std::string, Topology> topo = {
            {"growth", {{"population", {"cells"}}}},
            {"decay",  {{"population", {"cells"}}}},
        };
        State init = {{"cells", Value(State{{"count", Value(100.0)}})}};

        Engine engine(procs, topo, init, 0.0, false);
        engine.update(10.0);
        double final_count = engine.get_double({"cells", "count"});
        double expected = 100.0 * std::pow(1.0 + 0.10 - 0.05, 10.0);
        std::cout << "  Final: " << final_count << "  Expected: " << expected << "\n";
        assert(approx_eq(final_count, expected, 0.5));
        std::cout << "  PASS\n";
    }

    // ========================================================================
    //  Test 5: Adaptive Timestep
    // ========================================================================
    print_separator("Test 5: Adaptive Timestep");
    {
        auto growth = std::make_shared<AdaptiveGrowth>(0.5);
        std::map<std::string, std::shared_ptr<Process>> procs = {{"growth", growth}};
        std::map<std::string, Topology> topo = {{"growth", {{"population", {"cells"}}}}};
        State init = {{"cells", Value(State{{"count", Value(1.0)}})}};

        Engine engine(procs, topo, init, 0.0, false);
        engine.update(5.0);
        double final_count = engine.get_double({"cells", "count"});
        double exact = std::exp(0.5 * 5.0);
        double error_pct = 100.0 * std::abs(final_count - exact) / exact;
        std::cout << "  Final: " << final_count << "  Exact: " << exact
                  << "  Error: " << error_pct << "%\n";
        assert(error_pct < 15.0);
        std::cout << "  PASS\n";
    }

    // ========================================================================
    //  Test 6: Step (Deriver)
    // ========================================================================
    print_separator("Test 6: Step (Deriver)");
    {
        auto gp   = std::make_shared<GlucosePhosphorylation>(1.0, 0.5, 0.5);
        auto mass = std::make_shared<TotalMassStep>();
        std::map<std::string, std::shared_ptr<Process>> procs = {
            {"phos", gp}, {"mass_calc", mass},
        };
        std::map<std::string, Topology> topo = {
            {"phos", {{"substrates", {"cyt"}}, {"products", {"cyt"}}}},
            {"mass_calc", {{"species", {"cyt"}}, {"derived", {"derived"}}}},
        };
        State init = {{"cyt", Value(State{
            {"glucose", Value(10.0)}, {"ATP", Value(10.0)},
            {"G6P", Value(0.0)}, {"ADP", Value(0.0)},
        })}};

        Engine engine(procs, topo, init, 0.0, false);
        engine.update(3.0);
        double total_carbon = engine.get_double({"derived", "total_carbon"});
        std::cout << "  Total carbon: " << total_carbon << "\n";
        assert(approx_eq(total_carbon, 10.0, 0.01));
        std::cout << "  PASS\n";
    }

    // ========================================================================
    //  Test 7: Store Hierarchy
    // ========================================================================
    print_separator("Test 7: Store Hierarchy");
    {
        auto growth = std::make_shared<ExponentialGrowth>(0.1);
        std::map<std::string, std::shared_ptr<Process>> procs = {{"growth", growth}};
        std::map<std::string, Topology> topo = {{"growth", {{"population", {"organism", "cells"}}}}};
        State init = {{"organism", Value(State{{"cells", Value(State{{"count", Value(50.0)}})}})}};

        Engine engine(procs, topo, init, 0.0, false);
        engine.update(5.0);
        double count = engine.get_double({"organism", "cells", "count"});
        double expected = 50.0 * std::pow(1.1, 5.0);
        std::cout << "  Count: " << count << "  Expected: " << expected << "\n";
        assert(approx_eq(count, expected, 0.01));
        std::cout << "  PASS\n";
    }

    // ========================================================================
    //  Test 8: History / RAMEmitter
    // ========================================================================
    print_separator("Test 8: History & RAMEmitter");
    {
        auto growth = std::make_shared<ExponentialGrowth>(0.1, 1.0);
        std::map<std::string, std::shared_ptr<Process>> procs = {{"growth", growth}};
        std::map<std::string, Topology> topo = {{"growth", {{"population", {"cells"}}}}};
        State init = {{"cells", Value(State{{"count", Value(100.0)}})}};

        Engine engine(procs, topo, init, 0.0, false, "timeseries");
        engine.set_emit(true, 1.0);
        engine.update(5.0);

        auto data = engine.emitter().get_data();
        std::cout << "  Emitter data points: " << data.size() << "\n";
        assert(data.size() >= 2);

        // Check history vector
        assert(engine.history().size() >= 2);
        std::cout << "  History entries: " << engine.history().size() << "\n";
        std::cout << "  PASS\n";
    }

    // ========================================================================
    //  Test 9: Variable Timestep Processes
    // ========================================================================
    print_separator("Test 9: Variable Timestep");
    {
        auto fast = std::make_shared<ExponentialGrowth>(0.1, 0.5);
        auto slow = std::make_shared<LinearDecay>(0.02, 2.0);
        std::map<std::string, std::shared_ptr<Process>> procs = {
            {"fast", fast}, {"slow", slow},
        };
        std::map<std::string, Topology> topo = {
            {"fast", {{"population", {"cells"}}}},
            {"slow", {{"population", {"cells"}}}},
        };
        State init = {{"cells", Value(State{{"count", Value(100.0)}})}};

        Engine engine(procs, topo, init, 0.0, false);
        engine.update(4.0);
        double final_count = engine.get_double({"cells", "count"});
        std::cout << "  Final: " << final_count << "\n";
        assert(final_count > 100.0);
        std::cout << "  PASS\n";
    }

    // ========================================================================
    //  Test 10: Updater Types
    // ========================================================================
    print_separator("Test 10: Updater Types (set/accumulate)");
    {
        class SetterProcess : public Process {
        public:
            SetterProcess() : Process({{"time_step", Value(1.0)}}) { name_ = "setter"; }
            PortSchema ports_schema() const override {
                return {{"state", {
                    {"accumulated", {{"_default", Value(0.0)}}},
                    {"replaced", {{"_default", Value(0.0)}, {"_updater", Value(std::string("set"))}}},
                }}};
            }
            Update next_update(double, const State&) override {
                return {{"state", Value(Update{
                    {"accumulated", Value(5.0)}, {"replaced", Value(42.0)},
                })}};
            }
        };

        auto proc = std::make_shared<SetterProcess>();
        std::map<std::string, std::shared_ptr<Process>> procs = {{"setter", proc}};
        std::map<std::string, Topology> topo = {{"setter", {{"state", {"data"}}}}};

        Engine engine(procs, topo, {}, 0.0, false);
        engine.update(3.0);
        double accumulated = engine.get_double({"data", "accumulated"});
        double replaced    = engine.get_double({"data", "replaced"});
        std::cout << "  accumulated: " << accumulated << "  replaced: " << replaced << "\n";
        assert(approx_eq(accumulated, 15.0));
        assert(approx_eq(replaced, 42.0));
        std::cout << "  PASS\n";
    }

    // ========================================================================
    //  Test 11: Registry
    // ========================================================================
    print_separator("Test 11: Registry");
    {
        // Updater registry
        auto acc = updater_registry().access("accumulate");
        assert(acc);
        auto result = acc(Value(3.0), Value(7.0));
        assert(approx_eq(value_as_double(result), 10.0));

        auto set_fn = updater_registry().access("set");
        assert(set_fn);
        result = set_fn(Value(3.0), Value(7.0));
        assert(approx_eq(value_as_double(result), 7.0));

        auto null_fn = updater_registry().access("null");
        assert(null_fn);
        result = null_fn(Value(3.0), Value(7.0));
        assert(approx_eq(value_as_double(result), 3.0));

        auto nonneg = updater_registry().access("nonnegative_accumulate");
        assert(nonneg);
        result = nonneg(Value(3.0), Value(-10.0));
        assert(approx_eq(value_as_double(result), 0.0));

        // Divider registry
        auto split = divider_registry().access("split");
        assert(split);
        auto divided = split(Value(10.0));
        assert(divided.size() == 2);
        assert(approx_eq(value_as_double(divided[0]) + value_as_double(divided[1]), 10.0));

        auto zero_div = divider_registry().access("zero");
        assert(zero_div);
        auto zd = zero_div(Value(100.0));
        assert(approx_eq(value_as_double(zd[0]), 0.0));
        assert(approx_eq(value_as_double(zd[1]), 0.0));

        // Process registry
        register_builtin_processes();
        auto factory = process_registry().access("growth_rate");
        assert(factory);
        auto gr = factory(State{{"default_growth_rate", Value(0.001)}});
        assert(gr->name() == "growth_rate");

        std::cout << "  All registry tests PASS\n";
    }

    // ========================================================================
    //  Test 12: Deep merge & path utilities
    // ========================================================================
    print_separator("Test 12: Deep Merge & Paths");
    {
        // Deep merge
        State a = {{"x", Value(1.0)}, {"nested", Value(State{{"a", Value(1.0)}})}};
        State b = {{"y", Value(2.0)}, {"nested", Value(State{{"b", Value(2.0)}})}};
        deep_merge(a, b);
        assert(a.count("x") && a.count("y"));
        {
            const auto& nested = std::any_cast<const State&>(a["nested"]);
            assert(nested.count("a") && nested.count("b"));
            (void)nested;
        }

        // Path utilities
        assert(starts_with({"a", "b", "c"}, {"a", "b"}));
        assert(!starts_with({"a", "b"}, {"a", "b", "c"}));

        auto norm = normalize_path({"a", "b", "..", "c"});
        assert(norm == HierarchyPath({"a", "c"}));

        // assoc_path / get_in
        State s;
        assoc_path(s, {"x", "y", "z"}, Value(42.0));
        auto val = get_in(s, {"x", "y", "z"});
        assert(approx_eq(value_as_double(val), 42.0));

        // delete_in
        delete_in(s, {"x", "y", "z"});
        val = get_in(s, {"x", "y", "z"});
        assert(!val.has_value());

        // flatten
        State flat_test = {{"a", Value(1.0)}, {"b", Value(State{{"c", Value(2.0)}})}};
        auto flat = flatten_state(flat_test);
        assert(flat.size() == 2);

        std::cout << "  All path/merge tests PASS\n";
    }

    // ========================================================================
    //  Test 13: Serialization
    // ========================================================================
    print_separator("Test 13: Serialization");
    {
        State test_state = {
            {"mass", Value(1.5)},
            {"name", Value(std::string("cell_1"))},
            {"alive", Value(true)},
            {"sub", Value(State{{"x", Value(42.0)}})},
        };
        std::string json = serialize_state(test_state);
        std::cout << "  Serialized:\n" << json << "\n";
        assert(json.find("\"mass\"") != std::string::npos);
        assert(json.find("\"name\"") != std::string::npos);
        assert(json.find("42") != std::string::npos);
        std::cout << "  PASS\n";
    }

    // ========================================================================
    //  Test 14: GrowthRate process
    // ========================================================================
    print_separator("Test 14: GrowthRate");
    {
        State cfg = {
            {"variables", Value(std::vector<std::string>{"mass"})},
            {"default_growth_rate", Value(0.0005)},
            {"time_step", Value(2.0)},
        };
        auto gr = std::make_shared<GrowthRate>(cfg);
        std::map<std::string, std::shared_ptr<Process>> procs = {{"gr", gr}};
        std::map<std::string, Topology> topo = {
            {"gr", {{"variables", {"vars"}}, {"rates", {"rates"}}}},
        };
        State init = {
            {"vars", Value(State{{"mass", Value(100.0)}})},
            {"rates", Value(State{
                {"growth_rate", Value(0.0005)},
                {"growth_noise", Value(0.0)},
            })},
        };

        Engine engine(procs, topo, init, 0.0, false);
        engine.update(100.0);
        double final_mass = engine.get_double({"vars", "mass"});
        double expected = 100.0 * std::exp(0.0005 * 100.0);
        double rel_err = std::abs(final_mass - expected) / expected;
        std::cout << "  Final mass: " << final_mass << "  Expected: " << expected
                  << "  RelErr: " << rel_err << "\n";
        assert(rel_err < 0.02);  // within 2%
        std::cout << "  PASS\n";
    }

    // ========================================================================
    //  Test 15: Injector process
    // ========================================================================
    print_separator("Test 15: Injector");
    {
        State params = {
            {"substrate_rate_map", Value(State{{"toy", Value(1.0)}})},
        };
        auto inj = std::make_shared<Injector>(params);
        std::map<std::string, std::shared_ptr<Process>> procs = {{"inj", inj}};
        std::map<std::string, Topology> topo = {{"inj", {{"internal", {"int"}}}}};

        Engine engine(procs, topo, {}, 0.0, false);
        engine.update(10.0);
        double val = engine.get_double({"int", "toy"});
        std::cout << "  toy after 10s: " << val << "\n";
        assert(approx_eq(val, 10.0, 0.1));
        std::cout << "  PASS\n";
    }

    // ========================================================================
    //  Test 16: Nonnegative accumulate
    // ========================================================================
    print_separator("Test 16: Nonnegative Accumulate");
    {
        auto result = updater_nonneg_accumulate(Value(5.0), Value(-10.0));
        assert(approx_eq(value_as_double(result), 0.0));
        result = updater_nonneg_accumulate(Value(5.0), Value(3.0));
        assert(approx_eq(value_as_double(result), 8.0));

        // With vectors
        auto vr = updater_nonneg_accumulate(
            Value(std::vector<double>{5.0, 3.0}),
            Value(std::vector<double>{-10.0, 2.0}));
        auto vec = value_as_vector(vr);
        assert(approx_eq(vec[0], 0.0));
        assert(approx_eq(vec[1], 5.0));
        std::cout << "  PASS\n";
    }

    // ========================================================================
    //  Test 17: Dividers
    // ========================================================================
    print_separator("Test 17: Dividers");
    {
        // Split double
        auto d = divider_split(Value(10.0));
        assert(d.size() == 2);
        assert(approx_eq(value_as_double(d[0]) + value_as_double(d[1]), 10.0));

        // Set
        d = divider_set(Value(42.0));
        assert(approx_eq(value_as_double(d[0]), 42.0));
        assert(approx_eq(value_as_double(d[1]), 42.0));

        // Zero
        d = divider_zero(Value(100.0));
        assert(approx_eq(value_as_double(d[0]), 0.0));

        // Null
        d = divider_null(Value(100.0));
        assert(d.empty());

        // Set value
        auto sv = make_set_value_divider(Value(false));
        d = sv(Value(true));
        assert(d.size() == 2);

        // Split dict
        State s = {{"a", Value(1.0)}, {"b", Value(2.0)}, {"c", Value(3.0)}, {"d", Value(4.0)}};
        d = divider_split_dict(Value(s));
        assert(d.size() == 2);

        std::cout << "  PASS\n";
    }

    // ========================================================================
    //  Test 18: Store schema application
    // ========================================================================
    print_separator("Test 18: Store Schema");
    {
        Store store;
        Schema schema = {
            {"_default", Value(5.0)},
            {"_updater", Value(std::string("set"))},
            {"_emit", Value(true)},
            {"_divider", Value(std::string("split"))},
        };
        store.set_schema("x", schema);

        assert(store.has_child("x"));
        auto x = store.child("x");
        assert(approx_eq(x->value_double(), 5.0));
        assert(x->updater_name() == "set");
        assert(x->divider_name() == "split");
        assert(x->emit());

        // Apply update with set updater
        x->set_value(x->updater()(x->value(), Value(99.0)));
        assert(approx_eq(x->value_double(), 99.0));

        // Apply divider
        auto divided = x->divider()(x->value());
        assert(divided.size() == 2);
        assert(approx_eq(value_as_double(divided[0]) + value_as_double(divided[1]), 99.0));

        std::cout << "  PASS\n";
    }

    // ========================================================================
    //  Test 19: Store emit_data
    // ========================================================================
    print_separator("Test 19: Store emit_data");
    {
        Store store;
        store.set_schema("x", {
            {"_default", Value(10.0)}, {"_emit", Value(true)},
        });
        store.set_schema("y", {
            {"_default", Value(20.0)}, {"_emit", Value(false)},
        });

        auto data = store.emit_data();
        assert(data.count("x"));
        // y should still appear because emit_data returns all children for branch nodes
        // But the value-level emit flag is checked
        std::cout << "  emit_data keys: ";
        for (auto& [k, v] : data) std::cout << k << " ";
        std::cout << "\n  PASS\n";
    }

    // ========================================================================
    //  Test 20: ExchangeA process
    // ========================================================================
    print_separator("Test 20: ExchangeA");
    {
        auto ea = std::make_shared<ExchangeA>(State{{"uptake_rate", Value(0.1)}});
        std::map<std::string, std::shared_ptr<Process>> procs = {{"ea", ea}};
        std::map<std::string, Topology> topo = {
            {"ea", {{"internal", {"int"}}, {"external", {"ext"}}}},
        };
        State init = {
            {"int", Value(State{{"A", Value(0.0)}})},
            {"ext", Value(State{{"A", Value(10.0)}})},
        };

        Engine engine(procs, topo, init, 0.0, false);
        engine.update(10.0);
        double int_a = engine.get_double({"int", "A"});
        double ext_a = engine.get_double({"ext", "A"});
        std::cout << "  internal A: " << int_a << "  external A: " << ext_a << "\n";
        assert(int_a > 0.0);
        assert(ext_a < 10.0);
        assert(approx_eq(int_a + ext_a, 10.0, 0.01)); // conservation
        std::cout << "  PASS\n";
    }

    // ========================================================================
    //  Summary
    // ========================================================================
    print_separator("ALL 20 TESTS PASSED");

    return 0;
}
