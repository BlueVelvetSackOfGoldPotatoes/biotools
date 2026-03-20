// vivarium_processes.h -- Built-in processes, composers, and utilities
//
// Ported from:
//   vivarium/processes/growth_rate.py   -- GrowthRate
//   vivarium/processes/timeline.py      -- TimelineProcess
//   vivarium/processes/clock.py         -- Clock
//   vivarium/processes/division.py      -- Division, get_divide_update
//   vivarium/processes/divide_condition.py -- DivideCondition
//   vivarium/processes/meta_division.py -- MetaDivision
//   vivarium/processes/injector.py      -- Injector
//   vivarium/processes/remove.py        -- Remove
//   vivarium/composites/toys.py         -- toy processes and composers

#pragma once

#include "vivarium.h"

#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

namespace vivarium {

// ============================================================================
//  GrowthRate -- Exponential growth of biomass (growth_rate.py)
// ============================================================================
class GrowthRate : public Process {
public:
    explicit GrowthRate(const State& params = {})
        : Process(params)
    {
        merge_defaults(State{
            {"default_growth_rate", Value(0.0005)},
            {"default_growth_noise", Value(0.0)},
            {"variables", Value(std::vector<std::string>{"mass"})},
        });
        if (name_ == "Process") name_ = "growth_rate";
    }

    State defaults() const override {
        return {
            {"default_growth_rate", Value(0.0005)},
            {"default_growth_noise", Value(0.0)},
            {"variables", Value(std::vector<std::string>{"mass"})},
        };
    }

    PortSchema ports_schema() const override {
        auto vars = get_variables();
        PortSchema schema;

        std::map<std::string, Schema> var_schemas;
        for (auto& v : vars) {
            var_schemas[v] = {
                {"_default", Value(1.0)},
                {"_divider", Value(std::string("split"))},
                {"_emit", Value(true)},
            };
        }
        schema["variables"] = var_schemas;

        // Rates
        double default_rate = value_as_double(parameters_.at("default_growth_rate"), 0.0005);
        double default_noise = value_as_double(parameters_.at("default_growth_noise"), 0.0);

        std::map<std::string, Schema> rate_schemas;
        (void)vars; // rates schema uses flat structure

        schema["rates"] = {
            {"growth_rate", {{"_default", Value(default_rate)}}},
            {"growth_noise", {{"_default", Value(default_noise)}}},
        };

        return schema;
    }

    Update next_update(double timestep, const State& states) override {
        auto vars = get_variables();
        auto variables = value_as<State>(states.at("variables"));
        auto rates = value_as<State>(states.at("rates"));
        double growth_rate = value_as_double(rates.count("growth_rate") ?
            rates.at("growth_rate") : Value(0.0005));
        double growth_noise = value_as_double(rates.count("growth_noise") ?
            rates.at("growth_noise") : Value(0.0));

        Update var_update;
        static thread_local std::mt19937 rng{std::random_device{}()};

        for (auto& v : vars) {
            double value = value_as_double(variables.count(v) ?
                variables.at(v) : Value(1.0));
            double noise = 0.0;
            if (growth_noise > 0.0) {
                std::normal_distribution<double> dist(0.0, growth_noise);
                noise = dist(rng);
            }
            double delta = value * (std::exp((growth_rate + noise) * timestep) - 1.0);
            var_update[v] = Value(delta);
        }

        return {{"variables", Value(var_update)}};
    }

private:
    std::vector<std::string> get_variables() const {
        try {
            return std::any_cast<std::vector<std::string>>(parameters_.at("variables"));
        } catch (...) {
            return {"mass"};
        }
    }
};


// ============================================================================
//  Clock -- Tracks global time (clock.py)
// ============================================================================
class Clock : public Process {
public:
    explicit Clock(const State& params = {})
        : Process(params)
    {
        merge_defaults(State{{"time_step", Value(1.0)}});
        if (name_ == "Process") name_ = "clock";
    }

    State defaults() const override {
        return {{"time_step", Value(1.0)}};
    }

    PortSchema ports_schema() const override {
        return {
            {"global_time", {
                {"time", {{"_default", Value(0.0)}, {"_updater", Value(std::string("accumulate"))}}},
            }},
        };
    }

    Update next_update(double timestep, const State&) override {
        return {{"global_time", Value(State{{"time", Value(timestep)}})}};
    }
};


// ============================================================================
//  Injector -- Injects substrates at a given rate (injector.py)
// ============================================================================
class Injector : public Process {
public:
    explicit Injector(const State& params = {})
        : Process(params)
    {
        merge_defaults(State{});
        if (name_ == "Process") name_ = "injector";
        if (parameters_.count("substrate_rate_map")) {
            try {
                substrate_rate_map_ = std::any_cast<State>(
                    parameters_.at("substrate_rate_map"));
            } catch (...) {}
        }
    }

    PortSchema ports_schema() const override {
        std::map<std::string, Schema> internal_schema;
        for (auto& [substrate, rate] : substrate_rate_map_) {
            internal_schema[substrate] = {
                {"_default", Value(0.0)},
                {"_emit", Value(true)},
            };
        }
        return {{"internal", internal_schema}};
    }

    Update next_update(double timestep, const State&) override {
        Update internal_update;
        for (auto& [substrate, rate_val] : substrate_rate_map_) {
            double rate = value_as_double(rate_val);
            internal_update[substrate] = Value(timestep * rate);
        }
        return {{"internal", Value(internal_update)}};
    }

private:
    State substrate_rate_map_;
};


// ============================================================================
//  TimelineProcess -- Apply state changes at scheduled times (timeline.py)
// ============================================================================
class TimelineProcess : public Process {
public:
    explicit TimelineProcess(const State& params = {})
        : Process(params)
    {
        merge_defaults(State{
            {"time_step", Value(1.0)},
        });
        if (name_ == "Process") name_ = "timeline";
    }

    State defaults() const override {
        return {
            {"time_step", Value(1.0)},
            {"timeline", Value(std::vector<std::pair<double, State>>())},
        };
    }

    void set_timeline(const std::vector<std::pair<double, State>>& tl) {
        timeline_ = tl;
        // Sort by time
        std::sort(timeline_.begin(), timeline_.end(),
            [](auto& a, auto& b) { return a.first < b.first; });
        initialized_ = true;
    }

    PortSchema ports_schema() const override {
        if (!initialized_) {
            // Try to initialize from parameters
            try {
                auto& tl_val = parameters_.at("timeline");
                auto tl = std::any_cast<std::vector<std::pair<double, State>>>(tl_val);
                const_cast<TimelineProcess*>(this)->set_timeline(tl);
            } catch (...) {}
        }

        // Collect ports from timeline events
        std::set<std::string> port_names;
        port_names.insert("global");
        for (auto& [t, changes] : timeline_) {
            for (auto& [path_key, val] : changes) {
                // Path keys are the first segment
                auto sep = path_key.find('/');
                if (sep != std::string::npos) {
                    port_names.insert(path_key.substr(0, sep));
                } else {
                    port_names.insert(path_key);
                }
            }
        }

        PortSchema schema;
        schema["global"] = {
            {"time", {{"_default", Value(0.0)}, {"_updater", Value(std::string("accumulate"))}}},
        };
        for (auto& port : port_names) {
            if (port != "global") {
                schema[port] = {};  // wildcard port
            }
        }
        return schema;
    }

    Update next_update(double timestep, const State& states) override {
        double time = 0.0;
        try {
            auto global = value_as<State>(states.at("global"));
            time = value_as_double(global.at("time"));
        } catch (...) {}

        Update update;
        update["global"] = Value(Update{{"time", Value(timestep)}});

        // Apply timeline events
        while (!timeline_.empty() && time >= timeline_.front().first) {
            auto& [t, changes] = timeline_.front();
            for (auto& [key, val] : changes) {
                // Parse path: port/var1/var2/...
                auto parts = string_to_path(key, '/');
                if (parts.empty()) continue;

                // Build nested update with _value/_updater set
                Update set_update;
                set_update["_value"] = val;
                set_update["_updater"] = Value(std::string("set"));

                // Build path
                if (parts.size() == 1) {
                    update[parts[0]] = Value(set_update);
                } else {
                    // Nest
                    Update nested = set_update;
                    for (int i = static_cast<int>(parts.size()) - 1; i >= 1; --i) {
                        Update wrapper;
                        wrapper[parts[i]] = Value(nested);
                        nested = wrapper;
                    }
                    if (update.count(parts[0]) && update[parts[0]].type() == typeid(Update)) {
                        auto existing = std::any_cast<Update>(update[parts[0]]);
                        deep_merge(existing, nested);
                        update[parts[0]] = Value(existing);
                    } else {
                        update[parts[0]] = Value(nested);
                    }
                }
            }
            timeline_.erase(timeline_.begin());
        }

        return update;
    }

private:
    mutable std::vector<std::pair<double, State>> timeline_;
    mutable bool initialized_ = false;
};


// ============================================================================
//  DivideCondition -- Step that sets divide flag (divide_condition.py)
// ============================================================================
class DivideCondition : public Deriver {
public:
    explicit DivideCondition(const State& params = {})
        : Deriver(params)
    {
        merge_defaults(State{{"threshold", Value(2.0)}});
        if (name_ == "Process") name_ = "divide_condition";
        threshold_ = value_as_double(parameters_.at("threshold"));
    }

    PortSchema ports_schema() const override {
        return {
            {"variable", {}},
            {"divide", {
                {"_default", {{"_default", Value(false)}}},
            }},
        };
    }

    Update next_update(double, const State& states) override {
        double val = value_as_double(states.count("variable") ?
            states.at("variable") : Value(0.0));
        if (val >= threshold_) {
            return {{"divide", Value(true)}};
        }
        return {};
    }

private:
    double threshold_;
};


// ============================================================================
//  Remove -- Remove an agent when trigger is true (remove.py)
// ============================================================================
class Remove : public Deriver {
public:
    explicit Remove(const State& params = {})
        : Deriver(params)
    {
        merge_defaults(State{{"agent_id", Value(std::string(""))}});
        if (name_ == "Process") name_ = "remove";
        agent_id_ = value_as_string(parameters_.at("agent_id"));
    }

    PortSchema ports_schema() const override {
        return {
            {"trigger", {
                {"_default", {{"_default", Value(false)}, {"_emit", Value(true)}}},
            }},
            {"agents", {}},
        };
    }

    Update next_update(double, const State& states) override {
        bool trigger = false;
        try {
            trigger = value_as_bool(states.at("trigger"));
        } catch (...) {}

        if (trigger && !agent_id_.empty()) {
            return {
                {"agents", Value(Update{
                    {"_delete", Value(std::vector<std::string>{agent_id_})},
                })},
            };
        }
        return {};
    }

private:
    std::string agent_id_;
};


// ============================================================================
//  MetaDivision -- Step that triggers cell division (meta_division.py)
// ============================================================================
class MetaDivision : public Step {
public:
    explicit MetaDivision(const State& params = {})
        : Step(params)
    {
        merge_defaults(State{{"agent_id", Value(std::string(""))}});
        if (name_ == "Process") name_ = "meta_division";
        agent_id_ = value_as_string(parameters_.at("agent_id"));
    }

    PortSchema ports_schema() const override {
        return {
            {"global", {
                {"divide", {
                    {"_default", Value(false)},
                    {"_updater", Value(std::string("set"))},
                    {"_divider", Value(State{
                        {"divider", Value(std::string("set_value"))},
                        {"config", Value(State{{"value", Value(false)}})},
                    })},
                }},
            }},
            {"agents", {}},
        };
    }

    Update next_update(double, const State& states) override {
        bool divide = false;
        try {
            auto global = value_as<State>(states.at("global"));
            divide = value_as_bool(global.at("divide"));
        } catch (...) {}

        if (!divide) return {};

        // Generate daughter IDs
        auto daughter_ids = daughter_phylogeny_id(agent_id_);

        std::vector<State> daughter_updates;
        for (auto& did : daughter_ids) {
            State du;
            du["key"] = Value(did);
            du["initial_state"] = Value(State{});
            daughter_updates.push_back(du);
        }

        return {
            {"agents", Value(Update{
                {"_divide", Value(State{
                    {"mother", Value(agent_id_)},
                    {"daughters", Value(daughter_updates)},
                })},
            })},
        };
    }

    /// Default daughter ID function: appends 0 and 1.
    static std::vector<std::string> daughter_phylogeny_id(const std::string& mother_id) {
        return {mother_id + "0", mother_id + "1"};
    }

private:
    std::string agent_id_;
};


// ============================================================================
//  ExponentialGrowth -- Simple forward-Euler growth (for testing)
// ============================================================================
class ExponentialGrowth : public Process {
public:
    explicit ExponentialGrowth(double rate = 0.1, double dt = 1.0)
        : Process({{"time_step", Value(dt)}})
        , rate_(rate)
    {
        name_ = "ExponentialGrowth";
    }

    PortSchema ports_schema() const override {
        return {
            {"population", {
                {"count", {
                    {"_default", Value(1.0)},
                    {"_emit", Value(true)},
                }},
            }},
        };
    }

    Update next_update(double timestep, const State& states) override {
        auto pop = value_as<State>(states.at("population"));
        double count = value_as_double(pop.at("count"));
        double delta = rate_ * count * timestep;
        return {
            {"population", Value(Update{{"count", Value(delta)}})},
        };
    }

private:
    double rate_;
};


// ============================================================================
//  LinearDecay -- Simple decay process (for testing)
// ============================================================================
class LinearDecay : public Process {
public:
    explicit LinearDecay(double decay_rate = 0.05, double dt = 1.0)
        : Process({{"time_step", Value(dt)}})
        , decay_rate_(decay_rate)
    {
        name_ = "LinearDecay";
    }

    PortSchema ports_schema() const override {
        return {
            {"population", {
                {"count", {
                    {"_default", Value(0.0)},
                    {"_emit", Value(true)},
                }},
            }},
        };
    }

    Update next_update(double timestep, const State& states) override {
        auto pop = value_as<State>(states.at("population"));
        double count = value_as_double(pop.at("count"));
        return {
            {"population", Value(Update{
                {"count", Value(-decay_rate_ * count * timestep)},
            })},
        };
    }

private:
    double decay_rate_;
};


// ============================================================================
//  GlucosePhosphorylation -- Michaelis-Menten kinetics (toys.py / glucose_phosphorylation.py)
// ============================================================================
class GlucosePhosphorylation : public Process {
public:
    GlucosePhosphorylation(double vmax = 1.0, double km = 0.5, double dt = 0.1)
        : Process({{"time_step", Value(dt)}})
        , vmax_(vmax), km_(km)
    {
        name_ = "GlucosePhosphorylation";
    }

    PortSchema ports_schema() const override {
        return {
            {"substrates", {
                {"glucose", {{"_default", Value(10.0)}, {"_emit", Value(true)}}},
                {"ATP",     {{"_default", Value(10.0)}, {"_emit", Value(true)}}},
            }},
            {"products", {
                {"G6P",  {{"_default", Value(0.0)}, {"_emit", Value(true)}}},
                {"ADP",  {{"_default", Value(0.0)}, {"_emit", Value(true)}}},
            }},
        };
    }

    Update next_update(double timestep, const State& states) override {
        auto subs = value_as<State>(states.at("substrates"));
        double glucose = value_as_double(subs.at("glucose"));
        double atp     = value_as_double(subs.at("ATP"));
        double rate = vmax_ * glucose / (km_ + glucose);
        double flux = std::min({rate * timestep, glucose, atp});
        return {
            {"substrates", Value(Update{
                {"glucose", Value(-flux)}, {"ATP", Value(-flux)},
            })},
            {"products", Value(Update{
                {"G6P", Value(flux)}, {"ADP", Value(flux)},
            })},
        };
    }

private:
    double vmax_, km_;
};


// ============================================================================
//  TotalMassStep -- Deriver that computes total mass (toys.py)
// ============================================================================
class TotalMassStep : public Step {
public:
    TotalMassStep() : Step() { name_ = "TotalMassStep"; }

    PortSchema ports_schema() const override {
        return {
            {"species", {
                {"glucose", {{"_default", Value(0.0)}}},
                {"G6P",     {{"_default", Value(0.0)}}},
            }},
            {"derived", {
                {"total_carbon", {
                    {"_default", Value(0.0)},
                    {"_updater", Value(std::string("set"))},
                    {"_emit", Value(true)},
                }},
            }},
        };
    }

    Update next_update(double, const State& states) override {
        auto sp = value_as<State>(states.at("species"));
        double glucose = value_as_double(sp.count("glucose") ? sp.at("glucose") : Value(0.0));
        double g6p     = value_as_double(sp.count("G6P") ? sp.at("G6P") : Value(0.0));
        return {
            {"derived", Value(Update{{"total_carbon", Value(glucose + g6p)}})},
        };
    }
};


// ============================================================================
//  ToyTransport -- Toy process from composites/toys.py
// ============================================================================
class ToyTransport : public Process {
public:
    explicit ToyTransport(const State& params = {})
        : Process(params)
    {
        merge_defaults(State{{"intake_rate", Value(2.0)}});
        if (name_ == "Process") name_ = "toy_transport";
        intake_rate_ = value_as_double(parameters_.at("intake_rate"));
    }

    State defaults() const override {
        return {{"intake_rate", Value(2.0)}};
    }

    PortSchema ports_schema() const override {
        return {
            {"external", {
                {"GLC", {
                    {"_default", Value(0.0)},
                    {"_emit", Value(true)},
                    {"_divider", Value(std::string("set"))},
                }},
            }},
            {"internal", {
                {"GLC", {
                    {"_default", Value(0.0)},
                    {"_emit", Value(true)},
                    {"_divider", Value(std::string("split"))},
                }},
            }},
        };
    }

    Update next_update(double timestep, const State& states) override {
        auto ext = value_as<State>(states.at("external"));
        double ext_glc = value_as_double(ext.count("GLC") ? ext.at("GLC") : Value(0.0));
        double intake = timestep * intake_rate_;

        if (ext_glc >= intake) {
            return {
                {"external", Value(Update{{"GLC", Value(-2.0)}, {"MASS", Value(1.0)}})},
                {"internal", Value(Update{{"GLC", Value(2.0)}})},
            };
        }
        return {};
    }

private:
    double intake_rate_;
};


// ============================================================================
//  ExchangeA -- Simple exchange process (composites/toys.py)
// ============================================================================
class ExchangeA : public Process {
public:
    explicit ExchangeA(const State& params = {})
        : Process(params)
    {
        merge_defaults(State{{"uptake_rate", Value(0.1)}});
        if (name_ == "Process") name_ = "exchange_a";
        uptake_rate_ = value_as_double(parameters_.at("uptake_rate"));
    }

    State defaults() const override {
        return {{"uptake_rate", Value(0.1)}};
    }

    PortSchema ports_schema() const override {
        return {
            {"internal", {
                {"A", {{"_default", Value(0.0)}, {"_emit", Value(true)}}},
            }},
            {"external", {
                {"A", {{"_default", Value(0.0)}, {"_emit", Value(true)}}},
            }},
        };
    }

    Update next_update(double timestep, const State& states) override {
        auto ext = value_as<State>(states.at("external"));
        double ext_a = value_as_double(ext.count("A") ? ext.at("A") : Value(0.0));
        double uptake = std::min(ext_a, uptake_rate_ * timestep);

        return {
            {"internal", Value(Update{{"A", Value(uptake)}})},
            {"external", Value(Update{{"A", Value(-uptake)}})},
        };
    }

private:
    double uptake_rate_;
};


// ============================================================================
//  ToyDeriveVolume -- Derives volume from mass and density (toys.py)
// ============================================================================
class ToyDeriveVolume : public Deriver {
public:
    ToyDeriveVolume() : Deriver() { name_ = "toy_derive_volume"; }

    PortSchema ports_schema() const override {
        return {
            {"compartment", {
                {"MASS",    {{"_default", Value(1.0)}}},
                {"DENSITY", {{"_default", Value(1.0)}}},
                {"VOLUME",  {
                    {"_default", Value(1.0)},
                    {"_updater", Value(std::string("set"))},
                }},
            }},
        };
    }

    Update next_update(double, const State& states) override {
        auto comp = value_as<State>(states.at("compartment"));
        double mass = value_as_double(comp.count("MASS") ? comp.at("MASS") : Value(1.0));
        double density = value_as_double(comp.count("DENSITY") ? comp.at("DENSITY") : Value(1.0));
        double volume = mass / density;
        return {
            {"compartment", Value(Update{{"VOLUME", Value(volume)}})},
        };
    }
};


// ============================================================================
//  AdaptiveGrowth -- Growth with adaptive timestep (for testing)
// ============================================================================
class AdaptiveGrowth : public Process {
public:
    explicit AdaptiveGrowth(double rate = 0.5)
        : Process({{"time_step", Value(1.0)}})
        , rate_(rate)
    {
        name_ = "AdaptiveGrowth";
    }

    PortSchema ports_schema() const override {
        return {
            {"population", {
                {"count", {
                    {"_default", Value(1.0)},
                    {"_emit", Value(true)},
                }},
            }},
        };
    }

    double calculate_timestep(const State& states) const override {
        auto pop = value_as<State>(states.at("population"));
        double count = value_as_double(pop.at("count"));
        double max_dt = 0.1 / (rate_ * std::max(count, 1.0));
        return std::clamp(max_dt, 0.001, 1.0);
    }

    Update next_update(double timestep, const State& states) override {
        auto pop = value_as<State>(states.at("population"));
        double count = value_as_double(pop.at("count"));
        return {
            {"population", Value(Update{{"count", Value(rate_ * count * timestep)}})},
        };
    }

private:
    double rate_;
};


// ============================================================================
//  GrowthComposer -- Composer wiring ExponentialGrowth (for testing)
// ============================================================================
class GrowthComposer : public Composer {
public:
    explicit GrowthComposer(double rate = 0.1)
        : Composer({{"rate", Value(rate)}}) {}

    std::map<std::string, std::shared_ptr<Process>>
    generate_processes(const State& config) const override {
        double rate = value_as_double(config.count("rate") ? config.at("rate") : Value(0.1));
        return {{"growth", std::make_shared<ExponentialGrowth>(rate)}};
    }

    std::map<std::string, Topology>
    generate_topology(const State&) const override {
        return {{"growth", {{"population", {"cells"}}}}};
    }

    State generate_initial_state(const State&) const override {
        return {{"cells", Value(State{{"count", Value(100.0)}})}};
    }
};


// ============================================================================
//  Convenience: register all built-in processes
// ============================================================================
inline void register_builtin_processes() {
    process_registry().register_item("growth_rate",
        [](const State& p) -> std::shared_ptr<Process> { return std::make_shared<GrowthRate>(p); });
    process_registry().register_item("clock",
        [](const State& p) -> std::shared_ptr<Process> { return std::make_shared<Clock>(p); });
    process_registry().register_item("injector",
        [](const State& p) -> std::shared_ptr<Process> { return std::make_shared<Injector>(p); });
    process_registry().register_item("timeline",
        [](const State& p) -> std::shared_ptr<Process> { return std::make_shared<TimelineProcess>(p); });
    process_registry().register_item("divide_condition",
        [](const State& p) -> std::shared_ptr<Process> { return std::make_shared<DivideCondition>(p); });
    process_registry().register_item("remove",
        [](const State& p) -> std::shared_ptr<Process> { return std::make_shared<Remove>(p); });
    process_registry().register_item("meta_division",
        [](const State& p) -> std::shared_ptr<Process> { return std::make_shared<MetaDivision>(p); });
    process_registry().register_item("exponential_growth",
        [](const State& p) -> std::shared_ptr<Process> {
            double rate = value_as_double(p.count("rate") ? p.at("rate") : Value(0.1));
            double dt = value_as_double(p.count("time_step") ? p.at("time_step") : Value(1.0));
            return std::make_shared<ExponentialGrowth>(rate, dt);
        });
    process_registry().register_item("glucose_phosphorylation",
        [](const State& p) -> std::shared_ptr<Process> {
            double vmax = value_as_double(p.count("vmax") ? p.at("vmax") : Value(1.0));
            double km = value_as_double(p.count("km") ? p.at("km") : Value(0.5));
            double dt = value_as_double(p.count("time_step") ? p.at("time_step") : Value(0.1));
            return std::make_shared<GlucosePhosphorylation>(vmax, km, dt);
        });
    process_registry().register_item("exchange_a",
        [](const State& p) -> std::shared_ptr<Process> { return std::make_shared<ExchangeA>(p); });
    process_registry().register_item("toy_transport",
        [](const State& p) -> std::shared_ptr<Process> { return std::make_shared<ToyTransport>(p); });
}

}  // namespace vivarium
