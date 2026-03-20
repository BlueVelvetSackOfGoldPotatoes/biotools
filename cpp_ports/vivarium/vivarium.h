// vivarium.h -- Header-only C++17 port of the Vivarium-core framework
//
// Full port of the vivarium-core Python framework (~18K lines) into C++17.
// Split across:
//   vivarium_types.h    -- types, updaters, dividers, registry, serialization
//   vivarium.h          -- Store, Process, Step, Composer, Emitter, Engine
//   vivarium_processes.h -- built-in processes (growth, division, timeline, etc.)
//
// Ported from: https://github.com/vivarium-collective/vivarium-core

#pragma once

#include "vivarium_types.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace vivarium {

// Forward declarations
class Store;
class Process;
class Step;
class Deriver;
class Engine;
class Composer;
class Emitter;

// ============================================================================
//  Store -- Hierarchical state tree (vivarium/core/store.py, ~1800 lines)
// ============================================================================
// Each Store node holds:
//   - a scalar value (std::any)
//   - child stores (keyed by string)
//   - schema metadata (_default, _updater, _emit, _divider, _subschema, etc.)
//   - topology (if storing a process)
//   - topology_view (cached port->Store* mappings)
// ============================================================================
class Store : public std::enable_shared_from_this<Store> {
public:
    // ---- construction -------------------------------------------------------
    Store() = default;
    explicit Store(Value val) : value_(std::move(val)) {}

    /// Construct from a full nested State/config dict (mirrors Python Store.__init__).
    explicit Store(const State& config, Store* outer = nullptr,
                   const std::string& source = "")
        : outer_(outer)
    {
        apply_config(config, source);
    }

    /// Quick construction from state dict (no schema, just values).
    static std::shared_ptr<Store> from_state(const State& state) {
        auto root = std::make_shared<Store>();
        for (auto& [key, val] : state) {
            if (val.type() == typeid(State)) {
                root->inner_[key] = from_state(std::any_cast<State>(val));
                root->inner_[key]->outer_ = root.get();
            } else {
                auto child = std::make_shared<Store>(val);
                child->outer_ = root.get();
                root->inner_[key] = child;
            }
        }
        return root;
    }

    // ---- value access -------------------------------------------------------
    const Value& value() const { return value_; }
    Value& value_ref() { return value_; }
    void set_value(const Value& v) { value_ = v; }

    double value_double(double fallback = 0.0) const {
        return value_as_double(value_, fallback);
    }

    /// Recursively set values from a State dict.
    void set_value(const State& state) {
        for (auto& [key, val] : state) {
            if (val.type() == typeid(State)) {
                auto c = child(key);
                c->set_value(std::any_cast<const State&>(val));
            } else {
                child(key)->set_value(val);
            }
        }
    }

    // ---- child access -------------------------------------------------------
    bool has_child(const std::string& key) const {
        return inner_.count(key) > 0;
    }

    std::shared_ptr<Store> child(const std::string& key) {
        auto it = inner_.find(key);
        if (it != inner_.end()) return it->second;
        auto c = std::make_shared<Store>();
        c->outer_ = this;
        inner_[key] = c;
        return c;
    }

    std::shared_ptr<Store> child_or_null(const std::string& key) const {
        auto it = inner_.find(key);
        return it != inner_.end() ? it->second : nullptr;
    }

    const std::map<std::string, std::shared_ptr<Store>>& children() const { return inner_; }
    std::map<std::string, std::shared_ptr<Store>>& children_mut() { return inner_; }
    bool is_leaf() const { return inner_.empty(); }

    /// Remove a child.
    void remove_child(const std::string& key) {
        inner_.erase(key);
    }

    // ---- path-based access --------------------------------------------------
    Store* get_path_raw(const HierarchyPath& path) {
        Store* cur = this;
        for (auto& seg : path) {
            if (seg == "..") {
                if (cur->outer_) cur = cur->outer_;
            } else {
                cur = cur->child(seg).get();
            }
        }
        return cur;
    }

    std::shared_ptr<Store> get_path(const HierarchyPath& path) {
        Store* cur = this;
        for (auto& seg : path) {
            if (seg == "..") {
                if (cur->outer_) cur = cur->outer_;
            } else {
                auto it = cur->inner_.find(seg);
                if (it != cur->inner_.end()) {
                    cur = it->second.get();
                } else {
                    return cur->child(seg); // auto-vivify
                }
            }
        }
        // Return shared_ptr; if cur == this, share from this
        // Otherwise find in parent's children
        for (auto& [k, v] : inner_) {
            if (v.get() == cur) return v;
        }
        if (cur == this) return shared_from_this();
        // Walk to find shared_ptr - fallback
        auto result = std::make_shared<Store>();
        *result = *cur;
        return result;
    }

    Value get_value_at(const HierarchyPath& path) const {
        const Store* cur = this;
        for (auto& seg : path) {
            if (seg == "..") {
                if (cur->outer_) cur = cur->outer_;
                continue;
            }
            auto it = cur->inner_.find(seg);
            if (it == cur->inner_.end()) return Value{};
            cur = it->second.get();
        }
        return cur->value_;
    }

    void set_value_at(const HierarchyPath& path, const Value& val) {
        Store* cur = this;
        for (size_t i = 0; i + 1 < path.size(); ++i) {
            cur = cur->child(path[i]).get();
        }
        cur->child(path.back())->set_value(val);
    }

    // ---- schema application -------------------------------------------------
    /// Apply schema keys to this node or its children.
    /// Mirrors Store._apply_config in Python.
    void apply_config(const State& config, const std::string& source = "") {
        State cfg = config;

        // Handle special keys
        if (cfg.count("_default")) {
            default_ = cfg["_default"];
            if (!value_.has_value()) value_ = default_;
            cfg.erase("_default");
            leaf_ = true;
        }
        if (cfg.count("_value")) {
            value_ = cfg["_value"];
            cfg.erase("_value");
            leaf_ = true;
        }
        if (cfg.count("_updater")) {
            auto uname = value_as_string(cfg["_updater"], "accumulate");
            updater_name_ = uname;
            updater_ = get_updater(uname);
            cfg.erase("_updater");
            leaf_ = true;
        }
        if (cfg.count("_emit")) {
            emit_ = value_as_bool(cfg["_emit"], true);
            cfg.erase("_emit");
        }
        if (cfg.count("_divider")) {
            auto& dval = cfg["_divider"];
            if (dval.type() == typeid(std::string)) {
                divider_name_ = std::any_cast<std::string>(dval);
                divider_ = get_divider(divider_name_);
            } else if (dval.type() == typeid(State)) {
                auto& dcfg = std::any_cast<const State&>(dval);
                auto dit = dcfg.find("divider");
                if (dit != dcfg.end()) {
                    divider_name_ = value_as_string(dit->second, "set");
                    if (divider_name_ == "set_value") {
                        auto cit = dcfg.find("config");
                        if (cit != dcfg.end() && cit->second.type() == typeid(State)) {
                            auto& cc = std::any_cast<const State&>(cit->second);
                            auto vit = cc.find("value");
                            if (vit != cc.end()) {
                                divider_ = make_set_value_divider(vit->second);
                            } else {
                                divider_ = get_divider(divider_name_);
                            }
                        } else {
                            divider_ = get_divider(divider_name_);
                        }
                    } else {
                        divider_ = get_divider(divider_name_);
                    }
                }
            }
            cfg.erase("_divider");
            leaf_ = true;
        }
        if (cfg.count("_subschema")) {
            subschema_ = cfg["_subschema"];
            cfg.erase("_subschema");
        }
        if (cfg.count("_subtopology")) {
            subtopology_ = cfg["_subtopology"];
            cfg.erase("_subtopology");
        }
        if (cfg.count("_topology")) {
            topology_ = value_as<Topology>(cfg["_topology"]);
            cfg.erase("_topology");
        }
        if (cfg.count("_flow")) {
            cfg.erase("_flow"); // stored but not used in C++ port yet
        }
        if (cfg.count("_properties")) {
            if (cfg["_properties"].type() == typeid(State)) {
                auto props = std::any_cast<State>(cfg["_properties"]);
                deep_merge(properties_, props);
            }
            cfg.erase("_properties");
        }
        if (cfg.count("_output")) cfg.erase("_output");
        if (cfg.count("*")) {
            // Wildcard subschema
            subschema_ = cfg["*"];
            cfg.erase("*");
        }

        // Remaining keys are child configs
        if (!leaf_ || !cfg.empty()) {
            for (auto& [key, child_cfg] : cfg) {
                if (key.empty() || key[0] == '_') continue;
                if (child_cfg.type() == typeid(State)) {
                    if (!has_child(key)) {
                        auto c = std::make_shared<Store>();
                        c->outer_ = this;
                        inner_[key] = c;
                    }
                    inner_[key]->apply_config(std::any_cast<const State&>(child_cfg), source);
                } else if (child_cfg.type() == typeid(Schema)) {
                    if (!has_child(key)) {
                        auto c = std::make_shared<Store>();
                        c->outer_ = this;
                        inner_[key] = c;
                    }
                    inner_[key]->apply_config(std::any_cast<const Schema&>(child_cfg), source);
                }
            }
        }

        // Ensure all leaf nodes have defaults for updater/divider
        if (leaf_) {
            if (!updater_) {
                updater_ = updater_accumulate;
                updater_name_ = "accumulate";
            }
            if (!divider_) {
                divider_ = divider_set;
                divider_name_ = "set";
            }
        }
    }

    /// Set schema for a single variable under this node.
    void set_schema(const std::string& variable, const Schema& schema) {
        auto c = child(variable);
        for (auto& [k, v] : schema) {
            if (k == "_default") {
                if (!c->value_.has_value()) c->value_ = v;
                c->default_ = v;
                c->leaf_ = true;
            } else if (k == "_updater") {
                c->updater_name_ = value_as_string(v, "accumulate");
                c->updater_ = get_updater(c->updater_name_);
                c->leaf_ = true;
            } else if (k == "_emit") {
                c->emit_ = value_as_bool(v, true);
            } else if (k == "_divider") {
                if (v.type() == typeid(std::string)) {
                    c->divider_name_ = std::any_cast<std::string>(v);
                    c->divider_ = get_divider(c->divider_name_);
                } else if (v.type() == typeid(State)) {
                    auto& dcfg = std::any_cast<const State&>(v);
                    auto dit = dcfg.find("divider");
                    if (dit != dcfg.end()) {
                        c->divider_name_ = value_as_string(dit->second, "set");
                        if (c->divider_name_ == "set_value") {
                            auto cit = dcfg.find("config");
                            if (cit != dcfg.end() && cit->second.type() == typeid(State)) {
                                auto vit = std::any_cast<const State&>(cit->second).find("value");
                                if (vit != std::any_cast<const State&>(cit->second).end()) {
                                    c->divider_ = make_set_value_divider(vit->second);
                                }
                            }
                        } else {
                            c->divider_ = get_divider(c->divider_name_);
                        }
                    }
                }
                c->leaf_ = true;
            }
        }
        if (c->leaf_ && !c->updater_) {
            c->updater_ = updater_accumulate;
            c->updater_name_ = "accumulate";
        }
        if (c->leaf_ && !c->divider_) {
            c->divider_ = divider_set;
            c->divider_name_ = "set";
        }
    }

    /// Apply schemas from a PortSchema via topology.
    void apply_port_schemas(const PortSchema& port_schemas,
                            const Topology& topology) {
        for (auto& [port_name, var_schemas] : port_schemas) {
            auto it = topology.find(port_name);
            if (it == topology.end()) continue;
            Store* target = get_path_raw(it->second);
            for (auto& [var_name, schema] : var_schemas) {
                target->set_schema(var_name, schema);
            }
        }
    }

    // ---- updater/divider access ---------------------------------------------
    UpdaterFn updater() const {
        if (updater_) return updater_;
        return updater_accumulate;
    }
    const std::string& updater_name() const { return updater_name_; }

    DividerFn divider() const {
        if (divider_) return divider_;
        return divider_set;
    }
    const std::string& divider_name() const { return divider_name_; }

    // ---- apply_update -------------------------------------------------------
    /// Apply an Update dict to this store.
    /// Handles special keys: _updater override, _value, _add, _delete, _divide.
    /// Returns true if topology views need to be rebuilt.
    bool apply_update(const Update& update) {
        bool view_expire = false;
        for (auto& [key, delta] : update) {
            // Handle in-update updater override
            if (delta.type() == typeid(State) || delta.type() == typeid(Update)) {
                State sub;
                try { sub = std::any_cast<State>(delta); }
                catch (...) { sub = std::any_cast<Update>(delta); }

                // Check for special _updater key in the update
                auto upd_it = sub.find("_updater");
                if (upd_it != sub.end() && sub.count("_value")) {
                    auto updater_fn = get_updater(value_as_string(upd_it->second, "accumulate"));
                    auto c = child(key);
                    c->value_ = updater_fn(c->value_, sub["_value"]);
                    continue;
                }

                // Check for _add/_delete (agent management)
                if (sub.count("_add") || sub.count("_delete") || sub.count("_divide")) {
                    view_expire |= apply_special_update(key, sub);
                    continue;
                }

                // Recurse
                view_expire |= child(key)->apply_update(sub);
            } else {
                auto c = child(key);
                c->value_ = c->updater()(c->value_, delta);
            }
        }
        return view_expire;
    }

    /// Handle _add, _delete, _divide special updates.
    bool apply_special_update(const std::string& key, const State& update) {
        bool view_expire = false;
        auto c = child(key);

        if (update.count("_delete")) {
            try {
                auto dels = std::any_cast<std::vector<std::string>>(update.at("_delete"));
                for (auto& dk : dels) {
                    c->remove_child(dk);
                    view_expire = true;
                }
            } catch (...) {}
        }

        if (update.count("_add")) {
            try {
                auto adds = std::any_cast<std::vector<State>>(update.at("_add"));
                for (auto& add_entry : adds) {
                    auto k = value_as_string(add_entry.at("key"));
                    auto new_child = std::make_shared<Store>();
                    new_child->outer_ = c.get();
                    if (add_entry.count("state") && add_entry.at("state").type() == typeid(State)) {
                        new_child->set_value(std::any_cast<const State&>(add_entry.at("state")));
                    }
                    c->inner_[k] = new_child;
                    view_expire = true;
                }
            } catch (...) {}
        }

        if (update.count("_divide")) {
            view_expire |= apply_divide(key, update.at("_divide"));
        }

        return view_expire;
    }

    /// Apply division: split mother into two daughters.
    bool apply_divide(const std::string& agents_key, const Value& divide_config) {
        if (divide_config.type() != typeid(State)) return false;
        auto& dcfg = std::any_cast<const State&>(divide_config);

        auto mother_id = value_as_string(dcfg.at("mother"));
        auto agents = child(agents_key);
        auto mother_store = agents->child_or_null(mother_id);
        if (!mother_store) return false;

        // Get mother's state for division
        State mother_state = mother_store->get_state_for_divide();

        // Get daughter configs
        std::vector<State> daughter_updates;
        try {
            daughter_updates = std::any_cast<std::vector<State>>(dcfg.at("daughters"));
        } catch (...) { return false; }

        // Divide mother state for each daughter
        std::vector<State> daughter_states;
        for (size_t i = 0; i < daughter_updates.size() && i < 2; ++i) {
            daughter_states.push_back(divide_state(mother_state, i));
        }

        // Remove mother
        agents->remove_child(mother_id);

        // Create daughters
        for (size_t i = 0; i < daughter_updates.size(); ++i) {
            auto& du = daughter_updates[i];
            auto daughter_id = value_as_string(du.at("key"));
            auto daughter = std::make_shared<Store>();
            daughter->outer_ = agents.get();

            // Apply divided state
            if (i < daughter_states.size()) {
                daughter->set_value(daughter_states[i]);
            }

            // Apply initial_state overrides from daughter config
            if (du.count("initial_state") && du.at("initial_state").type() == typeid(State)) {
                auto& init = std::any_cast<const State&>(du.at("initial_state"));
                daughter->set_value(init);
            }

            agents->inner_[daughter_id] = daughter;
        }

        return true; // topology views expired
    }

    /// Get state suitable for division (recursive, applies dividers).
    State get_state_for_divide() const {
        State s;
        for (auto& [name, cstore] : inner_) {
            if (cstore->is_leaf()) {
                s[name] = cstore->value_;
            } else {
                s[name] = Value(cstore->get_state_for_divide());
            }
        }
        return s;
    }

    /// Divide a mother's state into daughter state at index (0 or 1).
    State divide_state(const State& mother_state, size_t daughter_index) const {
        State daughter;
        for (auto& [key, val] : mother_state) {
            auto child_store = child_or_null(key);
            if (child_store) {
                auto div = child_store->divider();
                if (div) {
                    auto divided = div(val);
                    if (!divided.empty() && daughter_index < divided.size()) {
                        daughter[key] = divided[daughter_index];
                    }
                } else {
                    daughter[key] = val;
                }
            } else {
                // No schema, use set divider
                daughter[key] = val;
            }
        }
        return daughter;
    }

    // ---- topology view ------------------------------------------------------
    /// The topology for a process stored at this node.
    Topology topology_;

    /// Cached topology view: port_name -> Store* for fast access.
    std::map<std::string, Store*> topology_view_;

    /// Build topology views for this node and all descendants.
    void build_topology_views() {
        if (!topology_.empty() && outer_) {
            for (auto& [port_name, path] : topology_) {
                Store* target = outer_;
                for (auto& seg : path) {
                    if (seg == "..") {
                        if (target->outer_) target = target->outer_;
                    } else {
                        target = target->child(seg).get();
                    }
                }
                topology_view_[port_name] = target;
            }
        }
        for (auto& [key, child_store] : inner_) {
            child_store->build_topology_views();
        }
    }

    /// Collect the current state a process sees through its topology_view.
    State get_topology_view_state() const {
        State view;
        for (auto& [port_name, store] : topology_view_) {
            State port_state;
            for (auto& [cname, cstore] : store->inner_) {
                if (cstore->is_leaf()) {
                    port_state[cname] = cstore->value_;
                } else {
                    port_state[cname] = Value(cstore->get_state());
                }
            }
            view[port_name] = Value(port_state);
        }
        return view;
    }

    /// topology_view from a Topology (legacy interface).
    State topology_view(const Topology& topology) const {
        State view;
        for (auto& [port_name, path] : topology) {
            const Store* target = this;
            bool valid = true;
            for (auto& seg : path) {
                if (seg == "..") {
                    if (target->outer_) target = target->outer_;
                    continue;
                }
                auto it = target->inner_.find(seg);
                if (it == target->inner_.end()) { valid = false; break; }
                target = it->second.get();
            }
            if (!valid) { view[port_name] = State{}; continue; }
            State port_state;
            for (auto& [cname, cstore] : target->inner_) {
                if (cstore->is_leaf()) port_state[cname] = cstore->value_;
                else port_state[cname] = Value(cstore->get_state());
            }
            view[port_name] = Value(port_state);
        }
        return view;
    }

    // ---- emit ---------------------------------------------------------------
    bool emit() const { return emit_; }
    void set_emit(bool v) { emit_ = v; }

    void set_emit_recursive(bool v) {
        emit_ = v;
        for (auto& [k, c] : inner_) c->set_emit_recursive(v);
    }

    State emit_data() const {
        State data;
        if (!inner_.empty()) {
            for (auto& [name, cstore] : inner_) {
                State sub = cstore->emit_data();
                if (!sub.empty()) data[name] = Value(sub);
                else if (cstore->is_leaf() && cstore->emit_) {
                    data[name] = cstore->value_;
                }
            }
            return data;
        }
        if (emit_) {
            data["_value"] = value_;
        }
        return data;
    }

    // ---- get_state ----------------------------------------------------------
    State get_state() const {
        State s;
        for (auto& [name, cstore] : inner_) {
            if (cstore->is_leaf()) {
                s[name] = cstore->value_;
            } else {
                s[name] = Value(cstore->get_state());
            }
        }
        return s;
    }

    /// Get value: if leaf return value_, if branch return state dict.
    Value get_value() const {
        if (inner_.empty()) return value_;
        return Value(get_state());
    }

    /// Get the config dict that could recreate this node.
    State get_config() const {
        State config;
        if (!inner_.empty()) {
            for (auto& [key, c] : inner_) {
                config[key] = Value(c->get_config());
            }
        } else {
            config["_default"] = default_;
            config["_value"] = value_;
            if (!updater_name_.empty()) config["_updater"] = Value(updater_name_);
            if (!divider_name_.empty()) config["_divider"] = Value(divider_name_);
            config["_emit"] = Value(emit_);
        }
        if (!properties_.empty()) config["_properties"] = Value(properties_);
        return config;
    }

    // ---- hierarchy navigation -----------------------------------------------
    Store* outer() const { return outer_; }

    Store* top() {
        Store* cur = this;
        while (cur->outer_) cur = cur->outer_;
        return cur;
    }

    HierarchyPath path_for() const {
        if (!outer_) return {};
        for (auto& [k, v] : outer_->inner_) {
            if (v.get() == this) {
                auto parent_path = outer_->path_for();
                parent_path.push_back(k);
                return parent_path;
            }
        }
        return {};
    }

    // ---- printing -----------------------------------------------------------
    void print(int indent = 0) const {
        std::string pad(indent * 2, ' ');
        for (auto& [name, cstore] : inner_) {
            if (cstore->is_leaf()) {
                std::cout << pad << name << ": " << value_to_string(cstore->value_) << "\n";
            } else {
                std::cout << pad << name << "/\n";
                cstore->print(indent + 1);
            }
        }
    }

    /// Pretty print with types.
    void print_verbose(int indent = 0) const {
        std::string pad(indent * 2, ' ');
        for (auto& [name, cstore] : inner_) {
            if (cstore->is_leaf()) {
                std::cout << pad << name << " = " << value_to_string(cstore->value_)
                          << " [updater=" << cstore->updater_name_
                          << " divider=" << cstore->divider_name_
                          << " emit=" << (cstore->emit_ ? "T" : "F") << "]\n";
            } else {
                std::cout << pad << name << "/\n";
                cstore->print_verbose(indent + 1);
            }
        }
    }

    // ---- data members -------------------------------------------------------
    Value                                         value_;
    std::map<std::string, std::shared_ptr<Store>> inner_;
    Store*                                        outer_ = nullptr;
    Value                                         default_;
    std::string                                   updater_name_ = "accumulate";
    UpdaterFn                                     updater_;
    std::string                                   divider_name_ = "set";
    DividerFn                                     divider_;
    bool                                          emit_ = true;
    bool                                          leaf_ = false;
    State                                         properties_;
    Value                                         subschema_;
    Value                                         subtopology_;
};


/// Generate a Store from processes, topology, and initial state.
/// Mirrors vivarium.core.store.generate_state.
inline std::shared_ptr<Store> generate_state(
    const std::map<std::string, std::shared_ptr<Process>>& processes,
    const std::map<std::string, Topology>& topology,
    const State& initial_state);


// ============================================================================
//  Process -- Base class (vivarium/core/process.py)
// ============================================================================
class Process {
public:
    static constexpr double DEFAULT_TIME_STEP = 1.0;

    explicit Process(const State& parameters = {})
        : parameters_(parameters)
    {
        // Note: defaults() is virtual but called from base ctor,
        // so it returns Process::defaults() = {}.
        // Subclasses must call merge_defaults() in their own ctors.
        apply_parameters();
    }

    /// Call this from subclass constructors after setting parameters_.
    void apply_parameters() {
        if (parameters_.count("name"))
            name_ = value_as_string(parameters_["name"], name_);
        if (parameters_.count("time_step"))
            timestep_ = value_as_double(parameters_["time_step"], DEFAULT_TIME_STEP);
        if (parameters_.count("timestep"))
            timestep_ = value_as_double(parameters_["timestep"], DEFAULT_TIME_STEP);
        if (parameters_.count("_parallel"))
            parallel_ = value_as_bool(parameters_["_parallel"]);
    }

    /// Merge class defaults with parameters (call from subclass ctor).
    void merge_defaults(const State& class_defaults) {
        State merged = deep_copy(class_defaults);
        deep_merge(merged, parameters_);
        parameters_ = merged;
        apply_parameters();
    }

    virtual ~Process() = default;

    // ---- class defaults (subclasses set via override) -----------------------
    virtual State defaults() const { return {}; }

    // ---- interface that subclasses MUST implement ---------------------------
    virtual PortSchema ports_schema() const = 0;
    virtual Update next_update(double timestep, const State& states) = 0;

    // ---- interface that subclasses MAY override -----------------------------
    virtual double calculate_timestep(const State&) const { return timestep_; }
    virtual State initial_state(const State* config = nullptr) const { (void)config; return {}; }
    virtual bool is_step() const { return false; }
    virtual bool is_deriver() const { return false; }
    virtual bool update_condition(double, const State&) const { return true; }

    // ---- command interface (for parallel processes) -------------------------
    virtual void send_command(const std::string& cmd,
                              const std::pair<double, State>& args) {
        if (cmd == "next_update") {
            command_result_ = next_update(args.first, args.second);
        }
    }

    virtual Update get_command_result() {
        return command_result_;
    }

    // ---- schema override merging -------------------------------------------
    void merge_overrides(const Schema& overrides) {
        deep_merge(schema_override_, overrides);
    }

    // ---- generate composite from a single process --------------------------
    struct SingleComposite {
        std::map<std::string, std::shared_ptr<Process>> processes;
        std::map<std::string, Topology> topology;
        State state;
    };

    /// Generate a minimal Composite wrapping this process.
    /// Mirrors Process.generate() in Python.
    virtual SingleComposite generate() {
        auto schema = ports_schema();
        Topology topo;
        for (auto& [port, vars] : schema) {
            topo[port] = {port};
        }
        SingleComposite comp;
        // We need shared_from_this but Process doesn't inherit enable_shared_from_this.
        // Instead, just don't use this for engines directly -- use the process pointer.
        // This is a convenience method.
        comp.topology[name_] = topo;
        return comp;
    }

    // ---- accessors ----------------------------------------------------------
    const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }
    double timestep() const { return timestep_; }
    const State& parameters() const { return parameters_; }
    bool parallel() const { return parallel_; }

    std::vector<std::string> ports() const {
        std::vector<std::string> out;
        for (auto& [p, _] : ports_schema()) out.push_back(p);
        return out;
    }

    State default_state() const {
        State ds;
        for (auto& [port, vars] : ports_schema()) {
            State port_state;
            for (auto& [var, schema] : vars) {
                auto dit = schema.find("_default");
                if (dit != schema.end()) port_state[var] = dit->second;
            }
            if (!port_state.empty()) ds[port] = Value(port_state);
        }
        return ds;
    }

    std::string name_ = "Process";
    double      timestep_ = DEFAULT_TIME_STEP;
    State       parameters_;
    bool        parallel_ = false;
    Schema      schema_override_;
    Update      command_result_;
};


// ============================================================================
//  Step -- A Process that runs every engine tick
// ============================================================================
class Step : public Process {
public:
    using Process::Process;
    bool is_step() const override { return true; }
};

// ============================================================================
//  Deriver -- Legacy step (backward compat)
// ============================================================================
class Deriver : public Step {
public:
    using Step::Step;
    bool is_deriver() const override { return true; }
};


// ============================================================================
//  Composite -- A bundle of processes + topology + state
// ============================================================================
struct Composite {
    std::map<std::string, std::shared_ptr<Process>> processes;
    std::map<std::string, std::shared_ptr<Process>> steps;
    std::map<std::string, Topology>                 topology;
    Flow                                            flow;
    State                                           state;

    /// Merge another composite into this one.
    void merge(const Composite& other,
               const std::map<std::string, std::shared_ptr<Process>>& extra_procs = {},
               const std::map<std::string, Topology>& extra_topo = {},
               const HierarchyPath& /*path*/ = {}) {
        for (auto& [k, v] : other.processes) processes[k] = v;
        for (auto& [k, v] : other.steps)     steps[k] = v;
        for (auto& [k, v] : other.topology)  topology[k] = v;
        for (auto& [k, v] : other.flow)      flow[k] = v;
        deep_merge(state, other.state);
        for (auto& [k, v] : extra_procs) processes[k] = v;
        for (auto& [k, v] : extra_topo)  topology[k] = v;
    }

    /// Merge individual components.
    void merge(const std::map<std::string, std::shared_ptr<Process>>& procs,
               const std::map<std::string, Topology>& topo,
               const State& st = {}) {
        for (auto& [k, v] : procs) processes[k] = v;
        for (auto& [k, v] : topo)  topology[k] = v;
        deep_merge(state, st);
    }

    /// Access by string key (mimics Python's dict-like Composite).
    State& operator[](const std::string& key) {
        if (key == "state") return state;
        static State empty;
        return empty;
    }
};


// ============================================================================
//  Composer -- Factory that generates Composites (vivarium/core/composer.py)
// ============================================================================
class Composer {
public:
    explicit Composer(const State& config = {}) : config_(config) {}
    virtual ~Composer() = default;

    virtual std::map<std::string, std::shared_ptr<Process>>
    generate_processes(const State& config) const = 0;

    virtual std::map<std::string, Topology>
    generate_topology(const State& config) const = 0;

    virtual std::map<std::string, std::shared_ptr<Process>>
    generate_steps(const State& config) const { (void)config; return {}; }

    virtual Flow generate_flow(const State& config) const { (void)config; return {}; }

    virtual State generate_initial_state(const State& config) const { (void)config; return {}; }

    Composite generate(const State* config_override = nullptr,
                       const HierarchyPath& /*path*/ = {}) const {
        State cfg = config_;
        if (config_override) deep_merge(cfg, *config_override);

        Composite comp;
        comp.processes = generate_processes(cfg);
        comp.steps     = generate_steps(cfg);
        comp.topology  = generate_topology(cfg);
        comp.flow      = generate_flow(cfg);
        comp.state     = generate_initial_state(cfg);
        return comp;
    }

    State initial_state(const State* config = nullptr) const {
        auto comp = generate(config);
        return comp.state;
    }

    const State& config() const { return config_; }

protected:
    State config_;
};


// ============================================================================
//  Emitter -- Data emission (vivarium/core/emitter.py)
// ============================================================================

/// Base emitter: prints to stdout.
class Emitter {
public:
    explicit Emitter(const State& config = {}) : config_(config) {}
    virtual ~Emitter() = default;

    virtual void emit(const State& data) {
        // Default: print emitter
        auto table_it = data.find("table");
        if (table_it != data.end()) {
            std::cout << "[" << value_as_string(table_it->second) << "] ";
        }
        auto data_it = data.find("data");
        if (data_it != data.end() && data_it->second.type() == typeid(State)) {
            auto& d = std::any_cast<const State&>(data_it->second);
            auto time_it = d.find("time");
            if (time_it != d.end()) {
                std::cout << "t=" << value_as_double(time_it->second);
            }
        }
        std::cout << "\n";
    }

    /// Get data (for subclasses that store it).
    virtual std::map<double, State> get_data(
        const std::vector<HierarchyPath>& query = {}) const {
        (void)query;
        return {};
    }

    /// Get deserialized data.
    virtual std::map<double, State> get_data_deserialized(
        const std::vector<HierarchyPath>& query = {}) const {
        return get_data(query);
    }

    /// Get as embedded timeseries: { "var_name": [v0, v1, ...], "time": [t0, t1, ...] }.
    virtual State get_timeseries(const std::vector<HierarchyPath>& query = {}) const {
        auto data = get_data(query);
        State timeseries;
        std::vector<double> times;

        for (auto& [t, snapshot] : data) {
            times.push_back(t);
            // Flatten snapshot into timeseries
            accumulate_timeseries(timeseries, snapshot);
        }

        // Convert accumulated vectors
        timeseries["time"] = Value(times);
        return timeseries;
    }

protected:
    State config_;

private:
    void accumulate_timeseries(State& ts, const State& snapshot,
                               const std::string& prefix = "") const {
        for (auto& [key, val] : snapshot) {
            std::string full_key = prefix.empty() ? key : prefix + "." + key;
            if (val.type() == typeid(State)) {
                auto& sub = std::any_cast<const State&>(val);
                accumulate_timeseries(ts, sub, full_key);
            } else {
                // Append to vector
                auto it = ts.find(full_key);
                if (it == ts.end()) {
                    ts[full_key] = Value(std::vector<double>{value_as_double(val)});
                } else {
                    try {
                        auto& vec = std::any_cast<std::vector<double>&>(it->second);
                        vec.push_back(value_as_double(val));
                    } catch (...) {}
                }
            }
        }
    }
};

/// NullEmitter: emits nothing.
class NullEmitter : public Emitter {
public:
    using Emitter::Emitter;
    void emit(const State&) override {}
};

/// RAMEmitter: accumulates timeseries in memory (mirrors Python RAMEmitter).
class RAMEmitter : public Emitter {
public:
    explicit RAMEmitter(const State& config = {})
        : Emitter(config) {}

    void emit(const State& data) override {
        auto table_it = data.find("table");
        if (table_it == data.end()) return;
        auto table = value_as_string(table_it->second);

        if (table == "history") {
            auto data_it = data.find("data");
            if (data_it == data.end()) return;
            if (data_it->second.type() != typeid(State)) return;

            State emit_data = std::any_cast<State>(data_it->second);
            double time = 0.0;
            auto time_it = emit_data.find("time");
            if (time_it != emit_data.end()) {
                time = value_as_double(time_it->second);
                emit_data.erase("time");
            }
            saved_data_[time] = emit_data;
        }
    }

    std::map<double, State> get_data(
        const std::vector<HierarchyPath>& query = {}) const override {
        if (query.empty()) return saved_data_;

        std::map<double, State> result;
        for (auto& [t, data] : saved_data_) {
            State filtered;
            for (auto& path : query) {
                auto val = vivarium::get_in(data, path);
                if (val.has_value()) {
                    vivarium::assoc_path(filtered, path, val);
                }
            }
            result[t] = filtered;
        }
        return result;
    }

    State get_timeseries(const std::vector<HierarchyPath>& query = {}) const override {
        auto data = get_data(query);
        State timeseries;
        std::vector<double> times;

        // Collect all keys across all timepoints
        for (auto& [t, snapshot] : data) {
            times.push_back(t);
        }

        // Build embedded timeseries
        State embedded;
        for (auto& [t, snapshot] : data) {
            embed_in_timeseries(embedded, snapshot);
        }
        embedded["time"] = Value(times);
        return embedded;
    }

protected:
    std::map<double, State> saved_data_;

private:
    void embed_in_timeseries(State& ts, const State& snapshot) const {
        for (auto& [key, val] : snapshot) {
            if (val.type() == typeid(State)) {
                if (!ts.count(key) || ts[key].type() != typeid(State)) {
                    ts[key] = Value(State{});
                }
                auto& sub_ts = std::any_cast<State&>(ts[key]);
                embed_in_timeseries(sub_ts, std::any_cast<const State&>(val));
            } else {
                if (!ts.count(key)) {
                    ts[key] = Value(std::vector<Value>{});
                }
                try {
                    auto& vec = std::any_cast<std::vector<Value>&>(ts[key]);
                    vec.push_back(val);
                } catch (...) {}
            }
        }
    }
};

/// FileEmitter: writes timeseries data to a CSV file.
class FileEmitter : public RAMEmitter {
public:
    explicit FileEmitter(const State& config = {})
        : RAMEmitter(config) {
        auto fn_it = config.find("filename");
        if (fn_it != config.end())
            filename_ = value_as_string(fn_it->second, "vivarium_output.csv");
        else
            filename_ = "vivarium_output.csv";
    }

    ~FileEmitter() { flush(); }

    void emit(const State& data) override {
        RAMEmitter::emit(data);
    }

    void flush() {
        std::ofstream ofs(filename_);
        if (!ofs.is_open()) return;

        auto ts = get_timeseries();
        // Extract time vector
        std::vector<double> times;
        auto time_it = ts.find("time");
        if (time_it != ts.end()) {
            try {
                times = std::any_cast<std::vector<double>>(time_it->second);
            } catch (...) {}
        }

        // Collect all flat keys
        std::vector<std::string> keys;
        std::map<std::string, std::vector<Value>> columns;
        collect_flat_columns("", ts, keys, columns);

        // Write header
        ofs << "time";
        for (auto& k : keys) ofs << "," << k;
        ofs << "\n";

        // Write rows
        for (size_t i = 0; i < times.size(); i++) {
            ofs << times[i];
            for (auto& k : keys) {
                auto& col = columns[k];
                if (i < col.size()) {
                    try { ofs << "," << value_as_double(col[i]); }
                    catch (...) { ofs << ","; }
                } else {
                    ofs << ",";
                }
            }
            ofs << "\n";
        }
    }

private:
    std::string filename_;

    void collect_flat_columns(const std::string& prefix, const State& ts,
                               std::vector<std::string>& keys,
                               std::map<std::string, std::vector<Value>>& columns) {
        for (auto& [key, val] : ts) {
            if (key == "time") continue;
            std::string full_key = prefix.empty() ? key : prefix + "." + key;
            if (val.type() == typeid(State)) {
                collect_flat_columns(full_key, std::any_cast<const State&>(val), keys, columns);
            } else if (val.type() == typeid(std::vector<Value>)) {
                keys.push_back(full_key);
                columns[full_key] = std::any_cast<std::vector<Value>>(val);
            }
        }
    }
};

/// Create an emitter by type name.
inline std::shared_ptr<Emitter> get_emitter(const State& config) {
    auto type = value_as_string(config.count("type") ?
        config.at("type") : Value(std::string("timeseries")), "timeseries");

    if (type == "null")       return std::make_shared<NullEmitter>(config);
    if (type == "timeseries") return std::make_shared<RAMEmitter>(config);
    if (type == "ram")        return std::make_shared<RAMEmitter>(config);
    if (type == "file")       return std::make_shared<FileEmitter>(config);
    // Default: print emitter
    return std::make_shared<Emitter>(config);
}


// ============================================================================
//  Engine -- Simulation runner (vivarium/core/engine.py)
// ============================================================================
class Engine {
public:
    // -- construct from a Composite -------------------------------------------
    explicit Engine(const Composite& composite,
                    const State& initial_state = {},
                    double initial_global_time = 0.0,
                    bool display_info = true,
                    const std::string& emitter_type = "timeseries",
                    double emit_step = 1.0,
                    bool progress_bar = false,
                    const std::string& experiment_id = "")
        : global_time_(initial_global_time)
        , display_info_(display_info)
        , emit_step_(emit_step)
        , progress_bar_(progress_bar)
        , experiment_id_(experiment_id)
    {
        // Merge processes and steps
        auto all_procs = composite.processes;
        for (auto& [k, v] : composite.steps) all_procs[k] = v;

        State merged_state = composite.state;
        deep_merge(merged_state, initial_state);

        // Emitter
        State emitter_config;
        emitter_config["type"] = Value(emitter_type);
        emitter_ = get_emitter(emitter_config);

        init(all_procs, composite.topology, merged_state);
    }

    // -- construct from processes + topology directly -------------------------
    Engine(std::map<std::string, std::shared_ptr<Process>> processes,
           std::map<std::string, Topology> topology,
           const State& initial_state = {},
           double initial_global_time = 0.0,
           bool display_info = true,
           const std::string& emitter_type = "timeseries",
           double emit_step = 1.0)
        : global_time_(initial_global_time)
        , display_info_(display_info)
        , emit_step_(emit_step)
    {
        State emitter_config;
        emitter_config["type"] = Value(emitter_type);
        emitter_ = get_emitter(emitter_config);

        init(std::move(processes), std::move(topology), initial_state);
    }

    // -- run ------------------------------------------------------------------
    void update(double interval) {
        auto wall_start = std::chrono::steady_clock::now();
        run_for(interval, true);
        auto wall_end = std::chrono::steady_clock::now();
        if (display_info_) {
            double secs = std::chrono::duration<double>(wall_end - wall_start).count();
            if (secs < 1.0)
                std::cout << "Completed in " << secs * 1e6 << " us\n";
            else
                std::cout << "Completed in " << secs << " s\n";
        }
    }

    /// Run for an interval without forcing completion.
    void run_for(double interval, bool force_complete = false) {
        double end_time = global_time_ + interval;
        double next_emit_time = global_time_ + emit_step_;

        while (global_time_ < end_time || force_complete) {
            double full_step = std::numeric_limits<double>::infinity();

            struct PendingUpdate {
                std::string proc_name;
                Update      update;
                Topology    topo;
                double      future_time;
            };
            std::vector<PendingUpdate> pending;

            for (auto& [proc_name, proc] : processes_) {
                if (proc->is_step()) continue;

                if (front_.find(proc_name) == front_.end())
                    front_[proc_name] = global_time_;

                double process_time = front_[proc_name];

                if (process_time <= global_time_) {
                    auto topo_it = topology_.find(proc_name);
                    if (topo_it == topology_.end()) continue;

                    State states = store_.topology_view(topo_it->second);
                    double process_dt = proc->calculate_timestep(states);

                    double future;
                    if (force_complete)
                        future = std::min(process_time + process_dt, end_time);
                    else
                        future = process_time + process_dt;

                    if (future <= end_time) {
                        if (proc->update_condition(process_dt, states)) {
                            Update upd = proc->next_update(process_dt, states);
                            pending.push_back({proc_name, std::move(upd),
                                               topo_it->second, future});
                        }
                        full_step = std::min(full_step, future - global_time_);
                    } else {
                        full_step = std::min(full_step, future - global_time_);
                    }
                } else {
                    full_step = std::min(full_step, process_time - global_time_);
                }
            }

            if (std::isinf(full_step)) {
                // No processes ran; jump to next event or end
                double next_event = end_time;
                for (auto& [p, t] : front_) {
                    if (t < next_event) next_event = t;
                }
                global_time_ = next_event;
                if (global_time_ >= end_time) {
                    global_time_ = end_time;
                    if (force_complete) {
                        force_complete = false;
                    }
                }
            } else if (global_time_ + full_step <= end_time) {
                global_time_ += full_step;

                for (auto& pu : pending) {
                    if (pu.future_time <= global_time_) {
                        apply_process_update(pu.update, pu.topo);
                        front_[pu.proc_name] = pu.future_time;
                    }
                }

                run_steps();

                // Emit
                if (emit_step_ == 1.0) {
                    emit_store_data();
                } else if (next_emit_time <= global_time_) {
                    while (next_emit_time <= global_time_) {
                        emit_store_data();
                        next_emit_time += emit_step_;
                    }
                }

                if (progress_bar_) {
                    print_progress_bar(global_time_, end_time);
                }
            } else {
                global_time_ = end_time;
            }

            if (force_complete && global_time_ >= end_time) {
                for (auto& pu : pending) {
                    if (front_.count(pu.proc_name) == 0 ||
                        front_[pu.proc_name] < pu.future_time) {
                        apply_process_update(pu.update, pu.topo);
                        front_[pu.proc_name] = pu.future_time;
                    }
                }
                force_complete = false;
            }
        }

        // Final emit
        emit_store_data();
    }

    // -- state access ---------------------------------------------------------
    Store& state() { return store_; }
    const Store& state() const { return store_; }
    double global_time() const { return global_time_; }
    const std::string& experiment_id() const { return experiment_id_; }

    State get_state() const { return store_.get_state(); }

    Value get_value(const HierarchyPath& path) const {
        return store_.get_value_at(path);
    }

    double get_double(const HierarchyPath& path) const {
        return value_as_double(store_.get_value_at(path));
    }

    State emit_data() const {
        State data = store_.emit_data();
        data["time"] = Value(global_time_);
        return data;
    }

    // -- emitter access -------------------------------------------------------
    Emitter& emitter() { return *emitter_; }
    const Emitter& emitter() const { return *emitter_; }
    std::shared_ptr<Emitter> emitter_ptr() { return emitter_; }

    // -- history (compatibility with old interface) ---------------------------
    const std::vector<State>& history() const { return history_; }

    void set_emit(bool on, double emit_step = 1.0) {
        emit_on_ = on;
        emit_step_ = emit_step;
    }

    // -- end (cleanup parallel processes) -------------------------------------
    void end() {
        // No parallel cleanup needed in this C++ implementation
    }

private:
    void init(const std::map<std::string, std::shared_ptr<Process>>& processes,
              const std::map<std::string, Topology>& topology,
              const State& initial_state)
    {
        processes_ = processes;
        topology_  = topology;

        // Apply schemas
        for (auto& [proc_name, proc] : processes_) {
            auto it = topology_.find(proc_name);
            if (it == topology_.end()) continue;
            store_.apply_port_schemas(proc->ports_schema(), it->second);
        }

        // Apply initial state
        apply_initial_state_recursive(&store_, initial_state);

        // Apply each process's own initial_state
        for (auto& [proc_name, proc] : processes_) {
            State proc_init = proc->initial_state();
            if (proc_init.empty()) continue;
            auto topo_it = topology_.find(proc_name);
            if (topo_it == topology_.end()) continue;
            apply_process_initial_state(proc_init, topo_it->second);
        }

        // Initialize front
        for (auto& [proc_name, proc] : processes_) {
            if (!proc->is_step()) {
                front_[proc_name] = global_time_;
            }
        }

        // Run steps at init
        run_steps();

        // Emit configuration
        emit_configuration();

        // Emit initial state
        emit_store_data();

        // Also store in old-style history
        if (emit_on_) {
            history_.push_back(emit_data());
        }

        if (display_info_) {
            std::cout << "\n=== Vivarium Engine ===\n";
            std::cout << "Processes: " << processes_.size() << "\n";
            std::cout << "Initial time: " << global_time_ << "\n";
        }
    }

    void apply_initial_state_recursive(Store* target, const State& init) {
        for (auto& [key, val] : init) {
            if (val.type() == typeid(State)) {
                apply_initial_state_recursive(
                    target->child(key).get(),
                    std::any_cast<const State&>(val));
            } else {
                target->child(key)->set_value(val);
            }
        }
    }

    void apply_process_initial_state(const State& proc_init, const Topology& topo) {
        for (auto& [port, port_val] : proc_init) {
            auto topo_it = topo.find(port);
            if (topo_it == topo.end()) continue;
            if (port_val.type() == typeid(State)) {
                auto& sub = std::any_cast<const State&>(port_val);
                Store* target = &store_;
                for (auto& seg : topo_it->second) target = target->child(seg).get();
                for (auto& [var, val] : sub) {
                    if (!target->has_child(var) ||
                        !target->child_or_null(var)->value().has_value()) {
                        target->child(var)->set_value(val);
                    }
                }
            }
        }
    }

    void run_steps() {
        for (auto& [proc_name, proc] : processes_) {
            if (!proc->is_step()) continue;
            auto topo_it = topology_.find(proc_name);
            if (topo_it == topology_.end()) continue;

            State states = store_.topology_view(topo_it->second);
            if (!proc->update_condition(0, states)) continue;

            Update update = proc->next_update(0, states);
            apply_process_update(update, topo_it->second);
        }
    }

    void apply_process_update(const Update& update, const Topology& topo) {
        for (auto& [port_name, port_update_val] : update) {
            auto topo_it = topo.find(port_name);
            if (topo_it == topo.end()) continue;

            Store* target = &store_;
            for (auto& seg : topo_it->second) {
                target = target->child(seg).get();
            }

            if (port_update_val.type() == typeid(Update) ||
                port_update_val.type() == typeid(State)) {
                State sub_update;
                try { sub_update = std::any_cast<State>(port_update_val); }
                catch (...) {
                    try { sub_update = std::any_cast<Update>(port_update_val); }
                    catch (...) { continue; }
                }
                bool view_expire = target->apply_update(sub_update);
                if (view_expire) {
                    // Would need to rebuild topology views
                }
            } else {
                target->set_value(target->updater()(target->value(), port_update_val));
            }
        }
    }

    void emit_configuration() {
        State config_data;
        config_data["experiment_id"] = Value(experiment_id_);
        // Topology
        State topo_state;
        for (auto& [name, t] : topology_) {
            State port_map;
            for (auto& [port, path] : t) {
                port_map[port] = Value(path_to_string(path, "/"));
            }
            topo_state[name] = Value(port_map);
        }
        config_data["topology"] = Value(topo_state);

        State emit_config;
        emit_config["table"] = Value(std::string("configuration"));
        emit_config["data"] = Value(config_data);
        emitter_->emit(emit_config);
    }

    void emit_store_data() {
        State data = store_.emit_data();
        data["time"] = Value(global_time_);

        State emit_config;
        emit_config["table"] = Value(std::string("history"));
        emit_config["data"] = Value(data);
        emitter_->emit(emit_config);

        // Also append to legacy history
        if (emit_on_) {
            State snapshot = data;
            history_.push_back(snapshot);
        }
    }

    static void print_progress_bar(double iteration, double total,
                                   int length = 50) {
        int filled = static_cast<int>(length * iteration / total);
        std::string bar(filled, '#');
        bar += std::string(length - filled, '-');
        std::cout << "\rProgress: [" << bar << "] "
                  << (total - iteration) << "/" << total << " remaining    "
                  << std::flush;
        if (iteration >= total) std::cout << "\n";
    }

    // ---- data members -------------------------------------------------------
    Store store_;
    std::map<std::string, std::shared_ptr<Process>> processes_;
    std::map<std::string, Topology>                 topology_;

    double      global_time_ = 0.0;
    bool        display_info_ = true;
    double      emit_step_ = 1.0;
    bool        progress_bar_ = false;
    std::string experiment_id_;

    std::map<std::string, double> front_;

    std::shared_ptr<Emitter> emitter_;
    bool                emit_on_   = true;
    std::vector<State>  history_;
};


// ============================================================================
//  generate_state implementation (needs Process to be complete)
// ============================================================================
inline std::shared_ptr<Store> generate_state(
    const std::map<std::string, std::shared_ptr<Process>>& processes,
    const std::map<std::string, Topology>& topology,
    const State& initial_state)
{
    auto store = std::make_shared<Store>();

    // Apply port schemas
    for (auto& [proc_name, proc] : processes) {
        auto topo_it = topology.find(proc_name);
        if (topo_it == topology.end()) continue;
        store->apply_port_schemas(proc->ports_schema(), topo_it->second);
    }

    // Apply initial state
    for (auto& [key, val] : initial_state) {
        if (val.type() == typeid(State)) {
            auto target = store->child(key);
            auto& sub = std::any_cast<const State&>(val);
            for (auto& [k2, v2] : sub) {
                target->child(k2)->set_value(v2);
            }
        } else {
            store->child(key)->set_value(val);
        }
    }

    store->build_topology_views();
    return store;
}


// ============================================================================
//  Experiment -- High-level experiment runner (vivarium/core/experiment.py)
// ============================================================================
struct ExperimentConfig {
    Composite composite;
    State initial_state;
    double total_time = 10.0;
    double emit_step = 1.0;
    std::string emitter_type = "timeseries";
    std::string experiment_id = "";
    bool display_info = true;
    bool progress_bar = false;
    double initial_global_time = 0.0;
};

/// Run a complete experiment and return the emitter data.
inline std::map<double, State> run_experiment(const ExperimentConfig& config) {
    Engine engine(
        config.composite,
        config.initial_state,
        config.initial_global_time,
        config.display_info,
        config.emitter_type,
        config.emit_step,
        config.progress_bar,
        config.experiment_id
    );

    engine.update(config.total_time);
    engine.end();

    return engine.emitter().get_data();
}

}  // namespace vivarium
