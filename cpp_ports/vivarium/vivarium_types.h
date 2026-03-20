// vivarium_types.h -- Type system, updaters, dividers, serialization, registry
//
// Ported from:
//   vivarium/core/types.py
//   vivarium/core/registry.py
//   vivarium/core/serialize.py
//   vivarium/library/dict_utils.py
//   vivarium/library/topology.py

#pragma once

#include <algorithm>
#include <any>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace vivarium {

// ============================================================================
//  Type aliases (vivarium/core/types.py)
// ============================================================================

using HierarchyPath = std::vector<std::string>;
using Value = std::any;
using State = std::map<std::string, Value>;
using Update = std::map<std::string, Value>;
using Schema = std::map<std::string, Value>;
using Topology = std::map<std::string, HierarchyPath>;
using PortSchema = std::map<std::string, std::map<std::string, Schema>>;
using Flow = std::map<std::string, std::vector<HierarchyPath>>;

// ============================================================================
//  Value casting utilities
// ============================================================================
template <typename T>
T value_as(const Value& v, T fallback = T{}) {
    try { return std::any_cast<T>(v); }
    catch (const std::bad_any_cast&) { return fallback; }
}

inline double value_as_double(const Value& v, double fallback = 0.0) {
    if (!v.has_value()) return fallback;
    if (v.type() == typeid(double))        return std::any_cast<double>(v);
    if (v.type() == typeid(float))         return static_cast<double>(std::any_cast<float>(v));
    if (v.type() == typeid(int))           return static_cast<double>(std::any_cast<int>(v));
    if (v.type() == typeid(long))          return static_cast<double>(std::any_cast<long>(v));
    if (v.type() == typeid(long long))     return static_cast<double>(std::any_cast<long long>(v));
    if (v.type() == typeid(unsigned))      return static_cast<double>(std::any_cast<unsigned>(v));
    if (v.type() == typeid(unsigned long)) return static_cast<double>(std::any_cast<unsigned long>(v));
    if (v.type() == typeid(size_t))        return static_cast<double>(std::any_cast<size_t>(v));
    return fallback;
}

inline bool value_as_bool(const Value& v, bool fallback = false) {
    if (!v.has_value()) return fallback;
    if (v.type() == typeid(bool))   return std::any_cast<bool>(v);
    if (v.type() == typeid(int))    return std::any_cast<int>(v) != 0;
    if (v.type() == typeid(double)) return std::any_cast<double>(v) != 0.0;
    return fallback;
}

inline std::string value_as_string(const Value& v, const std::string& fallback = "") {
    if (!v.has_value()) return fallback;
    try { return std::any_cast<std::string>(v); } catch (...) {}
    try { return std::string(std::any_cast<const char*>(v)); } catch (...) {}
    return fallback;
}

inline std::vector<double> value_as_vector(const Value& v) {
    if (!v.has_value()) return {};
    try { return std::any_cast<std::vector<double>>(v); } catch (...) { return {}; }
}

inline std::string value_to_string(const Value& v) {
    if (!v.has_value()) return "null";
    if (v.type() == typeid(double))      return std::to_string(std::any_cast<double>(v));
    if (v.type() == typeid(float))       return std::to_string(std::any_cast<float>(v));
    if (v.type() == typeid(int))         return std::to_string(std::any_cast<int>(v));
    if (v.type() == typeid(long))        return std::to_string(std::any_cast<long>(v));
    if (v.type() == typeid(bool))        return std::any_cast<bool>(v) ? "true" : "false";
    if (v.type() == typeid(std::string)) return "\"" + std::any_cast<std::string>(v) + "\"";
    if (v.type() == typeid(const char*)) return std::string("\"") + std::any_cast<const char*>(v) + "\"";
    return std::string("<") + v.type().name() + ">";
}

// ============================================================================
//  HierarchyPath utilities (vivarium/library/topology.py)
// ============================================================================
inline std::string path_to_string(const HierarchyPath& path, const std::string& sep = ".") {
    std::string s;
    for (size_t i = 0; i < path.size(); ++i) {
        if (i > 0) s += sep;
        s += path[i];
    }
    return s;
}

inline HierarchyPath string_to_path(const std::string& s, char sep = '.') {
    HierarchyPath path;
    std::istringstream iss(s);
    std::string seg;
    while (std::getline(iss, seg, sep)) {
        if (!seg.empty()) path.push_back(seg);
    }
    return path;
}

inline bool starts_with(const HierarchyPath& full, const HierarchyPath& sub) {
    if (sub.size() > full.size()) return false;
    for (size_t i = 0; i < sub.size(); ++i) {
        if (full[i] != sub[i]) return false;
    }
    return true;
}

inline HierarchyPath normalize_path(const HierarchyPath& path) {
    HierarchyPath result;
    for (auto& seg : path) {
        if (seg == "..") { if (!result.empty()) result.pop_back(); }
        else result.push_back(seg);
    }
    return result;
}

inline HierarchyPath concat_paths(const HierarchyPath& a, const HierarchyPath& b) {
    HierarchyPath result = a;
    result.insert(result.end(), b.begin(), b.end());
    return result;
}

// ============================================================================
//  Nested State access (vivarium/library/topology.py)
// ============================================================================
inline Value get_in(const State& state, const HierarchyPath& path) {
    if (path.empty()) return Value(state);
    const State* cur = &state;
    for (size_t i = 0; i < path.size(); ++i) {
        auto it = cur->find(path[i]);
        if (it == cur->end()) return Value{};
        if (i + 1 == path.size()) return it->second;
        try { cur = &std::any_cast<const State&>(it->second); }
        catch (...) { return Value{}; }
    }
    return Value{};
}

inline void assoc_path(State& state, const HierarchyPath& path, const Value& val) {
    if (path.empty()) return;
    State* cur = &state;
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        auto it = cur->find(path[i]);
        if (it == cur->end() || it->second.type() != typeid(State)) {
            (*cur)[path[i]] = Value(State{});
        }
        cur = &std::any_cast<State&>((*cur)[path[i]]);
    }
    (*cur)[path.back()] = val;
}

inline void delete_in(State& state, const HierarchyPath& path) {
    if (path.empty()) return;
    State* cur = &state;
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        auto it = cur->find(path[i]);
        if (it == cur->end()) return;
        try { cur = &std::any_cast<State&>(it->second); } catch (...) { return; }
    }
    cur->erase(path.back());
}

// ============================================================================
//  dict_utils (vivarium/library/dict_utils.py)
// ============================================================================
inline void deep_merge(State& dst, const State& src) {
    for (auto& [key, val] : src) {
        if (val.type() == typeid(State)) {
            auto dst_it = dst.find(key);
            if (dst_it != dst.end() && dst_it->second.type() == typeid(State)) {
                auto& dst_sub = std::any_cast<State&>(dst_it->second);
                deep_merge(dst_sub, std::any_cast<const State&>(val));
            } else { dst[key] = val; }
        } else { dst[key] = val; }
    }
}

inline State deep_merge_copy(const State& a, const State& b) {
    State result = a; deep_merge(result, b); return result;
}

inline State deep_copy(const State& s) {
    State result;
    for (auto& [key, val] : s) {
        if (val.type() == typeid(State))
            result[key] = Value(deep_copy(std::any_cast<const State&>(val)));
        else result[key] = val;
    }
    return result;
}

inline void apply_func_to_leaves(State& state, const std::function<void(Value&)>& f) {
    for (auto& [key, val] : state) {
        if (val.type() == typeid(State)) {
            auto& sub = std::any_cast<State&>(val);
            apply_func_to_leaves(sub, f);
        } else { f(val); }
    }
}

inline void flatten_state(const State& state, const HierarchyPath& prefix,
                          std::vector<std::pair<HierarchyPath, Value>>& out) {
    for (auto& [key, val] : state) {
        HierarchyPath path = prefix;
        path.push_back(key);
        if (val.type() == typeid(State))
            flatten_state(std::any_cast<const State&>(val), path, out);
        else out.push_back({path, val});
    }
}

inline std::vector<std::pair<HierarchyPath, Value>> flatten_state(const State& state) {
    std::vector<std::pair<HierarchyPath, Value>> out;
    flatten_state(state, {}, out);
    return out;
}

inline State inverse_topology(const HierarchyPath& path, const State& update,
                               const Topology& topology) {
    State result;
    for (auto& [port_name, port_update] : update) {
        auto topo_it = topology.find(port_name);
        if (topo_it == topology.end()) continue;
        HierarchyPath abs_path = normalize_path(concat_paths(path, topo_it->second));
        assoc_path(result, abs_path, port_update);
    }
    return result;
}

// ============================================================================
//  Updater functions (vivarium/core/registry.py)
// ============================================================================
using UpdaterFn = std::function<Value(const Value&, const Value&)>;

inline Value updater_accumulate(const Value& current, const Value& delta) {
    if (current.type() == typeid(std::vector<double>) &&
        delta.type() == typeid(std::vector<double>)) {
        auto c = std::any_cast<std::vector<double>>(current);
        auto d = std::any_cast<std::vector<double>>(delta);
        for (size_t i = 0; i < std::min(c.size(), d.size()); ++i) c[i] += d[i];
        return Value(c);
    }
    if (current.type() == typeid(State) && delta.type() == typeid(State)) {
        State result = std::any_cast<State>(current);
        for (auto& [k, v] : std::any_cast<const State&>(delta)) {
            auto it = result.find(k);
            if (it != result.end()) result[k] = updater_accumulate(it->second, v);
            else result[k] = v;
        }
        return Value(result);
    }
    return Value(value_as_double(current) + value_as_double(delta));
}

inline Value updater_set(const Value&, const Value& incoming) { return incoming; }
inline Value updater_null(const Value& current, const Value&) { return current; }

inline Value updater_nonneg_accumulate(const Value& current, const Value& delta) {
    if (current.type() == typeid(std::vector<double>) &&
        delta.type() == typeid(std::vector<double>)) {
        auto c = std::any_cast<std::vector<double>>(current);
        auto d = std::any_cast<std::vector<double>>(delta);
        for (size_t i = 0; i < std::min(c.size(), d.size()); ++i)
            c[i] = std::max(0.0, c[i] + d[i]);
        return Value(c);
    }
    return Value(std::max(0.0, value_as_double(current) + value_as_double(delta)));
}

inline Value updater_merge(const Value& current, const Value& new_value) {
    if (current.type() != typeid(State) || new_value.type() != typeid(State))
        return new_value;
    State result = std::any_cast<State>(current);
    deep_merge(result, std::any_cast<const State&>(new_value));
    return Value(result);
}

inline Value updater_dictionary(const Value& current, const Value& update_val) {
    if (current.type() != typeid(State) || update_val.type() != typeid(State))
        return update_val;
    State result = std::any_cast<State>(current);
    for (auto& [key, val] : std::any_cast<const State&>(update_val)) {
        if (key == "_add") {
            try {
                auto adds = std::any_cast<std::vector<State>>(val);
                for (auto& ae : adds) result[value_as_string(ae.at("key"))] = ae.at("state");
            } catch (...) {}
        } else if (key == "_delete") {
            try {
                auto dels = std::any_cast<std::vector<std::string>>(val);
                for (auto& dk : dels) result.erase(dk);
            } catch (...) {}
        } else {
            auto it = result.find(key);
            if (it != result.end() && it->second.type() == typeid(State) &&
                val.type() == typeid(State)) {
                deep_merge(std::any_cast<State&>(it->second), std::any_cast<const State&>(val));
            }
        }
    }
    return Value(result);
}

inline UpdaterFn make_bounds_updater(double lo, double hi) {
    return [lo, hi](const Value& cur, const Value& delta) -> Value {
        return Value(std::clamp(value_as_double(cur) + value_as_double(delta), lo, hi));
    };
}

// ============================================================================
//  Divider functions (vivarium/core/registry.py)
// ============================================================================
using DividerFn = std::function<std::vector<Value>(const Value&)>;

inline std::vector<Value> divider_set(const Value& s) { return {s, s}; }

inline std::vector<Value> divider_split(const Value& s) {
    if (s.type() == typeid(double)) {
        double h = std::any_cast<double>(s) / 2.0;
        return {Value(h), Value(h)};
    }
    if (s.type() == typeid(int)) {
        int v = std::any_cast<int>(s), h = v/2, r = v%2;
        static thread_local std::mt19937 rng{std::random_device{}()};
        std::uniform_int_distribution<int> d(0,1);
        return d(rng) ? std::vector<Value>{Value(h+r), Value(h)}
                      : std::vector<Value>{Value(h), Value(h+r)};
    }
    if (s.type() == typeid(std::vector<double>)) {
        auto vec = std::any_cast<std::vector<double>>(s);
        std::vector<double> h(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) h[i] = vec[i]/2.0;
        return {Value(h), Value(h)};
    }
    return divider_set(s);
}

inline std::vector<Value> divider_binomial(const Value& s) {
    if (s.type() == typeid(int)) {
        int n = std::any_cast<int>(s);
        static thread_local std::mt19937 rng{std::random_device{}()};
        int d1 = std::binomial_distribution<int>(n, 0.5)(rng);
        return {Value(d1), Value(n - d1)};
    }
    return divider_split(s);
}

inline std::vector<Value> divider_zero(const Value&) { return {Value(0.0), Value(0.0)}; }
inline std::vector<Value> divider_null(const Value&) { return {}; }

inline DividerFn make_set_value_divider(const Value& v) {
    return [v](const Value&) -> std::vector<Value> { return {v, v}; };
}

inline std::vector<Value> divider_split_dict(const Value& s) {
    if (s.type() != typeid(State)) return divider_set(s);
    auto& st = std::any_cast<const State&>(s);
    State d1, d2; size_t h = st.size()/2, i = 0;
    for (auto& [k,v] : st) { if (i++ < h) d1[k]=v; else d2[k]=v; }
    return {Value(d1), Value(d2)};
}

// ============================================================================
//  Registry class and global instances
// ============================================================================
template <typename T>
class Registry {
public:
    void register_item(const std::string& key, T item,
                       const std::vector<std::string>& alt = {}) {
        std::lock_guard<std::mutex> lock(mu_);
        registry_[key] = item;
        main_keys_.push_back(key);
        for (auto& a : alt) registry_[a] = item;
    }
    T access(const std::string& key) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = registry_.find(key);
        return it != registry_.end() ? it->second : T{};
    }
    bool has(const std::string& key) const {
        std::lock_guard<std::mutex> lock(mu_);
        return registry_.count(key) > 0;
    }
    std::vector<std::string> list() const {
        std::lock_guard<std::mutex> lock(mu_);
        return main_keys_;
    }
private:
    mutable std::mutex mu_;
    std::map<std::string, T> registry_;
    std::vector<std::string> main_keys_;
};

inline Registry<UpdaterFn>& updater_registry() {
    static Registry<UpdaterFn> reg;
    static bool init = false;
    if (!init) {
        init = true;
        reg.register_item("accumulate", updater_accumulate);
        reg.register_item("set", updater_set);
        reg.register_item("null", updater_null);
        reg.register_item("nonnegative_accumulate", updater_nonneg_accumulate);
        reg.register_item("merge", updater_merge);
        reg.register_item("dictionary", updater_dictionary);
    }
    return reg;
}

inline Registry<DividerFn>& divider_registry() {
    static Registry<DividerFn> reg;
    static bool init = false;
    if (!init) {
        init = true;
        reg.register_item("set", divider_set);
        reg.register_item("split", divider_split);
        reg.register_item("binomial", divider_binomial);
        reg.register_item("zero", divider_zero);
        reg.register_item("null", divider_null);
        reg.register_item("split_dict", divider_split_dict);
    }
    return reg;
}

inline UpdaterFn get_updater(const std::string& name) {
    auto fn = updater_registry().access(name);
    if (fn) return fn;
    return updater_accumulate;
}

inline DividerFn get_divider(const std::string& name) {
    auto fn = divider_registry().access(name);
    if (fn) return fn;
    return divider_set;
}

// ============================================================================
//  Serialization helpers (vivarium/core/serialize.py)
// ============================================================================
inline std::string serialize_state(const State& state, int indent = 0);

inline std::string serialize_value_repr(const Value& v, int indent = 0) {
    if (!v.has_value()) return "null";
    if (v.type() == typeid(State))
        return serialize_state(std::any_cast<const State&>(v), indent);
    if (v.type() == typeid(std::vector<double>)) {
        auto vec = std::any_cast<std::vector<double>>(v);
        std::string s = "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) s += ", ";
            s += std::to_string(vec[i]);
        }
        return s + "]";
    }
    if (v.type() == typeid(bool)) return std::any_cast<bool>(v) ? "true" : "false";
    return value_to_string(v);
}

inline std::string serialize_state(const State& state, int indent) {
    std::string pad(indent * 2, ' '), pad2((indent + 1) * 2, ' ');
    std::string s = "{\n";
    size_t i = 0;
    for (auto& [key, val] : state) {
        s += pad2 + "\"" + key + "\": " + serialize_value_repr(val, indent + 1);
        if (++i < state.size()) s += ",";
        s += "\n";
    }
    return s + pad + "}";
}

// ============================================================================
//  Process registry (for named process lookup)
// ============================================================================
class Process; // forward
using ProcessFactory = std::function<std::shared_ptr<Process>(const State&)>;

inline Registry<ProcessFactory>& process_registry() {
    static Registry<ProcessFactory> reg;
    return reg;
}

}  // namespace vivarium
