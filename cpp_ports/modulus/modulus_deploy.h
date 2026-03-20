// modulus_deploy.h - Model export, serialization, ONNX-compatible utilities
// Port of Modulus deploy functionality
//
// C++17, no external dependencies.

#ifndef MODULUS_DEPLOY_H
#define MODULUS_DEPLOY_H

#include "modulus.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace modulus {
namespace deploy {

// ============================================================
// Model serialization format
// ============================================================
struct ModelInfo {
    std::string model_type;     // e.g., "FNO1D", "MLP", "MeshGraphNet"
    std::string activation;
    std::vector<int> architecture; // key dimensions
    int num_parameters;
};

// ============================================================
// Parameter export to various formats
// ============================================================

// Export parameters to raw binary (for ONNX runtime or custom inference)
inline void export_parameters_binary(const std::string& path,
                                      const std::vector<Parameter*>& params,
                                      const ModelInfo& info) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write: " + path);

    // Magic number and version
    uint32_t magic = 0x4D4F4455; // "MODU"
    uint32_t version = 1;
    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&version), 4);

    // Model info
    uint32_t type_len = (uint32_t)info.model_type.size();
    f.write(reinterpret_cast<const char*>(&type_len), 4);
    f.write(info.model_type.data(), type_len);

    uint32_t act_len = (uint32_t)info.activation.size();
    f.write(reinterpret_cast<const char*>(&act_len), 4);
    f.write(info.activation.data(), act_len);

    uint32_t n_arch = (uint32_t)info.architecture.size();
    f.write(reinterpret_cast<const char*>(&n_arch), 4);
    for (auto& a : info.architecture) {
        int32_t v = a;
        f.write(reinterpret_cast<const char*>(&v), 4);
    }

    // Parameters
    uint32_t n_params = (uint32_t)params.size();
    f.write(reinterpret_cast<const char*>(&n_params), 4);

    for (auto* p : params) {
        uint32_t name_len = (uint32_t)p->name.size();
        f.write(reinterpret_cast<const char*>(&name_len), 4);
        f.write(p->name.data(), name_len);

        uint32_t ndim = (uint32_t)p->value.shape.size();
        f.write(reinterpret_cast<const char*>(&ndim), 4);
        for (auto& s : p->value.shape) {
            int32_t dim = s;
            f.write(reinterpret_cast<const char*>(&dim), 4);
        }

        uint32_t n = (uint32_t)p->value.data.size();
        f.write(reinterpret_cast<const char*>(&n), 4);
        f.write(reinterpret_cast<const char*>(p->value.data.data()), n * sizeof(double));
    }
}

// Import parameters from binary file
inline ModelInfo import_parameters_binary(const std::string& path,
                                           std::vector<Parameter*>& params) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot read: " + path);

    uint32_t magic, version;
    f.read(reinterpret_cast<char*>(&magic), 4);
    f.read(reinterpret_cast<char*>(&version), 4);

    if (magic != 0x4D4F4455) throw std::runtime_error("Invalid model file");

    ModelInfo info;

    uint32_t type_len;
    f.read(reinterpret_cast<char*>(&type_len), 4);
    info.model_type.resize(type_len);
    f.read(&info.model_type[0], type_len);

    uint32_t act_len;
    f.read(reinterpret_cast<char*>(&act_len), 4);
    info.activation.resize(act_len);
    f.read(&info.activation[0], act_len);

    uint32_t n_arch;
    f.read(reinterpret_cast<char*>(&n_arch), 4);
    info.architecture.resize(n_arch);
    for (uint32_t i = 0; i < n_arch; ++i) {
        int32_t v;
        f.read(reinterpret_cast<char*>(&v), 4);
        info.architecture[i] = v;
    }

    uint32_t n_params;
    f.read(reinterpret_cast<char*>(&n_params), 4);
    info.num_parameters = n_params;

    // Build param map
    std::unordered_map<std::string, Parameter*> param_map;
    for (auto* p : params) param_map[p->name] = p;

    for (uint32_t i = 0; i < n_params; ++i) {
        uint32_t name_len;
        f.read(reinterpret_cast<char*>(&name_len), 4);
        std::string name(name_len, '\0');
        f.read(&name[0], name_len);

        uint32_t ndim;
        f.read(reinterpret_cast<char*>(&ndim), 4);
        std::vector<int> shape(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            int32_t dim;
            f.read(reinterpret_cast<char*>(&dim), 4);
            shape[d] = dim;
        }

        uint32_t n;
        f.read(reinterpret_cast<char*>(&n), 4);
        std::vector<double> data(n);
        f.read(reinterpret_cast<char*>(data.data()), n * sizeof(double));

        auto it = param_map.find(name);
        if (it != param_map.end() && it->second->value.numel() == (int)n) {
            it->second->value.data = data;
        }
    }

    return info;
}

// ============================================================
// Export parameters to NumPy-compatible format (.npy)
// Single-parameter export for interop
// ============================================================
inline void export_npy(const std::string& path, const Tensor& t) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write: " + path);

    // NumPy .npy format header
    uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    f.write(reinterpret_cast<const char*>(magic), 6);

    uint8_t major_ver = 1, minor_ver = 0;
    f.write(reinterpret_cast<const char*>(&major_ver), 1);
    f.write(reinterpret_cast<const char*>(&minor_ver), 1);

    // Build header string
    std::string header = "{'descr': '<f8', 'fortran_order': False, 'shape': (";
    for (int i = 0; i < t.ndim(); ++i) {
        if (i) header += ", ";
        header += std::to_string(t.shape[i]);
    }
    if (t.ndim() == 1) header += ",";
    header += "), }";

    // Pad to multiple of 64
    while ((header.size() + 10 + 1) % 64 != 0) header += ' ';
    header += '\n';

    uint16_t header_len = (uint16_t)header.size();
    f.write(reinterpret_cast<const char*>(&header_len), 2);
    f.write(header.data(), header.size());

    // Write data
    f.write(reinterpret_cast<const char*>(t.data.data()), t.numel() * sizeof(double));
}

// ============================================================
// ONNX-compatible operator descriptions
// (For generating ONNX graph metadata, not actual ONNX protobuf)
// ============================================================
struct ONNXOpDesc {
    std::string op_type;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<std::pair<std::string, std::string>> attributes;
};

class ONNXGraphBuilder {
public:
    std::vector<ONNXOpDesc> ops;
    std::string model_name;
    int ir_version;
    int opset_version;

    ONNXGraphBuilder(const std::string& name = "modulus_model")
        : model_name(name), ir_version(7), opset_version(13) {}

    void add_linear(const std::string& name,
                    const std::string& input, const std::string& output,
                    int in_features, int out_features) {
        ops.push_back({"MatMul", name + "_matmul",
                       {input, name + "_weight"}, {name + "_mm_out"}, {}});
        ops.push_back({"Add", name + "_add",
                       {name + "_mm_out", name + "_bias"}, {output}, {}});
    }

    void add_activation(const std::string& name,
                        const std::string& input, const std::string& output,
                        const std::string& act_type) {
        std::string onnx_op = "Relu";
        if (act_type == "gelu") onnx_op = "Gelu";
        else if (act_type == "sigmoid") onnx_op = "Sigmoid";
        else if (act_type == "tanh") onnx_op = "Tanh";
        else if (act_type == "silu") onnx_op = "Sigmoid"; // approximate
        ops.push_back({onnx_op, name, {input}, {output}, {}});
    }

    void add_conv2d(const std::string& name,
                    const std::string& input, const std::string& output,
                    int kernel_size, int stride, int padding) {
        ops.push_back({"Conv", name,
                       {input, name + "_weight", name + "_bias"}, {output},
                       {{"kernel_shape", std::to_string(kernel_size)},
                        {"strides", std::to_string(stride)},
                        {"pads", std::to_string(padding)}}});
    }

    // Print graph description
    void print() const {
        std::cout << "ONNX Graph: " << model_name << std::endl;
        std::cout << "  IR version: " << ir_version << std::endl;
        std::cout << "  Opset: " << opset_version << std::endl;
        std::cout << "  Operators: " << ops.size() << std::endl;
        for (auto& op : ops) {
            std::cout << "    " << op.op_type << " (" << op.name << "): ";
            for (auto& in : op.inputs) std::cout << in << " ";
            std::cout << "-> ";
            for (auto& out : op.outputs) std::cout << out << " ";
            std::cout << std::endl;
        }
    }

    // Export graph description to text file
    void export_text(const std::string& path) const {
        std::ofstream f(path);
        f << "# Modulus ONNX Graph Description\n";
        f << "model: " << model_name << "\n";
        f << "ir_version: " << ir_version << "\n";
        f << "opset: " << opset_version << "\n\n";
        for (auto& op : ops) {
            f << op.op_type << " " << op.name << "\n";
            f << "  inputs: ";
            for (auto& in : op.inputs) f << in << " ";
            f << "\n  outputs: ";
            for (auto& out : op.outputs) f << out << " ";
            f << "\n";
            for (auto& [k, v] : op.attributes)
                f << "  " << k << ": " << v << "\n";
            f << "\n";
        }
    }
};

// ============================================================
// Model profiling utilities
// ============================================================
namespace profiling {

struct ProfileResult {
    std::string name;
    double total_time_ms;
    int num_calls;
    double avg_time_ms() const { return total_time_ms / std::max(num_calls, 1); }
};

class Profiler {
    std::vector<ProfileResult> results;
    std::chrono::high_resolution_clock::time_point start_time;
    std::string current_name;

public:
    void start(const std::string& name) {
        current_name = name;
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start_time).count();

        for (auto& r : results) {
            if (r.name == current_name) {
                r.total_time_ms += elapsed;
                r.num_calls++;
                return;
            }
        }
        results.push_back({current_name, elapsed, 1});
    }

    void print_summary() const {
        std::cout << "\n  Profiling Summary:" << std::endl;
        std::cout << "  " << std::setw(30) << "Operation"
                  << std::setw(12) << "Total(ms)"
                  << std::setw(12) << "Calls"
                  << std::setw(12) << "Avg(ms)" << std::endl;
        std::cout << "  " << std::string(66, '-') << std::endl;
        for (auto& r : results) {
            std::cout << "  " << std::setw(30) << r.name
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.total_time_ms
                      << std::setw(12) << r.num_calls
                      << std::setw(12) << r.avg_time_ms() << std::endl;
        }
    }

    void reset() { results.clear(); }
};

// Count model parameters
inline int count_parameters(const std::vector<Parameter*>& params) {
    int total = 0;
    for (auto* p : params) total += p->value.numel();
    return total;
}

// Estimate model memory usage (bytes)
inline size_t estimate_memory(const std::vector<Parameter*>& params) {
    size_t bytes = 0;
    for (auto* p : params) {
        bytes += p->value.numel() * sizeof(double); // values
        bytes += p->grad.numel() * sizeof(double);  // gradients
    }
    return bytes;
}

// Print model summary
inline void print_model_summary(const std::string& name,
                                 const std::vector<Parameter*>& params) {
    int n_params = count_parameters(params);
    size_t mem = estimate_memory(params);
    std::cout << "\n  Model: " << name << std::endl;
    std::cout << "  Parameters: " << n_params << std::endl;
    std::cout << "  Memory: " << std::fixed << std::setprecision(2)
              << mem / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Parameter groups: " << params.size() << std::endl;

    // Per-layer breakdown
    std::cout << "  Layer breakdown:" << std::endl;
    for (auto* p : params) {
        std::cout << "    " << std::setw(40) << p->name
                  << "  shape=[";
        for (int d = 0; d < (int)p->value.shape.size(); ++d) {
            if (d) std::cout << ",";
            std::cout << p->value.shape[d];
        }
        std::cout << "]  (" << p->value.numel() << " params)" << std::endl;
    }
}

} // namespace profiling

} // namespace deploy
} // namespace modulus

#endif // MODULUS_DEPLOY_H
