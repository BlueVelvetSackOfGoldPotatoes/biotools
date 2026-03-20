#include "run_logger.h"

#include <atomic>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <system_error>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {
std::string join_path(const std::string& a, const std::string& b) {
    return (std::filesystem::path(a) / b).string();
}

std::string trim_copy(std::string s) {
    auto is_space = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    while (!s.empty() && is_space(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && is_space(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

std::string to_lower_ascii(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return s;
}

std::string sanitize_token(const std::string& raw, const std::string& fallback = "unknown") {
    std::string out = to_lower_ascii(raw);
    for (char& c : out) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
        if (!ok) c = '_';
    }
    out = trim_copy(out);
    if (out.empty()) return fallback;
    return out;
}

std::int64_t process_id() {
#if defined(_WIN32)
    return static_cast<std::int64_t>(_getpid());
#else
    return static_cast<std::int64_t>(::getpid());
#endif
}

std::atomic<std::uint64_t>& run_id_counter() {
    static std::atomic<std::uint64_t> ctr{0};
    return ctr;
}

std::string random_hex_suffix(std::size_t n = 8) {
    thread_local std::mt19937_64 rng{
        static_cast<std::uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count() ^
            (run_id_counter().fetch_add(1, std::memory_order_relaxed) + 0x9e3779b97f4a7c15ULL)
        )
    };
    static constexpr char kHex[] = "0123456789abcdef";
    std::uniform_int_distribution<int> dist(0, 15);
    std::string out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) out.push_back(kHex[dist(rng)]);
    return out;
}

bool is_probably_valid_json_value(const std::string& raw) {
    const std::string s = trim_copy(raw);
    if (s.empty()) return false;
    if (s == "null" || s == "true" || s == "false") return true;
    if (s.front() == '"' && s.back() == '"' && s.size() >= 2) return true;

    if ((s.front() >= '0' && s.front() <= '9') || s.front() == '-' || s.front() == '+') {
        char* end = nullptr;
        std::strtod(s.c_str(), &end);
        if (end && *end == '\0') return true;
    }

    if (!((s.front() == '{' && s.back() == '}') || (s.front() == '[' && s.back() == ']'))) {
        return false;
    }

    std::vector<char> stack;
    stack.reserve(32);
    bool in_string = false;
    bool escape = false;
    for (char c : s) {
        if (in_string) {
            if (escape) {
                escape = false;
                continue;
            }
            if (c == '\\') {
                escape = true;
                continue;
            }
            if (c == '"') in_string = false;
            continue;
        }

        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '{' || c == '[') {
            stack.push_back(c);
            continue;
        }
        if (c == '}' || c == ']') {
            if (stack.empty()) return false;
            const char open = stack.back();
            stack.pop_back();
            if ((open == '{' && c != '}') || (open == '[' && c != ']')) return false;
        }
    }
    return !in_string && !escape && stack.empty();
}

std::string exec_capture(const char* cmd) {
    if (!cmd || !*cmd) return "";
#if defined(_WIN32)
    return "";
#else
    std::array<char, 256> buf{};
    std::string out;
    FILE* pipe = ::popen(cmd, "r");
    if (!pipe) return "";
    while (fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
        out += buf.data();
    }
    ::pclose(pipe);
    return trim_copy(out);
#endif
}

struct GitInfo {
    std::string commit;
    bool dirty_known = false;
    bool dirty = false;
};

GitInfo resolve_git_info() {
    GitInfo info;
    const char* env_commit = std::getenv("GIT_COMMIT");
    if (env_commit && *env_commit) {
        info.commit = trim_copy(env_commit);
    } else {
        info.commit = exec_capture("git rev-parse --short HEAD 2>/dev/null");
    }

#if !defined(_WIN32)
    const std::string rc = exec_capture("git diff --quiet --ignore-submodules HEAD >/dev/null 2>&1; echo $?");
    if (!rc.empty()) {
        info.dirty_known = true;
        info.dirty = (rc != "0");
    }
#endif
    return info;
}

std::string infer_benchmark_id(const std::string& data_version) {
    const std::string dv = to_lower_ascii(data_version);
    if (dv.find("mnist") != std::string::npos) return "mnist";
    if (dv.find("tictactoe") != std::string::npos) return "tictactoe";
    if (dv.empty()) return "unknown";

    const size_t dash = dv.find('-');
    if (dash == std::string::npos) return dv;
    return dv.substr(0, dash);
}

std::string infer_benchmark_name(const std::string& benchmark_id) {
    if (benchmark_id == "mnist") return "MNIST";
    if (benchmark_id == "tictactoe") return "TicTacToe";
    if (benchmark_id.empty()) return "Unknown";
    return benchmark_id;
}
} // namespace

RunLogger::RunLogger(const std::string& model_family,
                     const std::string& model_variant,
                     int seed,
                     const std::string& data_version,
                     const std::string& params_json,
                     const std::string& benchmark_id,
                     const std::string& benchmark_name,
                     const std::string& task_type)
    : model_family_(sanitize_token(model_family, "unknown")),
      model_variant_(model_variant),
      seed_(seed),
      data_version_(data_version),
      benchmark_id_(benchmark_id.empty() ? infer_benchmark_id(data_version) : benchmark_id),
      benchmark_name_(benchmark_name.empty() ? infer_benchmark_name(benchmark_id_) : benchmark_name),
      task_type_(task_type.empty() ? "classification" : task_type),
      params_json_(params_json),
      train_start_utc_(utc_now_iso8601()) {
    process_id_ = process_id();
    params_raw_json_ = params_json;

    const std::string trimmed_params = trim_copy(params_json);
    if (trimmed_params.empty()) {
        params_json_ = "{}";
        params_valid_json_ = true;
    } else if (is_probably_valid_json_value(trimmed_params)) {
        params_json_ = trimmed_params;
        params_valid_json_ = true;
    } else {
        params_json_ = "{}";
        params_valid_json_ = false;
        params_parse_error_ = "invalid_json_value";
    }

    const GitInfo git = resolve_git_info();
    git_commit_ = git.commit;
    git_dirty_known_ = git.dirty_known;
    git_dirty_ = git.dirty;
    const char* trial_env = std::getenv("TRIAL_UUID");
    if (trial_env && *trial_env) trial_uuid_ = trim_copy(trial_env);
    const char* origin_env = std::getenv("JOB_ORIGIN");
    if (origin_env && *origin_env) job_origin_ = trim_copy(origin_env);

    std::filesystem::create_directories("runs");
    constexpr int kMaxAttempts = 64;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        const std::string candidate = make_run_id(model_family_);
        const std::string candidate_dir = join_path("runs", candidate);

        std::error_code ec;
        const bool created = std::filesystem::create_directory(candidate_dir, ec);
        if (created) {
            run_id_ = candidate;
            run_dir_ = candidate_dir;
            break;
        }
        if (ec && ec != std::errc::file_exists) {
            throw std::runtime_error(
                "RunLogger: failed to create run directory '" + candidate_dir + "': " + ec.message()
            );
        }
    }

    if (run_id_.empty() || run_dir_.empty()) {
        throw std::runtime_error("RunLogger: failed to allocate unique run_id after multiple attempts");
    }

    ensure_base_layout();
    write_manifest(false);
    write_runtime_heartbeat("running");
}

void RunLogger::ensure_base_layout() const {
    std::filesystem::create_directories(join_path(run_dir_, "learning"));
    std::filesystem::create_directories(join_path(run_dir_, "deployment"));
    std::filesystem::create_directories(join_path(run_dir_, "model_specific"));
    std::filesystem::create_directories(join_path(join_path(run_dir_, "model_specific"), model_family_));
    std::filesystem::create_directories(join_path(run_dir_, "runtime"));
}

void RunLogger::write_manifest(bool finished) {
    const std::string out_path = join_path(run_dir_, "manifest.json");
    const std::string tmp_path = out_path + ".tmp";

    std::ofstream f(tmp_path, std::ios::trunc);
    if (!f.is_open()) {
        throw std::runtime_error("RunLogger: cannot open manifest tmp file '" + tmp_path + "'");
    }

    f << "{\n";
    f << "  \"run_id\": \"" << json_escape(run_id_) << "\",\n";
    f << "  \"model_family\": \"" << json_escape(model_family_) << "\",\n";
    f << "  \"model_variant\": \"" << json_escape(model_variant_) << "\",\n";
    if (git_commit_.empty()) {
        f << "  \"git_commit\": null,\n";
    } else {
        f << "  \"git_commit\": \"" << json_escape(git_commit_) << "\",\n";
    }
    if (git_dirty_known_) {
        f << "  \"git_dirty\": " << (git_dirty_ ? "true" : "false") << ",\n";
    } else {
        f << "  \"git_dirty\": null,\n";
    }
    f << "  \"pid\": " << process_id_ << ",\n";
    f << "  \"seed\": " << seed_ << ",\n";
    f << "  \"train_start_utc\": \"" << json_escape(train_start_utc_) << "\",\n";
    if (finished) {
        f << "  \"train_end_utc\": \"" << json_escape(utc_now_iso8601()) << "\",\n";
    } else {
        f << "  \"train_end_utc\": null,\n";
    }
    f << "  \"data_version\": \"" << json_escape(data_version_) << "\",\n";
    f << "  \"benchmark_id\": \"" << json_escape(benchmark_id_) << "\",\n";
    f << "  \"benchmark_name\": \"" << json_escape(benchmark_name_) << "\",\n";
    f << "  \"task_type\": \"" << json_escape(task_type_) << "\",\n";
    if (!trial_uuid_.empty()) {
        f << "  \"trial_uuid\": \"" << json_escape(trial_uuid_) << "\",\n";
    } else {
        f << "  \"trial_uuid\": null,\n";
    }
    if (!job_origin_.empty()) {
        f << "  \"job_origin\": \"" << json_escape(job_origin_) << "\",\n";
    } else {
        f << "  \"job_origin\": null,\n";
    }
    f << "  \"params_valid_json\": " << (params_valid_json_ ? "true" : "false") << ",\n";
    if (params_valid_json_) {
        f << "  \"params\": " << params_json_;
    } else {
        f << "  \"params\": {},\n";
        f << "  \"params_raw\": \"" << json_escape(params_raw_json_) << "\",\n";
        f << "  \"params_parse_error\": \"" << json_escape(params_parse_error_) << "\"";
    }
    f << "\n}\n";
    f.close();

    std::error_code ec;
    std::filesystem::remove(out_path, ec);
    ec.clear();
    std::filesystem::rename(tmp_path, out_path, ec);
    if (ec) {
        throw std::runtime_error(
            "RunLogger: failed to finalize manifest '" + out_path + "': " + ec.message()
        );
    }
}

void RunLogger::write_runtime_heartbeat(const std::string& status) const {
    const std::string hb_path = join_path(join_path(run_dir_, "runtime"), "heartbeat.json");
    const std::string tmp_path = hb_path + ".tmp";
    std::ofstream f(tmp_path, std::ios::trunc);
    if (!f.is_open()) return;
    f << "{\n";
    f << "  \"run_id\": \"" << json_escape(run_id_) << "\",\n";
    f << "  \"pid\": " << process_id_ << ",\n";
    f << "  \"status\": \"" << json_escape(status) << "\",\n";
    f << "  \"started_utc\": \"" << json_escape(train_start_utc_) << "\",\n";
    f << "  \"last_update_utc\": \"" << json_escape(utc_now_iso8601()) << "\"\n";
    f << "}\n";
    f.close();
    std::error_code ec;
    std::filesystem::remove(hb_path, ec);
    ec.clear();
    std::filesystem::rename(tmp_path, hb_path, ec);
}

void RunLogger::write_manifest_end() {
    write_manifest(true);
    write_runtime_heartbeat("finished");
}

void RunLogger::write_header_if_needed(const std::string& path, const std::vector<std::string>& header) {
    if (wrote_header_[path]) return;
    if (std::filesystem::exists(path) && std::filesystem::file_size(path) > 0) {
        wrote_header_[path] = true;
        return;
    }
    append_line(path, header);
    wrote_header_[path] = true;
}

void RunLogger::append_line(const std::string& path, const std::vector<std::string>& fields) {
    std::ofstream f(path, std::ios::app);
    for (size_t i = 0; i < fields.size(); ++i) {
        if (i) f << ",";
        f << csv_escape(fields[i]);
    }
    f << "\n";
    write_runtime_heartbeat("running");
}

std::string RunLogger::csv_escape(const std::string& s) {
    bool needs_quotes = s.find(',') != std::string::npos ||
                        s.find('"') != std::string::npos ||
                        s.find('\n') != std::string::npos;
    if (!needs_quotes) return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += "\"";
    return out;
}

std::string RunLogger::json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20) {
                std::ostringstream esc;
                esc << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(c);
                out += esc.str();
            } else {
                out += static_cast<char>(c);
            }
            break;
        }
    }
    return out;
}

std::string RunLogger::to_s(double v) {
    std::ostringstream oss;
    oss << std::setprecision(12) << v;
    return oss.str();
}

std::string RunLogger::to_s(int v) {
    return std::to_string(v);
}

void RunLogger::log_epoch_metric(int epoch,
                                 const std::string& split,
                                 double loss,
                                 double accuracy,
                                 double precision_macro,
                                 double recall_macro,
                                 double f1_macro,
                                 double lr,
                                 double grad_norm_mean,
                                 double grad_norm_max,
                                 double param_norm_mean,
                                 double param_norm_max,
                                 double epoch_time_ms,
                                 double samples_per_sec) {
    const std::string path = join_path(join_path(run_dir_, "learning"), "epoch_metrics.csv");
    write_header_if_needed(path, {
        "run_id","model_family","model_variant","epoch","split","loss","accuracy",
        "precision_macro","recall_macro","f1_macro","lr","grad_norm_mean","grad_norm_max",
        "param_norm_mean","param_norm_max","epoch_time_ms","samples_per_sec"
    });
    append_line(path, {
        run_id_, model_family_, model_variant_, to_s(epoch), split, to_s(loss), to_s(accuracy),
        to_s(precision_macro), to_s(recall_macro), to_s(f1_macro), to_s(lr), to_s(grad_norm_mean),
        to_s(grad_norm_max), to_s(param_norm_mean), to_s(param_norm_max), to_s(epoch_time_ms),
        to_s(samples_per_sec)
    });
}

void RunLogger::log_class_metric(int epoch,
                                 const std::string& split,
                                 int class_id,
                                 double precision,
                                 double recall,
                                 double f1,
                                 int support) {
    const std::string path = join_path(join_path(run_dir_, "learning"), "class_metrics.csv");
    write_header_if_needed(path, {
        "run_id","epoch","split","class_id","precision","recall","f1","support"
    });
    append_line(path, {
        run_id_, to_s(epoch), split, to_s(class_id), to_s(precision), to_s(recall),
        to_s(f1), to_s(support)
    });
}

void RunLogger::log_confusion_cell(int epoch,
                                   const std::string& split,
                                   int true_class,
                                   int pred_class,
                                   int count) {
    const std::string path = join_path(join_path(run_dir_, "learning"), "confusion_matrix.csv");
    write_header_if_needed(path, {
        "run_id","epoch","split","true_class","pred_class","count"
    });
    append_line(path, {
        run_id_, to_s(epoch), split, to_s(true_class), to_s(pred_class), to_s(count)
    });
}

void RunLogger::log_inference_metric(const std::string& split,
                                     int sample_id,
                                     int true_class,
                                     int pred_class,
                                     double confidence,
                                     double entropy,
                                     double top2_margin,
                                     double latency_ms,
                                     int batch_size,
                                     int is_correct) {
    const std::string path = join_path(join_path(run_dir_, "deployment"), "inference_metrics.csv");
    write_header_if_needed(path, {
        "run_id","split","sample_id","true_class","pred_class","confidence","entropy",
        "top2_margin","latency_ms","batch_size","is_correct"
    });
    append_line(path, {
        run_id_, split, to_s(sample_id), to_s(true_class), to_s(pred_class), to_s(confidence),
        to_s(entropy), to_s(top2_margin), to_s(latency_ms), to_s(batch_size), to_s(is_correct)
    });
}

void RunLogger::log_calibration_bin(const std::string& split,
                                    int bin_id,
                                    double conf_low,
                                    double conf_high,
                                    int count,
                                    double avg_conf,
                                    double empirical_acc,
                                    double ece_contrib) {
    const std::string path = join_path(join_path(run_dir_, "deployment"), "calibration_bins.csv");
    write_header_if_needed(path, {
        "run_id","split","bin_id","conf_low","conf_high","count","avg_conf","empirical_acc","ece_contrib"
    });
    append_line(path, {
        run_id_, split, to_s(bin_id), to_s(conf_low), to_s(conf_high), to_s(count), to_s(avg_conf),
        to_s(empirical_acc), to_s(ece_contrib)
    });
}

void RunLogger::log_system_metric(const std::string& timestamp_utc,
                                  double qps,
                                  double p50_ms,
                                  double p95_ms,
                                  double p99_ms,
                                  double cpu_pct,
                                  double mem_mb) {
    const std::string path = join_path(join_path(run_dir_, "deployment"), "system_metrics.csv");
    write_header_if_needed(path, {
        "run_id","timestamp_utc","qps","p50_ms","p95_ms","p99_ms","cpu_pct","mem_mb"
    });
    append_line(path, {
        run_id_, timestamp_utc, to_s(qps), to_s(p50_ms), to_s(p95_ms), to_s(p99_ms),
        to_s(cpu_pct), to_s(mem_mb)
    });
}

void RunLogger::append_csv_row(const std::string& relative_csv_path,
                               const std::vector<std::string>& header,
                               const std::vector<std::string>& values) {
    const std::string path = join_path(run_dir_, relative_csv_path);
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    write_header_if_needed(path, header);
    append_line(path, values);
}

std::string RunLogger::utc_now_iso8601() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now();
    const std::time_t t = clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

std::string RunLogger::make_run_id(const std::string& model_family) {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto t = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    const auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000;
    const std::uint64_t seq = run_id_counter().fetch_add(1, std::memory_order_relaxed);

    std::ostringstream oss;
    oss << sanitize_token(model_family, "unknown") << "_";
    oss << std::put_time(&tm, "%Y%m%dT%H%M%S");
    oss << std::setw(3) << std::setfill('0') << ms;
    oss << "_p" << process_id();
    oss << "_c" << std::setw(6) << std::setfill('0') << (seq % 1000000ULL);
    oss << "_" << random_hex_suffix(8);
    return oss.str();
}
