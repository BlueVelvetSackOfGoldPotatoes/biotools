#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

class RunLogger {
public:
    RunLogger(const std::string& model_family,
              const std::string& model_variant,
              int seed,
              const std::string& data_version,
              const std::string& params_json = "{}",
              const std::string& benchmark_id = "",
              const std::string& benchmark_name = "",
              const std::string& task_type = "");

    const std::string& run_id() const { return run_id_; }
    const std::string& run_dir() const { return run_dir_; }

    void write_manifest_end();

    void log_epoch_metric(int epoch,
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
                          double samples_per_sec);

    void log_class_metric(int epoch,
                          const std::string& split,
                          int class_id,
                          double precision,
                          double recall,
                          double f1,
                          int support);

    void log_confusion_cell(int epoch,
                            const std::string& split,
                            int true_class,
                            int pred_class,
                            int count);

    void log_inference_metric(const std::string& split,
                              int sample_id,
                              int true_class,
                              int pred_class,
                              double confidence,
                              double entropy,
                              double top2_margin,
                              double latency_ms,
                              int batch_size,
                              int is_correct);

    void log_calibration_bin(const std::string& split,
                             int bin_id,
                             double conf_low,
                             double conf_high,
                             int count,
                             double avg_conf,
                             double empirical_acc,
                             double ece_contrib);

    void log_system_metric(const std::string& timestamp_utc,
                           double qps,
                           double p50_ms,
                           double p95_ms,
                           double p99_ms,
                           double cpu_pct,
                           double mem_mb);

    // Generic helper for model-specific logging.
    void append_csv_row(const std::string& relative_csv_path,
                        const std::vector<std::string>& header,
                        const std::vector<std::string>& values);

    static std::string utc_now_iso8601();
    static std::string make_run_id(const std::string& model_family);

private:
    std::string run_id_;
    std::string run_dir_;
    std::string model_family_;
    std::string model_variant_;
    int seed_;
    std::string data_version_;
    std::string benchmark_id_;
    std::string benchmark_name_;
    std::string task_type_;
    std::string params_json_;
    std::string train_start_utc_;
    std::map<std::string, bool> wrote_header_;

    void ensure_base_layout() const;
    void write_header_if_needed(const std::string& path, const std::vector<std::string>& header);
    void append_line(const std::string& path, const std::vector<std::string>& fields);
    static std::string csv_escape(const std::string& s);
    static std::string json_escape(const std::string& s);
    static std::string to_s(double v);
    static std::string to_s(int v);
    void write_manifest(bool finished);
    void write_runtime_heartbeat(const std::string& status) const;

    std::string git_commit_;
    bool git_dirty_known_ = false;
    bool git_dirty_ = false;
    bool params_valid_json_ = true;
    std::string params_raw_json_;
    std::string params_parse_error_;
    std::string trial_uuid_;
    std::string job_origin_;
    std::int64_t process_id_ = 0;
};
