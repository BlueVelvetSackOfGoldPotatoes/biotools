#pragma once

#include "core/bio/bio_runtime.h"
#include "core/io/run_logger.h"
#include "core/losses/losses.h"
#include "core/metrics/metrics.h"
#include "hpo_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace benchlog {

inline std::vector<std::string> default_param_names(std::size_t n, const std::string& prefix = "param") {
    std::vector<std::string> names;
    names.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        names.push_back(prefix + "_" + std::to_string(i));
    }
    return names;
}

class BenchBioHarness {
public:
    BenchBioHarness(const std::string& family, const std::string& variant, int seed = 42)
        : runtime_(family, variant, seed, make_config()) {}

    void attach(const std::vector<Tensor*>& params,
                const std::vector<Tensor*>& grads,
                const std::string& prefix = "param") {
        runtime_.attach(params, grads, default_param_names(params.size(), prefix));
    }

    void begin_epoch(int epoch) { runtime_.begin_epoch(epoch); }
    void before_forward() { runtime_.before_forward(); }
    void after_backward(double loss,
                        double reward = std::numeric_limits<double>::quiet_NaN(),
                        double lr = 0.0) {
        runtime_.after_backward(loss, reward, lr);
    }
    void after_optimizer_step() { runtime_.after_optimizer_step(); }
    void end_epoch(RunLogger& logger, int epoch) {
        auto snapshot = runtime_.end_epoch(epoch);
        runtime_.log_epoch(logger, epoch, snapshot);
    }

private:
    static bio::BioConfig make_config() {
        bio::BioConfig cfg;
        auto clampd = [](double v, double lo, double hi) {
            return std::max(lo, std::min(hi, v));
        };

        cfg.active_parameter_updates = hpo::env_bool("BIO_ACTIVE", cfg.active_parameter_updates);

        cfg.enable_structural_plasticity =
            hpo::env_bool("BIO_ENABLE_STRUCTURAL_PLASTICITY", cfg.enable_structural_plasticity);
        cfg.enable_homeostasis = hpo::env_bool("BIO_ENABLE_HOMEOSTASIS", cfg.enable_homeostasis);
        cfg.enable_three_factor = hpo::env_bool("BIO_ENABLE_THREE_FACTOR", cfg.enable_three_factor);
        cfg.enable_myelination = hpo::env_bool("BIO_ENABLE_MYELINATION", cfg.enable_myelination);
        cfg.enable_bioelectric = hpo::env_bool("BIO_ENABLE_BIOELECTRIC", cfg.enable_bioelectric);
        cfg.enable_ei_sign_constraints =
            hpo::env_bool("BIO_ENABLE_EI_SIGN_CONSTRAINTS", cfg.enable_ei_sign_constraints);
        const std::string ei_axis = hpo::to_lower(hpo::env_string("BIO_EI_AXIS", "source"));
        cfg.ei_constraint_axis =
            (ei_axis == "destination" || ei_axis == "dst") ? bio::EiConstraintAxis::Destination
                                                             : bio::EiConstraintAxis::Source;
        cfg.ei_skip_input_layer = hpo::env_bool("BIO_EI_SKIP_INPUT_LAYER", cfg.ei_skip_input_layer);
        cfg.ei_skip_output_layer = hpo::env_bool("BIO_EI_SKIP_OUTPUT_LAYER", cfg.ei_skip_output_layer);
        cfg.ei_projection_strength =
            clampd(hpo::env_double("BIO_EI_PROJECTION_STRENGTH", cfg.ei_projection_strength), 0.0, 1.0);
        cfg.ei_excitatory_probability =
            clampd(hpo::env_double("BIO_EI_EXCITATORY_PROB", cfg.ei_excitatory_probability), 0.0, 1.0);

        cfg.rewire_every_epochs =
            std::max(1, hpo::env_int("BIO_REWIRE_EVERY_EPOCHS", cfg.rewire_every_epochs));
        cfg.myelin_every_epochs =
            std::max(1, hpo::env_int("BIO_MYELIN_EVERY_EPOCHS", cfg.myelin_every_epochs));
        cfg.prune_fraction = clampd(hpo::env_double("BIO_PRUNE_FRACTION", cfg.prune_fraction), 0.0, 0.95);
        cfg.growth_noise_std = std::max(0.0, hpo::env_double("BIO_GROWTH_NOISE_STD", cfg.growth_noise_std));
        cfg.structural_sample_cap = hpo::env_size("BIO_STRUCTURAL_SAMPLE_CAP", cfg.structural_sample_cap);

        cfg.eligibility_decay =
            clampd(hpo::env_double("BIO_ELIGIBILITY_DECAY", cfg.eligibility_decay), 0.0, 1.0);
        cfg.three_factor_lr = std::max(0.0, hpo::env_double("BIO_THREE_FACTOR_LR", cfg.three_factor_lr));
        cfg.modulator_clip = std::max(0.0, hpo::env_double("BIO_MODULATOR_CLIP", cfg.modulator_clip));

        cfg.homeostasis_target_norm =
            std::max(1e-8, hpo::env_double("BIO_HOMEOSTASIS_TARGET_NORM", cfg.homeostasis_target_norm));
        cfg.homeostasis_lr =
            clampd(hpo::env_double("BIO_HOMEOSTASIS_LR", cfg.homeostasis_lr), 0.0, 1.0);

        cfg.myelin_budget_fraction =
            clampd(hpo::env_double("BIO_MYELIN_BUDGET_FRACTION", cfg.myelin_budget_fraction), 0.0, 1.0);
        cfg.conduction_base_speed =
            std::max(1e-8, hpo::env_double("BIO_CONDUCTION_BASE_SPEED", cfg.conduction_base_speed));
        cfg.conduction_myelin_boost =
            std::max(0.0, hpo::env_double("BIO_CONDUCTION_MYELIN_BOOST", cfg.conduction_myelin_boost));
        cfg.conduction_delay_bins =
            std::max(1, hpo::env_int("BIO_CONDUCTION_DELAY_BINS", cfg.conduction_delay_bins));
        cfg.conduction_delay_mix =
            clampd(hpo::env_double("BIO_CONDUCTION_DELAY_MIX", cfg.conduction_delay_mix), 0.0, 1.0);

        cfg.bioelectric_leak = clampd(hpo::env_double("BIO_BIOELECTRIC_LEAK", cfg.bioelectric_leak), 0.0, 1.0);
        cfg.bioelectric_coupling =
            std::max(0.0, hpo::env_double("BIO_BIOELECTRIC_COUPLING", cfg.bioelectric_coupling));

        cfg.wiring_distance_cost =
            std::max(0.0, hpo::env_double("BIO_WIRING_DISTANCE_COST", cfg.wiring_distance_cost));
        cfg.wiring_cost_sample_stride =
            std::max<std::size_t>(1, hpo::env_size("BIO_WIRING_COST_SAMPLE_STRIDE", cfg.wiring_cost_sample_stride));
        return cfg;
    }

    bio::BioRuntime runtime_;
};

inline std::pair<double, double> norm_stats(const std::vector<Tensor*>& ts) {
    if (ts.empty()) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan};
    }
    double sum = 0.0;
    double mx = 0.0;
    for (const auto* t : ts) {
        const double n = t->norm();
        sum += n;
        mx = std::max(mx, n);
    }
    return {sum / static_cast<double>(ts.size()), mx};
}

inline void argmax_rows_to_vec(const Tensor& t, std::vector<int>& out) {
    out.resize(t.rows);
    for (size_t i = 0; i < t.rows; ++i) {
        size_t idx = 0;
        double best = t(i, 0);
        for (size_t j = 1; j < t.cols; ++j) {
            if (t(i, j) > best) {
                best = t(i, j);
                idx = j;
            }
        }
        out[i] = static_cast<int>(idx);
    }
}

inline std::tuple<double, double, double> macro_prf_from_preds(const std::vector<int>& preds,
                                                                const std::vector<int>& truths,
                                                                int num_classes);

inline void log_confusion_and_class_metrics(RunLogger& logger,
                                            int epoch,
                                            const std::string& split,
                                            const std::vector<int>& preds,
                                            const std::vector<int>& labels,
                                            int num_classes) {
    Tensor cm = Metrics::confusion_matrix(preds, labels, num_classes);
    auto per_class = Metrics::per_class_metrics(cm);
    for (int c = 0; c < num_classes; ++c) {
        int support = 0;
        for (int j = 0; j < num_classes; ++j) {
            support += static_cast<int>(cm(c, j));
        }
        logger.log_class_metric(
            epoch, split, c, per_class[c].precision, per_class[c].recall, per_class[c].f1, support
        );
    }

    for (int i = 0; i < num_classes; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            logger.log_confusion_cell(epoch, split, i, j, static_cast<int>(cm(i, j)));
        }
    }
}

inline void log_calibration_from_probs(RunLogger& logger,
                                       const std::string& split,
                                       const Tensor& probs,
                                       const std::vector<int>& preds,
                                       const std::vector<int>& truths,
                                       int n_bins = 10) {
    std::vector<int> counts(n_bins, 0);
    std::vector<double> sum_conf(n_bins, 0.0);
    std::vector<double> sum_acc(n_bins, 0.0);

    for (size_t i = 0; i < probs.rows; ++i) {
        const double conf = probs(i, preds[i]);
        const int bin = std::min(n_bins - 1, static_cast<int>(conf * n_bins));
        counts[bin]++;
        sum_conf[bin] += conf;
        sum_acc[bin] += (preds[i] == truths[i]) ? 1.0 : 0.0;
    }

    const double n = static_cast<double>(probs.rows);
    for (int b = 0; b < n_bins; ++b) {
        const double low = static_cast<double>(b) / static_cast<double>(n_bins);
        const double high = static_cast<double>(b + 1) / static_cast<double>(n_bins);
        const double avg_conf = counts[b] ? sum_conf[b] / counts[b] : 0.0;
        const double empirical_acc = counts[b] ? sum_acc[b] / counts[b] : 0.0;
        const double ece_contrib = (counts[b] / n) * std::abs(avg_conf - empirical_acc);
        logger.log_calibration_bin(split, b, low, high, counts[b], avg_conf, empirical_acc, ece_contrib);
    }
}

inline void log_calibration_from_preds(RunLogger& logger,
                                       const std::string& split,
                                       const std::vector<int>& preds,
                                       const std::vector<int>& truths,
                                       int n_bins = 10) {
    std::vector<int> counts(n_bins, 0);
    std::vector<double> sum_conf(n_bins, 0.0);
    std::vector<double> sum_acc(n_bins, 0.0);

    // Label-only models are treated as deterministic confidence=1.0.
    for (size_t i = 0; i < preds.size(); ++i) {
        const double conf = 1.0;
        const int bin = n_bins - 1;
        counts[bin]++;
        sum_conf[bin] += conf;
        sum_acc[bin] += (preds[i] == truths[i]) ? 1.0 : 0.0;
    }

    const double n = std::max(1.0, static_cast<double>(preds.size()));
    for (int b = 0; b < n_bins; ++b) {
        const double low = static_cast<double>(b) / static_cast<double>(n_bins);
        const double high = static_cast<double>(b + 1) / static_cast<double>(n_bins);
        const double avg_conf = counts[b] ? sum_conf[b] / counts[b] : 0.0;
        const double empirical_acc = counts[b] ? sum_acc[b] / counts[b] : 0.0;
        const double ece_contrib = (counts[b] / n) * std::abs(avg_conf - empirical_acc);
        logger.log_calibration_bin(split, b, low, high, counts[b], avg_conf, empirical_acc, ece_contrib);
    }
}

inline void log_inference_from_probs(RunLogger& logger,
                                     const std::string& split,
                                     const Tensor& probs,
                                     const std::vector<int>& preds,
                                     const std::vector<int>& truths,
                                     double per_sample_ms,
                                     int batch_size) {
    for (size_t i = 0; i < probs.rows; ++i) {
        const double conf = probs(i, preds[i]);
        double top2 = 0.0;
        for (size_t j = 0; j < probs.cols; ++j) {
            if (static_cast<int>(j) == preds[i]) continue;
            top2 = std::max(top2, probs(i, j));
        }

        double entropy = 0.0;
        for (size_t j = 0; j < probs.cols; ++j) {
            const double p = probs(i, j);
            entropy -= p * std::log(p + 1e-12);
        }

        logger.log_inference_metric(
            split,
            static_cast<int>(i),
            truths[i],
            preds[i],
            conf,
            entropy,
            conf - top2,
            per_sample_ms,
            batch_size,
            preds[i] == truths[i] ? 1 : 0
        );
    }
}

inline void log_inference_from_preds(RunLogger& logger,
                                     const std::string& split,
                                     const std::vector<int>& preds,
                                     const std::vector<int>& truths,
                                     double per_sample_ms,
                                     int batch_size) {
    for (size_t i = 0; i < preds.size(); ++i) {
        const int is_correct = preds[i] == truths[i] ? 1 : 0;
        logger.log_inference_metric(
            split,
            static_cast<int>(i),
            truths[i],
            preds[i],
            1.0,
            0.0,
            1.0,
            per_sample_ms,
            batch_size,
            is_correct
        );
    }
}

inline std::size_t resolve_eval_batch_size(std::size_t fallback) {
    const std::size_t safe_default = std::max<std::size_t>(1, std::min<std::size_t>(fallback, 512));
    const char* env = std::getenv("EVAL_BATCH_SIZE");
    if (!env || env[0] == '\0') return safe_default;

    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (end == env || parsed == 0) return safe_default;
    return static_cast<std::size_t>(std::min<unsigned long long>(parsed, 2048ULL));
}

struct BatchedEvalResult {
    double loss = std::numeric_limits<double>::quiet_NaN();
    double accuracy = std::numeric_limits<double>::quiet_NaN();
    double precision_macro = std::numeric_limits<double>::quiet_NaN();
    double recall_macro = std::numeric_limits<double>::quiet_NaN();
    double f1_macro = std::numeric_limits<double>::quiet_NaN();
    double elapsed_ms = 0.0;
    double per_sample_ms = 0.0;
    double samples_per_sec = 0.0;
    std::vector<int> preds;
    std::vector<int> truths;
};

template <typename ModelT>
BatchedEvalResult evaluate_classification_batched(ModelT& model,
                                                  const Tensor& images,
                                                  const Tensor& labels_one_hot,
                                                  Loss& loss_fn,
                                                  std::size_t batch_size,
                                                  RunLogger* logger = nullptr,
                                                  const std::string& split = "test",
                                                  bool log_inference = false,
                                                  bool log_calibration = false,
                                                  int n_bins = 10) {
    BatchedEvalResult out;
    const std::size_t n = std::min(images.rows, labels_one_hot.rows);
    if (n == 0) return out;

    const std::size_t bs_eval = std::max<std::size_t>(1, std::min(batch_size, n));
    const int bins = std::max(1, n_bins);
    std::vector<int> calib_counts(bins, 0);
    std::vector<double> calib_sum_conf(bins, 0.0);
    std::vector<double> calib_sum_acc(bins, 0.0);

    struct InferenceRow {
        int sample_idx = 0;
        int truth = 0;
        int pred = 0;
        double conf = 0.0;
        double entropy = 0.0;
        double margin = 0.0;
        int is_correct = 0;
    };
    std::vector<InferenceRow> inference_rows;
    if (logger && log_inference) inference_rows.reserve(n);

    out.preds.reserve(n);
    out.truths.reserve(n);

    auto t0 = std::chrono::high_resolution_clock::now();
    double weighted_loss_sum = 0.0;
    std::size_t correct = 0;

    for (std::size_t start = 0; start < n; start += bs_eval) {
        const std::size_t cur = std::min(bs_eval, n - start);
        Tensor bx = images.slice_rows(start, cur);
        Tensor by = labels_one_hot.slice_rows(start, cur);

        Tensor logits = model.forward(bx);
        Tensor probs = LossFunctions::softmax(logits);
        weighted_loss_sum += loss_fn.forward(probs, by) * static_cast<double>(cur);

        std::vector<int> batch_preds;
        std::vector<int> batch_truths;
        argmax_rows_to_vec(probs, batch_preds);
        argmax_rows_to_vec(by, batch_truths);

        for (std::size_t i = 0; i < cur; ++i) {
            const int pred = batch_preds[i];
            const int truth = batch_truths[i];
            out.preds.push_back(pred);
            out.truths.push_back(truth);
            const bool is_correct = pred == truth;
            if (is_correct) correct++;

            const double conf = probs(i, static_cast<std::size_t>(pred));
            if (log_calibration) {
                const int bin = std::min(bins - 1, static_cast<int>(conf * bins));
                calib_counts[bin]++;
                calib_sum_conf[bin] += conf;
                calib_sum_acc[bin] += is_correct ? 1.0 : 0.0;
            }

            if (logger && log_inference) {
                double top2 = 0.0;
                double entropy = 0.0;
                for (std::size_t j = 0; j < probs.cols; ++j) {
                    const double p = probs(i, j);
                    entropy -= p * std::log(p + 1e-12);
                    if (static_cast<int>(j) == pred) continue;
                    top2 = std::max(top2, p);
                }
                inference_rows.push_back(
                    InferenceRow{
                        static_cast<int>(start + i),
                        truth,
                        pred,
                        conf,
                        entropy,
                        conf - top2,
                        is_correct ? 1 : 0
                    }
                );
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    out.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    out.per_sample_ms = out.elapsed_ms / static_cast<double>(n);
    out.samples_per_sec = n / (out.elapsed_ms / 1000.0 + 1e-12);
    out.loss = weighted_loss_sum / static_cast<double>(n);
    out.accuracy = static_cast<double>(correct) / static_cast<double>(n);
    std::tie(out.precision_macro, out.recall_macro, out.f1_macro) =
        macro_prf_from_preds(out.preds, out.truths, static_cast<int>(labels_one_hot.cols));

    if (logger && log_inference) {
        for (const auto& row : inference_rows) {
            logger->log_inference_metric(
                split,
                row.sample_idx,
                row.truth,
                row.pred,
                row.conf,
                row.entropy,
                row.margin,
                out.per_sample_ms,
                static_cast<int>(bs_eval),
                row.is_correct
            );
        }
    }

    if (logger && log_calibration) {
        const double n_d = static_cast<double>(n);
        for (int b = 0; b < bins; ++b) {
            const double low = static_cast<double>(b) / static_cast<double>(bins);
            const double high = static_cast<double>(b + 1) / static_cast<double>(bins);
            const double avg_conf = calib_counts[b] ? calib_sum_conf[b] / calib_counts[b] : 0.0;
            const double empirical_acc = calib_counts[b] ? calib_sum_acc[b] / calib_counts[b] : 0.0;
            const double ece_contrib = (calib_counts[b] / n_d) * std::abs(avg_conf - empirical_acc);
            logger->log_calibration_bin(split, b, low, high, calib_counts[b], avg_conf, empirical_acc, ece_contrib);
        }
    }

    return out;
}

inline std::tuple<double, double, double> macro_prf_from_preds(const std::vector<int>& preds,
                                                                const std::vector<int>& truths,
                                                                int num_classes) {
    Tensor cm = Metrics::confusion_matrix(preds, truths, num_classes);
    auto class_metrics = Metrics::per_class_metrics(cm);
    double precision_macro = 0.0;
    double recall_macro = 0.0;
    for (const auto& m : class_metrics) {
        precision_macro += m.precision;
        recall_macro += m.recall;
    }
    precision_macro /= static_cast<double>(class_metrics.size());
    recall_macro /= static_cast<double>(class_metrics.size());
    const double f1_macro = Metrics::macro_f1(cm);
    return {precision_macro, recall_macro, f1_macro};
}

} // namespace benchlog
