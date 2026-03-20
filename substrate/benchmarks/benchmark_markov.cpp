#include "core/data/dataloader.h"
#include "core/io/run_logger.h"
#include "core/losses/losses.h"
#include "core/metrics/metrics.h"
#include "models/markov/src/markov_models.h"
#include "benchmarks/logging_utils.h"
#include "benchmarks/hpo_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

namespace {

struct EvalResult {
    double loss = std::numeric_limits<double>::quiet_NaN();
    double accuracy = std::numeric_limits<double>::quiet_NaN();
    double precision_macro = std::numeric_limits<double>::quiet_NaN();
    double recall_macro = std::numeric_limits<double>::quiet_NaN();
    double f1_macro = std::numeric_limits<double>::quiet_NaN();
    double elapsed_ms = 0.0;
    double samples_per_sec = 0.0;
    Tensor probs;
    std::vector<int> preds;
    std::vector<int> truths;
};

Tensor gather_rows(const Tensor& src, const std::vector<std::size_t>& indices) {
    Tensor out(indices.size(), src.cols, 0.0);
    for (std::size_t r = 0; r < indices.size(); ++r) {
        const std::size_t src_row = indices[r];
        for (std::size_t c = 0; c < src.cols; ++c) {
            out(r, c) = src(src_row, c);
        }
    }
    return out;
}

std::vector<int> gather_labels(const std::vector<int>& labels, const std::vector<std::size_t>& indices) {
    std::vector<int> out(indices.size(), 0);
    for (std::size_t i = 0; i < indices.size(); ++i) out[i] = labels[indices[i]];
    return out;
}

EvalResult evaluate_classifier(
    MarkovChainClassifier& model, const Tensor& x, const Tensor& y_one_hot, const std::vector<int>& labels
) {
    EvalResult out;
    if (x.rows == 0) return out;

    auto start = std::chrono::high_resolution_clock::now();
    out.probs = model.predict_proba(x);
    auto end = std::chrono::high_resolution_clock::now();
    out.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    out.samples_per_sec = x.rows / (out.elapsed_ms / 1000.0 + 1e-12);

    CrossEntropyLoss ce_loss;
    out.loss = ce_loss.forward(out.probs, y_one_hot);
    out.preds.resize(x.rows);
    for (std::size_t i = 0; i < out.probs.rows; ++i) {
        std::size_t best = 0;
        double best_v = out.probs(i, 0);
        for (std::size_t c = 1; c < out.probs.cols; ++c) {
            if (out.probs(i, c) > best_v) {
                best_v = out.probs(i, c);
                best = c;
            }
        }
        out.preds[i] = static_cast<int>(best);
    }
    out.truths = labels;
    out.accuracy = Metrics::accuracy(out.preds, out.truths);
    std::tie(out.precision_macro, out.recall_macro, out.f1_macro) =
        benchlog::macro_prf_from_preds(out.preds, out.truths, 10);
    return out;
}

void append_class_diagnostics(
    RunLogger& logger,
    int epoch,
    const std::vector<MarkovChainClassifier::ClassDiagnostics>& diagnostics
) {
    for (std::size_t class_id = 0; class_id < diagnostics.size(); ++class_id) {
        const auto& d = diagnostics[class_id];
        logger.append_csv_row(
            "model_specific/markov/class_state_metrics.csv",
            {
                "run_id",
                "epoch",
                "class_id",
                "sample_count",
                "prior",
                "transition_entropy",
                "context_coverage"
            },
            {
                logger.run_id(),
                std::to_string(epoch),
                std::to_string(class_id),
                std::to_string(d.sample_count),
                std::to_string(d.prior),
                std::to_string(d.transition_entropy),
                std::to_string(d.context_coverage)
            }
        );
    }
}

void append_symbol_emissions(RunLogger& logger, int epoch, const Tensor& class_bin_counts) {
    for (std::size_t class_id = 0; class_id < class_bin_counts.rows; ++class_id) {
        double row_total = 0.0;
        for (std::size_t bin_id = 0; bin_id < class_bin_counts.cols; ++bin_id) {
            row_total += class_bin_counts(class_id, bin_id);
        }
        const double inv_total = 1.0 / (row_total + 1e-12);

        for (std::size_t bin_id = 0; bin_id < class_bin_counts.cols; ++bin_id) {
            const double count = class_bin_counts(class_id, bin_id);
            logger.append_csv_row(
                "model_specific/markov/symbol_emissions.csv",
                {"run_id", "epoch", "class_id", "bin", "count", "fraction"},
                {
                    logger.run_id(),
                    std::to_string(epoch),
                    std::to_string(class_id),
                    std::to_string(bin_id),
                    std::to_string(count),
                    std::to_string(count * inv_total)
                }
            );
        }
    }
}

void append_position_nll(RunLogger& logger, int epoch, const std::vector<double>& nll) {
    for (std::size_t pos = 0; pos < nll.size(); ++pos) {
        logger.append_csv_row(
            "model_specific/markov/position_nll.csv",
            {"run_id", "epoch", "position", "row", "col", "nll"},
            {
                logger.run_id(),
                std::to_string(epoch),
                std::to_string(pos),
                std::to_string(pos / 28),
                std::to_string(pos % 28),
                std::to_string(nll[pos])
            }
        );
    }
}

void append_class_likelihood_summary(RunLogger& logger, const EvalResult& eval) {
    std::vector<double> true_prob_sum(10, 0.0);
    std::vector<double> conf_sum(10, 0.0);
    std::vector<double> correct_sum(10, 0.0);
    std::vector<int> class_count(10, 0);

    for (std::size_t i = 0; i < eval.preds.size(); ++i) {
        const int y = eval.truths[i];
        if (y < 0 || y >= 10) continue;
        const int pred = eval.preds[i];
        class_count[y] += 1;
        true_prob_sum[y] += eval.probs(i, static_cast<std::size_t>(y));
        conf_sum[y] += eval.probs(i, static_cast<std::size_t>(pred));
        correct_sum[y] += (pred == y) ? 1.0 : 0.0;
    }

    for (int c = 0; c < 10; ++c) {
        if (class_count[c] == 0) continue;
        const double inv = 1.0 / static_cast<double>(class_count[c]);
        logger.append_csv_row(
            "model_specific/markov/class_likelihoods.csv",
            {"run_id", "class_id", "support", "avg_true_prob", "avg_pred_confidence", "accuracy"},
            {
                logger.run_id(),
                std::to_string(c),
                std::to_string(class_count[c]),
                std::to_string(true_prob_sum[c] * inv),
                std::to_string(conf_sum[c] * inv),
                std::to_string(correct_sum[c] * inv)
            }
        );
    }
}

} // namespace

int main() {
    std::cout << "=== Markov Chain Benchmark (MNIST) ===" << std::endl;

    const int seed = hpo::env_int("MARKOV_SEED", 42);
    std::mt19937 rng(static_cast<unsigned int>(seed));

    const int order = std::max(1, hpo::env_int("MARKOV_ORDER", 2));
    const int bins = std::max(2, hpo::env_int("MARKOV_BINS", 8));
    const int epochs = std::max(1, hpo::env_int("MARKOV_EPOCHS", 8));
    const std::size_t train_cap = hpo::env_size("MARKOV_TRAIN_N", 30000);
    const std::size_t test_cap = hpo::env_size("MARKOV_TEST_N", 5000);
    const std::size_t nll_eval_n = hpo::env_size("MARKOV_NLL_EVAL_N", 1024);
    const double alpha = std::max(1e-8, hpo::env_double("MARKOV_ALPHA", 0.25));
    const bool shuffle = hpo::env_bool("MARKOV_SHUFFLE", true);
    const double pass_acc = hpo::env_double("MARKOV_PASS_ACC", 0.30);

    auto train = MNISTLoader::load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    auto test = MNISTLoader::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    const std::size_t train_n = std::min(train_cap, train.images.rows);
    const std::size_t test_n = std::min(test_cap, test.images.rows);
    if (train_n == 0 || test_n == 0) {
        std::cerr << "Empty dataset slice: train_n=" << train_n << " test_n=" << test_n << std::endl;
        return 2;
    }

    Tensor train_x = train.images.slice_rows(0, train_n);
    Tensor train_onehot = train.one_hot.slice_rows(0, train_n);
    std::vector<int> train_y(train.labels.begin(), train.labels.begin() + static_cast<long>(train_n));

    Tensor test_x = test.images.slice_rows(0, test_n);
    Tensor test_onehot = test.one_hot.slice_rows(0, test_n);
    std::vector<int> test_y(test.labels.begin(), test.labels.begin() + static_cast<long>(test_n));

    std::vector<std::size_t> train_perm(train_n);
    std::iota(train_perm.begin(), train_perm.end(), static_cast<std::size_t>(0));
    if (shuffle) std::shuffle(train_perm.begin(), train_perm.end(), rng);

    const std::size_t chunk_size = (train_n + static_cast<std::size_t>(epochs) - 1) / static_cast<std::size_t>(epochs);

    const std::string variant =
        "markov_order" + std::to_string(order) + "_bins" + std::to_string(bins) + "_e" + std::to_string(epochs);
    std::ostringstream params_json;
    params_json << "{"
                << "\"order\":" << order
                << ",\"bins\":" << bins
                << ",\"epochs\":" << epochs
                << ",\"alpha\":" << alpha
                << ",\"shuffle\":" << (shuffle ? "true" : "false")
                << ",\"train_n\":" << train_n
                << ",\"test_n\":" << test_n
                << ",\"nll_eval_n\":" << nll_eval_n
                << "}";

    RunLogger logger("markov", variant, seed, "mnist-idx-v1", params_json.str());
    benchlog::BenchBioHarness bio("markov", variant, seed);
    bio.attach({}, {}, "markov_param");

    MarkovChainClassifier model(10, bins, order, alpha);
    EvalResult final_eval;
    double best_acc = 0.0;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        bio.begin_epoch(epoch);
        const std::size_t begin = std::min(static_cast<std::size_t>(epoch - 1) * chunk_size, train_n);
        const std::size_t end = std::min(static_cast<std::size_t>(epoch) * chunk_size, train_n);

        std::vector<std::size_t> batch_indices;
        if (begin < end) {
            batch_indices.assign(train_perm.begin() + static_cast<long>(begin), train_perm.begin() + static_cast<long>(end));
        }

        auto train_start = std::chrono::high_resolution_clock::now();
        if (!batch_indices.empty()) {
            model.partial_fit(train_x, train_y, batch_indices);
        }
        auto train_end = std::chrono::high_resolution_clock::now();
        const double train_ms = std::chrono::duration<double, std::milli>(train_end - train_start).count();
        const double train_sps = batch_indices.empty() ? 0.0 : batch_indices.size() / (train_ms / 1000.0 + 1e-12);

        double train_loss = std::numeric_limits<double>::quiet_NaN();
        double train_acc = std::numeric_limits<double>::quiet_NaN();
        if (!batch_indices.empty()) {
            Tensor train_batch_x = gather_rows(train_x, batch_indices);
            Tensor train_batch_y = gather_rows(train_onehot, batch_indices);
            std::vector<int> train_batch_labels = gather_labels(train_y, batch_indices);
            EvalResult train_eval = evaluate_classifier(model, train_batch_x, train_batch_y, train_batch_labels);
            train_loss = train_eval.loss;
            train_acc = train_eval.accuracy;
        }

        EvalResult test_eval = evaluate_classifier(model, test_x, test_onehot, test_y);
        final_eval = test_eval;
        best_acc = std::max(best_acc, test_eval.accuracy);

        logger.log_epoch_metric(
            epoch,
            "train",
            train_loss,
            train_acc,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            0.0,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            train_ms,
            train_sps
        );
        logger.log_epoch_metric(
            epoch,
            "test",
            test_eval.loss,
            test_eval.accuracy,
            test_eval.precision_macro,
            test_eval.recall_macro,
            test_eval.f1_macro,
            0.0,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            test_eval.elapsed_ms,
            test_eval.samples_per_sec
        );
        benchlog::log_confusion_and_class_metrics(logger, epoch, "test", test_eval.preds, test_eval.truths, 10);

        append_class_diagnostics(logger, epoch, model.diagnostics());
        append_symbol_emissions(logger, epoch, model.class_bin_counts());

        const std::size_t nll_n = std::min(nll_eval_n, test_x.rows);
        Tensor nll_x = test_x.slice_rows(0, nll_n);
        std::vector<int> nll_labels(test_y.begin(), test_y.begin() + static_cast<long>(nll_n));
        append_position_nll(logger, epoch, model.position_nll(nll_x, nll_labels));

        logger.append_csv_row(
            "model_specific/markov/training_progress.csv",
            {"run_id", "epoch", "samples_seen", "chunk_samples", "test_loss", "test_accuracy", "num_contexts", "order", "bins"},
            {
                logger.run_id(),
                std::to_string(epoch),
                std::to_string(model.samples_seen()),
                std::to_string(batch_indices.size()),
                std::to_string(test_eval.loss),
                std::to_string(test_eval.accuracy),
                std::to_string(model.num_contexts()),
                std::to_string(model.order()),
                std::to_string(model.num_bins())
            }
        );

        bio.end_epoch(logger, epoch);

        std::cout << "Epoch " << epoch << "/" << epochs
                  << " | chunk=" << batch_indices.size()
                  << " | test_acc=" << std::fixed << std::setprecision(4) << test_eval.accuracy
                  << " | test_loss=" << std::setprecision(5) << test_eval.loss << std::endl;
    }

    const double per_sample_ms = final_eval.probs.rows
                                     ? final_eval.elapsed_ms / static_cast<double>(final_eval.probs.rows)
                                     : 0.0;
    benchlog::log_inference_from_probs(
        logger,
        "test",
        final_eval.probs,
        final_eval.preds,
        final_eval.truths,
        per_sample_ms,
        static_cast<int>(benchlog::resolve_eval_batch_size(256))
    );
    benchlog::log_calibration_from_probs(logger, "test", final_eval.probs, final_eval.preds, final_eval.truths, 10);
    logger.log_system_metric(
        RunLogger::utc_now_iso8601(),
        final_eval.samples_per_sec,
        per_sample_ms,
        per_sample_ms,
        per_sample_ms,
        0.0,
        0.0
    );
    append_class_likelihood_summary(logger, final_eval);
    logger.write_manifest_end();

    std::cout << "\n=== Markov Benchmark Complete ===" << std::endl;
    std::cout << "Best test accuracy: " << std::fixed << std::setprecision(4) << best_acc
              << " | threshold: " << pass_acc << std::endl;
    return best_acc >= pass_acc ? 0 : 1;
}
