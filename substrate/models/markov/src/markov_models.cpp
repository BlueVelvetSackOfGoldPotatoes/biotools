#include "markov_models.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#ifdef _OPENMP
#define MARKOV_OMP_PAR_FOR _Pragma("omp parallel for schedule(static)")
#else
#define MARKOV_OMP_PAR_FOR
#endif

MarkovChainClassifier::MarkovChainClassifier(int num_classes, int num_bins, int order, double alpha)
    : num_classes_(num_classes), num_bins_(num_bins), order_(order), alpha_(alpha) {
    if (num_classes_ < 2) throw std::runtime_error("MarkovChainClassifier: num_classes must be >= 2");
    if (num_bins_ < 2) throw std::runtime_error("MarkovChainClassifier: num_bins must be >= 2");
    if (order_ < 1) throw std::runtime_error("MarkovChainClassifier: order must be >= 1");
    if (!(alpha_ > 0.0)) throw std::runtime_error("MarkovChainClassifier: alpha must be > 0");

    num_contexts_ = 1;
    for (int i = 0; i < order_; ++i) {
        if (num_contexts_ > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(num_bins_)) {
            throw std::runtime_error("MarkovChainClassifier: context space overflow");
        }
        num_contexts_ *= static_cast<std::size_t>(num_bins_);
    }
    context_shift_mod_ = num_contexts_ / static_cast<std::size_t>(num_bins_);

    transition_counts_.assign(
        static_cast<std::size_t>(num_classes_) * num_contexts_ * static_cast<std::size_t>(num_bins_), 0.0
    );
    context_totals_.assign(static_cast<std::size_t>(num_classes_) * num_contexts_, 0.0);
    class_counts_.assign(static_cast<std::size_t>(num_classes_), 0.0);
}

void MarkovChainClassifier::reset() {
    std::fill(transition_counts_.begin(), transition_counts_.end(), 0.0);
    std::fill(context_totals_.begin(), context_totals_.end(), 0.0);
    std::fill(class_counts_.begin(), class_counts_.end(), 0.0);
    samples_seen_ = 0;
}

int MarkovChainClassifier::quantize(double value) const {
    const double v = std::clamp(value, 0.0, 1.0);
    int bin = static_cast<int>(v * static_cast<double>(num_bins_));
    if (bin >= num_bins_) bin = num_bins_ - 1;
    if (bin < 0) bin = 0;
    return bin;
}

std::size_t MarkovChainClassifier::shift_context(std::size_t context, int symbol) const {
    return ((context % context_shift_mod_) * static_cast<std::size_t>(num_bins_)) +
           static_cast<std::size_t>(symbol);
}

std::size_t MarkovChainClassifier::context_offset(int class_id, std::size_t context) const {
    return static_cast<std::size_t>(class_id) * num_contexts_ + context;
}

std::size_t MarkovChainClassifier::transition_offset(int class_id, std::size_t context, int symbol) const {
    return (static_cast<std::size_t>(class_id) * num_contexts_ + context) * static_cast<std::size_t>(num_bins_) +
           static_cast<std::size_t>(symbol);
}

void MarkovChainClassifier::update_row(const Tensor& X, std::size_t row, int class_id) {
    if (class_id < 0 || class_id >= num_classes_) return;
    class_counts_[static_cast<std::size_t>(class_id)] += 1.0;
    samples_seen_ += 1;

    std::size_t context = 0;
    for (std::size_t col = 0; col < X.cols; ++col) {
        const int symbol = quantize(X(row, col));
        transition_counts_[transition_offset(class_id, context, symbol)] += 1.0;
        context_totals_[context_offset(class_id, context)] += 1.0;
        context = shift_context(context, symbol);
    }
}

void MarkovChainClassifier::partial_fit(
    const Tensor& X, const std::vector<int>& y, const std::vector<std::size_t>& indices
) {
    if (X.rows != y.size()) throw std::runtime_error("MarkovChainClassifier::partial_fit: X/y size mismatch");
    for (std::size_t idx : indices) {
        if (idx >= X.rows) continue;
        update_row(X, idx, y[idx]);
    }
}

void MarkovChainClassifier::partial_fit(const Tensor& X, const std::vector<int>& y, std::size_t start, std::size_t count) {
    if (X.rows != y.size()) throw std::runtime_error("MarkovChainClassifier::partial_fit: X/y size mismatch");
    const std::size_t begin = std::min(start, X.rows);
    const std::size_t end = std::min(X.rows, begin + count);
    for (std::size_t i = begin; i < end; ++i) {
        update_row(X, i, y[i]);
    }
}

void MarkovChainClassifier::log_scores_for_row(
    const Tensor& X, std::size_t row, std::vector<double>& out_scores
) const {
    out_scores.assign(static_cast<std::size_t>(num_classes_), 0.0);

    const double total_samples = std::accumulate(class_counts_.begin(), class_counts_.end(), 0.0);
    const double prior_den = total_samples + alpha_ * static_cast<double>(num_classes_);

    for (int c = 0; c < num_classes_; ++c) {
        const double prior_num = class_counts_[static_cast<std::size_t>(c)] + alpha_;
        double logp = std::log(prior_num / prior_den);

        std::size_t context = 0;
        for (std::size_t col = 0; col < X.cols; ++col) {
            const int symbol = quantize(X(row, col));
            const double count = transition_counts_[transition_offset(c, context, symbol)];
            const double total = context_totals_[context_offset(c, context)];
            const double prob =
                (count + alpha_) / (total + alpha_ * static_cast<double>(num_bins_));
            logp += std::log(prob);
            context = shift_context(context, symbol);
        }
        out_scores[static_cast<std::size_t>(c)] = logp;
    }
}

Tensor MarkovChainClassifier::predict_log_scores(const Tensor& X) const {
    Tensor out(X.rows, static_cast<std::size_t>(num_classes_), 0.0);
    MARKOV_OMP_PAR_FOR
    for (std::size_t i = 0; i < X.rows; ++i) {
        std::vector<double> scores(static_cast<std::size_t>(num_classes_), 0.0);
        log_scores_for_row(X, i, scores);
        for (int c = 0; c < num_classes_; ++c) out(i, static_cast<std::size_t>(c)) = scores[static_cast<std::size_t>(c)];
    }
    return out;
}

Tensor MarkovChainClassifier::predict_proba(const Tensor& X) const {
    Tensor log_scores = predict_log_scores(X);
    Tensor probs(log_scores.rows, log_scores.cols, 0.0);

    MARKOV_OMP_PAR_FOR
    for (std::size_t i = 0; i < log_scores.rows; ++i) {
        double max_v = log_scores(i, 0);
        for (std::size_t c = 1; c < log_scores.cols; ++c) max_v = std::max(max_v, log_scores(i, c));

        double sum = 0.0;
        for (std::size_t c = 0; c < log_scores.cols; ++c) {
            const double e = std::exp(log_scores(i, c) - max_v);
            probs(i, c) = e;
            sum += e;
        }
        const double inv = 1.0 / (sum + 1e-12);
        for (std::size_t c = 0; c < log_scores.cols; ++c) probs(i, c) *= inv;
    }

    return probs;
}

std::vector<int> MarkovChainClassifier::predict(const Tensor& X) const {
    Tensor scores = predict_log_scores(X);
    std::vector<int> preds(X.rows, 0);

    MARKOV_OMP_PAR_FOR
    for (std::size_t i = 0; i < scores.rows; ++i) {
        std::size_t best_idx = 0;
        double best_val = scores(i, 0);
        for (std::size_t c = 1; c < scores.cols; ++c) {
            if (scores(i, c) > best_val) {
                best_val = scores(i, c);
                best_idx = c;
            }
        }
        preds[i] = static_cast<int>(best_idx);
    }

    return preds;
}

std::vector<MarkovChainClassifier::ClassDiagnostics> MarkovChainClassifier::diagnostics() const {
    std::vector<ClassDiagnostics> out(static_cast<std::size_t>(num_classes_));
    const double total_samples = std::accumulate(class_counts_.begin(), class_counts_.end(), 0.0);

    for (int c = 0; c < num_classes_; ++c) {
        ClassDiagnostics d;
        d.sample_count = class_counts_[static_cast<std::size_t>(c)];
        d.prior = (d.sample_count + alpha_) / (total_samples + alpha_ * static_cast<double>(num_classes_));

        double weighted_entropy = 0.0;
        double weighted_total = 0.0;
        std::size_t covered = 0;
        for (std::size_t ctx = 0; ctx < num_contexts_; ++ctx) {
            const double ctx_total = context_totals_[context_offset(c, ctx)];
            if (ctx_total > 0.0) covered++;
            if (ctx_total <= 0.0) continue;

            const double den = ctx_total + alpha_ * static_cast<double>(num_bins_);
            double h = 0.0;
            for (int b = 0; b < num_bins_; ++b) {
                const double count = transition_counts_[transition_offset(c, ctx, b)];
                const double p = (count + alpha_) / den;
                h -= p * std::log(p + 1e-12);
            }
            weighted_entropy += ctx_total * h;
            weighted_total += ctx_total;
        }

        d.transition_entropy = weighted_total > 0.0 ? (weighted_entropy / weighted_total) : 0.0;
        d.context_coverage =
            num_contexts_ > 0 ? static_cast<double>(covered) / static_cast<double>(num_contexts_) : 0.0;
        out[static_cast<std::size_t>(c)] = d;
    }

    return out;
}

std::vector<double> MarkovChainClassifier::position_nll(const Tensor& X, const std::vector<int>& y) const {
    if (X.rows != y.size()) throw std::runtime_error("MarkovChainClassifier::position_nll: X/y size mismatch");
    std::vector<double> nll_sum(X.cols, 0.0);
    std::vector<double> nll_count(X.cols, 0.0);

    for (std::size_t row = 0; row < X.rows; ++row) {
        const int class_id = y[row];
        if (class_id < 0 || class_id >= num_classes_) continue;

        std::size_t context = 0;
        for (std::size_t col = 0; col < X.cols; ++col) {
            const int symbol = quantize(X(row, col));
            const double count = transition_counts_[transition_offset(class_id, context, symbol)];
            const double total = context_totals_[context_offset(class_id, context)];
            const double prob =
                (count + alpha_) / (total + alpha_ * static_cast<double>(num_bins_));
            nll_sum[col] += -std::log(prob + 1e-12);
            nll_count[col] += 1.0;
            context = shift_context(context, symbol);
        }
    }

    std::vector<double> out(X.cols, 0.0);
    for (std::size_t col = 0; col < X.cols; ++col) {
        out[col] = nll_count[col] > 0.0 ? (nll_sum[col] / nll_count[col]) : 0.0;
    }
    return out;
}

Tensor MarkovChainClassifier::class_bin_counts() const {
    Tensor out(static_cast<std::size_t>(num_classes_), static_cast<std::size_t>(num_bins_), 0.0);
    for (int c = 0; c < num_classes_; ++c) {
        for (std::size_t ctx = 0; ctx < num_contexts_; ++ctx) {
            for (int b = 0; b < num_bins_; ++b) {
                out(static_cast<std::size_t>(c), static_cast<std::size_t>(b)) +=
                    transition_counts_[transition_offset(c, ctx, b)];
            }
        }
    }
    return out;
}
