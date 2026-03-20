#pragma once

#include "../../../core/tensor/tensor.h"

#include <cstddef>
#include <vector>

class MarkovChainClassifier {
public:
    struct ClassDiagnostics {
        double sample_count = 0.0;
        double prior = 0.0;
        double transition_entropy = 0.0;
        double context_coverage = 0.0;
    };

    MarkovChainClassifier(int num_classes, int num_bins, int order, double alpha);

    void reset();
    void partial_fit(const Tensor& X, const std::vector<int>& y, const std::vector<std::size_t>& indices);
    void partial_fit(const Tensor& X, const std::vector<int>& y, std::size_t start, std::size_t count);

    Tensor predict_log_scores(const Tensor& X) const;
    Tensor predict_proba(const Tensor& X) const;
    std::vector<int> predict(const Tensor& X) const;

    std::vector<ClassDiagnostics> diagnostics() const;
    std::vector<double> position_nll(const Tensor& X, const std::vector<int>& y) const;
    Tensor class_bin_counts() const;

    int num_classes() const { return num_classes_; }
    int num_bins() const { return num_bins_; }
    int order() const { return order_; }
    std::size_t num_contexts() const { return num_contexts_; }
    std::size_t samples_seen() const { return samples_seen_; }

private:
    int num_classes_ = 10;
    int num_bins_ = 8;
    int order_ = 1;
    double alpha_ = 0.25;

    std::size_t num_contexts_ = 0;
    std::size_t context_shift_mod_ = 1;
    std::size_t samples_seen_ = 0;

    std::vector<double> transition_counts_;
    std::vector<double> context_totals_;
    std::vector<double> class_counts_;

    int quantize(double value) const;
    std::size_t shift_context(std::size_t context, int symbol) const;
    std::size_t context_offset(int class_id, std::size_t context) const;
    std::size_t transition_offset(int class_id, std::size_t context, int symbol) const;

    void update_row(const Tensor& X, std::size_t row, int class_id);
    void log_scores_for_row(const Tensor& X, std::size_t row, std::vector<double>& out_scores) const;
};
