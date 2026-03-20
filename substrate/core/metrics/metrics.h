#pragma once
#include "../tensor/tensor.h"
#include <vector>
#include <cstdint>
#include <cmath>
#include <map>

namespace Metrics {

// Classification accuracy
double accuracy(const std::vector<int>& predictions, const std::vector<uint8_t>& labels);
double accuracy(const std::vector<int>& predictions, const std::vector<int>& labels);

// Accuracy from Tensor: logits/probs (batch x classes) vs one-hot targets
double accuracy_from_tensor(const Tensor& output, const Tensor& targets);

// Confusion matrix: rows=true, cols=predicted
Tensor confusion_matrix(const std::vector<int>& predictions,
                       const std::vector<int>& labels, int num_classes);

// Per-class precision, recall, F1
struct ClassMetrics {
    double precision, recall, f1;
};
std::vector<ClassMetrics> per_class_metrics(const Tensor& cm);
double macro_f1(const Tensor& cm);

// Perplexity from average loss
double perplexity(double avg_loss);

// Regression metrics
double mse(const Tensor& pred, const Tensor& target);
double mae(const Tensor& pred, const Tensor& target);
double rmse(const Tensor& pred, const Tensor& target);
double r_squared(const Tensor& pred, const Tensor& target);

// Clustering metrics
double silhouette_score(const Tensor& data, const std::vector<int>& labels, int k);
double davies_bouldin_index(const Tensor& data, const std::vector<int>& labels, int k);

// Adjusted Rand Index
double adjusted_rand_index(const std::vector<int>& labels_true, const std::vector<int>& labels_pred);

// Normalized Mutual Information
double normalized_mutual_info(const std::vector<int>& labels_true, const std::vector<int>& labels_pred);

} // namespace Metrics
