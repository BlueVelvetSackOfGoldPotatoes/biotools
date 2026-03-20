#pragma once

#include "../../../core/tensor/tensor.h"
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <map>

// ---------------------------------------------------------------------------
// TreeNode -- shared node representation for classification and regression
// ---------------------------------------------------------------------------
struct TreeNode {
    int feature_index = -1;     // split feature
    double threshold = 0.0;     // split threshold
    int class_label = -1;       // leaf prediction (classification)
    double value = 0.0;         // leaf value (regression)
    int left = -1, right = -1;  // child indices in nodes vector
    bool is_leaf = false;
    double impurity_decrease = 0.0;
    size_t n_samples = 0;       // how many training samples reached this node
    std::vector<double> class_counts; // per-class counts at this leaf (classification)
};

// ---------------------------------------------------------------------------
// DecisionTreeClassifier -- CART-style, Gini impurity
// ---------------------------------------------------------------------------
class DecisionTreeClassifier {
public:
    std::vector<TreeNode> nodes;

    DecisionTreeClassifier(int max_depth,
                           int min_samples_split,
                           int num_classes,
                           std::mt19937& rng,
                           int max_features = -1,
                           bool extra_random_splits = false);

    void fit(const Tensor& X, const std::vector<int>& y);
    std::vector<int> predict(const Tensor& X) const;
    Tensor predict_proba(const Tensor& X) const;              // N x num_classes
    std::vector<double> feature_importance() const;
    std::vector<int> decision_path(size_t sample_idx, const Tensor& X) const;

    int get_num_classes() const { return num_classes; }
    int get_num_features() const { return num_features; }

private:
    int max_depth;
    int min_samples_split;
    int num_classes;
    int num_features;
    std::mt19937& rng;
    int max_features;  // -1 = all features, else random subset size
    bool extra_random_splits;  // if true, pick random threshold per feature (Extra-Trees)

    // Internal split description
    struct SplitResult {
        int feature = -1;
        double threshold = 0.0;
        double gain = -1.0;
        std::vector<size_t> left_idx;
        std::vector<size_t> right_idx;
    };

    int build_tree(const Tensor& X,
                   const std::vector<int>& y,
                   const std::vector<size_t>& indices,
                   int depth);

    double gini(const std::vector<int>& labels,
                const std::vector<size_t>& indices) const;

    SplitResult find_best_split(const Tensor& X,
                                const std::vector<int>& y,
                                const std::vector<size_t>& indices);

    int predict_single(const Tensor& X, size_t row) const;
    std::vector<double> predict_proba_single(const Tensor& X, size_t row) const;
};

// ---------------------------------------------------------------------------
// DecisionTreeRegressor -- CART-style, variance reduction (MSE)
// ---------------------------------------------------------------------------
class DecisionTreeRegressor {
public:
    std::vector<TreeNode> nodes;

    DecisionTreeRegressor(int max_depth,
                          int min_samples_split,
                          std::mt19937& rng);

    void fit(const Tensor& X, const std::vector<double>& y);
    std::vector<double> predict(const Tensor& X) const;
    std::vector<double> feature_importance() const;

private:
    int max_depth;
    int min_samples_split;
    int num_features;
    std::mt19937& rng;

    struct SplitResult {
        int feature = -1;
        double threshold = 0.0;
        double gain = -1.0;
        std::vector<size_t> left_idx;
        std::vector<size_t> right_idx;
    };

    int build_tree(const Tensor& X,
                   const std::vector<double>& y,
                   const std::vector<size_t>& indices,
                   int depth);

    double mse(const std::vector<double>& y,
               const std::vector<size_t>& indices) const;

    double variance_reduction(const std::vector<double>& y,
                              const std::vector<size_t>& left,
                              const std::vector<size_t>& right,
                              const std::vector<size_t>& parent) const;

    SplitResult find_best_split(const Tensor& X,
                                const std::vector<double>& y,
                                const std::vector<size_t>& indices);

    double predict_single(const Tensor& X, size_t row) const;
};
