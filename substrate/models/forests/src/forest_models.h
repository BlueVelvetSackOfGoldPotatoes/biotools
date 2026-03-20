#pragma once

#include "../../trees/src/tree_models.h"
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

// ---------------------------------------------------------------------------
// RandomForestClassifier -- bagging + feature subsampling (sqrt(n_features))
// ---------------------------------------------------------------------------
class RandomForestClassifier {
public:
    RandomForestClassifier(int n_trees,
                           int max_depth,
                           int min_samples_split,
                           int num_classes,
                           std::mt19937& rng,
                           double sample_ratio = 0.8);

    void fit(const Tensor& X, const std::vector<int>& y);
    std::vector<int> predict(const Tensor& X) const;
    Tensor predict_proba(const Tensor& X) const;
    std::vector<double> feature_importance() const;

    // Permutation importance: measures accuracy drop when feature j is shuffled
    std::vector<double> permutation_importance(const Tensor& X,
                                               const std::vector<int>& y) const;

    // Out-of-bag accuracy estimate
    double oob_score(const Tensor& X, const std::vector<int>& y) const;

    int get_n_trees() const { return n_trees; }
    const std::vector<DecisionTreeClassifier>& get_trees() const { return trees; }

private:
    std::vector<DecisionTreeClassifier> trees;
    // For each tree, store which sample indices were used for training (for OOB)
    std::vector<std::vector<size_t>> bootstrap_indices;

    int n_trees;
    int max_depth;
    int min_samples_split;
    int num_classes;
    double sample_ratio;
    std::mt19937& rng;
};

// ---------------------------------------------------------------------------
// ExtraTreesClassifier -- Extremely Randomized Trees
//   Uses all data (no bootstrap) but picks random split thresholds.
//   Trees use max_features = sqrt(n_features) for feature subsampling,
//   but the split threshold is chosen randomly within the feature range
//   rather than scanning all possible thresholds.
// ---------------------------------------------------------------------------
class ExtraTreesClassifier {
public:
    ExtraTreesClassifier(int n_trees,
                         int max_depth,
                         int min_samples_split,
                         int num_classes,
                         std::mt19937& rng);

    void fit(const Tensor& X, const std::vector<int>& y);
    std::vector<int> predict(const Tensor& X) const;
    Tensor predict_proba(const Tensor& X) const;
    std::vector<double> feature_importance() const;

    int get_n_trees() const { return n_trees; }
    const std::vector<DecisionTreeClassifier>& get_trees() const { return trees; }

private:
    std::vector<DecisionTreeClassifier> trees;
    int n_trees;
    int max_depth;
    int min_samples_split;
    int num_classes;
    std::mt19937& rng;
};
