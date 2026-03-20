#include "forest_models.h"
#include <set>
#include <iostream>

#ifdef _OPENMP
#define FOREST_OMP_PAR_FOR _Pragma("omp parallel for schedule(static)")
#define FOREST_OMP_PAR_FOR_COLLAPSE2 _Pragma("omp parallel for collapse(2) schedule(static)")
#define FOREST_OMP_PAR_FOR_REDUCE_CORRECT _Pragma("omp parallel for reduction(+ : correct) schedule(static)")
#define FOREST_OMP_PAR_FOR_REDUCE_PERM _Pragma("omp parallel for reduction(+ : perm_correct) schedule(static)")
#else
#define FOREST_OMP_PAR_FOR
#define FOREST_OMP_PAR_FOR_COLLAPSE2
#define FOREST_OMP_PAR_FOR_REDUCE_CORRECT
#define FOREST_OMP_PAR_FOR_REDUCE_PERM
#endif

// ===================================================================
//  RandomForestClassifier
// ===================================================================

RandomForestClassifier::RandomForestClassifier(int n_trees,
                                               int max_depth,
                                               int min_samples_split,
                                               int num_classes,
                                               std::mt19937& rng,
                                               double sample_ratio)
    : n_trees(n_trees),
      max_depth(max_depth),
      min_samples_split(min_samples_split),
      num_classes(num_classes),
      sample_ratio(sample_ratio),
      rng(rng) {}

// ----- fit -----
void RandomForestClassifier::fit(const Tensor& X, const std::vector<int>& y) {
    int num_features = static_cast<int>(X.cols);
    int max_feat = static_cast<int>(std::round(std::sqrt(static_cast<double>(num_features))));
    if (max_feat < 1) max_feat = 1;

    size_t n_samples = X.rows;
    size_t boot_size = static_cast<size_t>(std::round(sample_ratio * static_cast<double>(n_samples)));
    if (boot_size < 1) boot_size = 1;

    trees.clear();
    trees.reserve(n_trees);
    bootstrap_indices.clear();
    bootstrap_indices.resize(n_trees);

    std::uniform_int_distribution<size_t> dist(0, n_samples - 1);

    for (int t = 0; t < n_trees; ++t) {
        std::cout << "  Training tree " << (t + 1) << "/" << n_trees << "...\r" << std::flush;

        // Bootstrap sample (with replacement)
        std::vector<size_t> boot_idx(boot_size);
        for (size_t i = 0; i < boot_size; ++i) {
            boot_idx[i] = dist(rng);
        }
        bootstrap_indices[t] = boot_idx;

        // Build bootstrap dataset
        Tensor X_boot(boot_size, X.cols);
        std::vector<int> y_boot(boot_size);
        for (size_t i = 0; i < boot_size; ++i) {
            for (size_t j = 0; j < X.cols; ++j) {
                X_boot(i, j) = X(boot_idx[i], j);
            }
            y_boot[i] = y[boot_idx[i]];
        }

        // Create and train tree with feature subsampling
        trees.emplace_back(max_depth, min_samples_split, num_classes, rng, max_feat);
        trees.back().fit(X_boot, y_boot);
    }
    std::cout << std::endl;
}

// ----- predict -----
std::vector<int> RandomForestClassifier::predict(const Tensor& X) const {
    Tensor proba = predict_proba(X);
    std::vector<int> preds(X.rows);
    FOREST_OMP_PAR_FOR
    for (size_t i = 0; i < X.rows; ++i) {
        int best_class = 0;
        double best_prob = proba(i, 0);
        for (int c = 1; c < num_classes; ++c) {
            if (proba(i, c) > best_prob) {
                best_prob = proba(i, c);
                best_class = c;
            }
        }
        preds[i] = best_class;
    }
    return preds;
}

// ----- predict_proba -----
Tensor RandomForestClassifier::predict_proba(const Tensor& X) const {
    Tensor avg_proba(X.rows, num_classes, 0.0);

    for (const auto& tree : trees) {
        Tensor tree_proba = tree.predict_proba(X);
        FOREST_OMP_PAR_FOR_COLLAPSE2
        for (size_t i = 0; i < X.rows; ++i) {
            for (int c = 0; c < num_classes; ++c) {
                avg_proba(i, c) += tree_proba(i, c);
            }
        }
    }

    // Average
    double inv_n = 1.0 / static_cast<double>(trees.size());
    FOREST_OMP_PAR_FOR_COLLAPSE2
    for (size_t i = 0; i < X.rows; ++i) {
        for (int c = 0; c < num_classes; ++c) {
            avg_proba(i, c) *= inv_n;
        }
    }

    return avg_proba;
}

// ----- feature_importance -----
std::vector<double> RandomForestClassifier::feature_importance() const {
    if (trees.empty()) return {};

    int num_features = trees[0].get_num_features();
    std::vector<double> importance(num_features, 0.0);

    for (const auto& tree : trees) {
        std::vector<double> tree_imp = tree.feature_importance();
        for (int f = 0; f < num_features; ++f) {
            importance[f] += tree_imp[f];
        }
    }

    double inv_n = 1.0 / static_cast<double>(trees.size());
    for (auto& v : importance) v *= inv_n;

    return importance;
}

// ----- permutation_importance -----
std::vector<double> RandomForestClassifier::permutation_importance(
        const Tensor& X, const std::vector<int>& y) const {

    int num_features = static_cast<int>(X.cols);
    size_t n = X.rows;

    // Baseline accuracy
    std::vector<int> base_preds = predict(X);
    int correct = 0;
    FOREST_OMP_PAR_FOR_REDUCE_CORRECT
    for (size_t i = 0; i < n; ++i) {
        if (base_preds[i] == y[i]) correct++;
    }
    double base_acc = static_cast<double>(correct) / static_cast<double>(n);

    std::vector<double> importance(num_features, 0.0);

    // For each feature, shuffle it and measure accuracy drop
    // We make a mutable copy of X for permutation
    Tensor X_perm(X.rows, X.cols, X.data);

    // We need a mutable rng for shuffling -- cast away const since
    // permutation_importance is conceptually const but needs random state.
    std::mt19937 local_rng(rng);  // copy the rng state

    for (int f = 0; f < num_features; ++f) {
        // Save original column values
        std::vector<double> original_col(n);
        for (size_t i = 0; i < n; ++i) {
            original_col[i] = X_perm(i, f);
        }

        // Create a shuffled permutation of the column
        std::vector<double> shuffled_col = original_col;
        std::shuffle(shuffled_col.begin(), shuffled_col.end(), local_rng);

        // Set the shuffled values
        for (size_t i = 0; i < n; ++i) {
            X_perm(i, f) = shuffled_col[i];
        }

        // Measure accuracy with permuted feature
        std::vector<int> perm_preds = predict(X_perm);
        int perm_correct = 0;
        FOREST_OMP_PAR_FOR_REDUCE_PERM
        for (size_t i = 0; i < n; ++i) {
            if (perm_preds[i] == y[i]) perm_correct++;
        }
        double perm_acc = static_cast<double>(perm_correct) / static_cast<double>(n);

        importance[f] = base_acc - perm_acc;

        // Restore original column
        for (size_t i = 0; i < n; ++i) {
            X_perm(i, f) = original_col[i];
        }
    }

    return importance;
}

// ----- oob_score -----
double RandomForestClassifier::oob_score(const Tensor& X,
                                         const std::vector<int>& y) const {
    size_t n = X.rows;

    // For each sample, accumulate votes from trees that did NOT use it
    // in their bootstrap sample.
    std::vector<std::vector<double>> oob_votes(n, std::vector<double>(num_classes, 0.0));
    std::vector<int> oob_count(n, 0);

    for (int t = 0; t < static_cast<int>(trees.size()); ++t) {
        // Determine which samples are OOB for this tree
        std::set<size_t> in_bag(bootstrap_indices[t].begin(),
                                bootstrap_indices[t].end());

        // For OOB samples, get predictions
        for (size_t i = 0; i < n; ++i) {
            if (in_bag.find(i) == in_bag.end()) {
                // This sample is OOB for tree t
                // Get probability prediction for this single sample
                // We pass a 1-row tensor (the i-th row of X)
                Tensor single_row = X.row(i);
                Tensor proba = trees[t].predict_proba(single_row);
                for (int c = 0; c < num_classes; ++c) {
                    oob_votes[i][c] += proba(0, c);
                }
                oob_count[i]++;
            }
        }
    }

    // Compute accuracy over samples that have at least one OOB tree
    int correct = 0;
    int total = 0;
    for (size_t i = 0; i < n; ++i) {
        if (oob_count[i] == 0) continue;
        total++;
        int pred = 0;
        double best = oob_votes[i][0];
        for (int c = 1; c < num_classes; ++c) {
            if (oob_votes[i][c] > best) {
                best = oob_votes[i][c];
                pred = c;
            }
        }
        if (pred == y[i]) correct++;
    }

    if (total == 0) return 0.0;
    return static_cast<double>(correct) / static_cast<double>(total);
}


// ===================================================================
//  ExtraTreesClassifier
// ===================================================================

ExtraTreesClassifier::ExtraTreesClassifier(int n_trees,
                                           int max_depth,
                                           int min_samples_split,
                                           int num_classes,
                                           std::mt19937& rng)
    : n_trees(n_trees),
      max_depth(max_depth),
      min_samples_split(min_samples_split),
      num_classes(num_classes),
      rng(rng) {}

// ----- fit -----
// ExtraTrees uses all training data (no bootstrap) but with truly random
// split thresholds (Geurts et al. 2006). For each candidate feature at each
// node, a random threshold is drawn uniformly from [min, max] of that
// feature's values in the current node, rather than scanning all midpoints.
// Combined with max_features = sqrt(n_features) for feature subsampling,
// this is the defining characteristic of Extremely Randomized Trees.
void ExtraTreesClassifier::fit(const Tensor& X, const std::vector<int>& y) {
    int num_features = static_cast<int>(X.cols);
    int max_feat = static_cast<int>(std::round(std::sqrt(static_cast<double>(num_features))));
    if (max_feat < 1) max_feat = 1;

    trees.clear();
    trees.reserve(n_trees);

    for (int t = 0; t < n_trees; ++t) {
        std::cout << "  Training ExtraTree " << (t + 1) << "/" << n_trees << "...\r" << std::flush;

        // No bootstrap -- use all data, with random feature subsampling
        // and extra-random threshold selection (true Extra-Trees)
        trees.emplace_back(max_depth, min_samples_split, num_classes, rng, max_feat,
                           /*extra_random_splits=*/true);
        trees.back().fit(X, y);
    }
    std::cout << std::endl;
}

// ----- predict -----
std::vector<int> ExtraTreesClassifier::predict(const Tensor& X) const {
    Tensor proba = predict_proba(X);
    std::vector<int> preds(X.rows);
    FOREST_OMP_PAR_FOR
    for (size_t i = 0; i < X.rows; ++i) {
        int best_class = 0;
        double best_prob = proba(i, 0);
        for (int c = 1; c < num_classes; ++c) {
            if (proba(i, c) > best_prob) {
                best_prob = proba(i, c);
                best_class = c;
            }
        }
        preds[i] = best_class;
    }
    return preds;
}

// ----- predict_proba -----
Tensor ExtraTreesClassifier::predict_proba(const Tensor& X) const {
    Tensor avg_proba(X.rows, num_classes, 0.0);

    for (const auto& tree : trees) {
        Tensor tree_proba = tree.predict_proba(X);
        FOREST_OMP_PAR_FOR_COLLAPSE2
        for (size_t i = 0; i < X.rows; ++i) {
            for (int c = 0; c < num_classes; ++c) {
                avg_proba(i, c) += tree_proba(i, c);
            }
        }
    }

    double inv_n = 1.0 / static_cast<double>(trees.size());
    FOREST_OMP_PAR_FOR_COLLAPSE2
    for (size_t i = 0; i < X.rows; ++i) {
        for (int c = 0; c < num_classes; ++c) {
            avg_proba(i, c) *= inv_n;
        }
    }

    return avg_proba;
}

// ----- feature_importance -----
std::vector<double> ExtraTreesClassifier::feature_importance() const {
    if (trees.empty()) return {};

    int num_features = trees[0].get_num_features();
    std::vector<double> importance(num_features, 0.0);

    for (const auto& tree : trees) {
        std::vector<double> tree_imp = tree.feature_importance();
        for (int f = 0; f < num_features; ++f) {
            importance[f] += tree_imp[f];
        }
    }

    double inv_n = 1.0 / static_cast<double>(trees.size());
    for (auto& v : importance) v *= inv_n;

    return importance;
}
