#include "tree_models.h"
#include <limits>
#include <cstring>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#define TREE_OMP_PAR_FOR _Pragma("omp parallel for schedule(static)")
#else
#define TREE_OMP_PAR_FOR
#endif

// ===================================================================
//  DecisionTreeClassifier
// ===================================================================

DecisionTreeClassifier::DecisionTreeClassifier(int max_depth,
                                               int min_samples_split,
                                               int num_classes,
                                               std::mt19937& rng,
                                               int max_features,
                                               bool extra_random_splits)
    : max_depth(max_depth),
      min_samples_split(min_samples_split),
      num_classes(num_classes),
      num_features(0),
      rng(rng),
      max_features(max_features),
      extra_random_splits(extra_random_splits) {}

// ----- fit -----
void DecisionTreeClassifier::fit(const Tensor& X, const std::vector<int>& y) {
    if (X.rows != y.size()) {
        throw std::runtime_error("DecisionTreeClassifier::fit requires X.rows == y.size()");
    }
    num_features = static_cast<int>(X.cols);
    nodes.clear();
    nodes.reserve(1024);

    std::vector<size_t> indices(X.rows);
    std::iota(indices.begin(), indices.end(), 0);

    build_tree(X, y, indices, 0);
}

// ----- build_tree (recursive, returns node index) -----
int DecisionTreeClassifier::build_tree(const Tensor& X,
                                       const std::vector<int>& y,
                                       const std::vector<size_t>& indices,
                                       int depth) {
    int node_idx = static_cast<int>(nodes.size());
    nodes.emplace_back();
    TreeNode& node = nodes[node_idx];
    node.n_samples = indices.size();

    // Count classes at this node
    std::vector<double> counts(num_classes, 0.0);
    for (size_t i : indices) {
        counts[y[i]] += 1.0;
    }
    node.class_counts = counts;

    // Majority class
    int best_class = 0;
    double best_count = 0;
    for (int c = 0; c < num_classes; ++c) {
        if (counts[c] > best_count) {
            best_count = counts[c];
            best_class = c;
        }
    }
    node.class_label = best_class;

    // Check stopping conditions
    bool pure = (best_count == static_cast<double>(indices.size()));
    if (pure ||
        depth >= max_depth ||
        static_cast<int>(indices.size()) < min_samples_split) {
        node.is_leaf = true;
        return node_idx;
    }

    // Find best split
    SplitResult split = find_best_split(X, y, indices);

    if (split.gain <= 0.0 || split.left_idx.empty() || split.right_idx.empty()) {
        node.is_leaf = true;
        return node_idx;
    }

    node.feature_index = split.feature;
    node.threshold = split.threshold;
    node.impurity_decrease = split.gain * static_cast<double>(indices.size());

    // Build children -- note: build_tree may reallocate `nodes`, so we
    // must NOT hold a reference to nodes[node_idx] across recursive calls.
    int left_idx = build_tree(X, y, split.left_idx, depth + 1);
    int right_idx = build_tree(X, y, split.right_idx, depth + 1);

    nodes[node_idx].left = left_idx;
    nodes[node_idx].right = right_idx;

    return node_idx;
}

// ----- gini impurity -----
double DecisionTreeClassifier::gini(const std::vector<int>& labels,
                                    const std::vector<size_t>& indices) const {
    if (indices.empty()) return 0.0;

    std::vector<int> counts(num_classes, 0);
    for (size_t i : indices) {
        counts[labels[i]]++;
    }
    double n = static_cast<double>(indices.size());
    double impurity = 1.0;
    for (int c = 0; c < num_classes; ++c) {
        double p = static_cast<double>(counts[c]) / n;
        impurity -= p * p;
    }
    return impurity;
}

// ----- find_best_split -----
DecisionTreeClassifier::SplitResult
DecisionTreeClassifier::find_best_split(const Tensor& X,
                                        const std::vector<int>& y,
                                        const std::vector<size_t>& indices) {
    SplitResult best;
    best.gain = -1.0;

    double parent_impurity = gini(y, indices);
    double n_parent = static_cast<double>(indices.size());

    // Determine which features to evaluate
    int n_feats_to_try = num_features;
    std::vector<int> feature_candidates(num_features);
    std::iota(feature_candidates.begin(), feature_candidates.end(), 0);

    if (max_features > 0 && max_features < num_features) {
        n_feats_to_try = max_features;
        // Fisher-Yates partial shuffle to pick max_features random features
        for (int i = 0; i < max_features; ++i) {
            std::uniform_int_distribution<int> dist(i, num_features - 1);
            std::swap(feature_candidates[i], feature_candidates[dist(rng)]);
        }
    }

    // For efficiency: build a sorted index per feature, then scan
    // We use a single working vector for the feature values + sorted indices
    std::vector<std::pair<double, size_t>> feat_vals(indices.size());

    for (int fi = 0; fi < n_feats_to_try; ++fi) {
        int f = feature_candidates[fi];

        // Collect feature values for the current subset
        for (size_t k = 0; k < indices.size(); ++k) {
            feat_vals[k] = {X(indices[k], f), indices[k]};
        }

        if (extra_random_splits) {
            // Extra-Trees: draw a random threshold from [min_val, max_val]
            // for this feature (Geurts et al. 2006)
            double min_val = feat_vals[0].first;
            double max_val = feat_vals[0].first;
            for (size_t k = 1; k < indices.size(); ++k) {
                if (feat_vals[k].first < min_val) min_val = feat_vals[k].first;
                if (feat_vals[k].first > max_val) max_val = feat_vals[k].first;
            }

            // Skip constant features
            if (min_val >= max_val) continue;

            std::uniform_real_distribution<double> threshold_dist(min_val, max_val);
            double rand_threshold = threshold_dist(rng);

            // Compute Gini for this random split
            std::vector<int> left_counts(num_classes, 0);
            std::vector<int> right_counts(num_classes, 0);
            int n_left = 0;
            int n_right = 0;
            for (size_t k = 0; k < indices.size(); ++k) {
                if (feat_vals[k].first <= rand_threshold) {
                    left_counts[y[feat_vals[k].second]]++;
                    n_left++;
                } else {
                    right_counts[y[feat_vals[k].second]]++;
                    n_right++;
                }
            }

            if (n_left == 0 || n_right == 0) continue;

            double gini_left = 1.0;
            for (int c = 0; c < num_classes; ++c) {
                double p = static_cast<double>(left_counts[c]) / n_left;
                gini_left -= p * p;
            }
            double gini_right = 1.0;
            for (int c = 0; c < num_classes; ++c) {
                double p = static_cast<double>(right_counts[c]) / n_right;
                gini_right -= p * p;
            }

            double w_left = static_cast<double>(n_left) / n_parent;
            double w_right = static_cast<double>(n_right) / n_parent;
            double gain = parent_impurity - w_left * gini_left - w_right * gini_right;

            if (gain > best.gain) {
                best.gain = gain;
                best.feature = f;
                best.threshold = rand_threshold;
            }
        } else {
            // Standard CART: sort and scan all midpoints

            // Sort by feature value
            std::sort(feat_vals.begin(), feat_vals.end(),
                      [](const std::pair<double, size_t>& a,
                         const std::pair<double, size_t>& b) {
                          return a.first < b.first;
                      });

            // Incrementally compute Gini for left/right as we sweep
            std::vector<int> left_counts(num_classes, 0);
            std::vector<int> right_counts(num_classes, 0);
            for (size_t k = 0; k < indices.size(); ++k) {
                right_counts[y[feat_vals[k].second]]++;
            }

            int n_left = 0;
            int n_right = static_cast<int>(indices.size());

            for (size_t k = 0; k < indices.size() - 1; ++k) {
                int label = y[feat_vals[k].second];
                left_counts[label]++;
                right_counts[label]--;
                n_left++;
                n_right--;

                // Skip if next sample has same feature value (no valid split here)
                if (feat_vals[k].first == feat_vals[k + 1].first) continue;

                // Gini for left
                double gini_left = 1.0;
                for (int c = 0; c < num_classes; ++c) {
                    double p = static_cast<double>(left_counts[c]) / n_left;
                    gini_left -= p * p;
                }

                // Gini for right
                double gini_right = 1.0;
                for (int c = 0; c < num_classes; ++c) {
                    double p = static_cast<double>(right_counts[c]) / n_right;
                    gini_right -= p * p;
                }

                // Weighted impurity decrease
                double w_left = static_cast<double>(n_left) / n_parent;
                double w_right = static_cast<double>(n_right) / n_parent;
                double gain = parent_impurity - w_left * gini_left - w_right * gini_right;

                if (gain > best.gain) {
                    best.gain = gain;
                    best.feature = f;
                    best.threshold = 0.5 * (feat_vals[k].first + feat_vals[k + 1].first);
                }
            }
        }
    }

    // Build left/right index vectors for the best split
    if (best.gain > 0.0) {
        best.left_idx.reserve(indices.size());
        best.right_idx.reserve(indices.size());
        for (size_t i : indices) {
            if (X(i, best.feature) <= best.threshold) {
                best.left_idx.push_back(i);
            } else {
                best.right_idx.push_back(i);
            }
        }
    }

    return best;
}

// ----- predict_single -----
int DecisionTreeClassifier::predict_single(const Tensor& X, size_t row) const {
    int idx = 0;
    while (!nodes[idx].is_leaf) {
        if (X(row, nodes[idx].feature_index) <= nodes[idx].threshold) {
            idx = nodes[idx].left;
        } else {
            idx = nodes[idx].right;
        }
    }
    return nodes[idx].class_label;
}

// ----- predict -----
std::vector<int> DecisionTreeClassifier::predict(const Tensor& X) const {
    std::vector<int> preds(X.rows);
    TREE_OMP_PAR_FOR
    for (size_t i = 0; i < X.rows; ++i) {
        preds[i] = predict_single(X, i);
    }
    return preds;
}

// ----- predict_proba_single -----
std::vector<double> DecisionTreeClassifier::predict_proba_single(const Tensor& X,
                                                                  size_t row) const {
    int idx = 0;
    while (!nodes[idx].is_leaf) {
        if (X(row, nodes[idx].feature_index) <= nodes[idx].threshold) {
            idx = nodes[idx].left;
        } else {
            idx = nodes[idx].right;
        }
    }

    const auto& counts = nodes[idx].class_counts;
    double total = 0.0;
    for (double c : counts) total += c;

    std::vector<double> proba(num_classes, 0.0);
    if (total > 0.0) {
        for (int c = 0; c < num_classes; ++c) {
            proba[c] = counts[c] / total;
        }
    }
    return proba;
}

// ----- predict_proba -----
Tensor DecisionTreeClassifier::predict_proba(const Tensor& X) const {
    Tensor proba(X.rows, num_classes, 0.0);
    TREE_OMP_PAR_FOR
    for (size_t i = 0; i < X.rows; ++i) {
        std::vector<double> p = predict_proba_single(X, i);
        for (int c = 0; c < num_classes; ++c) {
            proba(i, c) = p[c];
        }
    }
    return proba;
}

// ----- feature_importance -----
std::vector<double> DecisionTreeClassifier::feature_importance() const {
    std::vector<double> importance(num_features, 0.0);
    double total = 0.0;

    for (const auto& node : nodes) {
        if (!node.is_leaf && node.feature_index >= 0) {
            importance[node.feature_index] += node.impurity_decrease;
            total += node.impurity_decrease;
        }
    }

    if (total > 0.0) {
        for (auto& v : importance) v /= total;
    }
    return importance;
}

// ----- decision_path -----
std::vector<int> DecisionTreeClassifier::decision_path(size_t sample_idx,
                                                        const Tensor& X) const {
    std::vector<int> path;
    int idx = 0;
    while (true) {
        path.push_back(idx);
        if (nodes[idx].is_leaf) break;
        if (X(sample_idx, nodes[idx].feature_index) <= nodes[idx].threshold) {
            idx = nodes[idx].left;
        } else {
            idx = nodes[idx].right;
        }
    }
    return path;
}


// ===================================================================
//  DecisionTreeRegressor
// ===================================================================

DecisionTreeRegressor::DecisionTreeRegressor(int max_depth,
                                             int min_samples_split,
                                             std::mt19937& rng)
    : max_depth(max_depth),
      min_samples_split(min_samples_split),
      num_features(0),
      rng(rng) {}

// ----- fit -----
void DecisionTreeRegressor::fit(const Tensor& X, const std::vector<double>& y) {
    if (X.rows != y.size()) {
        throw std::runtime_error("DecisionTreeRegressor::fit requires X.rows == y.size()");
    }
    num_features = static_cast<int>(X.cols);
    nodes.clear();
    nodes.reserve(1024);

    std::vector<size_t> indices(X.rows);
    std::iota(indices.begin(), indices.end(), 0);

    build_tree(X, y, indices, 0);
}

// ----- mse -----
double DecisionTreeRegressor::mse(const std::vector<double>& y,
                                  const std::vector<size_t>& indices) const {
    if (indices.empty()) return 0.0;
    double mean_val = 0.0;
    for (size_t i : indices) mean_val += y[i];
    mean_val /= static_cast<double>(indices.size());

    double s = 0.0;
    for (size_t i : indices) {
        double d = y[i] - mean_val;
        s += d * d;
    }
    return s / static_cast<double>(indices.size());
}

// ----- variance_reduction -----
double DecisionTreeRegressor::variance_reduction(const std::vector<double>& y,
                                                 const std::vector<size_t>& left,
                                                 const std::vector<size_t>& right,
                                                 const std::vector<size_t>& parent) const {
    double n = static_cast<double>(parent.size());
    double mse_parent = mse(y, parent);
    double mse_left = mse(y, left);
    double mse_right = mse(y, right);

    double w_left = static_cast<double>(left.size()) / n;
    double w_right = static_cast<double>(right.size()) / n;

    return mse_parent - w_left * mse_left - w_right * mse_right;
}

// ----- build_tree (recursive) -----
int DecisionTreeRegressor::build_tree(const Tensor& X,
                                      const std::vector<double>& y,
                                      const std::vector<size_t>& indices,
                                      int depth) {
    int node_idx = static_cast<int>(nodes.size());
    nodes.emplace_back();
    TreeNode& node = nodes[node_idx];
    node.n_samples = indices.size();

    // Compute mean value at this node
    double mean_val = 0.0;
    for (size_t i : indices) mean_val += y[i];
    mean_val /= static_cast<double>(indices.size());
    node.value = mean_val;

    // Check stopping conditions
    if (depth >= max_depth ||
        static_cast<int>(indices.size()) < min_samples_split ||
        indices.size() <= 1) {
        node.is_leaf = true;
        return node_idx;
    }

    // Check if all targets are identical
    bool all_same = true;
    double first_val = y[indices[0]];
    for (size_t k = 1; k < indices.size(); ++k) {
        if (y[indices[k]] != first_val) {
            all_same = false;
            break;
        }
    }
    if (all_same) {
        node.is_leaf = true;
        return node_idx;
    }

    SplitResult split = find_best_split(X, y, indices);

    if (split.gain <= 0.0 || split.left_idx.empty() || split.right_idx.empty()) {
        node.is_leaf = true;
        return node_idx;
    }

    node.feature_index = split.feature;
    node.threshold = split.threshold;
    node.impurity_decrease = split.gain * static_cast<double>(indices.size());

    int left_idx = build_tree(X, y, split.left_idx, depth + 1);
    int right_idx = build_tree(X, y, split.right_idx, depth + 1);

    nodes[node_idx].left = left_idx;
    nodes[node_idx].right = right_idx;

    return node_idx;
}

// ----- find_best_split (regressor) -----
DecisionTreeRegressor::SplitResult
DecisionTreeRegressor::find_best_split(const Tensor& X,
                                       const std::vector<double>& y,
                                       const std::vector<size_t>& indices) {
    SplitResult best;
    best.gain = -1.0;

    double parent_mse = mse(y, indices);
    double n_parent = static_cast<double>(indices.size());

    // Use all features for the regressor (no max_features param)
    std::vector<std::pair<double, size_t>> feat_vals(indices.size());

    for (int f = 0; f < num_features; ++f) {
        for (size_t k = 0; k < indices.size(); ++k) {
            feat_vals[k] = {X(indices[k], f), indices[k]};
        }

        std::sort(feat_vals.begin(), feat_vals.end(),
                  [](const std::pair<double, size_t>& a,
                     const std::pair<double, size_t>& b) {
                      return a.first < b.first;
                  });

        // Incremental MSE computation
        // left: sum, sum_sq, count   right: same
        double left_sum = 0.0, left_sum_sq = 0.0;
        double right_sum = 0.0, right_sum_sq = 0.0;
        int n_left = 0, n_right = static_cast<int>(indices.size());

        for (size_t k = 0; k < indices.size(); ++k) {
            double val = y[feat_vals[k].second];
            right_sum += val;
            right_sum_sq += val * val;
        }

        for (size_t k = 0; k < indices.size() - 1; ++k) {
            double val = y[feat_vals[k].second];
            left_sum += val;
            left_sum_sq += val * val;
            right_sum -= val;
            right_sum_sq -= val * val;
            n_left++;
            n_right--;

            if (feat_vals[k].first == feat_vals[k + 1].first) continue;

            // MSE left = sum_sq/n - (sum/n)^2
            double mse_left = left_sum_sq / n_left - (left_sum / n_left) * (left_sum / n_left);
            double mse_right = right_sum_sq / n_right - (right_sum / n_right) * (right_sum / n_right);

            double w_left = static_cast<double>(n_left) / n_parent;
            double w_right = static_cast<double>(n_right) / n_parent;
            double gain = parent_mse - w_left * mse_left - w_right * mse_right;

            if (gain > best.gain) {
                best.gain = gain;
                best.feature = f;
                best.threshold = 0.5 * (feat_vals[k].first + feat_vals[k + 1].first);
            }
        }
    }

    if (best.gain > 0.0) {
        best.left_idx.reserve(indices.size());
        best.right_idx.reserve(indices.size());
        for (size_t i : indices) {
            if (X(i, best.feature) <= best.threshold) {
                best.left_idx.push_back(i);
            } else {
                best.right_idx.push_back(i);
            }
        }
    }

    return best;
}

// ----- predict_single -----
double DecisionTreeRegressor::predict_single(const Tensor& X, size_t row) const {
    int idx = 0;
    while (!nodes[idx].is_leaf) {
        if (X(row, nodes[idx].feature_index) <= nodes[idx].threshold) {
            idx = nodes[idx].left;
        } else {
            idx = nodes[idx].right;
        }
    }
    return nodes[idx].value;
}

// ----- predict -----
std::vector<double> DecisionTreeRegressor::predict(const Tensor& X) const {
    std::vector<double> preds(X.rows);
    TREE_OMP_PAR_FOR
    for (size_t i = 0; i < X.rows; ++i) {
        preds[i] = predict_single(X, i);
    }
    return preds;
}

// ----- feature_importance -----
std::vector<double> DecisionTreeRegressor::feature_importance() const {
    std::vector<double> importance(num_features, 0.0);
    double total = 0.0;

    for (const auto& node : nodes) {
        if (!node.is_leaf && node.feature_index >= 0) {
            importance[node.feature_index] += node.impurity_decrease;
            total += node.impurity_decrease;
        }
    }

    if (total > 0.0) {
        for (auto& v : importance) v /= total;
    }
    return importance;
}
