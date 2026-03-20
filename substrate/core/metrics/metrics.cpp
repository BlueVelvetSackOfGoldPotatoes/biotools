#include "metrics.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <set>
#include <map>

#ifdef _OPENMP
#define METRICS_OMP_PAR_FOR _Pragma("omp parallel for schedule(static)")
#define METRICS_OMP_PAR_FOR_REDUCE_SUM _Pragma("omp parallel for reduction(+ : s) schedule(static)")
#define METRICS_OMP_PAR_FOR_REDUCE_CORRECT _Pragma("omp parallel for reduction(+ : correct) schedule(static)")
#define METRICS_OMP_PAR_FOR_REDUCE_TOTAL _Pragma("omp parallel for reduction(+ : total) schedule(static)")
#define METRICS_OMP_PAR_FOR_REDUCE_SS _Pragma("omp parallel for reduction(+ : ss_res, ss_tot) schedule(static)")
#else
#define METRICS_OMP_PAR_FOR
#define METRICS_OMP_PAR_FOR_REDUCE_SUM
#define METRICS_OMP_PAR_FOR_REDUCE_CORRECT
#define METRICS_OMP_PAR_FOR_REDUCE_TOTAL
#define METRICS_OMP_PAR_FOR_REDUCE_SS
#endif

namespace Metrics {

double accuracy(const std::vector<int>& preds, const std::vector<uint8_t>& labels) {
    if (preds.empty()) return 0.0;
    TENSOR_CHECK(preds.size() == labels.size(), "accuracy: predictions/labels size mismatch");
    size_t correct = 0;
    METRICS_OMP_PAR_FOR_REDUCE_CORRECT
    for (size_t i = 0; i < preds.size(); ++i)
        if (preds[i] == (int)labels[i]) ++correct;
    return (double)correct / preds.size();
}

double accuracy(const std::vector<int>& preds, const std::vector<int>& labels) {
    if (preds.empty()) return 0.0;
    TENSOR_CHECK(preds.size() == labels.size(), "accuracy: predictions/labels size mismatch");
    size_t correct = 0;
    METRICS_OMP_PAR_FOR_REDUCE_CORRECT
    for (size_t i = 0; i < preds.size(); ++i)
        if (preds[i] == labels[i]) ++correct;
    return (double)correct / preds.size();
}

double accuracy_from_tensor(const Tensor& output, const Tensor& targets) {
    if (output.rows == 0) return 0.0;
    size_t correct = 0;
    METRICS_OMP_PAR_FOR_REDUCE_CORRECT
    for (size_t i = 0; i < output.rows; ++i) {
        size_t pred = 0, truth = 0;
        double pmax = output(i, 0), tmax = targets(i, 0);
        for (size_t j = 1; j < output.cols; ++j) {
            if (output(i, j) > pmax) { pmax = output(i, j); pred = j; }
            if (targets(i, j) > tmax) { tmax = targets(i, j); truth = j; }
        }
        if (pred == truth) ++correct;
    }
    return (double)correct / output.rows;
}

Tensor confusion_matrix(const std::vector<int>& preds, const std::vector<int>& labels, int nc) {
    TENSOR_CHECK(preds.size() == labels.size(), "confusion_matrix: predictions/labels size mismatch");
    TENSOR_CHECK(nc > 0, "confusion_matrix: number of classes must be positive");
    Tensor cm(nc, nc);
    for (size_t i = 0; i < preds.size(); ++i) {
        TENSOR_CHECK(labels[i] >= 0 && labels[i] < nc && preds[i] >= 0 && preds[i] < nc,
                     "confusion_matrix: index out of range");
        cm(labels[i], preds[i]) += 1.0;
    }
    return cm;
}

std::vector<ClassMetrics> per_class_metrics(const Tensor& cm) {
    int nc = cm.rows;
    std::vector<ClassMetrics> result(nc);
    for (int c = 0; c < nc; ++c) {
        double tp = cm(c, c);
        double fp = 0, fn = 0;
        for (int i = 0; i < nc; ++i) {
            if (i != c) { fp += cm(i, c); fn += cm(c, i); }
        }
        result[c].precision = (tp + fp > 0) ? tp / (tp + fp) : 0;
        result[c].recall = (tp + fn > 0) ? tp / (tp + fn) : 0;
        result[c].f1 = (result[c].precision + result[c].recall > 0)
            ? 2 * result[c].precision * result[c].recall / (result[c].precision + result[c].recall)
            : 0;
    }
    return result;
}

double macro_f1(const Tensor& cm) {
    auto metrics = per_class_metrics(cm);
    double sum = 0;
    for (auto& m : metrics) sum += m.f1;
    return sum / metrics.size();
}

double perplexity(double avg_loss) { return std::exp(avg_loss); }

double mse(const Tensor& pred, const Tensor& target) {
    double s = 0;
    METRICS_OMP_PAR_FOR_REDUCE_SUM
    for (size_t i = 0; i < pred.data.size(); ++i) {
        double d = pred.data[i] - target.data[i];
        s += d * d;
    }
    return s / pred.data.size();
}

double mae(const Tensor& pred, const Tensor& target) {
    double s = 0;
    METRICS_OMP_PAR_FOR_REDUCE_SUM
    for (size_t i = 0; i < pred.data.size(); ++i)
        s += std::abs(pred.data[i] - target.data[i]);
    return s / pred.data.size();
}

double rmse(const Tensor& pred, const Tensor& target) { return std::sqrt(mse(pred, target)); }

double r_squared(const Tensor& pred, const Tensor& target) {
    double mean_t = target.mean();
    double ss_res = 0, ss_tot = 0;
    METRICS_OMP_PAR_FOR_REDUCE_SS
    for (size_t i = 0; i < pred.data.size(); ++i) {
        double d = pred.data[i] - target.data[i];
        ss_res += d * d;
        double dt = target.data[i] - mean_t;
        ss_tot += dt * dt;
    }
    return 1.0 - ss_res / (ss_tot + 1e-12);
}

double silhouette_score(const Tensor& data, const std::vector<int>& labels, int k) {
    size_t n = data.rows;
    double total = 0;
    METRICS_OMP_PAR_FOR_REDUCE_TOTAL
    for (size_t i = 0; i < n; ++i) {
        // a(i) = mean distance to same cluster
        // b(i) = min mean distance to other clusters
        std::vector<double> cluster_dist(k, 0);
        std::vector<int> cluster_count(k, 0);
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            double dist = 0;
            for (size_t f = 0; f < data.cols; ++f) {
                double d = data(i, f) - data(j, f);
                dist += d * d;
            }
            dist = std::sqrt(dist);
            cluster_dist[labels[j]] += dist;
            cluster_count[labels[j]]++;
        }
        double a = (cluster_count[labels[i]] > 0) ? cluster_dist[labels[i]] / cluster_count[labels[i]] : 0;
        double b = std::numeric_limits<double>::max();
        for (int c = 0; c < k; ++c) {
            if (c == labels[i] || cluster_count[c] == 0) continue;
            double avg = cluster_dist[c] / cluster_count[c];
            b = std::min(b, avg);
        }
        if (b == std::numeric_limits<double>::max()) b = 0;
        double s = (std::max(a, b) > 0) ? (b - a) / std::max(a, b) : 0;
        total += s;
    }
    return total / n;
}

double davies_bouldin_index(const Tensor& data, const std::vector<int>& labels, int k) {
    // Compute cluster centers and avg distances
    std::vector<Tensor> centers(k, Tensor(1, data.cols));
    std::vector<int> counts(k, 0);
    for (size_t i = 0; i < data.rows; ++i) {
        for (size_t j = 0; j < data.cols; ++j)
            centers[labels[i]](0, j) += data(i, j);
        counts[labels[i]]++;
    }
    for (int c = 0; c < k; ++c)
        if (counts[c] > 0) centers[c] *= (1.0 / counts[c]);

    std::vector<double> scatter(k, 0);
    for (size_t i = 0; i < data.rows; ++i) {
        double dist = 0;
        for (size_t j = 0; j < data.cols; ++j) {
            double d = data(i, j) - centers[labels[i]](0, j);
            dist += d * d;
        }
        scatter[labels[i]] += std::sqrt(dist);
    }
    for (int c = 0; c < k; ++c)
        if (counts[c] > 0) scatter[c] /= counts[c];

    double db = 0;
    for (int i = 0; i < k; ++i) {
        double max_ratio = 0;
        for (int j = 0; j < k; ++j) {
            if (i == j) continue;
            double center_dist = 0;
            for (size_t f = 0; f < data.cols; ++f) {
                double d = centers[i](0, f) - centers[j](0, f);
                center_dist += d * d;
            }
            center_dist = std::sqrt(center_dist);
            double ratio = (scatter[i] + scatter[j]) / (center_dist + 1e-12);
            max_ratio = std::max(max_ratio, ratio);
        }
        db += max_ratio;
    }
    return db / k;
}

double adjusted_rand_index(const std::vector<int>& true_labels, const std::vector<int>& pred_labels) {
    size_t n = true_labels.size();
    std::set<int> true_set(true_labels.begin(), true_labels.end());
    std::set<int> pred_set(pred_labels.begin(), pred_labels.end());
    int nt = true_set.size(), np = pred_set.size();

    std::map<int, int> true_map, pred_map;
    int idx = 0;
    for (int t : true_set) true_map[t] = idx++;
    idx = 0;
    for (int p : pred_set) pred_map[p] = idx++;

    // Contingency table
    std::vector<std::vector<int>> ct(nt, std::vector<int>(np, 0));
    for (size_t i = 0; i < n; ++i)
        ct[true_map[true_labels[i]]][pred_map[pred_labels[i]]]++;

    auto comb2 = [](int x) -> double { return x >= 2 ? (double)x * (x - 1) / 2.0 : 0.0; };

    double sum_comb = 0;
    std::vector<int> a(nt, 0), b(np, 0);
    for (int i = 0; i < nt; ++i)
        for (int j = 0; j < np; ++j) {
            sum_comb += comb2(ct[i][j]);
            a[i] += ct[i][j];
            b[j] += ct[i][j];
        }

    double sum_a = 0, sum_b = 0;
    for (int i = 0; i < nt; ++i) sum_a += comb2(a[i]);
    for (int j = 0; j < np; ++j) sum_b += comb2(b[j]);

    double expected = sum_a * sum_b / comb2(n);
    double max_idx = 0.5 * (sum_a + sum_b);
    double denom = max_idx - expected;
    if (std::abs(denom) < 1e-12) return 0;
    return (sum_comb - expected) / denom;
}

double normalized_mutual_info(const std::vector<int>& true_labels, const std::vector<int>& pred_labels) {
    size_t n = true_labels.size();
    std::map<int, int> true_counts, pred_counts;
    std::map<std::pair<int,int>, int> joint;
    for (size_t i = 0; i < n; ++i) {
        true_counts[true_labels[i]]++;
        pred_counts[pred_labels[i]]++;
        joint[{true_labels[i], pred_labels[i]}]++;
    }

    double mi = 0;
    for (auto& [pair, count] : joint) {
        double pij = (double)count / n;
        double pi = (double)true_counts[pair.first] / n;
        double pj = (double)pred_counts[pair.second] / n;
        mi += pij * std::log(pij / (pi * pj + 1e-12) + 1e-12);
    }

    double h_true = 0, h_pred = 0;
    for (auto& [k, c] : true_counts) { double p = (double)c / n; h_true -= p * std::log(p + 1e-12); }
    for (auto& [k, c] : pred_counts) { double p = (double)c / n; h_pred -= p * std::log(p + 1e-12); }

    double denom = std::sqrt(h_true * h_pred);
    return (denom > 1e-12) ? mi / denom : 0;
}

} // namespace Metrics
