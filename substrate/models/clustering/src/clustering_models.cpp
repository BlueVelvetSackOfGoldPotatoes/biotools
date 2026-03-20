#include "clustering_models.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <queue>
#include <unordered_set>
#include <stdexcept>

#ifdef _OPENMP
#define CLUSTER_OMP_PAR_FOR _Pragma("omp parallel for schedule(static)")
#define CLUSTER_OMP_PAR_FOR_REDUCE_SHIFT _Pragma("omp parallel for reduction(+ : shift) schedule(static)")
#define CLUSTER_OMP_PAR_FOR_REDUCE_TOTAL _Pragma("omp parallel for reduction(+ : total) schedule(static)")
#define CLUSTER_OMP_PAR_FOR_REDUCE_LL _Pragma("omp parallel for reduction(+ : ll) schedule(static)")
#else
#define CLUSTER_OMP_PAR_FOR
#define CLUSTER_OMP_PAR_FOR_REDUCE_SHIFT
#define CLUSTER_OMP_PAR_FOR_REDUCE_TOTAL
#define CLUSTER_OMP_PAR_FOR_REDUCE_LL
#endif

// ============================================================================
// Helper: squared Euclidean distance between row i of A and row j of B
// ============================================================================
static double sq_dist_rows(const Tensor& A, size_t i, const Tensor& B, size_t j) {
    double d = 0.0;
    for (size_t f = 0; f < A.cols; ++f) {
        double diff = A(i, f) - B(j, f);
        d += diff * diff;
    }
    return d;
}

// Squared distance between row i of A and a raw pointer array of length D
[[maybe_unused]] static double sq_dist_row_ptr(const Tensor& A, size_t i, const double* p, size_t D) {
    double d = 0.0;
    for (size_t f = 0; f < D; ++f) {
        double diff = A(i, f) - p[f];
        d += diff * diff;
    }
    return d;
}

// Find nearest centroid for row i of X among centroids (k x D)
static int nearest_centroid(const Tensor& X, size_t i, const Tensor& centroids) {
    int best = 0;
    double best_d = sq_dist_rows(X, i, centroids, 0);
    for (size_t c = 1; c < centroids.rows; ++c) {
        double d = sq_dist_rows(X, i, centroids, c);
        if (d < best_d) {
            best_d = d;
            best = static_cast<int>(c);
        }
    }
    return best;
}

// ============================================================================
// KMeans
// ============================================================================

KMeans::KMeans(int k, int max_iter, std::mt19937& rng)
    : k(k), max_iter(max_iter), rng(rng) {}

void KMeans::fit(const Tensor& X) {
    size_t N = X.rows;
    size_t D = X.cols;
    if (k <= 0) {
        throw std::runtime_error("KMeans::fit requires k > 0");
    }
    if (N < static_cast<size_t>(k)) {
        throw std::runtime_error("KMeans::fit requires number of samples >= k");
    }

    // KMeans++ initialization
    centroids = Tensor(static_cast<size_t>(k), D);

    // Pick first centroid uniformly at random
    std::uniform_int_distribution<size_t> unif(0, N - 1);
    size_t first = unif(rng);
    for (size_t f = 0; f < D; ++f)
        centroids(0, f) = X(first, f);

    // Distance of each point to its nearest chosen centroid
    std::vector<double> min_dist(N, std::numeric_limits<double>::max());

    for (int c = 1; c < k; ++c) {
        // Update min_dist w.r.t. centroid c-1
        CLUSTER_OMP_PAR_FOR
        for (size_t i = 0; i < N; ++i) {
            double d = sq_dist_rows(X, i, centroids, static_cast<size_t>(c - 1));
            if (d < min_dist[i]) min_dist[i] = d;
        }
        // Sample proportional to min_dist
        std::discrete_distribution<size_t> dist(min_dist.begin(), min_dist.end());
        size_t chosen = dist(rng);
        for (size_t f = 0; f < D; ++f)
            centroids(static_cast<size_t>(c), f) = X(chosen, f);
    }

    // Lloyd iterations
    std::vector<int> assignments(N);
    for (int iter = 0; iter < max_iter; ++iter) {
        // Assign
        CLUSTER_OMP_PAR_FOR
        for (size_t i = 0; i < N; ++i)
            assignments[i] = nearest_centroid(X, i, centroids);

        // Recompute centroids
        Tensor new_centroids(static_cast<size_t>(k), D, 0.0);
        std::vector<int> counts(k, 0);

        for (size_t i = 0; i < N; ++i) {
            int c = assignments[i];
            counts[c]++;
            for (size_t f = 0; f < D; ++f)
                new_centroids(static_cast<size_t>(c), f) += X(i, f);
        }

        for (int c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                for (size_t f = 0; f < D; ++f)
                    new_centroids(static_cast<size_t>(c), f) /= counts[c];
            } else {
                // Reinitialize empty cluster to a random data point
                size_t idx = unif(rng);
                for (size_t f = 0; f < D; ++f)
                    new_centroids(static_cast<size_t>(c), f) = X(idx, f);
            }
        }

        // Check convergence
        double shift = 0.0;
        CLUSTER_OMP_PAR_FOR_REDUCE_SHIFT
        for (size_t c = 0; c < static_cast<size_t>(k); ++c)
            for (size_t f = 0; f < D; ++f) {
                double diff = new_centroids(c, f) - centroids(c, f);
                shift += diff * diff;
            }

        centroids = new_centroids;

        if (shift < 1e-10)
            break;
    }
}

std::vector<int> KMeans::predict(const Tensor& X) const {
    size_t N = X.rows;
    std::vector<int> labels(N);
    CLUSTER_OMP_PAR_FOR
    for (size_t i = 0; i < N; ++i)
        labels[i] = nearest_centroid(X, i, centroids);
    return labels;
}

double KMeans::inertia(const Tensor& X) const {
    size_t N = X.rows;
    double total = 0.0;
    CLUSTER_OMP_PAR_FOR_REDUCE_TOTAL
    for (size_t i = 0; i < N; ++i) {
        int c = nearest_centroid(X, i, centroids);
        total += sq_dist_rows(X, i, centroids, static_cast<size_t>(c));
    }
    return total;
}

// ============================================================================
// MiniBatchKMeans
// ============================================================================

MiniBatchKMeans::MiniBatchKMeans(int k, int max_iter, int batch_size, std::mt19937& rng)
    : k(k), max_iter(max_iter), batch_size(batch_size), rng(rng) {}

void MiniBatchKMeans::fit(const Tensor& X) {
    size_t N = X.rows;
    size_t D = X.cols;
    if (k <= 0) {
        throw std::runtime_error("MiniBatchKMeans::fit requires k > 0");
    }
    if (batch_size <= 0) {
        throw std::runtime_error("MiniBatchKMeans::fit requires batch_size > 0");
    }
    if (N < static_cast<size_t>(k)) {
        throw std::runtime_error("MiniBatchKMeans::fit requires number of samples >= k");
    }

    // KMeans++ initialization (same as KMeans)
    centroids = Tensor(static_cast<size_t>(k), D);

    std::uniform_int_distribution<size_t> unif(0, N - 1);
    size_t first = unif(rng);
    for (size_t f = 0; f < D; ++f)
        centroids(0, f) = X(first, f);

    std::vector<double> min_dist(N, std::numeric_limits<double>::max());
    for (int c = 1; c < k; ++c) {
        CLUSTER_OMP_PAR_FOR
        for (size_t i = 0; i < N; ++i) {
            double d = sq_dist_rows(X, i, centroids, static_cast<size_t>(c - 1));
            if (d < min_dist[i]) min_dist[i] = d;
        }
        std::discrete_distribution<size_t> dist(min_dist.begin(), min_dist.end());
        size_t chosen = dist(rng);
        for (size_t f = 0; f < D; ++f)
            centroids(static_cast<size_t>(c), f) = X(chosen, f);
    }

    // Per-centroid sample count for averaging
    std::vector<int> centroid_counts(k, 0);

    size_t actual_batch = std::min(static_cast<size_t>(batch_size), N);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Sample a mini-batch (random indices without full shuffle)
        std::vector<size_t> batch_idx(actual_batch);
        for (size_t b = 0; b < actual_batch; ++b)
            batch_idx[b] = unif(rng);

        // Assign each sample in the batch to its nearest centroid
        std::vector<int> batch_assign(actual_batch);
        for (size_t b = 0; b < actual_batch; ++b)
            batch_assign[b] = nearest_centroid(X, batch_idx[b], centroids);

        // Update centroids with streaming average
        for (size_t b = 0; b < actual_batch; ++b) {
            int c = batch_assign[b];
            centroid_counts[c]++;
            double eta = 1.0 / centroid_counts[c]; // learning rate
            size_t row = batch_idx[b];
            for (size_t f = 0; f < D; ++f)
                centroids(static_cast<size_t>(c), f) +=
                    eta * (X(row, f) - centroids(static_cast<size_t>(c), f));
        }
    }
}

std::vector<int> MiniBatchKMeans::predict(const Tensor& X) const {
    size_t N = X.rows;
    std::vector<int> labels(N);
    CLUSTER_OMP_PAR_FOR
    for (size_t i = 0; i < N; ++i)
        labels[i] = nearest_centroid(X, i, centroids);
    return labels;
}

// ============================================================================
// GMM (diagonal covariance)
// ============================================================================

GMM::GMM(int k, int max_iter, std::mt19937& rng, double tol)
    : k(k), max_iter(max_iter), tol(tol), rng(rng) {}

// Log of diagonal multivariate Gaussian pdf for a single point x (1 x D)
// given mean (1 x D) and variance (1 x D, diagonal entries)
static double log_gaussian_diag(const Tensor& x, const Tensor& mean,
                                const Tensor& var, size_t D) {
    // log N(x | mu, diag(var)) = -0.5 * [ D*log(2pi) + sum(log(var_j)) + sum((x_j-mu_j)^2/var_j) ]
    static const double LOG2PI = std::log(2.0 * M_PI);
    double log_det = 0.0;
    double mahal = 0.0;
    for (size_t j = 0; j < D; ++j) {
        double v = var(0, j);
        log_det += std::log(v);
        double diff = x(0, j) - mean(0, j);
        mahal += diff * diff / v;
    }
    return -0.5 * (D * LOG2PI + log_det + mahal);
}

// Log-sum-exp of a vector
static double log_sum_exp(const std::vector<double>& v) {
    double mx = *std::max_element(v.begin(), v.end());
    double s = 0.0;
    for (double x : v) s += std::exp(x - mx);
    return mx + std::log(s);
}

void GMM::fit(const Tensor& X) {
    size_t N = X.rows;
    size_t D = X.cols;
    if (k <= 0) {
        throw std::runtime_error("GMM::fit requires k > 0");
    }
    if (N < static_cast<size_t>(k)) {
        throw std::runtime_error("GMM::fit requires number of samples >= k");
    }

    // Initialize with KMeans
    KMeans km(k, 20, rng);
    km.fit(X);
    std::vector<int> init_labels = km.predict(X);
    Tensor km_centroids = km.get_centroids();

    means.resize(k);
    covariances.resize(k);
    weights.resize(k);

    for (int c = 0; c < k; ++c) {
        means[c] = km_centroids.row(static_cast<size_t>(c)); // 1 x D
        covariances[c] = Tensor(1, D, 1.0); // initialize to unit variance
        weights[c] = 1.0 / k;
    }

    // Compute initial variances from KMeans assignments
    std::vector<int> counts(k, 0);
    for (size_t i = 0; i < N; ++i)
        counts[init_labels[i]]++;

    for (int c = 0; c < k; ++c) {
        if (counts[c] > 1) {
            Tensor var(1, D, 0.0);
            for (size_t i = 0; i < N; ++i) {
                if (init_labels[i] == c) {
                    for (size_t j = 0; j < D; ++j) {
                        double diff = X(i, j) - means[c](0, j);
                        var(0, j) += diff * diff;
                    }
                }
            }
            for (size_t j = 0; j < D; ++j) {
                var(0, j) /= counts[c];
                // Floor variance to avoid singularity
                if (var(0, j) < 1e-6) var(0, j) = 1e-6;
            }
            covariances[c] = var;
        }
        weights[c] = static_cast<double>(counts[c]) / N;
        if (weights[c] < 1e-10) weights[c] = 1e-10;
    }

    // EM iterations
    // responsibilities: N x k
    Tensor resp(N, static_cast<size_t>(k));
    double prev_ll = -std::numeric_limits<double>::max();

    for (int iter = 0; iter < max_iter; ++iter) {
        // --- E-step ---
        for (size_t i = 0; i < N; ++i) {
            std::vector<double> log_probs(k);
            Tensor xi = X.row(i); // 1 x D
            for (int c = 0; c < k; ++c) {
                log_probs[c] = std::log(weights[c]) +
                               log_gaussian_diag(xi, means[c], covariances[c], D);
            }
            double lse = log_sum_exp(log_probs);
            for (int c = 0; c < k; ++c) {
                resp(i, static_cast<size_t>(c)) = std::exp(log_probs[c] - lse);
            }
        }

        // --- M-step ---
        for (int c = 0; c < k; ++c) {
            double Nc = 0.0;
            for (size_t i = 0; i < N; ++i)
                Nc += resp(i, static_cast<size_t>(c));

            if (Nc < 1e-10) {
                // Dead component: reinitialize to random point
                std::uniform_int_distribution<size_t> uid(0, N - 1);
                size_t ri = uid(rng);
                means[c] = X.row(ri);
                covariances[c] = Tensor(1, D, 1.0);
                weights[c] = 1.0 / k;
                continue;
            }

            weights[c] = Nc / N;

            // New mean
            Tensor new_mean(1, D, 0.0);
            for (size_t i = 0; i < N; ++i) {
                double r = resp(i, static_cast<size_t>(c));
                for (size_t j = 0; j < D; ++j)
                    new_mean(0, j) += r * X(i, j);
            }
            for (size_t j = 0; j < D; ++j)
                new_mean(0, j) /= Nc;
            means[c] = new_mean;

            // New variance (diagonal)
            Tensor new_var(1, D, 0.0);
            for (size_t i = 0; i < N; ++i) {
                double r = resp(i, static_cast<size_t>(c));
                for (size_t j = 0; j < D; ++j) {
                    double diff = X(i, j) - new_mean(0, j);
                    new_var(0, j) += r * diff * diff;
                }
            }
            for (size_t j = 0; j < D; ++j) {
                new_var(0, j) /= Nc;
                if (new_var(0, j) < 1e-6) new_var(0, j) = 1e-6;
            }
            covariances[c] = new_var;
        }

        // Check convergence via log-likelihood
        double ll = log_likelihood(X);
        if (std::abs(ll - prev_ll) < tol)
            break;
        prev_ll = ll;
    }
}

std::vector<int> GMM::predict(const Tensor& X) const {
    size_t N = X.rows;
    size_t D = X.cols;
    std::vector<int> labels(N);
    CLUSTER_OMP_PAR_FOR
    for (size_t i = 0; i < N; ++i) {
        Tensor xi = X.row(i);
        int best = 0;
        double best_lp = -std::numeric_limits<double>::max();
        for (int c = 0; c < k; ++c) {
            double lp = std::log(weights[c]) +
                        log_gaussian_diag(xi, means[c], covariances[c], D);
            if (lp > best_lp) {
                best_lp = lp;
                best = c;
            }
        }
        labels[i] = best;
    }
    return labels;
}

Tensor GMM::predict_proba(const Tensor& X) const {
    size_t N = X.rows;
    size_t D = X.cols;
    Tensor proba(N, static_cast<size_t>(k));
    CLUSTER_OMP_PAR_FOR
    for (size_t i = 0; i < N; ++i) {
        Tensor xi = X.row(i);
        std::vector<double> log_probs(k);
        for (int c = 0; c < k; ++c) {
            log_probs[c] = std::log(weights[c]) +
                           log_gaussian_diag(xi, means[c], covariances[c], D);
        }
        double lse = log_sum_exp(log_probs);
        for (int c = 0; c < k; ++c)
            proba(i, static_cast<size_t>(c)) = std::exp(log_probs[c] - lse);
    }
    return proba;
}

double GMM::log_likelihood(const Tensor& X) const {
    size_t N = X.rows;
    size_t D = X.cols;
    double ll = 0.0;
    CLUSTER_OMP_PAR_FOR_REDUCE_LL
    for (size_t i = 0; i < N; ++i) {
        Tensor xi = X.row(i);
        std::vector<double> log_probs(k);
        for (int c = 0; c < k; ++c) {
            log_probs[c] = std::log(weights[c]) +
                           log_gaussian_diag(xi, means[c], covariances[c], D);
        }
        ll += log_sum_exp(log_probs);
    }
    return ll;
}

// ============================================================================
// DBSCAN
// ============================================================================

DBSCAN::DBSCAN(double eps, int min_pts)
    : eps(eps), min_pts(min_pts) {}

std::vector<int> DBSCAN::fit_predict(const Tensor& X) const {
    size_t N = X.rows;
    double eps_sq = eps * eps;

    // Precompute neighbor lists
    std::vector<std::vector<size_t>> neighbors(N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            double d = sq_dist_rows(X, i, X, j);
            if (d <= eps_sq) {
                neighbors[i].push_back(j);
                neighbors[j].push_back(i);
            }
        }
    }

    std::vector<int> labels(N, -1);
    int cluster_id = 0;

    // visited tracking
    std::vector<bool> visited(N, false);

    for (size_t i = 0; i < N; ++i) {
        if (visited[i]) continue;
        visited[i] = true;

        if (static_cast<int>(neighbors[i].size()) + 1 < min_pts) {
            // Noise (may be claimed by a cluster later)
            continue;
        }

        // Expand cluster
        labels[i] = cluster_id;
        std::queue<size_t> q;
        for (size_t nb : neighbors[i])
            q.push(nb);

        while (!q.empty()) {
            size_t pt = q.front();
            q.pop();

            if (!visited[pt]) {
                visited[pt] = true;
                // Check if pt is a core point
                if (static_cast<int>(neighbors[pt].size()) + 1 >= min_pts) {
                    for (size_t nb : neighbors[pt])
                        q.push(nb);
                }
            }

            if (labels[pt] == -1) {
                labels[pt] = cluster_id;
            }
        }

        cluster_id++;
    }

    return labels;
}

// ============================================================================
// AgglomerativeClustering
// ============================================================================

AgglomerativeClustering::AgglomerativeClustering(int n_clusters, const std::string& linkage)
    : n_clusters(n_clusters), linkage(linkage) {}

std::vector<AgglomerativeClustering::MergeStep>
AgglomerativeClustering::get_dendrogram(const Tensor& X) const {
    size_t N = X.rows;

    // Compute pairwise distance matrix (upper triangle stored in condensed form)
    // But for efficiency with linkage updates, use a full NxN matrix approach
    // with cluster tracking.

    // dist_matrix: distance between cluster i and cluster j
    // We use a flat vector indexed as dist_matrix[i * N + j]
    std::vector<double> dist(N * N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            double d = std::sqrt(sq_dist_rows(X, i, X, j));
            dist[i * N + j] = d;
            dist[j * N + i] = d;
        }
    }

    // Track which original points belong to each cluster
    std::vector<std::vector<size_t>> cluster_members(N);
    for (size_t i = 0; i < N; ++i)
        cluster_members[i].push_back(i);

    std::vector<bool> active(N, true);
    std::vector<MergeStep> merges;

    for (size_t step = 0; step < N - 1; ++step) {
        // Find closest pair of active clusters
        double best_d = std::numeric_limits<double>::max();
        size_t best_i = 0, best_j = 0;

        for (size_t i = 0; i < N; ++i) {
            if (!active[i]) continue;
            for (size_t j = i + 1; j < N; ++j) {
                if (!active[j]) continue;
                if (dist[i * N + j] < best_d) {
                    best_d = dist[i * N + j];
                    best_i = i;
                    best_j = j;
                }
            }
        }

        merges.push_back({static_cast<int>(best_i), static_cast<int>(best_j), best_d});

        // Merge best_j into best_i
        for (size_t idx : cluster_members[best_j])
            cluster_members[best_i].push_back(idx);
        active[best_j] = false;

        // Update distances from merged cluster (best_i) to all other active clusters
        for (size_t m = 0; m < N; ++m) {
            if (!active[m] || m == best_i) continue;

            double new_d;
            if (linkage == "single") {
                new_d = std::min(dist[best_i * N + m], dist[best_j * N + m]);
            } else if (linkage == "complete") {
                new_d = std::max(dist[best_i * N + m], dist[best_j * N + m]);
            } else {
                // "average" linkage: recompute from members
                double sum_d = 0.0;
                size_t count = 0;
                for (size_t a : cluster_members[best_i]) {
                    for (size_t b : cluster_members[m]) {
                        sum_d += std::sqrt(sq_dist_rows(X, a, X, b));
                        count++;
                    }
                }
                new_d = sum_d / count;
            }
            dist[best_i * N + m] = new_d;
            dist[m * N + best_i] = new_d;
        }
    }

    return merges;
}

std::vector<int> AgglomerativeClustering::fit_predict(const Tensor& X) const {
    size_t N = X.rows;
    size_t D = X.cols;

    // For large datasets, subsample to 1000, cluster, then assign rest by nearest centroid
    const size_t MAX_AGGLOM = 1000;

    if (N <= MAX_AGGLOM) {
        // Directly cluster
        auto merges = get_dendrogram(X);

        // Reconstruct cluster assignments from the dendrogram
        // Start: each point is its own cluster
        // Apply merges until we have n_clusters remaining
        std::vector<int> label(N);
        std::iota(label.begin(), label.end(), 0);

        size_t n_merges_to_apply = N - static_cast<size_t>(n_clusters);
        // Union-find style: when merging c1 and c2, relabel all c2 to c1
        for (size_t s = 0; s < n_merges_to_apply && s < merges.size(); ++s) {
            int old_label = label[merges[s].c2];
            int new_label = label[merges[s].c1];
            for (size_t i = 0; i < N; ++i) {
                if (label[i] == old_label)
                    label[i] = new_label;
            }
        }

        // Renumber labels to 0..n_clusters-1
        std::unordered_set<int> unique_labels(label.begin(), label.end());
        std::vector<int> sorted_labels(unique_labels.begin(), unique_labels.end());
        std::sort(sorted_labels.begin(), sorted_labels.end());
        std::unordered_map<int, int> label_map;
        for (size_t i = 0; i < sorted_labels.size(); ++i)
            label_map[sorted_labels[i]] = static_cast<int>(i);

        std::vector<int> result(N);
        for (size_t i = 0; i < N; ++i)
            result[i] = label_map[label[i]];
        return result;
    }

    // Subsample
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    // Deterministic shuffle using a copy of some seed
    // We don't have rng here, so use a fixed seed derived from N
    std::mt19937 local_rng(static_cast<unsigned>(N * 42 + 7));
    std::shuffle(indices.begin(), indices.end(), local_rng);

    size_t sub_n = MAX_AGGLOM;
    Tensor sub_X(sub_n, D);
    for (size_t i = 0; i < sub_n; ++i)
        for (size_t j = 0; j < D; ++j)
            sub_X(i, j) = X(indices[i], j);

    // Cluster the subsample
    AgglomerativeClustering sub_clust(n_clusters, linkage);
    std::vector<int> sub_labels = sub_clust.fit_predict(sub_X);

    // Compute centroids of each cluster in the subsample
    Tensor centroids_mat(static_cast<size_t>(n_clusters), D, 0.0);
    std::vector<int> counts(n_clusters, 0);
    for (size_t i = 0; i < sub_n; ++i) {
        int c = sub_labels[i];
        counts[c]++;
        for (size_t j = 0; j < D; ++j)
            centroids_mat(static_cast<size_t>(c), j) += sub_X(i, j);
    }
    for (int c = 0; c < n_clusters; ++c) {
        if (counts[c] > 0) {
            for (size_t j = 0; j < D; ++j)
                centroids_mat(static_cast<size_t>(c), j) /= counts[c];
        }
    }

    // Assign all points to nearest centroid
    std::vector<int> result(N);
    for (size_t i = 0; i < N; ++i)
        result[i] = nearest_centroid(X, i, centroids_mat);

    return result;
}

// ============================================================================
// SpectralClustering
// ============================================================================

SpectralClustering::SpectralClustering(int k, double sigma, std::mt19937& rng)
    : k(k), sigma(sigma), rng(rng) {}

Tensor SpectralClustering::compute_laplacian(const Tensor& X) const {
    size_t N = X.rows;
    double two_sigma_sq = 2.0 * sigma * sigma;

    // Affinity matrix W (Gaussian kernel)
    Tensor W(N, N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            double d = sq_dist_rows(X, i, X, j);
            double w = std::exp(-d / two_sigma_sq);
            W(i, j) = w;
            W(j, i) = w;
        }
    }

    // Degree vector D (diagonal)
    std::vector<double> deg(N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j)
            deg[i] += W(i, j);
    }

    // Normalized Laplacian: L_sym = D^{-1/2} (D - W) D^{-1/2}
    //   = I - D^{-1/2} W D^{-1/2}
    Tensor L(N, N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        double di_inv_sqrt = (deg[i] > 1e-15) ? 1.0 / std::sqrt(deg[i]) : 0.0;
        for (size_t j = 0; j < N; ++j) {
            double dj_inv_sqrt = (deg[j] > 1e-15) ? 1.0 / std::sqrt(deg[j]) : 0.0;
            L(i, j) = -di_inv_sqrt * W(i, j) * dj_inv_sqrt;
        }
        L(i, i) += 1.0; // diagonal of I
    }

    return L;
}

Tensor SpectralClustering::compute_eigenvectors(const Tensor& L, int num) const {
    // We need the smallest eigenvectors of L (L_sym).
    // Since L is PSD with smallest eigenvalues near 0, we find them by
    // computing largest eigenvectors of (max_eigenvalue * I - L), i.e., shifted.
    // But estimating max eigenvalue is costly.
    //
    // Alternative: use inverse power iteration. But L may be singular (eigenvalue 0).
    // Instead, use power iteration on (I - L) = D^{-1/2} W D^{-1/2}
    // which has the LARGEST eigenvalues corresponding to the smallest of L.

    size_t N = L.rows;

    // Compute M = I - L = D^{-1/2} W D^{-1/2}
    // M's largest eigenvalues correspond to L's smallest eigenvalues.
    Tensor M(N, N);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            M(i, j) = (i == j ? 1.0 : 0.0) - L(i, j);

    // Power iteration with deflation to find top-`num` eigenvectors of M
    Tensor eigvecs(N, static_cast<size_t>(num));

    // Store found eigenvectors for deflation
    std::vector<std::vector<double>> found_vecs;

    std::mt19937 local_rng(rng()); // local copy to not perturb caller's state too much

    for (int ev = 0; ev < num; ++ev) {
        // Random initial vector
        std::vector<double> v(N);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (size_t i = 0; i < N; ++i)
            v[i] = nd(local_rng);

        // Orthogonalize against previously found eigenvectors
        for (const auto& prev : found_vecs) {
            double dot = 0.0;
            for (size_t i = 0; i < N; ++i)
                dot += v[i] * prev[i];
            for (size_t i = 0; i < N; ++i)
                v[i] -= dot * prev[i];
        }

        // Normalize
        double norm = 0.0;
        for (size_t i = 0; i < N; ++i) norm += v[i] * v[i];
        norm = std::sqrt(norm);
        if (norm > 1e-15)
            for (size_t i = 0; i < N; ++i) v[i] /= norm;

        // Power iterations
        int power_iters = 300;
        for (int it = 0; it < power_iters; ++it) {
            // w = M * v
            std::vector<double> w(N, 0.0);
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < N; ++j)
                    w[i] += M(i, j) * v[j];

            // Deflation: orthogonalize against found eigenvectors
            for (const auto& prev : found_vecs) {
                double dot = 0.0;
                for (size_t i = 0; i < N; ++i)
                    dot += w[i] * prev[i];
                for (size_t i = 0; i < N; ++i)
                    w[i] -= dot * prev[i];
            }

            // Normalize
            norm = 0.0;
            for (size_t i = 0; i < N; ++i) norm += w[i] * w[i];
            norm = std::sqrt(norm);
            if (norm < 1e-15) break;
            for (size_t i = 0; i < N; ++i)
                v[i] = w[i] / norm;
        }

        found_vecs.push_back(v);
        for (size_t i = 0; i < N; ++i)
            eigvecs(i, static_cast<size_t>(ev)) = v[i];
    }

    return eigvecs; // N x num
}

std::vector<int> SpectralClustering::fit_predict(const Tensor& X) const {
    size_t N = X.rows;

    // Compute normalized graph Laplacian
    Tensor L = compute_laplacian(X);

    // Extract k smallest eigenvectors (via power iteration on I - L)
    Tensor V = compute_eigenvectors(L, k); // N x k

    // Row-normalize V so each row has unit norm
    for (size_t i = 0; i < N; ++i) {
        double norm = 0.0;
        for (int j = 0; j < k; ++j)
            norm += V(i, static_cast<size_t>(j)) * V(i, static_cast<size_t>(j));
        norm = std::sqrt(norm);
        if (norm > 1e-15) {
            for (int j = 0; j < k; ++j)
                V(i, static_cast<size_t>(j)) /= norm;
        }
    }

    // Run KMeans on V
    // We need a non-const rng copy since KMeans::fit needs a mutable rng
    std::mt19937 local_rng(rng); // copy
    KMeans km(k, 100, local_rng);
    km.fit(V);
    return km.predict(V);
}
