#pragma once

#include "../../../core/tensor/tensor.h"
#include <vector>
#include <string>
#include <random>

// ============================================================================
// KMeans
// ============================================================================
class KMeans {
    int k, max_iter;
    Tensor centroids; // k x features
    std::mt19937& rng;

public:
    KMeans(int k, int max_iter, std::mt19937& rng);
    void fit(const Tensor& X);
    std::vector<int> predict(const Tensor& X) const;
    Tensor get_centroids() const { return centroids; }
    double inertia(const Tensor& X) const;
};

// ============================================================================
// MiniBatchKMeans
// ============================================================================
class MiniBatchKMeans {
    int k, max_iter, batch_size;
    Tensor centroids;
    std::mt19937& rng;

public:
    MiniBatchKMeans(int k, int max_iter, int batch_size, std::mt19937& rng);
    void fit(const Tensor& X);
    std::vector<int> predict(const Tensor& X) const;
    Tensor get_centroids() const { return centroids; }
};

// ============================================================================
// GMM (Gaussian Mixture Model) - diagonal covariance
// ============================================================================
class GMM {
    int k, max_iter;
    double tol;
    std::vector<Tensor> means;       // k tensors of 1 x D
    std::vector<Tensor> covariances; // k tensors of 1 x D (diagonal variances)
    std::vector<double> weights;     // mixing weights
    std::mt19937& rng;

public:
    GMM(int k, int max_iter, std::mt19937& rng, double tol = 1e-4);
    void fit(const Tensor& X);
    std::vector<int> predict(const Tensor& X) const;
    Tensor predict_proba(const Tensor& X) const; // N x k
    double log_likelihood(const Tensor& X) const;
};

// ============================================================================
// DBSCAN
// ============================================================================
class DBSCAN {
    double eps;
    int min_pts;

public:
    DBSCAN(double eps, int min_pts);
    std::vector<int> fit_predict(const Tensor& X) const; // -1 for noise
};

// ============================================================================
// AgglomerativeClustering
// ============================================================================
class AgglomerativeClustering {
    int n_clusters;
    std::string linkage; // "single", "complete", "average"

public:
    AgglomerativeClustering(int n_clusters, const std::string& linkage = "average");
    std::vector<int> fit_predict(const Tensor& X) const;

    struct MergeStep {
        int c1, c2;
        double dist;
    };
    std::vector<MergeStep> get_dendrogram(const Tensor& X) const;
};

// ============================================================================
// SpectralClustering
// ============================================================================
class SpectralClustering {
    int k;
    double sigma;
    std::mt19937& rng;

public:
    SpectralClustering(int k, double sigma, std::mt19937& rng);
    std::vector<int> fit_predict(const Tensor& X) const;

private:
    Tensor compute_laplacian(const Tensor& X) const;
    Tensor compute_eigenvectors(const Tensor& L, int num) const;
};
