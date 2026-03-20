#include "core/tensor/tensor.h"
#include "core/data/dataloader.h"
#include "core/metrics/metrics.h"
#include "core/io/run_logger.h"
#include "models/clustering/src/clustering_models.h"
#include "benchmarks/logging_utils.h"
#include "benchmarks/hpo_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

static void log_common_clustering(RunLogger& logger,
                                  const std::vector<int>& pred,
                                  const std::vector<int>& true_labels,
                                  double quality_metric,
                                  double loss_like,
                                  double elapsed_s) {
    const double elapsed_ms = elapsed_s * 1000.0;
    const double per_sample_ms = elapsed_ms / static_cast<double>(pred.size());
    logger.log_epoch_metric(
        1,
        "test",
        loss_like,
        quality_metric,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        0.0,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        elapsed_ms,
        pred.size() / (elapsed_s + 1e-12)
    );
    benchlog::log_confusion_and_class_metrics(logger, 1, "test", pred, true_labels, 10);
    benchlog::log_inference_from_preds(logger, "test", pred, true_labels, per_sample_ms, static_cast<int>(pred.size()));
    benchlog::log_calibration_from_preds(logger, "test", pred, true_labels, 10);
    logger.log_system_metric(
        RunLogger::utc_now_iso8601(),
        pred.size() / (elapsed_s + 1e-12),
        per_sample_ms,
        per_sample_ms,
        per_sample_ms,
        0.0,
        0.0
    );
}

int main() {
    std::cout << "=== Clustering Benchmark (MNIST) ===" << std::endl;
    const int seed = hpo::env_int("CLUSTERING_SEED", 42);
    std::mt19937 rng(static_cast<unsigned int>(seed));
    const int k = hpo::env_int("CLUSTERING_K", 10);
    const int max_iter = hpo::env_int("CLUSTERING_MAX_ITER", 100);
    const int minibatch_size = hpo::env_int("CLUSTERING_MINIBATCH_SIZE", 256);
    const size_t n = hpo::env_size("CLUSTERING_N", 3000);
    const size_t agg_n = hpo::env_size("CLUSTERING_AGG_N", 500);
    const std::string algorithm_filter = hpo::to_lower(hpo::env_string("CLUSTERING_ALGORITHM", "all"));

    auto test = MNISTLoader::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    Tensor data = test.images.slice_rows(0, n);
    std::vector<int> true_labels(test.labels.begin(), test.labels.begin() + static_cast<long>(n));
    double best_nmi = 0.0;

    if (algorithm_filter == "all" || algorithm_filter == "kmeans") {
        std::cout << "\n--- KMeans (k=10) ---" << std::endl;
        std::ostringstream params_json;
        params_json << "{\"algorithm\":\"kmeans\",\"k\":" << k << ",\"max_iter\":" << max_iter << "}";
        RunLogger logger(
            "clustering",
            "kmeans_k10",
            seed,
            "mnist-idx-v1",
            params_json.str()
        );
        benchlog::BenchBioHarness bio("clustering", "kmeans_k10", seed);
        bio.attach({}, {}, "cluster_param");
        bio.begin_epoch(1);
        auto start = std::chrono::high_resolution_clock::now();
        KMeans km(k, max_iter, rng);
        km.fit(data);
        auto pred = km.predict(data);
        auto end = std::chrono::high_resolution_clock::now();
        const double elapsed = std::chrono::duration<double>(end - start).count();

        const double ari = Metrics::adjusted_rand_index(true_labels, pred);
        const double nmi = Metrics::normalized_mutual_info(true_labels, pred);
        best_nmi = std::max(best_nmi, nmi);
        const double inertia = km.inertia(data);

        log_common_clustering(logger, pred, true_labels, nmi, inertia, elapsed);
        logger.append_csv_row(
            "model_specific/clustering/cluster_metrics.csv",
            {"run_id", "step", "inertia"},
            {logger.run_id(), "1", std::to_string(inertia)}
        );
        Tensor c = km.get_centroids();
        for (size_t cid = 0; cid < c.rows; ++cid) {
            double norm = 0.0;
            for (size_t j = 0; j < c.cols; ++j) {
                norm += c(cid, j) * c(cid, j);
            }
            logger.append_csv_row(
                "model_specific/clustering/centroid_shift.csv",
                {"run_id", "step", "cluster_id", "shift_l2"},
                {logger.run_id(), "1", std::to_string(cid), std::to_string(std::sqrt(norm))}
            );
        }

        // Cluster size distribution
        {
            std::vector<int> sizes(10, 0);
            for (int c : pred) {
                if (c >= 0 && c < 10) sizes[c]++;
            }
            for (int c = 0; c < 10; ++c) {
                logger.append_csv_row(
                    "model_specific/clustering/cluster_sizes.csv",
                    {"run_id", "algorithm", "cluster_id", "size", "fraction"},
                    {logger.run_id(), "kmeans", std::to_string(c),
                     std::to_string(sizes[c]),
                     std::to_string(static_cast<double>(sizes[c]) / static_cast<double>(pred.size()))}
                );
            }
        }

        std::cout << "ARI: " << std::fixed << std::setprecision(4) << ari << " | NMI: " << nmi
                  << " | Inertia: " << std::setprecision(1) << inertia << " (" << elapsed << "s)"
                  << std::endl;
        bio.end_epoch(logger, 1);
        logger.write_manifest_end();
    }

    if (algorithm_filter == "all" || algorithm_filter == "minibatch" || algorithm_filter == "minibatch_kmeans") {
        std::cout << "\n--- MiniBatchKMeans (k=10) ---" << std::endl;
        std::ostringstream params_json;
        params_json << "{\"algorithm\":\"minibatch_kmeans\",\"k\":" << k
                    << ",\"max_iter\":" << max_iter
                    << ",\"batch_size\":" << minibatch_size << "}";
        RunLogger logger(
            "clustering",
            "minibatch_kmeans_k10",
            seed,
            "mnist-idx-v1",
            params_json.str()
        );
        benchlog::BenchBioHarness bio("clustering", "minibatch_kmeans_k10", seed);
        bio.attach({}, {}, "cluster_param");
        bio.begin_epoch(1);
        auto start = std::chrono::high_resolution_clock::now();
        MiniBatchKMeans mbk(k, max_iter, minibatch_size, rng);
        mbk.fit(data);
        auto pred = mbk.predict(data);
        auto end = std::chrono::high_resolution_clock::now();
        const double elapsed = std::chrono::duration<double>(end - start).count();

        const double ari = Metrics::adjusted_rand_index(true_labels, pred);
        const double nmi = Metrics::normalized_mutual_info(true_labels, pred);
        best_nmi = std::max(best_nmi, nmi);

        log_common_clustering(
            logger, pred, true_labels, nmi, std::numeric_limits<double>::quiet_NaN(), elapsed
        );
        logger.append_csv_row(
            "model_specific/clustering/cluster_metrics.csv",
            {"run_id", "step", "inertia"},
            {logger.run_id(), "1", "nan"}
        );
        Tensor c = mbk.get_centroids();
        for (size_t cid = 0; cid < c.rows; ++cid) {
            double norm = 0.0;
            for (size_t j = 0; j < c.cols; ++j) {
                norm += c(cid, j) * c(cid, j);
            }
            logger.append_csv_row(
                "model_specific/clustering/centroid_shift.csv",
                {"run_id", "step", "cluster_id", "shift_l2"},
                {logger.run_id(), "1", std::to_string(cid), std::to_string(std::sqrt(norm))}
            );
        }

        // Cluster size distribution
        {
            std::vector<int> sizes(10, 0);
            for (int c : pred) {
                if (c >= 0 && c < 10) sizes[c]++;
            }
            for (int c = 0; c < 10; ++c) {
                logger.append_csv_row(
                    "model_specific/clustering/cluster_sizes.csv",
                    {"run_id", "algorithm", "cluster_id", "size", "fraction"},
                    {logger.run_id(), "minibatch", std::to_string(c),
                     std::to_string(sizes[c]),
                     std::to_string(static_cast<double>(sizes[c]) / static_cast<double>(pred.size()))}
                );
            }
        }

        std::cout << "ARI: " << std::fixed << std::setprecision(4) << ari << " | NMI: " << nmi
                  << " (" << elapsed << "s)" << std::endl;
        bio.end_epoch(logger, 1);
        logger.write_manifest_end();
    }

    if (algorithm_filter == "all" || algorithm_filter == "gmm") {
        std::cout << "\n--- GMM (k=10, diagonal) ---" << std::endl;
        std::ostringstream params_json;
        params_json << "{\"algorithm\":\"gmm\",\"k\":" << k << ",\"max_iter\":50}";
        RunLogger logger(
            "clustering",
            "gmm_k10_diag",
            seed,
            "mnist-idx-v1",
            params_json.str()
        );
        benchlog::BenchBioHarness bio("clustering", "gmm_k10_diag", seed);
        bio.attach({}, {}, "cluster_param");
        bio.begin_epoch(1);
        auto start = std::chrono::high_resolution_clock::now();
        GMM gmm(k, 50, rng);
        gmm.fit(data);
        auto pred = gmm.predict(data);
        auto end = std::chrono::high_resolution_clock::now();
        const double elapsed = std::chrono::duration<double>(end - start).count();

        const double ari = Metrics::adjusted_rand_index(true_labels, pred);
        const double nmi = Metrics::normalized_mutual_info(true_labels, pred);
        best_nmi = std::max(best_nmi, nmi);
        const double ll = gmm.log_likelihood(data);

        log_common_clustering(logger, pred, true_labels, nmi, -ll, elapsed);
        logger.append_csv_row(
            "model_specific/clustering/cluster_metrics.csv",
            {"run_id", "step", "inertia"},
            {logger.run_id(), "1", std::to_string(-ll)}
        );
        Tensor c = gmm.predict_proba(data.slice_rows(0, 1));
        for (size_t cid = 0; cid < 10; ++cid) {
            logger.append_csv_row(
                "model_specific/clustering/centroid_shift.csv",
                {"run_id", "step", "cluster_id", "shift_l2"},
                {logger.run_id(), "1", std::to_string(cid), std::to_string(c(0, cid))}
            );
        }

        // Cluster size distribution
        {
            std::vector<int> sizes(10, 0);
            for (int c : pred) {
                if (c >= 0 && c < 10) sizes[c]++;
            }
            for (int c = 0; c < 10; ++c) {
                logger.append_csv_row(
                    "model_specific/clustering/cluster_sizes.csv",
                    {"run_id", "algorithm", "cluster_id", "size", "fraction"},
                    {logger.run_id(), "gmm", std::to_string(c),
                     std::to_string(sizes[c]),
                     std::to_string(static_cast<double>(sizes[c]) / static_cast<double>(pred.size()))}
                );
            }
        }

        std::cout << "ARI: " << std::fixed << std::setprecision(4) << ari << " | NMI: " << nmi
                  << " (" << elapsed << "s)" << std::endl;
        bio.end_epoch(logger, 1);
        logger.write_manifest_end();
    }

    if (algorithm_filter == "all" || algorithm_filter == "agglomerative") {
        Tensor agg_data = data.slice_rows(0, agg_n);
        std::vector<int> agg_labels(true_labels.begin(), true_labels.begin() + static_cast<long>(agg_n));

        std::cout << "\n--- Agglomerative (n=" << agg_n << ", k=10, average) ---" << std::endl;
        RunLogger logger(
            "clustering",
            "agglomerative_k10_average",
            seed,
            "mnist-idx-v1",
            "{\"algorithm\":\"agglomerative\",\"k\":10,\"linkage\":\"average\"}"
        );
        benchlog::BenchBioHarness bio("clustering", "agglomerative_k10_average", seed);
        bio.attach({}, {}, "cluster_param");
        bio.begin_epoch(1);
        auto start = std::chrono::high_resolution_clock::now();
        AgglomerativeClustering agg(k, "average");
        auto pred = agg.fit_predict(agg_data);
        auto end = std::chrono::high_resolution_clock::now();
        const double elapsed = std::chrono::duration<double>(end - start).count();

        const double ari = Metrics::adjusted_rand_index(agg_labels, pred);
        const double nmi = Metrics::normalized_mutual_info(agg_labels, pred);
        best_nmi = std::max(best_nmi, nmi);

        log_common_clustering(
            logger, pred, agg_labels, nmi, std::numeric_limits<double>::quiet_NaN(), elapsed
        );
        logger.append_csv_row(
            "model_specific/clustering/cluster_metrics.csv",
            {"run_id", "step", "inertia"},
            {logger.run_id(), "1", "nan"}
        );
        auto dendro = agg.get_dendrogram(agg_data);
        size_t n_steps = std::min(static_cast<size_t>(10), dendro.size());
        for (size_t i = 0; i < n_steps; ++i) {
            logger.append_csv_row(
                "model_specific/clustering/centroid_shift.csv",
                {"run_id", "step", "cluster_id", "shift_l2"},
                {logger.run_id(), std::to_string(i + 1), std::to_string(i % 10), std::to_string(dendro[i].dist)}
            );
        }

        // Cluster size distribution
        {
            std::vector<int> sizes(10, 0);
            for (int c : pred) {
                if (c >= 0 && c < 10) sizes[c]++;
            }
            for (int c = 0; c < 10; ++c) {
                logger.append_csv_row(
                    "model_specific/clustering/cluster_sizes.csv",
                    {"run_id", "algorithm", "cluster_id", "size", "fraction"},
                    {logger.run_id(), "agglomerative", std::to_string(c),
                     std::to_string(sizes[c]),
                     std::to_string(static_cast<double>(sizes[c]) / static_cast<double>(pred.size()))}
                );
            }
        }

        std::cout << "ARI: " << std::fixed << std::setprecision(4) << ari << " | NMI: " << nmi
                  << " (" << elapsed << "s)" << std::endl;
        bio.end_epoch(logger, 1);
        logger.write_manifest_end();
    }

    std::cout << "\n=== Clustering Benchmark Complete ===" << std::endl;
    std::cout << "Note: Clustering uses ARI/NMI metrics (mapped to quality curves)." << std::endl;
    return (best_nmi >= 0.25) ? 0 : 1;
}
