#include "core/tensor/tensor.h"
#include "core/data/dataloader.h"
#include "core/metrics/metrics.h"
#include "core/io/run_logger.h"
#include "models/trees/src/tree_models.h"
#include "benchmarks/logging_utils.h"
#include "benchmarks/hpo_utils.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stack>
#include <vector>

static std::map<int, int> depth_histogram(const DecisionTreeClassifier& tree) {
    std::map<int, int> hist;
    if (tree.nodes.empty()) return hist;
    std::stack<std::pair<int, int>> st;
    st.push({0, 0});
    while (!st.empty()) {
        auto [idx, depth] = st.top();
        st.pop();
        hist[depth]++;
        const auto& node = tree.nodes[static_cast<size_t>(idx)];
        if (!node.is_leaf) {
            if (node.left >= 0) st.push({node.left, depth + 1});
            if (node.right >= 0) st.push({node.right, depth + 1});
        }
    }
    return hist;
}

int main() {
    std::cout << "=== Decision Tree Benchmark (MNIST) ===" << std::endl;
    const int seed = hpo::env_int("TREE_SEED", 42);
    std::mt19937 rng(static_cast<unsigned int>(seed));
    const int min_samples_split = hpo::env_int("TREE_MIN_SAMPLES_SPLIT", 5);
    std::vector<int> depths{20, 30, 50};
    const std::string depth_csv = hpo::env_string("TREE_DEPTHS", "20,30,50");
    std::vector<int> parsed_depths;
    std::string cur;
    for (char ch : depth_csv) {
        if (ch == ',') {
            if (!cur.empty()) parsed_depths.push_back(std::max(1, std::stoi(cur)));
            cur.clear();
        } else if (!std::isspace(static_cast<unsigned char>(ch))) {
            cur.push_back(ch);
        }
    }
    if (!cur.empty()) parsed_depths.push_back(std::max(1, std::stoi(cur)));
    if (!parsed_depths.empty()) depths = parsed_depths;

    auto train = MNISTLoader::load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    auto test = MNISTLoader::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    std::vector<int> train_labels(train.labels.begin(), train.labels.end());
    std::vector<int> test_labels(test.labels.begin(), test.labels.end());

    Tensor train_x = train.images;
    std::vector<int> train_y = train_labels;
    Tensor test_x = test.images;
    std::vector<int> test_y = test_labels;

    double best_acc = 0.0;
    for (int depth : depths) {
        std::cout << "\n--- DecisionTree (max_depth=" << depth << ") ---" << std::endl;
        std::ostringstream params_json;
        params_json << "{\"max_depth\":" << depth
                    << ",\"min_samples_split\":" << min_samples_split << "}";
        RunLogger logger(
            "trees",
            "decision_tree_depth_" + std::to_string(depth),
            seed,
            "mnist-idx-v1",
            params_json.str()
        );
        benchlog::BenchBioHarness bio("trees", "decision_tree_depth_" + std::to_string(depth), seed);
        bio.attach({}, {}, "tree_param");
        bio.begin_epoch(1);

        auto start = std::chrono::high_resolution_clock::now();
        DecisionTreeClassifier tree(depth, min_samples_split, 10, rng);
        tree.fit(train_x, train_y);
        auto preds = tree.predict(test_x);
        const double acc = Metrics::accuracy(preds, test_y);
        best_acc = std::max(best_acc, acc);
        auto end = std::chrono::high_resolution_clock::now();
        const double elapsed = std::chrono::duration<double>(end - start).count();
        const double elapsed_ms = elapsed * 1000.0;

        auto [precision_macro, recall_macro, f1_macro] = benchlog::macro_prf_from_preds(preds, test_y, 10);
        logger.log_epoch_metric(
            1,
            "test",
            std::numeric_limits<double>::quiet_NaN(),
            acc,
            precision_macro,
            recall_macro,
            f1_macro,
            0.0,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            elapsed_ms,
            test_x.rows / (elapsed + 1e-12)
        );
        benchlog::log_confusion_and_class_metrics(logger, 1, "test", preds, test_y, 10);

        const double per_sample_ms = elapsed_ms / static_cast<double>(test_x.rows);
        benchlog::log_inference_from_preds(
            logger, "test", preds, test_y, per_sample_ms, static_cast<int>(test_x.rows)
        );
        benchlog::log_calibration_from_preds(logger, "test", preds, test_y, 10);
        logger.log_system_metric(
            RunLogger::utc_now_iso8601(),
            test_x.rows / (elapsed + 1e-12),
            per_sample_ms,
            per_sample_ms,
            per_sample_ms,
            0.0,
            0.0
        );

        auto depth_hist = depth_histogram(tree);
        for (const auto& [d, n_nodes] : depth_hist) {
            logger.append_csv_row(
                "model_specific/trees/tree_growth.csv",
                {"run_id", "depth", "nodes"},
                {logger.run_id(), std::to_string(d), std::to_string(n_nodes)}
            );
        }

        auto imp = tree.feature_importance();
        for (size_t i = 0; i < imp.size(); ++i) {
            logger.append_csv_row(
                "model_specific/trees/split_importance.csv",
                {"run_id", "feature", "gain_total"},
                {logger.run_id(), std::to_string(i), std::to_string(imp[i])}
            );
        }
        // Pixel importance as 28x28 spatial map
        for (size_t i = 0; i < std::min(imp.size(), static_cast<size_t>(784)); ++i) {
            if (imp[i] > 0.0) {
                logger.append_csv_row(
                    "model_specific/trees/pixel_importance_map.csv",
                    {"run_id", "feature", "row", "col", "importance"},
                    {logger.run_id(), std::to_string(i),
                     std::to_string(i / 28), std::to_string(i % 28),
                     std::to_string(imp[i])}
                );
            }
        }
        bio.end_epoch(logger, 1);

        std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) << acc * 100 << "%"
                  << " (" << std::setprecision(1) << elapsed << "s)"
                  << " | " << (acc >= 0.80 ? "PASS" : "FAIL") << std::endl;
        std::vector<size_t> idx(imp.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(
            idx.begin(), idx.begin() + 5, idx.end(), [&](size_t a, size_t b) { return imp[a] > imp[b]; }
        );
        std::cout << "Top-5 features: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << idx[i] << "(" << std::setprecision(4) << imp[idx[i]] << ") ";
        }
        std::cout << std::endl;
        logger.write_manifest_end();
    }

    std::cout << "\n=== Decision Tree Benchmark Complete ===" << std::endl;
    return (best_acc >= 0.80) ? 0 : 1;
}
