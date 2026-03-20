#include "core/tensor/tensor.h"
#include "core/data/dataloader.h"
#include "core/metrics/metrics.h"
#include "core/io/run_logger.h"
#include "models/forests/src/forest_models.h"
#include "benchmarks/logging_utils.h"
#include "benchmarks/hpo_utils.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <vector>

template <typename ForestModel>
double run_forest(const std::string& family_name,
                  const std::string& variant,
                  ForestModel& model,
                  const Tensor& train_x,
                  const std::vector<int>& train_y,
                  const Tensor& test_x,
                  const std::vector<int>& test_y,
                  int n_trees,
                  int max_depth,
                  int min_samples_split,
                  int seed) {
    std::ostringstream params_json;
    params_json << "{\"n_trees\":" << n_trees
                << ",\"max_depth\":" << max_depth
                << ",\"min_samples_split\":" << min_samples_split
                << "}";
    RunLogger logger(
        "forests",
        variant,
        seed,
        "mnist-idx-v1",
        params_json.str()
    );
    benchlog::BenchBioHarness bio("forests", variant, seed);
    bio.attach({}, {}, "forest_param");
    bio.begin_epoch(1);

    auto start = std::chrono::high_resolution_clock::now();
    model.fit(train_x, train_y);
    auto preds = model.predict(test_x);
    const double acc = Metrics::accuracy(preds, test_y);
    auto end = std::chrono::high_resolution_clock::now();

    const double elapsed = std::chrono::duration<double>(end - start).count();
    const double elapsed_ms = elapsed * 1000.0;
    const double per_sample_ms = elapsed_ms / static_cast<double>(test_x.rows);

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
    benchlog::log_inference_from_preds(logger, "test", preds, test_y, per_sample_ms, static_cast<int>(test_x.rows));
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

    double oob = std::numeric_limits<double>::quiet_NaN();
    if constexpr (std::is_same_v<ForestModel, RandomForestClassifier>) {
        oob = model.oob_score(train_x, train_y);
    }
    logger.append_csv_row(
        "model_specific/forests/oob_curve.csv",
        {"run_id", "n_trees", "oob_acc"},
        {logger.run_id(), std::to_string(n_trees), std::to_string(oob)}
    );

    const auto& trees = model.get_trees();
    const size_t n_samples = std::min(static_cast<size_t>(300), test_x.rows);
    for (size_t i = 0; i < n_samples; ++i) {
        Tensor one = test_x.slice_rows(i, 1);
        std::vector<int> votes(10, 0);
        for (const auto& t : trees) {
            auto p = t.predict(one);
            votes[p[0]]++;
        }
        const int max_vote = *std::max_element(votes.begin(), votes.end());
        const double agreement = trees.empty() ? 0.0 : static_cast<double>(max_vote) / trees.size();
        logger.append_csv_row(
            "model_specific/forests/tree_agreement.csv",
            {"run_id", "sample", "agreement_ratio"},
            {logger.run_id(), std::to_string(i), std::to_string(agreement)}
        );
    }

    // Feature importance from ensemble
    auto imp = model.feature_importance();
    for (size_t i = 0; i < imp.size(); ++i) {
        if (imp[i] > 0.0) {
            logger.append_csv_row(
                "model_specific/forests/feature_importance.csv",
                {"run_id", "feature", "importance", "row", "col"},
                {logger.run_id(), std::to_string(i), std::to_string(imp[i]),
                 std::to_string(i / 28), std::to_string(i % 28)}
            );
        }
    }
    bio.end_epoch(logger, 1);
    logger.write_manifest_end();
    std::cout << family_name << " Test Accuracy: " << std::fixed << std::setprecision(2) << acc * 100 << "%"
              << " (" << std::setprecision(1) << elapsed << "s)"
              << " | " << (acc >= 0.90 ? "PASS" : "FAIL") << std::endl;
    return acc;
}

int main() {
    std::cout << "=== Random Forest Benchmark (MNIST) ===" << std::endl;
    const int seed = hpo::env_int("FORESTS_SEED", 42);
    std::mt19937 rng(static_cast<unsigned int>(seed));
    const int n_trees = hpo::env_int("FORESTS_N_TREES", 50);
    const int max_depth = hpo::env_int("FORESTS_MAX_DEPTH", 15);
    const int min_samples_split = hpo::env_int("FORESTS_MIN_SAMPLES_SPLIT", 5);
    const double rf_sample_ratio = hpo::env_double("FORESTS_RF_SAMPLE_RATIO", 0.8);
    const std::string variant_filter = hpo::to_lower(hpo::env_string("FORESTS_VARIANT", "all"));

    auto train = MNISTLoader::load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    auto test = MNISTLoader::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    std::vector<int> train_labels(train.labels.begin(), train.labels.end());
    std::vector<int> test_labels(test.labels.begin(), test.labels.end());

    size_t train_n = std::min(static_cast<size_t>(20000), train.images.rows);
    Tensor train_x = train.images.slice_rows(0, train_n);
    std::vector<int> train_y(train_labels.begin(), train_labels.begin() + static_cast<long>(train_n));
    Tensor test_x = test.images;
    std::vector<int> test_y = test_labels;

    double min_acc = 1.0;
    bool ran_any = false;
    if (variant_filter == "all" || variant_filter == "rf" || variant_filter == "random_forest") {
        std::cout << "\n--- RandomForest (50 trees, depth=15) ---" << std::endl;
        RandomForestClassifier rf(n_trees, max_depth, min_samples_split, 10, rng, rf_sample_ratio);
        ran_any = true;
        min_acc = std::min(
            min_acc,
            run_forest(
                "RandomForest",
                "random_forest_" + std::to_string(n_trees),
                rf,
                train_x,
                train_y,
                test_x,
                test_y,
                n_trees,
                max_depth,
                min_samples_split,
                seed
            )
        );
    }

    if (variant_filter == "all" || variant_filter == "et" || variant_filter == "extra_trees") {
        std::cout << "\n--- ExtraTrees (50 trees, depth=15) ---" << std::endl;
        ExtraTreesClassifier et(n_trees, max_depth, min_samples_split, 10, rng);
        ran_any = true;
        min_acc = std::min(
            min_acc,
            run_forest(
                "ExtraTrees",
                "extra_trees_" + std::to_string(n_trees),
                et,
                train_x,
                train_y,
                test_x,
                test_y,
                n_trees,
                max_depth,
                min_samples_split,
                seed
            )
        );
    }

    if (!ran_any) {
        std::cerr << "No forest variant selected for FORESTS_VARIANT='" << variant_filter
                  << "'. Valid values: all,rf,et" << std::endl;
        return 2;
    }
    std::cout << "\n=== Forest Benchmark Complete ===" << std::endl;
    return (min_acc >= 0.90) ? 0 : 1;
}
