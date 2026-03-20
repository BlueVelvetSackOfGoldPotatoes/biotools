#include "core/data/dataloader.h"
#include "core/io/run_logger.h"
#include "models/cells/src/cell_circuit.h"
#include "models/cells/src/cell_developmental.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

int env_int(const char* key, int fallback) {
    const char* v = std::getenv(key);
    if (!v || !*v) return fallback;
    try { return std::stoi(v); } catch (...) { return fallback; }
}

double env_double(const char* key, double fallback) {
    const char* v = std::getenv(key);
    if (!v || !*v) return fallback;
    try { return std::stod(v); } catch (...) { return fallback; }
}

std::vector<double> row_to_vec(const Tensor& t, std::size_t row) {
    std::vector<double> out(t.cols, 0.0);
    for (std::size_t j = 0; j < t.cols; ++j) out[j] = t(row, j);
    return out;
}

std::vector<double> central_occlusion(const std::vector<double>& x) {
    std::vector<double> out = x;
    for (int y = 9; y < 19; ++y) {
        for (int x0 = 9; x0 < 19; ++x0) {
            out[static_cast<std::size_t>(y * 28 + x0)] = 0.0;
        }
    }
    return out;
}

std::vector<double> brightness_shift(const std::vector<double>& x) {
    std::vector<double> out(28U * 28U, 0.0);
    for (int y = 0; y < 28; ++y) {
        for (int x0 = 0; x0 < 28; ++x0) {
            const int src_x = std::max(0, x0 - 2);
            const double v = x[static_cast<std::size_t>(y * 28 + src_x)];
            out[static_cast<std::size_t>(y * 28 + x0)] = std::min(1.0, 0.08 + 1.10 * v);
        }
    }
    return out;
}

struct EvalSummary {
    double loss = 0.0;
    double acc = 0.0;
    double mean_energy = 0.0;
    double mean_homeostasis = 0.0;
};

} // namespace

int main() {
    std::cout << "=== Developmental Tissue vs CellCircuit (MNIST) ===" << std::endl;

    const int epochs = env_int("CELL_DEV_EPOCHS", 3);
    const int train_limit = env_int("CELL_DEV_TRAIN_SAMPLES", 2000);
    const int test_limit = env_int("CELL_DEV_TEST_SAMPLES", 500);
    const int seed = env_int("CELL_DEV_SEED", 42);
    const int internal_after_epoch = env_int("CELL_DEV_INTERNAL_AFTER_EPOCH", -1);

    MNISTData train = MNISTLoader::load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    MNISTData test = MNISTLoader::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    const int n_train = std::min(train_limit, static_cast<int>(train.images.rows));
    const int n_test = std::min(test_limit, static_cast<int>(test.images.rows));

    cells::DevelopmentalTissueConfig dev_cfg;
    dev_cfg.seed = static_cast<unsigned int>(std::max(0, seed));
    dev_cfg.seed_grid_x = static_cast<std::size_t>(env_int("CELL_DEV_GRID_X", 12));
    dev_cfg.seed_grid_y = static_cast<std::size_t>(env_int("CELL_DEV_GRID_Y", 12));
    dev_cfg.gene_dim = static_cast<std::size_t>(env_int("CELL_DEV_GENE_DIM", 8));
    dev_cfg.free_settle_steps = env_int("CELL_DEV_FREE_SETTLE", 2);
    dev_cfg.taught_settle_steps = env_int("CELL_DEV_TAUGHT_SETTLE", 2);
    dev_cfg.edge_lr = env_double("CELL_DEV_EDGE_LR", 0.010);
    dev_cfg.readout_lr = env_double("CELL_DEV_READOUT_LR", 0.035);
    dev_cfg.contact_radius = env_double("CELL_DEV_CONTACT_RADIUS", 0.18);
    dev_cfg.contact_time_threshold = env_double("CELL_DEV_CONTACT_TIME", 0.05);
    dev_cfg.internal_teacher = env_int("CELL_DEV_INTERNAL_TEACHER", 0) != 0;
    dev_cfg.internal_teacher_blend = env_double("CELL_DEV_INTERNAL_BLEND", 0.5);
    dev_cfg.tissue_params.dt = env_double("CELL_DEV_DT", 0.02);
    dev_cfg.tissue_params.max_cells = dev_cfg.seed_grid_x * dev_cfg.seed_grid_y + 32U;
    dev_cfg.tissue_params.max_speed = env_double("CELL_DEV_MAX_SPEED", 0.18);
    dev_cfg.tissue_params.drag = env_double("CELL_DEV_DRAG", 2.2);

    cells::CellCircuitConfig circuit_cfg;
    circuit_cfg.input_dim = 784;
    circuit_cfg.retina_dim = static_cast<std::size_t>(env_int("CELL_DEV_RETINA", 144));
    circuit_cfg.v1_dim = static_cast<std::size_t>(env_int("CELL_DEV_V1", 128));
    circuit_cfg.assoc_dim = static_cast<std::size_t>(env_int("CELL_DEV_ASSOC", 96));
    circuit_cfg.decision_dim = 10;
    circuit_cfg.lr = env_double("CELL_DEV_CIRCUIT_LR", 0.004);
    circuit_cfg.dt = env_double("CELL_DEV_CIRCUIT_DT", 0.02);
    circuit_cfg.seed = static_cast<unsigned int>(std::max(0, seed));

    cells::DevelopmentalTissueClassifier developmental(dev_cfg);
    cells::CellCircuit circuit(circuit_cfg);

    std::ostringstream params_json;
    params_json << "{" << '"' << "epochs" << '"' << ":" << epochs << ","
                << '"' << "train_limit" << '"' << ":" << n_train << ","
                << '"' << "test_limit" << '"' << ":" << n_test << ","
                << '"' << "seed_grid_x" << '"' << ":" << dev_cfg.seed_grid_x << ","
                << '"' << "seed_grid_y" << '"' << ":" << dev_cfg.seed_grid_y << ","
                << '"' << "free_settle" << '"' << ":" << dev_cfg.free_settle_steps << ","
                << '"' << "taught_settle" << '"' << ":" << dev_cfg.taught_settle_steps << ","
                << '"' << "internal_teacher" << '"' << ":" << (dev_cfg.internal_teacher ? 1 : 0) << ","
                << '"' << "internal_teacher_blend" << '"' << ":" << dev_cfg.internal_teacher_blend << ","
                << '"' << "internal_after_epoch" << '"' << ":" << internal_after_epoch << "}";

    RunLogger logger("cells", "developmental_mnist_compare", seed, "mnist-idx-v1",
                     params_json.str(), "mnist", "MNIST", "classification");

    std::vector<int> order(n_train);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(static_cast<unsigned int>(seed));

    auto eval_model = [&](auto&& input_xform,
                          bool use_developmental,
                          bool damage_dev) -> EvalSummary {
        EvalSummary s;
        int correct = 0;

        if (use_developmental) {
            developmental.reset_dynamics();
            if (damage_dev) developmental.apply_damage(0.25, static_cast<unsigned int>(seed + 9001));
        } else {
            circuit.reset_state();
        }

        for (int i = 0; i < n_test; ++i) {
            std::vector<double> x = input_xform(row_to_vec(test.images, static_cast<std::size_t>(i)));
            const int y = static_cast<int>(test.labels[static_cast<std::size_t>(i)]);
            if (use_developmental) {
                const auto step = developmental.evaluate_one(x, y);
                s.loss += step.loss;
                s.mean_energy += step.mean_energy;
                s.mean_homeostasis += step.mean_homeostasis;
                if (step.pred == y) correct++;
            } else {
                const auto step = circuit.evaluate_one(x, y);
                s.loss += step.loss;
                s.mean_energy += step.mean_energy;
                s.mean_homeostasis += step.mean_homeostasis;
                if (step.pred == y) correct++;
            }
        }
        s.loss /= std::max(1, n_test);
        s.acc = static_cast<double>(correct) / std::max(1, n_test);
        s.mean_energy /= std::max(1, n_test);
        s.mean_homeostasis /= std::max(1, n_test);
        return s;
    };

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        if (internal_after_epoch >= 0 && epoch > internal_after_epoch) {
            developmental.set_internal_teacher(true);
        } else {
            developmental.set_internal_teacher(dev_cfg.internal_teacher);
        }
        std::shuffle(order.begin(), order.end(), rng);
        developmental.reset_dynamics();
        circuit.reset_state();

        double dev_train_loss = 0.0;
        double dev_train_energy = 0.0;
        double dev_train_homeo = 0.0;
        int dev_train_correct = 0;

        double circ_train_loss = 0.0;
        double circ_train_energy = 0.0;
        double circ_train_homeo = 0.0;
        int circ_train_correct = 0;

        for (int i = 0; i < n_train; ++i) {
            const int idx = order[static_cast<std::size_t>(i)];
            std::vector<double> x = row_to_vec(train.images, static_cast<std::size_t>(idx));
            const int y = static_cast<int>(train.labels[static_cast<std::size_t>(idx)]);

            const auto ds = developmental.train_one(x, y);
            dev_train_loss += ds.loss;
            dev_train_energy += ds.mean_energy;
            dev_train_homeo += ds.mean_homeostasis;
            if (ds.pred == y) dev_train_correct++;

            const auto cs = circuit.train_one(x, y);
            circ_train_loss += cs.loss;
            circ_train_energy += cs.mean_energy;
            circ_train_homeo += cs.mean_homeostasis;
            if (cs.pred == y) circ_train_correct++;
        }

        const double dev_train_acc = static_cast<double>(dev_train_correct) / std::max(1, n_train);
        const double circ_train_acc = static_cast<double>(circ_train_correct) / std::max(1, n_train);
        dev_train_loss /= std::max(1, n_train);
        circ_train_loss /= std::max(1, n_train);
        dev_train_energy /= std::max(1, n_train);
        circ_train_energy /= std::max(1, n_train);
        dev_train_homeo /= std::max(1, n_train);
        circ_train_homeo /= std::max(1, n_train);

        const EvalSummary dev_clean = eval_model([](const std::vector<double>& x) { return x; }, true, false);
        const EvalSummary dev_damage = eval_model(central_occlusion, true, false);
        const EvalSummary dev_shift = eval_model(brightness_shift, true, false);
        const EvalSummary dev_internal_damage = eval_model([](const std::vector<double>& x) { return x; }, true, true);

        const EvalSummary circ_clean = eval_model([](const std::vector<double>& x) { return x; }, false, false);
        const EvalSummary circ_damage = eval_model(central_occlusion, false, false);
        const EvalSummary circ_shift = eval_model(brightness_shift, false, false);

        logger.append_csv_row(
            "model_specific/cells/developmental_compare.csv",
            {"run_id","epoch","dev_train_loss","dev_train_acc","dev_test_acc","dev_damage_acc","dev_shift_acc",
             "dev_internal_damage_acc","dev_train_energy","dev_train_homeostasis","dev_test_energy","dev_test_homeostasis",
             "dev_edges","dev_parameters","dev_internal_teacher","circuit_train_loss","circuit_train_acc","circuit_test_acc","circuit_damage_acc",
             "circuit_shift_acc","circuit_train_energy","circuit_train_homeostasis","circuit_test_energy","circuit_test_homeostasis",
             "circuit_parameters"},
            {logger.run_id(), std::to_string(epoch),
             std::to_string(dev_train_loss), std::to_string(dev_train_acc), std::to_string(dev_clean.acc),
             std::to_string(dev_damage.acc), std::to_string(dev_shift.acc), std::to_string(dev_internal_damage.acc),
             std::to_string(dev_train_energy), std::to_string(dev_train_homeo), std::to_string(dev_clean.mean_energy),
             std::to_string(dev_clean.mean_homeostasis), std::to_string(developmental.edge_count()),
             std::to_string(developmental.parameter_count()),
             std::to_string(developmental.internal_teacher_enabled() ? 1 : 0),
             std::to_string(circ_train_loss),
             std::to_string(circ_train_acc), std::to_string(circ_clean.acc), std::to_string(circ_damage.acc),
             std::to_string(circ_shift.acc), std::to_string(circ_train_energy), std::to_string(circ_train_homeo),
             std::to_string(circ_clean.mean_energy), std::to_string(circ_clean.mean_homeostasis),
             std::to_string(circuit.parameter_count())});

        std::cout << "[dev-compare] epoch=" << epoch
                  << " dev_train_acc=" << std::fixed << std::setprecision(4) << dev_train_acc
                  << " dev_test_acc=" << dev_clean.acc
                  << " dev_shift_acc=" << dev_shift.acc
                  << " dev_edges=" << developmental.edge_count()
                  << " dev_internal_teacher=" << (developmental.internal_teacher_enabled() ? 1 : 0)
                  << " circuit_test_acc=" << circ_clean.acc
                  << " circuit_shift_acc=" << circ_shift.acc
                  << std::endl;
    }

    logger.write_manifest_end();
    return 0;
}
