#include "core/data/dataloader.h"
#include "models/cells/src/cell_developmental.h"
#include "models/cells/src/cell_tissue2d.h"

#include <cmath>
#include <filesystem>
#include <iostream>

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const std::string& name) {
    if (cond) {
        std::cout << "  PASS  " << name << std::endl;
        g_pass++;
    } else {
        std::cout << "  FAIL  " << name << std::endl;
        g_fail++;
    }
}

static cells::Genome2D make_genome(unsigned int seed) {
    cells::Genome2D g = cells::Genome2D::random(6, seed);
    g.mutation_std = 0.01;
    g.division_energy = 1.2;
    return g;
}

static std::vector<double> row_to_vec(const Tensor& t, std::size_t row) {
    std::vector<double> out(t.cols, 0.0);
    for (std::size_t j = 0; j < t.cols; ++j) out[j] = t(row, j);
    return out;
}

static cells::DevelopmentalTissueConfig make_dev_cfg(unsigned int seed) {
    cells::DevelopmentalTissueConfig cfg;
    cfg.seed = seed;
    cfg.seed_grid_x = 8;
    cfg.seed_grid_y = 8;
    cfg.gene_dim = 8;
    cfg.free_settle_steps = 1;
    cfg.taught_settle_steps = 1;
    cfg.contact_radius = 0.35;
    cfg.contact_time_threshold = 0.02;
    cfg.max_edges = 1500;
    cfg.tissue_params.dt = 0.02;
    cfg.tissue_params.max_cells = 96;
    cfg.tissue_params.max_speed = 0.14;
    cfg.tissue_params.drag = 2.0;
    return cfg;
}

static void test_leaf_field_sign() {
    cells::LeafField2D leaf;
    const double center_sdf = leaf.signed_distance(cells::Vec2(0.0, 0.0));
    const double far_sdf = leaf.signed_distance(cells::Vec2(2.0, 2.0));
    check(center_sdf <= 0.0, "leaf sdf negative at center");
    check(far_sdf > 0.0, "leaf sdf positive far away");
}

static void test_tissue_determinism() {
    cells::LeafField2D leaf;
    cells::Tissue2DParams p;
    p.max_cells = 900;
    p.substeps = 1;
    p.dt = 0.02;

    const auto g = make_genome(77);
    cells::CellSheet2D a(g, p, 123);
    cells::CellSheet2D b(g, p, 123);
    a.seed_disc(60, 0.06);
    b.seed_disc(60, 0.06);

    for (int i = 0; i < 200; ++i) {
        a.step(leaf);
        b.step(leaf);
    }

    const auto sa = a.compute_stats(&leaf, 96);
    const auto sb = b.compute_stats(&leaf, 96);

    check(sa.alive_cells == sb.alive_cells, "deterministic alive cell count");
    check(std::abs(sa.iou_to_leaf - sb.iou_to_leaf) < 1e-12, "deterministic iou");
    check(std::abs(sa.mean_energy - sb.mean_energy) < 1e-12, "deterministic mean energy");
}

static void test_leaf_growth_sanity() {
    cells::LeafField2D leaf;
    cells::Tissue2DParams p;
    p.max_cells = 1200;
    p.substeps = 1;
    p.dt = 0.02;
    p.division_jitter = 0.008;

    cells::CellSheet2D sheet(make_genome(99), p, 99);
    sheet.seed_disc(80, 0.08);

    for (int i = 0; i < 300; ++i) {
        sheet.step(leaf);
    }
    const auto s1 = sheet.compute_stats(&leaf, 128);

    check(s1.alive_cells > 0, "growth sanity alive cells > 0");
    check(s1.alive_cells <= p.max_cells, "growth sanity respects max cells");
    check(std::isfinite(s1.iou_to_leaf), "growth sanity finite iou");
    check(s1.inside_fraction > 0.15, "growth sanity keeps non-trivial mass inside leaf");
    check(s1.mean_abs_sdf < 1.0, "growth sanity bounded target distance");
}

static void test_developmental_field_stability() {
    cells::DevelopmentalMNISTField field;
    std::vector<double> image(28U * 28U, 0.0);
    image[static_cast<std::size_t>(14 * 28 + 14)] = 1.0;
    field.set_image(image);
    field.set_teacher_label(3, true);
    field.set_novelty(0.5);

    const auto a = field.sample(cells::Vec2(0.10, -0.20));
    const auto b = field.sample(cells::Vec2(0.10, -0.20));

    check(a.morphogen == b.morphogen, "developmental field morphogen stable");
    check(a.teacher_signal == b.teacher_signal, "developmental field teacher stable");
    check(a.guidance.x == b.guidance.x && a.guidance.y == b.guidance.y, "developmental field guidance stable");
}

static void test_developmental_graph_determinism_and_pruning() {
    const auto cfg = make_dev_cfg(55);
    cells::DevelopmentalTissueClassifier a(cfg);
    cells::DevelopmentalTissueClassifier b(cfg);

    std::vector<double> image(28U * 28U, 0.0);
    for (int y = 6; y < 22; ++y) {
        image[static_cast<std::size_t>(y * 28 + 10)] = 1.0;
        image[static_cast<std::size_t>(y * 28 + 18)] = 1.0;
    }

    for (int i = 0; i < 4; ++i) {
        a.train_one(image, 1);
        b.train_one(image, 1);
    }

    check(a.edge_count() > 0, "developmental graph forms edges");
    check(a.edge_count() == b.edge_count(), "developmental graph edge count deterministic");

    const auto& ea = a.edges();
    const auto& eb = b.edges();
    bool same_edges = ea.size() == eb.size();
    for (std::size_t i = 0; same_edges && i < ea.size(); ++i) {
        same_edges =
            ea[i].src_id == eb[i].src_id &&
            ea[i].dst_id == eb[i].dst_id &&
            std::abs(ea[i].weight - eb[i].weight) < 1e-12;
    }
    check(same_edges, "developmental graph edge weights deterministic");

    const std::size_t before = a.edge_count();
    a.apply_damage(1.0, 909);
    check(a.edge_count() < before, "developmental graph damage prunes edges");
}

static void test_internal_teacher_toggle() {
    const auto cfg = make_dev_cfg(88);
    cells::DevelopmentalTissueClassifier model(cfg);
    check(!model.internal_teacher_enabled(), "internal teacher disabled by default");
    model.set_internal_teacher(true);
    check(model.internal_teacher_enabled(), "internal teacher toggle on");
    model.set_internal_teacher(false);
    check(!model.internal_teacher_enabled(), "internal teacher toggle off");
}

static void test_developmental_mnist_smoke() {
    if (!std::filesystem::exists("data/train-images-idx3-ubyte") ||
        !std::filesystem::exists("data/t10k-images-idx3-ubyte")) {
        check(true, "developmental mnist smoke skipped (data missing)");
        return;
    }

    const auto cfg = make_dev_cfg(77);
    cells::DevelopmentalTissueClassifier a(cfg);
    cells::DevelopmentalTissueClassifier b(cfg);

    MNISTData train = MNISTLoader::load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte");
    MNISTData test = MNISTLoader::load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte");

    const int n_train = 120;
    const int n_test = 40;
    for (int i = 0; i < n_train; ++i) {
        const auto x = row_to_vec(train.images, static_cast<std::size_t>(i));
        const int y = static_cast<int>(train.labels[static_cast<std::size_t>(i)]);
        a.train_one(x, y);
        b.train_one(x, y);
    }

    auto eval_acc = [&](cells::DevelopmentalTissueClassifier& model) {
        model.reset_dynamics();
        int correct = 0;
        for (int i = 0; i < n_test; ++i) {
            const auto x = row_to_vec(test.images, static_cast<std::size_t>(i));
            const int y = static_cast<int>(test.labels[static_cast<std::size_t>(i)]);
            const auto step = model.evaluate_one(x, y);
            if (step.pred == y) correct++;
        }
        return static_cast<double>(correct) / static_cast<double>(n_test);
    };

    const double acc_a = eval_acc(a);
    const double acc_b = eval_acc(b);
    check(std::abs(acc_a - acc_b) < 1e-12, "developmental mnist deterministic accuracy");
    check(acc_a > 0.10, "developmental mnist above random floor");
}

int main() {
    std::cout << "=== Leaf Cells Tests ===" << std::endl;

    test_leaf_field_sign();
    test_tissue_determinism();
    test_leaf_growth_sanity();
    test_developmental_field_stability();
    test_developmental_graph_determinism_and_pruning();
    test_internal_teacher_toggle();
    test_developmental_mnist_smoke();

    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===" << std::endl;
    return g_fail == 0 ? 0 : 1;
}
