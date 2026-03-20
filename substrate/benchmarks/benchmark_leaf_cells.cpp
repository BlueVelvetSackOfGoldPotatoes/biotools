#include "models/cells/src/cell_tissue2d.h"
#include "models/cells/src/core/log.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
    if (v == nullptr || *v == '\0') return fallback;
    try { return std::stoi(v); } catch (...) { return fallback; }
}

double env_double(const char* key, double fallback) {
    const char* v = std::getenv(key);
    if (v == nullptr || *v == '\0') return fallback;
    try { return std::stod(v); } catch (...) { return fallback; }
}

std::string fmt(double x) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << x;
    return oss.str();
}

struct EvalResult {
    double fitness = -1e9;
    cells::Tissue2DStats stats;
};

EvalResult eval_genome(const cells::Genome2D& genome,
                       const cells::Tissue2DParams& params,
                       const cells::LeafField2D& leaf,
                       int steps,
                       int seed_cells,
                       int init_cells) {
    cells::CellSheet2D sheet(genome, params, static_cast<unsigned int>(seed_cells));
    sheet.seed_disc(static_cast<std::size_t>(std::max(1, init_cells)), 0.06);
    sheet.run(leaf, static_cast<std::size_t>(std::max(1, steps)));

    EvalResult r;
    r.stats = sheet.compute_stats(&leaf, 128);
    const double iou = r.stats.iou_to_leaf;
    const double inside = r.stats.inside_fraction;
    const double sdf = r.stats.mean_abs_sdf;
    const double cell_count_penalty = 0.0004 * std::abs(static_cast<double>(r.stats.alive_cells) - 900.0);
    r.fitness = 2.0 * iou + 1.0 * inside - 0.35 * sdf - cell_count_penalty;
    return r;
}

void write_leaf_outline_csv(const cells::LeafField2D& leaf, const std::string& path, int samples) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return;
    out << "x,upper_y,lower_y\n";
    const auto cfg = leaf.config();
    const int n = std::max(64, samples);
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n - 1);
        const double xn = -1.0 + 2.0 * t;
        const double x = cfg.center.x + xn * cfg.half_length;
        const double hw = leaf.half_width_for_normalized_x(xn);
        out << x << "," << (cfg.center.y + hw) << "," << (cfg.center.y - hw) << "\n";
    }
}

} // namespace

int main() {
    const int seed = env_int("LEAF_SEED", 123);
    const int gene_dim = env_int("LEAF_GENE_DIM", 8);
    const int population = env_int("LEAF_POPULATION", 12);
    const int generations = env_int("LEAF_GENERATIONS", 10);
    const int eval_steps = env_int("LEAF_EVAL_STEPS", 500);
    const int final_steps = env_int("LEAF_FINAL_STEPS", 3000);
    const int init_cells = env_int("LEAF_INIT_CELLS", 100);
    const int log_every = std::max(1, env_int("LEAF_LOG_EVERY", 10));

    cells::LeafField2D::Config leaf_cfg;
    leaf_cfg.half_length = env_double("LEAF_HALF_LENGTH", leaf_cfg.half_length);
    leaf_cfg.max_half_width = env_double("LEAF_HALF_WIDTH", leaf_cfg.max_half_width);
    leaf_cfg.taper_power = env_double("LEAF_TAPER_POWER", leaf_cfg.taper_power);
    leaf_cfg.serration = env_double("LEAF_SERRATION", leaf_cfg.serration);
    leaf_cfg.serration_freq = env_double("LEAF_SERRATION_FREQ", leaf_cfg.serration_freq);
    cells::LeafField2D leaf(leaf_cfg);

    cells::Tissue2DParams params;
    params.dt = env_double("LEAF_DT", params.dt);
    params.max_cells = static_cast<std::size_t>(std::max(100, env_int("LEAF_MAX_CELLS", static_cast<int>(params.max_cells))));
    params.substeps = std::max(1, env_int("LEAF_SUBSTEPS", params.substeps));
    params.repulsion_k = env_double("LEAF_REPULSION", params.repulsion_k);
    params.adhesion_k = env_double("LEAF_ADHESION", params.adhesion_k);
    params.energy_gain = env_double("LEAF_ENERGY_GAIN", params.energy_gain);
    params.metabolic_cost = env_double("LEAF_METABOLIC_COST", params.metabolic_cost);
    params.division_jitter = env_double("LEAF_DIVISION_JITTER", params.division_jitter);

    std::filesystem::create_directories("output/cells");
    cells::CsvLogger evo_log;
    const std::string evo_path = "output/cells/leaf_evolution.csv";
    if (!evo_log.open(evo_path, "generation,best_fitness,mean_fitness,best_iou,best_inside,best_abs_sdf,best_cells")) {
        std::cerr << "failed to open evolution log: " << evo_path << std::endl;
        return 1;
    }

    std::mt19937 rng(static_cast<unsigned int>(seed));
    std::vector<cells::Genome2D> pop;
    pop.reserve(static_cast<std::size_t>(std::max(2, population)));
    for (int i = 0; i < std::max(2, population); ++i) {
        pop.push_back(cells::Genome2D::random(static_cast<std::size_t>(std::max(2, gene_dim)),
                                              static_cast<unsigned int>(seed + i * 37 + 1)));
    }

    cells::Genome2D best_genome = pop.front();
    EvalResult best_global;

    std::cout << "=== Leaf Morphogenesis (Genome + 2D Cell Sheet) ===" << std::endl;
    std::cout << "population=" << pop.size()
              << " generations=" << generations
              << " eval_steps=" << eval_steps
              << " final_steps=" << final_steps
              << std::endl;

    for (int gen = 0; gen < std::max(1, generations); ++gen) {
        struct Ranked {
            int idx = -1;
            EvalResult eval;
        };
        std::vector<Ranked> ranked;
        ranked.reserve(pop.size());

        double fitness_sum = 0.0;
        for (std::size_t i = 0; i < pop.size(); ++i) {
            EvalResult ev = eval_genome(pop[i], params, leaf, eval_steps, seed + static_cast<int>(i) * 101 + gen * 7919, init_cells);
            ranked.push_back(Ranked{static_cast<int>(i), ev});
            fitness_sum += ev.fitness;
            if (ev.fitness > best_global.fitness) {
                best_global = ev;
                best_genome = pop[i];
            }
        }

        std::sort(ranked.begin(), ranked.end(), [](const Ranked& a, const Ranked& b) {
            return a.eval.fitness > b.eval.fitness;
        });

        const Ranked& best = ranked.front();
        const double mean_fit = fitness_sum / static_cast<double>(ranked.size());
        evo_log.write_row(
            std::to_string(gen) + "," +
            fmt(best.eval.fitness) + "," +
            fmt(mean_fit) + "," +
            fmt(best.eval.stats.iou_to_leaf) + "," +
            fmt(best.eval.stats.inside_fraction) + "," +
            fmt(best.eval.stats.mean_abs_sdf) + "," +
            std::to_string(best.eval.stats.alive_cells));

        std::cout << "[leaf] gen=" << gen
                  << " best_fit=" << fmt(best.eval.fitness)
                  << " mean_fit=" << fmt(mean_fit)
                  << " iou=" << fmt(best.eval.stats.iou_to_leaf)
                  << " inside=" << fmt(best.eval.stats.inside_fraction)
                  << " cells=" << best.eval.stats.alive_cells
                  << std::endl;

        if (gen == generations - 1) {
            break;
        }

        // Elite retention + mutation offspring.
        const int elite_n = std::max(2, static_cast<int>(std::ceil(0.2 * pop.size())));
        std::vector<cells::Genome2D> next;
        next.reserve(pop.size());
        for (int e = 0; e < elite_n; ++e) {
            next.push_back(pop[static_cast<std::size_t>(ranked[static_cast<std::size_t>(e)].idx)]);
        }

        std::uniform_int_distribution<int> parent_pick(0, std::max(0, elite_n - 1));
        for (std::size_t i = next.size(); i < pop.size(); ++i) {
            const cells::Genome2D& parent = next[static_cast<std::size_t>(parent_pick(rng))];
            cells::Genome2D child = parent;
            const double scale = (i % 2 == 0) ? 1.0 : 0.5;
            child.mutate(rng, scale);
            next.push_back(child);
        }
        pop.swap(next);
    }

    cells::CellSheet2D final_sheet(best_genome, params, static_cast<unsigned int>(seed + 4242));
    final_sheet.seed_disc(static_cast<std::size_t>(std::max(1, init_cells)), 0.06);

    const std::string dyn_path = "output/cells/leaf_dynamics.csv";
    cells::CsvLogger dyn_log;
    if (!dyn_log.open(dyn_path, "step,alive_cells,divisions,deaths,mean_energy,mean_abs_sdf,inside_fraction,iou,centroid_x,centroid_y")) {
        std::cerr << "failed to open dynamics log: " << dyn_path << std::endl;
        return 1;
    }

    for (int step = 0; step < std::max(1, final_steps); ++step) {
        final_sheet.step(leaf);
        if ((step % log_every) == 0 || step == final_steps - 1) {
            const auto st = final_sheet.compute_stats(&leaf, 180);
            dyn_log.write_row(
                std::to_string(step) + "," +
                std::to_string(st.alive_cells) + "," +
                std::to_string(st.divisions) + "," +
                std::to_string(st.deaths) + "," +
                fmt(st.mean_energy) + "," +
                fmt(st.mean_abs_sdf) + "," +
                fmt(st.inside_fraction) + "," +
                fmt(st.iou_to_leaf) + "," +
                fmt(st.centroid.x) + "," +
                fmt(st.centroid.y));
        }
    }

    const auto final_stats = final_sheet.compute_stats(&leaf, 220);
    final_sheet.export_cells_csv("output/cells/leaf_best_cells.csv");
    final_sheet.export_genome_csv("output/cells/leaf_best_genome.csv");
    write_leaf_outline_csv(leaf, "output/cells/leaf_target_outline.csv", 320);
    evo_log.flush();
    dyn_log.flush();

    std::cout << "[leaf] done"
              << " best_global_fit=" << fmt(best_global.fitness)
              << " final_iou=" << fmt(final_stats.iou_to_leaf)
              << " final_inside=" << fmt(final_stats.inside_fraction)
              << " final_cells=" << final_stats.alive_cells
              << "\n[leaf] outputs:"
              << " evolution=" << evo_path
              << " dynamics=" << dyn_path
              << " cells=output/cells/leaf_best_cells.csv"
              << " genome=output/cells/leaf_best_genome.csv"
              << " target=output/cells/leaf_target_outline.csv"
              << std::endl;

    return 0;
}
