#include "models/cells/cellengine/cellengine.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split_csv_row(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) parts.push_back(item);
    return parts;
}

std::size_t count_csv_rows(const std::filesystem::path& path) {
    std::ifstream in(path);
    std::string line;
    if (!std::getline(in, line)) return 0;
    std::size_t rows = 0;
    while (std::getline(in, line)) {
        if (!line.empty()) rows += 1;
    }
    return rows;
}

int count_csv_unique_z(const std::filesystem::path& path) {
    std::ifstream in(path);
    std::string header;
    if (!std::getline(in, header)) return 0;
    const auto columns = split_csv_row(header);
    int z_index = -1;
    for (std::size_t i = 0; i < columns.size(); ++i) {
        if (columns[i] == "z") {
            z_index = static_cast<int>(i);
            break;
        }
    }
    if (z_index < 0) return 0;
    std::set<int> zs;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto row = split_csv_row(line);
        if (static_cast<int>(row.size()) <= z_index) continue;
        zs.insert(std::stoi(row[static_cast<std::size_t>(z_index)]));
    }
    return static_cast<int>(zs.size());
}

} // namespace

int main() {
    setenv("CELLENGINE_USE_CUDA", "0", 1);
    setenv("TENSOR_USE_CUDA", "0", 1);

    cells::cellengine::CellEngineConfig cfg;
    cfg.seed = 7;
    cfg.task_name = "cartpole_balance";
    cfg.body_mode = "fixed2d";
    cfg.rl_algorithm = "none";
    cfg.mode = "cell_only";
    cfg.population_size = 6;
    cfg.generations = 3;
    cfg.elitism = 2;
    cfg.tournament_size = 3;
    cfg.search_budget.num_trials = 1;
    cfg.search_budget.max_ticks = 8;
    cfg.final_budget.num_trials = 1;
    cfg.final_budget.max_ticks = 8;
    cfg.damage_budget.num_trials = 1;
    cfg.damage_budget.max_ticks = 8;
    cfg.auto_attempts = 1;
    cfg.rl_train_episodes = 8;
    cfg.rl_damage_finetune_episodes = 4;
    cfg.rl_eval_every = 4;
    cfg.solve_ratio = 0.02;
    cfg.hint_genomes = 1;

    const std::filesystem::path tmp_dir = "output/test_cellengine_replay";
    std::filesystem::create_directories(tmp_dir);
    std::array<double, cells::cellengine::kGenomeSize> base_genes{};
    base_genes[0] = 0.12;
    base_genes[1] = -0.08;
    base_genes[5] = 0.22;
    base_genes[6] = 0.18;
    base_genes[80] = 0.42;
    base_genes[81] = 0.34;
    base_genes[82] = 0.75;
    base_genes[83] = 0.48;
    const std::filesystem::path genome_csv = tmp_dir / "genome.csv";
    {
        std::ofstream out(genome_csv);
        out << "gene,value\n";
        for (std::size_t i = 0; i < base_genes.size(); ++i) {
            out << i << ',' << base_genes[i] << '\n';
        }
    }
    const std::filesystem::path growth_genome_csv = tmp_dir / "growth_genome.csv";
    {
        auto growth_genes = base_genes;
        growth_genes[80] = 0.9;
        growth_genes[81] = 0.6;
        growth_genes[82] = 1.0;
        growth_genes[83] = 1.0;
        growth_genes[84] = -0.7;
        growth_genes[85] = 0.8;
        growth_genes[86] = 0.7;
        std::ofstream out(growth_genome_csv);
        out << "gene,value,param_name\n";
        for (std::size_t i = 0; i < growth_genes.size(); ++i) {
            out << i << ',' << growth_genes[i] << ",\n";
        }
        out << "-1,2,body_mode_code\n";
        out << "-2,8,development_steps\n";
        out << "-3,2,development_seed_half_width\n";
        out << "-4,72,max_cells\n";
        out << "-5,5,body_extent_x\n";
        out << "-6,12,body_extent_y\n";
        out << "-7,2,body_extent_z\n";
        out << "-8,2,chemical_diffusion_steps\n";
        out << "-9,0.54,development_growth_threshold\n";
        out << "-10,0.32,chemical_diffusion_rate\n";
        out << "-11,0.08,chemical_decay\n";
    }

    cells::cellengine::CellEngineConfig replay_cfg = cfg;
    replay_cfg.mode = "replay";
    replay_cfg.output_dir = tmp_dir.string();
    replay_cfg.load_genome_csv = genome_csv.string();
    replay_cfg.rl_algorithm = "none";
    replay_cfg.final_budget.num_trials = 1;
    replay_cfg.final_budget.max_ticks = 8;
    replay_cfg.replay_sleep_ms = 0;
    replay_cfg.replay_frame_stride = 1000;
    replay_cfg.replay_clear_screen = false;
    replay_cfg.replay_emit_stdout = false;
    const auto replay = cells::cellengine::run_full_benchmark(replay_cfg);
    if (replay.cell_clean.max_ticks != 8.0) {
        throw std::runtime_error("replay mode did not honor max tick budget");
    }
    if (!std::filesystem::exists(tmp_dir / "replay_trace.csv")) {
        throw std::runtime_error("replay mode did not write trace csv");
    }
    if (!std::filesystem::exists(tmp_dir / "replay_body.csv")) {
        throw std::runtime_error("replay mode did not write body csv");
    }
    if (count_csv_rows(tmp_dir / "replay_body.csv") != static_cast<std::size_t>(replay.champion_cell_count)) {
        throw std::runtime_error("replay body csv row count does not match champion_cell_count");
    }
    if (count_csv_unique_z(tmp_dir / "replay_body.csv") != replay.champion_body_depth) {
        throw std::runtime_error("replay body csv depth does not match champion_body_depth");
    }
    if (!std::filesystem::exists(tmp_dir / "replay_cells.csv")) {
        throw std::runtime_error("replay mode did not write cell state csv");
    }
    if (!std::filesystem::exists(tmp_dir / "replay_edges.csv")) {
        throw std::runtime_error("replay mode did not write edge csv");
    }
    if (!std::filesystem::exists(tmp_dir / "decoded_genome.json")) {
        throw std::runtime_error("replay mode did not write decoded genome json");
    }
    if (!std::filesystem::exists(tmp_dir / "trial_context.json")) {
        throw std::runtime_error("replay mode did not write trial context json");
    }
    if (!std::filesystem::exists(tmp_dir / "trial_pre_reset_cells.csv") ||
        !std::filesystem::exists(tmp_dir / "trial_post_reset_cells.csv")) {
        throw std::runtime_error("replay mode did not write trial context snapshots");
    }
    if (!cells::cellengine::write_report(replay, replay_cfg, tmp_dir.string())) {
        throw std::runtime_error("replay mode report write failed");
    }
    {
        std::ifstream trace_in(tmp_dir / "replay_trace.csv");
        const std::string header = [] (std::ifstream& in) {
            std::string line;
            std::getline(in, line);
            return line;
        }(trace_in);
        if (header.find("x_ddot") == std::string::npos ||
            header.find("theta_ddot") == std::string::npos ||
            header.find("task_aux_a") == std::string::npos ||
            header.find("task_primary") == std::string::npos ||
            header.find("damage_event") == std::string::npos) {
            throw std::runtime_error("replay trace header missing enriched telemetry columns");
        }
    }
    {
        std::ifstream body_in(tmp_dir / "replay_body.csv");
        std::string header;
        std::getline(body_in, header);
        if (header.find("z") == std::string::npos ||
            header.find("birth_step") == std::string::npos ||
            header.find("hinge") == std::string::npos ||
            header.find("motor") == std::string::npos ||
            header.find("feature_neighbor_density") == std::string::npos ||
            header.find("gene_expr_0") == std::string::npos) {
            throw std::runtime_error("replay body header missing structural columns");
        }
    }
    {
        std::ifstream cells_in(tmp_dir / "replay_cells.csv");
        std::string header;
        std::getline(cells_in, header);
        if (header.find("cell_id") == std::string::npos ||
            header.find("energy") == std::string::npos ||
            header.find("chem_out") == std::string::npos ||
            header.find("I_elec") == std::string::npos ||
            header.find("I_chem") == std::string::npos ||
            header.find("I_mech") == std::string::npos ||
            header.find("I_self_chem") == std::string::npos ||
            header.find("I_total") == std::string::npos ||
            header.find("stress_hinge") == std::string::npos ||
            header.find("stress_lateral") == std::string::npos ||
            header.find("stress_gravity") == std::string::npos ||
            header.find("stress_drift") == std::string::npos ||
            header.find("stress_neighbor") == std::string::npos ||
            header.find("stress_final") == std::string::npos ||
            header.find("mech_a_gate") == std::string::npos ||
            header.find("mech_contract") == std::string::npos ||
            header.find("mech_prefactor") == std::string::npos ||
            header.find("mech_force_contrib") == std::string::npos) {
            throw std::runtime_error("replay cells header missing per-cell state columns");
        }
    }
    {
        std::ifstream edges_in(tmp_dir / "replay_edges.csv");
        std::string header;
        std::getline(edges_in, header);
        if (header.find("src_cell_id") == std::string::npos ||
            header.find("gap_mean") == std::string::npos ||
            header.find("corr_forward") == std::string::npos ||
            header.find("hebb_delta_forward") == std::string::npos ||
            header.find("decay_delta_forward") == std::string::npos ||
            header.find("old_gap_forward") == std::string::npos ||
            header.find("new_gap_forward") == std::string::npos) {
            throw std::runtime_error("replay edges header missing connectivity columns");
        }
    }
    {
        std::ifstream summary_in(tmp_dir / "summary.json");
        const std::string summary_text((std::istreambuf_iterator<char>(summary_in)),
                                       std::istreambuf_iterator<char>());
        if (summary_text.find("NaN") != std::string::npos || summary_text.find("nan") != std::string::npos) {
            throw std::runtime_error("summary.json must not contain NaN literals");
        }
    }

    cells::cellengine::CellEngineConfig preview_cfg = cfg;
    preview_cfg.mode = "preview";
    preview_cfg.output_dir = (tmp_dir / "preview").string();
    preview_cfg.load_genome_csv = genome_csv.string();
    const auto preview = cells::cellengine::run_full_benchmark(preview_cfg);
    if (!preview.solved) {
        throw std::runtime_error("preview mode should report success");
    }
    if (!std::filesystem::exists(tmp_dir / "preview" / "preview_body.csv")) {
        throw std::runtime_error("preview mode did not write preview_body.csv");
    }
    if (count_csv_rows(tmp_dir / "preview" / "preview_body.csv") != static_cast<std::size_t>(preview.champion_cell_count)) {
        throw std::runtime_error("preview body csv row count does not match champion_cell_count");
    }
    if (!std::filesystem::exists(tmp_dir / "preview" / "decoded_genome.json")) {
        throw std::runtime_error("preview mode did not write decoded_genome.json");
    }

    cells::cellengine::CellEngineConfig forced_3d_cfg = cfg;
    forced_3d_cfg.mode = "preview";
    forced_3d_cfg.output_dir = (tmp_dir / "forced_3d_preview").string();
    forced_3d_cfg.load_genome_csv = genome_csv.string();
    forced_3d_cfg.body_mode = "grown3d";
    forced_3d_cfg.body_extent_z = 0;
    const auto forced_3d_preview = cells::cellengine::run_full_benchmark(forced_3d_cfg);
    if (forced_3d_preview.body_mode != "grown3d" || forced_3d_preview.champion_body_depth <= 1) {
        throw std::runtime_error("fixed grown3d mode must yield a real multi-layer body");
    }
    if (count_csv_unique_z(tmp_dir / "forced_3d_preview" / "preview_body.csv") != forced_3d_preview.champion_body_depth) {
        throw std::runtime_error("forced 3d preview body depth does not match preview_body.csv");
    }

    const std::filesystem::path saved_report_dir = tmp_dir / "saved_full";
    if (!cells::cellengine::write_report(replay, replay_cfg, saved_report_dir.string())) {
        throw std::runtime_error("full report write failed");
    }
    if (!std::filesystem::exists(saved_report_dir / "clean_teacher_trial_context.json") ||
        !std::filesystem::exists(saved_report_dir / "clean_teacher_trial_pre_reset_cells.csv") ||
        !std::filesystem::exists(saved_report_dir / "clean_teacher_trial_post_reset_cells.csv")) {
        throw std::runtime_error("full report write did not emit clean teacher trial context artifacts");
    }
    if (!std::filesystem::exists(saved_report_dir / "clean_autonomous_trial_context.json") ||
        !std::filesystem::exists(saved_report_dir / "damaged_autonomous_trial_context.json")) {
        throw std::runtime_error("full report write did not emit autonomous trial context artifacts");
    }
    {
        std::ifstream champion_in(saved_report_dir / "champion_genome.csv");
        std::string header;
        std::getline(champion_in, header);
        if (header != "gene,value,param_name") {
            throw std::runtime_error("champion genome csv did not upgrade to param-aware format");
        }
        const std::string champion_text((std::istreambuf_iterator<char>(champion_in)),
                                        std::istreambuf_iterator<char>());
        if (champion_text.find("max_cells") == std::string::npos ||
            champion_text.find("body_mode_code") == std::string::npos) {
            throw std::runtime_error("champion genome csv did not persist evolved candidate params");
        }
    }

    cells::cellengine::CellEngineConfig grown_cfg = cfg;
    grown_cfg.body_mode = "fixed2d";
    grown_cfg.max_cells = 29;
    grown_cfg.body_extent_x = 3;
    grown_cfg.body_extent_y = 8;
    grown_cfg.body_extent_z = 0;
    grown_cfg.mode = "preview";
    grown_cfg.output_dir = (tmp_dir / "grown_preview").string();
    grown_cfg.load_genome_csv = growth_genome_csv.string();
    const auto grown_preview = cells::cellengine::run_full_benchmark(grown_cfg);
    if (!grown_preview.solved) {
        throw std::runtime_error("grown preview mode should report success");
    }
    if (grown_preview.champion_cell_count <= 29 && grown_preview.champion_body_depth <= 1) {
        throw std::runtime_error("saved candidate params did not override the fixed preview config");
    }

    cells::cellengine::CellEngineConfig task_cfg = cfg;
    task_cfg.body_mode = "grown2d";
    task_cfg.task_name = "mass_spring_balance";
    task_cfg.rl_algorithm = "none";
    task_cfg.mode = "cell_only";
    task_cfg.load_genome_csv = genome_csv.string();
    task_cfg.final_budget.num_trials = 1;
    task_cfg.final_budget.max_ticks = 8;
    task_cfg.output_dir = (tmp_dir / "massspring").string();
    const auto task_summary = cells::cellengine::run_full_benchmark(task_cfg);
    if (task_summary.task_name != "mass_spring_balance") {
        throw std::runtime_error("task-general interface did not preserve the requested task name");
    }
    if (task_summary.champion_cell_count <= 0) {
        throw std::runtime_error("task-general interface did not build a body");
    }

    cells::cellengine::CellEngineConfig worm_cfg = cfg;
    worm_cfg.task_name = "worm_drag_race";
    worm_cfg.body_mode = "grown2d";
    worm_cfg.rl_algorithm = "none";
    worm_cfg.mode = "cell_only";
    worm_cfg.load_genome_csv = growth_genome_csv.string();
    worm_cfg.final_budget.num_trials = 1;
    worm_cfg.final_budget.max_ticks = 8;
    worm_cfg.output_dir = (tmp_dir / "wormdrag").string();
    const auto worm_summary = cells::cellengine::run_full_benchmark(worm_cfg);
    if (worm_summary.task_name != "worm_drag_race" ||
        worm_summary.cell_clean.task_primary_label != "goal_progress") {
        throw std::runtime_error("worm drag task did not propagate task metrics");
    }

    cells::cellengine::CellEngineConfig pong_cfg = cfg;
    pong_cfg.task_name = "pong_return";
    pong_cfg.body_mode = "grown2d";
    pong_cfg.rl_algorithm = "none";
    pong_cfg.mode = "cell_only";
    pong_cfg.load_genome_csv = genome_csv.string();
    pong_cfg.final_budget.num_trials = 1;
    pong_cfg.final_budget.max_ticks = 8;
    pong_cfg.output_dir = (tmp_dir / "pong").string();
    const auto pong_summary = cells::cellengine::run_full_benchmark(pong_cfg);
    if (pong_summary.task_name != "pong_return" ||
        pong_summary.cell_clean.task_primary_label != "rally_completion") {
        throw std::runtime_error("pong task did not propagate task metrics");
    }

    std::cout << "test_cellengine: OK\n";
    return 0;
}
