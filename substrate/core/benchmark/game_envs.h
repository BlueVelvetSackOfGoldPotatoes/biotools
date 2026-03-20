#pragma once

#include "core/benchmark/discrete_env.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace benchmark {

struct ConnectFourConfig {
    double opponent_noise = 0.2;
    double win_reward = 1.0;
    double draw_reward = 0.2;
    double loss_reward = -1.0;
    double step_reward = -0.01;
};

class ConnectFourEnv final : public DiscreteEnv {
public:
    explicit ConnectFourEnv(ConnectFourConfig cfg = {});

    std::string benchmark_id() const override { return "connect_four"; }
    std::string benchmark_name() const override { return "Connect Four"; }
    std::string task_type() const override { return "control"; }

    std::size_t observation_bits() const override { return 7 * 6 * 3; }
    int num_actions() const override { return 7; }

    std::vector<uint8_t> reset(std::mt19937& rng) override;
    std::vector<int> legal_actions() const override;
    DiscreteStepResult step(int action, std::mt19937& rng) override;
    int teacher_action() const override;
    int tactical_immediate_win_action() const override;
    int tactical_immediate_block_action() const override;

private:
    static constexpr int kWidth = 7;
    static constexpr int kHeight = 6;
    static constexpr int kCells = kWidth * kHeight;

    std::array<int, kCells> board_{};
    ConnectFourConfig cfg_;

    static int cell_index(int row, int col) { return row * kWidth + col; }
    static bool board_full(const std::array<int, kCells>& board);
    static int check_winner(const std::array<int, kCells>& board);
    static std::vector<int> legal_actions_for(const std::array<int, kCells>& board);
    static bool apply_drop(std::array<int, kCells>& board, int col, int player);
    static int winning_action(std::array<int, kCells> board, int player);
    static int immediate_block_action(std::array<int, kCells> board, int player, int opponent);

    std::vector<uint8_t> encode_bits() const;
    int opponent_action(std::mt19937& rng) const;
};

struct BattleshipConfig {
    int board_size = 8;
    int max_steps = 64;
    std::vector<int> ship_lengths = {4, 3, 3, 2, 2};
    double hit_reward = 1.0;
    double sink_reward = 2.5;
    double miss_reward = -0.05;
    double repeat_penalty = -0.2;
    double win_reward = 5.0;
    double loss_reward = -1.0;
};

class BattleshipEnv final : public DiscreteEnv {
public:
    explicit BattleshipEnv(BattleshipConfig cfg = {});

    std::string benchmark_id() const override { return "battleship"; }
    std::string benchmark_name() const override { return "Battleship"; }
    std::string task_type() const override { return "control"; }

    std::size_t observation_bits() const override { return board_cells_ * 3; }
    int num_actions() const override { return static_cast<int>(board_cells_); }

    std::vector<uint8_t> reset(std::mt19937& rng) override;
    std::vector<int> legal_actions() const override;
    DiscreteStepResult step(int action, std::mt19937& rng) override;
    int teacher_action() const override;
    int tactical_immediate_win_action() const override;

private:
    enum ShotState : int { Unknown = 0, Miss = 1, Hit = 2 };

    BattleshipConfig cfg_;
    std::size_t board_cells_ = 0;
    int steps_ = 0;
    std::vector<int> ships_; // 0 empty, >0 ship id
    std::vector<ShotState> observed_;
    std::vector<int> ship_remaining_;

    static int index_of(int board_size, int row, int col) { return row * board_size + col; }

    std::vector<uint8_t> encode_bits() const;
    bool place_ships(std::mt19937& rng);
    bool all_sunk() const;
    bool is_hit_unobserved(int idx) const;
};

struct GoLiteConfig {
    int board_size = 5;
    int max_plies = 80;
    double opponent_noise = 0.3;
    double capture_reward = 0.1;
    double step_reward = -0.005;
    double win_reward = 1.0;
    double draw_reward = 0.2;
    double loss_reward = -1.0;
};

class GoLiteEnv final : public DiscreteEnv {
public:
    explicit GoLiteEnv(GoLiteConfig cfg = {});

    std::string benchmark_id() const override { return "go"; }
    std::string benchmark_name() const override { return "Go (Lite)"; }
    std::string task_type() const override { return "control"; }

    std::size_t observation_bits() const override { return board_cells_ * 3; }
    int num_actions() const override { return static_cast<int>(board_cells_ + 1); } // + pass

    std::vector<uint8_t> reset(std::mt19937& rng) override;
    std::vector<int> legal_actions() const override;
    DiscreteStepResult step(int action, std::mt19937& rng) override;
    int teacher_action() const override;

private:
    GoLiteConfig cfg_;
    int plies_ = 0;
    int consecutive_passes_ = 0;
    std::size_t board_cells_ = 0;
    std::vector<int> board_; // 0 empty, 1 agent, 2 opponent

    int pass_action() const { return static_cast<int>(board_cells_); }
    int idx(int row, int col) const { return row * cfg_.board_size + col; }

    std::vector<uint8_t> encode_bits() const;
    std::vector<int> neighbors(int cell) const;
    std::vector<int> collect_group(int start) const;
    bool group_has_liberty(const std::vector<int>& group) const;
    int capture_adjacent_groups(int cell, int player, int opponent);
    bool would_be_suicide(int cell, int player, int opponent) const;
    int choose_opponent_action(std::mt19937& rng) const;
    int score_position() const;
};

struct ChessLiteConfig {
    int max_plies = 200;
    double opponent_noise = 0.2;
    double capture_bonus = 0.02;
    double step_reward = -0.001;
    double win_reward = 1.0;
    double draw_reward = 0.1;
    double loss_reward = -1.0;
};

class ChessLiteEnv final : public DiscreteEnv {
public:
    explicit ChessLiteEnv(ChessLiteConfig cfg = {});

    std::string benchmark_id() const override { return "chess"; }
    std::string benchmark_name() const override { return "Chess (Lite)"; }
    std::string task_type() const override { return "control"; }

    std::size_t observation_bits() const override { return 64 * 13; }
    int num_actions() const override { return 64 * 64; } // from*64 + to

    std::vector<uint8_t> reset(std::mt19937& rng) override;
    std::vector<int> legal_actions() const override;
    DiscreteStepResult step(int action, std::mt19937& rng) override;
    int teacher_action() const override;
    int tactical_immediate_win_action() const override;

    int num_action_owner_groups() const override { return 6; }
    std::vector<int> action_owner_groups() const override;
    std::vector<std::string> action_owner_group_names() const override {
        return {"pawn", "knight", "bishop", "rook", "queen", "king"};
    }

private:
    ChessLiteConfig cfg_;
    std::array<int, 64> board_{};
    int plies_ = 0;

    static int idx(int row, int col) { return row * 8 + col; }
    static bool on_board(int row, int col) { return row >= 0 && row < 8 && col >= 0 && col < 8; }

    struct Move {
        int from = 0;
        int to = 0;
    };

    static int piece_abs(int piece) { return piece < 0 ? -piece : piece; }
    static int piece_owner_group(int piece); // 0..5
    static int piece_value(int piece_abs_code);

    std::vector<uint8_t> encode_bits() const;
    static int encode_action(int from, int to) { return from * 64 + to; }
    static Move decode_action(int action) { return Move{action / 64, action % 64}; }

    void reset_board();
    std::vector<Move> generate_moves_for_side(int side_sign) const;
    void add_sliding_moves(std::vector<Move>& out, int from, int side_sign,
                           const std::vector<std::pair<int, int>>& dirs) const;
    bool apply_move(const Move& move, int side_sign, int* captured_piece = nullptr);
    bool has_king(int side_sign) const;
    int choose_opponent_action(std::mt19937& rng) const;
};

struct CartPoleBitsConfig {
    int max_steps = 500;
    int bins_per_dim = 32;
    double step_reward = 1.0;
    double failure_reward = -1.0;
};

class CartPoleBitsEnv final : public DiscreteEnv {
public:
    explicit CartPoleBitsEnv(CartPoleBitsConfig cfg = {});

    std::string benchmark_id() const override { return "cartpole"; }
    std::string benchmark_name() const override { return "CartPole"; }
    std::string task_type() const override { return "control"; }

    std::size_t observation_bits() const override { return static_cast<std::size_t>(cfg_.bins_per_dim * 4); }
    int num_actions() const override { return 2; } // left/right force

    std::vector<uint8_t> reset(std::mt19937& rng) override;
    std::vector<int> legal_actions() const override { return {0, 1}; }
    DiscreteStepResult step(int action, std::mt19937& rng) override;
    int teacher_action() const override;

private:
    CartPoleBitsConfig cfg_;
    int steps_ = 0;
    double x_ = 0.0;
    double x_dot_ = 0.0;
    double theta_ = 0.0;
    double theta_dot_ = 0.0;

    std::vector<uint8_t> encode_bits() const;
    int quantize(double value, double lo, double hi) const;
    bool is_failure() const;
};

} // namespace benchmark
