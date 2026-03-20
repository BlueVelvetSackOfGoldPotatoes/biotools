#pragma once

#include "core/benchmark/discrete_env.h"

#include <array>
#include <cstddef>
#include <random>
#include <unordered_map>
#include <vector>

namespace benchmark {

struct TicTacToeConfig {
    double opponent_noise = 0.15;
    double win_reward = 1.0;
    double draw_reward = 0.2;
    double loss_reward = -1.0;
    double step_reward = -0.01;
};

class TicTacToeEnv final : public DiscreteEnv {
public:
    explicit TicTacToeEnv(TicTacToeConfig cfg = {});

    std::string benchmark_id() const override { return "tictactoe"; }
    std::string benchmark_name() const override { return "TicTacToe"; }
    std::string task_type() const override { return "control"; }

    std::size_t observation_bits() const override { return 27; } // 9 cells x 3 one-hot bits
    int num_actions() const override { return 9; }

    std::vector<uint8_t> reset(std::mt19937& rng) override;
    std::vector<int> legal_actions() const override;
    DiscreteStepResult step(int action, std::mt19937& rng) override;
    int teacher_action() const override;
    int tactical_immediate_win_action() const override;
    int tactical_immediate_block_action() const override;

    void set_opponent_noise(double noise);
    double opponent_noise() const { return cfg_.opponent_noise; }

private:
    std::array<int, 9> board_{};
    TicTacToeConfig cfg_;
    mutable std::unordered_map<int, int> minimax_cache_;

    std::vector<uint8_t> encode_observation_bits() const;

    static int check_winner(const std::array<int, 9>& board);
    static bool board_full(const std::array<int, 9>& board);
    static std::vector<int> legal_actions_for(const std::array<int, 9>& board);
    static int forced_win_move(std::array<int, 9> board, int player);
    static int encode_board_state(const std::array<int, 9>& board);

    int opponent_action(std::mt19937& rng) const;
    int oracle_action() const;
    int minimax_score(std::array<int, 9>& board, int turn) const;
};

} // namespace benchmark
