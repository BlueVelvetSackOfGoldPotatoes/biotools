#include "core/benchmark/tictactoe_env.h"

#include "core/benchmark/bit_codec.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace benchmark {

namespace {

constexpr int kEmpty = 0;
constexpr int kAgent = 1;    // X
constexpr int kOpponent = 2; // O

constexpr int kOutcomeWin = 1;
constexpr int kOutcomeDraw = 0;
constexpr int kOutcomeLoss = -1;
constexpr int kOutcomeOngoing = 2;

} // namespace

TicTacToeEnv::TicTacToeEnv(TicTacToeConfig cfg) : cfg_(cfg) {
    board_.fill(kEmpty);
}

void TicTacToeEnv::set_opponent_noise(double noise) {
    cfg_.opponent_noise = std::clamp(noise, 0.0, 1.0);
}

std::vector<uint8_t> TicTacToeEnv::reset(std::mt19937&) {
    board_.fill(kEmpty);
    return encode_observation_bits();
}

std::vector<int> TicTacToeEnv::legal_actions_for(const std::array<int, 9>& board) {
    std::vector<int> out;
    out.reserve(9);
    for (int idx = 0; idx < 9; ++idx) {
        if (board[static_cast<std::size_t>(idx)] == kEmpty) out.push_back(idx);
    }
    return out;
}

std::vector<int> TicTacToeEnv::legal_actions() const {
    return legal_actions_for(board_);
}

int TicTacToeEnv::check_winner(const std::array<int, 9>& board) {
    static constexpr int lines[8][3] = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
        {0, 4, 8}, {2, 4, 6}
    };

    for (const auto& line : lines) {
        const int v0 = board[static_cast<std::size_t>(line[0])];
        if (v0 == kEmpty) continue;
        if (board[static_cast<std::size_t>(line[1])] == v0 &&
            board[static_cast<std::size_t>(line[2])] == v0) {
            return v0;
        }
    }
    return kEmpty;
}

bool TicTacToeEnv::board_full(const std::array<int, 9>& board) {
    return std::all_of(board.begin(), board.end(), [](int v) { return v != kEmpty; });
}

int TicTacToeEnv::forced_win_move(std::array<int, 9> board, int player) {
    const auto legal = legal_actions_for(board);
    for (const int action : legal) {
        board[static_cast<std::size_t>(action)] = player;
        if (check_winner(board) == player) return action;
        board[static_cast<std::size_t>(action)] = kEmpty;
    }
    return -1;
}

int TicTacToeEnv::opponent_action(std::mt19937& rng) const {
    const auto legal = legal_actions_for(board_);
    if (legal.empty()) return -1;

    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> random_legal(0, legal.size() - 1);

    if (u01(rng) < std::clamp(cfg_.opponent_noise, 0.0, 1.0)) {
        return legal[random_legal(rng)];
    }

    {
        auto copy = board_;
        const int win = forced_win_move(copy, kOpponent);
        if (win >= 0) return win;
    }
    {
        auto copy = board_;
        const int block = forced_win_move(copy, kAgent);
        if (block >= 0) return block;
    }

    if (board_[4] == kEmpty) return 4;

    std::vector<int> corners;
    for (const int idx : {0, 2, 6, 8}) {
        if (board_[static_cast<std::size_t>(idx)] == kEmpty) corners.push_back(idx);
    }
    if (!corners.empty()) {
        std::uniform_int_distribution<std::size_t> corner_pick(0, corners.size() - 1);
        return corners[corner_pick(rng)];
    }

    return legal[random_legal(rng)];
}

int TicTacToeEnv::encode_board_state(const std::array<int, 9>& board) {
    int code = 0;
    int factor = 1;
    for (const int cell : board) {
        code += cell * factor;
        factor *= 3;
    }
    return code;
}

int TicTacToeEnv::minimax_score(std::array<int, 9>& board, int turn) const {
    const int winner = check_winner(board);
    if (winner == kAgent) return 1;
    if (winner == kOpponent) return -1;
    if (board_full(board)) return 0;

    const int key = encode_board_state(board) + (turn == kOpponent ? 19683 : 0);
    const auto it = minimax_cache_.find(key);
    if (it != minimax_cache_.end()) return it->second;

    const auto legal = legal_actions_for(board);
    int best = (turn == kAgent) ? -2 : 2;

    for (const int action : legal) {
        board[static_cast<std::size_t>(action)] = turn;
        const int child = minimax_score(board, turn == kAgent ? kOpponent : kAgent);
        board[static_cast<std::size_t>(action)] = kEmpty;

        if (turn == kAgent) {
            best = std::max(best, child);
            if (best == 1) break;
        } else {
            best = std::min(best, child);
            if (best == -1) break;
        }
    }

    minimax_cache_[key] = best;
    return best;
}

int TicTacToeEnv::oracle_action() const {
    const auto legal = legal_actions_for(board_);
    if (legal.empty()) return -1;

    int best_score = -3;
    std::vector<int> best_actions;

    for (const int action : legal) {
        auto copy = board_;
        copy[static_cast<std::size_t>(action)] = kAgent;

        int score = 0;
        if (check_winner(copy) == kAgent) {
            score = 1;
        } else if (board_full(copy)) {
            score = 0;
        } else {
            score = minimax_score(copy, kOpponent);
        }

        if (score > best_score) {
            best_score = score;
            best_actions.clear();
            best_actions.push_back(action);
        } else if (score == best_score) {
            best_actions.push_back(action);
        }
    }

    // Stable preference order for deterministic labels.
    static constexpr int pref[9] = {4, 0, 2, 6, 8, 1, 3, 5, 7};
    for (const int idx : pref) {
        if (std::find(best_actions.begin(), best_actions.end(), idx) != best_actions.end()) {
            return idx;
        }
    }
    return best_actions.front();
}

int TicTacToeEnv::teacher_action() const {
    return oracle_action();
}

int TicTacToeEnv::tactical_immediate_win_action() const {
    auto copy = board_;
    return forced_win_move(copy, kAgent);
}

int TicTacToeEnv::tactical_immediate_block_action() const {
    auto copy = board_;
    return forced_win_move(copy, kOpponent);
}

std::vector<uint8_t> TicTacToeEnv::encode_observation_bits() const {
    std::vector<int> cells(board_.begin(), board_.end());
    return bitcodec::encode_categorical_vector_one_hot(cells, 3);
}

DiscreteStepResult TicTacToeEnv::step(int action, std::mt19937& rng) {
    DiscreteStepResult out;

    if (action < 0 || action >= 9 || board_[static_cast<std::size_t>(action)] != kEmpty) {
        out.observation_bits = encode_observation_bits();
        out.reward = cfg_.loss_reward;
        out.done = true;
        out.outcome = kOutcomeLoss;
        out.invalid_action = true;
        return out;
    }

    board_[static_cast<std::size_t>(action)] = kAgent;
    out.plies++;

    if (check_winner(board_) == kAgent) {
        out.observation_bits = encode_observation_bits();
        out.reward = cfg_.win_reward;
        out.done = true;
        out.outcome = kOutcomeWin;
        return out;
    }
    if (board_full(board_)) {
        out.observation_bits = encode_observation_bits();
        out.reward = cfg_.draw_reward;
        out.done = true;
        out.outcome = kOutcomeDraw;
        return out;
    }

    const int opp_action = opponent_action(rng);
    if (opp_action >= 0) {
        board_[static_cast<std::size_t>(opp_action)] = kOpponent;
        out.plies++;
    }

    if (check_winner(board_) == kOpponent) {
        out.observation_bits = encode_observation_bits();
        out.reward = cfg_.loss_reward;
        out.done = true;
        out.outcome = kOutcomeLoss;
        return out;
    }
    if (board_full(board_)) {
        out.observation_bits = encode_observation_bits();
        out.reward = cfg_.draw_reward;
        out.done = true;
        out.outcome = kOutcomeDraw;
        return out;
    }

    out.observation_bits = encode_observation_bits();
    out.reward = cfg_.step_reward;
    out.done = false;
    out.outcome = kOutcomeOngoing;
    return out;
}

} // namespace benchmark
