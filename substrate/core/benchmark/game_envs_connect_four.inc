// ========================= Connect Four =========================

ConnectFourEnv::ConnectFourEnv(ConnectFourConfig cfg) : cfg_(cfg) {
    board_.fill(kEmpty);
}

std::vector<uint8_t> ConnectFourEnv::reset(std::mt19937&) {
    board_.fill(kEmpty);
    return encode_bits();
}

std::vector<int> ConnectFourEnv::legal_actions_for(const std::array<int, kCells>& board) {
    std::vector<int> out;
    out.reserve(kWidth);
    for (int c = 0; c < kWidth; ++c) {
        if (board[cell_index(0, c)] == kEmpty) out.push_back(c);
    }
    return out;
}

std::vector<int> ConnectFourEnv::legal_actions() const {
    return legal_actions_for(board_);
}

bool ConnectFourEnv::board_full(const std::array<int, kCells>& board) {
    for (int c = 0; c < kWidth; ++c) {
        if (board[cell_index(0, c)] == kEmpty) return false;
    }
    return true;
}

bool ConnectFourEnv::apply_drop(std::array<int, kCells>& board, int col, int player) {
    if (col < 0 || col >= kWidth) return false;
    for (int r = kHeight - 1; r >= 0; --r) {
        const int id = cell_index(r, col);
        if (board[id] == kEmpty) {
            board[id] = player;
            return true;
        }
    }
    return false;
}

int ConnectFourEnv::check_winner(const std::array<int, kCells>& board) {
    auto get = [&](int r, int c) -> int {
        if (r < 0 || r >= kHeight || c < 0 || c >= kWidth) return kEmpty;
        return board[cell_index(r, c)];
    };
    const std::array<std::pair<int, int>, 4> dirs = {
        std::pair<int, int>{0, 1}, std::pair<int, int>{1, 0},
        std::pair<int, int>{1, 1}, std::pair<int, int>{1, -1}
    };
    for (int r = 0; r < kHeight; ++r) {
        for (int c = 0; c < kWidth; ++c) {
            const int p = get(r, c);
            if (p == kEmpty) continue;
            for (const auto& [dr, dc] : dirs) {
                bool ok = true;
                for (int k = 1; k < 4; ++k) {
                    if (get(r + dr * k, c + dc * k) != p) {
                        ok = false;
                        break;
                    }
                }
                if (ok) return p;
            }
        }
    }
    return kEmpty;
}

int ConnectFourEnv::winning_action(std::array<int, kCells> board, int player) {
    const auto legal = legal_actions_for(board);
    for (const int a : legal) {
        auto copy = board;
        if (!apply_drop(copy, a, player)) continue;
        if (check_winner(copy) == player) return a;
    }
    return -1;
}

int ConnectFourEnv::immediate_block_action(std::array<int, kCells> board, int player, int opponent) {
    const auto legal = legal_actions_for(board);
    for (const int a : legal) {
        auto copy = board;
        if (!apply_drop(copy, a, player)) continue;
        if (winning_action(copy, opponent) < 0) return a;
    }
    return -1;
}

int ConnectFourEnv::tactical_immediate_win_action() const {
    return winning_action(board_, kAgent);
}

int ConnectFourEnv::tactical_immediate_block_action() const {
    return winning_action(board_, kOpponent);
}

int ConnectFourEnv::teacher_action() const {
    const int win = winning_action(board_, kAgent);
    if (win >= 0) return win;

    const int block = winning_action(board_, kOpponent);
    if (block >= 0) return block;

    const auto legal = legal_actions();
    if (legal.empty()) return -1;
    if (std::find(legal.begin(), legal.end(), 3) != legal.end()) return 3;

    int best = legal.front();
    int best_dist = std::abs(best - 3);
    for (int a : legal) {
        const int d = std::abs(a - 3);
        if (d < best_dist) {
            best = a;
            best_dist = d;
        }
    }
    return best;
}

int ConnectFourEnv::opponent_action(std::mt19937& rng) const {
    const auto legal = legal_actions();
    if (legal.empty()) return -1;

    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
    if (u01(rng) < clamp01(cfg_.opponent_noise)) {
        return legal[pick(rng)];
    }

    const int win = winning_action(board_, kOpponent);
    if (win >= 0) return win;
    const int block = winning_action(board_, kAgent);
    if (block >= 0) return block;
    if (std::find(legal.begin(), legal.end(), 3) != legal.end()) return 3;
    return legal[pick(rng)];
}

std::vector<uint8_t> ConnectFourEnv::encode_bits() const {
    std::vector<int> flat;
    flat.reserve(kCells);
    for (int v : board_) flat.push_back(v);
    return bitcodec::encode_categorical_vector_one_hot(flat, 3);
}

DiscreteStepResult ConnectFourEnv::step(int action, std::mt19937& rng) {
    DiscreteStepResult out;

    if (action < 0 || action >= kWidth || !apply_drop(board_, action, kAgent)) {
        out.observation_bits = encode_bits();
        out.reward = cfg_.loss_reward;
        out.done = true;
        out.outcome = kOutcomeLoss;
        out.invalid_action = true;
        return out;
    }
    out.plies++;

    if (check_winner(board_) == kAgent) {
        out.observation_bits = encode_bits();
        out.reward = cfg_.win_reward;
        out.done = true;
        out.outcome = kOutcomeWin;
        return out;
    }
    if (board_full(board_)) {
        out.observation_bits = encode_bits();
        out.reward = cfg_.draw_reward;
        out.done = true;
        out.outcome = kOutcomeDraw;
        return out;
    }

    const int opp = opponent_action(rng);
    if (opp >= 0) {
        apply_drop(board_, opp, kOpponent);
        out.plies++;
    }

    if (check_winner(board_) == kOpponent) {
        out.observation_bits = encode_bits();
        out.reward = cfg_.loss_reward;
        out.done = true;
        out.outcome = kOutcomeLoss;
        return out;
    }
    if (board_full(board_)) {
        out.observation_bits = encode_bits();
        out.reward = cfg_.draw_reward;
        out.done = true;
        out.outcome = kOutcomeDraw;
        return out;
    }

    out.observation_bits = encode_bits();
    out.reward = cfg_.step_reward;
    out.done = false;
    out.outcome = kOutcomeOngoing;
    return out;
}
