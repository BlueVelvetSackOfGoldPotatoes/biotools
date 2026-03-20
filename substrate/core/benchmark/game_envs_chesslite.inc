// ========================= Chess (Lite) =========================

ChessLiteEnv::ChessLiteEnv(ChessLiteConfig cfg) : cfg_(std::move(cfg)) {
    cfg_.max_plies = std::max(20, cfg_.max_plies);
    reset_board();
}

void ChessLiteEnv::reset_board() {
    board_.fill(0);
    // Black back rank.
    board_[idx(0, 0)] = -4; board_[idx(0, 1)] = -2; board_[idx(0, 2)] = -3; board_[idx(0, 3)] = -5;
    board_[idx(0, 4)] = -6; board_[idx(0, 5)] = -3; board_[idx(0, 6)] = -2; board_[idx(0, 7)] = -4;
    for (int c = 0; c < 8; ++c) board_[idx(1, c)] = -1;

    // White back rank.
    for (int c = 0; c < 8; ++c) board_[idx(6, c)] = 1;
    board_[idx(7, 0)] = 4; board_[idx(7, 1)] = 2; board_[idx(7, 2)] = 3; board_[idx(7, 3)] = 5;
    board_[idx(7, 4)] = 6; board_[idx(7, 5)] = 3; board_[idx(7, 6)] = 2; board_[idx(7, 7)] = 4;
    plies_ = 0;
}

std::vector<uint8_t> ChessLiteEnv::reset(std::mt19937&) {
    reset_board();
    return encode_bits();
}

int ChessLiteEnv::piece_owner_group(int piece) {
    switch (piece_abs(piece)) {
        case 1: return 0; // pawn
        case 2: return 1; // knight
        case 3: return 2; // bishop
        case 4: return 3; // rook
        case 5: return 4; // queen
        case 6: return 5; // king
        default: return 0;
    }
}

int ChessLiteEnv::piece_value(int piece_abs_code) {
    switch (piece_abs_code) {
        case 1: return 1;
        case 2: return 3;
        case 3: return 3;
        case 4: return 5;
        case 5: return 9;
        case 6: return 100;
        default: return 0;
    }
}

std::vector<uint8_t> ChessLiteEnv::encode_bits() const {
    std::vector<int> cells;
    cells.reserve(64);
    for (int p : board_) {
        // 0: empty
        // 1..6: white pawn..king
        // 7..12: black pawn..king
        if (p == 0) {
            cells.push_back(0);
        } else if (p > 0) {
            cells.push_back(piece_abs(p));
        } else {
            cells.push_back(6 + piece_abs(p));
        }
    }
    return bitcodec::encode_categorical_vector_one_hot(cells, 13);
}

void ChessLiteEnv::add_sliding_moves(std::vector<Move>& out,
                                     int from,
                                     int side_sign,
                                     const std::vector<std::pair<int, int>>& dirs) const {
    const int fr = from / 8;
    const int fc = from % 8;
    for (const auto& [dr, dc] : dirs) {
        int r = fr + dr;
        int c = fc + dc;
        while (on_board(r, c)) {
            const int to = idx(r, c);
            const int target = board_[to];
            if (target == 0) {
                out.push_back({from, to});
            } else {
                if ((target > 0 ? 1 : -1) != side_sign) out.push_back({from, to});
                break;
            }
            r += dr;
            c += dc;
        }
    }
}

std::vector<ChessLiteEnv::Move> ChessLiteEnv::generate_moves_for_side(int side_sign) const {
    std::vector<Move> moves;
    moves.reserve(128);

    for (int from = 0; from < 64; ++from) {
        const int piece = board_[static_cast<std::size_t>(from)];
        if (piece == 0) continue;
        if ((piece > 0 ? 1 : -1) != side_sign) continue;

        const int abs_piece = piece_abs(piece);
        const int fr = from / 8;
        const int fc = from % 8;
        const auto add_if_legal = [&](int tr, int tc) {
            if (!on_board(tr, tc)) return;
            const int to = idx(tr, tc);
            const int tgt = board_[static_cast<std::size_t>(to)];
            if (tgt == 0 || (tgt > 0 ? 1 : -1) != side_sign) {
                moves.push_back({from, to});
            }
        };

        if (abs_piece == 1) {
            const int dir = side_sign > 0 ? -1 : 1;
            const int start_row = side_sign > 0 ? 6 : 1;
            const int one_r = fr + dir;
            if (on_board(one_r, fc) && board_[static_cast<std::size_t>(idx(one_r, fc))] == 0) {
                moves.push_back({from, idx(one_r, fc)});
                const int two_r = fr + 2 * dir;
                if (fr == start_row && on_board(two_r, fc) &&
                    board_[static_cast<std::size_t>(idx(two_r, fc))] == 0) {
                    moves.push_back({from, idx(two_r, fc)});
                }
            }
            for (int dc : {-1, 1}) {
                const int cr = fr + dir;
                const int cc = fc + dc;
                if (!on_board(cr, cc)) continue;
                const int to = idx(cr, cc);
                const int tgt = board_[static_cast<std::size_t>(to)];
                if (tgt != 0 && (tgt > 0 ? 1 : -1) != side_sign) {
                    moves.push_back({from, to});
                }
            }
        } else if (abs_piece == 2) {
            static const std::array<std::pair<int, int>, 8> hops = {
                std::pair<int, int>{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
                {1, -2}, {1, 2}, {2, -1}, {2, 1}
            };
            for (const auto& [dr, dc] : hops) add_if_legal(fr + dr, fc + dc);
        } else if (abs_piece == 3) {
            add_sliding_moves(moves, from, side_sign, {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}});
        } else if (abs_piece == 4) {
            add_sliding_moves(moves, from, side_sign, {{-1, 0}, {1, 0}, {0, -1}, {0, 1}});
        } else if (abs_piece == 5) {
            add_sliding_moves(
                moves,
                from,
                side_sign,
                {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}}
            );
        } else if (abs_piece == 6) {
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    add_if_legal(fr + dr, fc + dc);
                }
            }
        }
    }

    return moves;
}

bool ChessLiteEnv::apply_move(const Move& move, int side_sign, int* captured_piece) {
    if (move.from < 0 || move.from >= 64 || move.to < 0 || move.to >= 64) return false;
    const int piece = board_[static_cast<std::size_t>(move.from)];
    if (piece == 0 || (piece > 0 ? 1 : -1) != side_sign) return false;

    const int target = board_[static_cast<std::size_t>(move.to)];
    if (target != 0 && (target > 0 ? 1 : -1) == side_sign) return false;
    if (captured_piece) *captured_piece = target;

    board_[static_cast<std::size_t>(move.to)] = piece;
    board_[static_cast<std::size_t>(move.from)] = 0;

    // Auto-promote pawns to queen for simplicity.
    const int tr = move.to / 8;
    if (piece_abs(piece) == 1) {
        if ((side_sign > 0 && tr == 0) || (side_sign < 0 && tr == 7)) {
            board_[static_cast<std::size_t>(move.to)] = side_sign * 5;
        }
    }
    return true;
}

bool ChessLiteEnv::has_king(int side_sign) const {
    const int king = side_sign * 6;
    for (int p : board_) {
        if (p == king) return true;
    }
    return false;
}

std::vector<int> ChessLiteEnv::legal_actions() const {
    const auto moves = generate_moves_for_side(+1);
    std::vector<int> out;
    out.reserve(moves.size());
    for (const auto& m : moves) out.push_back(encode_action(m.from, m.to));
    return out;
}

std::vector<int> ChessLiteEnv::action_owner_groups() const {
    std::vector<int> owners(static_cast<std::size_t>(num_actions()), -1);
    const auto moves = generate_moves_for_side(+1);
    for (const auto& m : moves) {
        const int action = encode_action(m.from, m.to);
        const int piece = board_[static_cast<std::size_t>(m.from)];
        owners[static_cast<std::size_t>(action)] = piece_owner_group(piece);
    }
    return owners;
}

int ChessLiteEnv::tactical_immediate_win_action() const {
    const auto moves = generate_moves_for_side(+1);
    for (const auto& m : moves) {
        const int target = board_[static_cast<std::size_t>(m.to)];
        if (target == -6) return encode_action(m.from, m.to);
    }
    return -1;
}

int ChessLiteEnv::teacher_action() const {
    const auto moves = generate_moves_for_side(+1);
    if (moves.empty()) return -1;

    int best_action = encode_action(moves.front().from, moves.front().to);
    int best_score = -100000;
    for (const auto& m : moves) {
        const int target = board_[static_cast<std::size_t>(m.to)];
        int score = 0;
        if (target == -6) score += 100000;
        else if (target < 0) score += 10 * piece_value(piece_abs(target));
        const int to_row = m.to / 8;
        const int to_col = m.to % 8;
        score += (3 - std::abs(3 - to_row)) + (3 - std::abs(3 - to_col));
        if (score > best_score) {
            best_score = score;
            best_action = encode_action(m.from, m.to);
        }
    }
    return best_action;
}

int ChessLiteEnv::choose_opponent_action(std::mt19937& rng) const {
    const auto moves = generate_moves_for_side(-1);
    if (moves.empty()) return -1;

    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> pick(0, moves.size() - 1);
    if (u01(rng) < clamp01(cfg_.opponent_noise)) {
        const auto& m = moves[pick(rng)];
        return encode_action(m.from, m.to);
    }

    int best = encode_action(moves.front().from, moves.front().to);
    int best_score = -100000;
    for (const auto& m : moves) {
        const int target = board_[static_cast<std::size_t>(m.to)];
        int score = 0;
        if (target == 6) score += 100000;
        else if (target > 0) score += 10 * piece_value(piece_abs(target));
        if (score > best_score) {
            best_score = score;
            best = encode_action(m.from, m.to);
        }
    }
    return best;
}

DiscreteStepResult ChessLiteEnv::step(int action, std::mt19937& rng) {
    DiscreteStepResult out;
    out.reward = cfg_.step_reward;
    out.outcome = kOutcomeOngoing;

    if (action < 0 || action >= num_actions()) {
        out.observation_bits = encode_bits();
        out.reward = cfg_.loss_reward;
        out.done = true;
        out.outcome = kOutcomeLoss;
        out.invalid_action = true;
        return out;
    }

    const Move move = decode_action(action);
    int captured = 0;
    if (!apply_move(move, +1, &captured)) {
        out.observation_bits = encode_bits();
        out.reward = cfg_.loss_reward;
        out.done = true;
        out.outcome = kOutcomeLoss;
        out.invalid_action = true;
        return out;
    }
    out.plies++;
    plies_++;

    if (captured != 0) {
        out.reward += cfg_.capture_bonus * static_cast<double>(piece_value(piece_abs(captured)));
    }

    if (!has_king(-1) || captured == -6) {
        out.observation_bits = encode_bits();
        out.reward += cfg_.win_reward;
        out.done = true;
        out.outcome = kOutcomeWin;
        return out;
    }

    const auto opp_moves = generate_moves_for_side(-1);
    if (opp_moves.empty()) {
        out.observation_bits = encode_bits();
        out.reward += cfg_.draw_reward;
        out.done = true;
        out.outcome = kOutcomeDraw;
        return out;
    }

    const int opp_action = choose_opponent_action(rng);
    if (opp_action >= 0) {
        int opp_captured = 0;
        apply_move(decode_action(opp_action), -1, &opp_captured);
        out.plies++;
        plies_++;
    }

    if (!has_king(+1)) {
        out.observation_bits = encode_bits();
        out.reward += cfg_.loss_reward;
        out.done = true;
        out.outcome = kOutcomeLoss;
        return out;
    }

    if (plies_ >= cfg_.max_plies) {
        int material = 0;
        for (int p : board_) {
            if (p > 0) material += piece_value(piece_abs(p));
            else if (p < 0) material -= piece_value(piece_abs(p));
        }
        out.done = true;
        if (material > 0) {
            out.outcome = kOutcomeWin;
            out.reward += cfg_.win_reward;
        } else if (material < 0) {
            out.outcome = kOutcomeLoss;
            out.reward += cfg_.loss_reward;
        } else {
            out.outcome = kOutcomeDraw;
            out.reward += cfg_.draw_reward;
        }
    }

    out.observation_bits = encode_bits();
    return out;
}
