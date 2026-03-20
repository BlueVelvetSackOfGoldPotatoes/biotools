// ========================= Battleship =========================

BattleshipEnv::BattleshipEnv(BattleshipConfig cfg) : cfg_(std::move(cfg)) {
    cfg_.board_size = std::max(4, cfg_.board_size);
    cfg_.max_steps = std::max(1, cfg_.max_steps);
    if (cfg_.ship_lengths.empty()) cfg_.ship_lengths = {4, 3, 3, 2, 2};
    board_cells_ = static_cast<std::size_t>(cfg_.board_size * cfg_.board_size);
    ships_.assign(board_cells_, 0);
    observed_.assign(board_cells_, Unknown);
}

std::vector<uint8_t> BattleshipEnv::reset(std::mt19937& rng) {
    ships_.assign(board_cells_, 0);
    observed_.assign(board_cells_, Unknown);
    steps_ = 0;
    if (!place_ships(rng)) {
        throw std::runtime_error("BattleshipEnv: failed to place ships");
    }
    return encode_bits();
}

bool BattleshipEnv::place_ships(std::mt19937& rng) {
    ships_.assign(board_cells_, 0);
    ship_remaining_.assign(cfg_.ship_lengths.size(), 0);
    std::uniform_int_distribution<int> dir_pick(0, 1);
    std::uniform_int_distribution<int> coord_pick(0, cfg_.board_size - 1);

    for (std::size_t ship_id = 0; ship_id < cfg_.ship_lengths.size(); ++ship_id) {
        const int len = std::max(1, cfg_.ship_lengths[ship_id]);
        ship_remaining_[ship_id] = len;
        bool placed = false;
        for (int tries = 0; tries < 3000 && !placed; ++tries) {
            const bool horizontal = dir_pick(rng) == 0;
            const int r0 = coord_pick(rng);
            const int c0 = coord_pick(rng);

            int r1 = r0;
            int c1 = c0;
            if (horizontal) c1 = c0 + len - 1;
            else r1 = r0 + len - 1;
            if (r1 >= cfg_.board_size || c1 >= cfg_.board_size) continue;

            bool ok = true;
            for (int k = 0; k < len; ++k) {
                const int r = horizontal ? r0 : r0 + k;
                const int c = horizontal ? c0 + k : c0;
                if (ships_[static_cast<std::size_t>(index_of(cfg_.board_size, r, c))] != 0) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;
            for (int k = 0; k < len; ++k) {
                const int r = horizontal ? r0 : r0 + k;
                const int c = horizontal ? c0 + k : c0;
                ships_[static_cast<std::size_t>(index_of(cfg_.board_size, r, c))] = static_cast<int>(ship_id + 1);
            }
            placed = true;
        }
        if (!placed) return false;
    }
    return true;
}

std::vector<uint8_t> BattleshipEnv::encode_bits() const {
    std::vector<int> flat;
    flat.reserve(board_cells_);
    for (ShotState s : observed_) flat.push_back(static_cast<int>(s));
    return bitcodec::encode_categorical_vector_one_hot(flat, 3);
}

bool BattleshipEnv::all_sunk() const {
    for (int rem : ship_remaining_) {
        if (rem > 0) return false;
    }
    return true;
}

bool BattleshipEnv::is_hit_unobserved(int idx) const {
    return idx >= 0 &&
           static_cast<std::size_t>(idx) < ships_.size() &&
           ships_[static_cast<std::size_t>(idx)] > 0 &&
           observed_[static_cast<std::size_t>(idx)] == Unknown;
}

std::vector<int> BattleshipEnv::legal_actions() const {
    std::vector<int> out;
    out.reserve(board_cells_);
    for (std::size_t i = 0; i < observed_.size(); ++i) {
        if (observed_[i] == Unknown) out.push_back(static_cast<int>(i));
    }
    return out;
}

int BattleshipEnv::teacher_action() const {
    for (std::size_t i = 0; i < ships_.size(); ++i) {
        if (ships_[i] > 0 && observed_[i] == Unknown) return static_cast<int>(i);
    }
    const auto legal = legal_actions();
    return legal.empty() ? -1 : legal.front();
}

int BattleshipEnv::tactical_immediate_win_action() const {
    int remaining = 0;
    int last = -1;
    for (std::size_t i = 0; i < ships_.size(); ++i) {
        if (ships_[i] > 0 && observed_[i] == Unknown) {
            remaining++;
            last = static_cast<int>(i);
        }
    }
    return remaining == 1 ? last : -1;
}

DiscreteStepResult BattleshipEnv::step(int action, std::mt19937&) {
    DiscreteStepResult out;
    out.plies = 1;

    if (action < 0 || static_cast<std::size_t>(action) >= board_cells_) {
        out.observation_bits = encode_bits();
        out.reward = cfg_.repeat_penalty;
        out.done = false;
        out.outcome = kOutcomeOngoing;
        out.invalid_action = true;
        return out;
    }

    const std::size_t a = static_cast<std::size_t>(action);
    if (observed_[a] != Unknown) {
        out.observation_bits = encode_bits();
        out.reward = cfg_.repeat_penalty;
        out.done = false;
        out.outcome = kOutcomeOngoing;
        out.invalid_action = true;
        return out;
    }

    steps_++;
    const int ship_id = ships_[a];
    if (ship_id > 0) {
        observed_[a] = Hit;
        out.reward = cfg_.hit_reward;
        const std::size_t sid = static_cast<std::size_t>(ship_id - 1);
        if (sid < ship_remaining_.size() && ship_remaining_[sid] > 0) {
            ship_remaining_[sid]--;
            if (ship_remaining_[sid] == 0) out.reward += cfg_.sink_reward;
        }
        if (all_sunk()) {
            out.reward += cfg_.win_reward;
            out.done = true;
            out.outcome = kOutcomeWin;
        }
    } else {
        observed_[a] = Miss;
        out.reward = cfg_.miss_reward;
    }

    if (!out.done && steps_ >= cfg_.max_steps) {
        out.done = true;
        out.outcome = kOutcomeLoss;
        out.reward += cfg_.loss_reward;
    }
    if (!out.done) out.outcome = kOutcomeOngoing;

    out.observation_bits = encode_bits();
    return out;
}
