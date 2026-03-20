// ========================= Go (Lite) =========================

GoLiteEnv::GoLiteEnv(GoLiteConfig cfg) : cfg_(std::move(cfg)) {
    cfg_.board_size = std::max(3, cfg_.board_size);
    cfg_.max_plies = std::max(8, cfg_.max_plies);
    board_cells_ = static_cast<std::size_t>(cfg_.board_size * cfg_.board_size);
    board_.assign(board_cells_, kEmpty);
}

std::vector<uint8_t> GoLiteEnv::reset(std::mt19937&) {
    board_.assign(board_cells_, kEmpty);
    plies_ = 0;
    consecutive_passes_ = 0;
    return encode_bits();
}

std::vector<uint8_t> GoLiteEnv::encode_bits() const {
    return bitcodec::encode_categorical_vector_one_hot(board_, 3);
}

std::vector<int> GoLiteEnv::neighbors(int cell) const {
    const int n = cfg_.board_size;
    const int r = cell / n;
    const int c = cell % n;
    std::vector<int> out;
    out.reserve(4);
    if (r > 0) out.push_back(idx(r - 1, c));
    if (r + 1 < n) out.push_back(idx(r + 1, c));
    if (c > 0) out.push_back(idx(r, c - 1));
    if (c + 1 < n) out.push_back(idx(r, c + 1));
    return out;
}

std::vector<int> GoLiteEnv::collect_group(int start) const {
    const int color = board_[static_cast<std::size_t>(start)];
    if (color == kEmpty) return {};
    std::vector<int> group;
    std::vector<int> stack = {start};
    std::vector<uint8_t> seen(board_cells_, 0);
    seen[static_cast<std::size_t>(start)] = 1;
    while (!stack.empty()) {
        const int cur = stack.back();
        stack.pop_back();
        group.push_back(cur);
        for (int nb : neighbors(cur)) {
            const std::size_t nbi = static_cast<std::size_t>(nb);
            if (seen[nbi]) continue;
            if (board_[nbi] != color) continue;
            seen[nbi] = 1;
            stack.push_back(nb);
        }
    }
    return group;
}

bool GoLiteEnv::group_has_liberty(const std::vector<int>& group) const {
    for (int cell : group) {
        for (int nb : neighbors(cell)) {
            if (board_[static_cast<std::size_t>(nb)] == kEmpty) return true;
        }
    }
    return false;
}

int GoLiteEnv::capture_adjacent_groups(int cell, int player, int opponent) {
    (void)player;
    int captured = 0;
    std::vector<uint8_t> seen(board_cells_, 0);
    for (int nb : neighbors(cell)) {
        const std::size_t nbi = static_cast<std::size_t>(nb);
        if (seen[nbi]) continue;
        if (board_[nbi] != opponent) continue;
        const auto group = collect_group(nb);
        for (int g : group) seen[static_cast<std::size_t>(g)] = 1;
        if (!group_has_liberty(group)) {
            for (int g : group) board_[static_cast<std::size_t>(g)] = kEmpty;
            captured += static_cast<int>(group.size());
        }
    }
    return captured;
}

bool GoLiteEnv::would_be_suicide(int cell, int player, int opponent) const {
    if (board_[static_cast<std::size_t>(cell)] != kEmpty) return true;
    auto copy = board_;
    copy[static_cast<std::size_t>(cell)] = player;

    auto neighbors_of = [&](int c) {
        const int n = cfg_.board_size;
        const int r = c / n;
        const int col = c % n;
        std::vector<int> out;
        if (r > 0) out.push_back((r - 1) * n + col);
        if (r + 1 < n) out.push_back((r + 1) * n + col);
        if (col > 0) out.push_back(r * n + (col - 1));
        if (col + 1 < n) out.push_back(r * n + (col + 1));
        return out;
    };
    auto collect_group_copy = [&](int start) {
        const int color = copy[static_cast<std::size_t>(start)];
        std::vector<int> group;
        if (color == kEmpty) return group;
        std::vector<int> stack = {start};
        std::vector<uint8_t> seen(copy.size(), 0);
        seen[static_cast<std::size_t>(start)] = 1;
        while (!stack.empty()) {
            const int cur = stack.back();
            stack.pop_back();
            group.push_back(cur);
            for (int nb : neighbors_of(cur)) {
                const std::size_t nbi = static_cast<std::size_t>(nb);
                if (seen[nbi] || copy[nbi] != color) continue;
                seen[nbi] = 1;
                stack.push_back(nb);
            }
        }
        return group;
    };
    auto has_lib = [&](const std::vector<int>& group) {
        for (int g : group) {
            for (int nb : neighbors_of(g)) {
                if (copy[static_cast<std::size_t>(nb)] == kEmpty) return true;
            }
        }
        return false;
    };

    std::vector<uint8_t> seen(copy.size(), 0);
    for (int nb : neighbors_of(cell)) {
        const std::size_t nbi = static_cast<std::size_t>(nb);
        if (seen[nbi] || copy[nbi] != opponent) continue;
        const auto grp = collect_group_copy(nb);
        for (int g : grp) seen[static_cast<std::size_t>(g)] = 1;
        if (!has_lib(grp)) {
            for (int g : grp) copy[static_cast<std::size_t>(g)] = kEmpty;
        }
    }
    const auto own = collect_group_copy(cell);
    return !has_lib(own);
}

std::vector<int> GoLiteEnv::legal_actions() const {
    std::vector<int> out;
    out.reserve(board_cells_ + 1);
    out.push_back(pass_action());
    for (std::size_t i = 0; i < board_cells_; ++i) {
        if (board_[i] != kEmpty) continue;
        if (!would_be_suicide(static_cast<int>(i), kAgent, kOpponent)) {
            out.push_back(static_cast<int>(i));
        }
    }
    return out;
}

int GoLiteEnv::teacher_action() const {
    const auto legal = legal_actions();
    if (legal.empty()) return -1;

    int best = pass_action();
    int best_capture = -1;
    for (int a : legal) {
        if (a == pass_action()) continue;
        auto copy = *this;
        copy.board_[static_cast<std::size_t>(a)] = kAgent;
        const int cap = copy.capture_adjacent_groups(a, kAgent, kOpponent);
        if (cap > best_capture) {
            best_capture = cap;
            best = a;
        }
    }
    if (best_capture > 0) return best;

    const int center = idx(cfg_.board_size / 2, cfg_.board_size / 2);
    if (std::find(legal.begin(), legal.end(), center) != legal.end()) return center;
    for (int a : legal) {
        if (a != pass_action()) return a;
    }
    return pass_action();
}

int GoLiteEnv::choose_opponent_action(std::mt19937& rng) const {
    std::vector<int> legal;
    legal.reserve(board_cells_ + 1);
    legal.push_back(pass_action());
    for (std::size_t i = 0; i < board_cells_; ++i) {
        if (board_[i] != kEmpty) continue;
        if (!would_be_suicide(static_cast<int>(i), kOpponent, kAgent)) {
            legal.push_back(static_cast<int>(i));
        }
    }
    if (legal.empty()) return pass_action();

    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> pick(0, legal.size() - 1);
    if (u01(rng) < clamp01(cfg_.opponent_noise)) {
        return legal[pick(rng)];
    }

    int best = pass_action();
    int best_capture = -1;
    for (int a : legal) {
        if (a == pass_action()) continue;
        auto copy = *this;
        copy.board_[static_cast<std::size_t>(a)] = kOpponent;
        const int cap = copy.capture_adjacent_groups(a, kOpponent, kAgent);
        if (cap > best_capture) {
            best_capture = cap;
            best = a;
        }
    }
    if (best_capture > 0) return best;
    return legal[pick(rng)];
}

int GoLiteEnv::score_position() const {
    int agent = 0;
    int opp = 0;
    for (int c : board_) {
        if (c == kAgent) agent++;
        else if (c == kOpponent) opp++;
    }
    return agent - opp;
}

DiscreteStepResult GoLiteEnv::step(int action, std::mt19937& rng) {
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

    auto finish_if_terminal = [&](bool force_terminal = false) {
        if (force_terminal || consecutive_passes_ >= 2 || plies_ >= cfg_.max_plies) {
            const int score = score_position();
            out.done = true;
            if (score > 0) {
                out.outcome = kOutcomeWin;
                out.reward += cfg_.win_reward;
            } else if (score < 0) {
                out.outcome = kOutcomeLoss;
                out.reward += cfg_.loss_reward;
            } else {
                out.outcome = kOutcomeDraw;
                out.reward += cfg_.draw_reward;
            }
            return true;
        }
        return false;
    };

    if (action == pass_action()) {
        consecutive_passes_++;
        plies_++;
        out.plies++;
        if (finish_if_terminal()) {
            out.observation_bits = encode_bits();
            return out;
        }
    } else {
        const std::size_t a = static_cast<std::size_t>(action);
        if (board_[a] != kEmpty || would_be_suicide(action, kAgent, kOpponent)) {
            out.observation_bits = encode_bits();
            out.reward = cfg_.loss_reward;
            out.done = true;
            out.outcome = kOutcomeLoss;
            out.invalid_action = true;
            return out;
        }

        board_[a] = kAgent;
        const int cap = capture_adjacent_groups(action, kAgent, kOpponent);
        if (cap > 0) out.reward += cfg_.capture_reward * static_cast<double>(cap);
        consecutive_passes_ = 0;
        plies_++;
        out.plies++;
        if (finish_if_terminal()) {
            out.observation_bits = encode_bits();
            return out;
        }
    }

    const int opp_action = choose_opponent_action(rng);
    if (opp_action == pass_action()) {
        consecutive_passes_++;
        plies_++;
        out.plies++;
    } else if (opp_action >= 0 && opp_action < pass_action()) {
        const std::size_t oa = static_cast<std::size_t>(opp_action);
        if (board_[oa] == kEmpty && !would_be_suicide(opp_action, kOpponent, kAgent)) {
            board_[oa] = kOpponent;
            capture_adjacent_groups(opp_action, kOpponent, kAgent);
            consecutive_passes_ = 0;
            plies_++;
            out.plies++;
        } else {
            consecutive_passes_++;
            plies_++;
            out.plies++;
        }
    }

    finish_if_terminal();
    out.observation_bits = encode_bits();
    return out;
}
