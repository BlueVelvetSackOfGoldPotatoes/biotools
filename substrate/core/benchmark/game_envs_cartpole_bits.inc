// ========================= CartPole =========================

CartPoleBitsEnv::CartPoleBitsEnv(CartPoleBitsConfig cfg) : cfg_(std::move(cfg)) {
    cfg_.max_steps = std::max(10, cfg_.max_steps);
    cfg_.bins_per_dim = std::max(8, cfg_.bins_per_dim);
}

std::vector<uint8_t> CartPoleBitsEnv::reset(std::mt19937& rng) {
    std::uniform_real_distribution<double> init(-0.05, 0.05);
    x_ = init(rng);
    x_dot_ = init(rng);
    theta_ = init(rng);
    theta_dot_ = init(rng);
    steps_ = 0;
    return encode_bits();
}

int CartPoleBitsEnv::quantize(double value, double lo, double hi) const {
    const double clamped = std::clamp(value, lo, hi);
    const double t = (clamped - lo) / std::max(1e-9, hi - lo);
    int q = static_cast<int>(std::floor(t * static_cast<double>(cfg_.bins_per_dim)));
    if (q >= cfg_.bins_per_dim) q = cfg_.bins_per_dim - 1;
    if (q < 0) q = 0;
    return q;
}

std::vector<uint8_t> CartPoleBitsEnv::encode_bits() const {
    std::vector<int> qs;
    qs.reserve(4);
    qs.push_back(quantize(x_, -2.4, 2.4));
    qs.push_back(quantize(x_dot_, -3.0, 3.0));
    qs.push_back(quantize(theta_, -0.2095, 0.2095));
    qs.push_back(quantize(theta_dot_, -3.5, 3.5));
    return bitcodec::encode_categorical_vector_one_hot(qs, cfg_.bins_per_dim);
}

bool CartPoleBitsEnv::is_failure() const {
    constexpr double x_threshold = 2.4;
    constexpr double theta_threshold = 12.0 * 3.14159265358979323846 / 180.0;
    return std::abs(x_) > x_threshold || std::abs(theta_) > theta_threshold;
}

int CartPoleBitsEnv::teacher_action() const {
    const double control = theta_ + 0.25 * theta_dot_ + 0.02 * x_ + 0.05 * x_dot_;
    return control > 0.0 ? 1 : 0;
}

DiscreteStepResult CartPoleBitsEnv::step(int action, std::mt19937&) {
    DiscreteStepResult out;
    out.plies = 1;
    out.outcome = kOutcomeOngoing;

    if (action < 0 || action > 1) {
        out.observation_bits = encode_bits();
        out.reward = cfg_.failure_reward;
        out.done = true;
        out.outcome = kOutcomeLoss;
        out.invalid_action = true;
        return out;
    }

    constexpr double gravity = 9.8;
    constexpr double masscart = 1.0;
    constexpr double masspole = 0.1;
    constexpr double total_mass = masscart + masspole;
    constexpr double length = 0.5;
    constexpr double polemass_length = masspole * length;
    constexpr double force_mag = 10.0;
    constexpr double tau = 0.02;

    const double force = action == 1 ? force_mag : -force_mag;
    const double cos_theta = std::cos(theta_);
    const double sin_theta = std::sin(theta_);
    const double temp = (force + polemass_length * theta_dot_ * theta_dot_ * sin_theta) / total_mass;
    const double theta_acc =
        (gravity * sin_theta - cos_theta * temp) /
        (length * (4.0 / 3.0 - masspole * cos_theta * cos_theta / total_mass));
    const double x_acc = temp - polemass_length * theta_acc * cos_theta / total_mass;

    x_ += tau * x_dot_;
    x_dot_ += tau * x_acc;
    theta_ += tau * theta_dot_;
    theta_dot_ += tau * theta_acc;
    steps_++;

    if (is_failure()) {
        out.reward = cfg_.failure_reward;
        out.done = true;
        out.outcome = kOutcomeLoss;
    } else if (steps_ >= cfg_.max_steps) {
        out.reward = cfg_.step_reward;
        out.done = true;
        out.outcome = kOutcomeWin;
    } else {
        out.reward = cfg_.step_reward;
        out.done = false;
        out.outcome = kOutcomeOngoing;
    }

    out.observation_bits = encode_bits();
    return out;
}
