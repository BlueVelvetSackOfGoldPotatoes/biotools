#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace benchmark {

// Generic step payload for discrete-action benchmark environments.
struct DiscreteStepResult {
    std::vector<uint8_t> observation_bits;
    double reward = 0.0;
    bool done = false;

    // Convention: >0 win/success, 0 draw/neutral, <0 loss/failure.
    // Environments that do not expose terminal classes can keep this at 0.
    int outcome = 0;

    // Optional counters/flags for diagnostics.
    int plies = 0;
    bool invalid_action = false;
};

// Stable contract between benchmark environments and generic runtimes.
class DiscreteEnv {
public:
    virtual ~DiscreteEnv() = default;

    virtual std::string benchmark_id() const = 0;
    virtual std::string benchmark_name() const = 0;
    virtual std::string task_type() const = 0;

    virtual std::size_t observation_bits() const = 0;
    virtual int num_actions() const = 0;

    virtual std::vector<uint8_t> reset(std::mt19937& rng) = 0;
    virtual std::vector<int> legal_actions() const = 0;
    virtual DiscreteStepResult step(int action, std::mt19937& rng) = 0;

    // Optional teacher policy action for supervised diagnostics/training.
    // Return -1 when unavailable.
    virtual int teacher_action() const { return -1; }

    // Optional tactical signals for turn-based board/control tasks.
    // Return -1 when unavailable.
    // Immediate win: action that wins the game now (for the learning agent).
    virtual int tactical_immediate_win_action() const { return -1; }
    // Immediate block: action that blocks an opponent immediate win.
    virtual int tactical_immediate_block_action() const { return -1; }

    // Optional action-owner metadata for specialist routing (e.g. per-piece
    // control policies). Default means "single owner group".
    virtual int num_action_owner_groups() const { return 1; }
    // Return one owner-group index per action id in [0, num_actions).
    // Use -1 for actions without a meaningful owner in the current state.
    virtual std::vector<int> action_owner_groups() const { return {}; }
    virtual std::vector<std::string> action_owner_group_names() const { return {}; }
};

} // namespace benchmark
