#pragma once

#include <algorithm>

namespace benchmark {

constexpr int kEmpty = 0;
constexpr int kAgent = 1;
constexpr int kOpponent = 2;

constexpr int kOutcomeWin = 1;
constexpr int kOutcomeDraw = 0;
constexpr int kOutcomeLoss = -1;
constexpr int kOutcomeOngoing = 2;

inline double clamp01(double v) { return std::clamp(v, 0.0, 1.0); }

} // namespace benchmark
