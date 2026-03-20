#pragma once

#include <chrono>

namespace cells {

class ScopedTimer {
public:
    explicit ScopedTimer(double* out_ms)
        : out_ms_(out_ms), start_(Clock::now()) {}

    ~ScopedTimer() {
        if (out_ms_ == nullptr) {
            return;
        }
        const auto end = Clock::now();
        const auto dur = std::chrono::duration<double, std::milli>(end - start_);
        *out_ms_ = dur.count();
    }

private:
    using Clock = std::chrono::high_resolution_clock;

    double* out_ms_;
    Clock::time_point start_;
};

} // namespace cells

