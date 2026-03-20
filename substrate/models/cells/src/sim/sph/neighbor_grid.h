#pragma once

#include "models/cells/src/core/vec3.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cells {
namespace sph {

struct GridCell {
    int x;
    int y;
    int z;
};

class NeighborGrid {
public:
    void clear() {
        h_ = 1.0;
        inv_h_ = 1.0;
        bucket_count_ = 0;
        sorted_indices_.clear();
        sorted_cells_.clear();
        bucket_offsets_.clear();
    }

    static GridCell cell_from_pos(const Vec3& p, double h) {
        return GridCell{
            static_cast<int>(std::floor(p.x / h)),
            static_cast<int>(std::floor(p.y / h)),
            static_cast<int>(std::floor(p.z / h))
        };
    }

    void build(const std::vector<Vec3>& positions, double h) {
        clear();
        if (positions.empty()) {
            return;
        }

        h_ = std::max(h, 1e-8);
        inv_h_ = 1.0 / h_;
        bucket_count_ = next_prime(static_cast<std::size_t>(positions.size() * 2 + 1));

        bucket_offsets_.assign(bucket_count_ + 1, 0);
        sorted_indices_.resize(positions.size());
        sorted_cells_.resize(positions.size());

        std::vector<std::size_t> bucket_id(positions.size(), 0);
        for (std::size_t i = 0; i < positions.size(); ++i) {
            const GridCell c = cell_from_pos(positions[i], h_);
            const std::size_t b = bucket_of(c);
            bucket_id[i] = b;
            bucket_offsets_[b + 1] += 1;
        }

        for (std::size_t b = 1; b < bucket_offsets_.size(); ++b) {
            bucket_offsets_[b] += bucket_offsets_[b - 1];
        }

        std::vector<std::size_t> cursor = bucket_offsets_;
        for (std::size_t i = 0; i < positions.size(); ++i) {
            const GridCell c = cell_from_pos(positions[i], h_);
            const std::size_t b = bucket_id[i];
            const std::size_t at = cursor[b]++;
            sorted_indices_[at] = static_cast<std::uint32_t>(i);
            sorted_cells_[at] = c;
        }
    }

    template <typename Fn>
    void for_neighbor_candidates(const Vec3& p, Fn&& fn) const {
        if (bucket_count_ == 0) {
            return;
        }
        const GridCell center = cell_from_pos(p, h_);
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const GridCell q{center.x + dx, center.y + dy, center.z + dz};
                    for_exact_cell(q, fn);
                }
            }
        }
    }

    double h() const { return h_; }

private:
    std::size_t bucket_of(const GridCell& c) const {
        const std::uint64_t hx = static_cast<std::uint64_t>(static_cast<std::int64_t>(c.x) * 73856093LL);
        const std::uint64_t hy = static_cast<std::uint64_t>(static_cast<std::int64_t>(c.y) * 19349663LL);
        const std::uint64_t hz = static_cast<std::uint64_t>(static_cast<std::int64_t>(c.z) * 83492791LL);
        const std::uint64_t key = hx ^ hy ^ hz;
        return static_cast<std::size_t>(key % static_cast<std::uint64_t>(bucket_count_));
    }

    template <typename Fn>
    void for_exact_cell(const GridCell& q, Fn&& fn) const {
        const std::size_t b = bucket_of(q);
        const std::size_t begin = bucket_offsets_[b];
        const std::size_t end = bucket_offsets_[b + 1];
        for (std::size_t k = begin; k < end; ++k) {
            const GridCell& c = sorted_cells_[k];
            if (c.x == q.x && c.y == q.y && c.z == q.z) {
                fn(sorted_indices_[k]);
            }
        }
    }

    static bool is_prime(std::size_t x) {
        if (x < 2) return false;
        if (x % 2 == 0) return x == 2;
        for (std::size_t d = 3; d * d <= x; d += 2) {
            if (x % d == 0) return false;
        }
        return true;
    }

    static std::size_t next_prime(std::size_t x) {
        while (!is_prime(x)) {
            ++x;
        }
        return x;
    }

    double h_ = 1.0;
    double inv_h_ = 1.0;
    std::size_t bucket_count_ = 0;

    std::vector<std::uint32_t> sorted_indices_;
    std::vector<GridCell> sorted_cells_;
    std::vector<std::size_t> bucket_offsets_;
};

} // namespace sph
} // namespace cells

