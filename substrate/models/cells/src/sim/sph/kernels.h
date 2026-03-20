#pragma once

#include "models/cells/src/core/vec3.h"

#include <cmath>

namespace cells {
namespace sph {

constexpr double kPi = 3.14159265358979323846;

inline double poly6(double r, double h) {
    if (r < 0.0 || r > h) {
        return 0.0;
    }
    const double h2 = h * h;
    const double t = h2 - r * r;
    const double coeff = 315.0 / (64.0 * kPi * std::pow(h, 9));
    return coeff * t * t * t;
}

inline double viscosity_laplacian(double r, double h) {
    if (r < 0.0 || r > h) {
        return 0.0;
    }
    const double coeff = 45.0 / (kPi * std::pow(h, 6));
    return coeff * (h - r);
}

inline Vec3 spiky_gradient(const Vec3& rij, double h) {
    const double r = norm(rij);
    if (r <= 1e-12 || r > h) {
        return Vec3(0.0, 0.0, 0.0);
    }
    const double coeff = -45.0 / (kPi * std::pow(h, 6));
    const double s = coeff * (h - r) * (h - r) / r;
    return rij * s;
}

} // namespace sph
} // namespace cells
