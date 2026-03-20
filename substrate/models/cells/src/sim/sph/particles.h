#pragma once

#include "models/cells/src/core/vec3.h"

#include <vector>

namespace cells {
namespace sph {

struct Particle {
    Vec3 position;
    Vec3 velocity;
    double mass = 1.0;
    double density = 0.0;
    double pressure = 0.0;
};

struct Particles {
    std::vector<Particle> data;
};

} // namespace sph
} // namespace cells

