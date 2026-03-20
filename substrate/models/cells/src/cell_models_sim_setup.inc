namespace {

inline Vec3 random_unit_vec3(std::mt19937& rng) {
    std::uniform_real_distribution<double> ud(-1.0, 1.0);
    for (int tries = 0; tries < 64; ++tries) {
        Vec3 v(ud(rng), ud(rng), ud(rng));
        const double n2 = norm2(v);
        if (n2 > 1e-12 && n2 <= 1.0) {
            return v / std::sqrt(n2);
        }
    }
    return Vec3(1.0, 0.0, 0.0);
}

inline bool ray_intersects_triangle_x(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
    // Moller-Trumbore with +X ray.
    const Vec3 dir(1.0, 0.0, 0.0);
    const Vec3 e1 = b - a;
    const Vec3 e2 = c - a;
    const Vec3 h = cross(dir, e2);
    const double det = dot(e1, h);
    if (std::abs(det) < 1e-12) return false;
    const double inv_det = 1.0 / det;
    const Vec3 s = p - a;
    const double u = inv_det * dot(s, h);
    if (u < 0.0 || u > 1.0) return false;
    const Vec3 q = cross(s, e1);
    const double v = inv_det * dot(dir, q);
    if (v < 0.0 || (u + v) > 1.0) return false;
    const double t = inv_det * dot(e2, q);
    return t > 1e-9;
}

inline bool segment_intersects_triangle(
    const Vec3& p0,
    const Vec3& p1,
    const Vec3& a,
    const Vec3& b,
    const Vec3& c,
    double& t_out,
    Vec3& bary_out) {
    const Vec3 dir = p1 - p0;
    const Vec3 e1 = b - a;
    const Vec3 e2 = c - a;
    const Vec3 pvec = cross(dir, e2);
    const double det = dot(e1, pvec);
    if (std::abs(det) < 1e-12) {
        return false;
    }
    const double inv_det = 1.0 / det;
    const Vec3 tvec = p0 - a;
    const double u = dot(tvec, pvec) * inv_det;
    if (u < 0.0 || u > 1.0) {
        return false;
    }
    const Vec3 qvec = cross(tvec, e1);
    const double v = dot(dir, qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0) {
        return false;
    }
    const double t = dot(e2, qvec) * inv_det;
    if (t < 0.0 || t > 1.0) {
        return false;
    }
    t_out = t;
    bary_out = Vec3(1.0 - u - v, u, v);
    return true;
}

inline Vec3 closest_point_on_triangle(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c, Vec3& bary) {
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    const Vec3 ap = p - a;
    const double d1 = dot(ab, ap);
    const double d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        bary = Vec3(1.0, 0.0, 0.0);
        return a;
    }

    const Vec3 bp = p - b;
    const double d3 = dot(ab, bp);
    const double d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        bary = Vec3(0.0, 1.0, 0.0);
        return b;
    }

    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v = d1 / (d1 - d3);
        bary = Vec3(1.0 - v, v, 0.0);
        return a + ab * v;
    }

    const Vec3 cp = p - c;
    const double d5 = dot(ab, cp);
    const double d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        bary = Vec3(0.0, 0.0, 1.0);
        return c;
    }

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = d2 / (d2 - d6);
        bary = Vec3(1.0 - w, 0.0, w);
        return a + ac * w;
    }

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const Vec3 bc = c - b;
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        bary = Vec3(0.0, 1.0 - w, w);
        return b + bc * w;
    }

    const double denom = 1.0 / (va + vb + vc);
    const double v = vb * denom;
    const double w = vc * denom;
    bary = Vec3(1.0 - v - w, v, w);
    return a + ab * v + ac * w;
}

} // namespace

SimParams SimParams::soft_blob() {
    SimParams p;
    p.stretch_compliance = 8e-6;
    p.area_compliance = 4e-4;
    p.volume_compliance = 2e-7;
    p.global_damping = 0.985;
    p.enable_bending = false;
    return p;
}

SimParams SimParams::stiff_blob() {
    SimParams p;
    p.stretch_compliance = 3e-7;
    p.area_compliance = 8e-5;
    p.volume_compliance = 5e-8;
    p.bend_compliance = 3e-4;
    p.enable_bending = true;
    p.global_damping = 0.975;
    return p;
}

SimParams SimParams::watery_cytoplasm() {
    SimParams p;
    p.viscosity = 0.005;
    p.smoothing_length = 0.14;
    p.fluid_iterations = 3;
    return p;
}

SimParams SimParams::viscous_cytoplasm() {
    SimParams p;
    p.viscosity = 0.08;
    p.smoothing_length = 0.12;
    p.fluid_iterations = 5;
    return p;
}

SimParams SimParams::preset(const std::string& name) {
    if (name == "soft" || name == "soft_blob") return soft_blob();
    if (name == "stiff" || name == "stiff_blob") return stiff_blob();
    if (name == "watery") return watery_cytoplasm();
    if (name == "viscous") return viscous_cytoplasm();
    return SimParams{};
}

CellSim::CellSim(const CellInit& init)
    : params_(init.params), rng_(init.seed) {
    membrane::generate_icosphere(
        init.icosphere_subdivisions,
        init.radius,
        state_.membrane_positions,
        state_.membrane_triangles);

    state_.membrane_velocities.assign(state_.membrane_positions.size(), Vec3(0.0, 0.0, 0.0));
    state_.membrane_pred_positions = state_.membrane_positions;

    membrane::build_edges(state_.membrane_positions, state_.membrane_triangles, state_.membrane_edges);
    membrane::flatten_indices(state_.membrane_triangles, state_.membrane_indices);
    membrane::compute_vertex_normals(state_.membrane_positions, state_.membrane_triangles, state_.membrane_normals);
    state_.membrane_stretch_lambdas.assign(state_.membrane_edges.size(), 0.0);
    state_.membrane_area_lambda = 0.0;
    state_.membrane_volume_lambda = 0.0;

    const double thickness = std::max(1e-6, params_.membrane_thickness * std::max(init.radius, 1e-6));
    membrane::compute_vertex_inv_masses(
        state_.membrane_positions,
        state_.membrane_triangles,
        params_.membrane_surface_density,
        thickness,
        state_.membrane_inv_masses);

    const double area0 = membrane::compute_area(state_.membrane_positions, state_.membrane_triangles);
    const double volume0 = membrane::compute_volume(state_.membrane_positions, state_.membrane_triangles);

    if (params_.target_area <= 0.0) {
        params_.target_area = area0;
    }
    if (params_.target_volume <= 0.0) {
        params_.target_volume = volume0;
    }

    state_.energy = params_.energy_max;
    state_.diagnostics.energy = state_.energy;
    init_particles(init.fluid_particles, init.radius * 0.92, init.seed + 1U);

    recompute_geometry();
}

void CellSim::init_particles(std::size_t n, double radius, unsigned int seed) {
    state_.particle_positions.clear();
    state_.particle_velocities.clear();
    state_.particle_pred_positions.clear();
    state_.particle_position_prev.clear();
    state_.particle_masses.clear();
    state_.particle_densities.clear();
    state_.particle_pressures.clear();
    state_.particle_lambdas.clear();

    if (n == 0) return;

    const double sphere_vol = (4.0 / 3.0) * sph::kPi * radius * radius * radius;
    const double spacing = std::cbrt(std::max(sphere_vol / static_cast<double>(n), 1e-9));

    state_.particle_positions.reserve(n);
    for (double z = -radius; z <= radius && state_.particle_positions.size() < n; z += spacing) {
        for (double y = -radius; y <= radius && state_.particle_positions.size() < n; y += spacing) {
            for (double x = -radius; x <= radius && state_.particle_positions.size() < n; x += spacing) {
                const Vec3 p(x, y, z);
                if (norm2(p) <= radius * radius) {
                    state_.particle_positions.push_back(p);
                }
            }
        }
    }

    if (state_.particle_positions.size() < n) {
        std::mt19937 rr(seed);
        std::uniform_real_distribution<double> ud(-radius, radius);
        while (state_.particle_positions.size() < n) {
            Vec3 p(ud(rr), ud(rr), ud(rr));
            if (norm2(p) <= radius * radius) {
                state_.particle_positions.push_back(p);
            }
        }
    }

    const std::size_t m = state_.particle_positions.size();
    state_.particle_velocities.assign(m, Vec3(0.0, 0.0, 0.0));
    state_.particle_pred_positions = state_.particle_positions;
    state_.particle_position_prev = state_.particle_positions;

    const double total_mass = params_.rest_density * sphere_vol;
    const double pmass = (m == 0) ? 0.0 : (total_mass / static_cast<double>(m));

    state_.particle_masses.assign(m, std::max(1e-8, (params_.fluid_particle_mass > 0.0 ? params_.fluid_particle_mass : pmass)));
    state_.particle_densities.assign(m, params_.rest_density);
    state_.particle_pressures.assign(m, 0.0);
    state_.particle_lambdas.assign(m, 0.0);
}

void CellSim::apply_active_behavior(const CellControlInput& control, double dt) {
    if (params_.enable_active_forces) {
        const Vec3 noise = random_unit_vec3(rng_) * params_.polarity_noise;
        const Vec3 chemo = Vec3(params_.polarity_chemo, 0.0, 0.0) + control.chemo_gradient;
        const double keep = std::max(0.0, std::min(1.0, params_.polarity_persistence));
        Vec3 next_polarity = state_.polarity * keep + noise + chemo * (1.0 - keep);
        if (norm2(control.polarity_hint) > 1e-12) {
            const double blend = std::max(0.0, std::min(1.0, control.polarity_blend));
            next_polarity = next_polarity * (1.0 - blend) + normalized(control.polarity_hint) * blend;
        }
        state_.polarity = normalized(next_polarity);
        if (norm2(state_.polarity) < 1e-12) {
            state_.polarity = Vec3(1.0, 0.0, 0.0);
        }

        const double active_force = params_.active_force * std::max(0.0, control.active_force_scale);
        for (std::size_t i = 0; i < state_.membrane_velocities.size(); ++i) {
            const double align = dot(state_.polarity, state_.membrane_normals[i]);
            const Vec3 force = state_.membrane_normals[i] * (active_force * align * state_.energy);
            state_.membrane_velocities[i] += force * (dt * state_.membrane_inv_masses[i]);
        }
    }

    if (norm2(control.body_force) > 0.0) {
        for (std::size_t i = 0; i < state_.membrane_velocities.size(); ++i) {
            state_.membrane_velocities[i] += control.body_force * (dt * state_.membrane_inv_masses[i]);
        }
    }

    const double friction = params_.substrate_friction * std::max(0.0, control.substrate_friction_scale);
    if (friction > 0.0) {
        for (std::size_t i = 0; i < state_.membrane_velocities.size(); ++i) {
            if (state_.membrane_positions[i].z <= params_.substrate_z + params_.substrate_band) {
                state_.membrane_velocities[i].x *= std::max(0.0, 1.0 - friction * dt);
                state_.membrane_velocities[i].y *= std::max(0.0, 1.0 - friction * dt);
            }
        }
    }
}

void CellSim::step_fluid(double dt) {
    if (state_.particle_positions.empty()) {
        state_.diagnostics.fluid_density_error_avg = 0.0;
        state_.diagnostics.fluid_density_error_max = 0.0;
        return;
    }

    // Predict.
    const double max_disp = std::max(1e-8, 0.75 * params_.smoothing_length);
    for (std::size_t i = 0; i < state_.particle_positions.size(); ++i) {
        state_.particle_position_prev[i] = state_.particle_positions[i];
        state_.particle_velocities[i].z += params_.gravity_z * dt;
        state_.particle_velocities[i] *= params_.global_damping;
        Vec3 disp = state_.particle_velocities[i] * dt;
        const double dn = norm(disp);
        if (dn > max_disp) {
            disp *= (max_disp / dn);
        }
        state_.particle_pred_positions[i] = state_.particle_positions[i] + disp;
    }

    {
        ScopedTimer timer(&state_.diagnostics.neighbor_build_time_ms);
        state_.particle_grid.build(state_.particle_pred_positions, params_.smoothing_length);
    }

    std::vector<Vec3> delta(state_.particle_positions.size(), Vec3(0.0, 0.0, 0.0));
    const double rho0 = std::max(1e-8, params_.rest_density);

    ScopedTimer timer(&state_.diagnostics.fluid_solve_time_ms);
    for (int iter = 0; iter < std::max(1, params_.fluid_iterations); ++iter) {
        // Density.
        for (std::size_t i = 0; i < state_.particle_pred_positions.size(); ++i) {
            const Vec3 pi = state_.particle_pred_positions[i];
            double rho = 0.0;
            state_.particle_grid.for_neighbor_candidates(pi, [&](std::uint32_t j) {
                const Vec3 rij = pi - state_.particle_pred_positions[j];
                const double r = norm(rij);
                if (r <= params_.smoothing_length) {
                    rho += state_.particle_masses[j] * sph::poly6(r, params_.smoothing_length);
                }
            });
            state_.particle_densities[i] = std::max(1e-8, rho);
        }

        // Lambdas.
        for (std::size_t i = 0; i < state_.particle_pred_positions.size(); ++i) {
            const Vec3 pi = state_.particle_pred_positions[i];
            const double c = state_.particle_densities[i] / rho0 - 1.0;

            Vec3 grad_i(0.0, 0.0, 0.0);
            double sum_grad2 = 0.0;
            state_.particle_grid.for_neighbor_candidates(pi, [&](std::uint32_t j) {
                if (j == i) return;
                const Vec3 rij = pi - state_.particle_pred_positions[j];
                const Vec3 grad = sph::spiky_gradient(rij, params_.smoothing_length) * (state_.particle_masses[j] / rho0);
                sum_grad2 += norm2(grad);
                grad_i += grad;
            });
            sum_grad2 += norm2(grad_i);
            state_.particle_lambdas[i] = -c / (sum_grad2 + params_.pbf_epsilon);
        }

        // Position deltas.
        std::fill(delta.begin(), delta.end(), Vec3(0.0, 0.0, 0.0));
        for (std::size_t i = 0; i < state_.particle_pred_positions.size(); ++i) {
            const Vec3 pi = state_.particle_pred_positions[i];
            Vec3 d(0.0, 0.0, 0.0);
            state_.particle_grid.for_neighbor_candidates(pi, [&](std::uint32_t j) {
                if (j == i) return;
                const Vec3 rij = pi - state_.particle_pred_positions[j];
                const Vec3 grad = sph::spiky_gradient(rij, params_.smoothing_length);
                d += grad * ((state_.particle_lambdas[i] + state_.particle_lambdas[j]) * (state_.particle_masses[j] / rho0));
            });
            delta[i] = d;
        }

        for (std::size_t i = 0; i < state_.particle_pred_positions.size(); ++i) {
            state_.particle_pred_positions[i] += delta[i];
        }

        state_.particle_grid.build(state_.particle_pred_positions, params_.smoothing_length);
    }

    // Density diagnostics.
    double err_sum = 0.0;
    double err_max = 0.0;
    for (double rho : state_.particle_densities) {
        const double e = std::abs(rho - rho0) / rho0;
        err_sum += e;
        err_max = std::max(err_max, e);
    }
    state_.diagnostics.fluid_density_error_avg = err_sum / static_cast<double>(state_.particle_densities.size());
    state_.diagnostics.fluid_density_error_max = err_max;

    // Velocity + viscosity (XSPH style smoothing).
    std::vector<Vec3> new_vel = state_.particle_velocities;
    for (std::size_t i = 0; i < state_.particle_pred_positions.size(); ++i) {
        new_vel[i] = (state_.particle_pred_positions[i] - state_.particle_positions[i]) / dt;
    }

    if (params_.viscosity > 0.0) {
        for (std::size_t i = 0; i < state_.particle_pred_positions.size(); ++i) {
            Vec3 visc(0.0, 0.0, 0.0);
            const Vec3 pi = state_.particle_pred_positions[i];
            state_.particle_grid.for_neighbor_candidates(pi, [&](std::uint32_t j) {
                if (j == i) return;
                const Vec3 rij = pi - state_.particle_pred_positions[j];
                const double r = norm(rij);
                if (r > params_.smoothing_length) return;
                const double lap = sph::viscosity_laplacian(r, params_.smoothing_length);
                const double rhoj = std::max(1e-8, state_.particle_densities[j]);
                visc += (new_vel[j] - new_vel[i]) * (state_.particle_masses[j] / rhoj * lap);
            });
            new_vel[i] += visc * (params_.viscosity * dt);
        }
    }

    state_.particle_velocities.swap(new_vel);
}

