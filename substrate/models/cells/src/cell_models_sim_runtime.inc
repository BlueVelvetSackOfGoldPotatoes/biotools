bool CellSim::point_inside_membrane(const Vec3& p) const {
    int hits = 0;
    const Vec3 q = p + Vec3(0.0, 1e-9, -1e-9);
    for (const Triangle& t : state_.membrane_triangles) {
        const Vec3& a = state_.membrane_pred_positions[t.a];
        const Vec3& b = state_.membrane_pred_positions[t.b];
        const Vec3& c = state_.membrane_pred_positions[t.c];
        if (ray_intersects_triangle_x(q, a, b, c)) {
            ++hits;
        }
    }
    return (hits % 2) == 1;
}

std::size_t CellSim::nearest_triangle(const Vec3& p, Vec3& closest, Vec3& normal, Vec3& bary) const {
    std::size_t best = 0;
    double best_d2 = std::numeric_limits<double>::infinity();

    for (std::size_t ti = 0; ti < state_.membrane_triangles.size(); ++ti) {
        const Triangle& t = state_.membrane_triangles[ti];
        const Vec3& a = state_.membrane_pred_positions[t.a];
        const Vec3& b = state_.membrane_pred_positions[t.b];
        const Vec3& c = state_.membrane_pred_positions[t.c];

        Vec3 bar;
        const Vec3 cp = closest_point_on_triangle(p, a, b, c, bar);
        const double d2 = norm2(cp - p);
        if (d2 < best_d2) {
            best_d2 = d2;
            best = ti;
            closest = cp;
            bary = bar;
            normal = normalized(cross(b - a, c - a));
        }
    }

    if (dot(closest - state_.membrane_centroid, normal) < 0.0) {
        normal *= -1.0;
    }

    return best;
}

void CellSim::build_boundary_sdf() {
    auto& sdf = state_.boundary_sdf;
    sdf.valid = false;
    sdf.sdf.clear();
    sdf.nearest_tri.clear();
    sdf.nearest_normal.clear();

    if (state_.membrane_pred_positions.empty() || state_.membrane_triangles.empty()) {
        return;
    }

    Vec3 pmin = state_.membrane_pred_positions[0];
    Vec3 pmax = state_.membrane_pred_positions[0];
    for (const Vec3& p : state_.membrane_pred_positions) {
        pmin.x = std::min(pmin.x, p.x);
        pmin.y = std::min(pmin.y, p.y);
        pmin.z = std::min(pmin.z, p.z);
        pmax.x = std::max(pmax.x, p.x);
        pmax.y = std::max(pmax.y, p.y);
        pmax.z = std::max(pmax.z, p.z);
    }

    const double h = std::max(1e-6, params_.smoothing_length);
    const double cell = std::max(0.06, std::min(0.35, 2.0 * h));
    const double pad = 2.5 * h;
    pmin -= Vec3(pad, pad, pad);
    pmax += Vec3(pad, pad, pad);

    const int nx = std::max(4, static_cast<int>(std::ceil((pmax.x - pmin.x) / cell)) + 2);
    const int ny = std::max(4, static_cast<int>(std::ceil((pmax.y - pmin.y) / cell)) + 2);
    const int nz = std::max(4, static_cast<int>(std::ceil((pmax.z - pmin.z) / cell)) + 2);
    const std::size_t total = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);

    sdf.origin = pmin;
    sdf.cell_size = cell;
    sdf.nx = nx;
    sdf.ny = ny;
    sdf.nz = nz;
    sdf.sdf.assign(total, 0.0f);
    sdf.nearest_tri.assign(total, 0U);
    sdf.nearest_normal.assign(total, Vec3(1.0, 0.0, 0.0));

    auto idx_of = [&](int ix, int iy, int iz) -> std::size_t {
        return static_cast<std::size_t>(iz) * static_cast<std::size_t>(nx * ny) +
               static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(ix);
    };

    for (int iz = 0; iz < nz; ++iz) {
        for (int iy = 0; iy < ny; ++iy) {
            for (int ix = 0; ix < nx; ++ix) {
                const Vec3 c = sdf.origin + Vec3((ix + 0.5) * cell, (iy + 0.5) * cell, (iz + 0.5) * cell);

                double best_d2 = std::numeric_limits<double>::infinity();
                std::size_t best_tri = 0;
                Vec3 best_n(1.0, 0.0, 0.0);
                for (std::size_t ti = 0; ti < state_.membrane_triangles.size(); ++ti) {
                    const Triangle& t = state_.membrane_triangles[ti];
                    const Vec3& a = state_.membrane_pred_positions[t.a];
                    const Vec3& b = state_.membrane_pred_positions[t.b];
                    const Vec3& cc = state_.membrane_pred_positions[t.c];
                    Vec3 bar;
                    const Vec3 cp = closest_point_on_triangle(c, a, b, cc, bar);
                    const double d2 = norm2(cp - c);
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best_tri = ti;
                        best_n = normalized(cross(b - a, cc - a));
                    }
                }

                if (dot(c - state_.membrane_centroid, best_n) < 0.0) {
                    best_n *= -1.0;
                }
                const bool inside = point_inside_membrane(c);
                const double dist = std::sqrt(std::max(0.0, best_d2));
                const std::size_t at = idx_of(ix, iy, iz);
                sdf.sdf[at] = static_cast<float>(inside ? -dist : dist);
                sdf.nearest_tri[at] = static_cast<std::uint32_t>(best_tri);
                sdf.nearest_normal[at] = best_n;
            }
        }
    }

    sdf.valid = true;
}

bool CellSim::sample_boundary_sdf(const Vec3& p, double& signed_distance, Vec3& gradient, std::size_t& tri_id) const {
    const auto& sdf = state_.boundary_sdf;
    if (!sdf.valid || sdf.nx < 3 || sdf.ny < 3 || sdf.nz < 3) {
        return false;
    }

    const double gx = (p.x - sdf.origin.x) / sdf.cell_size - 0.5;
    const double gy = (p.y - sdf.origin.y) / sdf.cell_size - 0.5;
    const double gz = (p.z - sdf.origin.z) / sdf.cell_size - 0.5;

    const int ix = static_cast<int>(std::round(gx));
    const int iy = static_cast<int>(std::round(gy));
    const int iz = static_cast<int>(std::round(gz));
    if (ix < 1 || iy < 1 || iz < 1 || ix >= sdf.nx - 1 || iy >= sdf.ny - 1 || iz >= sdf.nz - 1) {
        return false;
    }

    auto idx_of = [&](int x, int y, int z) -> std::size_t {
        return static_cast<std::size_t>(z) * static_cast<std::size_t>(sdf.nx * sdf.ny) +
               static_cast<std::size_t>(y) * static_cast<std::size_t>(sdf.nx) +
               static_cast<std::size_t>(x);
    };

    const std::size_t c = idx_of(ix, iy, iz);
    signed_distance = static_cast<double>(sdf.sdf[c]);
    tri_id = static_cast<std::size_t>(sdf.nearest_tri[c]);

    const double sx = (static_cast<double>(sdf.sdf[idx_of(ix + 1, iy, iz)]) -
                       static_cast<double>(sdf.sdf[idx_of(ix - 1, iy, iz)])) / (2.0 * sdf.cell_size);
    const double sy = (static_cast<double>(sdf.sdf[idx_of(ix, iy + 1, iz)]) -
                       static_cast<double>(sdf.sdf[idx_of(ix, iy - 1, iz)])) / (2.0 * sdf.cell_size);
    const double sz = (static_cast<double>(sdf.sdf[idx_of(ix, iy, iz + 1)]) -
                       static_cast<double>(sdf.sdf[idx_of(ix, iy, iz - 1)])) / (2.0 * sdf.cell_size);
    gradient = Vec3(sx, sy, sz);
    if (norm2(gradient) < 1e-16) {
        gradient = sdf.nearest_normal[c];
    }
    return true;
}

bool CellSim::segment_hits_membrane(
    const Vec3& p0,
    const Vec3& p1,
    Vec3& hit,
    Vec3& normal,
    Vec3& bary,
    std::size_t& tri_id) const {
    bool found = false;
    double best_t = std::numeric_limits<double>::infinity();
    Vec3 best_bary(0.0, 0.0, 0.0);
    std::size_t best_tri = 0;

    for (std::size_t ti = 0; ti < state_.membrane_triangles.size(); ++ti) {
        const Triangle& tri = state_.membrane_triangles[ti];
        const Vec3& a = state_.membrane_pred_positions[tri.a];
        const Vec3& b = state_.membrane_pred_positions[tri.b];
        const Vec3& c = state_.membrane_pred_positions[tri.c];
        double t = 0.0;
        Vec3 bar;
        if (!segment_intersects_triangle(p0, p1, a, b, c, t, bar)) {
            continue;
        }
        if (t < best_t) {
            best_t = t;
            best_bary = bar;
            best_tri = ti;
            found = true;
        }
    }

    if (!found) {
        return false;
    }

    const Triangle& tri = state_.membrane_triangles[best_tri];
    const Vec3& a = state_.membrane_pred_positions[tri.a];
    const Vec3& b = state_.membrane_pred_positions[tri.b];
    const Vec3& c = state_.membrane_pred_positions[tri.c];
    hit = p0 + (p1 - p0) * best_t;
    normal = normalized(cross(b - a, c - a));
    if (dot(hit - state_.membrane_centroid, normal) < 0.0) {
        normal *= -1.0;
    }
    bary = best_bary;
    tri_id = best_tri;
    return true;
}

void CellSim::step_coupling(double dt) {
    state_.diagnostics.penetration_count = 0;
    state_.diagnostics.max_penetration_depth = 0.0;

    if (state_.particle_pred_positions.empty()) {
        return;
    }

    build_boundary_sdf();

    ScopedTimer timer(&state_.diagnostics.coupling_time_ms);
    for (int it = 0; it < std::max(1, params_.coupling_iterations); ++it) {
        for (std::size_t i = 0; i < state_.particle_pred_positions.size(); ++i) {
            Vec3 p = state_.particle_pred_positions[i];

            // Continuous collision detection for fast particles.
            if (params_.coupling_enable_ccd) {
                const Vec3 p_prev = state_.particle_position_prev[i];
                const double disp = norm(p - p_prev);
                if (disp > params_.ccd_displacement_threshold * params_.smoothing_length) {
                    Vec3 ccd_hit;
                    Vec3 ccd_normal;
                    Vec3 ccd_bary;
                    std::size_t ccd_tri = 0;
                    if (segment_hits_membrane(p_prev, p, ccd_hit, ccd_normal, ccd_bary, ccd_tri)) {
                        p = ccd_hit - ccd_normal * (params_.noleak_inset * params_.smoothing_length);
                        state_.particle_pred_positions[i] = p;
                    }
                }
            }

            double sd = 0.0;
            Vec3 grad(0.0, 0.0, 0.0);
            std::size_t tri_hint = 0;
            const bool have_sdf = sample_boundary_sdf(p, sd, grad, tri_hint);
            bool outside = false;
            if (have_sdf) {
                outside = (sd > 0.0);
            } else {
                const double radial = norm(p - state_.membrane_centroid);
                if (radial < 0.85 * state_.membrane_mean_radius) {
                    continue;
                }
                outside = !point_inside_membrane(p);
            }
            if (!outside) {
                continue;
            }

            Vec3 closest;
            Vec3 normal;
            Vec3 bary;
            std::size_t tri_id = tri_hint;
            if (tri_id < state_.membrane_triangles.size()) {
                const Triangle& tri = state_.membrane_triangles[tri_id];
                const Vec3& a = state_.membrane_pred_positions[tri.a];
                const Vec3& b = state_.membrane_pred_positions[tri.b];
                const Vec3& c = state_.membrane_pred_positions[tri.c];
                closest = closest_point_on_triangle(p, a, b, c, bary);
                normal = normalized(cross(b - a, c - a));
                if (dot(closest - state_.membrane_centroid, normal) < 0.0) {
                    normal *= -1.0;
                }
            } else {
                tri_id = nearest_triangle(p, closest, normal, bary);
            }
            const Triangle& tri = state_.membrane_triangles[tri_id];

            const double depth = norm(p - closest);
            state_.diagnostics.penetration_count += 1;
            state_.diagnostics.max_penetration_depth = std::max(state_.diagnostics.max_penetration_depth, depth);

            Vec3 outward = normal;
            if (have_sdf && norm2(grad) > 1e-16) {
                outward = normalized(grad);
                if (dot(outward, normal) < 0.0) {
                    outward *= -1.0;
                }
            }

            const Vec3 projected = have_sdf
                ? (p - outward * (std::max(0.0, sd) + params_.noleak_inset * params_.smoothing_length))
                : (closest - normal * (params_.noleak_inset * params_.smoothing_length));
            Vec3 corr = projected - p;
            const double max_corr = std::max(1e-8, params_.coupling_impulse_cap * params_.smoothing_length);
            const double corr_n = norm(corr);
            if (corr_n > max_corr) {
                corr *= (max_corr / corr_n);
            }
            state_.particle_pred_positions[i] = p + corr;

            // Apply a bounded opposite positional correction to membrane vertices.
            const Vec3 push = corr * (-std::max(0.0, params_.coupling_damping));

            const double wa = std::max(0.0, bary.x);
            const double wb = std::max(0.0, bary.y);
            const double wc = std::max(0.0, bary.z);
            const double wia = wa * state_.membrane_inv_masses[tri.a];
            const double wib = wb * state_.membrane_inv_masses[tri.b];
            const double wic = wc * state_.membrane_inv_masses[tri.c];
            const double wsum = std::max(1e-12, wia + wib + wic);

            state_.membrane_pred_positions[tri.a] += push * (wia / wsum);
            state_.membrane_pred_positions[tri.b] += push * (wib / wsum);
            state_.membrane_pred_positions[tri.c] += push * (wic / wsum);
        }
    }

    (void)dt;
}

void CellSim::update_homeostasis(double dt) {
    Vec3 mean_v(0.0, 0.0, 0.0);
    double mean_speed = 0.0;
    for (const Vec3& v : state_.membrane_velocities) {
        mean_v += v;
        mean_speed += norm(v);
    }
    if (!state_.membrane_velocities.empty()) {
        mean_v /= static_cast<double>(state_.membrane_velocities.size());
        mean_speed /= static_cast<double>(state_.membrane_velocities.size());
    }
    state_.diagnostics.centroid_speed = norm(mean_v);

    const double active_cost = params_.energy_cost_active * std::abs(params_.active_force) *
                               (params_.enable_active_forces ? 1.0 : 0.0);
    const double motion_cost = params_.energy_cost_motion * mean_speed;
    state_.energy += dt * (params_.energy_replenish - active_cost - motion_cost);
    state_.energy = std::max(0.0, std::min(params_.energy_max, state_.energy));

    const double v_over = std::max(0.0, state_.diagnostics.volume_drift_pct - params_.homeostasis_tol_volume_pct);
    const double a_over = std::max(0.0, state_.diagnostics.area_drift_pct - params_.homeostasis_tol_area_pct);
    const double activity_dev = std::abs(mean_speed - params_.homeostasis_target_activity);
    const double act_over = std::max(0.0, activity_dev - params_.homeostasis_tol_activity);
    const double energy_floor = 0.2 * params_.energy_max;
    const double e_over = std::max(0.0, energy_floor - state_.energy);

    state_.diagnostics.homeostasis_score =
        v_over * v_over + a_over * a_over + act_over * act_over + e_over * e_over;
    state_.diagnostics.energy = state_.energy;
}

void CellSim::step(double dt) {
    step(CellControlInput{}, dt);
}

void CellSim::step(const CellControlInput& control, double dt) {
    const double use_dt = (dt > 0.0) ? dt : params_.dt;

    ScopedTimer step_timer(&state_.diagnostics.step_time_ms);
    apply_active_behavior(control, use_dt);

    {
        ScopedTimer timer(&state_.diagnostics.predict_time_ms);
        const std::size_t n = state_.membrane_positions.size();
        for (std::size_t i = 0; i < n; ++i) {
            state_.membrane_velocities[i] *= params_.global_damping;
            state_.membrane_pred_positions[i] = state_.membrane_positions[i] + state_.membrane_velocities[i] * use_dt;
        }
        // XPBD warm-start state decay keeps convergence benefit while avoiding
        // long-run multiplier accumulation under strongly coupled constraints.
        for (double& l : state_.membrane_stretch_lambdas) {
            l *= 0.995;
        }
        state_.membrane_area_lambda *= 0.98;
        state_.membrane_volume_lambda *= 0.98;
    }

    step_fluid(use_dt);

    {
        ScopedTimer solve_timer(&state_.diagnostics.constraint_solve_time_ms);
        for (int it = 0; it < std::max(1, params_.membrane_iterations); ++it) {
            state_.diagnostics.stretch_residual = membrane::project_stretch(
                state_.membrane_pred_positions,
                state_.membrane_edges,
                state_.membrane_inv_masses,
                state_.membrane_stretch_lambdas,
                params_.stretch_compliance,
                use_dt);

            if (params_.enable_bending) {
                state_.diagnostics.bend_residual = membrane::project_bend(
                    state_.membrane_pred_positions,
                    state_.membrane_triangles,
                    state_.membrane_edges,
                    state_.membrane_inv_masses,
                    params_.bend_compliance,
                    use_dt);
            } else {
                state_.diagnostics.bend_residual = 0.0;
            }

            state_.diagnostics.area_residual = membrane::project_area(
                state_.membrane_pred_positions,
                state_.membrane_triangles,
                state_.membrane_inv_masses,
                params_.target_area,
                state_.membrane_area_lambda,
                params_.area_compliance,
                use_dt);

            state_.diagnostics.volume_residual = membrane::project_volume(
                state_.membrane_pred_positions,
                state_.membrane_triangles,
                state_.membrane_inv_masses,
                params_.target_volume,
                state_.membrane_volume_lambda,
                params_.volume_compliance,
                use_dt);
        }
    }

    step_coupling(use_dt);

    // Re-project membrane constraints after coupling because boundary reaction
    // updates can violate stretch/area/volume invariants.
    double stabilize_ms = 0.0;
    {
        ScopedTimer solve_timer(&stabilize_ms);
        const int stabilize_iters = std::max(1, params_.membrane_iterations / 3);
        for (int it = 0; it < stabilize_iters; ++it) {
            state_.diagnostics.stretch_residual = membrane::project_stretch(
                state_.membrane_pred_positions,
                state_.membrane_edges,
                state_.membrane_inv_masses,
                state_.membrane_stretch_lambdas,
                params_.stretch_compliance,
                use_dt);

            if (params_.enable_bending) {
                state_.diagnostics.bend_residual = membrane::project_bend(
                    state_.membrane_pred_positions,
                    state_.membrane_triangles,
                    state_.membrane_edges,
                    state_.membrane_inv_masses,
                    params_.bend_compliance,
                    use_dt);
            }

            state_.diagnostics.area_residual = membrane::project_area(
                state_.membrane_pred_positions,
                state_.membrane_triangles,
                state_.membrane_inv_masses,
                params_.target_area,
                state_.membrane_area_lambda,
                params_.area_compliance,
                use_dt);

            state_.diagnostics.volume_residual = membrane::project_volume(
                state_.membrane_pred_positions,
                state_.membrane_triangles,
                state_.membrane_inv_masses,
                params_.target_volume,
                state_.membrane_volume_lambda,
                params_.volume_compliance,
                use_dt);
        }
    }
    state_.diagnostics.constraint_solve_time_ms += stabilize_ms;

    // Commit membrane.
    for (std::size_t i = 0; i < state_.membrane_positions.size(); ++i) {
        state_.membrane_velocities[i] = (state_.membrane_pred_positions[i] - state_.membrane_positions[i]) / use_dt;
        state_.membrane_positions[i] = state_.membrane_pred_positions[i];
    }

    // Commit particles.
    for (std::size_t i = 0; i < state_.particle_positions.size(); ++i) {
        state_.particle_velocities[i] = (state_.particle_pred_positions[i] - state_.particle_positions[i]) / use_dt;
        state_.particle_positions[i] = state_.particle_pred_positions[i];
    }

    membrane::compute_vertex_normals(state_.membrane_positions, state_.membrane_triangles, state_.membrane_normals);
    recompute_geometry();
    update_homeostasis(use_dt);
}

void CellSim::poke_vertex(std::size_t index, const Vec3& impulse) {
    if (index >= state_.membrane_velocities.size()) {
        return;
    }
    state_.membrane_velocities[index] += impulse;
}

std::size_t CellSim::vertex_count() const {
    return state_.membrane_positions.size();
}

std::size_t CellSim::triangle_count() const {
    return state_.membrane_triangles.size();
}

std::size_t CellSim::edge_count() const {
    return state_.membrane_edges.size();
}

std::size_t CellSim::particle_count() const {
    return state_.particle_positions.size();
}

std::size_t CellSim::damage_random_vertices(double frac, unsigned int seed) {
    if (state_.membrane_positions.empty()) return 0;
    const std::size_t k = static_cast<std::size_t>(std::round(
        std::max(0.0, std::min(1.0, frac)) * state_.membrane_positions.size()));
    std::mt19937 rr(seed ? seed : rng_());
    std::vector<std::size_t> idx(state_.membrane_positions.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rr);

    for (std::size_t t = 0; t < k; ++t) {
        const std::size_t i = idx[t];
        state_.membrane_inv_masses[i] = 0.0;
        state_.membrane_velocities[i] = Vec3(0.0, 0.0, 0.0);
    }
    return k;
}

std::size_t CellSim::damage_random_particles(double frac, unsigned int seed) {
    if (state_.particle_positions.empty()) return 0;
    const std::size_t k = static_cast<std::size_t>(std::round(
        std::max(0.0, std::min(1.0, frac)) * state_.particle_positions.size()));
    if (k == 0) return 0;

    std::mt19937 rr(seed ? seed : rng_());
    std::vector<std::size_t> idx(state_.particle_positions.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rr);

    std::vector<char> dead(state_.particle_positions.size(), 0);
    for (std::size_t t = 0; t < k; ++t) dead[idx[t]] = 1;

    auto compact_vec3 = [&](std::vector<Vec3>& v) {
        std::vector<Vec3> out;
        out.reserve(v.size() - k);
        for (std::size_t i = 0; i < v.size(); ++i) if (!dead[i]) out.push_back(v[i]);
        v.swap(out);
    };
    auto compact_double = [&](std::vector<double>& v) {
        std::vector<double> out;
        out.reserve(v.size() - k);
        for (std::size_t i = 0; i < v.size(); ++i) if (!dead[i]) out.push_back(v[i]);
        v.swap(out);
    };

    compact_vec3(state_.particle_positions);
    compact_vec3(state_.particle_velocities);
    compact_vec3(state_.particle_pred_positions);
    compact_vec3(state_.particle_position_prev);
    compact_double(state_.particle_masses);
    compact_double(state_.particle_densities);
    compact_double(state_.particle_pressures);
    compact_double(state_.particle_lambdas);
    return k;
}

const SimState& CellSim::state() const {
    return state_;
}

const Diagnostics& CellSim::diagnostics() const {
    return state_.diagnostics;
}

SimView CellSim::view() const {
    return SimView{
        Span<const Vec3>(state_.membrane_positions),
        Span<const Vec3>(state_.membrane_normals),
        Span<const std::uint32_t>(state_.membrane_indices),
        Span<const Vec3>(state_.particle_positions)
    };
}

SimParams& CellSim::params() {
    return params_;
}

const SimParams& CellSim::params() const {
    return params_;
}

void CellSim::recompute_geometry() {
    state_.diagnostics.current_area = membrane::compute_area(state_.membrane_positions, state_.membrane_triangles);
    state_.diagnostics.current_volume = membrane::compute_volume(state_.membrane_positions, state_.membrane_triangles);

    const double area_target = std::max(params_.target_area, 1e-12);
    const double vol_target = std::max(params_.target_volume, 1e-12);

    state_.diagnostics.area_drift_pct =
        100.0 * std::abs(state_.diagnostics.current_area - area_target) / area_target;
    state_.diagnostics.volume_drift_pct =
        100.0 * std::abs(state_.diagnostics.current_volume - vol_target) / vol_target;

    state_.membrane_centroid = Vec3(0.0, 0.0, 0.0);
    for (const Vec3& p : state_.membrane_positions) {
        state_.membrane_centroid += p;
    }
    if (!state_.membrane_positions.empty()) {
        state_.membrane_centroid /= static_cast<double>(state_.membrane_positions.size());
    }

    double mean_r = 0.0;
    for (const Vec3& p : state_.membrane_positions) {
        mean_r += norm(p - state_.membrane_centroid);
    }
    state_.membrane_mean_radius = state_.membrane_positions.empty()
        ? 1.0
        : (mean_r / static_cast<double>(state_.membrane_positions.size()));
}

