namespace membrane {
namespace {

constexpr double kPi = 3.14159265358979323846;

inline std::uint64_t edge_key(std::uint32_t i, std::uint32_t j) {
    const std::uint32_t a = std::min(i, j);
    const std::uint32_t b = std::max(i, j);
    return (static_cast<std::uint64_t>(a) << 32U) | static_cast<std::uint64_t>(b);
}

struct MidpointCache {
    std::unordered_map<std::uint64_t, std::uint32_t> idx;
};

inline double signed_volume(const std::vector<Vec3>& vertices, const std::vector<Triangle>& triangles) {
    double v = 0.0;
    for (const Triangle& tri : triangles) {
        const Vec3& a = vertices[tri.a];
        const Vec3& b = vertices[tri.b];
        const Vec3& c = vertices[tri.c];
        v += dot(a, cross(b, c)) / 6.0;
    }
    return v;
}

inline std::array<std::uint32_t, 3> tri_vertices(const Triangle& t) {
    return {t.a, t.b, t.c};
}

} // namespace

void generate_icosphere(
    int subdivisions,
    double radius,
    std::vector<Vec3>& out_vertices,
    std::vector<Triangle>& out_triangles) {
    const double t = (1.0 + std::sqrt(5.0)) * 0.5;

    out_vertices = {
        Vec3(-1,  t,  0), Vec3( 1,  t,  0), Vec3(-1, -t,  0), Vec3( 1, -t,  0),
        Vec3( 0, -1,  t), Vec3( 0,  1,  t), Vec3( 0, -1, -t), Vec3( 0,  1, -t),
        Vec3( t,  0, -1), Vec3( t,  0,  1), Vec3(-t,  0, -1), Vec3(-t,  0,  1)
    };

    for (Vec3& v : out_vertices) {
        v = normalized(v) * radius;
    }

    out_triangles = {
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
        {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
        {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
    };

    auto midpoint = [&](std::uint32_t i, std::uint32_t j, MidpointCache& cache) -> std::uint32_t {
        const std::uint64_t key = edge_key(i, j);
        auto it = cache.idx.find(key);
        if (it != cache.idx.end()) {
            return it->second;
        }
        const Vec3 m = normalized((out_vertices[i] + out_vertices[j]) * 0.5) * radius;
        const std::uint32_t idx = static_cast<std::uint32_t>(out_vertices.size());
        out_vertices.push_back(m);
        cache.idx.emplace(key, idx);
        return idx;
    };

    for (int s = 0; s < subdivisions; ++s) {
        MidpointCache cache;
        std::vector<Triangle> next;
        next.reserve(out_triangles.size() * 4);

        for (const Triangle& tri : out_triangles) {
            const std::uint32_t a = tri.a;
            const std::uint32_t b = tri.b;
            const std::uint32_t c = tri.c;

            const std::uint32_t ab = midpoint(a, b, cache);
            const std::uint32_t bc = midpoint(b, c, cache);
            const std::uint32_t ca = midpoint(c, a, cache);

            next.push_back({a, ab, ca});
            next.push_back({b, bc, ab});
            next.push_back({c, ca, bc});
            next.push_back({ab, bc, ca});
        }
        out_triangles.swap(next);
    }
}

void build_edges(
    const std::vector<Vec3>& vertices,
    const std::vector<Triangle>& triangles,
    std::vector<Edge>& out_edges) {
    struct EdgeBuild {
        std::uint32_t i = 0;
        std::uint32_t j = 0;
        std::uint32_t opp0 = 0;
        std::uint32_t opp1 = 0;
        std::int32_t tri0 = -1;
        std::int32_t tri1 = -1;
        double rest_length = 0.0;
        double rest_dihedral_cos = 1.0;
    };

    std::map<std::uint64_t, EdgeBuild> ordered;

    auto add_edge = [&](std::uint32_t i, std::uint32_t j, std::uint32_t opp, std::int32_t tri_id) {
        const std::uint64_t key = edge_key(i, j);
        auto it = ordered.find(key);
        if (it == ordered.end()) {
            EdgeBuild eb;
            eb.i = std::min(i, j);
            eb.j = std::max(i, j);
            eb.opp0 = opp;
            eb.tri0 = tri_id;
            eb.rest_length = norm(vertices[eb.i] - vertices[eb.j]);
            ordered.emplace(key, eb);
        } else {
            it->second.opp1 = opp;
            it->second.tri1 = tri_id;
        }
    };

    for (std::size_t ti = 0; ti < triangles.size(); ++ti) {
        const Triangle& tri = triangles[ti];
        add_edge(tri.a, tri.b, tri.c, static_cast<std::int32_t>(ti));
        add_edge(tri.b, tri.c, tri.a, static_cast<std::int32_t>(ti));
        add_edge(tri.c, tri.a, tri.b, static_cast<std::int32_t>(ti));
    }

    auto tri_normal = [&](const Triangle& t) {
        const Vec3& a = vertices[t.a];
        const Vec3& b = vertices[t.b];
        const Vec3& c = vertices[t.c];
        return normalized(cross(b - a, c - a));
    };

    out_edges.clear();
    out_edges.reserve(ordered.size());
    for (const auto& kv : ordered) {
        const EdgeBuild& eb = kv.second;
        Edge e;
        e.i = eb.i;
        e.j = eb.j;
        e.rest_length = eb.rest_length;
        e.tri0 = eb.tri0;
        e.tri1 = eb.tri1;
        e.opp0 = eb.opp0;
        e.opp1 = eb.opp1;
        if (eb.tri0 >= 0 && eb.tri1 >= 0) {
            const Vec3 n0 = tri_normal(triangles[static_cast<std::size_t>(eb.tri0)]);
            const Vec3 n1 = tri_normal(triangles[static_cast<std::size_t>(eb.tri1)]);
            e.rest_dihedral_cos = std::max(-1.0, std::min(1.0, dot(n0, n1)));
        } else {
            e.rest_dihedral_cos = 1.0;
        }
        out_edges.push_back(e);
    }
}

void flatten_indices(
    const std::vector<Triangle>& triangles,
    std::vector<std::uint32_t>& out_indices) {
    out_indices.clear();
    out_indices.reserve(triangles.size() * 3);
    for (const Triangle& tri : triangles) {
        out_indices.push_back(tri.a);
        out_indices.push_back(tri.b);
        out_indices.push_back(tri.c);
    }
}

void compute_vertex_normals(
    const std::vector<Vec3>& vertices,
    const std::vector<Triangle>& triangles,
    std::vector<Vec3>& out_normals) {
    out_normals.assign(vertices.size(), Vec3(0.0, 0.0, 0.0));
    for (const Triangle& tri : triangles) {
        const Vec3& a = vertices[tri.a];
        const Vec3& b = vertices[tri.b];
        const Vec3& c = vertices[tri.c];
        const Vec3 n = cross(b - a, c - a);
        out_normals[tri.a] += n;
        out_normals[tri.b] += n;
        out_normals[tri.c] += n;
    }
    for (Vec3& n : out_normals) {
        n = normalized(n);
    }
}

double compute_area(
    const std::vector<Vec3>& vertices,
    const std::vector<Triangle>& triangles) {
    double area = 0.0;
    for (const Triangle& tri : triangles) {
        const Vec3& a = vertices[tri.a];
        const Vec3& b = vertices[tri.b];
        const Vec3& c = vertices[tri.c];
        area += 0.5 * norm(cross(b - a, c - a));
    }
    return area;
}

double compute_volume(
    const std::vector<Vec3>& vertices,
    const std::vector<Triangle>& triangles) {
    return std::abs(signed_volume(vertices, triangles));
}

void compute_vertex_inv_masses(
    const std::vector<Vec3>& vertices,
    const std::vector<Triangle>& triangles,
    double surface_density,
    double thickness,
    std::vector<double>& out_inv_masses) {
    std::vector<double> masses(vertices.size(), 0.0);
    for (const Triangle& tri : triangles) {
        const Vec3& a = vertices[tri.a];
        const Vec3& b = vertices[tri.b];
        const Vec3& c = vertices[tri.c];
        const double area = 0.5 * norm(cross(b - a, c - a));
        const double tri_mass = surface_density * area * thickness;
        const double share = tri_mass / 3.0;
        masses[tri.a] += share;
        masses[tri.b] += share;
        masses[tri.c] += share;
    }

    out_inv_masses.resize(masses.size());
    for (std::size_t i = 0; i < masses.size(); ++i) {
        const double m = std::max(masses[i], 1e-9);
        out_inv_masses[i] = 1.0 / m;
    }
}

double project_stretch(
    std::vector<Vec3>& predicted_positions,
    const std::vector<Edge>& edges,
    const std::vector<double>& inv_masses,
    std::vector<double>& lambdas,
    double compliance,
    double dt) {
    const double dt2 = dt * dt;
    const double alpha = compliance / std::max(dt2, 1e-12);

    for (std::size_t eidx = 0; eidx < edges.size(); ++eidx) {
        const Edge& edge = edges[eidx];
        const std::size_t i = edge.i;
        const std::size_t j = edge.j;

        const Vec3 d = predicted_positions[i] - predicted_positions[j];
        const double dist = norm(d);
        if (dist < 1e-12) {
            continue;
        }

        const double c = dist - edge.rest_length;
        const double wi = inv_masses[i];
        const double wj = inv_masses[j];
        const double denom = wi + wj + alpha;
        if (denom < 1e-12) {
            continue;
        }

        const double lambda_prev = (eidx < lambdas.size()) ? lambdas[eidx] : 0.0;
        const double dl = (-c - alpha * lambda_prev) / denom;
        const Vec3 n = d / dist;
        predicted_positions[i] += n * (dl * wi);
        predicted_positions[j] -= n * (dl * wj);
        if (eidx < lambdas.size()) {
            lambdas[eidx] = lambda_prev + dl;
        }
    }

    double residual = 0.0;
    for (const Edge& edge : edges) {
        const double dist = norm(predicted_positions[edge.i] - predicted_positions[edge.j]);
        residual += std::abs(dist - edge.rest_length);
    }

    return edges.empty() ? 0.0 : residual / static_cast<double>(edges.size());
}

double project_area(
    std::vector<Vec3>& predicted_positions,
    const std::vector<Triangle>& triangles,
    const std::vector<double>& inv_masses,
    double target_area,
    double& lambda,
    double compliance,
    double dt) {
    const double current_area = compute_area(predicted_positions, triangles);
    const double c = current_area - target_area;

    std::vector<Vec3> grad(predicted_positions.size(), Vec3(0.0, 0.0, 0.0));
    for (const Triangle& tri : triangles) {
        const Vec3& a = predicted_positions[tri.a];
        const Vec3& b = predicted_positions[tri.b];
        const Vec3& cc = predicted_positions[tri.c];
        const Vec3 n = cross(b - a, cc - a);
        const double nlen = norm(n);
        if (nlen < 1e-12) {
            continue;
        }
        const Vec3 nh = n / nlen;

        grad[tri.a] += 0.5 * cross(nh, cc - b);
        grad[tri.b] += 0.5 * cross(nh, a - cc);
        grad[tri.c] += 0.5 * cross(nh, b - a);
    }

    double denom = 0.0;
    for (std::size_t i = 0; i < grad.size(); ++i) {
        denom += inv_masses[i] * norm2(grad[i]);
    }
    const double dt2 = dt * dt;
    const double alpha = compliance / std::max(dt2, 1e-12);
    denom += alpha;

    if (denom > 1e-12) {
        const double s = (-c - alpha * lambda) / denom;
        for (std::size_t i = 0; i < predicted_positions.size(); ++i) {
            predicted_positions[i] += grad[i] * (s * inv_masses[i]);
        }
        lambda += s;
    }

    const double after = compute_area(predicted_positions, triangles);
    return std::abs(after - target_area);
}

double project_volume(
    std::vector<Vec3>& predicted_positions,
    const std::vector<Triangle>& triangles,
    const std::vector<double>& inv_masses,
    double target_volume,
    double& lambda,
    double compliance,
    double dt) {
    const double current_signed = signed_volume(predicted_positions, triangles);
    const double sign = (current_signed >= 0.0) ? 1.0 : -1.0;
    const double current_volume = std::abs(current_signed);
    const double c = current_volume - target_volume;

    std::vector<Vec3> grad(predicted_positions.size(), Vec3(0.0, 0.0, 0.0));
    for (const Triangle& tri : triangles) {
        const Vec3& a = predicted_positions[tri.a];
        const Vec3& b = predicted_positions[tri.b];
        const Vec3& cc = predicted_positions[tri.c];
        grad[tri.a] += sign * cross(b, cc) / 6.0;
        grad[tri.b] += sign * cross(cc, a) / 6.0;
        grad[tri.c] += sign * cross(a, b) / 6.0;
    }

    double denom = 0.0;
    for (std::size_t i = 0; i < grad.size(); ++i) {
        denom += inv_masses[i] * norm2(grad[i]);
    }
    const double dt2 = dt * dt;
    const double alpha = compliance / std::max(dt2, 1e-12);
    denom += alpha;

    if (denom > 1e-12) {
        const double s = (-c - alpha * lambda) / denom;
        for (std::size_t i = 0; i < predicted_positions.size(); ++i) {
            predicted_positions[i] += grad[i] * (s * inv_masses[i]);
        }
        lambda += s;
    }

    const double after = compute_volume(predicted_positions, triangles);
    return std::abs(after - target_volume);
}

double project_bend(
    std::vector<Vec3>& predicted_positions,
    const std::vector<Triangle>& triangles,
    const std::vector<Edge>& edges,
    const std::vector<double>& inv_masses,
    double compliance,
    double dt) {
    const double dt2 = dt * dt;
    const double alpha = compliance / std::max(dt2, 1e-12);
    const double strength = 0.05;
    double residual = 0.0;
    std::size_t n_used = 0;

    for (const Edge& e : edges) {
        if (e.tri0 < 0 || e.tri1 < 0) {
            continue;
        }
        const Triangle& t0 = triangles[static_cast<std::size_t>(e.tri0)];
        const Triangle& t1 = triangles[static_cast<std::size_t>(e.tri1)];

        const Vec3 n0 = normalized(cross(
            predicted_positions[t0.b] - predicted_positions[t0.a],
            predicted_positions[t0.c] - predicted_positions[t0.a]));
        const Vec3 n1 = normalized(cross(
            predicted_positions[t1.b] - predicted_positions[t1.a],
            predicted_positions[t1.c] - predicted_positions[t1.a]));

        const double cos_now = std::max(-1.0, std::min(1.0, dot(n0, n1)));
        const double c = cos_now - e.rest_dihedral_cos;
        residual += std::abs(c);
        n_used++;

        const std::size_t k0 = e.opp0;
        const std::size_t k1 = e.opp1;

        const double denom = inv_masses[k0] + inv_masses[k1] + alpha;

        if (denom < 1e-12) {
            continue;
        }
        const double s = -strength * c / denom;
        predicted_positions[k0] += n0 * (s * inv_masses[k0]);
        predicted_positions[k1] -= n1 * (s * inv_masses[k1]);
    }

    return (n_used == 0) ? 0.0 : residual / static_cast<double>(n_used);
}

} // namespace membrane
