// deepxde_icbc.h - Extended boundary/initial conditions for DeepXDE C++ port
// Ports: RobinBC, PeriodicBC, PointSetBC, OperatorBC, PointSetOperatorBC, Interface2DBC
#ifndef DEEPXDE_ICBC_H
#define DEEPXDE_ICBC_H

#include "deepxde.h"

namespace deepxde {

// ============================================================================
// RobinBC: dy/dn(x) = func(x, y(x))
// Ports deepxde.icbc.boundary_conditions.RobinBC
// ============================================================================
using RobinFunc = std::function<double(const std::vector<double>&, double)>;

struct RobinBC : BCBase {
    std::shared_ptr<Geometry> geom;
    RobinFunc func;  // func(x, y) -> target normal derivative
    BoundaryPred on_boundary;
    int component;
    const FNN* net;

    RobinBC(std::shared_ptr<Geometry> g, RobinFunc f, BoundaryPred pred,
            const FNN* network, int comp = 0)
        : geom(g), func(std::move(f)), on_boundary(std::move(pred)),
          component(comp), net(network) {}

    Matrix collocation_points(const Matrix& all_boundary) const override {
        std::vector<std::vector<double>> pts;
        for (int i = 0; i < all_boundary.rows; ++i) {
            auto row = all_boundary.row(i);
            if (on_boundary(row, geom->on_boundary(row))) pts.push_back(row);
        }
        Matrix m(static_cast<int>(pts.size()), all_boundary.cols);
        for (int i = 0; i < m.rows; ++i) m.set_row(i, pts[i]);
        return m;
    }

    std::vector<double> error(const Matrix& bc_x,
                              const std::vector<std::vector<double>>& /*outputs_at_bc*/) const override {
        const double h = 1e-5;
        std::vector<double> err(bc_x.rows);
        for (int i = 0; i < bc_x.rows; ++i) {
            auto x = bc_x.row(i);
            auto n = geom->boundary_normal(x);
            std::vector<double> xp(x.size()), xm(x.size());
            for (size_t d = 0; d < x.size(); ++d) {
                xp[d] = x[d] + h * n[d];
                xm[d] = x[d] - h * n[d];
            }
            double dudn = (net->forward(xp)[component] - net->forward(xm)[component]) / (2 * h);
            double y_val = net->forward(x)[component];
            err[i] = dudn - func(x, y_val);
        }
        return err;
    }
};

// ============================================================================
// PeriodicBC: u(x_min) = u(x_max) on component_x
// Ports deepxde.icbc.boundary_conditions.PeriodicBC
// ============================================================================
struct PeriodicBC : BCBase {
    std::shared_ptr<Geometry> geom;
    BoundaryPred on_boundary;
    int component_x;
    int derivative_order;
    int component;
    const FNN* net;

    // Periodic point function: maps boundary point to its periodic image
    std::function<std::vector<double>(const std::vector<double>&, int)> periodic_point_fn;

    PeriodicBC(std::shared_ptr<Geometry> g, int comp_x, BoundaryPred pred,
               const FNN* network, int deriv_order = 0, int comp = 0,
               std::function<std::vector<double>(const std::vector<double>&, int)> ppfn = nullptr)
        : geom(g), on_boundary(std::move(pred)), component_x(comp_x),
          derivative_order(deriv_order), component(comp), net(network),
          periodic_point_fn(std::move(ppfn)) {}

    Matrix collocation_points(const Matrix& all_boundary) const override {
        // Collect boundary points on one side, then add their periodic images
        std::vector<std::vector<double>> pts1;
        for (int i = 0; i < all_boundary.rows; ++i) {
            auto row = all_boundary.row(i);
            if (on_boundary(row, geom->on_boundary(row))) pts1.push_back(row);
        }
        // Create periodic images
        std::vector<std::vector<double>> pts2;
        for (auto& p : pts1) {
            if (periodic_point_fn) {
                pts2.push_back(periodic_point_fn(p, component_x));
            } else {
                // Default: mirror across center of bounding box
                auto p2 = p;
                double lo = geom->bbox_lo[component_x];
                double hi = geom->bbox_hi[component_x];
                if (std::fabs(p[component_x] - lo) < 1e-12) p2[component_x] = hi;
                else if (std::fabs(p[component_x] - hi) < 1e-12) p2[component_x] = lo;
                pts2.push_back(p2);
            }
        }
        int n1 = static_cast<int>(pts1.size());
        Matrix m(2 * n1, all_boundary.cols);
        for (int i = 0; i < n1; ++i) m.set_row(i, pts1[i]);
        for (int i = 0; i < n1; ++i) m.set_row(n1 + i, pts2[i]);
        return m;
    }

    std::vector<double> error(const Matrix& bc_x,
                              const std::vector<std::vector<double>>& outputs_at_bc) const override {
        int mid = bc_x.rows / 2;
        std::vector<double> err(mid);
        if (derivative_order == 0) {
            for (int i = 0; i < mid; ++i)
                err[i] = outputs_at_bc[i][component] - outputs_at_bc[mid + i][component];
        } else {
            // First derivative comparison via finite differences
            const double h = 1e-5;
            for (int i = 0; i < mid; ++i) {
                auto x1 = bc_x.row(i);
                auto x2 = bc_x.row(mid + i);
                auto x1p = x1; x1p[component_x] += h;
                auto x1m = x1; x1m[component_x] -= h;
                auto x2p = x2; x2p[component_x] += h;
                auto x2m = x2; x2m[component_x] -= h;
                double d1 = (net->forward(x1p)[component] - net->forward(x1m)[component]) / (2*h);
                double d2 = (net->forward(x2p)[component] - net->forward(x2m)[component]) / (2*h);
                err[i] = d1 - d2;
            }
        }
        return err;
    }
};

// ============================================================================
// PointSetBC: Dirichlet BC for a fixed set of points
// Ports deepxde.icbc.boundary_conditions.PointSetBC
// ============================================================================
struct PointSetBC : BCBase {
    Matrix points;
    std::vector<double> values;
    int component;

    PointSetBC(const Matrix& pts, const std::vector<double>& vals, int comp = 0)
        : points(pts), values(vals), component(comp) {}

    Matrix collocation_points(const Matrix& /*all_boundary*/) const override {
        return points;
    }

    std::vector<double> error(const Matrix& bc_x,
                              const std::vector<std::vector<double>>& outputs_at_bc) const override {
        std::vector<double> err(bc_x.rows);
        for (int i = 0; i < bc_x.rows; ++i)
            err[i] = outputs_at_bc[i][component] - values[i];
        return err;
    }
};

// ============================================================================
// OperatorBC: General operator BC func(inputs, outputs, X) = 0
// Ports deepxde.icbc.boundary_conditions.OperatorBC
// ============================================================================
using OperatorFunc = std::function<double(const std::vector<double>& x,
                                          const std::vector<double>& output)>;

struct OperatorBC : BCBase {
    std::shared_ptr<Geometry> geom;
    OperatorFunc func;
    BoundaryPred on_boundary;

    OperatorBC(std::shared_ptr<Geometry> g, OperatorFunc f, BoundaryPred pred)
        : geom(g), func(std::move(f)), on_boundary(std::move(pred)) {}

    Matrix collocation_points(const Matrix& all_boundary) const override {
        std::vector<std::vector<double>> pts;
        for (int i = 0; i < all_boundary.rows; ++i) {
            auto row = all_boundary.row(i);
            if (on_boundary(row, geom->on_boundary(row))) pts.push_back(row);
        }
        Matrix m(static_cast<int>(pts.size()), all_boundary.cols);
        for (int i = 0; i < m.rows; ++i) m.set_row(i, pts[i]);
        return m;
    }

    std::vector<double> error(const Matrix& bc_x,
                              const std::vector<std::vector<double>>& outputs_at_bc) const override {
        std::vector<double> err(bc_x.rows);
        for (int i = 0; i < bc_x.rows; ++i)
            err[i] = func(bc_x.row(i), outputs_at_bc[i]);
        return err;
    }
};

// ============================================================================
// PointSetOperatorBC: Operator BC for fixed set of points
// Ports deepxde.icbc.boundary_conditions.PointSetOperatorBC
// ============================================================================
struct PointSetOperatorBC : BCBase {
    Matrix points;
    std::vector<double> values;
    OperatorFunc func;

    PointSetOperatorBC(const Matrix& pts, const std::vector<double>& vals, OperatorFunc f)
        : points(pts), values(vals), func(std::move(f)) {}

    Matrix collocation_points(const Matrix& /*all_boundary*/) const override {
        return points;
    }

    std::vector<double> error(const Matrix& bc_x,
                              const std::vector<std::vector<double>>& outputs_at_bc) const override {
        std::vector<double> err(bc_x.rows);
        for (int i = 0; i < bc_x.rows; ++i)
            err[i] = func(bc_x.row(i), outputs_at_bc[i]) - values[i];
        return err;
    }
};

// ============================================================================
// Interface2DBC: Interface boundary condition for 2D geometries
// Ports deepxde.icbc.boundary_conditions.Interface2DBC
// ============================================================================
struct Interface2DBC : BCBase {
    std::shared_ptr<Geometry> geom;
    ScalarFunc func;
    BoundaryPred on_boundary1, on_boundary2;
    std::string direction;  // "normal" or "tangent"

    Interface2DBC(std::shared_ptr<Geometry> g, ScalarFunc f,
                  BoundaryPred pred1, BoundaryPred pred2,
                  const std::string& dir = "normal")
        : geom(g), func(std::move(f)),
          on_boundary1(std::move(pred1)), on_boundary2(std::move(pred2)),
          direction(dir) {}

    Matrix collocation_points(const Matrix& all_boundary) const override {
        std::vector<std::vector<double>> pts1, pts2;
        for (int i = 0; i < all_boundary.rows; ++i) {
            auto row = all_boundary.row(i);
            bool ob = geom->on_boundary(row);
            if (on_boundary1(row, ob)) pts1.push_back(row);
            if (on_boundary2(row, ob)) pts2.push_back(row);
        }
        int n = std::min(static_cast<int>(pts1.size()), static_cast<int>(pts2.size()));
        Matrix m(2 * n, all_boundary.cols);
        for (int i = 0; i < n; ++i) m.set_row(i, pts1[i]);
        for (int i = 0; i < n; ++i) m.set_row(n + i, pts2[i]);
        return m;
    }

    std::vector<double> error(const Matrix& bc_x,
                              const std::vector<std::vector<double>>& outputs_at_bc) const override {
        int mid = bc_x.rows / 2;
        std::vector<double> err(mid);
        for (int i = 0; i < mid; ++i) {
            auto x1 = bc_x.row(i);
            auto n1 = geom->boundary_normal(x1);
            auto x2 = bc_x.row(mid + i);
            auto n2 = geom->boundary_normal(x2);
            double left = 0, right = 0;
            if (direction == "normal") {
                for (size_t d = 0; d < n1.size() && d < outputs_at_bc[i].size(); ++d)
                    left += outputs_at_bc[i][d] * n1[d];
                for (size_t d = 0; d < n2.size() && d < outputs_at_bc[mid+i].size(); ++d)
                    right += outputs_at_bc[mid+i][d] * n2[d];
            } else {  // tangent: rotate normal 90 degrees clockwise
                if (outputs_at_bc[i].size() >= 2) {
                    left = outputs_at_bc[i][0] * n1[1] - outputs_at_bc[i][1] * n1[0];
                    right = outputs_at_bc[mid+i][0] * n2[1] - outputs_at_bc[mid+i][1] * n2[0];
                }
            }
            err[i] = left + right - func(x1);
        }
        return err;
    }
};

} // namespace deepxde
#endif // DEEPXDE_ICBC_H
