// test_all.cpp - Compilation test exercising all DeepXDE C++ port components
#include "deepxde_all.h"
#include <cassert>
#include <iostream>

int main() {
    using namespace deepxde;
    std::cout << "=== DeepXDE C++ Port: Full Compilation Test ===\n\n";

    // --- Core AD ---
    {
        auto x = make_var(2.0, true);
        auto y = ad::sin(x) + ad::exp(x) * make_var(3.0);
        y->backward();
        std::cout << "[OK] AD: sin(2)+3*exp(2) = " << y->val
                  << ", grad = " << x->grad << "\n";
    }

    // --- Matrix ---
    {
        Matrix m(3, 2, 1.0);
        m(0,0) = 5; m(1,1) = 7;
        assert(m(0,0) == 5 && m(1,1) == 7);
        std::cout << "[OK] Matrix operations\n";
    }

    // --- Geometry: Interval, Rectangle (from deepxde.h) ---
    {
        Interval intv(0, 1);
        assert(intv.inside({0.5}));
        assert(intv.on_boundary({0.0}));
        auto pts = intv.random_points(10);
        assert(pts.rows == 10);

        Rectangle rect({0,0}, {1,1});
        assert(rect.inside({0.5, 0.5}));
        auto bpts = rect.random_boundary_points(20);
        assert(bpts.rows == 20);
        std::cout << "[OK] Interval, Rectangle\n";
    }

    // --- Extended Geometry ---
    {
        Hypercube hc({0,0,0,0}, {1,1,1,1});
        assert(hc.inside({0.5,0.5,0.5,0.5}));
        assert(hc.dim == 4);
        auto pts = hc.random_points(10);
        assert(pts.rows == 10 && pts.cols == 4);
        std::cout << "[OK] Hypercube (4D)\n";

        Hypersphere hs({0,0,0}, 1.0);
        assert(hs.inside({0.1,0.1,0.1}));
        assert(!hs.inside({2,2,2}));
        std::cout << "[OK] Hypersphere (3D)\n";

        Disk disk({0,0}, 1.0);
        assert(disk.inside({0.5, 0.5}));
        auto dpts = disk.random_points(20);
        assert(dpts.rows == 20);
        auto dbdy = disk.uniform_boundary_points(16);
        assert(dbdy.rows == 16);
        std::cout << "[OK] Disk\n";

        Sphere sphere({0,0,0}, 2.0);
        assert(sphere.inside({1,0,0}));
        std::cout << "[OK] Sphere\n";

        Cuboid cuboid({0,0,0}, {1,2,3});
        assert(cuboid.inside({0.5, 1.0, 1.5}));
        assert(cuboid.dim == 3);
        std::cout << "[OK] Cuboid\n";

        Ellipse ellipse({0,0}, 2.0, 1.0);
        assert(ellipse.inside({1.0, 0.0}));
        assert(ellipse.inside({0.0, 0.5}));
        auto epts = ellipse.random_points(15);
        assert(epts.rows == 15);
        std::cout << "[OK] Ellipse\n";

        Triangle tri({0,0}, {1,0}, {0.5, 1.0});
        assert(tri.inside({0.5, 0.3}));
        assert(!tri.inside({1.5, 1.5}));
        auto tpts = tri.random_boundary_points(12);
        assert(tpts.rows == 12);
        std::cout << "[OK] Triangle\n";

        Polygon poly({{0,0},{1,0},{1,1},{0,1}});
        assert(poly.inside({0.5, 0.5}));
        assert(poly.on_boundary({0.0, 0.5}));
        std::cout << "[OK] Polygon\n";
    }

    // --- PointCloud ---
    {
        Matrix pts(5, 2);
        for (int i = 0; i < 5; ++i) { pts(i,0) = i*0.2; pts(i,1) = i*0.1; }
        PointCloud pc(pts);
        assert(pc.dim == 2);
        auto rp = pc.random_points(3);
        assert(rp.rows == 3);
        std::cout << "[OK] PointCloud\n";
    }

    // --- CSG Operations ---
    {
        auto d1 = std::make_shared<Disk>(std::vector<double>{0,0}, 1.0);
        auto d2 = std::make_shared<Disk>(std::vector<double>{0.5,0}, 1.0);
        CSGUnion cu(d1, d2);
        assert(cu.inside({-0.5, 0}));
        assert(cu.inside({1.2, 0}));

        CSGDifference cd(d1, d2);
        assert(cd.inside({-0.8, 0}));

        CSGIntersection ci(d1, d2);
        assert(ci.inside({0.3, 0}));
        std::cout << "[OK] CSG Union/Difference/Intersection\n";
    }

    // --- TimeDomain, GeometryXTime ---
    {
        TimeDomain td(0, 1);
        assert(td.on_initial(0.0));
        GeometryXTime gxt(std::make_shared<Interval>(0,1), std::make_shared<TimeDomain>(0,1));
        assert(gxt.inside({0.5, 0.5}));
        auto icp = gxt.random_initial_points(10);
        assert(icp.rows == 10);
        std::cout << "[OK] TimeDomain, GeometryXTime\n";
    }

    // --- FNN ---
    {
        FNN net({2, 16, 16, 1}, Activation::Tanh);
        auto out = net.forward({0.5, 0.3});
        assert(out.size() == 1);
        auto batch = net.forward_batch(Matrix(5, 2, 0.1));
        assert(batch.rows == 5 && batch.cols == 1);
        std::cout << "[OK] FNN forward + batch\n";
    }

    // --- ResNet ---
    {
        ResNet rn(2, 1, 16, 2, Activation::Tanh);
        auto out = rn.forward({0.5, 0.3});
        assert(out.size() == 1);
        auto batch = rn.forward_batch(Matrix(5, 2, 0.1));
        assert(batch.rows == 5);
        std::cout << "[OK] ResNet\n";
    }

    // --- PFNN ---
    {
        PFNN pfnn(2, 3, {8, 8}, Activation::Tanh);
        auto out = pfnn.forward({0.5, 0.3});
        assert(out.size() == 3);
        std::cout << "[OK] PFNN (" << pfnn.num_trainable_parameters() << " params)\n";
    }

    // --- MsFFN ---
    {
        MsFFN msffn({2, 16, 8, 1}, Activation::Tanh, {1.0, 10.0});
        auto out = msffn.forward({0.5, 0.3});
        assert(out.size() == 1);
        std::cout << "[OK] MsFFN\n";
    }

    // --- STMsFFN ---
    {
        STMsFFN stmsffn({3, 16, 8, 1}, Activation::Tanh, {1.0}, {1.0});
        auto out = stmsffn.forward({0.5, 0.3, 0.1});
        assert(out.size() == 1);
        std::cout << "[OK] STMsFFN\n";
    }

    // --- DeepONet ---
    {
        DeepONet don({10, 16, 8}, {2, 16, 8}, Activation::Tanh);
        std::vector<double> fv(10, 0.1), loc = {0.3, 0.5};
        auto out = don.forward(fv, loc);
        assert(out.size() == 1);
        std::cout << "[OK] DeepONet\n";
    }

    // --- DeepONetCartesianProd ---
    {
        DeepONetCartesianProd doncp({10, 16, 8}, {2, 16, 8}, Activation::Tanh);
        Matrix fv(3, 10, 0.1), locs(5, 2, 0.2);
        auto out = doncp.forward_cartesian(fv, locs);
        assert(out.rows == 3 && out.cols == 5);
        std::cout << "[OK] DeepONetCartesianProd\n";
    }

    // --- PODDeepONet ---
    {
        Matrix basis(4, 10, 0.1);
        PODDeepONet pod(basis, {10, 16, 4}, Activation::Tanh);
        Matrix fv(2, 10, 0.1);
        auto out = pod.forward_cartesian(fv, Matrix());
        assert(out.rows == 2 && out.cols == 10);
        std::cout << "[OK] PODDeepONet\n";
    }

    // --- MIONet ---
    {
        MIONet mio({5, 8, 4}, {5, 8, 4}, {2, 8, 4}, Activation::Tanh);
        Matrix f1(2, 5, 0.1), f2(2, 5, 0.2), locs(3, 2, 0.3);
        auto out = mio.forward_cartesian(f1, f2, locs);
        assert(out.rows == 2 && out.cols == 3);
        std::cout << "[OK] MIONet\n";
    }

    // --- MfNN ---
    {
        MfNN mfnn({1, 8, 1}, {8, 1}, Activation::Tanh);
        auto [lo, hi] = mfnn.forward({0.5});
        assert(lo.size() == 1 && hi.size() == 1);
        std::cout << "[OK] MfNN\n";
    }

    // --- ModifiedMLP ---
    {
        ModifiedMLP mmlp({2, 16, 16, 1}, Activation::Tanh);
        auto out = mmlp.forward({0.5, 0.3});
        assert(out.size() == 1);
        std::cout << "[OK] ModifiedMLP\n";
    }

    // --- Extended BCs ---
    {
        auto geom = std::make_shared<Interval>(0, 1);
        FNN net({1, 8, 1}, Activation::Tanh);

        // PointSetBC
        Matrix pspts(3, 1); pspts(0,0)=0.0; pspts(1,0)=0.5; pspts(2,0)=1.0;
        PointSetBC psbc(pspts, {0, 0.5, 0});
        auto cp = psbc.collocation_points(Matrix());
        assert(cp.rows == 3);
        std::cout << "[OK] PointSetBC\n";

        // OperatorBC
        OperatorBC obc(geom, [](const std::vector<double>&, const std::vector<double>& o){ return o[0]; },
                       [](const std::vector<double>&, bool ob){ return ob; });
        std::cout << "[OK] OperatorBC\n";

        // RobinBC
        RobinBC rbc(geom, [](const std::vector<double>&, double y){ return -y; },
                    [](const std::vector<double>&, bool ob){ return ob; }, &net);
        std::cout << "[OK] RobinBC\n";

        // PeriodicBC
        PeriodicBC pbc(geom, 0, [](const std::vector<double>&, bool ob){ return ob; }, &net);
        std::cout << "[OK] PeriodicBC\n";

        // Interface2DBC
        auto rect = std::make_shared<Rectangle>(std::vector<double>{0,0}, std::vector<double>{1,1});
        Interface2DBC ibc(rect, [](const std::vector<double>&){return 0.0;},
                          [](const std::vector<double>& x, bool){ return std::fabs(x[0]) < 1e-12; },
                          [](const std::vector<double>& x, bool){ return std::fabs(x[0]-1) < 1e-12; });
        std::cout << "[OK] Interface2DBC\n";
    }

    // --- Data types ---
    {
        BatchSampler bs(100);
        auto idx = bs.get_next(10);
        assert(idx.size() == 10);
        std::cout << "[OK] BatchSampler\n";

        Matrix tx(10, 2, 0.1), ty(10, 1, 0.5);
        DataSet ds(tx, ty, tx, ty);
        auto [x, y] = ds.train_next_batch();
        assert(x.rows == 10);
        std::cout << "[OK] DataSet\n";

        PowerSeries ps(5, 1.0);
        auto feats = ps.random_features(3);
        assert(feats.rows == 3 && feats.cols == 5);
        double v = ps.eval_one(feats.row(0), 0.5);
        (void)v;
        std::cout << "[OK] PowerSeries FunctionSpace\n";

        ChebyshevSpace cs(5, 1.0);
        auto cf = cs.random_features(2);
        double cv = cs.eval_one(cf.row(0), 0.5);
        (void)cv;
        std::cout << "[OK] Chebyshev FunctionSpace\n";

        auto intv = std::make_shared<Interval>(0, 1);
        IDEData ide(intv, 4);
        auto qpts = ide.quad_points(Matrix(3, 1, 0.5));
        assert(qpts.rows == 12);
        std::cout << "[OK] IDEData\n";

        FractionalScheme fs("dynamic", {10, 20});
        auto glw = FractionalScheme::gl_weights(1.5, 10);
        assert(glw.size() == 11);
        std::cout << "[OK] FractionalScheme\n";
    }

    // --- Callbacks ---
    {
        CallbackList cbl;
        auto es = std::make_shared<EarlyStopping>(0.001, 10);
        es->set_loss_fn([]{ return 0.1; });
        cbl.add(es);
        auto mc = std::make_shared<ModelCheckpoint>("test_ckpt");
        cbl.add(mc);
        auto timer = std::make_shared<TimerCallback>(60.0);
        cbl.add(timer);
        auto resampler = std::make_shared<PDEPointResampler>(50);
        cbl.add(resampler);
        auto du = std::make_shared<DropoutUncertainty>(500);
        cbl.add(du);
        auto vv = std::make_shared<VariableValue>(std::vector<double*>{});
        cbl.add(vv);
        auto op = std::make_shared<OperatorPredictor>(10);
        cbl.add(op);
        cbl.on_train_begin();
        cbl.on_epoch_begin();
        cbl.on_epoch_end();
        cbl.on_train_end();
        std::cout << "[OK] All callbacks\n";
    }

    // --- Losses ---
    {
        std::vector<double> yt = {1, 2, 3}, yp = {1.1, 2.2, 2.8};
        double mse = losses::mean_squared_error(yt, yp);
        double mae = losses::mean_absolute_error(yt, yp);
        double mape = losses::mean_absolute_percentage_error(yt, yp);
        double l2r = losses::mean_l2_relative_error(yt, yp);
        assert(mse > 0 && mae > 0 && mape > 0 && l2r > 0);

        auto loss_fn = losses::get_loss("MSE");
        assert(loss_fn(yt, yp) > 0);
        std::cout << "[OK] Loss functions (MSE=" << mse << ", MAE=" << mae << ")\n";
    }

    // --- Metrics ---
    {
        Matrix yt(3, 1), yp(3, 1);
        yt(0,0) = 1; yt(1,0) = 2; yt(2,0) = 3;
        yp(0,0) = 1.1; yp(1,0) = 2.2; yp(2,0) = 2.8;
        double l2 = metrics::l2_relative_error(yt, yp);
        double mse = metrics::mean_squared_error(yt, yp);
        assert(l2 > 0 && mse > 0);
        std::cout << "[OK] Metrics (L2_rel=" << l2 << ")\n";
    }

    // --- Optimizers ---
    {
        // Adam (already tested in base)
        Adam adam(1e-3);
        adam.init(3);

        SGD sgd(0.01, 0.9);
        sgd.init(3);

        RMSProp rmsp(0.001);
        rmsp.init(3);

        LBFGS lbfgs(10);
        std::cout << "[OK] Optimizers (Adam, SGD, RMSProp, L-BFGS)\n";
    }

    // --- LR Schedulers ---
    {
        lr_schedule::StepLR slr(1e-3, 100, 0.5);
        assert(slr(0) > slr(200));

        lr_schedule::CosineAnnealingLR calr(1e-3, 1000);
        assert(calr(0) > calr(500));

        lr_schedule::ExponentialLR elr(1e-3, 0.99);
        assert(elr(0) > elr(100));

        lr_schedule::InverseTimeLR itlr(1e-3, 100, 0.5);
        assert(itlr(0) > itlr(100));
        std::cout << "[OK] LR Schedulers\n";
    }

    // --- Gradients ---
    {
        FNN net({2, 8, 1}, Activation::Tanh);
        auto J = gradients::jacobian_ad(net, {0.5, 0.3});
        assert(J.rows == 1 && J.cols == 2);

        auto H = gradients::hessian_ad(net, {0.5, 0.3});
        assert(H.rows == 2 && H.cols == 2);

        double lap = gradients::laplacian_ad(net, {0.5, 0.3});
        (void)lap;
        std::cout << "[OK] Jacobian, Hessian, Laplacian\n";
    }

    // --- Fractional calculus ---
    {
        auto w = fractional::gl_weights(1.5, 10);
        assert(w.size() == 11);
        double c1 = fractional::frac_laplacian_constant_1d(1.5);
        double c2 = fractional::frac_laplacian_constant(1.5, 2);
        (void)c1; (void)c2;
        std::cout << "[OK] Fractional calculus\n";
    }

    // --- Feature transforms ---
    {
        transforms::FourierFeatures ff(2, 8, 1.0);
        auto feat = ff.transform({0.5, 0.3});
        assert(feat.size() == 16);
        auto batch = ff.transform_batch(Matrix(5, 2, 0.3));
        assert(batch.rows == 5 && batch.cols == 16);

        transforms::Standardizer scaler;
        Matrix data(10, 2);
        for (int i = 0; i < 10; ++i) { data(i,0) = i*0.1; data(i,1) = i*0.2; }
        scaler.fit(data);
        auto scaled = scaler.transform(data);
        auto inv = scaler.inverse_transform(scaled);
        assert(std::fabs(inv(5,0) - data(5,0)) < 1e-10);
        std::cout << "[OK] FourierFeatures, Standardizer\n";
    }

    // --- FullModel (extended) ---
    {
        FullModel fm;
        fm.set_network(FNN({2, 8, 1}, Activation::Tanh));
        auto geom = std::make_shared<Interval>(0, 1);
        auto td = std::make_shared<TimeDomain>(0, 1);
        auto gxt = std::make_shared<GeometryXTime>(geom, td);
        fm.set_geometry_time(gxt);
        fm.num_domain = 10;
        fm.num_boundary = 5;
        fm.num_initial = 5;
        fm.set_pde([&fm](const std::vector<VarPtr>& x, const std::vector<VarPtr>& u) -> VarPtr {
            return u[0];  // Trivial PDE
        });
        fm.compile("adam", 1e-3);
        // Just 2 epochs to verify it runs
        auto hist = fm.train(2, 1);
        assert(!hist.empty());
        std::cout << "[OK] FullModel compile/train/predict\n";
    }

    std::cout << "\n=== ALL TESTS PASSED ===\n";
    std::cout << "Total lines of C++ code across all headers:\n";
    return 0;
}
