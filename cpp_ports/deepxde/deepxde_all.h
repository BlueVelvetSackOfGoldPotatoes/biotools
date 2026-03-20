// deepxde_all.h - Master include for the complete DeepXDE C++ port
//
// Includes all components:
//   deepxde.h          - Core: AD, Matrix, Geometry base, Interval, Rectangle,
//                        TimeDomain, GeometryXTime, FNN, Adam, DirichletBC,
//                        NeumannBC, IC, Model (basic), AD derivative helpers
//   deepxde_geometry.h - Extended geometry: Hypercube, Hypersphere, Disk, Sphere,
//                        Cuboid, Ellipse, Triangle, Polygon, PointCloud,
//                        CSGUnion, CSGDifference, CSGIntersection
//   deepxde_nn.h       - Extended NN: ResNet, PFNN, MsFFN, STMsFFN, DeepONet,
//                        DeepONetCartesianProd, PODDeepONet, MIONet, MfNN, ModifiedMLP
//   deepxde_icbc.h     - Extended BC/IC: RobinBC, PeriodicBC, PointSetBC,
//                        OperatorBC, PointSetOperatorBC, Interface2DBC
//   deepxde_data.h     - Data types: BatchSampler, DataSet, FunctionData, Triple,
//                        Quadruple, MfFuncData, FunctionSpace, PowerSeries,
//                        ChebyshevSpace, IDEData, FractionalScheme, PDEOperatorData
//   deepxde_callbacks.h- Callbacks: CallbackList, EarlyStopping, ModelCheckpoint,
//                        TimerCallback, PDEPointResampler, DropoutUncertainty,
//                        MovieDumper, OperatorPredictor, VariableValue
//   deepxde_losses.h   - Losses: MSE, MAE, MAPE, L2Relative, SoftmaxCE, Zero
//                        Metrics: l2_relative_error, MSE, MAPE, accuracy
//                        Optimizers: SGD, RMSProp, Adam, L-BFGS
//                        LR schedulers: StepLR, CosineAnnealing, Exponential, InverseTime
//                        Gradients: Jacobian, Hessian, Laplacian (AD-based)
//                        Fractional: GL weights, fractional Laplacian constants
//                        Transforms: FourierFeatures, Standardizer
//                        FullModel: compile, train, predict, save, restore
#ifndef DEEPXDE_ALL_H
#define DEEPXDE_ALL_H

#include "deepxde.h"
#include "deepxde_geometry.h"
#include "deepxde_nn.h"
#include "deepxde_icbc.h"
#include "deepxde_data.h"
#include "deepxde_callbacks.h"
#include "deepxde_losses.h"

#endif // DEEPXDE_ALL_H
