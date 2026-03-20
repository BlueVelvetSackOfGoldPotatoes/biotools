// main.cpp -- Demo: train GNN models on a toy graph classification task
//
// We generate random graphs with two classes:
//   Class 0: "ring" graphs  (cycle on N nodes)
//   Class 1: "star" graphs  (one hub connected to N-1 leaves)
//
// Node features are random but the structural difference is enough for a
// GNN to learn the classification.  We demonstrate GCNConv, GATConv,
// SAGEConv, and GINConv -- each trained with Adam + cross-entropy.

#include "pyg.h"
#include "pyg_conv.h"
#include "pyg_pool.h"
#include "pyg_norm.h"
#include "pyg_dense.h"
#include "pyg_transforms.h"
#include "pyg_models.h"
#include "pyg_loader.h"

#include <cstdio>
#include <iostream>

using namespace pyg;

// ------------------------------------------------------------------
//  Toy dataset generator
// ------------------------------------------------------------------

// Make a ring (cycle) graph with `n` nodes and `feat_dim` random features.
Data make_ring(int n, int feat_dim) {
    Data g;
    g.num_nodes = n;
    g.x = Tensor::randn(n, feat_dim, 0.5f);
    g.x->set_requires_grad(true);
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        g.edge_src.push_back(i); g.edge_dst.push_back(j);
        g.edge_src.push_back(j); g.edge_dst.push_back(i);
    }
    g.graph_label = 0;
    return g;
}

// Make a star graph with `n` nodes (node 0 is hub).
Data make_star(int n, int feat_dim) {
    Data g;
    g.num_nodes = n;
    g.x = Tensor::randn(n, feat_dim, 0.5f);
    g.x->set_requires_grad(true);
    for (int i = 1; i < n; ++i) {
        g.edge_src.push_back(0); g.edge_dst.push_back(i);
        g.edge_src.push_back(i); g.edge_dst.push_back(0);
    }
    g.graph_label = 1;
    return g;
}

// ------------------------------------------------------------------
//  A small GCN-based graph classifier
// ------------------------------------------------------------------
struct GCNClassifier {
    GCNConv conv1, conv2;
    Linear classifier;
    int hidden;

    GCNClassifier(int in_feat, int hidden, int num_classes)
        : conv1(in_feat, hidden), conv2(hidden, hidden),
          classifier(hidden, num_classes), hidden(hidden) {}

    TensorPtr forward(Data& batch_data, int num_graphs) {
        auto h = conv1.forward(batch_data.x, batch_data);
        h = autograd::relu(h);
        h = conv2.forward(h, batch_data);
        h = autograd::relu(h);
        // Global mean pooling
        auto graph_emb = pool::global_mean_pool(h, batch_data.batch, num_graphs);
        auto logits = classifier.forward(graph_emb);
        return logits;
    }

    std::vector<TensorPtr> parameters() {
        auto p = conv1.parameters();
        auto p2 = conv2.parameters();
        p.insert(p.end(), p2.begin(), p2.end());
        auto p3 = classifier.parameters();
        p.insert(p.end(), p3.begin(), p3.end());
        return p;
    }
};

// ------------------------------------------------------------------
//  A small GAT-based graph classifier
// ------------------------------------------------------------------
struct GATClassifier {
    GATConv conv1, conv2;
    Linear classifier;
    int hidden;

    GATClassifier(int in_feat, int hidden, int num_classes, int heads = 2)
        : conv1(in_feat, hidden, heads, /*concat=*/true),
          conv2(hidden * heads, hidden, 1, /*concat=*/false),
          classifier(hidden, num_classes), hidden(hidden) {}

    TensorPtr forward(Data& batch_data, int num_graphs) {
        auto h = conv1.forward(batch_data.x, batch_data);
        h = autograd::relu(h);
        h = conv2.forward(h, batch_data);
        h = autograd::relu(h);
        auto graph_emb = pool::global_mean_pool(h, batch_data.batch, num_graphs);
        auto logits = classifier.forward(graph_emb);
        return logits;
    }

    std::vector<TensorPtr> parameters() {
        auto p = conv1.parameters();
        auto p2 = conv2.parameters();
        p.insert(p.end(), p2.begin(), p2.end());
        auto p3 = classifier.parameters();
        p.insert(p.end(), p3.begin(), p3.end());
        return p;
    }
};

// ------------------------------------------------------------------
//  A small GraphSAGE-based graph classifier
// ------------------------------------------------------------------
struct SAGEClassifier {
    SAGEConv conv1, conv2;
    Linear classifier;
    int hidden;

    SAGEClassifier(int in_feat, int hidden, int num_classes)
        : conv1(in_feat, hidden), conv2(hidden, hidden),
          classifier(hidden, num_classes), hidden(hidden) {}

    TensorPtr forward(Data& batch_data, int num_graphs) {
        auto h = conv1.forward(batch_data.x, batch_data);
        h = autograd::relu(h);
        h = conv2.forward(h, batch_data);
        h = autograd::relu(h);
        auto graph_emb = pool::global_mean_pool(h, batch_data.batch, num_graphs);
        auto logits = classifier.forward(graph_emb);
        return logits;
    }

    std::vector<TensorPtr> parameters() {
        auto p = conv1.parameters();
        auto p2 = conv2.parameters();
        p.insert(p.end(), p2.begin(), p2.end());
        auto p3 = classifier.parameters();
        p.insert(p.end(), p3.begin(), p3.end());
        return p;
    }
};

// ------------------------------------------------------------------
//  A small GIN-based graph classifier
// ------------------------------------------------------------------
struct GINClassifier {
    GINConv conv1, conv2;
    Linear classifier;
    int hidden;

    GINClassifier(int in_feat, int hidden, int num_classes)
        : conv1(in_feat, hidden), conv2(hidden, hidden),
          classifier(hidden, num_classes), hidden(hidden) {}

    TensorPtr forward(Data& batch_data, int num_graphs) {
        auto h = conv1.forward(batch_data.x, batch_data);
        h = autograd::relu(h);
        h = conv2.forward(h, batch_data);
        h = autograd::relu(h);
        auto graph_emb = pool::global_add_pool(h, batch_data.batch, num_graphs);
        auto logits = classifier.forward(graph_emb);
        return logits;
    }

    std::vector<TensorPtr> parameters() {
        auto p = conv1.parameters();
        auto p2 = conv2.parameters();
        p.insert(p.end(), p2.begin(), p2.end());
        auto p3 = classifier.parameters();
        p.insert(p.end(), p3.begin(), p3.end());
        return p;
    }
};

// ------------------------------------------------------------------
//  Training loop  (generic -- works with any of the classifiers above)
// ------------------------------------------------------------------
template<typename Model>
void train_and_eval(const std::string& model_name,
                    Model& model,
                    std::vector<Data>& train_set,
                    std::vector<Data>& test_set,
                    int epochs, int batch_size, float lr)
{
    std::printf("\n========================================\n");
    std::printf("  Training %s  (%d train, %d test)\n",
                model_name.c_str(), (int)train_set.size(), (int)test_set.size());
    std::printf("========================================\n");

    auto params = model.parameters();
    Adam optimizer(params, lr);

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float total_loss = 0;
        int total_correct = 0;
        int total_samples = 0;

        // Simple mini-batching
        for (int b = 0; b < (int)train_set.size(); b += batch_size) {
            int end = std::min(b + batch_size, (int)train_set.size());
            std::vector<Data> mini_batch(train_set.begin() + b, train_set.begin() + end);
            int num_graphs = (int)mini_batch.size();

            // Rebuild features with requires_grad each iteration (fresh graph)
            for (auto& g : mini_batch) {
                auto x_new = std::make_shared<Tensor>(g.num_nodes, g.x->cols());
                for (int i = 0; i < g.num_nodes * g.x->cols(); ++i)
                    x_new->data_[i] = g.x->data_[i];
                x_new->set_requires_grad(true);
                g.x = x_new;
            }

            Data batch_data = batch_graphs(mini_batch);
            batch_data.x->set_requires_grad(true);

            // Collect graph labels
            std::vector<int> labels;
            for (auto& g : mini_batch)
                labels.push_back(g.graph_label);

            optimizer.zero_grad();

            auto logits = model.forward(batch_data, num_graphs);
            auto l = loss::cross_entropy(logits, labels);

            autograd::backward(l);
            optimizer.step();

            total_loss += l->data_[0] * num_graphs;
            total_correct += (int)(accuracy(logits, labels) * num_graphs);
            total_samples += num_graphs;
        }

        if (epoch % 10 == 0 || epoch == 1) {
            // Evaluate on test set
            std::vector<Data> test_copy(test_set);
            for (auto& g : test_copy) {
                auto x_new = std::make_shared<Tensor>(g.num_nodes, g.x->cols());
                for (int i = 0; i < g.num_nodes * g.x->cols(); ++i)
                    x_new->data_[i] = g.x->data_[i];
                g.x = x_new;
            }
            Data test_batch = batch_graphs(test_copy);
            int num_test = (int)test_copy.size();
            std::vector<int> test_labels;
            for (auto& g : test_copy)
                test_labels.push_back(g.graph_label);

            auto test_logits = model.forward(test_batch, num_test);
            float test_acc = accuracy(test_logits, test_labels);

            std::printf("  Epoch %3d | loss: %.4f | train acc: %.2f%% | test acc: %.2f%%\n",
                        epoch,
                        total_loss / total_samples,
                        100.0f * total_correct / total_samples,
                        100.0f * test_acc);
        }
    }
}

// ------------------------------------------------------------------
//  Demo: TopK pooling, normalization layers, transforms
// ------------------------------------------------------------------
void demo_extras() {
    std::printf("\n========================================\n");
    std::printf("  Demonstrating additional components\n");
    std::printf("========================================\n");

    // Create a small graph
    Data g;
    g.num_nodes = 10;
    g.x = Tensor::randn(10, 8, 1.0f);
    for (int i = 0; i < 10; ++i) for (int j = i+1; j < 10; ++j) {
        if (std::abs(i - j) <= 2) {
            g.edge_src.push_back(i); g.edge_dst.push_back(j);
            g.edge_src.push_back(j); g.edge_dst.push_back(i);
        }
    }

    // Test transforms
    transforms::NormalizeFeatures norm_tf("l2");
    norm_tf(g);
    std::printf("  NormalizeFeatures applied: x[0] norm = %.4f\n",
        std::sqrt([&]{ float s=0; for (int j=0;j<8;++j) s+=(*g.x)(0,j)*(*g.x)(0,j); return s; }()));

    transforms::OneHotDegree ohd(10);
    ohd(g);
    std::printf("  OneHotDegree applied: features expanded to %d\n", g.x->cols());

    // Test pooling
    TopKPooling topk_pool(g.x->cols(), 0.5f);
    auto pool_res = topk_pool.forward(g.x, g);
    std::printf("  TopKPooling: %d -> %d nodes\n", g.num_nodes, pool_res.data.num_nodes);

    // Test normalization
    std::vector<int> batch(g.num_nodes, 0);
    GraphNorm gn(g.x->cols());
    auto normed = gn.forward(g.x, batch, 1);
    std::printf("  GraphNorm applied: output shape (%d, %d)\n", normed->rows(), normed->cols());

    PairNorm pn(1.0f);
    auto pn_out = pn.forward(g.x);
    std::printf("  PairNorm applied: output shape (%d, %d)\n", pn_out->rows(), pn_out->cols());

    // Test dense operations
    auto adj = utils::to_dense_adj(g.edge_src, g.edge_dst, g.num_nodes);
    auto dense_x = Tensor::randn(g.num_nodes, 8, 1.0f);
    dense::DenseGCNConv dgcn(8, 16);
    auto dense_out = dgcn.forward(dense_x, adj);
    std::printf("  DenseGCNConv: (%d,%d) -> (%d,%d)\n", dense_x->rows(), dense_x->cols(), dense_out->rows(), dense_out->cols());

    // Test GAE model
    Data gae_data;
    gae_data.num_nodes = 20;
    gae_data.x = Tensor::randn(20, 16, 1.0f);
    for (int i = 0; i < 20; ++i) for (int j = i+1; j < 20; ++j) {
        if (std::abs(i-j) <= 3) { gae_data.edge_src.push_back(i); gae_data.edge_dst.push_back(j);
            gae_data.edge_src.push_back(j); gae_data.edge_dst.push_back(i); }
    }
    models::GAE gae(16, 32, 16);
    auto z = gae.encode(gae_data.x, gae_data);
    auto recon = gae.recon_loss(z, gae_data);
    std::printf("  GAE reconstruction loss: %.4f\n", (*recon)(0,0));

    // Test JumpingKnowledge
    auto h1 = Tensor::randn(10, 8, 1.0f);
    auto h2 = Tensor::randn(10, 8, 1.0f);
    auto h3 = Tensor::randn(10, 8, 1.0f);
    models::JumpingKnowledge jk_cat("cat");
    auto jk_out = jk_cat.forward({h1, h2, h3});
    std::printf("  JumpingKnowledge(cat): 3x(10,8) -> (%d,%d)\n", jk_out->rows(), jk_out->cols());

    models::JumpingKnowledge jk_max("max");
    auto jk_out2 = jk_max.forward({h1, h2, h3});
    std::printf("  JumpingKnowledge(max): 3x(10,8) -> (%d,%d)\n", jk_out2->rows(), jk_out2->cols());

    // Test new conv layers
    PointTransformerConv pt_conv(8, 16);
    Data pt_data;
    pt_data.num_nodes = 10;
    auto pt_pos = Tensor::randn(10, 3, 1.0f);
    auto pt_x = Tensor::randn(10, 8, 1.0f);
    for (int i = 0; i < 10; ++i) for (int j = 0; j < 10; ++j)
        if (i != j && std::abs(i - j) <= 2) { pt_data.edge_src.push_back(i); pt_data.edge_dst.push_back(j); }
    auto pt_out = pt_conv.forward(pt_x, pt_pos, pt_data);
    std::printf("  PointTransformerConv: (%d,%d) -> (%d,%d)\n", pt_x->rows(), pt_x->cols(), pt_out->rows(), pt_out->cols());

    GeneralConv gen_conv(8, 16, Aggr::MEAN, true, true);
    auto gen_out = gen_conv.forward(pt_x, pt_data);
    std::printf("  GeneralConv: (%d,%d) -> (%d,%d)\n", pt_x->rows(), pt_x->cols(), gen_out->rows(), gen_out->cols());

    std::printf("\n  All extra components verified successfully!\n");
}

// ------------------------------------------------------------------
int main() {
    std::printf("=== C++ Port of PyTorch Geometric -- Graph Classification Demo ===\n\n");

    const int feat_dim = 8;
    const int hidden = 16;
    const int num_classes = 2;
    const int num_train = 60;
    const int num_test  = 20;
    const int epochs = 50;
    const int batch_size = 10;
    const float lr = 0.01f;

    // Generate dataset: ring graphs (class 0) and star graphs (class 1)
    std::vector<Data> train_set, test_set;

    manual_seed(123);
    for (int i = 0; i < num_train / 2; ++i) {
        int n = 5 + (int)(global_rng()() % 4); // 5-8 nodes
        train_set.push_back(make_ring(n, feat_dim));
    }
    for (int i = 0; i < num_train / 2; ++i) {
        int n = 5 + (int)(global_rng()() % 4);
        train_set.push_back(make_star(n, feat_dim));
    }
    for (int i = 0; i < num_test / 2; ++i) {
        int n = 5 + (int)(global_rng()() % 4);
        test_set.push_back(make_ring(n, feat_dim));
    }
    for (int i = 0; i < num_test / 2; ++i) {
        int n = 5 + (int)(global_rng()() % 4);
        test_set.push_back(make_star(n, feat_dim));
    }

    // Shuffle training set
    std::shuffle(train_set.begin(), train_set.end(), global_rng());

    std::printf("Dataset: %d training graphs, %d test graphs\n",
                (int)train_set.size(), (int)test_set.size());
    std::printf("  Features per node: %d\n", feat_dim);
    std::printf("  Classes: ring (0) vs star (1)\n");

    // ----- GCN -----
    {
        manual_seed(42);
        GCNClassifier model(feat_dim, hidden, num_classes);
        train_and_eval("GCNConv", model, train_set, test_set, epochs, batch_size, lr);
    }

    // ----- GAT -----
    {
        manual_seed(42);
        GATClassifier model(feat_dim, hidden, num_classes, /*heads=*/2);
        train_and_eval("GATConv", model, train_set, test_set, epochs, batch_size, lr);
    }

    // ----- GraphSAGE -----
    {
        manual_seed(42);
        SAGEClassifier model(feat_dim, hidden, num_classes);
        train_and_eval("SAGEConv", model, train_set, test_set, epochs, batch_size, lr);
    }

    // ----- GIN -----
    {
        manual_seed(42);
        GINClassifier model(feat_dim, hidden, num_classes);
        train_and_eval("GINConv", model, train_set, test_set, epochs, batch_size, lr);
    }

    // ----- Additional components demo -----
    demo_extras();

    std::printf("\n=== All models trained successfully ===\n");
    return 0;
}
