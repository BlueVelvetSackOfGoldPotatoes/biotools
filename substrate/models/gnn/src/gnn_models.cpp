#include "gnn_models.h"
#include "core/bio/bio_affine.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <stdexcept>

// ============================================================
// GCNLayer
// ============================================================

GCNLayer::GCNLayer(size_t in_f, size_t out_f, std::mt19937& rng)
    : in_features(in_f), out_features(out_f) {
    // Xavier initialization for weights
    weights = Tensor(in_f, out_f);
    weights.fill_xavier(in_f, out_f, rng);

    bias = Tensor(1, out_f, 0.0);

    grad_weights = Tensor(in_f, out_f, 0.0);
    grad_bias = Tensor(1, out_f, 0.0);
}

Tensor GCNLayer::forward(const Tensor& X, const Tensor& A_norm) {
    // X: N x in_features
    // A_norm: N x N
    // Output: A_norm @ X @ W + b  => N x out_features
    if (X.cols != in_features) {
        throw std::runtime_error("GCNLayer::forward feature width mismatch");
    }
    if (A_norm.rows != X.rows || A_norm.cols != X.rows) {
        throw std::runtime_error("GCNLayer::forward adjacency shape mismatch");
    }

    cached_input = X;
    cached_adj_norm = A_norm;

    // AH = A_norm @ X  (N x in_features)
    cached_AH = A_norm.matmul(X);

    // output = AH @ W + b (N x out_features)
    return bio::affine(cached_AH, weights, &bias, false);
}

Tensor GCNLayer::backward(const Tensor& grad, const Tensor& A_norm) {
    // grad: N x out_features (gradient of loss w.r.t. output of this layer)
    //
    // Forward was: output = A_norm @ X @ W + b
    //   Let AH = A_norm @ X
    //   output = AH @ W + b
    //
    // d(Loss)/d(W) = AH^T @ grad              (in_features x out_features)
    // d(Loss)/d(b) = sum_rows(grad)             (1 x out_features)
    // d(Loss)/d(AH) = grad @ W^T               (N x in_features)
    // d(Loss)/d(X) = A_norm^T @ d(Loss)/d(AH)  (N x in_features)

    // grad_weights = AH^T @ grad
    grad_weights = cached_AH.transpose().matmul(grad);

    // grad_bias = sum over rows of grad
    grad_bias = grad.sum_rows(); // 1 x out_features

    // grad_AH = grad @ W^T
    Tensor grad_AH = grad.matmul(weights.transpose()); // N x in_features

    // grad_X = A_norm^T @ grad_AH
    // For symmetric normalization, A_norm is symmetric, so A_norm^T = A_norm
    Tensor grad_X = A_norm.transpose().matmul(grad_AH); // N x in_features

    return grad_X;
}

// ============================================================
// GATLayer
// ============================================================

GATLayer::GATLayer(size_t in_f, size_t out_f, std::mt19937& rng, double neg_slope)
    : in_features(in_f), out_features(out_f), negative_slope(neg_slope) {
    // Xavier init for W
    W = Tensor(in_f, out_f);
    W.fill_xavier(in_f, out_f, rng);

    // Attention vectors
    a_left = Tensor(out_f, 1);
    a_left.fill_xavier(out_f, 1, rng);

    a_right = Tensor(out_f, 1);
    a_right.fill_xavier(out_f, 1, rng);

    grad_W = Tensor(in_f, out_f, 0.0);
    grad_a_left = Tensor(out_f, 1, 0.0);
    grad_a_right = Tensor(out_f, 1, 0.0);
}

Tensor GATLayer::forward(const Tensor& X, const Tensor& adj) {
    // X: N x in_features
    // adj: N x N binary adjacency (1 = edge, 0 = no edge; should include self-loops)
    size_t N = X.rows;
    if (X.cols != in_features) {
        throw std::runtime_error("GATLayer::forward feature width mismatch");
    }
    if (adj.rows != N || adj.cols != N) {
        throw std::runtime_error("GATLayer::forward adjacency shape mismatch");
    }

    cached_input = X;

    // Wh = X @ W  (N x out_features)
    cached_Wh = bio::affine(X, W, nullptr, false);

    // Compute attention scores:
    // e_{ij} = LeakyReLU(a_left^T Wh_i + a_right^T Wh_j)
    // Only for (i,j) where adj(i,j) > 0

    // left_scores = Wh @ a_left  (N x 1)
    Tensor left_scores = bio::affine(cached_Wh, a_left, nullptr, false);  // N x 1

    // right_scores = Wh @ a_right  (N x 1)
    Tensor right_scores = bio::affine(cached_Wh, a_right, nullptr, false); // N x 1

    // Compute raw attention: e(i,j) = left_scores(i) + right_scores(j)
    // Then apply LeakyReLU and mask by adjacency
    Tensor e(N, N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (adj(i, j) > 0.0) {
                double val = left_scores(i, 0) + right_scores(j, 0);
                // LeakyReLU
                e(i, j) = (val >= 0.0) ? val : negative_slope * val;
            }
        }
    }

    // Softmax over neighbors (masked by adjacency)
    // alpha_{ij} = exp(e_{ij}) / sum_k(exp(e_{ik})) where adj(i,k) > 0
    cached_attn = Tensor(N, N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        // Find max for numerical stability
        double max_e = -1e30;
        for (size_t j = 0; j < N; ++j) {
            if (adj(i, j) > 0.0)
                max_e = std::max(max_e, e(i, j));
        }

        double sum_exp = 0.0;
        for (size_t j = 0; j < N; ++j) {
            if (adj(i, j) > 0.0) {
                cached_attn(i, j) = std::exp(e(i, j) - max_e);
                sum_exp += cached_attn(i, j);
            }
        }
        if (sum_exp > 0.0) {
            for (size_t j = 0; j < N; ++j)
                cached_attn(i, j) /= sum_exp;
        }
    }

    // Output: h'_i = sum_j alpha_{ij} * Wh_j
    // output = attn @ Wh  (N x out_features)
    Tensor output = cached_attn.matmul(cached_Wh);

    return output;
}

Tensor GATLayer::backward(const Tensor& grad, const Tensor& adj) {
    // grad: N x out_features
    // Forward was: output = attn @ Wh
    //   Wh = X @ W
    //   attn = softmax(e) where e_{ij} = LeakyReLU(a_left^T Wh_i + a_right^T Wh_j)
    //
    // This is a complex backward pass. We compute gradients for W, a_left, a_right.

    size_t N = grad.rows;

    // d(Loss)/d(Wh) from the output = attn @ Wh part:
    // d(Loss)/d(Wh)_matmul = attn^T @ grad   (N x out_features)
    Tensor grad_Wh_from_output = cached_attn.transpose().matmul(grad);

    // d(Loss)/d(attn) = grad @ Wh^T   (N x N)
    Tensor grad_attn = grad.matmul(cached_Wh.transpose()); // N x N

    // Backprop through softmax:
    // d(Loss)/d(e_{ij}) = sum_k [grad_attn(i,k) * attn(i,k) * (delta_{jk} - attn(i,j))]
    // Simplifies to: attn(i,j) * (grad_attn(i,j) - sum_k[grad_attn(i,k) * attn(i,k)])
    Tensor grad_e(N, N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        // Compute dot product of grad_attn[i,:] and attn[i,:]
        double dot = 0.0;
        for (size_t k = 0; k < N; ++k)
            dot += grad_attn(i, k) * cached_attn(i, k);

        for (size_t j = 0; j < N; ++j) {
            if (adj(i, j) > 0.0) {
                grad_e(i, j) = cached_attn(i, j) * (grad_attn(i, j) - dot);
            }
        }
    }

    // Backprop through LeakyReLU on e:
    // We need the pre-LeakyReLU values. Reconstruct from left_scores and right_scores.
    Tensor left_scores = cached_Wh.matmul(a_left);   // N x 1
    Tensor right_scores = cached_Wh.matmul(a_right);  // N x 1

    Tensor grad_e_pre_lrelu(N, N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (adj(i, j) > 0.0) {
                double val = left_scores(i, 0) + right_scores(j, 0);
                double lrelu_grad = (val >= 0.0) ? 1.0 : negative_slope;
                grad_e_pre_lrelu(i, j) = grad_e(i, j) * lrelu_grad;
            }
        }
    }

    // e_{ij} = a_left^T Wh_i + a_right^T Wh_j (before LeakyReLU)
    // d(Loss)/d(left_scores_i) = sum_j grad_e_pre_lrelu(i,j)
    // d(Loss)/d(right_scores_j) = sum_i grad_e_pre_lrelu(i,j)
    Tensor grad_left_scores(N, 1, 0.0);
    Tensor grad_right_scores(N, 1, 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (adj(i, j) > 0.0) {
                grad_left_scores(i, 0) += grad_e_pre_lrelu(i, j);
                grad_right_scores(j, 0) += grad_e_pre_lrelu(i, j);
            }
        }
    }

    // left_scores = Wh @ a_left => d/d(a_left) = Wh^T @ grad_left_scores
    // right_scores = Wh @ a_right => d/d(a_right) = Wh^T @ grad_right_scores
    grad_a_left = cached_Wh.transpose().matmul(grad_left_scores);   // out_features x 1
    grad_a_right = cached_Wh.transpose().matmul(grad_right_scores); // out_features x 1

    // d(Loss)/d(Wh) from attention path:
    // left_scores = Wh @ a_left => d(Loss)/d(Wh) += grad_left_scores @ a_left^T
    // right_scores = Wh @ a_right => d(Loss)/d(Wh) += grad_right_scores @ a_right^T
    Tensor grad_Wh_from_attn = grad_left_scores.matmul(a_left.transpose())
                              + grad_right_scores.matmul(a_right.transpose()); // N x out_features

    // Total gradient on Wh
    Tensor grad_Wh = grad_Wh_from_output + grad_Wh_from_attn; // N x out_features

    // Wh = X @ W => d(Loss)/d(W) = X^T @ grad_Wh
    grad_W = cached_input.transpose().matmul(grad_Wh);

    // d(Loss)/d(X) = grad_Wh @ W^T
    Tensor grad_X = grad_Wh.matmul(W.transpose()); // N x in_features

    return grad_X;
}

// ============================================================
// GNNClassifier
// ============================================================

GNNClassifier::GNNClassifier(size_t node_features, size_t hidden_size,
                             size_t num_classes, size_t num_gcn_layers,
                             std::mt19937& rng, size_t grid_h, size_t grid_w)
    : attn_fc(hidden_size, 1, "gnn_attn", rng),
      fc(hidden_size, num_classes, "gnn_fc", rng),
      num_nodes(grid_h * grid_w), num_classes(num_classes),
      hidden_size(hidden_size), adam_t_(0) {
    if (num_gcn_layers < 1) {
        throw std::runtime_error("GNNClassifier requires at least one GCN layer");
    }
    if (grid_h == 0 || grid_w == 0) {
        throw std::runtime_error("GNNClassifier requires non-zero grid dimensions");
    }

    // First GCN layer: node_features -> hidden_size
    gcn_layers.emplace_back(node_features, hidden_size, rng);

    // Subsequent GCN layers: hidden_size -> hidden_size
    for (size_t i = 1; i < num_gcn_layers; ++i)
        gcn_layers.emplace_back(hidden_size, hidden_size, rng);

    // Precompute the normalized adjacency matrix for the grid
    Tensor adj = create_grid_adjacency(grid_h, grid_w);
    adj_norm = normalize_adjacency(adj);
}

Tensor GNNClassifier::create_grid_adjacency(size_t h, size_t w) {
    size_t N = h * w;
    Tensor adj(N, N, 0.0);

    // 8-connected grid with self-loops
    for (size_t row = 0; row < h; ++row) {
        for (size_t col = 0; col < w; ++col) {
            size_t idx = row * w + col;

            // Self-loop
            adj(idx, idx) = 1.0;

            // 8 neighbors
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    int nr = (int)row + dr;
                    int nc = (int)col + dc;
                    if (nr >= 0 && nr < (int)h && nc >= 0 && nc < (int)w) {
                        size_t nidx = (size_t)nr * w + (size_t)nc;
                        adj(idx, nidx) = 1.0;
                    }
                }
            }
        }
    }
    return adj;
}

Tensor GNNClassifier::normalize_adjacency(const Tensor& adj) {
    // Symmetric normalization: D^{-1/2} A D^{-1/2}
    // A should already include self-loops.
    size_t N = adj.rows;
    if (adj.cols != N) {
        throw std::runtime_error("GNNClassifier::normalize_adjacency requires square adjacency");
    }

    // Compute degree vector
    std::vector<double> deg(N, 0.0);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            deg[i] += adj(i, j);

    // D^{-1/2}
    std::vector<double> deg_inv_sqrt(N, 0.0);
    for (size_t i = 0; i < N; ++i) {
        if (deg[i] > 0.0)
            deg_inv_sqrt[i] = 1.0 / std::sqrt(deg[i]);
    }

    // A_norm = D^{-1/2} A D^{-1/2}
    Tensor norm(N, N, 0.0);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            norm(i, j) = deg_inv_sqrt[i] * adj(i, j) * deg_inv_sqrt[j];

    return norm;
}

Tensor GNNClassifier::forward_single(const Tensor& image) {
    // image: 1 x (num_nodes * node_features) or num_nodes x node_features
    // Reshape to N x F (each node has F features)
    size_t node_feat = gcn_layers[0].parameters()[0]->rows; // in_features of first GCN layer
    Tensor X;
    if (image.rows == 1 && image.cols == num_nodes * node_feat) {
        // 1 x (N*F) -> N x F
        X = Tensor(num_nodes, node_feat);
        for (size_t i = 0; i < num_nodes; ++i)
            for (size_t j = 0; j < node_feat; ++j)
                X(i, j) = image(0, i * node_feat + j);
    } else if (image.rows == num_nodes && image.cols == node_feat) {
        X = image;
    } else if (image.rows == 1 && image.cols == num_nodes && node_feat == 1) {
        X = image.transpose();
    } else {
        // Try reshape
        X = Tensor(num_nodes, node_feat);
        size_t total = image.rows * image.cols;
        if (total != num_nodes * node_feat) {
            throw std::runtime_error("GNNClassifier::forward_single cannot reshape input into num_nodes x node_features");
        }
        for (size_t i = 0; i < total; ++i)
            X.data[i] = image.data[i];
    }

    // Save inputs for backward pass
    cached_gcn_inputs.clear();
    cached_gcn_inputs.push_back(X);

    // Pass through GCN layers with LeakyReLU activations (slope=0.01)
    // LeakyReLU prevents dying neurons, critical for GNN gradient flow
    Tensor H = X;
    for (size_t l = 0; l < gcn_layers.size(); ++l) {
        H = gcn_layers[l].forward(H, adj_norm);
        for (size_t i = 0; i < H.data.size(); ++i)
            H.data[i] = (H.data[i] > 0.0) ? H.data[i] : 0.01 * H.data[i];
        cached_gcn_inputs.push_back(H);
    }

    // Attention-based graph readout:
    // alpha_i = softmax(attn_fc(h_i)) for all nodes i
    // pooled = sum_i(alpha_i * h_i)
    cached_final_H = H;  // N x hidden_size

    // Compute attention scores: attn_fc(H) gives N x 1
    Tensor attn_scores = attn_fc.forward(H);  // N x 1

    // Softmax over nodes
    double max_score = attn_scores(0, 0);
    for (size_t i = 1; i < num_nodes; ++i)
        max_score = std::max(max_score, attn_scores(i, 0));
    double sum_exp = 0.0;
    cached_attn_weights = Tensor(num_nodes, 1);
    for (size_t i = 0; i < num_nodes; ++i) {
        cached_attn_weights(i, 0) = std::exp(attn_scores(i, 0) - max_score);
        sum_exp += cached_attn_weights(i, 0);
    }
    for (size_t i = 0; i < num_nodes; ++i)
        cached_attn_weights(i, 0) /= sum_exp;

    // Weighted sum: pooled = sum_i(alpha_i * h_i)
    cached_pooled = Tensor(1, hidden_size, 0.0);
    for (size_t i = 0; i < num_nodes; ++i)
        for (size_t j = 0; j < hidden_size; ++j)
            cached_pooled(0, j) += cached_attn_weights(i, 0) * H(i, j);

    // Classification: FC layer
    Tensor logits = fc.forward(cached_pooled); // 1 x num_classes
    return logits;
}

std::vector<int> GNNClassifier::predict(const Tensor& images) {
    // images: batch x 784
    size_t batch = images.rows;
    std::vector<int> predictions(batch);

    for (size_t b = 0; b < batch; ++b) {
        Tensor img = images.row(b); // 1 x 784
        Tensor logits = forward_single(img);

        // Find argmax
        int best = 0;
        double best_val = logits(0, 0);
        for (size_t c = 1; c < num_classes; ++c) {
            if (logits(0, c) > best_val) {
                best_val = logits(0, c);
                best = (int)c;
            }
        }
        predictions[b] = best;
    }
    return predictions;
}

void GNNClassifier::train_step(const Tensor& images, const std::vector<int>& labels,
                                double lr) {
    size_t batch = images.rows;
    if (labels.size() != batch) {
        throw std::runtime_error("GNNClassifier::train_step labels size mismatch");
    }

    // Mini-batch SGD: accumulate gradients over mini_batch_size samples, then update
    const size_t mini_batch_size = 16;

    auto all_params = parameters();
    auto all_grads = gradients();

    for (size_t start = 0; start < batch; start += mini_batch_size) {
        size_t mb = std::min(mini_batch_size, batch - start);

        // Zero gradients and create accumulator
        zero_grad();
        std::vector<Tensor> accum_grads(all_grads.size());
        for (size_t i = 0; i < all_grads.size(); ++i)
            accum_grads[i] = Tensor(all_grads[i]->rows, all_grads[i]->cols, 0.0);

        for (size_t b = start; b < start + mb; ++b) {
            Tensor img = images.row(b);
            Tensor logits = forward_single(img);
            Tensor probs = LossFunctions::softmax(logits);

            Tensor target(1, num_classes, 0.0);
            target(0, (size_t)labels[b]) = 1.0;

            Tensor grad_logits = (probs - target) / (double)mb;

            // Backward through FC
            Tensor grad_pooled = fc.backward(grad_logits);

            // Backward through attention readout
            // pooled = sum_i(alpha_i * H_i)
            // d(pooled)/d(H_i) = alpha_i (for each feature)
            // d(pooled)/d(alpha_i) = H_i
            size_t hidden = grad_pooled.cols;

            // Grad w.r.t. H from the weighted sum
            Tensor grad_H(num_nodes, hidden, 0.0);
            for (size_t i = 0; i < num_nodes; ++i)
                for (size_t j = 0; j < hidden; ++j)
                    grad_H(i, j) = cached_attn_weights(i, 0) * grad_pooled(0, j);

            // Grad w.r.t. attention weights
            // d(loss)/d(alpha_i) = sum_j(grad_pooled_j * H_{i,j})
            Tensor grad_alpha(num_nodes, 1, 0.0);
            for (size_t i = 0; i < num_nodes; ++i)
                for (size_t j = 0; j < hidden; ++j)
                    grad_alpha(i, 0) += grad_pooled(0, j) * cached_final_H(i, j);

            // Backprop through softmax
            // d(alpha_i)/d(score_j) = alpha_i * (delta_ij - alpha_j)
            Tensor grad_scores(num_nodes, 1, 0.0);
            double dot = 0.0;
            for (size_t i = 0; i < num_nodes; ++i)
                dot += grad_alpha(i, 0) * cached_attn_weights(i, 0);
            for (size_t i = 0; i < num_nodes; ++i)
                grad_scores(i, 0) = cached_attn_weights(i, 0) * (grad_alpha(i, 0) - dot);

            // Backprop through attn_fc: grad w.r.t. H from attention path
            Tensor grad_H_from_attn = attn_fc.backward(grad_scores); // N x hidden
            for (size_t i = 0; i < num_nodes * hidden; ++i)
                grad_H.data[i] += grad_H_from_attn.data[i];

            // Backward through GCN layers (LeakyReLU backward)
            for (int l = (int)gcn_layers.size() - 1; l >= 0; --l) {
                Tensor& post_act = cached_gcn_inputs[(size_t)l + 1];
                for (size_t i = 0; i < grad_H.data.size(); ++i)
                    grad_H.data[i] *= (post_act.data[i] > 0.0) ? 1.0 : 0.01;
                grad_H = gcn_layers[(size_t)l].backward(grad_H, adj_norm);
            }

            // Accumulate gradients
            auto sample_grads = gradients();
            for (size_t i = 0; i < accum_grads.size(); ++i) {
                for (size_t k = 0; k < accum_grads[i].data.size(); ++k)
                    accum_grads[i].data[k] += sample_grads[i]->data[k];
            }
        }

        // Adam update with accumulated gradients
        adam_t_++;
        if (adam_m_.empty()) {
            for (auto* p : all_params) {
                adam_m_.push_back(Tensor::zeros(p->rows, p->cols));
                adam_v_.push_back(Tensor::zeros(p->rows, p->cols));
            }
        }
        for (size_t i = 0; i < all_params.size(); ++i) {
            for (size_t k = 0; k < all_params[i]->data.size(); ++k) {
                double g = accum_grads[i].data[k];
                adam_m_[i].data[k] = 0.9 * adam_m_[i].data[k] + 0.1 * g;
                adam_v_[i].data[k] = 0.999 * adam_v_[i].data[k] + 0.001 * g * g;
                double mh = adam_m_[i].data[k] / (1.0 - std::pow(0.9, adam_t_));
                double vh = adam_v_[i].data[k] / (1.0 - std::pow(0.999, adam_t_));
                all_params[i]->data[k] -= lr * mh / (std::sqrt(vh) + 1e-8);
            }
        }
    }
}

std::vector<Tensor*> GNNClassifier::parameters() {
    std::vector<Tensor*> params;
    for (auto& layer : gcn_layers)
        for (auto* p : layer.parameters()) params.push_back(p);
    for (auto* p : attn_fc.parameters()) params.push_back(p);
    for (auto* p : fc.parameters()) params.push_back(p);
    return params;
}

std::vector<Tensor*> GNNClassifier::gradients() {
    std::vector<Tensor*> grads;
    for (auto& layer : gcn_layers)
        for (auto* g : layer.gradients()) grads.push_back(g);
    for (auto* g : attn_fc.gradients()) grads.push_back(g);
    for (auto* g : fc.gradients()) grads.push_back(g);
    return grads;
}

void GNNClassifier::zero_grad() {
    auto grads = gradients();
    for (auto* g : grads)
        g->fill_zeros();
}
