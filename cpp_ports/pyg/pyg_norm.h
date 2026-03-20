// pyg_norm.h -- Normalization layers for C++ PyG port
// Header-only C++17. Depends on pyg.h
//
// Implements: GraphNorm, InstanceNorm, LayerNorm, PairNorm,
//             GraphSizeNorm, DiffGroupNorm, MessageNorm
#pragma once
#include "pyg.h"

namespace pyg {

// =====================================================================
//  GraphNorm -- "GraphNorm: A Principled Approach to Accelerating Graph
//               Neural Network Training" (Cai et al., 2020)
//  Normalizes across all nodes in each graph, with a learnable shift alpha.
// =====================================================================
class GraphNorm {
public:
    int channels_; float eps_;
    TensorPtr gamma, beta, alpha;

    GraphNorm()=default;
    GraphNorm(int channels, float eps=1e-5f):channels_(channels),eps_(eps) {
        gamma=Tensor::ones(1,channels); gamma->set_requires_grad(true);
        beta=Tensor::zeros(1,channels); beta->set_requires_grad(true);
        alpha=std::make_shared<Tensor>(1,1,1.0f); alpha->set_requires_grad(true);
    }

    TensorPtr forward(const TensorPtr& x, const std::vector<int>& batch, int num_graphs) {
        int N=x->rows(), C=x->cols();
        auto out=std::make_shared<Tensor>(N,C);
        std::vector<std::vector<int>> gn(num_graphs);
        for (int i=0;i<N;++i) gn[batch[i]].push_back(i);
        float a=(*alpha)(0,0);
        for (int g=0;g<num_graphs;++g) {
            if (gn[g].empty()) continue;
            int gs=(int)gn[g].size();
            for (int j=0;j<C;++j) {
                // Compute mean and var for this graph, this channel
                float mean=0;
                for (int ni:gn[g]) mean+=(*x)(ni,j); mean/=gs;
                float var=0;
                for (int ni:gn[g]) { float d=(*x)(ni,j)-a*mean; var+=d*d; } var/=gs;
                float inv=1.0f/std::sqrt(var+eps_);
                for (int ni:gn[g]) (*out)(ni,j)=((*x)(ni,j)-a*mean)*inv*(*gamma)(0,j)+(*beta)(0,j);
            }
        }
        return out;
    }
    std::vector<TensorPtr> parameters() const { return {gamma,beta,alpha}; }
};

// =====================================================================
//  InstanceNorm -- per-graph per-channel normalization
//  (equivalent to InstanceNorm1d applied independently per graph)
// =====================================================================
class InstanceNorm {
public:
    int channels_; float eps_,mom_; bool affine_;
    TensorPtr gamma, beta;

    InstanceNorm()=default;
    InstanceNorm(int channels, float eps=1e-5f, float mom=0.1f, bool affine=true)
        :channels_(channels),eps_(eps),mom_(mom),affine_(affine) {
        if (affine) { gamma=Tensor::ones(1,channels); gamma->set_requires_grad(true);
                      beta=Tensor::zeros(1,channels); beta->set_requires_grad(true); }
    }

    TensorPtr forward(const TensorPtr& x, const std::vector<int>& batch, int num_graphs) {
        int N=x->rows(), C=x->cols();
        auto out=std::make_shared<Tensor>(N,C);
        std::vector<std::vector<int>> gn(num_graphs);
        for (int i=0;i<N;++i) gn[batch[i]].push_back(i);
        for (int g=0;g<num_graphs;++g) {
            if (gn[g].empty()) continue;
            int gs=(int)gn[g].size();
            for (int j=0;j<C;++j) {
                float mean=0;
                for (int ni:gn[g]) mean+=(*x)(ni,j); mean/=gs;
                float var=0;
                for (int ni:gn[g]) { float d=(*x)(ni,j)-mean; var+=d*d; } var/=gs;
                float inv=1.0f/std::sqrt(var+eps_);
                for (int ni:gn[g]) {
                    float v=((*x)(ni,j)-mean)*inv;
                    if (affine_) v=v*(*gamma)(0,j)+(*beta)(0,j);
                    (*out)(ni,j)=v;
                }
            }
        }
        return out;
    }
    std::vector<TensorPtr> parameters() const { std::vector<TensorPtr> p; if (affine_) { p.push_back(gamma); p.push_back(beta); } return p; }
};

// =====================================================================
//  LayerNorm -- applies layer normalization per node
// =====================================================================
class GraphLayerNorm {
public:
    int channels_; float eps_; bool affine_;
    TensorPtr gamma, beta;

    GraphLayerNorm()=default;
    GraphLayerNorm(int channels, float eps=1e-5f, bool affine=true)
        :channels_(channels),eps_(eps),affine_(affine) {
        if (affine) { gamma=Tensor::ones(1,channels); gamma->set_requires_grad(true);
                      beta=Tensor::zeros(1,channels); beta->set_requires_grad(true); }
    }

    TensorPtr forward(const TensorPtr& x) {
        int N=x->rows(), C=x->cols();
        auto out=std::make_shared<Tensor>(N,C);
        for (int i=0;i<N;++i) {
            float mean=0; for (int j=0;j<C;++j) mean+=(*x)(i,j); mean/=C;
            float var=0; for (int j=0;j<C;++j) { float d=(*x)(i,j)-mean; var+=d*d; } var/=C;
            float inv=1.0f/std::sqrt(var+eps_);
            for (int j=0;j<C;++j) {
                float v=((*x)(i,j)-mean)*inv;
                if (affine_) v=v*(*gamma)(0,j)+(*beta)(0,j);
                (*out)(i,j)=v;
            }
        }
        if (x->requires_grad_) {
            out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={x};
            int Nr=N,Cr=C; float epsr=eps_;
            nd->backward_fn=[Nr,Cr,epsr,x](const Tensor& g)->std::vector<TensorPtr> {
                auto dX=std::make_shared<Tensor>(Nr,Cr);
                for (int i=0;i<Nr;++i) {
                    float mean=0; for (int j=0;j<Cr;++j) mean+=(*x)(i,j); mean/=Cr;
                    float var=0; for (int j=0;j<Cr;++j) { float d=(*x)(i,j)-mean; var+=d*d; } var/=Cr;
                    float inv=1.0f/std::sqrt(var+epsr);
                    float gsum=0,gxsum=0;
                    for (int j=0;j<Cr;++j) { gsum+=g(i,j); gxsum+=g(i,j)*((*x)(i,j)-mean); }
                    for (int j=0;j<Cr;++j) (*dX)(i,j)=inv*(g(i,j)-gsum/Cr-((*x)(i,j)-mean)*gxsum*inv*inv/Cr);
                }
                return {dX};
            }; out->grad_node_=nd;
        }
        return out;
    }
    std::vector<TensorPtr> parameters() const { std::vector<TensorPtr> p; if (affine_) { p.push_back(gamma); p.push_back(beta); } return p; }
};

// =====================================================================
//  PairNorm -- "PairNorm: Tackling Oversmoothing in GNNs" (Zhao & Akoglu)
//  Centers and rescales node features to prevent oversmoothing.
// =====================================================================
class PairNorm {
public:
    float scale_; bool scale_individually_;

    PairNorm(float scale=1.0f, bool si=false):scale_(scale),scale_individually_(si){}

    TensorPtr forward(const TensorPtr& x) {
        int N=x->rows(), C=x->cols();
        auto out=std::make_shared<Tensor>(N,C);
        // Center
        std::vector<float> mean(C,0);
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) mean[j]+=(*x)(i,j);
        for (int j=0;j<C;++j) mean[j]/=N;
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*out)(i,j)=(*x)(i,j)-mean[j];
        // Scale
        if (scale_individually_) {
            for (int i=0;i<N;++i) {
                float norm=0; for (int j=0;j<C;++j) norm+=(*out)(i,j)*(*out)(i,j);
                norm=std::sqrt(norm/C)+1e-12f;
                for (int j=0;j<C;++j) (*out)(i,j)=(*out)(i,j)*scale_/norm;
            }
        } else {
            float norm=0; for (int i=0;i<N;++i) for (int j=0;j<C;++j) norm+=(*out)(i,j)*(*out)(i,j);
            norm=std::sqrt(norm/(N*C))+1e-12f;
            for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*out)(i,j)*=scale_/norm;
        }
        return out;
    }
    std::vector<TensorPtr> parameters() const { return {}; }
};

// =====================================================================
//  GraphSizeNorm -- normalizes by sqrt(|V_g|) per graph
// =====================================================================
class GraphSizeNorm {
public:
    GraphSizeNorm()=default;

    TensorPtr forward(const TensorPtr& x, const std::vector<int>& batch, int num_graphs) {
        int N=x->rows(), C=x->cols();
        std::vector<int> gsz(num_graphs,0);
        for (int i=0;i<N;++i) gsz[batch[i]]++;
        auto out=std::make_shared<Tensor>(N,C);
        for (int i=0;i<N;++i) { float s=1.0f/std::sqrt((float)gsz[batch[i]]);
            for (int j=0;j<C;++j) (*out)(i,j)=(*x)(i,j)*s; }
        return out;
    }
    std::vector<TensorPtr> parameters() const { return {}; }
};

// =====================================================================
//  DiffGroupNorm -- "Towards Deeper Graph Neural Networks with
//                    Differentiable Group Normalization" (Zhou et al.)
//  Clusters nodes into groups, normalizes within groups.
// =====================================================================
class DiffGroupNorm {
public:
    int channels_,num_groups_; float eps_;
    Linear group_lin;
    TensorPtr gamma, beta;

    DiffGroupNorm()=default;
    DiffGroupNorm(int channels, int num_groups=4, float eps=1e-5f)
        :channels_(channels),num_groups_(num_groups),eps_(eps),group_lin(channels,num_groups,true) {
        gamma=Tensor::ones(1,channels); gamma->set_requires_grad(true);
        beta=Tensor::zeros(1,channels); beta->set_requires_grad(true);
    }

    TensorPtr forward(const TensorPtr& x) {
        int N=x->rows(), C=x->cols(), G=num_groups_;
        // Soft group assignment
        auto logits=group_lin.forward(x); // (N, G)
        auto assign=autograd::softmax(logits); // (N, G)
        // For each group, compute weighted mean and var
        auto out=Tensor::zeros(N,C);
        for (int g=0;g<G;++g) {
            float total_w=0;
            std::vector<float> wmean(C,0);
            for (int i=0;i<N;++i) { float w=(*assign)(i,g); total_w+=w;
                for (int j=0;j<C;++j) wmean[j]+=w*(*x)(i,j); }
            if (total_w>1e-12f) for (int j=0;j<C;++j) wmean[j]/=total_w;
            std::vector<float> wvar(C,0);
            for (int i=0;i<N;++i) { float w=(*assign)(i,g);
                for (int j=0;j<C;++j) { float d=(*x)(i,j)-wmean[j]; wvar[j]+=w*d*d; } }
            if (total_w>1e-12f) for (int j=0;j<C;++j) wvar[j]/=total_w;
            for (int i=0;i<N;++i) { float w=(*assign)(i,g);
                for (int j=0;j<C;++j) (*out)(i,j)+=w*((*x)(i,j)-wmean[j])/std::sqrt(wvar[j]+eps_); }
        }
        // Affine
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*out)(i,j)=(*out)(i,j)*(*gamma)(0,j)+(*beta)(0,j);
        return out;
    }
    std::vector<TensorPtr> parameters() const {
        auto p=group_lin.parameters(); p.push_back(gamma); p.push_back(beta); return p;
    }
};

// =====================================================================
//  MessageNorm -- normalizes messages in message-passing layers
//  "DeeperGCN: All You Need to Train Deeper GCNs" (Li et al.)
// =====================================================================
class MessageNorm {
public:
    bool learn_scale_;
    TensorPtr scale;

    MessageNorm(bool learn_scale=true):learn_scale_(learn_scale) {
        scale=std::make_shared<Tensor>(1,1,1.0f);
        if (learn_scale) scale->set_requires_grad(true);
    }

    TensorPtr forward(const TensorPtr& x, const TensorPtr& msg) {
        int N=x->rows(), C=x->cols();
        auto out=std::make_shared<Tensor>(N,C);
        float s=(*scale)(0,0);
        for (int i=0;i<N;++i) {
            // x_norm / msg_norm ratio
            float xn=0,mn=0;
            for (int j=0;j<C;++j) { xn+=(*x)(i,j)*(*x)(i,j); mn+=(*msg)(i,j)*(*msg)(i,j); }
            xn=std::sqrt(xn)+1e-12f; mn=std::sqrt(mn)+1e-12f;
            float ratio=s*xn/mn;
            for (int j=0;j<C;++j) (*out)(i,j)=(*msg)(i,j)*ratio;
        }
        return out;
    }
    std::vector<TensorPtr> parameters() const { return learn_scale_?std::vector<TensorPtr>{scale}:std::vector<TensorPtr>{}; }
};

// =====================================================================
//  MeanSubtractionNorm -- simple centering norm
// =====================================================================
class MeanSubtractionNorm {
public:
    MeanSubtractionNorm()=default;

    TensorPtr forward(const TensorPtr& x) {
        int N=x->rows(), C=x->cols();
        auto out=std::make_shared<Tensor>(N,C);
        std::vector<float> mean(C,0);
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) mean[j]+=(*x)(i,j);
        for (int j=0;j<C;++j) mean[j]/=N;
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*out)(i,j)=(*x)(i,j)-mean[j];
        return out;
    }
    std::vector<TensorPtr> parameters() const { return {}; }
};

} // namespace pyg
