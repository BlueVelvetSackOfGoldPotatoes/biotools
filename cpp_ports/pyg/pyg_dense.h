// pyg_dense.h -- Dense graph convolution & pooling layers for C++ PyG port
// Header-only C++17. Depends on pyg.h
//
// Implements: DenseGCNConv, DenseGINConv, DenseSAGEConv,
//             dense_diff_pool, dense_mincut_pool
#pragma once
#include "pyg.h"

namespace pyg {
namespace dense {

// =====================================================================
//  DenseGCNConv -- GCN on dense adjacency matrix
//  Operates on batched tensors: X (B, N, F), Adj (B, N, N) -> (B, N, F')
//  For simplicity, we use single-graph (N, F) interface with adj (N, N).
// =====================================================================
class DenseGCNConv {
public:
    Linear lin; int in_c_,out_c_; bool add_sl_;
    DenseGCNConv()=default;
    DenseGCNConv(int in,int out,bool bias=true,bool add_self_loops=true)
        :lin(in,out,bias),in_c_(in),out_c_(out),add_sl_(add_self_loops){}

    TensorPtr forward(const TensorPtr& x, const TensorPtr& adj) {
        int N=x->rows(), C=x->cols(), OC=out_c_;
        // A_hat = adj + I (if add_self_loops)
        auto A=std::make_shared<Tensor>(N,N);
        for (int i=0;i<N;++i) for (int j=0;j<N;++j) (*A)(i,j)=(*adj)(i,j);
        if (add_sl_) for (int i=0;i<N;++i) (*A)(i,i)+=1.0f;
        // D^{-1/2} A D^{-1/2}
        std::vector<float> deg(N,0);
        for (int i=0;i<N;++i) for (int j=0;j<N;++j) deg[i]+=(*A)(i,j);
        std::vector<float> dis(N,0);
        for (int i=0;i<N;++i) if (deg[i]>0) dis[i]=1.0f/std::sqrt(deg[i]);
        for (int i=0;i<N;++i) for (int j=0;j<N;++j) (*A)(i,j)*=dis[i]*dis[j];
        // H = A * X
        auto h=std::make_shared<Tensor>(N,C,0.0f);
        for (int i=0;i<N;++i) for (int k=0;k<N;++k) { float a=(*A)(i,k);
            if (a!=0) for (int j=0;j<C;++j) (*h)(i,j)+=a*(*x)(k,j); }
        return lin.forward(h);
    }
    std::vector<TensorPtr> parameters() const { return lin.parameters(); }
};

// =====================================================================
//  DenseGINConv -- GIN on dense adjacency
// =====================================================================
class DenseGINConv {
public:
    Linear mlp1,mlp2; float eps_;
    DenseGINConv()=default;
    DenseGINConv(int in,int out,float eps=0):mlp1(in,out,true),mlp2(out,out,true),eps_(eps){}

    TensorPtr forward(const TensorPtr& x, const TensorPtr& adj) {
        int N=x->rows(), C=x->cols();
        // Agg = A * X
        auto agg=std::make_shared<Tensor>(N,C,0.0f);
        for (int i=0;i<N;++i) for (int k=0;k<N;++k) { float a=(*adj)(i,k);
            if (a!=0) for (int j=0;j<C;++j) (*agg)(i,j)+=a*(*x)(k,j); }
        // (1+eps)*x + agg
        auto combined=std::make_shared<Tensor>(N,C);
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*combined)(i,j)=(1+eps_)*(*x)(i,j)+(*agg)(i,j);
        return mlp2.forward(autograd::relu(mlp1.forward(combined)));
    }
    std::vector<TensorPtr> parameters() const {
        auto p=mlp1.parameters(); auto p2=mlp2.parameters(); p.insert(p.end(),p2.begin(),p2.end()); return p;
    }
};

// =====================================================================
//  DenseSAGEConv -- GraphSAGE on dense adjacency
// =====================================================================
class DenseSAGEConv {
public:
    Linear lin_l,lin_r; bool normalize_;
    DenseSAGEConv()=default;
    DenseSAGEConv(int in,int out,bool norm=false,bool bias=true)
        :lin_l(in,out,bias),lin_r(in,out,false),normalize_(norm){}

    TensorPtr forward(const TensorPtr& x, const TensorPtr& adj) {
        int N=x->rows(), C=x->cols();
        // Mean aggregation: agg_i = sum_j A[i,j]*x_j / sum_j A[i,j]
        auto agg=std::make_shared<Tensor>(N,C,0.0f);
        std::vector<float> deg(N,0);
        for (int i=0;i<N;++i) for (int k=0;k<N;++k) { float a=(*adj)(i,k);
            if (a!=0) { deg[i]+=a; for (int j=0;j<C;++j) (*agg)(i,j)+=a*(*x)(k,j); } }
        for (int i=0;i<N;++i) if (deg[i]>0) for (int j=0;j<C;++j) (*agg)(i,j)/=deg[i];
        auto out=autograd::add(lin_l.forward(agg),lin_r.forward(x));
        if (normalize_) { int OC=out->cols();
            for (int i=0;i<N;++i) { float n=0; for (int j=0;j<OC;++j) n+=(*out)(i,j)*(*out)(i,j);
                n=std::sqrt(n)+1e-12f; for (int j=0;j<OC;++j) (*out)(i,j)/=n; } }
        return out;
    }
    std::vector<TensorPtr> parameters() const { auto p=lin_l.parameters(); auto pr=lin_r.parameters(); p.insert(p.end(),pr.begin(),pr.end()); return p; }
};

// =====================================================================
//  DenseGATConv -- GAT on dense adjacency
// =====================================================================
class DenseGATConv {
public:
    Linear lin; TensorPtr att_l,att_r; int in_c_,out_c_,heads_; float neg_slope_; bool concat_;
    DenseGATConv()=default;
    DenseGATConv(int in,int out,int heads=1,bool concat=true,float ns=0.2f)
        :lin(in,heads*out,false),in_c_(in),out_c_(out),heads_(heads),neg_slope_(ns),concat_(concat) {
        att_l=Tensor::xavier_uniform(1,heads*out); att_l->set_requires_grad(true);
        att_r=Tensor::xavier_uniform(1,heads*out); att_r->set_requires_grad(true);
    }

    TensorPtr forward(const TensorPtr& x, const TensorPtr& adj) {
        int N=x->rows(),H=heads_,C=out_c_,HC=H*C;
        auto Wx=lin.forward(x); // (N, HC)
        // Attention
        int od=concat_?HC:C;
        auto out=Tensor::zeros(N,od);
        for (int h=0;h<H;++h) {
            // Compute e_ij = LeakyReLU(a_l^T Wx_i + a_r^T Wx_j)
            auto e=std::make_shared<Tensor>(N,N);
            for (int i=0;i<N;++i) { float sl=0; for (int c=0;c<C;++c) sl+=(*Wx)(i,h*C+c)*(*att_l)(0,h*C+c);
                for (int j=0;j<N;++j) { if ((*adj)(i,j)==0) { (*e)(i,j)=-1e30f; continue; }
                    float sr=0; for (int c=0;c<C;++c) sr+=(*Wx)(j,h*C+c)*(*att_r)(0,h*C+c);
                    float v=sl+sr; (*e)(i,j)=(v>0)?v:neg_slope_*v; } }
            // Softmax per row (masking zeros)
            for (int i=0;i<N;++i) { float mx=-1e30f; for (int j=0;j<N;++j) mx=std::max(mx,(*e)(i,j));
                float s=0; for (int j=0;j<N;++j) { (*e)(i,j)=std::exp((*e)(i,j)-mx); s+=(*e)(i,j); }
                for (int j=0;j<N;++j) (*e)(i,j)/=(s+1e-12f); }
            // Aggregate
            for (int i=0;i<N;++i) for (int j=0;j<N;++j) { float a=(*e)(i,j);
                for (int c=0;c<C;++c) { if (concat_) (*out)(i,h*C+c)+=a*(*Wx)(j,h*C+c);
                    else (*out)(i,c)+=a*(*Wx)(j,h*C+c)/H; } }
        }
        return out;
    }
    std::vector<TensorPtr> parameters() const { auto p=lin.parameters(); p.push_back(att_l); p.push_back(att_r); return p; }
};

// =====================================================================
//  dense_diff_pool -- "Hierarchical Graph Representation Learning with
//                      Differentiable Pooling" (Ying et al., 2018)
// =====================================================================
struct DiffPoolResult { TensorPtr x; TensorPtr adj; float link_loss; float entropy_loss; };

inline DiffPoolResult dense_diff_pool(const TensorPtr& x, const TensorPtr& adj, const TensorPtr& s) {
    // x: (N, F), adj: (N, N), s: (N, K) soft assignment matrix
    int N=x->rows(), F=x->cols(), K=s->cols();
    // S^T * X -> (K, F)
    auto new_x=std::make_shared<Tensor>(K,F,0.0f);
    for (int k=0;k<K;++k) for (int j=0;j<F;++j) { float v=0; for (int i=0;i<N;++i) v+=(*s)(i,k)*(*x)(i,j); (*new_x)(k,j)=v; }
    // S^T * A * S -> (K, K)
    // First compute A*S -> (N, K)
    auto AS=std::make_shared<Tensor>(N,K,0.0f);
    for (int i=0;i<N;++i) for (int k=0;k<K;++k) { float v=0; for (int j=0;j<N;++j) v+=(*adj)(i,j)*(*s)(j,k); (*AS)(i,k)=v; }
    // Then S^T * (A*S) -> (K, K)
    auto new_adj=std::make_shared<Tensor>(K,K,0.0f);
    for (int ki=0;ki<K;++ki) for (int kj=0;kj<K;++kj) { float v=0; for (int i=0;i<N;++i) v+=(*s)(i,ki)*(*AS)(i,kj); (*new_adj)(ki,kj)=v; }
    // Link prediction loss: ||A - S*S^T||_F^2 / (N*N)
    float link_loss=0;
    for (int i=0;i<N;++i) for (int j=0;j<N;++j) {
        float sst=0; for (int k=0;k<K;++k) sst+=(*s)(i,k)*(*s)(j,k);
        float d=(*adj)(i,j)-sst; link_loss+=d*d;
    }
    link_loss/=(N*N);
    // Entropy loss: H(S) = -sum(S * log(S+eps)) / N
    float entropy_loss=0;
    for (int i=0;i<N;++i) for (int k=0;k<K;++k) {
        float v=std::max(1e-7f,(*s)(i,k)); entropy_loss-=v*std::log(v);
    }
    entropy_loss/=N;
    return {new_x,new_adj,link_loss,entropy_loss};
}

// =====================================================================
//  dense_mincut_pool -- "Spectral Clustering with Graph Neural Networks
//                        for Graph Pooling" (Bianchi et al., 2020)
// =====================================================================
struct MinCutPoolResult { TensorPtr x; TensorPtr adj; float mincut_loss; float ortho_loss; };

inline MinCutPoolResult dense_mincut_pool(const TensorPtr& x, const TensorPtr& adj, const TensorPtr& s) {
    int N=x->rows(), F=x->cols(), K=s->cols();
    // Soft assignments via softmax
    auto s_soft=autograd::softmax(s);
    // New features: S^T X
    auto new_x=std::make_shared<Tensor>(K,F,0.0f);
    for (int k=0;k<K;++k) for (int j=0;j<F;++j) { float v=0; for (int i=0;i<N;++i) v+=(*s_soft)(i,k)*(*x)(i,j); (*new_x)(k,j)=v; }
    // New adjacency: S^T A S
    auto AS=std::make_shared<Tensor>(N,K,0.0f);
    for (int i=0;i<N;++i) for (int k=0;k<K;++k) { float v=0; for (int j=0;j<N;++j) v+=(*adj)(i,j)*(*s_soft)(j,k); (*AS)(i,k)=v; }
    auto new_adj=std::make_shared<Tensor>(K,K,0.0f);
    for (int ki=0;ki<K;++ki) for (int kj=0;kj<K;++kj) { float v=0; for (int i=0;i<N;++i) v+=(*s_soft)(i,ki)*(*AS)(i,kj); (*new_adj)(ki,kj)=v; }
    // MinCut loss: -Tr(S^T A S) / Tr(S^T D S) + 1
    float tr_SAS=0; for (int k=0;k<K;++k) tr_SAS+=(*new_adj)(k,k);
    // Compute D (degree matrix)
    std::vector<float> deg(N,0);
    for (int i=0;i<N;++i) for (int j=0;j<N;++j) deg[i]+=(*adj)(i,j);
    // S^T D S diagonal
    float tr_SDS=0;
    for (int k=0;k<K;++k) { float v=0; for (int i=0;i<N;++i) v+=(*s_soft)(i,k)*(*s_soft)(i,k)*deg[i]; tr_SDS+=v; }
    float mincut_loss=1.0f-tr_SAS/(tr_SDS+1e-12f);
    // Orthogonality loss: ||S^T S / ||S^T S||_F - I/K||_F^2
    auto STS=std::make_shared<Tensor>(K,K,0.0f);
    for (int ki=0;ki<K;++ki) for (int kj=0;kj<K;++kj) { float v=0; for (int i=0;i<N;++i) v+=(*s_soft)(i,ki)*(*s_soft)(i,kj); (*STS)(ki,kj)=v; }
    float sts_norm=0; for (int ki=0;ki<K;++ki) for (int kj=0;kj<K;++kj) sts_norm+=(*STS)(ki,kj)*(*STS)(ki,kj);
    sts_norm=std::sqrt(sts_norm)+1e-12f;
    float ortho_loss=0;
    for (int ki=0;ki<K;++ki) for (int kj=0;kj<K;++kj) {
        float target=(ki==kj)?1.0f/K:0.0f; float d=(*STS)(ki,kj)/sts_norm-target; ortho_loss+=d*d;
    }
    return {new_x,new_adj,mincut_loss,ortho_loss};
}

// =====================================================================
//  DenseGraphConv -- dense version of GraphConv (W1*x_i + W2*A*x)
// =====================================================================
class DenseGraphConv {
public:
    Linear lin_rel,lin_root;
    DenseGraphConv()=default;
    DenseGraphConv(int in,int out,bool bias=true):lin_rel(in,out,bias),lin_root(in,out,false){}

    TensorPtr forward(const TensorPtr& x, const TensorPtr& adj) {
        int N=x->rows(), C=x->cols();
        auto agg=std::make_shared<Tensor>(N,C,0.0f);
        for (int i=0;i<N;++i) for (int k=0;k<N;++k) { float a=(*adj)(i,k);
            if (a!=0) for (int j=0;j<C;++j) (*agg)(i,j)+=a*(*x)(k,j); }
        return autograd::add(lin_rel.forward(agg),lin_root.forward(x));
    }
    std::vector<TensorPtr> parameters() const { auto p=lin_rel.parameters(); auto p2=lin_root.parameters(); p.insert(p.end(),p2.begin(),p2.end()); return p; }
};

} // namespace dense
} // namespace pyg
