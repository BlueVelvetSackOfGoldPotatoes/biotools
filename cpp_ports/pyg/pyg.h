// pyg.h -- Header-only C++17 port of PyTorch Geometric (graph neural networks)
//
// Comprehensive reimplementation covering:
//   - Dense matrix (Tensor) with value-semantic storage and basic autograd
//   - Graph data structures (Data, HeteroData, Batch)
//   - Utility functions (degree, scatter, softmax, subgraph, negative_sampling, etc.)
//   - Cross-entropy, MSE, BCE losses
//   - Adam & SGD optimisers
//   - Reverse-mode autograd tape
//   - Dropout, Embedding, BatchNorm1d helper
//   - Linear, GRU cell
//
// Additional headers:
//   pyg_conv.h      -- 30+ convolution layers
//   pyg_pool.h      -- Pooling operators
//   pyg_norm.h      -- Normalization layers
//   pyg_models.h    -- Models (GAE, VGAE, JK, Node2Vec, SchNet, etc.)
//   pyg_transforms.h-- Data transforms

#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pyg {

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

inline std::mt19937& global_rng() { static std::mt19937 rng(42); return rng; }
inline void manual_seed(unsigned seed) { global_rng().seed(seed); }

struct GradNode {
    std::function<std::vector<TensorPtr>(const Tensor&)> backward_fn;
    std::vector<TensorPtr> inputs;
};

// =====================================================================
//  Tensor  -- row-major dense 2-D matrix  (rows x cols)
// =====================================================================
class Tensor {
public:
    std::vector<float> data_;
    int rows_ = 0, cols_ = 0;
    std::vector<float> grad_;
    bool requires_grad_ = false;
    std::shared_ptr<GradNode> grad_node_;

    Tensor() = default;
    Tensor(int r, int c, float val = 0.0f) : data_(r*c, val), rows_(r), cols_(c) {}
    Tensor(int r, int c, const std::vector<float>& d) : data_(d), rows_(r), cols_(c) { assert((int)d.size()==r*c); }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int numel() const { return rows_*cols_; }
    std::pair<int,int> shape() const { return {rows_,cols_}; }

    float& operator()(int r, int c) { return data_[r*cols_+c]; }
    float  operator()(int r, int c) const { return data_[r*cols_+c]; }
    float& operator[](int i) { return data_[i]; }
    float  operator[](int i) const { return data_[i]; }

    void zero_grad() { grad_.assign(data_.size(), 0.0f); }
    void set_requires_grad(bool v) { requires_grad_=v; if (v && grad_.empty()) grad_.resize(data_.size(),0.0f); }
    void fill_(float val) { std::fill(data_.begin(), data_.end(), val); }
    void clamp_(float lo, float hi) { for (auto& v:data_) v=std::max(lo,std::min(hi,v)); }

    TensorPtr slice_rows(int start, int end) const {
        assert(start>=0 && end<=rows_ && start<end);
        auto out=std::make_shared<Tensor>(end-start,cols_);
        std::copy(data_.begin()+start*cols_,data_.begin()+end*cols_,out->data_.begin());
        return out;
    }
    TensorPtr row(int r) const { return slice_rows(r,r+1); }

    TensorPtr t() const {
        auto out=std::make_shared<Tensor>(cols_,rows_);
        for (int i=0;i<rows_;++i) for (int j=0;j<cols_;++j) (*out)(j,i)=(*this)(i,j);
        return out;
    }

    float sum() const { float s=0; for (auto v:data_) s+=v; return s; }
    float mean() const { return sum()/(float)numel(); }
    float norm() const { float s=0; for (auto v:data_) s+=v*v; return std::sqrt(s); }

    std::vector<int> argmax_per_row() const {
        std::vector<int> r(rows_);
        for (int i=0;i<rows_;++i) { int b=0; float bv=(*this)(i,0);
            for (int j=1;j<cols_;++j) if ((*this)(i,j)>bv) { bv=(*this)(i,j); b=j; } r[i]=b; }
        return r;
    }

    static TensorPtr zeros(int r,int c) { return std::make_shared<Tensor>(r,c,0.0f); }
    static TensorPtr ones(int r,int c) { return std::make_shared<Tensor>(r,c,1.0f); }
    static TensorPtr full(int r,int c,float v) { return std::make_shared<Tensor>(r,c,v); }

    static TensorPtr randn(int r,int c,float scale=1.0f) {
        auto t=std::make_shared<Tensor>(r,c); std::normal_distribution<float> d(0,scale);
        for (auto& v:t->data_) v=d(global_rng()); return t;
    }
    static TensorPtr rand(int r,int c) {
        auto t=std::make_shared<Tensor>(r,c); std::uniform_real_distribution<float> d(0,1);
        for (auto& v:t->data_) v=d(global_rng()); return t;
    }
    static TensorPtr xavier_uniform(int r,int c) {
        float lim=std::sqrt(6.0f/(r+c)); auto t=std::make_shared<Tensor>(r,c);
        std::uniform_real_distribution<float> d(-lim,lim);
        for (auto& v:t->data_) v=d(global_rng()); return t;
    }
    static TensorPtr glorot(int r,int c) { return xavier_uniform(r,c); }
    static TensorPtr kaiming_uniform(int r,int c) {
        float lim=std::sqrt(6.0f/r); auto t=std::make_shared<Tensor>(r,c);
        std::uniform_real_distribution<float> d(-lim,lim);
        for (auto& v:t->data_) v=d(global_rng()); return t;
    }
    static TensorPtr uniform(int r,int c,float lo,float hi) {
        auto t=std::make_shared<Tensor>(r,c); std::uniform_real_distribution<float> d(lo,hi);
        for (auto& v:t->data_) v=d(global_rng()); return t;
    }
    static TensorPtr eye(int n) {
        auto t=std::make_shared<Tensor>(n,n,0.0f); for (int i=0;i<n;++i) (*t)(i,i)=1.0f; return t;
    }
    static TensorPtr one_hot(const std::vector<int>& labels,int nc) {
        int N=(int)labels.size(); auto t=zeros(N,nc);
        for (int i=0;i<N;++i) if (labels[i]>=0&&labels[i]<nc) (*t)(i,labels[i])=1.0f; return t;
    }
    static TensorPtr cat_rows(const std::vector<TensorPtr>& ts) {
        if (ts.empty()) return zeros(0,0); int C=ts[0]->cols(),tr=0;
        for (auto& t:ts) { assert(t->cols()==C); tr+=t->rows(); }
        auto out=std::make_shared<Tensor>(tr,C); int off=0;
        for (auto& t:ts) { std::copy(t->data_.begin(),t->data_.end(),out->data_.begin()+off*C); off+=t->rows(); }
        return out;
    }
    static TensorPtr cat_cols(const std::vector<TensorPtr>& ts) {
        if (ts.empty()) return zeros(0,0); int R=ts[0]->rows(),tc=0;
        for (auto& t:ts) { assert(t->rows()==R); tc+=t->cols(); }
        auto out=std::make_shared<Tensor>(R,tc);
        for (int i=0;i<R;++i) { int co=0; for (auto& t:ts) { for (int j=0;j<t->cols();++j) (*out)(i,co+j)=(*t)(i,j); co+=t->cols(); } }
        return out;
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        os<<"Tensor("<<t.rows_<<"x"<<t.cols_<<") [";
        int n=std::min((int)t.data_.size(),12);
        for (int i=0;i<n;++i) { if (i) os<<", "; os<<t.data_[i]; }
        if ((int)t.data_.size()>n) os<<" ..."; os<<"]"; return os;
    }
};

// =====================================================================
//  Autograd operations
// =====================================================================
namespace autograd {

inline TensorPtr matmul(const TensorPtr& A, const TensorPtr& B) {
    assert(A->cols()==B->rows()); int M=A->rows(),K=A->cols(),N=B->cols();
    auto C=std::make_shared<Tensor>(M,N,0.0f);
    for (int i=0;i<M;++i) for (int k=0;k<K;++k) { float a=(*A)(i,k); for (int j=0;j<N;++j) (*C)(i,j)+=a*(*B)(k,j); }
    if (A->requires_grad_||B->requires_grad_) {
        C->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={A,B};
        nd->backward_fn=[A,B,M,K,N](const Tensor& gC)->std::vector<TensorPtr> {
            auto dA=std::make_shared<Tensor>(M,K,0.0f), dB=std::make_shared<Tensor>(K,N,0.0f);
            for (int i=0;i<M;++i) for (int j=0;j<N;++j) { float g=gC(i,j);
                for (int k=0;k<K;++k) { (*dA)(i,k)+=g*(*B)(k,j); (*dB)(k,j)+=(*A)(i,k)*g; } }
            return {dA,dB};
        }; C->grad_node_=nd;
    } return C;
}

inline TensorPtr add(const TensorPtr& A, const TensorPtr& B) {
    assert(A->cols()==B->cols()); bool br=(B->rows()==1&&A->rows()!=1);
    int R=A->rows(),CC=A->cols(); auto out=std::make_shared<Tensor>(R,CC);
    for (int i=0;i<R;++i) for (int j=0;j<CC;++j) (*out)(i,j)=(*A)(i,j)+(*B)(br?0:i,j);
    if (A->requires_grad_||B->requires_grad_) {
        out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={A,B};
        nd->backward_fn=[A,B,R,CC,br](const Tensor& g)->std::vector<TensorPtr> {
            auto dA=std::make_shared<Tensor>(A->rows(),A->cols()); auto dB=std::make_shared<Tensor>(B->rows(),B->cols(),0.0f);
            for (int i=0;i<R;++i) for (int j=0;j<CC;++j) { (*dA)(i,j)=g(i,j); (*dB)(br?0:i,j)+=g(i,j); }
            return {dA,dB};
        }; out->grad_node_=nd;
    } return out;
}

inline TensorPtr sub(const TensorPtr& A, const TensorPtr& B) {
    assert(A->rows()==B->rows()&&A->cols()==B->cols()); int n=A->numel();
    auto out=std::make_shared<Tensor>(A->rows(),A->cols());
    for (int i=0;i<n;++i) (*out)[i]=(*A)[i]-(*B)[i];
    if (A->requires_grad_||B->requires_grad_) {
        out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={A,B};
        nd->backward_fn=[n](const Tensor& g)->std::vector<TensorPtr> {
            auto dA=std::make_shared<Tensor>(g.rows(),g.cols()), dB=std::make_shared<Tensor>(g.rows(),g.cols());
            for (int i=0;i<n;++i) { (*dA)[i]=g[i]; (*dB)[i]=-g[i]; } return {dA,dB};
        }; out->grad_node_=nd;
    } return out;
}

inline TensorPtr relu(const TensorPtr& A) {
    int n=A->numel(); auto out=std::make_shared<Tensor>(A->rows(),A->cols());
    for (int i=0;i<n;++i) (*out)[i]=std::max(0.0f,(*A)[i]);
    if (A->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={A};
        nd->backward_fn=[A,n](const Tensor& g)->std::vector<TensorPtr> {
            auto d=std::make_shared<Tensor>(A->rows(),A->cols());
            for (int i=0;i<n;++i) (*d)[i]=((*A)[i]>0)?g[i]:0; return {d};
        }; out->grad_node_=nd; } return out;
}

inline TensorPtr leaky_relu(const TensorPtr& A, float ns=0.2f) {
    int n=A->numel(); auto out=std::make_shared<Tensor>(A->rows(),A->cols());
    for (int i=0;i<n;++i) (*out)[i]=(*A)[i]>0?(*A)[i]:ns*(*A)[i];
    if (A->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={A};
        nd->backward_fn=[A,n,ns](const Tensor& g)->std::vector<TensorPtr> {
            auto d=std::make_shared<Tensor>(A->rows(),A->cols());
            for (int i=0;i<n;++i) (*d)[i]=((*A)[i]>0)?g[i]:ns*g[i]; return {d};
        }; out->grad_node_=nd; } return out;
}

inline TensorPtr elu(const TensorPtr& A, float alpha=1.0f) {
    int n=A->numel(); auto out=std::make_shared<Tensor>(A->rows(),A->cols());
    for (int i=0;i<n;++i) (*out)[i]=(*A)[i]>=0?(*A)[i]:alpha*(std::exp((*A)[i])-1.0f);
    if (A->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={A};
        nd->backward_fn=[A,out,n,alpha](const Tensor& g)->std::vector<TensorPtr> {
            auto d=std::make_shared<Tensor>(A->rows(),A->cols());
            for (int i=0;i<n;++i) (*d)[i]=(*A)[i]>=0?g[i]:g[i]*((*out)[i]+alpha); return {d};
        }; out->grad_node_=nd; } return out;
}

inline TensorPtr softplus(const TensorPtr& A, float bv=1.0f, float thr=20.0f) {
    int n=A->numel(); auto out=std::make_shared<Tensor>(A->rows(),A->cols());
    for (int i=0;i<n;++i) { float bx=bv*(*A)[i]; (*out)[i]=bx>thr?(*A)[i]:std::log(1.0f+std::exp(bx))/bv; }
    if (A->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={A};
        nd->backward_fn=[A,n,bv,thr](const Tensor& g)->std::vector<TensorPtr> {
            auto d=std::make_shared<Tensor>(A->rows(),A->cols());
            for (int i=0;i<n;++i) { float bx=bv*(*A)[i]; float s=bx>thr?1.0f:1.0f/(1.0f+std::exp(-bx)); (*d)[i]=g[i]*s; } return {d};
        }; out->grad_node_=nd; } return out;
}

inline TensorPtr tanh(const TensorPtr& A) {
    int n=A->numel(); auto out=std::make_shared<Tensor>(A->rows(),A->cols());
    for (int i=0;i<n;++i) (*out)[i]=std::tanh((*A)[i]);
    if (A->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); auto oc=out; nd->inputs={A};
        nd->backward_fn=[oc,n](const Tensor& g)->std::vector<TensorPtr> {
            auto d=std::make_shared<Tensor>(oc->rows(),oc->cols());
            for (int i=0;i<n;++i) { float t=(*oc)[i]; (*d)[i]=g[i]*(1-t*t); } return {d};
        }; out->grad_node_=nd; } return out;
}

inline TensorPtr sigmoid(const TensorPtr& A) {
    int n=A->numel(); auto out=std::make_shared<Tensor>(A->rows(),A->cols());
    for (int i=0;i<n;++i) (*out)[i]=1.0f/(1.0f+std::exp(-(*A)[i]));
    if (A->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); auto oc=out; nd->inputs={A};
        nd->backward_fn=[oc,n](const Tensor& g)->std::vector<TensorPtr> {
            auto d=std::make_shared<Tensor>(oc->rows(),oc->cols());
            for (int i=0;i<n;++i) { float s=(*oc)[i]; (*d)[i]=g[i]*s*(1-s); } return {d};
        }; out->grad_node_=nd; } return out;
}

inline TensorPtr log_softmax(const TensorPtr& A) {
    int R=A->rows(),C=A->cols(); auto out=std::make_shared<Tensor>(R,C);
    for (int i=0;i<R;++i) { float mx=-1e30f; for (int j=0;j<C;++j) mx=std::max(mx,(*A)(i,j));
        float s=0; for (int j=0;j<C;++j) s+=std::exp((*A)(i,j)-mx); float lse=mx+std::log(s);
        for (int j=0;j<C;++j) (*out)(i,j)=(*A)(i,j)-lse; }
    if (A->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); auto oc=out; nd->inputs={A};
        nd->backward_fn=[oc,R,C](const Tensor& g)->std::vector<TensorPtr> {
            auto d=std::make_shared<Tensor>(R,C);
            for (int i=0;i<R;++i) { float sg=0; for (int j=0;j<C;++j) sg+=g(i,j);
                for (int j=0;j<C;++j) (*d)(i,j)=g(i,j)-std::exp((*oc)(i,j))*sg; } return {d};
        }; out->grad_node_=nd; } return out;
}

inline TensorPtr softmax(const TensorPtr& A) {
    int R=A->rows(),C=A->cols(); auto out=std::make_shared<Tensor>(R,C);
    for (int i=0;i<R;++i) { float mx=-1e30f; for (int j=0;j<C;++j) mx=std::max(mx,(*A)(i,j));
        float s=0; for (int j=0;j<C;++j) { (*out)(i,j)=std::exp((*A)(i,j)-mx); s+=(*out)(i,j); }
        for (int j=0;j<C;++j) (*out)(i,j)/=s; }
    if (A->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); auto oc=out; nd->inputs={A};
        nd->backward_fn=[oc,R,C](const Tensor& g)->std::vector<TensorPtr> {
            auto d=std::make_shared<Tensor>(R,C);
            for (int i=0;i<R;++i) { float dot=0; for (int j=0;j<C;++j) dot+=g(i,j)*(*oc)(i,j);
                for (int j=0;j<C;++j) (*d)(i,j)=(*oc)(i,j)*(g(i,j)-dot); } return {d};
        }; out->grad_node_=nd; } return out;
}

inline TensorPtr mul(const TensorPtr& A, const TensorPtr& B) {
    assert(A->rows()==B->rows()&&A->cols()==B->cols()); int n=A->numel();
    auto out=std::make_shared<Tensor>(A->rows(),A->cols());
    for (int i=0;i<n;++i) (*out)[i]=(*A)[i]*(*B)[i];
    if (A->requires_grad_||B->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={A,B};
        nd->backward_fn=[A,B,n](const Tensor& g)->std::vector<TensorPtr> {
            auto dA=std::make_shared<Tensor>(A->rows(),A->cols()), dB=std::make_shared<Tensor>(B->rows(),B->cols());
            for (int i=0;i<n;++i) { (*dA)[i]=g[i]*(*B)[i]; (*dB)[i]=g[i]*(*A)[i]; } return {dA,dB};
        }; out->grad_node_=nd; } return out;
}

inline TensorPtr scale(const TensorPtr& A, float s) {
    int n=A->numel(); auto out=std::make_shared<Tensor>(A->rows(),A->cols());
    for (int i=0;i<n;++i) (*out)[i]=(*A)[i]*s;
    if (A->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={A};
        nd->backward_fn=[s,n](const Tensor& g)->std::vector<TensorPtr> {
            auto d=std::make_shared<Tensor>(g.rows(),g.cols()); for (int i=0;i<n;++i) (*d)[i]=g[i]*s; return {d};
        }; out->grad_node_=nd; } return out;
}

inline void backward(const TensorPtr& loss) {
    assert(loss->numel()==1); loss->grad_.assign(1,1.0f);
    std::vector<TensorPtr> order; std::unordered_map<Tensor*,bool> visited;
    std::function<void(const TensorPtr&)> topo=[&](const TensorPtr& t) {
        if (visited[t.get()]) return; visited[t.get()]=true;
        if (t->grad_node_) for (auto& inp:t->grad_node_->inputs) topo(inp); order.push_back(t);
    }; topo(loss); std::reverse(order.begin(),order.end());
    for (auto& t:order) { if (!t->grad_node_) continue;
        auto grads=t->grad_node_->backward_fn(Tensor(t->rows(),t->cols(),t->grad_));
        assert(grads.size()==t->grad_node_->inputs.size());
        for (size_t i=0;i<grads.size();++i) { auto& inp=t->grad_node_->inputs[i];
            if (inp->grad_.empty()) inp->grad_.resize(inp->numel(),0.0f);
            for (int k=0;k<inp->numel();++k) inp->grad_[k]+=(*grads[i])[k]; }
    }
}

} // namespace autograd

// =====================================================================
//  Loss functions
// =====================================================================
namespace loss {

inline TensorPtr nll_loss(const TensorPtr& lp, const std::vector<int>& targets) {
    int N=lp->rows(); float s=0; for (int i=0;i<N;++i) s-=(*lp)(i,targets[i]); s/=N;
    auto out=std::make_shared<Tensor>(1,1,s);
    if (lp->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={lp};
        nd->backward_fn=[lp,targets,N](const Tensor& g)->std::vector<TensorPtr> {
            auto dX=Tensor::zeros(lp->rows(),lp->cols()); for (int i=0;i<N;++i) (*dX)(i,targets[i])=-g[0]/N; return {dX};
        }; out->grad_node_=nd; } return out;
}

inline TensorPtr cross_entropy(const TensorPtr& logits, const std::vector<int>& targets) {
    return nll_loss(autograd::log_softmax(logits), targets);
}

inline TensorPtr mse_loss(const TensorPtr& pred, const TensorPtr& target) {
    int n=pred->numel(); float s=0; for (int i=0;i<n;++i) { float d=(*pred)[i]-(*target)[i]; s+=d*d; } s/=n;
    auto out=std::make_shared<Tensor>(1,1,s);
    if (pred->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={pred};
        nd->backward_fn=[pred,target,n](const Tensor& g)->std::vector<TensorPtr> {
            auto dP=std::make_shared<Tensor>(pred->rows(),pred->cols());
            for (int i=0;i<n;++i) (*dP)[i]=g[0]*2.0f*((*pred)[i]-(*target)[i])/n; return {dP};
        }; out->grad_node_=nd; } return out;
}

inline TensorPtr bce_loss(const TensorPtr& pred, const TensorPtr& target) {
    int n=pred->numel(); const float eps=1e-7f; float s=0;
    for (int i=0;i<n;++i) { float p=std::max(eps,std::min(1-eps,(*pred)[i]));
        s-=(*target)[i]*std::log(p)+(1-(*target)[i])*std::log(1-p); } s/=n;
    auto out=std::make_shared<Tensor>(1,1,s);
    if (pred->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={pred};
        nd->backward_fn=[pred,target,n,eps](const Tensor& g)->std::vector<TensorPtr> {
            auto dP=std::make_shared<Tensor>(pred->rows(),pred->cols());
            for (int i=0;i<n;++i) { float p=std::max(eps,std::min(1-eps,(*pred)[i]));
                (*dP)[i]=g[0]*(-(*target)[i]/p+(1-(*target)[i])/(1-p))/n; } return {dP};
        }; out->grad_node_=nd; } return out;
}

inline TensorPtr bce_with_logits(const TensorPtr& logits, const TensorPtr& target) {
    return bce_loss(autograd::sigmoid(logits), target);
}

} // namespace loss

// =====================================================================
//  Optimisers
// =====================================================================
class Adam {
public:
    float lr,beta1,beta2,eps; int t=0;
    struct State { std::vector<float> m,v; };
    std::vector<TensorPtr> params_; std::vector<State> states_;
    Adam(std::vector<TensorPtr> p, float lr=0.01f, float b1=0.9f, float b2=0.999f, float e=1e-8f)
        : lr(lr),beta1(b1),beta2(b2),eps(e),params_(std::move(p)) {
        for (auto& pp:params_) { State s; s.m.assign(pp->numel(),0); s.v.assign(pp->numel(),0); states_.push_back(std::move(s)); }
    }
    void step() { ++t; float bc1=1-std::pow(beta1,(float)t),bc2=1-std::pow(beta2,(float)t);
        for (size_t pi=0;pi<params_.size();++pi) { auto& p=params_[pi]; auto& s=states_[pi];
            if (p->grad_.empty()) continue; for (int i=0;i<p->numel();++i) { float g=p->grad_[i];
                s.m[i]=beta1*s.m[i]+(1-beta1)*g; s.v[i]=beta2*s.v[i]+(1-beta2)*g*g;
                p->data_[i]-=lr*(s.m[i]/bc1)/(std::sqrt(s.v[i]/bc2)+eps); } } }
    void zero_grad() { for (auto& p:params_) p->zero_grad(); }
};

class SGD {
public:
    float lr,momentum,wd; std::vector<TensorPtr> params_; std::vector<std::vector<float>> vel_;
    SGD(std::vector<TensorPtr> p, float lr=0.01f, float mom=0, float wd=0)
        : lr(lr),momentum(mom),wd(wd),params_(std::move(p)) { for (auto& pp:params_) vel_.push_back(std::vector<float>(pp->numel(),0)); }
    void step() { for (size_t pi=0;pi<params_.size();++pi) { auto& p=params_[pi]; auto& v=vel_[pi];
        if (p->grad_.empty()) continue; for (int i=0;i<p->numel();++i) { float g=p->grad_[i]+wd*p->data_[i];
            v[i]=momentum*v[i]+g; p->data_[i]-=lr*v[i]; } } }
    void zero_grad() { for (auto& p:params_) p->zero_grad(); }
};

// =====================================================================
//  Graph data structure
// =====================================================================
struct Data {
    std::vector<int> edge_src, edge_dst;
    int num_nodes = 0;
    TensorPtr x, edge_attr, pos, y_float;
    std::vector<int> y;
    int graph_label = -1;
    std::vector<int> batch;
    std::vector<bool> train_mask, val_mask, test_mask;

    int num_edges() const { return (int)edge_src.size(); }
    int num_features() const { return x?x->cols():0; }

    void add_self_loops() { for (int i=0;i<num_nodes;++i) { edge_src.push_back(i); edge_dst.push_back(i); } }

    void remove_self_loops() {
        std::vector<int> ns,nd;
        for (int e=0;e<num_edges();++e) if (edge_src[e]!=edge_dst[e]) { ns.push_back(edge_src[e]); nd.push_back(edge_dst[e]); }
        edge_src=std::move(ns); edge_dst=std::move(nd);
    }

    void to_undirected() {
        std::set<std::pair<int,int>> es; int E=num_edges();
        for (int e=0;e<E;++e) es.insert({edge_src[e],edge_dst[e]});
        for (int e=0;e<E;++e) { auto rev=std::make_pair(edge_dst[e],edge_src[e]);
            if (es.find(rev)==es.end()) { edge_src.push_back(edge_dst[e]); edge_dst.push_back(edge_src[e]); es.insert(rev); } }
    }

    void coalesce() {
        std::map<std::pair<int,int>,int> em; std::vector<int> ns,nd;
        for (int e=0;e<num_edges();++e) { auto k=std::make_pair(edge_src[e],edge_dst[e]);
            if (em.find(k)==em.end()) { em[k]=(int)ns.size(); ns.push_back(edge_src[e]); nd.push_back(edge_dst[e]); } }
        edge_src=std::move(ns); edge_dst=std::move(nd);
    }

    std::vector<std::vector<int>> adj_list_target_to_source() const {
        std::vector<std::vector<int>> adj(num_nodes);
        for (int e=0;e<num_edges();++e) adj[edge_dst[e]].push_back(edge_src[e]); return adj;
    }
    std::vector<std::vector<int>> adj_list_source_to_target() const {
        std::vector<std::vector<int>> adj(num_nodes);
        for (int e=0;e<num_edges();++e) adj[edge_src[e]].push_back(edge_dst[e]); return adj;
    }
    bool has_self_loops() const { for (int e=0;e<num_edges();++e) if (edge_src[e]==edge_dst[e]) return true; return false; }
    bool is_undirected() const {
        std::set<std::pair<int,int>> es;
        for (int e=0;e<num_edges();++e) es.insert({edge_src[e],edge_dst[e]});
        for (auto& [s,d]:es) if (es.find({d,s})==es.end()) return false; return true;
    }
};

struct HeteroData {
    std::unordered_map<std::string, TensorPtr> x_dict;
    std::unordered_map<std::string, int> num_nodes_dict;
    struct EdgeData { std::vector<int> edge_src, edge_dst; TensorPtr edge_attr; };
    std::unordered_map<std::string, EdgeData> edge_dict;
    static std::string edge_key(const std::string& s,const std::string& r,const std::string& d) { return s+"__"+r+"__"+d; }
    void set_edge_index(const std::string& st,const std::string& r,const std::string& dt,
                        const std::vector<int>& es,const std::vector<int>& ed) { edge_dict[edge_key(st,r,dt)]={es,ed,nullptr}; }
};

inline Data batch_graphs(const std::vector<Data>& graphs) {
    Data big; int no=0;
    for (size_t gi=0;gi<graphs.size();++gi) { auto& g=graphs[gi];
        for (int e=0;e<g.num_edges();++e) { big.edge_src.push_back(g.edge_src[e]+no); big.edge_dst.push_back(g.edge_dst[e]+no); }
        for (int i=0;i<g.num_nodes;++i) big.batch.push_back((int)gi); no+=g.num_nodes; }
    big.num_nodes=no;
    if (graphs[0].x) { int C=graphs[0].x->cols(); big.x=std::make_shared<Tensor>(no,C); int row=0;
        for (auto& g:graphs) for (int i=0;i<g.num_nodes;++i) { for (int j=0;j<C;++j) (*big.x)(row,j)=(*g.x)(i,j); ++row; } }
    if (graphs[0].edge_attr) { int C=graphs[0].edge_attr->cols(); big.edge_attr=std::make_shared<Tensor>(big.num_edges(),C); int row=0;
        for (auto& g:graphs) if (g.edge_attr) for (int i=0;i<g.num_edges();++i) { for (int j=0;j<C;++j) (*big.edge_attr)(row,j)=(*g.edge_attr)(i,j); ++row; } }
    if (graphs[0].pos) { int C=graphs[0].pos->cols(); big.pos=std::make_shared<Tensor>(no,C); int row=0;
        for (auto& g:graphs) if (g.pos) for (int i=0;i<g.num_nodes;++i) { for (int j=0;j<C;++j) (*big.pos)(row,j)=(*g.pos)(i,j); ++row; } }
    return big;
}

// =====================================================================
//  Utility functions
// =====================================================================
namespace utils {

inline std::vector<float> degree(const std::vector<int>& index, int num_nodes) {
    std::vector<float> deg(num_nodes,0); for (int idx:index) deg[idx]+=1.0f; return deg;
}

inline TensorPtr scatter_add(const TensorPtr& src, const std::vector<int>& index, int ds) {
    int C=src->cols(); auto out=Tensor::zeros(ds,C);
    for (int i=0;i<src->rows();++i) { int idx=index[i]; for (int j=0;j<C;++j) (*out)(idx,j)+=(*src)(i,j); } return out;
}

inline TensorPtr scatter_mean(const TensorPtr& src, const std::vector<int>& index, int ds) {
    int C=src->cols(); auto out=Tensor::zeros(ds,C); std::vector<int> cnt(ds,0);
    for (int i=0;i<src->rows();++i) { int idx=index[i]; cnt[idx]++; for (int j=0;j<C;++j) (*out)(idx,j)+=(*src)(i,j); }
    for (int i=0;i<ds;++i) if (cnt[i]>0) for (int j=0;j<C;++j) (*out)(i,j)/=cnt[i]; return out;
}

inline TensorPtr scatter_max(const TensorPtr& src, const std::vector<int>& index, int ds) {
    int C=src->cols(); auto out=std::make_shared<Tensor>(ds,C,-1e30f);
    for (int i=0;i<src->rows();++i) { int idx=index[i]; for (int j=0;j<C;++j) (*out)(idx,j)=std::max((*out)(idx,j),(*src)(i,j)); } return out;
}

inline TensorPtr segment_softmax(const TensorPtr& src, const std::vector<int>& index, int ns) {
    int N=src->rows(),C=src->cols(); auto out=std::make_shared<Tensor>(N,C,0.0f);
    std::vector<std::vector<float>> sm(ns,std::vector<float>(C,-1e30f));
    for (int i=0;i<N;++i) for (int j=0;j<C;++j) sm[index[i]][j]=std::max(sm[index[i]][j],(*src)(i,j));
    std::vector<std::vector<float>> ss(ns,std::vector<float>(C,0));
    for (int i=0;i<N;++i) for (int j=0;j<C;++j) { (*out)(i,j)=std::exp((*src)(i,j)-sm[index[i]][j]); ss[index[i]][j]+=(*out)(i,j); }
    for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*out)(i,j)/=(ss[index[i]][j]+1e-12f);
    return out;
}

inline TensorPtr to_dense_adj(const std::vector<int>& es,const std::vector<int>& ed,int N) {
    auto adj=Tensor::zeros(N,N); for (int e=0;e<(int)es.size();++e) (*adj)(es[e],ed[e])=1.0f; return adj;
}

inline std::vector<float> gcn_norm(const std::vector<int>& es,const std::vector<int>& ed,int N,bool asl=true) {
    auto src=es; auto dst=ed;
    if (asl) for (int i=0;i<N;++i) { src.push_back(i); dst.push_back(i); }
    int E=(int)src.size(); std::vector<float> deg(N,0);
    for (int e=0;e<E;++e) deg[dst[e]]+=1.0f;
    std::vector<float> dis(N,0); for (int i=0;i<N;++i) if (deg[i]>0) dis[i]=1.0f/std::sqrt(deg[i]);
    std::vector<float> w(E); for (int e=0;e<E;++e) w[e]=dis[src[e]]*dis[dst[e]]; return w;
}

inline std::pair<std::vector<int>,std::vector<int>> negative_sampling(
    const std::vector<int>& es,const std::vector<int>& ed,int N,int nn=-1) {
    std::set<std::pair<int,int>> ex; for (int e=0;e<(int)es.size();++e) ex.insert({es[e],ed[e]});
    if (nn<0) nn=(int)es.size(); std::vector<int> ns,nd; std::uniform_int_distribution<int> dist(0,N-1); int att=0;
    while ((int)ns.size()<nn&&att<nn*10) { int s=dist(global_rng()),d=dist(global_rng());
        if (s!=d&&!ex.count({s,d})) { ns.push_back(s); nd.push_back(d); ex.insert({s,d}); } ++att; } return {ns,nd};
}

inline void random_node_split(Data& data,float tr=0.6f,float vr=0.2f) {
    int N=data.num_nodes; std::vector<int> perm(N); std::iota(perm.begin(),perm.end(),0);
    std::shuffle(perm.begin(),perm.end(),global_rng()); int nt=(int)(N*tr),nv=(int)(N*vr);
    data.train_mask.assign(N,false); data.val_mask.assign(N,false); data.test_mask.assign(N,false);
    for (int i=0;i<N;++i) { if (i<nt) data.train_mask[perm[i]]=true; else if (i<nt+nv) data.val_mask[perm[i]]=true; else data.test_mask[perm[i]]=true; }
}

inline Data erdos_renyi_graph(int N,float p) {
    Data g; g.num_nodes=N; std::uniform_real_distribution<float> dist(0,1);
    for (int i=0;i<N;++i) for (int j=i+1;j<N;++j) if (dist(global_rng())<p) {
        g.edge_src.push_back(i); g.edge_dst.push_back(j); g.edge_src.push_back(j); g.edge_dst.push_back(i); } return g;
}

inline float homophily(const Data& data) {
    if (data.y.empty()) return 0; int same=0;
    for (int e=0;e<data.num_edges();++e) if (data.y[data.edge_src[e]]==data.y[data.edge_dst[e]]) ++same;
    return (float)same/std::max(1,data.num_edges());
}

inline Data subgraph(const Data& data,const std::vector<int>& nodes) {
    std::unordered_set<int> ns(nodes.begin(),nodes.end());
    std::unordered_map<int,int> nm; int idx=0; for (int n:nodes) nm[n]=idx++;
    Data sub; sub.num_nodes=(int)nodes.size();
    for (int e=0;e<data.num_edges();++e) { int s=data.edge_src[e],d=data.edge_dst[e];
        if (ns.count(s)&&ns.count(d)) { sub.edge_src.push_back(nm[s]); sub.edge_dst.push_back(nm[d]); } }
    if (data.x) { int C=data.x->cols(); sub.x=std::make_shared<Tensor>((int)nodes.size(),C);
        for (int i=0;i<(int)nodes.size();++i) for (int j=0;j<C;++j) (*sub.x)(i,j)=(*data.x)(nodes[i],j); }
    if (!data.y.empty()) for (int n:nodes) sub.y.push_back(data.y[n]); return sub;
}

inline std::vector<int> k_hop_subgraph(const Data& data,const std::vector<int>& ni,int nh) {
    auto adj=data.adj_list_source_to_target(); auto rev=data.adj_list_target_to_source();
    std::unordered_set<int> visited(ni.begin(),ni.end()); std::vector<int> frontier=ni;
    for (int h=0;h<nh;++h) { std::vector<int> next;
        for (int n:frontier) { for (int nb:adj[n]) if (!visited.count(nb)) { visited.insert(nb); next.push_back(nb); }
            for (int nb:rev[n]) if (!visited.count(nb)) { visited.insert(nb); next.push_back(nb); } }
        frontier=next; } return std::vector<int>(visited.begin(),visited.end());
}

} // namespace utils

// =====================================================================
//  Common building blocks
// =====================================================================
class Linear {
public:
    TensorPtr weight, bias; bool use_bias;
    Linear() : use_bias(false) {}
    Linear(int inf,int of,bool b=true) : use_bias(b) {
        weight=Tensor::xavier_uniform(of,inf); weight->set_requires_grad(true);
        if (b) { bias=Tensor::zeros(1,of); bias->set_requires_grad(true); }
    }
    TensorPtr forward(const TensorPtr& x) const {
        int of=weight->rows(),inf=weight->cols();
        auto wt=std::make_shared<Tensor>(inf,of);
        for (int i=0;i<of;++i) for (int j=0;j<inf;++j) (*wt)(j,i)=(*weight)(i,j);
        if (weight->requires_grad_) { wt->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={weight};
            nd->backward_fn=[of,inf](const Tensor& g)->std::vector<TensorPtr> {
                auto dW=std::make_shared<Tensor>(of,inf); for (int i=0;i<of;++i) for (int j=0;j<inf;++j) (*dW)(i,j)=g(j,i); return {dW};
            }; wt->grad_node_=nd; }
        auto out=autograd::matmul(x,wt); if (use_bias) out=autograd::add(out,bias); return out;
    }
    std::vector<TensorPtr> parameters() const { std::vector<TensorPtr> p={weight}; if (use_bias) p.push_back(bias); return p; }
};

class Embedding {
public:
    TensorPtr weight; int ne_,ed_;
    Embedding():ne_(0),ed_(0){}
    Embedding(int ne,int ed):ne_(ne),ed_(ed) { weight=Tensor::randn(ne,ed,0.01f); weight->set_requires_grad(true); }
    TensorPtr forward(const std::vector<int>& idx) const {
        int N=(int)idx.size(); auto out=std::make_shared<Tensor>(N,ed_);
        for (int i=0;i<N;++i) for (int j=0;j<ed_;++j) (*out)(i,j)=(*weight)(idx[i],j);
        if (weight->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={weight};
            auto idxs=idx; int ed=ed_,ne=ne_;
            nd->backward_fn=[idxs,ed,ne](const Tensor& g)->std::vector<TensorPtr> {
                auto dW=Tensor::zeros(ne,ed); for (int i=0;i<(int)idxs.size();++i) for (int j=0;j<ed;++j) (*dW)(idxs[i],j)+=g(i,j); return {dW};
            }; out->grad_node_=nd; } return out;
    }
    std::vector<TensorPtr> parameters() const { return {weight}; }
};

enum class Aggr { ADD, MEAN, MAX };

inline TensorPtr dropout(const TensorPtr& x,float p,bool training) {
    if (!training||p<=0) return x; int n=x->numel(); auto out=std::make_shared<Tensor>(x->rows(),x->cols());
    std::uniform_real_distribution<float> dist(0,1); float sc=1.0f/(1-p); std::vector<bool> mask(n);
    for (int i=0;i<n;++i) { mask[i]=dist(global_rng())>=p; (*out)[i]=mask[i]?(*x)[i]*sc:0; }
    if (x->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={x};
        nd->backward_fn=[mask,sc,n](const Tensor& g)->std::vector<TensorPtr> {
            auto d=std::make_shared<Tensor>(g.rows(),g.cols()); for (int i=0;i<n;++i) (*d)[i]=mask[i]?g[i]*sc:0; return {d};
        }; out->grad_node_=nd; } return out;
}

inline float accuracy(const TensorPtr& logits,const std::vector<int>& labels) {
    int N=logits->rows(),c=0;
    for (int i=0;i<N;++i) { int b=0; float bv=(*logits)(i,0);
        for (int j=1;j<logits->cols();++j) if ((*logits)(i,j)>bv) { bv=(*logits)(i,j); b=j; }
        if (b==labels[i]) ++c; } return (float)c/N;
}

inline float masked_accuracy(const TensorPtr& logits,const std::vector<int>& labels,const std::vector<bool>& mask) {
    int total=0,correct=0;
    for (int i=0;i<logits->rows();++i) { if (!mask[i]) continue; ++total;
        int b=0; float bv=(*logits)(i,0);
        for (int j=1;j<logits->cols();++j) if ((*logits)(i,j)>bv) { bv=(*logits)(i,j); b=j; }
        if (b==labels[i]) ++correct; } return total>0?(float)correct/total:0;
}

class GRUCell {
public:
    Linear lin_ir,lin_iz,lin_in,lin_hr,lin_hz,lin_hn; int hs_;
    GRUCell():hs_(0){}
    GRUCell(int is,int hs,bool bias=true):lin_ir(is,hs,bias),lin_iz(is,hs,bias),lin_in(is,hs,bias),
        lin_hr(hs,hs,bias),lin_hz(hs,hs,bias),lin_hn(hs,hs,bias),hs_(hs){}
    TensorPtr forward(const TensorPtr& x,const TensorPtr& h) {
        auto r=autograd::sigmoid(autograd::add(lin_ir.forward(x),lin_hr.forward(h)));
        auto z=autograd::sigmoid(autograd::add(lin_iz.forward(x),lin_hz.forward(h)));
        auto n=autograd::tanh(autograd::add(lin_in.forward(x),lin_hn.forward(autograd::mul(r,h))));
        int N=x->rows(),H=hs_; auto hn=std::make_shared<Tensor>(N,H);
        for (int i=0;i<N;++i) for (int j=0;j<H;++j) (*hn)(i,j)=(1-(*z)(i,j))*(*n)(i,j)+(*z)(i,j)*(*h)(i,j); return hn;
    }
    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p; auto add=[&](const Linear& l){ auto lp=l.parameters(); p.insert(p.end(),lp.begin(),lp.end()); };
        add(lin_ir);add(lin_iz);add(lin_in);add(lin_hr);add(lin_hz);add(lin_hn); return p;
    }
};

class BatchNorm1d {
public:
    int nf_; float eps_,mom_; bool affine_,training_=true;
    TensorPtr gamma,beta; std::vector<float> rm,rv;
    BatchNorm1d():nf_(0),eps_(1e-5f),mom_(0.1f),affine_(true){}
    BatchNorm1d(int nf,float eps=1e-5f,float mom=0.1f,bool aff=true):nf_(nf),eps_(eps),mom_(mom),affine_(aff) {
        if (aff) { gamma=Tensor::ones(1,nf); gamma->set_requires_grad(true); beta=Tensor::zeros(1,nf); beta->set_requires_grad(true); }
        rm.assign(nf,0); rv.assign(nf,1);
    }
    void train() { training_=true; } void eval() { training_=false; }
    TensorPtr forward(const TensorPtr& x) {
        int N=x->rows(),C=x->cols(); auto out=std::make_shared<Tensor>(N,C);
        if (training_) { std::vector<float> bm(C,0),bv(C,0);
            for (int j=0;j<C;++j) { for (int i=0;i<N;++i) bm[j]+=(*x)(i,j); bm[j]/=N; }
            for (int j=0;j<C;++j) { for (int i=0;i<N;++i) { float d=(*x)(i,j)-bm[j]; bv[j]+=d*d; } bv[j]/=N; }
            for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*out)(i,j)=((*x)(i,j)-bm[j])/std::sqrt(bv[j]+eps_);
            for (int j=0;j<C;++j) { rm[j]=(1-mom_)*rm[j]+mom_*bm[j]; rv[j]=(1-mom_)*rv[j]+mom_*bv[j]; }
        } else { for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*out)(i,j)=((*x)(i,j)-rm[j])/std::sqrt(rv[j]+eps_); }
        if (affine_) for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*out)(i,j)=(*out)(i,j)*(*gamma)(0,j)+(*beta)(0,j);
        return out;
    }
    std::vector<TensorPtr> parameters() const { std::vector<TensorPtr> p; if (affine_) { p.push_back(gamma); p.push_back(beta); } return p; }
};

class DataLoader {
public:
    std::vector<Data> dataset_; int batch_size_; bool shuffle_;
    DataLoader(const std::vector<Data>& ds,int bs,bool sh=true):dataset_(ds),batch_size_(bs),shuffle_(sh){}
    struct Iterator {
        const DataLoader* loader; std::vector<int> indices; int pos;
        Data operator*() const { int end=std::min(pos+loader->batch_size_,(int)indices.size());
            std::vector<Data> batch; for (int i=pos;i<end;++i) batch.push_back(loader->dataset_[indices[i]]); return batch_graphs(batch); }
        Iterator& operator++() { pos+=loader->batch_size_; return *this; }
        bool operator!=(const Iterator&) const { return pos<(int)indices.size(); }
    };
    Iterator begin() { std::vector<int> idx(dataset_.size()); std::iota(idx.begin(),idx.end(),0);
        if (shuffle_) std::shuffle(idx.begin(),idx.end(),global_rng()); return {this,idx,0}; }
    Iterator end() { return {this,{},(int)dataset_.size()}; }
};


// =====================================================================
//  Global Pooling  (pool namespace -- basic ops needed by many models)
// =====================================================================
namespace pool {

inline TensorPtr global_add_pool(const TensorPtr& x, const std::vector<int>& batch, int num_graphs) {
    int C=x->cols(); auto out=Tensor::zeros(num_graphs,C);
    for (int i=0;i<x->rows();++i) { int g=batch[i]; for (int j=0;j<C;++j) (*out)(g,j)+=(*x)(i,j); }
    if (x->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={x};
        auto bt=batch; int R=x->rows();
        nd->backward_fn=[bt,R,C](const Tensor& g)->std::vector<TensorPtr> {
            auto dX=std::make_shared<Tensor>(R,C); for (int i=0;i<R;++i) for (int j=0;j<C;++j) (*dX)(i,j)=g(bt[i],j); return {dX};
        }; out->grad_node_=nd; }
    return out;
}

inline TensorPtr global_mean_pool(const TensorPtr& x, const std::vector<int>& batch, int num_graphs) {
    int C=x->cols(); auto out=Tensor::zeros(num_graphs,C); std::vector<int> cnt(num_graphs,0);
    for (int i=0;i<x->rows();++i) { int g=batch[i]; cnt[g]++; for (int j=0;j<C;++j) (*out)(g,j)+=(*x)(i,j); }
    for (int g=0;g<num_graphs;++g) if (cnt[g]>0) for (int j=0;j<C;++j) (*out)(g,j)/=cnt[g];
    if (x->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={x};
        auto bt=batch; int R=x->rows();
        nd->backward_fn=[bt,cnt,R,C](const Tensor& g)->std::vector<TensorPtr> {
            auto dX=std::make_shared<Tensor>(R,C); for (int i=0;i<R;++i) { int gi=bt[i]; float d=cnt[gi]>0?(float)cnt[gi]:1;
                for (int j=0;j<C;++j) (*dX)(i,j)=g(gi,j)/d; } return {dX};
        }; out->grad_node_=nd; }
    return out;
}

inline TensorPtr global_max_pool(const TensorPtr& x, const std::vector<int>& batch, int num_graphs) {
    int C=x->cols(); auto out=std::make_shared<Tensor>(num_graphs,C,-1e30f);
    for (int i=0;i<x->rows();++i) { int g=batch[i]; for (int j=0;j<C;++j) (*out)(g,j)=std::max((*out)(g,j),(*x)(i,j)); }
    return out;
}

inline TensorPtr global_attention_pool(const TensorPtr& x, const TensorPtr& gate_logits, const std::vector<int>& batch, int num_graphs) {
    int N=x->rows(),C=x->cols();
    // Softmax gate per graph
    auto gate=std::make_shared<Tensor>(N,1,0.0f);
    std::vector<std::vector<int>> gn(num_graphs); for (int i=0;i<N;++i) gn[batch[i]].push_back(i);
    for (int g=0;g<num_graphs;++g) { if (gn[g].empty()) continue; float mx=-1e30f;
        for (int i:gn[g]) mx=std::max(mx,(*gate_logits)(i,0)); float s=0;
        for (int i:gn[g]) { (*gate)(i,0)=std::exp((*gate_logits)(i,0)-mx); s+=(*gate)(i,0); }
        for (int i:gn[g]) (*gate)(i,0)/=(s+1e-12f); }
    auto out=Tensor::zeros(num_graphs,C);
    for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*out)(batch[i],j)+=(*gate)(i,0)*(*x)(i,j);
    return out;
}

inline TensorPtr global_sort_pool(const TensorPtr& x, const std::vector<int>& batch, int num_graphs, int k) {
    int C=x->cols(); auto out=Tensor::zeros(num_graphs,k*C);
    std::vector<std::vector<int>> gn(num_graphs); for (int i=0;i<x->rows();++i) gn[batch[i]].push_back(i);
    for (int g=0;g<num_graphs;++g) {
        // Sort nodes by last feature channel
        auto& nodes=gn[g]; std::sort(nodes.begin(),nodes.end(),[&](int a,int b){ return (*x)(a,C-1)>(*x)(b,C-1); });
        int nk=std::min(k,(int)nodes.size());
        for (int i=0;i<nk;++i) for (int j=0;j<C;++j) (*out)(g,i*C+j)=(*x)(nodes[i],j);
    }
    return out;
}

inline TensorPtr set2set(const TensorPtr& x, const std::vector<int>& batch, int num_graphs, int processing_steps=3) {
    int C=x->cols(); int N=x->rows();
    // Simplified Set2Set: iterative attention-based readout
    auto q=Tensor::zeros(num_graphs,C); // query
    auto out=Tensor::zeros(num_graphs,2*C);
    std::vector<std::vector<int>> gn(num_graphs); for (int i=0;i<N;++i) gn[batch[i]].push_back(i);
    for (int step=0;step<processing_steps;++step) {
        // Attention
        for (int g=0;g<num_graphs;++g) { if (gn[g].empty()) continue;
            float mx=-1e30f; std::vector<float> scores(gn[g].size());
            for (size_t ni=0;ni<gn[g].size();++ni) { float dot=0; int node=gn[g][ni];
                for (int j=0;j<C;++j) dot+=(*q)(g,j)*(*x)(node,j); scores[ni]=dot; mx=std::max(mx,dot); }
            float s=0; for (auto& sc:scores) { sc=std::exp(sc-mx); s+=sc; }
            for (auto& sc:scores) sc/=(s+1e-12f);
            // Weighted sum
            std::vector<float> r(C,0);
            for (size_t ni=0;ni<gn[g].size();++ni) { int node=gn[g][ni]; for (int j=0;j<C;++j) r[j]+=scores[ni]*(*x)(node,j); }
            for (int j=0;j<C;++j) { (*q)(g,j)=r[j]; (*out)(g,j)=(*q)(g,j); (*out)(g,C+j)=r[j]; }
        }
    }
    return out;
}

} // namespace pool

// =====================================================================
//  Additional utils
// =====================================================================
namespace utils {

inline std::pair<std::vector<int>,std::vector<int>> add_self_loops(
    const std::vector<int>& es, const std::vector<int>& ed, int num_nodes) {
    auto ns=es; auto nd=ed;
    for (int i=0;i<num_nodes;++i) { ns.push_back(i); nd.push_back(i); }
    return {ns,nd};
}

inline std::pair<std::vector<int>,std::vector<int>> remove_self_loops(
    const std::vector<int>& es, const std::vector<int>& ed) {
    std::vector<int> ns,nd;
    for (int e=0;e<(int)es.size();++e) if (es[e]!=ed[e]) { ns.push_back(es[e]); nd.push_back(ed[e]); }
    return {ns,nd};
}

inline bool contains_self_loops(const std::vector<int>& es, const std::vector<int>& ed) {
    for (int e=0;e<(int)es.size();++e) if (es[e]==ed[e]) return true; return false;
}

inline std::pair<std::vector<int>,std::vector<int>> coalesce(
    const std::vector<int>& es, const std::vector<int>& ed) {
    std::set<std::pair<int,int>> seen; std::vector<int> ns,nd;
    for (int e=0;e<(int)es.size();++e) { auto p=std::make_pair(es[e],ed[e]);
        if (!seen.count(p)) { seen.insert(p); ns.push_back(es[e]); nd.push_back(ed[e]); } }
    return {ns,nd};
}

inline TensorPtr softmax_nodes(const TensorPtr& src, const std::vector<int>& batch, int num_graphs) {
    int N=src->rows(),C=src->cols(); auto out=std::make_shared<Tensor>(N,C);
    std::vector<std::vector<int>> gn(num_graphs); for (int i=0;i<N;++i) gn[batch[i]].push_back(i);
    for (int g=0;g<num_graphs;++g) { if (gn[g].empty()) continue;
        for (int j=0;j<C;++j) { float mx=-1e30f; for (int i:gn[g]) mx=std::max(mx,(*src)(i,j));
            float s=0; for (int i:gn[g]) { (*out)(i,j)=std::exp((*src)(i,j)-mx); s+=(*out)(i,j); }
            for (int i:gn[g]) (*out)(i,j)/=(s+1e-12f); } }
    return out;
}

inline Data line_graph(const Data& data) {
    Data lg; int E=data.num_edges(); lg.num_nodes=E;
    // Edge features become node features if available
    if (data.edge_attr) lg.x=data.edge_attr;
    // Two line-graph edges are connected if they share a node
    std::unordered_map<int, std::vector<int>> node_edges;
    for (int e=0;e<E;++e) { node_edges[data.edge_src[e]].push_back(e); node_edges[data.edge_dst[e]].push_back(e); }
    for (auto& [node, edges] : node_edges) {
        for (int i=0;i<(int)edges.size();++i) for (int j=0;j<(int)edges.size();++j) {
            if (i!=j) { lg.edge_src.push_back(edges[i]); lg.edge_dst.push_back(edges[j]); } } }
    lg.coalesce();
    return lg;
}

inline std::pair<std::vector<int>,std::vector<int>> to_undirected(
    const std::vector<int>& es, const std::vector<int>& ed) {
    std::set<std::pair<int,int>> seen; std::vector<int> ns,nd;
    for (int e=0;e<(int)es.size();++e) { seen.insert({es[e],ed[e]}); seen.insert({ed[e],es[e]}); }
    for (auto& [s,d] : seen) { ns.push_back(s); nd.push_back(d); }
    return {ns,nd};
}

} // namespace utils

} // namespace pyg
