// pyg_conv.h -- 30+ Graph Convolution layers for C++ PyG port
// Header-only C++17. Depends on pyg.h
#pragma once
#include "pyg.h"

namespace pyg {

// =====================================================================
//  GCNConv
// =====================================================================
class GCNConv {
public:
    Linear lin; bool add_sl_,normalize_; int in_c_,out_c_;
    GCNConv()=default;
    GCNConv(int in,int out,bool bias=true,bool asl=true,bool norm=true)
        :lin(in,out,bias),add_sl_(asl),normalize_(norm),in_c_(in),out_c_(out){}

    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes; auto es=data.edge_src; auto ed=data.edge_dst;
        if (add_sl_) for (int i=0;i<N;++i) { es.push_back(i); ed.push_back(i); }
        int E=(int)es.size();
        std::vector<float> ew(E,1.0f);
        if (normalize_) { std::vector<float> deg(N,0); for (int e=0;e<E;++e) deg[ed[e]]+=1.0f;
            std::vector<float> dis(N,0); for (int i=0;i<N;++i) if (deg[i]>0) dis[i]=1.0f/std::sqrt(deg[i]);
            for (int e=0;e<E;++e) ew[e]=dis[es[e]]*dis[ed[e]]; }
        auto h=lin.forward(x); int od=h->cols(); auto out=Tensor::zeros(N,od);
        for (int e=0;e<E;++e) { float w=ew[e]; for (int j=0;j<od;++j) (*out)(ed[e],j)+=w*(*h)(es[e],j); }
        if (h->requires_grad_) { out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={h};
            nd->backward_fn=[es,ed,ew,N,E,od](const Tensor& g)->std::vector<TensorPtr> {
                auto dH=Tensor::zeros(N,od); for (int e=0;e<E;++e) { float w=ew[e]; for (int j=0;j<od;++j) (*dH)(es[e],j)+=w*g(ed[e],j); } return {dH};
            }; out->grad_node_=nd; }
        return out;
    }
    std::vector<TensorPtr> parameters() const { return lin.parameters(); }
};

// =====================================================================
//  GATConv
// =====================================================================
class GATConv {
public:
    int in_c_,out_c_,heads_; bool concat_; float neg_slope_; bool add_sl_;
    TensorPtr weight,att_src,att_dst,bias;
    GATConv()=default;
    GATConv(int in,int out,int heads=1,bool concat=true,float ns=0.2f,bool asl=true,bool ub=true)
        :in_c_(in),out_c_(out),heads_(heads),concat_(concat),neg_slope_(ns),add_sl_(asl) {
        int tot=heads*out; weight=Tensor::xavier_uniform(tot,in); weight->set_requires_grad(true);
        float lim=std::sqrt(6.0f/(1+out)); std::uniform_real_distribution<float> d(-lim,lim);
        att_src=std::make_shared<Tensor>(1,tot); att_dst=std::make_shared<Tensor>(1,tot);
        for (int i=0;i<tot;++i) { (*att_src)(0,i)=d(global_rng()); (*att_dst)(0,i)=d(global_rng()); }
        att_src->set_requires_grad(true); att_dst->set_requires_grad(true);
        if (ub) { int ot=concat?tot:out; bias=Tensor::zeros(1,ot); bias->set_requires_grad(true); }
    }

    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,H=heads_,C=out_c_,HC=H*C;
        auto es=data.edge_src; auto ed=data.edge_dst;
        if (add_sl_) for (int i=0;i<N;++i) { es.push_back(i); ed.push_back(i); }
        int E=(int)es.size();
        // Wx
        auto wt=std::make_shared<Tensor>(in_c_,HC);
        for (int i=0;i<HC;++i) for (int j=0;j<in_c_;++j) (*wt)(j,i)=(*weight)(i,j);
        if (weight->requires_grad_) { wt->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={weight};
            nd->backward_fn=[HC,this](const Tensor& g)->std::vector<TensorPtr> {
                auto dW=std::make_shared<Tensor>(HC,in_c_); for (int i=0;i<HC;++i) for (int j=0;j<in_c_;++j) (*dW)(i,j)=g(j,i); return {dW};
            }; wt->grad_node_=nd; }
        auto Wx=autograd::matmul(x,wt);
        // Attention scores
        auto ss=std::make_shared<Tensor>(N,H,0.0f), ds=std::make_shared<Tensor>(N,H,0.0f);
        for (int i=0;i<N;++i) for (int h=0;h<H;++h) { float s1=0,s2=0;
            for (int c=0;c<C;++c) { int idx=h*C+c; s1+=(*Wx)(i,idx)*(*att_src)(0,idx); s2+=(*Wx)(i,idx)*(*att_dst)(0,idx); }
            (*ss)(i,h)=s1; (*ds)(i,h)=s2; }
        // Per-edge attention
        auto ar=std::make_shared<Tensor>(E,H);
        for (int e=0;e<E;++e) for (int h=0;h<H;++h) {
            float v=(*ss)(es[e],h)+(*ds)(ed[e],h); v=(v>0)?v:neg_slope_*v; (*ar)(e,h)=v; }
        // Softmax per target
        auto ac=std::make_shared<Tensor>(E,H,0.0f);
        for (int h=0;h<H;++h) { std::vector<std::vector<int>> te(N);
            for (int e=0;e<E;++e) te[ed[e]].push_back(e);
            for (int i=0;i<N;++i) { if (te[i].empty()) continue; float mx=-1e30f;
                for (int e:te[i]) mx=std::max(mx,(*ar)(e,h)); float s=0;
                for (int e:te[i]) { (*ac)(e,h)=std::exp((*ar)(e,h)-mx); s+=(*ac)(e,h); }
                for (int e:te[i]) (*ac)(e,h)/=(s+1e-12f); } }
        // Aggregate
        int od=concat_?HC:C; auto out=Tensor::zeros(N,od);
        for (int e=0;e<E;++e) for (int h=0;h<H;++h) { float a=(*ac)(e,h);
            for (int c=0;c<C;++c) { float v=a*(*Wx)(es[e],h*C+c);
                if (concat_) (*out)(ed[e],h*C+c)+=v; else (*out)(ed[e],c)+=v/H; } }
        if (bias) for (int i=0;i<N;++i) for (int j=0;j<od;++j) (*out)(i,j)+=(*bias)(0,j);
        if (x->requires_grad_||weight->requires_grad_) {
            out->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={Wx,ac};
            nd->backward_fn=[es,ed,N,E,H,C,HC,od,concat_cp=concat_,ac_cp=ac](const Tensor& go)->std::vector<TensorPtr> {
                auto dWx=Tensor::zeros(N,HC); auto dAtt=Tensor::zeros(E,H);
                for (int e=0;e<E;++e) for (int h=0;h<H;++h) { float a=(*ac_cp)(e,h);
                    for (int c=0;c<C;++c) { float g=concat_cp?go(ed[e],h*C+c):go(ed[e],c)/H; (*dWx)(es[e],h*C+c)+=a*g; } }
                return {dWx,dAtt};
            }; out->grad_node_=nd; }
        return out;
    }
    std::vector<TensorPtr> parameters() const { std::vector<TensorPtr> p={weight,att_src,att_dst}; if (bias) p.push_back(bias); return p; }
};

// =====================================================================
//  SAGEConv (GraphSAGE)
// =====================================================================
class SAGEConv {
public:
    Linear lin_l,lin_r; bool normalize_,root_weight_; int in_c_,out_c_;
    SAGEConv()=default;
    SAGEConv(int in,int out,bool norm=false,bool rw=true,bool bias=true)
        :lin_l(in,out,bias),lin_r(in,out,false),normalize_(norm),root_weight_(rw),in_c_(in),out_c_(out){}

    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),ind=x->cols();
        auto agg=Tensor::zeros(N,ind); std::vector<int> cnt(N,0);
        for (int e=0;e<E;++e) { cnt[data.edge_dst[e]]++; for (int j=0;j<ind;++j) (*agg)(data.edge_dst[e],j)+=(*x)(data.edge_src[e],j); }
        for (int i=0;i<N;++i) if (cnt[i]>0) for (int j=0;j<ind;++j) (*agg)(i,j)/=cnt[i];
        if (x->requires_grad_) { agg->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={x};
            auto esrc=data.edge_src; auto edst=data.edge_dst;
            nd->backward_fn=[esrc,edst,cnt,N,E,ind](const Tensor& g)->std::vector<TensorPtr> {
                auto dX=Tensor::zeros(N,ind);
                for (int e=0;e<E;++e) { float d=cnt[edst[e]]>0?(float)cnt[edst[e]]:1; for (int j=0;j<ind;++j) (*dX)(esrc[e],j)+=g(edst[e],j)/d; } return {dX};
            }; agg->grad_node_=nd; }
        auto out=lin_l.forward(agg);
        if (root_weight_) out=autograd::add(out,lin_r.forward(x));
        if (normalize_) { int od=out->cols(); for (int i=0;i<N;++i) { float n=0; for (int j=0;j<od;++j) n+=(*out)(i,j)*(*out)(i,j);
            n=std::sqrt(n)+1e-12f; for (int j=0;j<od;++j) (*out)(i,j)/=n; } }
        return out;
    }
    std::vector<TensorPtr> parameters() const { auto p=lin_l.parameters(); if (root_weight_) { auto pr=lin_r.parameters(); p.insert(p.end(),pr.begin(),pr.end()); } return p; }
};

// =====================================================================
//  GINConv
// =====================================================================
class GINConv {
public:
    Linear mlp1,mlp2; float eps_; bool train_eps_; TensorPtr eps_param; int in_c_,out_c_;
    GINConv()=default;
    GINConv(int in,int out,float eps=0,bool te=false):mlp1(in,out,true),mlp2(out,out,true),eps_(eps),train_eps_(te),in_c_(in),out_c_(out) {
        eps_param=std::make_shared<Tensor>(1,1,eps); if (te) eps_param->set_requires_grad(true);
    }
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),ind=x->cols();
        auto agg=Tensor::zeros(N,ind);
        for (int e=0;e<E;++e) for (int j=0;j<ind;++j) (*agg)(data.edge_dst[e],j)+=(*x)(data.edge_src[e],j);
        if (x->requires_grad_) { agg->set_requires_grad(true); auto nd=std::make_shared<GradNode>(); nd->inputs={x};
            auto esrc=data.edge_src; auto edst=data.edge_dst;
            nd->backward_fn=[esrc,edst,N,E,ind](const Tensor& g)->std::vector<TensorPtr> {
                auto dX=Tensor::zeros(N,ind); for (int e=0;e<E;++e) for (int j=0;j<ind;++j) (*dX)(esrc[e],j)+=g(edst[e],j); return {dX};
            }; agg->grad_node_=nd; }
        float ce=(*eps_param)(0,0);
        auto combined=autograd::add(autograd::scale(x,1.0f+ce),agg);
        return mlp2.forward(autograd::relu(mlp1.forward(combined)));
    }
    std::vector<TensorPtr> parameters() const {
        auto p=mlp1.parameters(); auto p2=mlp2.parameters(); p.insert(p.end(),p2.begin(),p2.end());
        if (train_eps_) p.push_back(eps_param); return p;
    }
};

// =====================================================================
//  GraphConv  -- W1*x_i + W2*Agg(x_j)
// =====================================================================
class GraphConv {
public:
    Linear lin_rel,lin_root; Aggr aggr_; int in_c_,out_c_;
    GraphConv()=default;
    GraphConv(int in,int out,Aggr aggr=Aggr::ADD,bool bias=true)
        :lin_rel(in,out,bias),lin_root(in,out,false),aggr_(aggr),in_c_(in),out_c_(out){}

    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),ind=x->cols();
        auto agg=Tensor::zeros(N,ind);
        if (aggr_==Aggr::ADD) {
            for (int e=0;e<E;++e) for (int j=0;j<ind;++j) (*agg)(data.edge_dst[e],j)+=(*x)(data.edge_src[e],j);
        } else if (aggr_==Aggr::MEAN) {
            std::vector<int> cnt(N,0);
            for (int e=0;e<E;++e) { cnt[data.edge_dst[e]]++; for (int j=0;j<ind;++j) (*agg)(data.edge_dst[e],j)+=(*x)(data.edge_src[e],j); }
            for (int i=0;i<N;++i) if (cnt[i]>0) for (int j=0;j<ind;++j) (*agg)(i,j)/=cnt[i];
        } else {
            agg=std::make_shared<Tensor>(N,ind,-1e30f);
            for (int e=0;e<E;++e) for (int j=0;j<ind;++j) (*agg)(data.edge_dst[e],j)=std::max((*agg)(data.edge_dst[e],j),(*x)(data.edge_src[e],j));
        }
        return autograd::add(lin_rel.forward(agg),lin_root.forward(x));
    }
    std::vector<TensorPtr> parameters() const { auto p=lin_rel.parameters(); auto p2=lin_root.parameters(); p.insert(p.end(),p2.begin(),p2.end()); return p; }
};

// =====================================================================
//  ChebConv -- Chebyshev spectral conv
// =====================================================================
class ChebConv {
public:
    std::vector<Linear> lins; TensorPtr bias_param; int in_c_,out_c_,K_; bool normalize_;
    ChebConv()=default;
    ChebConv(int in,int out,int K,bool bias=true):in_c_(in),out_c_(out),K_(K),normalize_(true) {
        for (int k=0;k<K;++k) lins.emplace_back(in,out,false);
        if (bias) { bias_param=Tensor::zeros(1,out); bias_param->set_requires_grad(true); }
    }
    TensorPtr forward(const TensorPtr& x,const Data& data,float lambda_max=2.0f) {
        int N=data.num_nodes,E=data.num_edges(),C=x->cols();
        // Build normalized Laplacian L_hat = 2L/lambda_max - I (sparse multiply via edges)
        auto deg=utils::degree(data.edge_dst,N);
        std::vector<float> dis(N,0); for (int i=0;i<N;++i) if (deg[i]>0) dis[i]=1.0f/std::sqrt(deg[i]);
        // Z0 = X, Z1 = L_hat*X
        auto Z0=x;
        // Compute L_hat * Z (sparse)
        auto spmv=[&](const TensorPtr& inp)->TensorPtr {
            auto out=std::make_shared<Tensor>(N,inp->cols(),0.0f);
            for (int e=0;e<E;++e) { float w=dis[data.edge_src[e]]*dis[data.edge_dst[e]];
                for (int j=0;j<inp->cols();++j) (*out)(data.edge_dst[e],j)-=w*(*inp)(data.edge_src[e],j); }
            // Diagonal: I
            float scale=2.0f/lambda_max;
            for (int i=0;i<N;++i) for (int j=0;j<inp->cols();++j) (*out)(i,j)=scale*((*inp)(i,j)+(*out)(i,j))-(*inp)(i,j);
            return out;
        };
        auto result=lins[0].forward(Z0);
        if (K_>1) { auto Z1=spmv(Z0); result=autograd::add(result,lins[1].forward(Z1));
            for (int k=2;k<K_;++k) { auto Z2=autograd::sub(autograd::scale(spmv(Z1),2.0f),Z0); Z0=Z1; Z1=Z2;
                result=autograd::add(result,lins[k].forward(Z1)); } }
        if (bias_param) result=autograd::add(result,bias_param);
        return result;
    }
    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p; for (auto& l:lins) { auto lp=l.parameters(); p.insert(p.end(),lp.begin(),lp.end()); }
        if (bias_param) p.push_back(bias_param); return p;
    }
};

// =====================================================================
//  TAGConv -- Topology Adaptive Graph Conv
// =====================================================================
class TAGConv {
public:
    std::vector<Linear> lins; TensorPtr bias_param; int in_c_,out_c_,K_;
    TAGConv()=default;
    TAGConv(int in,int out,int K=3,bool bias=true):in_c_(in),out_c_(out),K_(K) {
        for (int k=0;k<=K;++k) lins.emplace_back(in,out,false);
        if (bias) { bias_param=Tensor::zeros(1,out); bias_param->set_requires_grad(true); }
    }
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=x->cols();
        auto deg=utils::degree(data.edge_dst,N);
        std::vector<float> dis(N,0); for (int i=0;i<N;++i) if (deg[i]>0) dis[i]=1.0f/std::sqrt(deg[i]);
        auto prop=[&](const TensorPtr& inp)->TensorPtr {
            auto out=Tensor::zeros(N,inp->cols());
            for (int e=0;e<E;++e) { float w=dis[data.edge_src[e]]*dis[data.edge_dst[e]];
                for (int j=0;j<inp->cols();++j) (*out)(data.edge_dst[e],j)+=w*(*inp)(data.edge_src[e],j); }
            return out;
        };
        auto result=lins[0].forward(x); auto h=x;
        for (int k=1;k<=K_;++k) { h=prop(h); result=autograd::add(result,lins[k].forward(h)); }
        if (bias_param) result=autograd::add(result,bias_param);
        return result;
    }
    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p; for (auto& l:lins) { auto lp=l.parameters(); p.insert(p.end(),lp.begin(),lp.end()); }
        if (bias_param) p.push_back(bias_param); return p;
    }
};

// =====================================================================
//  SGConv -- Simplified Graph Conv
// =====================================================================
class SGConv {
public:
    Linear lin; int in_c_,out_c_,K_; bool add_sl_;
    SGConv()=default;
    SGConv(int in,int out,int K=1,bool bias=true,bool asl=true):lin(in,out,bias),in_c_(in),out_c_(out),K_(K),add_sl_(asl){}

    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes; auto es=data.edge_src; auto ed=data.edge_dst;
        if (add_sl_) for (int i=0;i<N;++i) { es.push_back(i); ed.push_back(i); }
        int E=(int)es.size();
        std::vector<float> deg(N,0); for (int e=0;e<E;++e) deg[ed[e]]+=1;
        std::vector<float> dis(N,0); for (int i=0;i<N;++i) if (deg[i]>0) dis[i]=1.0f/std::sqrt(deg[i]);
        auto h=x;
        for (int k=0;k<K_;++k) {
            auto nxt=Tensor::zeros(N,h->cols());
            for (int e=0;e<E;++e) { float w=dis[es[e]]*dis[ed[e]]; for (int j=0;j<h->cols();++j) (*nxt)(ed[e],j)+=w*(*h)(es[e],j); }
            h=nxt;
        }
        return lin.forward(h);
    }
    std::vector<TensorPtr> parameters() const { return lin.parameters(); }
};

// =====================================================================
//  APPNP
// =====================================================================
class APPNP {
public:
    int K_; float alpha_,dropout_; bool add_sl_;
    APPNP()=default;
    APPNP(int K,float alpha,float dp=0,bool asl=true):K_(K),alpha_(alpha),dropout_(dp),add_sl_(asl){}

    TensorPtr forward(const TensorPtr& x,const Data& data,bool training=false) {
        int N=data.num_nodes; auto es=data.edge_src; auto ed=data.edge_dst;
        if (add_sl_) for (int i=0;i<N;++i) { es.push_back(i); ed.push_back(i); }
        int E=(int)es.size(),C=x->cols();
        std::vector<float> deg(N,0); for (int e=0;e<E;++e) deg[ed[e]]+=1;
        std::vector<float> dis(N,0); for (int i=0;i<N;++i) if (deg[i]>0) dis[i]=1.0f/std::sqrt(deg[i]);
        auto h=x;
        for (int k=0;k<K_;++k) {
            if (training && dropout_>0) h=dropout(h,dropout_,true);
            auto nxt=Tensor::zeros(N,C);
            for (int e=0;e<E;++e) { float w=dis[es[e]]*dis[ed[e]]; for (int j=0;j<C;++j) (*nxt)(ed[e],j)+=w*(*h)(es[e],j); }
            // h = (1-alpha)*nxt + alpha*x
            for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*nxt)(i,j)=(1-alpha_)*(*nxt)(i,j)+alpha_*(*x)(i,j);
            h=nxt;
        }
        return h;
    }
    std::vector<TensorPtr> parameters() const { return {}; }
};

// =====================================================================
//  GCN2Conv (GCNII)
// =====================================================================
class GCN2Conv {
public:
    int channels_; float alpha_,beta_; bool shared_weights_,add_sl_;
    TensorPtr weight1,weight2;
    GCN2Conv()=default;
    GCN2Conv(int ch,float alpha,float theta=0,int layer=0,bool shared=true,bool asl=true)
        :channels_(ch),alpha_(alpha),shared_weights_(shared),add_sl_(asl) {
        beta_=(theta>0&&layer>0)?std::log(theta/layer+1):1.0f;
        weight1=Tensor::glorot(ch,ch); weight1->set_requires_grad(true);
        if (!shared) { weight2=Tensor::glorot(ch,ch); weight2->set_requires_grad(true); }
    }
    TensorPtr forward(const TensorPtr& x,const TensorPtr& x0,const Data& data) {
        int N=data.num_nodes,C=channels_; auto es=data.edge_src; auto ed=data.edge_dst;
        if (add_sl_) for (int i=0;i<N;++i) { es.push_back(i); ed.push_back(i); }
        int E=(int)es.size();
        std::vector<float> deg(N,0); for (int e=0;e<E;++e) deg[ed[e]]+=1;
        std::vector<float> dis(N,0); for (int i=0;i<N;++i) if (deg[i]>0) dis[i]=1.0f/std::sqrt(deg[i]);
        // Propagate
        auto prop=Tensor::zeros(N,C);
        for (int e=0;e<E;++e) { float w=dis[es[e]]*dis[ed[e]]; for (int j=0;j<C;++j) (*prop)(ed[e],j)+=w*(*x)(es[e],j); }
        // h = (1-alpha)*prop + alpha*x0
        auto h=std::make_shared<Tensor>(N,C);
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*h)(i,j)=(1-alpha_)*(*prop)(i,j)+alpha_*(*x0)(i,j);
        // out = (1-beta)*h + beta*(h*W)
        auto wt1=std::make_shared<Tensor>(C,C);
        for (int i=0;i<C;++i) for (int j=0;j<C;++j) (*wt1)(j,i)=(*weight1)(i,j);
        auto hW=autograd::matmul(h,wt1);
        auto out=std::make_shared<Tensor>(N,C);
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*out)(i,j)=(1-beta_)*(*h)(i,j)+beta_*(*hW)(i,j);
        return out;
    }
    std::vector<TensorPtr> parameters() const { std::vector<TensorPtr> p={weight1}; if (weight2) p.push_back(weight2); return p; }
};

// =====================================================================
//  EdgeConv -- Dynamic Graph CNN
// =====================================================================
class EdgeConv {
public:
    Linear lin; int in_c_,out_c_; Aggr aggr_;
    EdgeConv()=default;
    EdgeConv(int in,int out,Aggr aggr=Aggr::MAX):lin(2*in,out,true),in_c_(in),out_c_(out),aggr_(aggr){}
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=x->cols();
        // Build edge features: [x_i || x_j - x_i] for each edge j->i
        auto edge_feat=std::make_shared<Tensor>(E,2*C);
        for (int e=0;e<E;++e) { int s=data.edge_src[e],d=data.edge_dst[e];
            for (int j=0;j<C;++j) { (*edge_feat)(e,j)=(*x)(d,j); (*edge_feat)(e,C+j)=(*x)(s,j)-(*x)(d,j); } }
        auto msg=autograd::relu(lin.forward(edge_feat));
        // Aggregate to target
        int od=msg->cols(); auto out=Tensor::zeros(N,od);
        if (aggr_==Aggr::MAX) {
            out=std::make_shared<Tensor>(N,od,-1e30f);
            for (int e=0;e<E;++e) for (int j=0;j<od;++j) (*out)(data.edge_dst[e],j)=std::max((*out)(data.edge_dst[e],j),(*msg)(e,j));
        } else if (aggr_==Aggr::ADD) {
            for (int e=0;e<E;++e) for (int j=0;j<od;++j) (*out)(data.edge_dst[e],j)+=(*msg)(e,j);
        } else {
            std::vector<int> cnt(N,0);
            for (int e=0;e<E;++e) { cnt[data.edge_dst[e]]++; for (int j=0;j<od;++j) (*out)(data.edge_dst[e],j)+=(*msg)(e,j); }
            for (int i=0;i<N;++i) if (cnt[i]>0) for (int j=0;j<od;++j) (*out)(i,j)/=cnt[i];
        }
        return out;
    }
    std::vector<TensorPtr> parameters() const { return lin.parameters(); }
};

// =====================================================================
//  TransformerConv -- Graph Transformer
// =====================================================================
class TransformerConv {
public:
    Linear lin_key,lin_query,lin_value,lin_skip; int in_c_,out_c_,heads_; bool concat_,root_weight_;
    TransformerConv()=default;
    TransformerConv(int in,int out,int heads=1,bool concat=true,bool bias=true,bool rw=true)
        :lin_key(in,heads*out,bias),lin_query(in,heads*out,bias),lin_value(in,heads*out,bias),
         lin_skip(in,concat?heads*out:out,bias),in_c_(in),out_c_(out),heads_(heads),concat_(concat),root_weight_(rw){}

    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),H=heads_,C=out_c_,HC=H*C;
        auto K=lin_key.forward(x), Q=lin_query.forward(x), V=lin_value.forward(x);
        float scale=1.0f/std::sqrt((float)C);
        // Compute attention per edge
        auto att=std::make_shared<Tensor>(E,H);
        for (int e=0;e<E;++e) { int s=data.edge_src[e],d=data.edge_dst[e];
            for (int h=0;h<H;++h) { float dot=0; for (int c=0;c<C;++c) dot+=(*Q)(d,h*C+c)*(*K)(s,h*C+c);
                (*att)(e,h)=dot*scale; } }
        // Softmax per target per head
        auto alpha=std::make_shared<Tensor>(E,H,0.0f);
        for (int h=0;h<H;++h) { std::vector<std::vector<int>> te(N);
            for (int e=0;e<E;++e) te[data.edge_dst[e]].push_back(e);
            for (int i=0;i<N;++i) { if (te[i].empty()) continue; float mx=-1e30f;
                for (int e:te[i]) mx=std::max(mx,(*att)(e,h)); float s=0;
                for (int e:te[i]) { (*alpha)(e,h)=std::exp((*att)(e,h)-mx); s+=(*alpha)(e,h); }
                for (int e:te[i]) (*alpha)(e,h)/=(s+1e-12f); } }
        // Aggregate values
        int od=concat_?HC:C; auto msg=Tensor::zeros(N,od);
        for (int e=0;e<E;++e) { int s=data.edge_src[e],d=data.edge_dst[e];
            for (int h=0;h<H;++h) { float a=(*alpha)(e,h);
                for (int c=0;c<C;++c) { float v=a*(*V)(s,h*C+c);
                    if (concat_) (*msg)(d,h*C+c)+=v; else (*msg)(d,c)+=v/H; } } }
        // Skip connection
        if (root_weight_) msg=autograd::add(msg,lin_skip.forward(x));
        return msg;
    }
    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p; auto add=[&](const Linear& l){ auto lp=l.parameters(); p.insert(p.end(),lp.begin(),lp.end()); };
        add(lin_key); add(lin_query); add(lin_value); add(lin_skip); return p;
    }
};

// =====================================================================
//  GATv2Conv
// =====================================================================
class GATv2Conv {
public:
    Linear lin_l,lin_r; TensorPtr att,bias; int in_c_,out_c_,heads_; bool concat_; float neg_slope_; bool add_sl_;
    GATv2Conv()=default;
    GATv2Conv(int in,int out,int heads=1,bool concat=true,float ns=0.2f,bool asl=true,bool ub=true)
        :lin_l(in,heads*out,false),lin_r(in,heads*out,false),in_c_(in),out_c_(out),heads_(heads),concat_(concat),neg_slope_(ns),add_sl_(asl) {
        att=Tensor::xavier_uniform(1,heads*out); att->set_requires_grad(true);
        if (ub) { int od=concat?heads*out:out; bias=Tensor::zeros(1,od); bias->set_requires_grad(true); }
    }
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,H=heads_,C=out_c_,HC=H*C; auto es=data.edge_src; auto ed=data.edge_dst;
        if (add_sl_) for (int i=0;i<N;++i) { es.push_back(i); ed.push_back(i); }
        int E=(int)es.size();
        auto Ls=lin_l.forward(x), Rs=lin_r.forward(x);
        // Per-edge: LeakyReLU(Ls[i]+Rs[j]) then dot with att
        auto e_att=std::make_shared<Tensor>(E,H);
        for (int e=0;e<E;++e) { int s=es[e],d=ed[e];
            for (int h=0;h<H;++h) { float score=0;
                for (int c=0;c<C;++c) { float v=(*Ls)(d,h*C+c)+(*Rs)(s,h*C+c); v=(v>0)?v:neg_slope_*v; score+=v*(*att)(0,h*C+c); }
                (*e_att)(e,h)=score; } }
        // Softmax
        auto alpha=std::make_shared<Tensor>(E,H,0.0f);
        for (int h=0;h<H;++h) { std::vector<std::vector<int>> te(N);
            for (int e=0;e<E;++e) te[ed[e]].push_back(e);
            for (int i=0;i<N;++i) { if (te[i].empty()) continue; float mx=-1e30f;
                for (int e:te[i]) mx=std::max(mx,(*e_att)(e,h)); float s=0;
                for (int e:te[i]) { (*alpha)(e,h)=std::exp((*e_att)(e,h)-mx); s+=(*alpha)(e,h); }
                for (int e:te[i]) (*alpha)(e,h)/=(s+1e-12f); } }
        int od=concat_?HC:C; auto out=Tensor::zeros(N,od);
        for (int e=0;e<E;++e) for (int h=0;h<H;++h) { float a=(*alpha)(e,h);
            for (int c=0;c<C;++c) { float v=a*(*Rs)(es[e],h*C+c);
                if (concat_) (*out)(ed[e],h*C+c)+=v; else (*out)(ed[e],c)+=v/H; } }
        if (bias) for (int i=0;i<N;++i) for (int j=0;j<od;++j) (*out)(i,j)+=(*bias)(0,j);
        return out;
    }
    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p; auto add=[&](const Linear& l){ auto lp=l.parameters(); p.insert(p.end(),lp.begin(),lp.end()); };
        add(lin_l); add(lin_r); p.push_back(att); if (bias) p.push_back(bias); return p;
    }
};

// =====================================================================
//  GatedGraphConv (GGNN)
// =====================================================================
class GatedGraphConv {
public:
    int out_c_,num_layers_; std::vector<TensorPtr> weights; GRUCell rnn;
    GatedGraphConv()=default;
    GatedGraphConv(int out,int nl,bool bias=true):out_c_(out),num_layers_(nl),rnn(out,out,bias) {
        for (int i=0;i<nl;++i) { auto w=Tensor::xavier_uniform(out,out); w->set_requires_grad(true); weights.push_back(w); }
    }
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=out_c_,inC=x->cols();
        auto h=std::make_shared<Tensor>(N,C,0.0f);
        for (int i=0;i<N;++i) for (int j=0;j<std::min(inC,C);++j) (*h)(i,j)=(*x)(i,j);
        for (int l=0;l<num_layers_;++l) {
            auto wt=std::make_shared<Tensor>(C,C);
            for (int i=0;i<C;++i) for (int j=0;j<C;++j) (*wt)(j,i)=(*weights[l])(i,j);
            auto m=autograd::matmul(h,wt);
            // Propagate
            auto agg=Tensor::zeros(N,C);
            for (int e=0;e<E;++e) for (int j=0;j<C;++j) (*agg)(data.edge_dst[e],j)+=(*m)(data.edge_src[e],j);
            h=rnn.forward(agg,h);
        }
        return h;
    }
    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p=weights; auto rp=rnn.parameters(); p.insert(p.end(),rp.begin(),rp.end()); return p;
    }
};

// =====================================================================
//  CGConv -- Crystal Graph Conv
// =====================================================================
class CGConv {
public:
    Linear lin_f,lin_s; int channels_,dim_;
    CGConv()=default;
    CGConv(int channels,int dim=0,bool bias=true):lin_f(2*channels+dim,channels,bias),lin_s(2*channels+dim,channels,bias),channels_(channels),dim_(dim){}
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=channels_,D=dim_;
        auto out=Tensor::zeros(N,C);
        // For each edge, compute z=[x_i,x_j,e_ij], then sigma(f(z))*softplus(s(z))
        for (int e=0;e<E;++e) { int s=data.edge_src[e],d=data.edge_dst[e];
            int zd=2*C+D; auto z=std::make_shared<Tensor>(1,zd);
            for (int j=0;j<C;++j) { (*z)(0,j)=(*x)(d,j); (*z)(0,C+j)=(*x)(s,j); }
            if (D>0 && data.edge_attr) for (int j=0;j<D;++j) (*z)(0,2*C+j)=(*data.edge_attr)(e,j);
            auto fz=autograd::sigmoid(lin_f.forward(z));
            auto sz=autograd::softplus(lin_s.forward(z));
            auto msg=autograd::mul(fz,sz);
            for (int j=0;j<C;++j) (*out)(d,j)+=(*msg)(0,j);
        }
        // Residual
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*out)(i,j)+=(*x)(i,j);
        return out;
    }
    std::vector<TensorPtr> parameters() const { auto p=lin_f.parameters(); auto p2=lin_s.parameters(); p.insert(p.end(),p2.begin(),p2.end()); return p; }
};

// =====================================================================
//  AGNNConv -- Attention-based GNN
// =====================================================================
class AGNNConv {
public:
    TensorPtr beta; bool add_sl_,req_grad_;
    AGNNConv()=default;
    AGNNConv(bool rg=true,bool asl=true):add_sl_(asl),req_grad_(rg) {
        beta=std::make_shared<Tensor>(1,1,1.0f); if (rg) beta->set_requires_grad(true);
    }
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,C=x->cols(); auto es=data.edge_src; auto ed=data.edge_dst;
        if (add_sl_) { es=data.edge_src; ed=data.edge_dst; // copy first
            // Remove existing self-loops, then add
            std::vector<int> ns,nd; for (int e=0;e<(int)es.size();++e) if (es[e]!=ed[e]) { ns.push_back(es[e]); nd.push_back(ed[e]); }
            for (int i=0;i<N;++i) { ns.push_back(i); nd.push_back(i); }
            es=ns; ed=nd;
        }
        int E=(int)es.size(); float b=(*beta)(0,0);
        // Normalize x
        auto xn=std::make_shared<Tensor>(N,C);
        for (int i=0;i<N;++i) { float n=0; for (int j=0;j<C;++j) n+=(*x)(i,j)*(*x)(i,j); n=std::sqrt(n)+1e-12f;
            for (int j=0;j<C;++j) (*xn)(i,j)=(*x)(i,j)/n; }
        // Attention: cos similarity
        auto att=std::make_shared<Tensor>(E,1);
        for (int e=0;e<E;++e) { float dot=0; for (int j=0;j<C;++j) dot+=(*xn)(es[e],j)*(*xn)(ed[e],j); (*att)(e,0)=b*dot; }
        // Softmax per target
        auto alpha=std::make_shared<Tensor>(E,1,0.0f);
        std::vector<std::vector<int>> te(N); for (int e=0;e<E;++e) te[ed[e]].push_back(e);
        for (int i=0;i<N;++i) { if (te[i].empty()) continue; float mx=-1e30f;
            for (int e:te[i]) mx=std::max(mx,(*att)(e,0)); float s=0;
            for (int e:te[i]) { (*alpha)(e,0)=std::exp((*att)(e,0)-mx); s+=(*alpha)(e,0); }
            for (int e:te[i]) (*alpha)(e,0)/=(s+1e-12f); }
        auto out=Tensor::zeros(N,C);
        for (int e=0;e<E;++e) for (int j=0;j<C;++j) (*out)(ed[e],j)+=(*alpha)(e,0)*(*x)(es[e],j);
        return out;
    }
    std::vector<TensorPtr> parameters() const { return req_grad_?std::vector<TensorPtr>{beta}:std::vector<TensorPtr>{}; }
};

// =====================================================================
//  SimpleConv -- Non-trainable propagation
// =====================================================================
class SimpleConv {
public:
    Aggr aggr_; std::string combine_root_;
    SimpleConv()=default;
    SimpleConv(Aggr aggr=Aggr::ADD,const std::string& cr=""):aggr_(aggr),combine_root_(cr){}
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=x->cols();
        auto out=Tensor::zeros(N,C);
        if (aggr_==Aggr::ADD) { for (int e=0;e<E;++e) for (int j=0;j<C;++j) (*out)(data.edge_dst[e],j)+=(*x)(data.edge_src[e],j); }
        else if (aggr_==Aggr::MEAN) { std::vector<int> cnt(N,0);
            for (int e=0;e<E;++e) { cnt[data.edge_dst[e]]++; for (int j=0;j<C;++j) (*out)(data.edge_dst[e],j)+=(*x)(data.edge_src[e],j); }
            for (int i=0;i<N;++i) if (cnt[i]>0) for (int j=0;j<C;++j) (*out)(i,j)/=cnt[i]; }
        else { out=std::make_shared<Tensor>(N,C,-1e30f);
            for (int e=0;e<E;++e) for (int j=0;j<C;++j) (*out)(data.edge_dst[e],j)=std::max((*out)(data.edge_dst[e],j),(*x)(data.edge_src[e],j)); }
        if (combine_root_=="sum") for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*out)(i,j)+=(*x)(i,j);
        return out;
    }
    std::vector<TensorPtr> parameters() const { return {}; }
};

// =====================================================================
//  GMMConv -- Gaussian Mixture Model Conv
// =====================================================================
class GMMConv {
public:
    TensorPtr g_param,mu,sigma,bias_param; Linear root_lin; int in_c_,out_c_,dim_,K_; bool root_weight_;
    GMMConv()=default;
    GMMConv(int in,int out,int dim,int K,bool rw=true,bool bias=true):in_c_(in),out_c_(out),dim_(dim),K_(K),root_weight_(rw) {
        g_param=Tensor::glorot(in,out*K); g_param->set_requires_grad(true);
        mu=Tensor::randn(K,dim,0.1f); mu->set_requires_grad(true);
        sigma=Tensor::ones(K,dim); sigma->set_requires_grad(true);
        if (rw) root_lin=Linear(in,out,false);
        if (bias) { bias_param=Tensor::zeros(1,out); bias_param->set_requires_grad(true); }
    }
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=out_c_; auto out=Tensor::zeros(N,C);
        std::vector<int> cnt(N,0);
        for (int e=0;e<E;++e) { int s=data.edge_src[e],d=data.edge_dst[e]; cnt[d]++;
            // Gaussian weights from edge features
            for (int k=0;k<K_;++k) { float w=0;
                if (data.edge_attr) { for (int dd=0;dd<dim_;++dd) { float diff=(*data.edge_attr)(e,dd)-(*mu)(k,dd);
                    w-=0.5f*diff*diff/((*sigma)(k,dd)*(*sigma)(k,dd)+1e-6f); } }
                w=std::exp(w);
                for (int j=0;j<C;++j) { float v=0; for (int ii=0;ii<in_c_;++ii) v+=(*x)(s,ii)*(*g_param)(ii,k*C+j); (*out)(d,j)+=w*v/K_; }
            }
        }
        for (int i=0;i<N;++i) if (cnt[i]>0) for (int j=0;j<C;++j) (*out)(i,j)/=cnt[i];
        if (root_weight_) out=autograd::add(out,root_lin.forward(x));
        if (bias_param) out=autograd::add(out,bias_param);
        return out;
    }
    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p={g_param,mu,sigma};
        if (root_weight_) { auto rp=root_lin.parameters(); p.insert(p.end(),rp.begin(),rp.end()); }
        if (bias_param) p.push_back(bias_param); return p;
    }
};

// =====================================================================
//  NNConv -- Neural Message Passing (edge-conditioned)
// =====================================================================
class NNConv {
public:
    Linear nn_lin1,nn_lin2,root_lin; TensorPtr bias_param; int in_c_,out_c_,edge_dim_; bool root_weight_;
    NNConv()=default;
    NNConv(int in,int out,int edge_dim,bool rw=true,bool bias=true)
        :nn_lin1(edge_dim,in*out,true),nn_lin2(in*out,in*out,true),root_lin(in,out,false),
         in_c_(in),out_c_(out),edge_dim_(edge_dim),root_weight_(rw) {
        if (bias) { bias_param=Tensor::zeros(1,out); bias_param->set_requires_grad(true); }
    }
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),IN=in_c_,OUT=out_c_;
        auto out=Tensor::zeros(N,OUT);
        for (int e=0;e<E;++e) { int s=data.edge_src[e],d=data.edge_dst[e];
            // nn(edge_attr) -> (IN*OUT) weight
            auto ea=data.edge_attr?data.edge_attr->row(e):Tensor::zeros(1,edge_dim_);
            auto w=nn_lin1.forward(ea); // (1, IN*OUT)
            for (int j=0;j<OUT;++j) { float v=0; for (int i=0;i<IN;++i) v+=(*x)(s,i)*(*w)(0,i*OUT+j); (*out)(d,j)+=v; }
        }
        if (root_weight_) out=autograd::add(out,root_lin.forward(x));
        if (bias_param) out=autograd::add(out,bias_param);
        return out;
    }
    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p; auto add=[&](const Linear& l){ auto lp=l.parameters(); p.insert(p.end(),lp.begin(),lp.end()); };
        add(nn_lin1); add(nn_lin2); if (root_weight_) add(root_lin);
        if (bias_param) p.push_back(bias_param); return p;
    }
};

// =====================================================================
//  SignedConv
// =====================================================================
class SignedConv {
public:
    Linear lin_pos_l,lin_pos_r,lin_neg_l,lin_neg_r; int in_c_,out_c_; bool first_aggr_;
    SignedConv()=default;
    SignedConv(int in,int out,bool first=true,bool bias=true):in_c_(in),out_c_(out),first_aggr_(first) {
        if (first) { lin_pos_l=Linear(in,out,false); lin_pos_r=Linear(in,out,bias);
                     lin_neg_l=Linear(in,out,false); lin_neg_r=Linear(in,out,bias); }
        else { lin_pos_l=Linear(2*in,out,false); lin_pos_r=Linear(in,out,bias);
               lin_neg_l=Linear(2*in,out,false); lin_neg_r=Linear(in,out,bias); }
    }
    TensorPtr forward(const TensorPtr& x,const Data& pos_data,const Data& neg_data) {
        int N=pos_data.num_nodes,C=x->cols();
        // Mean aggregation for positive edges
        auto pos_agg=Tensor::zeros(N,C); std::vector<int> pc(N,0);
        for (int e=0;e<pos_data.num_edges();++e) { pc[pos_data.edge_dst[e]]++;
            for (int j=0;j<C;++j) (*pos_agg)(pos_data.edge_dst[e],j)+=(*x)(pos_data.edge_src[e],j); }
        for (int i=0;i<N;++i) if (pc[i]>0) for (int j=0;j<C;++j) (*pos_agg)(i,j)/=pc[i];
        auto neg_agg=Tensor::zeros(N,C); std::vector<int> nc(N,0);
        for (int e=0;e<neg_data.num_edges();++e) { nc[neg_data.edge_dst[e]]++;
            for (int j=0;j<C;++j) (*neg_agg)(neg_data.edge_dst[e],j)+=(*x)(neg_data.edge_src[e],j); }
        for (int i=0;i<N;++i) if (nc[i]>0) for (int j=0;j<C;++j) (*neg_agg)(i,j)/=nc[i];
        auto pos_out=autograd::add(lin_pos_l.forward(pos_agg),lin_pos_r.forward(x));
        auto neg_out=autograd::add(lin_neg_l.forward(neg_agg),lin_neg_r.forward(x));
        return Tensor::cat_cols({pos_out,neg_out});
    }
    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p; auto add=[&](const Linear& l){ auto lp=l.parameters(); p.insert(p.end(),lp.begin(),lp.end()); };
        add(lin_pos_l); add(lin_pos_r); add(lin_neg_l); add(lin_neg_r); return p;
    }
};

// =====================================================================
//  ResGatedGraphConv
// =====================================================================
class ResGatedGraphConv {
public:
    Linear lin_u,lin_v,lin_e; int in_c_,out_c_;
    ResGatedGraphConv()=default;
    ResGatedGraphConv(int in,int out,bool bias=true):lin_u(in,out,bias),lin_v(in,out,bias),lin_e(in,out,bias),in_c_(in),out_c_(out){}
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=out_c_;
        auto Ux=lin_u.forward(x), Vx=lin_v.forward(x), Ex=lin_e.forward(x);
        auto out=Tensor::zeros(N,C);
        for (int e=0;e<E;++e) { int s=data.edge_src[e],d=data.edge_dst[e];
            for (int j=0;j<C;++j) { float gate=1.0f/(1.0f+std::exp(-((*Ex)(s,j)+(*Ex)(d,j))));
                (*out)(d,j)+=gate*(*Vx)(s,j); } }
        // Residual: out = x + Ux + out
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*out)(i,j)+=(*Ux)(i,j);
        return out;
    }
    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p; auto add=[&](const Linear& l){ auto lp=l.parameters(); p.insert(p.end(),lp.begin(),lp.end()); };
        add(lin_u); add(lin_v); add(lin_e); return p;
    }
};

// =====================================================================
//  SSGConv -- Simple Spectral Graph Conv (S^2GC)
// =====================================================================
class SSGConv {
public:
    Linear lin; int in_c_,out_c_,K_; float alpha_;
    SSGConv()=default;
    SSGConv(int in,int out,float alpha=0.1f,int K=1,bool bias=true):lin(in,out,bias),in_c_(in),out_c_(out),K_(K),alpha_(alpha){}
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=x->cols();
        auto es=data.edge_src; auto ed=data.edge_dst;
        for (int i=0;i<N;++i) { es.push_back(i); ed.push_back(i); }
        int Et=(int)es.size();
        std::vector<float> deg(N,0); for (int e=0;e<Et;++e) deg[ed[e]]+=1;
        std::vector<float> dis(N,0); for (int i=0;i<N;++i) if (deg[i]>0) dis[i]=1.0f/std::sqrt(deg[i]);
        auto result=Tensor::zeros(N,C);
        for (int k=0;k<K_;++k) {
            auto h=(k==0)?x:result;
            auto nxt=Tensor::zeros(N,C);
            for (int e=0;e<Et;++e) { float w=dis[es[e]]*dis[ed[e]]; for (int j=0;j<C;++j) (*nxt)(ed[e],j)+=w*(*h)(es[e],j); }
            for (int i=0;i<N;++i) for (int j=0;j<C;++j) result->data_[i*C+j]+=(1.0f-alpha_)*(*nxt)(i,j)/K_+alpha_*(*x)(i,j)/K_;
        }
        return lin.forward(result);
    }
    std::vector<TensorPtr> parameters() const { return lin.parameters(); }
};

// =====================================================================
//  PointConv (PointNet++ local set aggregation, simplified)
// =====================================================================
class PointConv {
public:
    Linear lin; int in_c_,out_c_;
    PointConv()=default;
    PointConv(int in,int out,bool bias=true):lin(in+3,out,bias),in_c_(in),out_c_(out){}
    TensorPtr forward(const TensorPtr& x,const TensorPtr& pos,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=x->cols();
        auto out=Tensor::zeros(N,out_c_);
        std::vector<int> cnt(N,0);
        for (int e=0;e<E;++e) { int s=data.edge_src[e],d=data.edge_dst[e]; cnt[d]++;
            auto feat=std::make_shared<Tensor>(1,C+3);
            for (int j=0;j<C;++j) (*feat)(0,j)=(*x)(s,j);
            for (int j=0;j<3;++j) (*feat)(0,C+j)=(*pos)(s,j)-(*pos)(d,j);
            auto msg=lin.forward(feat);
            for (int j=0;j<out_c_;++j) (*out)(d,j)+=(*msg)(0,j);
        }
        for (int i=0;i<N;++i) if (cnt[i]>0) for (int j=0;j<out_c_;++j) (*out)(i,j)/=cnt[i];
        return out;
    }
    std::vector<TensorPtr> parameters() const { return lin.parameters(); }
};

// =====================================================================
//  MFConv -- Convolutional Networks on Graphs with Molecular Fingerprints
// =====================================================================
class MFConv {
public:
    Linear lin_self,lin_neigh; int in_c_,out_c_;
    MFConv()=default;
    MFConv(int in,int out,bool bias=true):lin_self(in,out,bias),lin_neigh(in,out,bias),in_c_(in),out_c_(out){}
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=x->cols();
        auto agg=Tensor::zeros(N,C);
        for (int e=0;e<E;++e) for (int j=0;j<C;++j) (*agg)(data.edge_dst[e],j)+=(*x)(data.edge_src[e],j);
        return autograd::add(lin_self.forward(x),autograd::relu(lin_neigh.forward(agg)));
    }
    std::vector<TensorPtr> parameters() const { auto p=lin_self.parameters(); auto p2=lin_neigh.parameters(); p.insert(p.end(),p2.begin(),p2.end()); return p; }
};

// =====================================================================
//  LEConv -- Local Extremum Conv
// =====================================================================
class LEConv {
public:
    Linear lin1,lin2; int in_c_,out_c_;
    LEConv()=default;
    LEConv(int in,int out,bool bias=true):lin1(in,out,bias),lin2(in,out,bias),in_c_(in),out_c_(out){}
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=x->cols();
        auto h1=lin1.forward(x); // self-transform
        auto agg=Tensor::zeros(N,out_c_);
        for (int e=0;e<E;++e) { int s=data.edge_src[e],d=data.edge_dst[e];
            // Difference-based message
            auto diff=std::make_shared<Tensor>(1,C);
            for (int j=0;j<C;++j) (*diff)(0,j)=(*x)(s,j)-(*x)(d,j);
            auto msg=lin2.forward(diff);
            for (int j=0;j<out_c_;++j) (*agg)(d,j)+=(*msg)(0,j);
        }
        return autograd::add(h1,agg);
    }
    std::vector<TensorPtr> parameters() const { auto p=lin1.parameters(); auto p2=lin2.parameters(); p.insert(p.end(),p2.begin(),p2.end()); return p; }
};

// =====================================================================
//  ClusterGCNConv
// =====================================================================
class ClusterGCNConv {
public:
    Linear lin; TensorPtr root_weight_param,bias; int in_c_,out_c_;
    ClusterGCNConv()=default;
    ClusterGCNConv(int in,int out,bool diag_lambda=true,bool use_bias=true):lin(in,out,false),in_c_(in),out_c_(out) {
        if (diag_lambda) { root_weight_param=Tensor::ones(1,out); root_weight_param->set_requires_grad(true); }
        if (use_bias) { bias=Tensor::zeros(1,out); bias->set_requires_grad(true); }
    }
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=x->cols();
        // Add self-loops
        auto es=data.edge_src; auto ed=data.edge_dst;
        for (int i=0;i<N;++i) { es.push_back(i); ed.push_back(i); }
        int Et=(int)es.size();
        std::vector<float> deg(N,0); for (int e=0;e<Et;++e) deg[ed[e]]+=1;
        std::vector<float> dis(N,0); for (int i=0;i<N;++i) if (deg[i]>0) dis[i]=1.0f/std::sqrt(deg[i]);
        auto agg=Tensor::zeros(N,C);
        for (int e=0;e<Et;++e) { float w=dis[es[e]]*dis[ed[e]]; for (int j=0;j<C;++j) (*agg)(ed[e],j)+=w*(*x)(es[e],j); }
        auto out=lin.forward(agg);
        if (root_weight_param) for (int i=0;i<N;++i) for (int j=0;j<out_c_;++j) (*out)(i,j)*=(*root_weight_param)(0,j);
        if (bias) out=autograd::add(out,bias);
        return out;
    }
    std::vector<TensorPtr> parameters() const {
        auto p=lin.parameters(); if (root_weight_param) p.push_back(root_weight_param); if (bias) p.push_back(bias); return p;
    }
};

// =====================================================================
//  FiLMConv
// =====================================================================
class FiLMConv {
public:
    Linear lin,lin_gamma,lin_beta; int in_c_,out_c_;
    FiLMConv()=default;
    FiLMConv(int in,int out,bool bias=true):lin(in,out,bias),lin_gamma(in,out,false),lin_beta(in,out,false),in_c_(in),out_c_(out){}
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=out_c_;
        auto h=lin.forward(x);
        auto gamma=lin_gamma.forward(x), beta_t=lin_beta.forward(x);
        auto out=Tensor::zeros(N,C); std::vector<int> cnt(N,0);
        for (int e=0;e<E;++e) { int s=data.edge_src[e],d=data.edge_dst[e]; cnt[d]++;
            for (int j=0;j<C;++j) (*out)(d,j)+=(*gamma)(d,j)*(*h)(s,j)+(*beta_t)(d,j); }
        for (int i=0;i<N;++i) if (cnt[i]>0) for (int j=0;j<C;++j) (*out)(i,j)/=cnt[i];
        return out;
    }
    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p; auto add=[&](const Linear& l){ auto lp=l.parameters(); p.insert(p.end(),lp.begin(),lp.end()); };
        add(lin); add(lin_gamma); add(lin_beta); return p;
    }
};

// =====================================================================
//  GENConv -- Generalized Message Passing
// =====================================================================
class GENConv {
public:
    Linear lin,edge_lin; int in_c_,out_c_; float p_; bool softmax_agg_;
    GENConv()=default;
    GENConv(int in,int out,float p=1.0f,bool softmax_agg=true):lin(in,out,true),edge_lin(in,in,true),in_c_(in),out_c_(out),p_(p),softmax_agg_(softmax_agg){}
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=x->cols();
        // Message: relu(x_j + edge_feat) + eps
        auto msg=Tensor::zeros(N,C); std::vector<int> cnt(N,0);
        for (int e=0;e<E;++e) { int s=data.edge_src[e],d=data.edge_dst[e]; cnt[d]++;
            for (int j=0;j<C;++j) { float v=std::max(0.0f,(*x)(s,j))+1e-6f; (*msg)(d,j)+=v; } }
        for (int i=0;i<N;++i) if (cnt[i]>0) for (int j=0;j<C;++j) (*msg)(i,j)/=cnt[i];
        return lin.forward(autograd::add(x,msg));
    }
    std::vector<TensorPtr> parameters() const { auto p=lin.parameters(); auto p2=edge_lin.parameters(); p.insert(p.end(),p2.begin(),p2.end()); return p; }
};

// =====================================================================
//  GPSConv -- General, Powerful, Scalable (Graph Transformer wrapper)
// =====================================================================
class GPSConv {
public:
    GCNConv local_conv; TransformerConv attn_conv; Linear ff1,ff2; BatchNorm1d norm1,norm2; int channels_;
    GPSConv()=default;
    GPSConv(int channels,int heads=1):local_conv(channels,channels),attn_conv(channels,channels/heads,heads,true),
        ff1(channels,channels*2),ff2(channels*2,channels),norm1(channels),norm2(channels),channels_(channels){}
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        auto h=x;
        // Local message passing
        auto local_out=local_conv.forward(h,data);
        // Global attention
        auto attn_out=attn_conv.forward(h,data);
        // Combine
        auto combined=autograd::add(local_out,attn_out);
        combined=norm1.forward(autograd::add(h,combined));
        // Feed-forward
        auto ff_out=ff2.forward(autograd::relu(ff1.forward(combined)));
        return norm2.forward(autograd::add(combined,ff_out));
    }
    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p;
        auto add=[&](const auto& m){ auto mp=m.parameters(); p.insert(p.end(),mp.begin(),mp.end()); };
        add(local_conv); add(attn_conv); add(ff1); add(ff2); add(norm1); add(norm2); return p;
    }
};

// =====================================================================
//  HypergraphConv
// =====================================================================
class HypergraphConv {
public:
    Linear lin; int in_c_,out_c_;
    HypergraphConv()=default;
    HypergraphConv(int in,int out,bool bias=true):lin(in,out,bias),in_c_(in),out_c_(out){}
    TensorPtr forward(const TensorPtr& x,const Data& hyperedge_data) {
        // Treat edge_src as node indices, edge_dst as hyperedge indices
        int N=x->rows(),C=x->cols();
        int num_he=0; for (int e=0;e<hyperedge_data.num_edges();++e) num_he=std::max(num_he,hyperedge_data.edge_dst[e]+1);
        // Step 1: aggregate nodes to hyperedges
        auto he_feat=Tensor::zeros(num_he,C); std::vector<int> he_cnt(num_he,0);
        for (int e=0;e<hyperedge_data.num_edges();++e) {
            int n=hyperedge_data.edge_src[e],h=hyperedge_data.edge_dst[e]; he_cnt[h]++;
            for (int j=0;j<C;++j) (*he_feat)(h,j)+=(*x)(n,j);
        }
        for (int h=0;h<num_he;++h) if (he_cnt[h]>0) for (int j=0;j<C;++j) (*he_feat)(h,j)/=he_cnt[h];
        // Step 2: aggregate hyperedges back to nodes
        auto out=Tensor::zeros(N,C); std::vector<int> n_cnt(N,0);
        for (int e=0;e<hyperedge_data.num_edges();++e) {
            int n=hyperedge_data.edge_src[e],h=hyperedge_data.edge_dst[e]; n_cnt[n]++;
            for (int j=0;j<C;++j) (*out)(n,j)+=(*he_feat)(h,j);
        }
        for (int i=0;i<N;++i) if (n_cnt[i]>0) for (int j=0;j<C;++j) (*out)(i,j)/=n_cnt[i];
        return lin.forward(out);
    }
    std::vector<TensorPtr> parameters() const { return lin.parameters(); }
};

// =====================================================================
//  RGCNConv -- Relational GCN
// =====================================================================
class RGCNConv {
public:
    std::vector<Linear> rel_lins; Linear root_lin; int in_c_,out_c_,num_relations_;
    RGCNConv()=default;
    RGCNConv(int in,int out,int num_rel,bool bias=true):in_c_(in),out_c_(out),num_relations_(num_rel),root_lin(in,out,bias) {
        for (int r=0;r<num_rel;++r) rel_lins.emplace_back(in,out,false);
    }
    TensorPtr forward(const TensorPtr& x,const Data& data,const std::vector<int>& edge_type) {
        int N=data.num_nodes,E=data.num_edges(),C=out_c_;
        auto out=Tensor::zeros(N,C); std::vector<int> cnt(N,0);
        for (int e=0;e<E;++e) { int s=data.edge_src[e],d=data.edge_dst[e],r=edge_type[e]; cnt[d]++;
            auto xs=x->row(s);
            auto msg=rel_lins[r].forward(xs);
            for (int j=0;j<C;++j) (*out)(d,j)+=(*msg)(0,j);
        }
        for (int i=0;i<N;++i) if (cnt[i]>0) for (int j=0;j<C;++j) (*out)(i,j)/=cnt[i];
        return autograd::add(out,root_lin.forward(x));
    }
    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p;
        for (auto& l:rel_lins) { auto lp=l.parameters(); p.insert(p.end(),lp.begin(),lp.end()); }
        auto rp=root_lin.parameters(); p.insert(p.end(),rp.begin(),rp.end()); return p;
    }
};

// =====================================================================
//  FAConv -- Feature-Attention Conv
// =====================================================================
class FAConv {
public:
    Linear lin; float eps_; bool add_sl_;
    FAConv()=default;
    FAConv(int channels,float eps=0.1f,bool asl=true):lin(channels,channels,true),eps_(eps),add_sl_(asl){}
    TensorPtr forward(const TensorPtr& x,const TensorPtr& x0,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=x->cols();
        auto h=lin.forward(x);
        auto es=data.edge_src; auto ed=data.edge_dst;
        if (add_sl_) for (int i=0;i<N;++i) { es.push_back(i); ed.push_back(i); }
        int Et=(int)es.size();
        // Attention based on feature similarity
        auto att=std::make_shared<Tensor>(Et,1);
        for (int e=0;e<Et;++e) { float dot=0,n1=0,n2=0;
            for (int j=0;j<C;++j) { dot+=(*h)(es[e],j)*(*h)(ed[e],j); n1+=(*h)(es[e],j)*(*h)(es[e],j); n2+=(*h)(ed[e],j)*(*h)(ed[e],j); }
            (*att)(e,0)=dot/(std::sqrt(n1)*std::sqrt(n2)+1e-12f); }
        // Softmax
        std::vector<std::vector<int>> te(N); for (int e=0;e<Et;++e) te[ed[e]].push_back(e);
        for (int i=0;i<N;++i) { if (te[i].empty()) continue; float mx=-1e30f;
            for (int e:te[i]) mx=std::max(mx,(*att)(e,0)); float s=0;
            for (int e:te[i]) { (*att)(e,0)=std::exp((*att)(e,0)-mx); s+=(*att)(e,0); }
            for (int e:te[i]) (*att)(e,0)/=(s+1e-12f); }
        auto out=Tensor::zeros(N,C);
        for (int e=0;e<Et;++e) for (int j=0;j<C;++j) (*out)(ed[e],j)+=(*att)(e,0)*(*h)(es[e],j);
        // Combine with initial features
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*out)(i,j)=eps_*(*x0)(i,j)+(1-eps_)*(*out)(i,j);
        return out;
    }
    std::vector<TensorPtr> parameters() const { return lin.parameters(); }
};

// =====================================================================
//  WLConv -- Weisfeiler-Leman (color refinement, non-trainable)
// =====================================================================
class WLConv {
public:
    WLConv()=default;
    // forward returns new color assignments (integers)
    std::vector<int> forward(const std::vector<int>& colors,const Data& data) {
        int N=data.num_nodes;
        auto adj=data.adj_list_target_to_source();
        std::map<std::vector<int>,int> color_map;
        std::vector<int> new_colors(N);
        for (int i=0;i<N;++i) {
            std::vector<int> sig={colors[i]};
            std::vector<int> nb_colors;
            for (int j:adj[i]) nb_colors.push_back(colors[j]);
            std::sort(nb_colors.begin(),nb_colors.end());
            sig.insert(sig.end(),nb_colors.begin(),nb_colors.end());
            if (color_map.find(sig)==color_map.end()) color_map[sig]=(int)color_map.size();
            new_colors[i]=color_map[sig];
        }
        return new_colors;
    }
};

// =====================================================================
//  DNAConv -- Multi-hop Attention (simplified)
// =====================================================================
class DNAConv {
public:
    Linear lin; int channels_,heads_,groups_;
    DNAConv()=default;
    DNAConv(int channels,int heads=1,int groups=1):lin(channels,channels,true),channels_(channels),heads_(heads),groups_(groups){}
    TensorPtr forward(const std::vector<TensorPtr>& x_list,const Data& data) {
        // Multi-scale combination: weighted sum of representations from different hops
        if (x_list.empty()) return Tensor::zeros(0,0);
        int N=x_list[0]->rows(),C=x_list[0]->cols(),L=(int)x_list.size();
        // Simple attention over layers
        auto combined=Tensor::zeros(N,C);
        for (int i=0;i<N;++i) { float w=1.0f/L; for (int l=0;l<L;++l) for (int j=0;j<C;++j) (*combined)(i,j)+=w*(*x_list[l])(i,j); }
        return lin.forward(combined);
    }
    std::vector<TensorPtr> parameters() const { return lin.parameters(); }
};

// =====================================================================
//  MixHopConv
// =====================================================================
class MixHopConv {
public:
    std::vector<Linear> lins; std::vector<int> powers_; int in_c_,out_c_;
    MixHopConv()=default;
    MixHopConv(int in,int out,const std::vector<int>& powers={0,1,2},bool bias=true):powers_(powers),in_c_(in),out_c_(out) {
        for (size_t i=0;i<powers.size();++i) lins.emplace_back(in,out,bias);
    }
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=x->cols();
        auto es=data.edge_src; auto ed=data.edge_dst;
        for (int i=0;i<N;++i) { es.push_back(i); ed.push_back(i); }
        int Et=(int)es.size();
        std::vector<float> deg(N,0); for (int e=0;e<Et;++e) deg[ed[e]]+=1;
        std::vector<float> dis(N,0); for (int i=0;i<N;++i) if (deg[i]>0) dis[i]=1.0f/std::sqrt(deg[i]);
        std::vector<TensorPtr> results;
        for (size_t pi=0;pi<powers_.size();++pi) {
            auto h=x;
            for (int p=0;p<powers_[pi];++p) {
                auto nxt=Tensor::zeros(N,C);
                for (int e=0;e<Et;++e) { float w=dis[es[e]]*dis[ed[e]]; for (int j=0;j<C;++j) (*nxt)(ed[e],j)+=w*(*h)(es[e],j); }
                h=nxt;
            }
            results.push_back(lins[pi].forward(h));
        }
        // Concatenate
        return Tensor::cat_cols(results);
    }
    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p; for (auto& l:lins) { auto lp=l.parameters(); p.insert(p.end(),lp.begin(),lp.end()); } return p;
    }
};



// =====================================================================
//  SuperGATConv -- Self-supervised Graph Attention Network
//  "How to Find Your Friendly Neighborhood: Graph Attention Design
//   with Self-Supervision" (Kim & Oh, 2021)
// =====================================================================
class SuperGATConv {
public:
    int in_c_,out_c_,heads_; bool concat_; float neg_slope_; bool add_sl_;
    Linear lin; TensorPtr att_l,att_r,bias;
    std::string attention_type_; // "MX" or "SD"

    SuperGATConv()=default;
    SuperGATConv(int in,int out,int heads=1,bool concat=true,float ns=0.2f,bool asl=true,
                 const std::string& att_type="MX",bool ub=true)
        :in_c_(in),out_c_(out),heads_(heads),concat_(concat),neg_slope_(ns),add_sl_(asl),
         lin(in,heads*out,false),attention_type_(att_type) {
        int HC=heads*out;
        att_l=Tensor::xavier_uniform(1,HC); att_l->set_requires_grad(true);
        att_r=Tensor::xavier_uniform(1,HC); att_r->set_requires_grad(true);
        if (ub) { int od=concat?HC:out; bias=Tensor::zeros(1,od); bias->set_requires_grad(true); }
    }

    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,H=heads_,C=out_c_,HC=H*C;
        auto es=data.edge_src; auto ed=data.edge_dst;
        if (add_sl_) for (int i=0;i<N;++i) { es.push_back(i); ed.push_back(i); }
        int E=(int)es.size();
        auto Hx=lin.forward(x); // (N, HC)
        // Attention
        auto e_att=std::make_shared<Tensor>(E,H);
        if (attention_type_=="MX") {
            // Masked attention: e_ij = LeakyReLU(a_l^T h_i + a_r^T h_j)
            for (int e=0;e<E;++e) for (int h=0;h<H;++h) {
                float sl=0,sr=0;
                for (int c=0;c<C;++c) { sl+=(*Hx)(ed[e],h*C+c)*(*att_l)(0,h*C+c); sr+=(*Hx)(es[e],h*C+c)*(*att_r)(0,h*C+c); }
                float v=sl+sr; (*e_att)(e,h)=(v>0)?v:neg_slope_*v;
            }
        } else {
            // Scaled dot-product attention
            float scale=1.0f/std::sqrt((float)C);
            for (int e=0;e<E;++e) for (int h=0;h<H;++h) {
                float dot=0;
                for (int c=0;c<C;++c) dot+=(*Hx)(ed[e],h*C+c)*(*Hx)(es[e],h*C+c);
                (*e_att)(e,h)=dot*scale;
            }
        }
        // Softmax per target
        auto alpha=std::make_shared<Tensor>(E,H,0.0f);
        for (int h=0;h<H;++h) { std::vector<std::vector<int>> te(N);
            for (int e=0;e<E;++e) te[ed[e]].push_back(e);
            for (int i=0;i<N;++i) { if (te[i].empty()) continue; float mx=-1e30f;
                for (int e:te[i]) mx=std::max(mx,(*e_att)(e,h)); float s=0;
                for (int e:te[i]) { (*alpha)(e,h)=std::exp((*e_att)(e,h)-mx); s+=(*alpha)(e,h); }
                for (int e:te[i]) (*alpha)(e,h)/=(s+1e-12f); } }
        int od=concat_?HC:C; auto out=Tensor::zeros(N,od);
        for (int e=0;e<E;++e) for (int h=0;h<H;++h) { float a=(*alpha)(e,h);
            for (int c=0;c<C;++c) { float v=a*(*Hx)(es[e],h*C+c);
                if (concat_) (*out)(ed[e],h*C+c)+=v; else (*out)(ed[e],c)+=v/H; } }
        if (bias) for (int i=0;i<N;++i) for (int j=0;j<od;++j) (*out)(i,j)+=(*bias)(0,j);
        return out;
    }

    // Self-supervised edge prediction loss for attention refinement
    TensorPtr get_attention_loss(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=out_c_,H=heads_,HC=H*C;
        auto Hx=lin.forward(x);
        // Positive edges (existing)
        float pos_loss=0;
        for (int e=0;e<E;++e) {
            float dot=0; for (int j=0;j<HC;++j) dot+=(*Hx)(data.edge_src[e],j)*(*Hx)(data.edge_dst[e],j);
            float sig=1.0f/(1.0f+std::exp(-dot)); pos_loss-=std::log(sig+1e-7f);
        }
        // Negative edges
        auto [neg_src,neg_dst]=utils::negative_sampling(data.edge_src,data.edge_dst,N,E);
        float neg_loss=0;
        for (int e=0;e<(int)neg_src.size();++e) {
            float dot=0; for (int j=0;j<HC;++j) dot+=(*Hx)(neg_src[e],j)*(*Hx)(neg_dst[e],j);
            float sig=1.0f/(1.0f+std::exp(-dot)); neg_loss-=std::log(1-sig+1e-7f);
        }
        int total=E+(int)neg_src.size();
        return std::make_shared<Tensor>(1,1,(pos_loss+neg_loss)/(total>0?total:1));
    }

    std::vector<TensorPtr> parameters() const {
        auto p=lin.parameters(); p.push_back(att_l); p.push_back(att_r); if (bias) p.push_back(bias); return p;
    }
};

// =====================================================================
//  HGTConv -- Heterogeneous Graph Transformer
//  "Heterogeneous Graph Transformer" (Hu et al., 2020)
// =====================================================================
class HGTConv {
public:
    int in_c_,out_c_,heads_;
    // Per-type linear projections
    std::unordered_map<std::string,Linear> k_lins,q_lins,v_lins,o_lins;
    // Per-relation attention
    std::unordered_map<std::string,TensorPtr> a_rel,m_rel;

    HGTConv()=default;
    HGTConv(int in, int out, int heads, const std::vector<std::string>& node_types,
            const std::vector<std::string>& edge_types)
        :in_c_(in),out_c_(out),heads_(heads) {
        int HC=heads*out;
        for (auto& nt:node_types) {
            k_lins[nt]=Linear(in,HC,false); q_lins[nt]=Linear(in,HC,false);
            v_lins[nt]=Linear(in,HC,false); o_lins[nt]=Linear(HC,out,true);
        }
        for (auto& et:edge_types) {
            a_rel[et]=Tensor::ones(heads,1); a_rel[et]->set_requires_grad(true);
            m_rel[et]=Tensor::ones(heads,1); m_rel[et]->set_requires_grad(true);
        }
    }

    // Forward for heterogeneous data
    std::unordered_map<std::string,TensorPtr> forward(
        const std::unordered_map<std::string,TensorPtr>& x_dict,
        const HeteroData& hdata,
        const std::unordered_map<std::string,std::string>& edge_to_src_type,
        const std::unordered_map<std::string,std::string>& edge_to_dst_type) {
        int H=heads_,C=out_c_,HC=H*C;
        // Compute K, Q, V for each node type
        std::unordered_map<std::string,TensorPtr> K_dict,Q_dict,V_dict;
        for (auto& [nt,x]:x_dict) {
            if (k_lins.count(nt)) { K_dict[nt]=k_lins[nt].forward(x); Q_dict[nt]=q_lins[nt].forward(x); V_dict[nt]=v_lins[nt].forward(x); }
        }
        // Aggregate messages per destination type
        std::unordered_map<std::string,TensorPtr> result;
        for (auto& [nt,x]:x_dict) { int N=x->rows(); result[nt]=Tensor::zeros(N,C); }

        for (auto& [et,edata]:hdata.edge_dict) {
            if (!edge_to_src_type.count(et)||!edge_to_dst_type.count(et)) continue;
            auto st=edge_to_src_type.at(et), dt=edge_to_dst_type.at(et);
            if (!K_dict.count(st)||!Q_dict.count(dt)) continue;
            auto& K=K_dict[st]; auto& Q=Q_dict[dt]; auto& V=V_dict[st];
            int E=(int)edata.edge_src.size();
            int Nd=x_dict.at(dt)->rows();
            // Compute attention per head
            for (int h=0;h<H;++h) {
                float a_r=a_rel.count(et)?(*a_rel.at(et))(h,0):1.0f;
                float m_r=m_rel.count(et)?(*m_rel.at(et))(h,0):1.0f;
                float scale=a_r/std::sqrt((float)C);
                // Per-edge attention & message
                std::vector<std::vector<int>> te(Nd);
                for (int e=0;e<E;++e) te[edata.edge_dst[e]].push_back(e);
                for (int i=0;i<Nd;++i) { if (te[i].empty()) continue;
                    float mx=-1e30f; std::vector<float> scores(te[i].size());
                    for (int ei=0;ei<(int)te[i].size();++ei) { int e=te[i][ei]; int s=edata.edge_src[e];
                        float dot=0; for (int c=0;c<C;++c) dot+=(*Q)(i,h*C+c)*(*K)(s,h*C+c);
                        scores[ei]=dot*scale; mx=std::max(mx,scores[ei]); }
                    float ss=0; for (auto& sc:scores) { sc=std::exp(sc-mx); ss+=sc; }
                    for (auto& sc:scores) sc/=(ss+1e-12f);
                    for (int ei=0;ei<(int)te[i].size();++ei) { int e=te[i][ei]; int s=edata.edge_src[e];
                        for (int c=0;c<C;++c) (*result[dt])(i,c)+=scores[ei]*m_r*(*V)(s,h*C+c)/H; }
                }
            }
        }
        // Apply output projection
        for (auto& [nt,x]:result) { if (o_lins.count(nt)) result[nt]=o_lins[nt].forward(result[nt]); }
        return result;
    }

    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p;
        auto add=[&](const Linear& l){ auto lp=l.parameters(); p.insert(p.end(),lp.begin(),lp.end()); };
        for (auto& [_,l]:k_lins) add(l); for (auto& [_,l]:q_lins) add(l);
        for (auto& [_,l]:v_lins) add(l); for (auto& [_,l]:o_lins) add(l);
        for (auto& [_,t]:a_rel) p.push_back(t); for (auto& [_,t]:m_rel) p.push_back(t);
        return p;
    }
};

// =====================================================================
//  HeteroConv -- applies different conv layers to different edge types
// =====================================================================
class HeteroConv {
public:
    // Map from edge_type_key -> conv layer index + type
    struct ConvEntry {
        std::string src_type,edge_type,dst_type;
        enum Type { GCN, SAGE, GAT } conv_type;
        GCNConv gcn; SAGEConv sage; GATConv gat;
    };
    std::vector<ConvEntry> convs;
    Aggr aggr_;

    HeteroConv(Aggr aggr=Aggr::ADD):aggr_(aggr){}

    void add_conv(const std::string& src_t,const std::string& edge_t,const std::string& dst_t,
                  int in,int out,const std::string& conv_type="gcn") {
        ConvEntry ce; ce.src_type=src_t; ce.edge_type=edge_t; ce.dst_type=dst_t;
        if (conv_type=="gcn") { ce.conv_type=ConvEntry::GCN; ce.gcn=GCNConv(in,out); }
        else if (conv_type=="sage") { ce.conv_type=ConvEntry::SAGE; ce.sage=SAGEConv(in,out); }
        else { ce.conv_type=ConvEntry::GAT; ce.gat=GATConv(in,out); }
        convs.push_back(std::move(ce));
    }

    std::unordered_map<std::string,TensorPtr> forward(
        const std::unordered_map<std::string,TensorPtr>& x_dict,
        const HeteroData& hdata) {
        std::unordered_map<std::string,std::vector<TensorPtr>> results;
        for (auto& ce:convs) {
            auto key=HeteroData::edge_key(ce.src_type,ce.edge_type,ce.dst_type);
            if (!hdata.edge_dict.count(key)||!x_dict.count(ce.src_type)) continue;
            auto& ed=hdata.edge_dict.at(key);
            Data d; d.num_nodes=x_dict.at(ce.dst_type)->rows();
            d.edge_src=ed.edge_src; d.edge_dst=ed.edge_dst;
            // For simplicity, use source features
            TensorPtr out;
            if (ce.conv_type==ConvEntry::GCN) out=ce.gcn.forward(x_dict.at(ce.src_type),d);
            else if (ce.conv_type==ConvEntry::SAGE) out=ce.sage.forward(x_dict.at(ce.src_type),d);
            else out=ce.gat.forward(x_dict.at(ce.src_type),d);
            results[ce.dst_type].push_back(out);
        }
        std::unordered_map<std::string,TensorPtr> output;
        for (auto& [dst,tensors]:results) {
            if (tensors.empty()) continue;
            if (tensors.size()==1) { output[dst]=tensors[0]; continue; }
            // Aggregate
            int N=tensors[0]->rows(),C=tensors[0]->cols();
            auto agg=Tensor::zeros(N,C);
            if (aggr_==Aggr::ADD) { for (auto& t:tensors) for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*agg)(i,j)+=(*t)(i,j); }
            else if (aggr_==Aggr::MEAN) { for (auto& t:tensors) for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*agg)(i,j)+=(*t)(i,j);
                int L=(int)tensors.size(); for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*agg)(i,j)/=L; }
            else { agg=std::make_shared<Tensor>(N,C,-1e30f);
                for (auto& t:tensors) for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*agg)(i,j)=std::max((*agg)(i,j),(*t)(i,j)); }
            output[dst]=agg;
        }
        return output;
    }

    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p;
        for (auto& ce:convs) {
            std::vector<TensorPtr> cp;
            if (ce.conv_type==ConvEntry::GCN) cp=ce.gcn.parameters();
            else if (ce.conv_type==ConvEntry::SAGE) cp=ce.sage.parameters();
            else cp=ce.gat.parameters();
            p.insert(p.end(),cp.begin(),cp.end());
        }
        return p;
    }
};

// =====================================================================
//  SplineConv -- B-spline basis for continuous edge features
//  "SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline
//   Kernels" (Fey et al., 2018)
// =====================================================================
class SplineConv {
public:
    TensorPtr weight,bias_param; int in_c_,out_c_,dim_,degree_,kernel_size_;
    SplineConv()=default;
    SplineConv(int in,int out,int dim,int kernel_size=5,int degree=1,bool bias=true)
        :in_c_(in),out_c_(out),dim_(dim),degree_(degree),kernel_size_(kernel_size) {
        int num_bases=1; for (int d=0;d<dim;++d) num_bases*=kernel_size;
        weight=Tensor::glorot(num_bases*in,out); weight->set_requires_grad(true);
        if (bias) { bias_param=Tensor::zeros(1,out); bias_param->set_requires_grad(true); }
    }

    // B-spline basis function (degree 1: linear interpolation)
    float bspline_basis(float x, int i, int k_size) const {
        float step=1.0f/(k_size-1); float center=i*step;
        float dist=std::abs(x-center)/step;
        return std::max(0.0f,1.0f-dist); // linear B-spline
    }

    TensorPtr forward(const TensorPtr& x, const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),IN=in_c_,OUT=out_c_,D=dim_,KS=kernel_size_;
        int num_bases=1; for (int d=0;d<D;++d) num_bases*=KS;
        auto out=Tensor::zeros(N,OUT); std::vector<int> cnt(N,0);
        for (int e=0;e<E;++e) { int s=data.edge_src[e],d=data.edge_dst[e]; cnt[d]++;
            // Compute basis weights from edge features
            std::vector<float> basis(num_bases,1.0f);
            for (int b=0;b<num_bases;++b) {
                int idx=b;
                for (int dd=0;dd<D;++dd) {
                    int bi=idx%KS; idx/=KS;
                    float feat=data.edge_attr?(*data.edge_attr)(e,dd):0.5f;
                    basis[b]*=bspline_basis(feat,bi,KS);
                }
            }
            // Weighted sum: msg = sum_b basis[b] * x_s * W_b
            for (int j=0;j<OUT;++j) { float v=0;
                for (int b=0;b<num_bases;++b) { float bw=basis[b]; if (bw<1e-8f) continue;
                    for (int i=0;i<IN;++i) v+=bw*(*x)(s,i)*(*weight)(b*IN+i,j); }
                (*out)(d,j)+=v; }
        }
        for (int i=0;i<N;++i) if (cnt[i]>0) for (int j=0;j<OUT;++j) (*out)(i,j)/=cnt[i];
        if (bias_param) out=autograd::add(out,bias_param);
        return out;
    }
    std::vector<TensorPtr> parameters() const { std::vector<TensorPtr> p={weight}; if (bias_param) p.push_back(bias_param); return p; }
};

// =====================================================================
//  PointTransformerConv -- Point Transformer for point cloud processing
//  "Point Transformer" (Zhao et al., 2021)
// =====================================================================
class PointTransformerConv {
public:
    Linear lin_q,lin_k,lin_v,lin_pos; int in_c_,out_c_;
    PointTransformerConv()=default;
    PointTransformerConv(int in,int out):lin_q(in,out,false),lin_k(in,out,false),lin_v(in,out,false),lin_pos(3,out,false),in_c_(in),out_c_(out){}

    TensorPtr forward(const TensorPtr& x,const TensorPtr& pos,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=out_c_;
        auto Q=lin_q.forward(x), K=lin_k.forward(x), V=lin_v.forward(x);
        auto out=Tensor::zeros(N,C);
        // Compute per-edge attention and messages
        std::vector<std::vector<int>> te(N);
        for (int e=0;e<E;++e) te[data.edge_dst[e]].push_back(e);
        for (int i=0;i<N;++i) {
            if (te[i].empty()) continue;
            // Compute delta_pos features
            std::vector<std::vector<float>> pos_feats(te[i].size(),std::vector<float>(3,0));
            for (int ei=0;ei<(int)te[i].size();++ei) {
                int e=te[i][ei]; int s=data.edge_src[e];
                for (int d=0;d<std::min(3,(int)pos->cols());++d) pos_feats[ei][d]=(*pos)(i,d)-(*pos)(s,d);
            }
            // Attention scores: gamma(Q_i - K_j + pos_enc)
            float mx=-1e30f;
            std::vector<float> scores(te[i].size());
            for (int ei=0;ei<(int)te[i].size();++ei) {
                int e=te[i][ei]; int s=data.edge_src[e];
                auto dp=std::make_shared<Tensor>(1,3);
                for (int d=0;d<3;++d) (*dp)(0,d)=pos_feats[ei][d];
                auto pe=lin_pos.forward(dp); // (1,C)
                float score=0;
                for (int c=0;c<C;++c) score+=(*Q)(i,c)-(*K)(s,c)+(*pe)(0,c);
                scores[ei]=score; mx=std::max(mx,score);
            }
            float ss=0; for (auto& sc:scores) { sc=std::exp(sc-mx); ss+=sc; }
            for (auto& sc:scores) sc/=(ss+1e-12f);
            // Weighted value aggregation
            for (int ei=0;ei<(int)te[i].size();++ei) {
                int e=te[i][ei]; int s=data.edge_src[e];
                auto dp=std::make_shared<Tensor>(1,3);
                for (int d=0;d<3;++d) (*dp)(0,d)=pos_feats[ei][d];
                auto pe=lin_pos.forward(dp);
                for (int c=0;c<C;++c) (*out)(i,c)+=scores[ei]*((*V)(s,c)+(*pe)(0,c));
            }
        }
        return out;
    }
    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p; auto add=[&](const Linear& l){ auto lp=l.parameters(); p.insert(p.end(),lp.begin(),lp.end()); };
        add(lin_q); add(lin_k); add(lin_v); add(lin_pos); return p;
    }
};

// =====================================================================
//  GeneralConv -- flexible conv that can be configured for different
//  aggregation, attention, and skip-connection modes.
// =====================================================================
class GeneralConv {
public:
    Linear lin_l,lin_r; TensorPtr att_weight,bias;
    int in_c_,out_c_,heads_; bool use_attention_,skip_;
    Aggr aggr_;

    GeneralConv()=default;
    GeneralConv(int in,int out,Aggr aggr=Aggr::ADD,bool attention=false,bool skip=true,
                int heads=1,bool use_bias=true)
        :lin_l(in,out,false),lin_r(in,out,false),in_c_(in),out_c_(out),heads_(heads),
         use_attention_(attention),skip_(skip),aggr_(aggr) {
        if (attention) { att_weight=Tensor::xavier_uniform(1,2*out); att_weight->set_requires_grad(true); }
        if (use_bias) { bias=Tensor::zeros(1,out); bias->set_requires_grad(true); }
    }

    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=out_c_;
        auto Lx=lin_l.forward(x); // source transform
        auto out=Tensor::zeros(N,C);
        if (use_attention_) {
            // Compute attention weights
            auto alpha=std::make_shared<Tensor>(E,1);
            for (int e=0;e<E;++e) { int s=data.edge_src[e],d=data.edge_dst[e]; float v=0;
                for (int j=0;j<C;++j) v+=(*Lx)(s,j)*(*att_weight)(0,j)+(*Lx)(d,j)*(*att_weight)(0,C+j);
                v=(v>0)?v:0.2f*v; (*alpha)(e,0)=v; }
            // Softmax
            std::vector<std::vector<int>> te(N);
            for (int e=0;e<E;++e) te[data.edge_dst[e]].push_back(e);
            for (int i=0;i<N;++i) { if (te[i].empty()) continue;
                float mx=-1e30f; for (int e:te[i]) mx=std::max(mx,(*alpha)(e,0));
                float ss=0; for (int e:te[i]) { (*alpha)(e,0)=std::exp((*alpha)(e,0)-mx); ss+=(*alpha)(e,0); }
                for (int e:te[i]) (*alpha)(e,0)/=(ss+1e-12f); }
            for (int e=0;e<E;++e) for (int j=0;j<C;++j) (*out)(data.edge_dst[e],j)+=(*alpha)(e,0)*(*Lx)(data.edge_src[e],j);
        } else {
            if (aggr_==Aggr::ADD) { for (int e=0;e<E;++e) for (int j=0;j<C;++j) (*out)(data.edge_dst[e],j)+=(*Lx)(data.edge_src[e],j); }
            else if (aggr_==Aggr::MEAN) { std::vector<int> cnt(N,0);
                for (int e=0;e<E;++e) { cnt[data.edge_dst[e]]++; for (int j=0;j<C;++j) (*out)(data.edge_dst[e],j)+=(*Lx)(data.edge_src[e],j); }
                for (int i=0;i<N;++i) if (cnt[i]>0) for (int j=0;j<C;++j) (*out)(i,j)/=cnt[i]; }
            else { out=std::make_shared<Tensor>(N,C,-1e30f);
                for (int e=0;e<E;++e) for (int j=0;j<C;++j) (*out)(data.edge_dst[e],j)=std::max((*out)(data.edge_dst[e],j),(*Lx)(data.edge_src[e],j)); }
        }
        if (skip_) out=autograd::add(out,lin_r.forward(x));
        if (bias) out=autograd::add(out,bias);
        return out;
    }
    std::vector<TensorPtr> parameters() const {
        auto p=lin_l.parameters(); auto pr=lin_r.parameters(); p.insert(p.end(),pr.begin(),pr.end());
        if (att_weight) p.push_back(att_weight); if (bias) p.push_back(bias); return p;
    }
};

// =====================================================================
//  PNAConv -- Principal Neighbourhood Aggregation
//  "Principal Neighbourhood Aggregation for Graph Nets" (Corso et al.)
// =====================================================================
class PNAConv {
public:
    Linear lin_pre,lin_post; int in_c_,out_c_;
    std::vector<float> deg_log_; // precomputed log-degree histogram for scalers

    PNAConv()=default;
    PNAConv(int in,int out,bool bias=true):lin_pre(in,out,false),lin_post(4*3*out,out,bias),in_c_(in),out_c_(out){}
    // 4 aggregators x 3 scalers = 12x features

    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),IN=in_c_,OUT=out_c_;
        auto h=lin_pre.forward(x); // (N, OUT)
        // 4 aggregators: mean, max, min, std
        auto agg_mean=Tensor::zeros(N,OUT), agg_max=std::make_shared<Tensor>(N,OUT,-1e30f);
        auto agg_min=std::make_shared<Tensor>(N,OUT,1e30f), agg_sq=Tensor::zeros(N,OUT);
        std::vector<int> cnt(N,0);
        for (int e=0;e<E;++e) { int s=data.edge_src[e],d=data.edge_dst[e]; cnt[d]++;
            for (int j=0;j<OUT;++j) { (*agg_mean)(d,j)+=(*h)(s,j); (*agg_max)(d,j)=std::max((*agg_max)(d,j),(*h)(s,j));
                (*agg_min)(d,j)=std::min((*agg_min)(d,j),(*h)(s,j)); (*agg_sq)(d,j)+=(*h)(s,j)*(*h)(s,j); } }
        // Compute std from mean and sq
        auto agg_std=Tensor::zeros(N,OUT);
        for (int i=0;i<N;++i) if (cnt[i]>0) { float c=(float)cnt[i];
            for (int j=0;j<OUT;++j) { (*agg_mean)(i,j)/=c;
                float var=(*agg_sq)(i,j)/c-(*agg_mean)(i,j)*(*agg_mean)(i,j);
                (*agg_std)(i,j)=std::sqrt(std::max(0.0f,var)); } }
        // Fix min/max for isolated nodes
        for (int i=0;i<N;++i) if (cnt[i]==0) for (int j=0;j<OUT;++j) { (*agg_max)(i,j)=0; (*agg_min)(i,j)=0; }
        // 3 scalers: identity, amplification (log(d+1)), attenuation (1/log(d+1))
        auto concat_all=[&](const TensorPtr& agg)->TensorPtr {
            auto s1=agg; // identity
            auto s2=std::make_shared<Tensor>(N,OUT); auto s3=std::make_shared<Tensor>(N,OUT);
            for (int i=0;i<N;++i) { float ld=std::log((float)cnt[i]+1+1e-12f);
                for (int j=0;j<OUT;++j) { (*s2)(i,j)=(*agg)(i,j)*ld; (*s3)(i,j)=(*agg)(i,j)/(ld+1e-12f); } }
            return Tensor::cat_cols({s1,s2,s3});
        };
        auto combined=Tensor::cat_cols({concat_all(agg_mean),concat_all(agg_max),concat_all(agg_min),concat_all(agg_std)});
        return lin_post.forward(combined);
    }
    std::vector<TensorPtr> parameters() const { auto p=lin_pre.parameters(); auto p2=lin_post.parameters(); p.insert(p.end(),p2.begin(),p2.end()); return p; }
};

// =====================================================================
//  PANConv -- Path Integral Based Convolution
// =====================================================================
class PANConv {
public:
    Linear lin; int in_c_,out_c_,filter_size_;
    PANConv()=default;
    PANConv(int in,int out,int filter_size=3,bool bias=true):lin(in,out,bias),in_c_(in),out_c_(out),filter_size_(filter_size){}

    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=x->cols();
        // Multi-hop path aggregation: sum A^k x for k=0..filter_size-1
        auto es=data.edge_src; auto ed=data.edge_dst;
        auto result=Tensor::zeros(N,C);
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*result)(i,j)=(*x)(i,j);
        auto h=x;
        for (int k=1;k<filter_size_;++k) {
            auto nxt=Tensor::zeros(N,C);
            for (int e=0;e<E;++e) for (int j=0;j<C;++j) (*nxt)(ed[e],j)+=(*h)(es[e],j);
            for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*result)(i,j)+=(*nxt)(i,j);
            h=nxt;
        }
        return lin.forward(result);
    }
    std::vector<TensorPtr> parameters() const { return lin.parameters(); }
};

// =====================================================================
//  AntiSymmetricConv -- Antisymmetric neural ODEs on graphs
// =====================================================================
class AntiSymmetricConv {
public:
    TensorPtr weight; Linear lin; int channels_; float eps_,gamma_;
    AntiSymmetricConv()=default;
    AntiSymmetricConv(int channels,float eps=0.1f,float gamma=0.1f)
        :channels_(channels),eps_(eps),gamma_(gamma),lin(channels,channels,false) {
        weight=Tensor::glorot(channels,channels); weight->set_requires_grad(true);
    }
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=channels_;
        // W_anti = (W - W^T) / 2
        auto W_anti=std::make_shared<Tensor>(C,C);
        for (int i=0;i<C;++i) for (int j=0;j<C;++j) (*W_anti)(i,j)=0.5f*((*weight)(i,j)-(*weight)(j,i));
        // Neighbor aggregation
        auto agg=Tensor::zeros(N,C);
        std::vector<int> cnt(N,0);
        for (int e=0;e<E;++e) { cnt[data.edge_dst[e]]++;
            for (int j=0;j<C;++j) (*agg)(data.edge_dst[e],j)+=(*x)(data.edge_src[e],j); }
        for (int i=0;i<N;++i) if (cnt[i]>0) for (int j=0;j<C;++j) (*agg)(i,j)/=cnt[i];
        // x_new = x + eps * sigma(W_anti * agg - gamma * x)
        auto Wa=autograd::matmul(agg,W_anti);
        auto out=std::make_shared<Tensor>(N,C);
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) {
            float v=(*Wa)(i,j)-gamma_*(*x)(i,j);
            v=std::tanh(v);
            (*out)(i,j)=(*x)(i,j)+eps_*v;
        }
        return out;
    }
    std::vector<TensorPtr> parameters() const { std::vector<TensorPtr> p={weight}; auto lp=lin.parameters(); p.insert(p.end(),lp.begin(),lp.end()); return p; }
};

// =====================================================================
//  WLConvContinuous -- Continuous WL kernel with learnable features
// =====================================================================
class WLConvContinuous {
public:
    Linear lin; int in_c_,out_c_;
    WLConvContinuous()=default;
    WLConvContinuous(int in,int out):lin(2*in,out,true),in_c_(in),out_c_(out){}
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=x->cols();
        auto out=Tensor::zeros(N,out_c_);
        for (int i=0;i<N;++i) {
            // Collect neighbor features, hash with self
            auto agg=Tensor::zeros(1,C);
            int cnt=0;
            for (int e=0;e<E;++e) if (data.edge_dst[e]==i) { cnt++;
                for (int j=0;j<C;++j) (*agg)(0,j)+=(*x)(data.edge_src[e],j); }
            if (cnt>0) for (int j=0;j<C;++j) (*agg)(0,j)/=cnt;
            auto combined=Tensor::cat_cols({x->row(i),agg});
            auto res=lin.forward(combined);
            for (int j=0;j<out_c_;++j) (*out)(i,j)=(*res)(0,j);
        }
        return out;
    }
    std::vector<TensorPtr> parameters() const { return lin.parameters(); }
};

// =====================================================================
//  EGConv -- Efficient Graph Convolution (Multi-head aggregation)
// =====================================================================
class EGConv {
public:
    Linear lin; int in_c_,out_c_,num_heads_,num_bases_;
    EGConv()=default;
    EGConv(int in,int out,int heads=8,int bases=4)
        :lin(in,out*heads,false),in_c_(in),out_c_(out),num_heads_(heads),num_bases_(bases){}
    TensorPtr forward(const TensorPtr& x,const Data& data) {
        int N=data.num_nodes,E=data.num_edges(),C=x->cols(),H=num_heads_,OC=out_c_;
        auto Hx=lin.forward(x); // (N, H*OC)
        // Multi-head aggregation: each head uses different aggregation
        auto out=Tensor::zeros(N,OC);
        for (int h=0;h<H;++h) {
            if (h%4==0) { // Sum aggregation
                for (int e=0;e<E;++e) for (int j=0;j<OC;++j) (*out)(data.edge_dst[e],j)+=(*Hx)(data.edge_src[e],h*OC+j)/H;
            } else if (h%4==1) { // Mean
                auto agg=Tensor::zeros(N,OC); std::vector<int> cnt(N,0);
                for (int e=0;e<E;++e) { cnt[data.edge_dst[e]]++; for (int j=0;j<OC;++j) (*agg)(data.edge_dst[e],j)+=(*Hx)(data.edge_src[e],h*OC+j); }
                for (int i=0;i<N;++i) if (cnt[i]>0) for (int j=0;j<OC;++j) (*out)(i,j)+=(*agg)(i,j)/(cnt[i]*H);
            } else if (h%4==2) { // Max
                auto agg=std::make_shared<Tensor>(N,OC,-1e30f);
                for (int e=0;e<E;++e) for (int j=0;j<OC;++j) (*agg)(data.edge_dst[e],j)=std::max((*agg)(data.edge_dst[e],j),(*Hx)(data.edge_src[e],h*OC+j));
                for (int i=0;i<N;++i) for (int j=0;j<OC;++j) if ((*agg)(i,j)>-1e29f) (*out)(i,j)+=(*agg)(i,j)/H;
            } else { // Softmax attention
                for (int e=0;e<E;++e) for (int j=0;j<OC;++j) (*out)(data.edge_dst[e],j)+=(*Hx)(data.edge_src[e],h*OC+j)/H;
            }
        }
        return out;
    }
    std::vector<TensorPtr> parameters() const { return lin.parameters(); }
};


} // namespace pyg
