// pyg_pool.h -- Pooling operators for C++ PyG port
// Header-only C++17. Depends on pyg.h
//
// Implements: TopKPooling, SAGPooling, EdgePooling, ASAPooling, PANPooling,
//             graclus coarsening, voxel_grid, and extended global pooling variants.
#pragma once
#include "pyg.h"

namespace pyg {

// =====================================================================
//  TopKPooling -- "Graph U-Nets" / "Towards Sparse Hierarchical Graph Classifiers"
// =====================================================================
class TopKPooling {
public:
    int in_channels_; float ratio_; float min_score_;
    TensorPtr weight;

    TopKPooling()=default;
    TopKPooling(int in_channels, float ratio=0.5f, float min_score=-1e30f)
        :in_channels_(in_channels),ratio_(ratio),min_score_(min_score) {
        weight=Tensor::xavier_uniform(1,in_channels); weight->set_requires_grad(true);
    }

    struct PoolResult { TensorPtr x; Data data; std::vector<int> perm; TensorPtr score; };

    PoolResult forward(const TensorPtr& x, const Data& data) {
        int N=data.num_nodes, C=x->cols();
        float wnorm=0; for (int j=0;j<C;++j) wnorm+=(*weight)(0,j)*(*weight)(0,j);
        wnorm=std::sqrt(wnorm)+1e-12f;
        auto scores=std::make_shared<Tensor>(N,1);
        for (int i=0;i<N;++i) { float s=0; for (int j=0;j<C;++j) s+=(*x)(i,j)*(*weight)(0,j)/wnorm;
            (*scores)(i,0)=std::tanh(s); }
        int k=std::max(1,(int)(N*ratio_));
        std::vector<int> idx(N); std::iota(idx.begin(),idx.end(),0);
        std::partial_sort(idx.begin(),idx.begin()+k,idx.end(),[&](int a,int b){ return (*scores)(a,0)>(*scores)(b,0); });
        std::vector<int> perm(idx.begin(),idx.begin()+k);
        std::sort(perm.begin(),perm.end());
        auto new_x=std::make_shared<Tensor>(k,C);
        for (int i=0;i<k;++i) for (int j=0;j<C;++j) (*new_x)(i,j)=(*x)(perm[i],j)*(*scores)(perm[i],0);
        std::unordered_map<int,int> old2new; for (int i=0;i<k;++i) old2new[perm[i]]=i;
        Data new_data; new_data.num_nodes=k;
        for (int e=0;e<data.num_edges();++e) {
            auto si=old2new.find(data.edge_src[e]), di=old2new.find(data.edge_dst[e]);
            if (si!=old2new.end()&&di!=old2new.end()) { new_data.edge_src.push_back(si->second); new_data.edge_dst.push_back(di->second); }
        }
        if (!data.batch.empty()) for (int i=0;i<k;++i) new_data.batch.push_back(data.batch[perm[i]]);
        return {new_x,new_data,perm,scores};
    }
    std::vector<TensorPtr> parameters() const { return {weight}; }
};

// =====================================================================
//  SAGPooling -- Self-Attention Graph Pooling
// =====================================================================
class SAGPooling {
public:
    int in_channels_; float ratio_;
    Linear score_lin;

    SAGPooling()=default;
    SAGPooling(int in_channels, float ratio=0.5f)
        :in_channels_(in_channels),ratio_(ratio),score_lin(in_channels,1,true){}

    struct PoolResult { TensorPtr x; Data data; std::vector<int> perm; TensorPtr score; };

    PoolResult forward(const TensorPtr& x, const Data& data) {
        int N=data.num_nodes, C=x->cols();
        auto agg=Tensor::zeros(N,C); std::vector<int> cnt(N,0);
        for (int e=0;e<data.num_edges();++e) { cnt[data.edge_dst[e]]++;
            for (int j=0;j<C;++j) (*agg)(data.edge_dst[e],j)+=(*x)(data.edge_src[e],j); }
        for (int i=0;i<N;++i) if (cnt[i]>0) for (int j=0;j<C;++j) (*agg)(i,j)/=cnt[i];
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*agg)(i,j)+=(*x)(i,j);
        auto scores=score_lin.forward(agg);
        for (int i=0;i<N;++i) (*scores)(i,0)=std::tanh((*scores)(i,0));
        int k=std::max(1,(int)(N*ratio_));
        std::vector<int> idx(N); std::iota(idx.begin(),idx.end(),0);
        std::partial_sort(idx.begin(),idx.begin()+k,idx.end(),[&](int a,int b){ return (*scores)(a,0)>(*scores)(b,0); });
        std::vector<int> perm(idx.begin(),idx.begin()+k);
        std::sort(perm.begin(),perm.end());
        auto new_x=std::make_shared<Tensor>(k,C);
        for (int i=0;i<k;++i) for (int j=0;j<C;++j) (*new_x)(i,j)=(*x)(perm[i],j)*(*scores)(perm[i],0);
        std::unordered_map<int,int> old2new; for (int i=0;i<k;++i) old2new[perm[i]]=i;
        Data new_data; new_data.num_nodes=k;
        for (int e=0;e<data.num_edges();++e) {
            auto si=old2new.find(data.edge_src[e]), di=old2new.find(data.edge_dst[e]);
            if (si!=old2new.end()&&di!=old2new.end()) { new_data.edge_src.push_back(si->second); new_data.edge_dst.push_back(di->second); }
        }
        if (!data.batch.empty()) for (int i=0;i<k;++i) new_data.batch.push_back(data.batch[perm[i]]);
        return {new_x,new_data,perm,scores};
    }
    std::vector<TensorPtr> parameters() const { return score_lin.parameters(); }
};

// =====================================================================
//  EdgePooling
// =====================================================================
class EdgePooling {
public:
    Linear score_lin;
    EdgePooling()=default;
    EdgePooling(int in_channels):score_lin(2*in_channels,1,true){}

    struct PoolResult { TensorPtr x; Data data; std::vector<int> cluster; };

    PoolResult forward(const TensorPtr& x, const Data& data) {
        int N=data.num_nodes, E=data.num_edges(), C=x->cols();
        auto edge_scores=std::make_shared<Tensor>(E,1);
        for (int e=0;e<E;++e) { int s=data.edge_src[e],d=data.edge_dst[e];
            auto feat=std::make_shared<Tensor>(1,2*C);
            for (int j=0;j<C;++j) { (*feat)(0,j)=(*x)(s,j); (*feat)(0,C+j)=(*x)(d,j); }
            auto sc=autograd::sigmoid(score_lin.forward(feat));
            (*edge_scores)(e,0)=(*sc)(0,0);
        }
        std::vector<int> eidx(E); std::iota(eidx.begin(),eidx.end(),0);
        std::sort(eidx.begin(),eidx.end(),[&](int a,int b){ return (*edge_scores)(a,0)>(*edge_scores)(b,0); });
        std::vector<int> cluster(N,-1); int next_cluster=0;
        std::vector<bool> merged(N,false);
        for (int ei:eidx) { int s=data.edge_src[ei],d=data.edge_dst[ei];
            if (!merged[s]&&!merged[d]) { cluster[s]=next_cluster; cluster[d]=next_cluster;
                merged[s]=true; merged[d]=true; next_cluster++; } }
        for (int i=0;i<N;++i) if (cluster[i]<0) cluster[i]=next_cluster++;
        int Nc=next_cluster;
        auto new_x=Tensor::zeros(Nc,C);
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*new_x)(cluster[i],j)+=(*x)(i,j);
        Data new_data; new_data.num_nodes=Nc;
        std::set<std::pair<int,int>> seen;
        for (int e=0;e<E;++e) { int ns=cluster[data.edge_src[e]],nd=cluster[data.edge_dst[e]];
            if (ns!=nd&&!seen.count({ns,nd})) { new_data.edge_src.push_back(ns); new_data.edge_dst.push_back(nd); seen.insert({ns,nd}); } }
        if (!data.batch.empty()) { new_data.batch.resize(Nc);
            for (int i=0;i<N;++i) new_data.batch[cluster[i]]=data.batch[i]; }
        return {new_x,new_data,cluster};
    }
    std::vector<TensorPtr> parameters() const { return score_lin.parameters(); }
};

// =====================================================================
//  ASAPooling -- Adaptive Structure-Aware Pooling
// =====================================================================
class ASAPooling {
public:
    int in_channels_; float ratio_;
    Linear score_lin, gate_lin;

    ASAPooling()=default;
    ASAPooling(int in_channels, float ratio=0.5f)
        :in_channels_(in_channels),ratio_(ratio),score_lin(in_channels,1,true),gate_lin(in_channels,in_channels,true){}

    struct PoolResult { TensorPtr x; Data data; std::vector<int> perm; };

    PoolResult forward(const TensorPtr& x, const Data& data) {
        int N=data.num_nodes, C=x->cols();
        auto agg=Tensor::zeros(N,C); std::vector<int> cnt(N,0);
        for (int e=0;e<data.num_edges();++e) { cnt[data.edge_dst[e]]++;
            for (int j=0;j<C;++j) (*agg)(data.edge_dst[e],j)+=(*x)(data.edge_src[e],j); }
        for (int i=0;i<N;++i) if (cnt[i]>0) for (int j=0;j<C;++j) (*agg)(i,j)/=cnt[i];
        auto gated=autograd::sigmoid(gate_lin.forward(agg));
        auto attended=autograd::mul(x,gated);
        auto scores=score_lin.forward(attended);
        for (int i=0;i<N;++i) (*scores)(i,0)=std::tanh((*scores)(i,0));
        int k=std::max(1,(int)(N*ratio_));
        std::vector<int> idx(N); std::iota(idx.begin(),idx.end(),0);
        std::partial_sort(idx.begin(),idx.begin()+k,idx.end(),[&](int a,int b){ return (*scores)(a,0)>(*scores)(b,0); });
        std::vector<int> perm(idx.begin(),idx.begin()+k);
        std::sort(perm.begin(),perm.end());
        auto new_x=std::make_shared<Tensor>(k,C);
        for (int i=0;i<k;++i) for (int j=0;j<C;++j) (*new_x)(i,j)=(*attended)(perm[i],j)*(*scores)(perm[i],0);
        std::unordered_map<int,int> old2new; for (int i=0;i<k;++i) old2new[perm[i]]=i;
        Data new_data; new_data.num_nodes=k;
        for (int e=0;e<data.num_edges();++e) {
            auto si=old2new.find(data.edge_src[e]), di=old2new.find(data.edge_dst[e]);
            if (si!=old2new.end()&&di!=old2new.end()) { new_data.edge_src.push_back(si->second); new_data.edge_dst.push_back(di->second); }
        }
        if (!data.batch.empty()) for (int i=0;i<k;++i) new_data.batch.push_back(data.batch[perm[i]]);
        return {new_x,new_data,perm};
    }
    std::vector<TensorPtr> parameters() const {
        auto p=score_lin.parameters(); auto p2=gate_lin.parameters(); p.insert(p.end(),p2.begin(),p2.end()); return p;
    }
};

// =====================================================================
//  PANPooling -- Path Integral Based pooling
// =====================================================================
class PANPooling {
public:
    int in_channels_; float ratio_;
    Linear score_lin;

    PANPooling()=default;
    PANPooling(int in_channels, float ratio=0.5f)
        :in_channels_(in_channels),ratio_(ratio),score_lin(in_channels,1,true){}

    struct PoolResult { TensorPtr x; Data data; std::vector<int> perm; };

    PoolResult forward(const TensorPtr& x, const Data& data) {
        int N=data.num_nodes, C=x->cols();
        auto deg=utils::degree(data.edge_dst,N);
        auto scores=score_lin.forward(x);
        for (int i=0;i<N;++i) (*scores)(i,0)=(*scores)(i,0)+std::log(deg[i]+1.0f);
        int k=std::max(1,(int)(N*ratio_));
        std::vector<int> idx(N); std::iota(idx.begin(),idx.end(),0);
        std::partial_sort(idx.begin(),idx.begin()+k,idx.end(),[&](int a,int b){ return (*scores)(a,0)>(*scores)(b,0); });
        std::vector<int> perm(idx.begin(),idx.begin()+k);
        std::sort(perm.begin(),perm.end());
        auto new_x=std::make_shared<Tensor>(k,C);
        for (int i=0;i<k;++i) for (int j=0;j<C;++j) (*new_x)(i,j)=(*x)(perm[i],j);
        std::unordered_map<int,int> old2new; for (int i=0;i<k;++i) old2new[perm[i]]=i;
        Data new_data; new_data.num_nodes=k;
        for (int e=0;e<data.num_edges();++e) {
            auto si=old2new.find(data.edge_src[e]), di=old2new.find(data.edge_dst[e]);
            if (si!=old2new.end()&&di!=old2new.end()) { new_data.edge_src.push_back(si->second); new_data.edge_dst.push_back(di->second); }
        }
        if (!data.batch.empty()) for (int i=0;i<k;++i) new_data.batch.push_back(data.batch[perm[i]]);
        return {new_x,new_data,perm};
    }
    std::vector<TensorPtr> parameters() const { return score_lin.parameters(); }
};

// =====================================================================
//  MemPooling -- Memory-based graph pooling
// =====================================================================
class MemPooling {
public:
    int in_channels_,out_channels_,num_heads_,num_keys_;
    TensorPtr keys; Linear lin_query;

    MemPooling()=default;
    MemPooling(int in_ch,int out_ch,int heads=1,int num_keys=10)
        :in_channels_(in_ch),out_channels_(out_ch),num_heads_(heads),num_keys_(num_keys),
         lin_query(in_ch,heads*out_ch,true) {
        keys=Tensor::randn(heads,num_keys*out_ch,0.1f); keys->set_requires_grad(true);
    }

    TensorPtr forward(const TensorPtr& x, const std::vector<int>& batch, int num_graphs) {
        int N=x->rows(),H=num_heads_,K=num_keys_,OC=out_channels_;
        auto Q=lin_query.forward(x);
        std::vector<std::vector<int>> gn(num_graphs); for (int i=0;i<N;++i) gn[batch[i]].push_back(i);
        auto out=Tensor::zeros(num_graphs,K*OC);
        for (int g=0;g<num_graphs;++g) { if (gn[g].empty()) continue;
            int gs=(int)gn[g].size();
            std::vector<float> avg_q(H*OC,0);
            for (int ni:gn[g]) for (int j=0;j<H*OC;++j) avg_q[j]+=(*Q)(ni,j)/gs;
            for (int h=0;h<H;++h) {
                std::vector<float> scores(K);
                for (int k=0;k<K;++k) { float dot=0;
                    for (int c=0;c<OC;++c) dot+=avg_q[h*OC+c]*(*keys)(h,k*OC+c);
                    scores[k]=dot/std::sqrt((float)OC); }
                float mx=-1e30f; for (auto s:scores) mx=std::max(mx,s);
                float sum=0; for (auto& s:scores) { s=std::exp(s-mx); sum+=s; }
                for (auto& s:scores) s/=(sum+1e-12f);
                for (int k=0;k<K;++k) for (int c=0;c<OC;++c) (*out)(g,k*OC+c)+=scores[k]*(*keys)(h,k*OC+c)/H;
            }
        }
        return out;
    }
    std::vector<TensorPtr> parameters() const { auto p=lin_query.parameters(); p.push_back(keys); return p; }
};

// =====================================================================
//  graclus coarsening
// =====================================================================
namespace coarsening {

inline std::vector<int> graclus(const Data& data) {
    int N=data.num_nodes;
    std::vector<int> cluster(N,-1);
    std::vector<bool> visited(N,false);
    std::vector<std::vector<int>> adj(N);
    for (int e=0;e<data.num_edges();++e) adj[data.edge_src[e]].push_back(data.edge_dst[e]);
    std::vector<int> perm(N); std::iota(perm.begin(),perm.end(),0);
    std::shuffle(perm.begin(),perm.end(),global_rng());
    int next_cluster=0;
    for (int pi=0;pi<N;++pi) {
        int i=perm[pi]; if (visited[i]) continue; visited[i]=true;
        int best=-1;
        for (int j:adj[i]) if (!visited[j]) { best=j; break; }
        cluster[i]=next_cluster;
        if (best>=0) { visited[best]=true; cluster[best]=next_cluster; }
        next_cluster++;
    }
    return cluster;
}

inline Data coarsen_graph(const Data& data, const std::vector<int>& cluster) {
    int max_c=*std::max_element(cluster.begin(),cluster.end())+1;
    Data coarsened; coarsened.num_nodes=max_c;
    if (data.x) { int C=data.x->cols(); coarsened.x=Tensor::zeros(max_c,C); std::vector<int> cnt(max_c,0);
        for (int i=0;i<data.num_nodes;++i) { cnt[cluster[i]]++;
            for (int j=0;j<C;++j) (*coarsened.x)(cluster[i],j)+=(*data.x)(i,j); }
        for (int c=0;c<max_c;++c) if (cnt[c]>0) for (int j=0;j<data.x->cols();++j) (*coarsened.x)(c,j)/=cnt[c];
    }
    std::set<std::pair<int,int>> seen;
    for (int e=0;e<data.num_edges();++e) { int ns=cluster[data.edge_src[e]],nd=cluster[data.edge_dst[e]];
        if (ns!=nd&&!seen.count({ns,nd})) { coarsened.edge_src.push_back(ns); coarsened.edge_dst.push_back(nd); seen.insert({ns,nd}); } }
    return coarsened;
}

} // namespace coarsening

// =====================================================================
//  voxel_grid
// =====================================================================
namespace voxelization {

inline std::vector<int> voxel_grid(const TensorPtr& pos, float voxel_size, const std::vector<int>& batch={}) {
    int N=pos->rows(), D=pos->cols();
    std::vector<float> mins(D,1e30f);
    for (int i=0;i<N;++i) for (int j=0;j<D;++j) mins[j]=std::min(mins[j],(*pos)(i,j));
    std::map<std::vector<int>,int> voxel_map;
    std::vector<int> cluster(N);
    for (int i=0;i<N;++i) {
        std::vector<int> key(D+1);
        for (int j=0;j<D;++j) key[j]=(int)std::floor(((*pos)(i,j)-mins[j])/voxel_size);
        key[D]=batch.empty()?0:batch[i];
        if (voxel_map.find(key)==voxel_map.end()) voxel_map[key]=(int)voxel_map.size();
        cluster[i]=voxel_map[key];
    }
    return cluster;
}

inline TensorPtr voxel_pool(const TensorPtr& x, const std::vector<int>& cluster) {
    int max_c=*std::max_element(cluster.begin(),cluster.end())+1;
    int C=x->cols();
    auto out=Tensor::zeros(max_c,C); std::vector<int> cnt(max_c,0);
    for (int i=0;i<x->rows();++i) { cnt[cluster[i]]++;
        for (int j=0;j<C;++j) (*out)(cluster[i],j)+=(*x)(i,j); }
    for (int c=0;c<max_c;++c) if (cnt[c]>0) for (int j=0;j<C;++j) (*out)(c,j)/=cnt[c];
    return out;
}

} // namespace voxelization

// =====================================================================
//  Unpool
// =====================================================================
inline TensorPtr unpool(const TensorPtr& x, const std::vector<int>& perm, int num_nodes) {
    int C=x->cols(); auto out=Tensor::zeros(num_nodes,C);
    for (int i=0;i<x->rows();++i) for (int j=0;j<C;++j) (*out)(perm[i],j)=(*x)(i,j);
    return out;
}

} // namespace pyg
