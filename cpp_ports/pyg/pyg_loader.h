// pyg_loader.h -- Data loaders for C++ PyG port
// Header-only C++17. Depends on pyg.h
//
// Implements: DataLoader (enhanced), NeighborLoader, ClusterLoader,
//             RandomNodeLoader, GraphSAINTRandomWalkSampler
#pragma once
#include "pyg.h"

namespace pyg {
namespace loader {

// =====================================================================
//  DataLoader -- enhanced version with graph-level batching
//  (The basic DataLoader is in pyg.h; this extends it.)
// =====================================================================
class GraphDataLoader {
public:
    std::vector<Data> dataset_;
    int batch_size_;
    bool shuffle_;
    bool drop_last_;

    GraphDataLoader(const std::vector<Data>& ds, int bs, bool shuffle=true, bool drop_last=false)
        :dataset_(ds),batch_size_(bs),shuffle_(shuffle),drop_last_(drop_last){}

    int num_batches() const {
        int n=(int)dataset_.size()/batch_size_;
        if (!drop_last_&&(int)dataset_.size()%batch_size_!=0) n++;
        return n;
    }

    struct Iterator {
        const GraphDataLoader* loader;
        std::vector<int> indices;
        int pos;
        Data operator*() const {
            int end=std::min(pos+loader->batch_size_,(int)indices.size());
            std::vector<Data> batch;
            for (int i=pos;i<end;++i) batch.push_back(loader->dataset_[indices[i]]);
            return batch_graphs(batch);
        }
        Iterator& operator++() { pos+=loader->batch_size_; return *this; }
        bool operator!=(const Iterator& other) const {
            if (loader->drop_last_) return pos+loader->batch_size_<=(int)indices.size();
            return pos<(int)indices.size();
        }
    };

    Iterator begin() {
        std::vector<int> idx(dataset_.size()); std::iota(idx.begin(),idx.end(),0);
        if (shuffle_) std::shuffle(idx.begin(),idx.end(),global_rng());
        return {this,idx,0};
    }
    Iterator end() { return {this,{},(int)dataset_.size()}; }
};

// =====================================================================
//  NeighborLoader -- mini-batch sampling via neighbor sampling
//  For each seed node, samples up to `num_neighbors` neighbors per hop.
// =====================================================================
class NeighborLoader {
public:
    const Data& data_;
    std::vector<int> num_neighbors_; // per hop
    int batch_size_;
    bool shuffle_;
    std::vector<std::vector<int>> adj_; // adjacency list (target -> sources)

    NeighborLoader(const Data& data, const std::vector<int>& num_neighbors, int batch_size, bool shuffle=true)
        :data_(data),num_neighbors_(num_neighbors),batch_size_(batch_size),shuffle_(shuffle) {
        adj_=data.adj_list_target_to_source();
    }

    struct SubgraphResult {
        Data subgraph;
        std::vector<int> node_ids; // mapping new_id -> original_id
        int num_seed_nodes;
    };

    SubgraphResult sample(const std::vector<int>& seed_nodes) const {
        std::unordered_set<int> sampled(seed_nodes.begin(),seed_nodes.end());
        std::vector<int> frontier=seed_nodes;

        for (int hop=0;hop<(int)num_neighbors_.size();++hop) {
            int max_nb=num_neighbors_[hop];
            std::vector<int> next_frontier;
            for (int node:frontier) {
                auto& neighbors=adj_[node];
                if (max_nb<0||(int)neighbors.size()<=max_nb) {
                    for (int nb:neighbors) {
                        if (!sampled.count(nb)) { sampled.insert(nb); next_frontier.push_back(nb); }
                    }
                } else {
                    // Random sample
                    std::vector<int> perm(neighbors.size());
                    std::iota(perm.begin(),perm.end(),0);
                    std::shuffle(perm.begin(),perm.end(),global_rng());
                    for (int i=0;i<max_nb;++i) {
                        int nb=neighbors[perm[i]];
                        if (!sampled.count(nb)) { sampled.insert(nb); next_frontier.push_back(nb); }
                    }
                }
            }
            frontier=next_frontier;
        }

        // Build subgraph
        std::vector<int> node_ids(sampled.begin(),sampled.end());
        std::sort(node_ids.begin(),node_ids.end());
        std::unordered_map<int,int> node_map;
        for (int i=0;i<(int)node_ids.size();++i) node_map[node_ids[i]]=i;

        Data sub; sub.num_nodes=(int)node_ids.size();
        for (int e=0;e<data_.num_edges();++e) {
            int s=data_.edge_src[e],d=data_.edge_dst[e];
            if (node_map.count(s)&&node_map.count(d)) {
                sub.edge_src.push_back(node_map[s]); sub.edge_dst.push_back(node_map[d]);
            }
        }
        if (data_.x) {
            int C=data_.x->cols(); sub.x=std::make_shared<Tensor>((int)node_ids.size(),C);
            for (int i=0;i<(int)node_ids.size();++i) for (int j=0;j<C;++j) (*sub.x)(i,j)=(*data_.x)(node_ids[i],j);
        }
        if (!data_.y.empty()) for (int i=0;i<(int)node_ids.size();++i) sub.y.push_back(data_.y[node_ids[i]]);
        return {sub,node_ids,(int)seed_nodes.size()};
    }

    struct Iterator {
        const NeighborLoader* loader;
        std::vector<int> node_perm;
        int pos;

        SubgraphResult operator*() const {
            int end=std::min(pos+loader->batch_size_,(int)node_perm.size());
            std::vector<int> seeds(node_perm.begin()+pos,node_perm.begin()+end);
            return loader->sample(seeds);
        }
        Iterator& operator++() { pos+=loader->batch_size_; return *this; }
        bool operator!=(const Iterator&) const { return pos<(int)node_perm.size(); }
    };

    Iterator begin() {
        std::vector<int> perm(data_.num_nodes); std::iota(perm.begin(),perm.end(),0);
        if (shuffle_) std::shuffle(perm.begin(),perm.end(),global_rng());
        return {this,perm,0};
    }
    Iterator end() { return {this,{},data_.num_nodes}; }
};

// =====================================================================
//  ClusterLoader -- METIS-style graph clustering for mini-batching
//  "Cluster-GCN: An Efficient Algorithm for Training Deep and Large
//   Graph Convolutional Networks" (Chiang et al.)
// =====================================================================
class ClusterLoader {
public:
    const Data& data_;
    int num_parts_;
    bool shuffle_;
    std::vector<std::vector<int>> partitions_;

    ClusterLoader(const Data& data, int num_parts, bool shuffle=true)
        :data_(data),num_parts_(num_parts),shuffle_(shuffle) {
        partition();
    }

    void partition() {
        // Simple balanced random partitioning (approximation of METIS)
        int N=data_.num_nodes;
        std::vector<int> perm(N); std::iota(perm.begin(),perm.end(),0);
        std::shuffle(perm.begin(),perm.end(),global_rng());
        partitions_.resize(num_parts_);
        for (int i=0;i<N;++i) partitions_[i%num_parts_].push_back(perm[i]);
        // Refine: try to improve cut by local swaps (simplified)
        auto adj_list=data_.adj_list_source_to_target();
        std::vector<int> part_of(N);
        for (int p=0;p<num_parts_;++p) for (int n:partitions_[p]) part_of[n]=p;
        // One pass of refinement
        for (int i=0;i<N;++i) {
            std::vector<int> counts(num_parts_,0);
            for (int nb:adj_list[i]) counts[part_of[nb]]++;
            int best_part=part_of[i]; int best_count=counts[best_part];
            for (int p=0;p<num_parts_;++p) {
                if (counts[p]>best_count&&(int)partitions_[p].size()<2*N/num_parts_) {
                    best_count=counts[p]; best_part=p;
                }
            }
            if (best_part!=part_of[i]) {
                // Move node
                auto& old_part=partitions_[part_of[i]];
                old_part.erase(std::find(old_part.begin(),old_part.end(),i));
                partitions_[best_part].push_back(i);
                part_of[i]=best_part;
            }
        }
    }

    Data get_partition(int idx) const {
        return utils::subgraph(data_,partitions_[idx]);
    }

    struct Iterator {
        const ClusterLoader* loader;
        std::vector<int> order;
        int pos;
        Data operator*() const { return loader->get_partition(order[pos]); }
        Iterator& operator++() { ++pos; return *this; }
        bool operator!=(const Iterator&) const { return pos<(int)order.size(); }
    };

    Iterator begin() {
        std::vector<int> order(num_parts_); std::iota(order.begin(),order.end(),0);
        if (shuffle_) std::shuffle(order.begin(),order.end(),global_rng());
        return {this,order,0};
    }
    Iterator end() { return {this,{},num_parts_}; }
};

// =====================================================================
//  RandomNodeLoader -- randomly samples nodes for mini-batching
// =====================================================================
class RandomNodeLoader {
public:
    const Data& data_;
    int batch_size_;
    bool shuffle_;

    RandomNodeLoader(const Data& data, int batch_size, bool shuffle=true)
        :data_(data),batch_size_(batch_size),shuffle_(shuffle){}

    struct Iterator {
        const RandomNodeLoader* loader;
        std::vector<int> node_perm;
        int pos;

        Data operator*() const {
            int end=std::min(pos+loader->batch_size_,(int)node_perm.size());
            std::vector<int> nodes(node_perm.begin()+pos,node_perm.begin()+end);
            auto sub=utils::subgraph(loader->data_,nodes);
            // Set masks for this batch
            sub.train_mask.assign(sub.num_nodes,true);
            return sub;
        }
        Iterator& operator++() { pos+=loader->batch_size_; return *this; }
        bool operator!=(const Iterator&) const { return pos<(int)node_perm.size(); }
    };

    Iterator begin() {
        std::vector<int> perm(data_.num_nodes); std::iota(perm.begin(),perm.end(),0);
        if (shuffle_) std::shuffle(perm.begin(),perm.end(),global_rng());
        return {this,perm,0};
    }
    Iterator end() { return {this,{},data_.num_nodes}; }
};

// =====================================================================
//  GraphSAINTRandomWalkSampler -- GraphSAINT random walk based sampler
// =====================================================================
class GraphSAINTRandomWalkSampler {
public:
    const Data& data_;
    int batch_size_,walk_length_;

    GraphSAINTRandomWalkSampler(const Data& data, int batch_size=200, int walk_length=4)
        :data_(data),batch_size_(batch_size),walk_length_(walk_length){}

    Data sample() const {
        int N=data_.num_nodes;
        auto adj=data_.adj_list_source_to_target();
        std::unordered_set<int> sampled;
        std::uniform_int_distribution<int> dist(0,N-1);
        // Start random walks from random nodes
        for (int w=0;w<batch_size_;++w) {
            int cur=dist(global_rng()); sampled.insert(cur);
            for (int step=0;step<walk_length_;++step) {
                if (adj[cur].empty()) break;
                cur=adj[cur][global_rng()()%adj[cur].size()];
                sampled.insert(cur);
            }
        }
        std::vector<int> nodes(sampled.begin(),sampled.end());
        return utils::subgraph(data_,nodes);
    }
};

// =====================================================================
//  LinkNeighborLoader -- for link prediction mini-batches
// =====================================================================
class LinkNeighborLoader {
public:
    const Data& data_;
    int batch_size_;
    std::vector<int> num_neighbors_;
    bool shuffle_;

    LinkNeighborLoader(const Data& data, const std::vector<int>& nn, int bs, bool shuffle=true)
        :data_(data),batch_size_(bs),num_neighbors_(nn),shuffle_(shuffle){}

    struct Batch {
        Data subgraph;
        std::vector<int> edge_src, edge_dst; // positive edges in this batch
        std::vector<int> neg_edge_src, neg_edge_dst; // negative edges
    };

    Batch sample(int start, int end) const {
        int E=data_.num_edges();
        std::vector<int> batch_src,batch_dst;
        for (int e=start;e<end&&e<E;++e) { batch_src.push_back(data_.edge_src[e]); batch_dst.push_back(data_.edge_dst[e]); }
        // Collect unique nodes
        std::unordered_set<int> seed_set;
        for (int s:batch_src) seed_set.insert(s);
        for (int d:batch_dst) seed_set.insert(d);
        std::vector<int> seed_nodes(seed_set.begin(),seed_set.end());
        // Sample neighbors
        auto adj=data_.adj_list_target_to_source();
        std::unordered_set<int> sampled(seed_nodes.begin(),seed_nodes.end());
        auto frontier=seed_nodes;
        for (int hop=0;hop<(int)num_neighbors_.size();++hop) {
            int max_nb=num_neighbors_[hop];
            std::vector<int> next;
            for (int node:frontier) { auto& nbs=adj[node];
                int lim=max_nb<0?(int)nbs.size():std::min(max_nb,(int)nbs.size());
                for (int i=0;i<lim;++i) { if (!sampled.count(nbs[i])) { sampled.insert(nbs[i]); next.push_back(nbs[i]); } } }
            frontier=next;
        }
        std::vector<int> all_nodes(sampled.begin(),sampled.end());
        auto sub=utils::subgraph(data_,all_nodes);
        // Negative sampling
        auto [neg_src,neg_dst]=utils::negative_sampling(batch_src,batch_dst,data_.num_nodes,(int)batch_src.size());
        return {sub,batch_src,batch_dst,neg_src,neg_dst};
    }

    struct Iterator {
        const LinkNeighborLoader* loader;
        std::vector<int> edge_perm;
        int pos;
        Batch operator*() const {
            int end=std::min(pos+loader->batch_size_,(int)edge_perm.size());
            return loader->sample(pos,end);
        }
        Iterator& operator++() { pos+=loader->batch_size_; return *this; }
        bool operator!=(const Iterator&) const { return pos<(int)edge_perm.size(); }
    };

    Iterator begin() {
        int E=data_.num_edges();
        std::vector<int> perm(E); std::iota(perm.begin(),perm.end(),0);
        if (shuffle_) std::shuffle(perm.begin(),perm.end(),global_rng());
        return {this,perm,0};
    }
    Iterator end() { return {this,{},data_.num_edges()}; }
};

} // namespace loader
} // namespace pyg
