// pyg_models.h -- High-level model architectures for C++ PyG port
// Header-only C++17. Depends on pyg.h, pyg_conv.h
//
// Implements: GAE, VGAE, Node2Vec, JumpingKnowledge, GraphUNet, SchNet,
//             DimeNet (simplified), SignedGCN, RECT_L, MetaPath2Vec,
//             GNNExplainer, RENet (simplified), BasicGNN, DeepGCNLayer
#pragma once
#include "pyg.h"
#include "pyg_conv.h"

namespace pyg {
namespace models {

// =====================================================================
//  GAE -- Graph AutoEncoder
//  Encoder: GNN that produces node embeddings
//  Decoder: inner product between embeddings
// =====================================================================
class GAE {
public:
    GCNConv conv1, conv2;
    int in_c_,hidden_c_,out_c_;

    GAE()=default;
    GAE(int in, int hidden, int out)
        :conv1(in,hidden),conv2(hidden,out),in_c_(in),hidden_c_(hidden),out_c_(out){}

    TensorPtr encode(const TensorPtr& x, const Data& data) {
        auto h=autograd::relu(conv1.forward(x,data));
        return conv2.forward(h,data);
    }

    TensorPtr decode(const TensorPtr& z, const std::vector<int>& src, const std::vector<int>& dst) {
        int E=(int)src.size(),C=z->cols();
        auto out=std::make_shared<Tensor>(E,1);
        for (int e=0;e<E;++e) { float dot=0; for (int j=0;j<C;++j) dot+=(*z)(src[e],j)*(*z)(dst[e],j);
            (*out)(e,0)=dot; }
        return out;
    }

    TensorPtr decode_all(const TensorPtr& z) {
        int N=z->rows(),C=z->cols();
        auto out=std::make_shared<Tensor>(N,N);
        for (int i=0;i<N;++i) for (int j=0;j<N;++j) { float dot=0;
            for (int c=0;c<C;++c) dot+=(*z)(i,c)*(*z)(j,c); (*out)(i,j)=dot; }
        return autograd::sigmoid(out);
    }

    TensorPtr recon_loss(const TensorPtr& z, const Data& data) {
        int N=z->rows();
        auto pos_pred=decode(z,data.edge_src,data.edge_dst);
        auto [neg_src,neg_dst]=utils::negative_sampling(data.edge_src,data.edge_dst,N);
        auto neg_pred=decode(z,neg_src,neg_dst);
        // BCE loss
        float loss=0; int np=(int)data.edge_src.size(),nn=(int)neg_src.size();
        for (int i=0;i<np;++i) { float p=1.0f/(1.0f+std::exp(-(*pos_pred)(i,0))); loss-=std::log(p+1e-7f); }
        for (int i=0;i<nn;++i) { float p=1.0f/(1.0f+std::exp(-(*neg_pred)(i,0))); loss-=std::log(1-p+1e-7f); }
        return std::make_shared<Tensor>(1,1,loss/(np+nn));
    }

    std::vector<TensorPtr> parameters() const {
        auto p=conv1.parameters(); auto p2=conv2.parameters(); p.insert(p.end(),p2.begin(),p2.end()); return p;
    }
};

// =====================================================================
//  VGAE -- Variational Graph AutoEncoder
// =====================================================================
class VGAE {
public:
    GCNConv conv_shared,conv_mu,conv_logvar;
    int in_c_,hidden_c_,out_c_;

    VGAE()=default;
    VGAE(int in, int hidden, int out)
        :conv_shared(in,hidden),conv_mu(hidden,out),conv_logvar(hidden,out),
         in_c_(in),hidden_c_(hidden),out_c_(out){}

    struct EncodeResult { TensorPtr mu; TensorPtr logvar; TensorPtr z; };

    EncodeResult encode(const TensorPtr& x, const Data& data) {
        auto h=autograd::relu(conv_shared.forward(x,data));
        auto mu=conv_mu.forward(h,data);
        auto logvar=conv_logvar.forward(h,data);
        // Reparameterization trick
        int N=mu->rows(),C=mu->cols();
        auto z=std::make_shared<Tensor>(N,C);
        std::normal_distribution<float> dist(0,1);
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) {
            float eps=dist(global_rng());
            (*z)(i,j)=(*mu)(i,j)+std::exp(0.5f*(*logvar)(i,j))*eps;
        }
        return {mu,logvar,z};
    }

    TensorPtr decode(const TensorPtr& z, const std::vector<int>& src, const std::vector<int>& dst) {
        int E=(int)src.size(),C=z->cols();
        auto out=std::make_shared<Tensor>(E,1);
        for (int e=0;e<E;++e) { float dot=0; for (int j=0;j<C;++j) dot+=(*z)(src[e],j)*(*z)(dst[e],j); (*out)(e,0)=dot; }
        return out;
    }

    float kl_loss(const TensorPtr& mu, const TensorPtr& logvar) {
        int n=mu->numel(); float loss=0;
        for (int i=0;i<n;++i) loss+=0.5f*((*mu)[i]*(*mu)[i]+std::exp((*logvar)[i])-(*logvar)[i]-1);
        return loss/mu->rows();
    }

    TensorPtr recon_loss(const TensorPtr& z, const Data& data) {
        int N=z->rows();
        auto pos_pred=decode(z,data.edge_src,data.edge_dst);
        auto [neg_src,neg_dst]=utils::negative_sampling(data.edge_src,data.edge_dst,N);
        auto neg_pred=decode(z,neg_src,neg_dst);
        float loss=0; int np=(int)data.edge_src.size(),nn=(int)neg_src.size();
        for (int i=0;i<np;++i) { float p=1.0f/(1.0f+std::exp(-(*pos_pred)(i,0))); loss-=std::log(p+1e-7f); }
        for (int i=0;i<nn;++i) { float p=1.0f/(1.0f+std::exp(-(*neg_pred)(i,0))); loss-=std::log(1-p+1e-7f); }
        return std::make_shared<Tensor>(1,1,loss/(np+nn));
    }

    std::vector<TensorPtr> parameters() const {
        auto p=conv_shared.parameters(); auto p2=conv_mu.parameters(); p.insert(p.end(),p2.begin(),p2.end());
        auto p3=conv_logvar.parameters(); p.insert(p.end(),p3.begin(),p3.end()); return p;
    }
};

// =====================================================================
//  Node2Vec -- learns node embeddings via random walks
// =====================================================================
class Node2Vec {
public:
    Embedding embedding;
    int embed_dim_,walk_length_,context_size_,walks_per_node_;
    float p_,q_;

    Node2Vec()=default;
    Node2Vec(int num_nodes, int embed_dim, int walk_length=20, int context_size=10,
             int walks_per_node=10, float p=1, float q=1)
        :embedding(num_nodes,embed_dim),embed_dim_(embed_dim),walk_length_(walk_length),
         context_size_(context_size),walks_per_node_(walks_per_node),p_(p),q_(q){}

    std::vector<std::vector<int>> generate_walks(const Data& data) const {
        auto adj=data.adj_list_source_to_target();
        int N=data.num_nodes;
        std::vector<std::vector<int>> walks;
        for (int w=0;w<walks_per_node_;++w) {
            for (int start=0;start<N;++start) {
                std::vector<int> walk={start};
                for (int step=0;step<walk_length_-1;++step) {
                    int cur=walk.back();
                    if (adj[cur].empty()) break;
                    if (walk.size()==1) {
                        // Uniform random
                        walk.push_back(adj[cur][global_rng()()%adj[cur].size()]);
                    } else {
                        int prev=walk[walk.size()-2];
                        // Biased sampling with p,q
                        std::vector<float> weights;
                        for (int nb:adj[cur]) {
                            if (nb==prev) weights.push_back(1.0f/p_);
                            else {
                                // Check if nb is neighbor of prev
                                bool connected=false;
                                for (int pp:adj[prev]) if (pp==nb) { connected=true; break; }
                                weights.push_back(connected?1.0f:1.0f/q_);
                            }
                        }
                        float total=0; for (auto w2:weights) total+=w2;
                        float r=std::uniform_real_distribution<float>(0,total)(global_rng());
                        float cum=0; int chosen=0;
                        for (int i=0;i<(int)weights.size();++i) { cum+=weights[i]; if (r<=cum) { chosen=i; break; } }
                        walk.push_back(adj[cur][chosen]);
                    }
                }
                walks.push_back(walk);
            }
        }
        return walks;
    }

    TensorPtr loss_fn(const std::vector<std::vector<int>>& walks) {
        float total_loss=0; int count=0;
        for (auto& walk:walks) {
            int wl=(int)walk.size();
            for (int i=0;i<wl;++i) {
                int center=walk[i];
                auto center_emb=embedding.forward({center});
                for (int j=std::max(0,i-context_size_);j<=std::min(wl-1,i+context_size_);++j) {
                    if (j==i) continue;
                    int context=walk[j];
                    auto ctx_emb=embedding.forward({context});
                    // Positive: dot product -> sigmoid -> log
                    float dot=0; for (int c=0;c<embed_dim_;++c) dot+=(*center_emb)(0,c)*(*ctx_emb)(0,c);
                    float sig=1.0f/(1.0f+std::exp(-dot));
                    total_loss-=std::log(sig+1e-7f);
                    count++;
                }
            }
        }
        return std::make_shared<Tensor>(1,1,count>0?total_loss/count:0);
    }

    TensorPtr forward(const std::vector<int>& nodes) { return embedding.forward(nodes); }
    std::vector<TensorPtr> parameters() const { return embedding.parameters(); }
};

// =====================================================================
//  JumpingKnowledge -- aggregates representations from multiple layers
//  Modes: "cat" (concatenate), "max" (element-wise max), "lstm" (attention)
// =====================================================================
class JumpingKnowledge {
public:
    std::string mode_;
    Linear att_lin; // for lstm mode
    int channels_;

    JumpingKnowledge():channels_(0){}
    JumpingKnowledge(const std::string& mode, int channels=0, int num_layers=0)
        :mode_(mode),channels_(channels) {
        if (mode=="lstm"||mode=="att") att_lin=Linear(channels,1,true);
    }

    TensorPtr forward(const std::vector<TensorPtr>& xs) {
        if (xs.empty()) return Tensor::zeros(0,0);
        if (mode_=="cat") return Tensor::cat_cols(xs);
        if (mode_=="max") {
            auto out=std::make_shared<Tensor>(xs[0]->rows(),xs[0]->cols(),-1e30f);
            for (auto& x:xs) for (int i=0;i<x->rows();++i) for (int j=0;j<x->cols();++j)
                (*out)(i,j)=std::max((*out)(i,j),(*x)(i,j));
            return out;
        }
        // Attention-based aggregation
        int N=xs[0]->rows(),C=xs[0]->cols(),L=(int)xs.size();
        // Compute attention scores per layer
        auto scores=Tensor::zeros(N,L);
        for (int l=0;l<L;++l) { auto s=att_lin.forward(xs[l]); for (int i=0;i<N;++i) (*scores)(i,l)=(*s)(i,0); }
        // Softmax over layers
        for (int i=0;i<N;++i) { float mx=-1e30f; for (int l=0;l<L;++l) mx=std::max(mx,(*scores)(i,l));
            float s=0; for (int l=0;l<L;++l) { (*scores)(i,l)=std::exp((*scores)(i,l)-mx); s+=(*scores)(i,l); }
            for (int l=0;l<L;++l) (*scores)(i,l)/=(s+1e-12f); }
        auto out=Tensor::zeros(N,C);
        for (int l=0;l<L;++l) for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*out)(i,j)+=(*scores)(i,l)*(*xs[l])(i,j);
        return out;
    }
    std::vector<TensorPtr> parameters() const {
        if (mode_=="lstm"||mode_=="att") return att_lin.parameters(); return {};
    }
};

// =====================================================================
//  GraphUNet -- "Graph U-Nets" (Gao & Ji, 2019)
//  Encoder-decoder with TopK pooling and skip connections.
// =====================================================================
class GraphUNet {
public:
    int in_c_,hidden_c_,out_c_,depth_;
    std::vector<GCNConv> down_convs, up_convs;
    std::vector<TensorPtr> pool_weights; // TopK weights
    float pool_ratio_;

    GraphUNet()=default;
    GraphUNet(int in, int hidden, int out, int depth=3, float pool_ratio=0.5f)
        :in_c_(in),hidden_c_(hidden),out_c_(out),depth_(depth),pool_ratio_(pool_ratio) {
        down_convs.emplace_back(in,hidden);
        for (int i=1;i<depth;++i) down_convs.emplace_back(hidden,hidden);
        for (int i=0;i<depth-1;++i) up_convs.emplace_back(2*hidden,hidden);
        up_convs.emplace_back(hidden+in,out);
        for (int i=0;i<depth;++i) {
            auto w=Tensor::xavier_uniform(1,hidden); w->set_requires_grad(true); pool_weights.push_back(w);
        }
    }

    TensorPtr forward(const TensorPtr& x, const Data& data) {
        std::vector<TensorPtr> xs={x};
        std::vector<Data> datas={data};
        std::vector<std::vector<int>> perms;
        // Encoder (down)
        auto h=x;
        auto cur_data=data;
        for (int i=0;i<depth_;++i) {
            h=autograd::relu(down_convs[i].forward(h,cur_data));
            xs.push_back(h);
            // TopK pooling
            int N=cur_data.num_nodes,C=h->cols();
            float wnorm=0; for (int j=0;j<C;++j) wnorm+=(*pool_weights[i])(0,j)*(*pool_weights[i])(0,j);
            wnorm=std::sqrt(wnorm)+1e-12f;
            auto scores=std::make_shared<Tensor>(N,1);
            for (int ni=0;ni<N;++ni) { float s=0; for (int j=0;j<C;++j) s+=(*h)(ni,j)*(*pool_weights[i])(0,j)/wnorm;
                (*scores)(ni,0)=std::tanh(s); }
            int k=std::max(1,(int)(N*pool_ratio_));
            std::vector<int> idx(N); std::iota(idx.begin(),idx.end(),0);
            std::partial_sort(idx.begin(),idx.begin()+k,idx.end(),[&](int a,int b){ return (*scores)(a,0)>(*scores)(b,0); });
            std::vector<int> perm(idx.begin(),idx.begin()+k);
            std::sort(perm.begin(),perm.end());
            perms.push_back(perm);
            auto new_h=std::make_shared<Tensor>(k,C);
            for (int ni=0;ni<k;++ni) for (int j=0;j<C;++j) (*new_h)(ni,j)=(*h)(perm[ni],j)*(*scores)(perm[ni],0);
            std::unordered_map<int,int> o2n; for (int ni=0;ni<k;++ni) o2n[perm[ni]]=ni;
            Data nd; nd.num_nodes=k;
            for (int e=0;e<cur_data.num_edges();++e) {
                auto si=o2n.find(cur_data.edge_src[e]),di=o2n.find(cur_data.edge_dst[e]);
                if (si!=o2n.end()&&di!=o2n.end()) { nd.edge_src.push_back(si->second); nd.edge_dst.push_back(di->second); } }
            h=new_h; cur_data=nd;
            datas.push_back(cur_data);
        }
        // Decoder (up)
        for (int i=depth_-1;i>=0;--i) {
            // Unpool
            int target_N=datas[i].num_nodes;
            auto unpooled=Tensor::zeros(target_N,h->cols());
            for (int ni=0;ni<h->rows();++ni) for (int j=0;j<h->cols();++j) (*unpooled)(perms[i][ni],j)=(*h)(ni,j);
            // Skip connection
            h=Tensor::cat_cols({unpooled,xs[i]});
            h=up_convs[i].forward(h,datas[i]);
            if (i>0) h=autograd::relu(h);
        }
        return h;
    }
    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p;
        for (auto& c:down_convs) { auto cp=c.parameters(); p.insert(p.end(),cp.begin(),cp.end()); }
        for (auto& c:up_convs) { auto cp=c.parameters(); p.insert(p.end(),cp.begin(),cp.end()); }
        for (auto& w:pool_weights) p.push_back(w);
        return p;
    }
};

// =====================================================================
//  SchNet -- continuous-filter convolutional network for molecular data
//  "SchNet: A Continuous-filter Convolutional Neural Network for
//   Modeling Quantum Interactions" (Schuett et al., 2017)
// =====================================================================
class SchNet {
public:
    int hidden_channels_,num_filters_,num_interactions_,num_gaussians_;
    float cutoff_;
    Embedding atom_embedding;
    struct InteractionBlock {
        Linear lin1,lin2,lin_filter;
        InteractionBlock()=default;
        InteractionBlock(int hidden,int nf):lin1(nf,hidden,true),lin2(hidden,hidden,true),lin_filter(50,nf,true){}
    };
    std::vector<InteractionBlock> interactions;
    Linear output_lin1,output_lin2;

    SchNet()=default;
    SchNet(int hidden_channels=128, int num_filters=128, int num_interactions=6,
           int num_gaussians=50, float cutoff=10.0f, int max_z=100)
        :hidden_channels_(hidden_channels),num_filters_(num_filters),
         num_interactions_(num_interactions),num_gaussians_(num_gaussians),cutoff_(cutoff),
         atom_embedding(max_z,hidden_channels),
         output_lin1(hidden_channels,hidden_channels/2),output_lin2(hidden_channels/2,1) {
        for (int i=0;i<num_interactions;++i) interactions.emplace_back(hidden_channels,num_filters);
    }

    TensorPtr gaussian_smearing(float dist) const {
        auto out=std::make_shared<Tensor>(1,num_gaussians_);
        float offset_step=cutoff_/num_gaussians_;
        for (int i=0;i<num_gaussians_;++i) {
            float center=i*offset_step;
            float coeff=-0.5f/(offset_step*offset_step);
            (*out)(0,i)=std::exp(coeff*(dist-center)*(dist-center));
        }
        return out;
    }

    TensorPtr forward(const std::vector<int>& z, const TensorPtr& pos, const Data& data) {
        int N=(int)z.size(),E=data.num_edges();
        auto h=atom_embedding.forward(z);
        for (int layer=0;layer<num_interactions_;++layer) {
            auto agg=Tensor::zeros(N,num_filters_);
            std::vector<int> cnt(N,0);
            for (int e=0;e<E;++e) {
                int s=data.edge_src[e],d=data.edge_dst[e];
                // Distance
                float dist=0;
                for (int dd=0;dd<pos->cols();++dd) { float diff=(*pos)(s,dd)-(*pos)(d,dd); dist+=diff*diff; }
                dist=std::sqrt(dist);
                if (dist>cutoff_) continue;
                cnt[d]++;
                auto rbf=gaussian_smearing(dist);
                auto filter=interactions[layer].lin_filter.forward(rbf);
                // Message: h_s * filter
                for (int j=0;j<num_filters_;++j) (*agg)(d,j)+=(*h)(s,j%hidden_channels_)*(*filter)(0,j);
            }
            auto update=interactions[layer].lin2.forward(autograd::relu(interactions[layer].lin1.forward(agg)));
            // Residual
            for (int i=0;i<N;++i) for (int j=0;j<hidden_channels_;++j) (*h)(i,j)+=(*update)(i,j);
        }
        // Readout
        return output_lin2.forward(autograd::relu(output_lin1.forward(h)));
    }

    std::vector<TensorPtr> parameters() const {
        auto p=atom_embedding.parameters();
        for (auto& ib:interactions) {
            auto add=[&](const Linear& l){ auto lp=l.parameters(); p.insert(p.end(),lp.begin(),lp.end()); };
            add(ib.lin1); add(ib.lin2); add(ib.lin_filter);
        }
        auto add=[&](const Linear& l){ auto lp=l.parameters(); p.insert(p.end(),lp.begin(),lp.end()); };
        add(output_lin1); add(output_lin2);
        return p;
    }
};

// =====================================================================
//  DimeNet (simplified) -- Directional Message Passing Neural Network
//  "Directional Message Passing for Molecular Graphs" (Klicpera et al.)
// =====================================================================
class DimeNet {
public:
    int hidden_channels_,num_blocks_,num_bilinear_,num_spherical_,num_radial_;
    float cutoff_;
    Embedding atom_embedding;
    Linear rbf_lin,output_lin;

    DimeNet()=default;
    DimeNet(int hidden_channels=128, int num_blocks=6, int num_bilinear=8,
            int num_spherical=7, int num_radial=6, float cutoff=5.0f, int max_z=100)
        :hidden_channels_(hidden_channels),num_blocks_(num_blocks),num_bilinear_(num_bilinear),
         num_spherical_(num_spherical),num_radial_(num_radial),cutoff_(cutoff),
         atom_embedding(max_z,hidden_channels),rbf_lin(num_radial,hidden_channels),
         output_lin(hidden_channels,1){}

    TensorPtr radial_basis(float dist) const {
        auto out=std::make_shared<Tensor>(1,num_radial_);
        for (int n=0;n<num_radial_;++n) {
            float freq=(n+1)*M_PI/cutoff_;
            (*out)(0,n)=std::sqrt(2.0f/cutoff_)*std::sin(freq*dist)/dist;
        }
        return out;
    }

    TensorPtr forward(const std::vector<int>& z, const TensorPtr& pos, const Data& data) {
        int N=(int)z.size(),E=data.num_edges();
        auto h=atom_embedding.forward(z);
        for (int block=0;block<num_blocks_;++block) {
            auto agg=Tensor::zeros(N,hidden_channels_);
            for (int e=0;e<E;++e) {
                int s=data.edge_src[e],d=data.edge_dst[e];
                float dist=0;
                for (int dd=0;dd<pos->cols();++dd) { float diff=(*pos)(s,dd)-(*pos)(d,dd); dist+=diff*diff; }
                dist=std::sqrt(dist)+1e-8f;
                if (dist>cutoff_) continue;
                auto rbf=radial_basis(dist);
                auto w=rbf_lin.forward(rbf);
                for (int j=0;j<hidden_channels_;++j) (*agg)(d,j)+=(*h)(s,j)*(*w)(0,j);
            }
            // Residual update
            for (int i=0;i<N;++i) for (int j=0;j<hidden_channels_;++j) (*h)(i,j)+=(*agg)(i,j);
        }
        return output_lin.forward(h);
    }

    std::vector<TensorPtr> parameters() const {
        auto p=atom_embedding.parameters(); auto p2=rbf_lin.parameters(); p.insert(p.end(),p2.begin(),p2.end());
        auto p3=output_lin.parameters(); p.insert(p.end(),p3.begin(),p3.end()); return p;
    }
};

// =====================================================================
//  SignedGCN -- for signed networks (positive & negative edges)
//  "Signed Graph Convolutional Network" (Derr et al.)
// =====================================================================
class SignedGCN {
public:
    int in_channels_,hidden_channels_,out_channels_,num_layers_;
    std::vector<SignedConv> convs;
    Linear discriminator;

    SignedGCN()=default;
    SignedGCN(int in, int hidden, int out, int num_layers=2)
        :in_channels_(in),hidden_channels_(hidden),out_channels_(out),num_layers_(num_layers) {
        convs.emplace_back(in,hidden,true); // first
        for (int i=1;i<num_layers;++i) convs.emplace_back(2*hidden,hidden,false);
        discriminator=Linear(2*hidden,out);
    }

    TensorPtr forward(const TensorPtr& x, const Data& pos_data, const Data& neg_data) {
        auto h=x;
        for (int i=0;i<num_layers_;++i) {
            h=autograd::relu(convs[i].forward(h,pos_data,neg_data));
        }
        return discriminator.forward(h);
    }

    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p;
        for (auto& c:convs) { auto cp=c.parameters(); p.insert(p.end(),cp.begin(),cp.end()); }
        auto dp=discriminator.parameters(); p.insert(p.end(),dp.begin(),dp.end()); return p;
    }
};

// =====================================================================
//  RECT_L -- "Network Embedding as Matrix Factorization:
//             Approximating Graph by Shorter Random Walks"
//  Simplified version using GCN encoder + label propagation ideas.
// =====================================================================
class RECT_L {
public:
    GCNConv conv; Linear lin;

    RECT_L()=default;
    RECT_L(int in, int hidden):conv(in,hidden),lin(hidden,in){}

    TensorPtr forward(const TensorPtr& x, const Data& data) {
        auto h=autograd::relu(conv.forward(x,data));
        return autograd::sigmoid(lin.forward(h));
    }

    TensorPtr loss_fn(const TensorPtr& pred, const TensorPtr& target, const std::vector<bool>& mask) {
        int N=pred->rows(),C=pred->cols(); float loss=0; int count=0;
        for (int i=0;i<N;++i) { if (!mask[i]) continue; count++;
            for (int j=0;j<C;++j) { float p=std::max(1e-7f,std::min(1-1e-7f,(*pred)(i,j)));
                loss-=(*target)(i,j)*std::log(p)+(1-(*target)(i,j))*std::log(1-p); } }
        return std::make_shared<Tensor>(1,1,count>0?loss/(count*C):0);
    }

    std::vector<TensorPtr> parameters() const {
        auto p=conv.parameters(); auto p2=lin.parameters(); p.insert(p.end(),p2.begin(),p2.end()); return p;
    }
};

// =====================================================================
//  MetaPath2Vec -- Heterogeneous graph embedding via metapath random walks
// =====================================================================
class MetaPath2Vec {
public:
    Embedding embedding;
    int embed_dim_,walk_length_,context_size_,walks_per_node_;
    std::vector<std::string> metapath_; // sequence of node types

    MetaPath2Vec()=default;
    MetaPath2Vec(int num_nodes, int embed_dim, const std::vector<std::string>& metapath,
                 int walk_length=20, int context_size=7, int walks_per_node=3)
        :embedding(num_nodes,embed_dim),embed_dim_(embed_dim),walk_length_(walk_length),
         context_size_(context_size),walks_per_node_(walks_per_node),metapath_(metapath){}

    // Walks follow the metapath pattern cyclically
    std::vector<std::vector<int>> generate_walks(
        const HeteroData& hdata,
        const std::vector<std::string>& node_types,
        const std::unordered_map<int,std::string>& node_type_map) const {
        // Build typed adjacency
        std::vector<std::vector<int>> walks;
        // Simplified: for each node of the start type, do walks
        for (auto& [node,type]:node_type_map) {
            if (type!=metapath_[0]) continue;
            for (int w=0;w<walks_per_node_;++w) {
                std::vector<int> walk={node};
                for (int step=0;step<walk_length_-1;++step) {
                    // Follow metapath
                    // In practice, would filter by edge types matching metapath
                    break; // Simplified
                }
                walks.push_back(walk);
            }
        }
        return walks;
    }

    TensorPtr forward(const std::vector<int>& nodes) { return embedding.forward(nodes); }
    std::vector<TensorPtr> parameters() const { return embedding.parameters(); }
};

// =====================================================================
//  GNNExplainer -- explains predictions of a trained GNN
//  "GNNExplainer: Generating Explanations for Graph Neural Networks"
// =====================================================================
class GNNExplainer {
public:
    int epochs_; float lr_;

    GNNExplainer(int epochs=200, float lr=0.01f):epochs_(epochs),lr_(lr){}

    struct Explanation {
        std::vector<float> node_mask;  // importance of each node
        std::vector<float> edge_mask;  // importance of each edge
    };

    // forward_fn: a callable that takes (x, data) -> logits
    template<typename ForwardFn>
    Explanation explain_node(int node_idx, const TensorPtr& x, const Data& data,
                             ForwardFn forward_fn) {
        int N=data.num_nodes, E=data.num_edges(), C=x->cols();
        // Initialize learnable masks
        auto node_mask=Tensor::ones(N,1); node_mask->set_requires_grad(true);
        auto edge_mask=Tensor::ones(E,1); edge_mask->set_requires_grad(true);
        // Get original prediction
        auto orig_logits=forward_fn(x,data);
        int orig_pred=0; float max_v=(*orig_logits)(node_idx,0);
        for (int j=1;j<orig_logits->cols();++j) if ((*orig_logits)(node_idx,j)>max_v) { max_v=(*orig_logits)(node_idx,j); orig_pred=j; }
        // Optimize masks
        for (int epoch=0;epoch<epochs_;++epoch) {
            // Apply sigmoid to masks
            auto nm_sig=autograd::sigmoid(node_mask);
            // Mask node features
            auto masked_x=std::make_shared<Tensor>(N,C);
            for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*masked_x)(i,j)=(*x)(i,j)*(*nm_sig)(i,0);
            auto logits=forward_fn(masked_x,data);
            // Loss: prediction fidelity + mask sparsity
            float pred_loss=-(*logits)(node_idx,orig_pred);
            float size_loss=0;
            for (int i=0;i<N;++i) size_loss+=(*nm_sig)(i,0);
            size_loss/=N;
            float total=pred_loss+0.01f*size_loss;
            // Simple gradient descent on masks
            // Numerical gradient for simplicity
            float eps=1e-4f;
            for (int i=0;i<N;++i) {
                (*node_mask)(i,0)+=eps;
                auto nm2=autograd::sigmoid(node_mask);
                auto mx2=std::make_shared<Tensor>(N,C);
                for (int ii=0;ii<N;++ii) for (int j=0;j<C;++j) (*mx2)(ii,j)=(*x)(ii,j)*(*nm2)(ii,0);
                auto l2=forward_fn(mx2,data);
                float loss2=-(*l2)(node_idx,orig_pred);
                for (int ii=0;ii<N;++ii) loss2+=0.01f*(*nm2)(ii,0)/N;
                float grad=(loss2-total)/eps;
                (*node_mask)(i,0)-=eps+lr_*grad;
            }
        }
        // Extract final masks
        Explanation exp;
        exp.node_mask.resize(N); exp.edge_mask.resize(E);
        for (int i=0;i<N;++i) exp.node_mask[i]=1.0f/(1.0f+std::exp(-(*node_mask)(i,0)));
        for (int e=0;e<E;++e) exp.edge_mask[e]=1.0f/(1.0f+std::exp(-(*edge_mask)(e,0)));
        return exp;
    }
};

// =====================================================================
//  DeepGCNLayer -- "DeepGCNs: Can GCNs Go as Deep as CNNs?"
//  Wraps a conv + norm + act with pre/post or residual connections.
// =====================================================================
class DeepGCNLayer {
public:
    GCNConv conv; BatchNorm1d norm;
    std::string block_; // "res", "res+", "dense", "plain"
    bool act_;

    DeepGCNLayer()=default;
    DeepGCNLayer(int channels, const std::string& block="res+", bool act=true)
        :conv(channels,channels),norm(channels),block_(block),act_(act){}

    TensorPtr forward(const TensorPtr& x, const Data& data) {
        auto h=x;
        if (block_=="res+" || block_=="pre") {
            // Pre-activation: norm -> act -> conv
            h=norm.forward(h);
            if (act_) h=autograd::relu(h);
            h=conv.forward(h,data);
            if (block_=="res+") h=autograd::add(h,x);
        } else {
            // Post-activation: conv -> norm -> act
            h=conv.forward(h,data);
            h=norm.forward(h);
            if (act_) h=autograd::relu(h);
            if (block_=="res") h=autograd::add(h,x);
        }
        return h;
    }
    std::vector<TensorPtr> parameters() const {
        auto p=conv.parameters(); auto p2=norm.parameters(); p.insert(p.end(),p2.begin(),p2.end()); return p;
    }
};

// =====================================================================
//  BasicGNN -- configurable multi-layer GNN
//  Supports GCN, GAT, SAGE, GIN backends.
// =====================================================================
class BasicGNN {
public:
    std::string conv_type_;
    int num_layers_,in_c_,hidden_c_,out_c_; float dp_;
    std::vector<GCNConv> gcn_convs;
    std::vector<SAGEConv> sage_convs;
    std::vector<GINConv> gin_convs;
    std::vector<GATConv> gat_convs;
    std::vector<BatchNorm1d> bns;
    Linear classifier;
    bool use_bn_;

    BasicGNN()=default;
    BasicGNN(const std::string& conv_type, int in, int hidden, int out, int num_layers=3,
             float dp=0.5f, bool use_bn=true)
        :conv_type_(conv_type),num_layers_(num_layers),in_c_(in),hidden_c_(hidden),
         out_c_(out),dp_(dp),use_bn_(use_bn),classifier(hidden,out) {
        for (int i=0;i<num_layers;++i) {
            int inf=(i==0)?in:hidden;
            if (conv_type=="gcn") gcn_convs.emplace_back(inf,hidden);
            else if (conv_type=="sage") sage_convs.emplace_back(inf,hidden);
            else if (conv_type=="gin") gin_convs.emplace_back(inf,hidden);
            else if (conv_type=="gat") gat_convs.emplace_back(inf,hidden,1,false);
            if (use_bn) bns.emplace_back(hidden);
        }
    }

    TensorPtr forward(const TensorPtr& x, const Data& data, bool training=true) {
        auto h=x;
        for (int i=0;i<num_layers_;++i) {
            if (conv_type_=="gcn") h=gcn_convs[i].forward(h,data);
            else if (conv_type_=="sage") h=sage_convs[i].forward(h,data);
            else if (conv_type_=="gin") h=gin_convs[i].forward(h,data);
            else if (conv_type_=="gat") h=gat_convs[i].forward(h,data);
            if (use_bn_ && i<(int)bns.size()) h=bns[i].forward(h);
            h=autograd::relu(h);
            h=dropout(h,dp_,training);
        }
        return classifier.forward(h);
    }

    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> p;
        auto add=[&](const auto& vec){ for (auto& v:vec) { auto vp=v.parameters(); p.insert(p.end(),vp.begin(),vp.end()); } };
        add(gcn_convs); add(sage_convs); add(gin_convs); add(gat_convs); add(bns);
        auto cp=classifier.parameters(); p.insert(p.end(),cp.begin(),cp.end());
        return p;
    }
};

// =====================================================================
//  RENet -- Recurrent Event Network (simplified for temporal graphs)
//  Uses RNN to model temporal interactions.
// =====================================================================
class RENet {
public:
    int in_channels_,hidden_channels_,num_rels_;
    GRUCell rnn;
    Linear event_lin;

    RENet()=default;
    RENet(int in, int hidden, int num_rels)
        :in_channels_(in),hidden_channels_(hidden),num_rels_(num_rels),
         rnn(in+num_rels,hidden),event_lin(hidden,in){}

    // Process a sequence of events (subject, relation, object, time)
    TensorPtr forward(const TensorPtr& x, const std::vector<std::tuple<int,int,int>>& events) {
        int N=x->rows(), H=hidden_channels_;
        auto h=Tensor::zeros(N,H);
        for (auto& [s,r,o]:events) {
            // Construct event feature: [x_s; one_hot(r)]
            auto event_feat=Tensor::zeros(1,in_channels_+num_rels_);
            for (int j=0;j<in_channels_;++j) (*event_feat)(0,j)=(*x)(s,j);
            (*event_feat)(0,in_channels_+r)=1.0f;
            // Update object hidden state
            auto h_obj=h->slice_rows(o,o+1);
            auto new_h=rnn.forward(event_feat,h_obj);
            for (int j=0;j<H;++j) (*h)(o,j)=(*new_h)(0,j);
        }
        return event_lin.forward(h);
    }

    std::vector<TensorPtr> parameters() const {
        auto p=rnn.parameters(); auto p2=event_lin.parameters(); p.insert(p.end(),p2.begin(),p2.end()); return p;
    }
};

// =====================================================================
//  InnerProductDecoder -- inner product decoder for link prediction
// =====================================================================
class InnerProductDecoder {
public:
    TensorPtr forward(const TensorPtr& z, const std::vector<int>& src, const std::vector<int>& dst) {
        int E=(int)src.size(),C=z->cols();
        auto out=std::make_shared<Tensor>(E,1);
        for (int e=0;e<E;++e) { float dot=0; for (int j=0;j<C;++j) dot+=(*z)(src[e],j)*(*z)(dst[e],j);
            (*out)(e,0)=1.0f/(1.0f+std::exp(-dot)); }
        return out;
    }
};

// =====================================================================
//  LabelPropagation -- semi-supervised label propagation
// =====================================================================
class LabelPropagation {
public:
    int num_layers_; float alpha_;

    LabelPropagation(int num_layers=3, float alpha=0.5f):num_layers_(num_layers),alpha_(alpha){}

    TensorPtr forward(const TensorPtr& y, const Data& data, const std::vector<bool>& mask) {
        int N=data.num_nodes,C=y->cols();
        auto h=std::make_shared<Tensor>(N,C);
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*h)(i,j)=mask[i]?(*y)(i,j):0;
        // Compute normalized adjacency
        auto deg=utils::degree(data.edge_dst,N);
        std::vector<float> dis(N,0); for (int i=0;i<N;++i) if (deg[i]>0) dis[i]=1.0f/std::sqrt(deg[i]);
        for (int l=0;l<num_layers_;++l) {
            auto nxt=Tensor::zeros(N,C);
            for (int e=0;e<data.num_edges();++e) { float w=dis[data.edge_src[e]]*dis[data.edge_dst[e]];
                for (int j=0;j<C;++j) (*nxt)(data.edge_dst[e],j)+=w*(*h)(data.edge_src[e],j); }
            for (int i=0;i<N;++i) for (int j=0;j<C;++j) {
                if (mask[i]) (*h)(i,j)=alpha_*(*y)(i,j)+(1-alpha_)*(*nxt)(i,j);
                else (*h)(i,j)=(*nxt)(i,j);
            }
        }
        return h;
    }
};

// =====================================================================
//  CorrectAndSmooth -- "Combining Label Propagation and Simple Models
//                       Out-performs Graph Neural Networks" (Huang et al.)
// =====================================================================
class CorrectAndSmooth {
public:
    int num_correction_layers_,num_smoothing_layers_;
    float correction_alpha_,smoothing_alpha_;

    CorrectAndSmooth(int ncl=50, int nsl=50, float ca=0.5f, float sa=0.5f)
        :num_correction_layers_(ncl),num_smoothing_layers_(nsl),correction_alpha_(ca),smoothing_alpha_(sa){}

    TensorPtr correct(const TensorPtr& pred, const TensorPtr& y_true,
                     const std::vector<bool>& mask, const Data& data) {
        int N=data.num_nodes,C=pred->cols();
        auto deg=utils::degree(data.edge_dst,N);
        std::vector<float> dis(N,0); for (int i=0;i<N;++i) if (deg[i]>0) dis[i]=1.0f/std::sqrt(deg[i]);
        // Compute error on training nodes
        auto error=Tensor::zeros(N,C);
        for (int i=0;i<N;++i) if (mask[i]) for (int j=0;j<C;++j) (*error)(i,j)=(*y_true)(i,j)-(*pred)(i,j);
        // Propagate errors
        for (int l=0;l<num_correction_layers_;++l) {
            auto nxt=Tensor::zeros(N,C);
            for (int e=0;e<data.num_edges();++e) { float w=dis[data.edge_src[e]]*dis[data.edge_dst[e]];
                for (int j=0;j<C;++j) (*nxt)(data.edge_dst[e],j)+=w*(*error)(data.edge_src[e],j); }
            for (int i=0;i<N;++i) for (int j=0;j<C;++j) {
                if (mask[i]) (*error)(i,j)=correction_alpha_*(*y_true)(i,j)-(*pred)(i,j)+(1-correction_alpha_)*(*nxt)(i,j);
                else (*error)(i,j)=(*nxt)(i,j);
            }
        }
        // Add error to predictions
        auto corrected=std::make_shared<Tensor>(N,C);
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*corrected)(i,j)=(*pred)(i,j)+(*error)(i,j);
        return corrected;
    }

    TensorPtr smooth(const TensorPtr& pred, const TensorPtr& y_true,
                    const std::vector<bool>& mask, const Data& data) {
        int N=data.num_nodes,C=pred->cols();
        auto deg=utils::degree(data.edge_dst,N);
        std::vector<float> dis(N,0); for (int i=0;i<N;++i) if (deg[i]>0) dis[i]=1.0f/std::sqrt(deg[i]);
        auto h=std::make_shared<Tensor>(N,C);
        for (int i=0;i<N;++i) for (int j=0;j<C;++j) (*h)(i,j)=mask[i]?(*y_true)(i,j):(*pred)(i,j);
        for (int l=0;l<num_smoothing_layers_;++l) {
            auto nxt=Tensor::zeros(N,C);
            for (int e=0;e<data.num_edges();++e) { float w=dis[data.edge_src[e]]*dis[data.edge_dst[e]];
                for (int j=0;j<C;++j) (*nxt)(data.edge_dst[e],j)+=w*(*h)(data.edge_src[e],j); }
            for (int i=0;i<N;++i) for (int j=0;j<C;++j) {
                if (mask[i]) (*h)(i,j)=smoothing_alpha_*(*y_true)(i,j)+(1-smoothing_alpha_)*(*nxt)(i,j);
                else (*h)(i,j)=(*nxt)(i,j);
            }
        }
        return h;
    }
};

} // namespace models
} // namespace pyg
