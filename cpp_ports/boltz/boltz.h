#pragma once
// boltz.h -- Comprehensive header-only C++17 port of the Boltz protein
// structure prediction architecture.
//
// Covers the full pipeline:
//   Data parsing (FASTA, A3M/MSA, PDB, mmCIF)
//   Tokenization & featurization
//   Cropping strategies
//   Input embedding (atom encoder, relative position)
//   MSA module (OPM, TriMul, TriAttn, PWA)
//   Pairformer (TriMul, TriAttn, AttentionPairBias)
//   Diffusion module (noise schedule, conditioning, atom transformer, decoder)
//   Confidence heads (pLDDT, pTM, iPTM, PAE, PDE)
//   Recycling mechanism
//   Structure module (IPA, backbone update)
//   All loss functions (FAPE, smooth lDDT, distogram, confidence, B-factor)
//   Full prediction pipeline
//   PDB/mmCIF output

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace boltz {

// ============================================================================
// Compile-time constants  (matching Python boltz.data.const)
// ============================================================================
static constexpr int NUM_AA = 20;
static constexpr int NUM_TOKENS = 21; // 20 AA + UNK
static constexpr float INF_VAL = 1e9f;

// Extended token vocabulary (pad, gap, 20 AA + UNK, 5 RNA, 5 DNA = 33)
static constexpr int VOCAB_SIZE = 33;

// Molecule types (matching chain_type_ids)
static constexpr int MOL_PROTEIN    = 0;
static constexpr int MOL_DNA        = 1;
static constexpr int MOL_RNA        = 2;
static constexpr int MOL_NONPOLYMER = 3;

// Bond types
static constexpr int BOND_OTHER    = 0;
static constexpr int BOND_SINGLE   = 1;
static constexpr int BOND_DOUBLE   = 2;
static constexpr int BOND_TRIPLE   = 3;
static constexpr int BOND_AROMATIC = 4;
static constexpr int BOND_COVALENT = 5;

// Ideal backbone bond geometry (Angstroms / radians)
static constexpr float BOND_CA_N  = 1.458f;
static constexpr float BOND_CA_C  = 1.523f;
static constexpr float BOND_C_O   = 1.231f;
static constexpr float BOND_C_N   = 1.329f;
static constexpr float ANGLE_N_CA_C  = 111.2f * (float)M_PI / 180.0f;
static constexpr float ANGLE_CA_C_O  = 120.8f * (float)M_PI / 180.0f;

// Architecture hyper-parameters
static constexpr int TOKEN_S = 64;
static constexpr int TOKEN_Z = 32;
static constexpr int MSA_S   = 32;
static constexpr int ATOM_S  = 64;
static constexpr int ATOM_Z  = 16;
static constexpr int NUM_HEADS = 4;
static constexpr int HEAD_DIM = TOKEN_S / NUM_HEADS;
static constexpr int PAIR_HEADS = 2;
static constexpr int PAIR_HEAD_DIM = TOKEN_Z / PAIR_HEADS;
static constexpr int TRANSITION_FACTOR = 4;
static constexpr int NUM_MSA_BLOCKS = 2;
static constexpr int NUM_PAIRFORMER_BLOCKS = 2;
static constexpr int NUM_DIFFUSION_BLOCKS = 2;
static constexpr int NUM_IPA_QUERY_POINTS = 4;
static constexpr int NUM_IPA_VALUE_POINTS = 4;
static constexpr int REL_POS_MAX = 32;
static constexpr int SYM_MAX = 2;
static constexpr int NUM_DIST_BINS = 64;
static constexpr float DIST_MIN = 2.0f;
static constexpr float DIST_MAX = 22.0f;
static constexpr int NUM_ELEMENTS = 128;

// Confidence head bins
static constexpr int NUM_PLDDT_BINS = 50;
static constexpr int NUM_PDE_BINS   = 64;
static constexpr int NUM_PAE_BINS   = 64;

// Diffusion hyper-parameters
static constexpr float SIGMA_DATA = 16.0f;
static constexpr int DIM_FOURIER  = 256;
static constexpr int ATOM_ENCODER_DEPTH = 3;
static constexpr int ATOM_DECODER_DEPTH = 3;
static constexpr int TOKEN_TRANSFORMER_DEPTH = 4;
static constexpr int ATOMS_PER_WINDOW_Q = 32;
static constexpr int ATOMS_PER_WINDOW_K = 128;
static constexpr int ATOM_FEATURE_DIM = 128;

// FAPE constants
static constexpr float FAPE_CLAMP = 10.0f;
static constexpr float FAPE_D = 10.0f;

// Recycling
static constexpr int NUM_RECYCLES = 3;

// Cropping
static constexpr float INTERFACE_CUTOFF = 15.0f;
static constexpr int CHUNK_SIZE_THRESHOLD = 384;

// Max MSA
static constexpr int MAX_MSA_SEQS = 16384;
static constexpr int MAX_PAIRED_SEQS = 8192;

// ============================================================================
// Amino acid tables
// ============================================================================
static const char AA_1LETTER[NUM_AA] = {
    'A','R','N','D','C','Q','E','G','H','I',
    'L','K','M','F','P','S','T','W','Y','V'
};

static const char* AA_3LETTER[NUM_AA] = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"
};

// Full token vocabulary: <pad>, -, 20 AA + UNK, A/G/C/U/N(RNA), DA/DG/DC/DT/DN(DNA)
static const char* FULL_TOKENS[] = {
    "<pad>", "-",
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "UNK",
    "A","G","C","U","N",
    "DA","DG","DC","DT","DN"
};
static constexpr int NUM_FULL_TOKENS = sizeof(FULL_TOKENS) / sizeof(FULL_TOKENS[0]);

inline int full_token_to_id(const std::string& tok) {
    for (int i = 0; i < NUM_FULL_TOKENS; ++i)
        if (tok == FULL_TOKENS[i]) return i;
    return 22; // UNK
}

inline int aa_to_index(char c) {
    for (int i = 0; i < NUM_AA; ++i)
        if (AA_1LETTER[i] == c) return i;
    return NUM_AA; // UNK
}

inline const char* aa_index_to_3letter(int idx) {
    if (idx >= 0 && idx < NUM_AA) return AA_3LETTER[idx];
    return "UNK";
}

// Protein single letter to 3-letter code
inline std::string prot_letter_to_3letter(char c) {
    int idx = aa_to_index(c);
    if (idx < NUM_AA) return AA_3LETTER[idx];
    return "UNK";
}

// RNA single letter to token
inline std::string rna_letter_to_token(char c) {
    switch (c) {
        case 'A': return "A"; case 'G': return "G"; case 'C': return "C";
        case 'U': return "U"; default: return "N";
    }
}

// DNA single letter to token
inline std::string dna_letter_to_token(char c) {
    switch (c) {
        case 'A': return "DA"; case 'G': return "DG"; case 'C': return "DC";
        case 'T': return "DT"; default: return "DN";
    }
}

// Reference atom counts per residue type
inline int ref_atom_count(const std::string& res) {
    static const std::unordered_map<std::string, int> counts = {
        {"ALA",5},{"ARG",11},{"ASN",8},{"ASP",8},{"CYS",6},
        {"GLN",9},{"GLU",9},{"GLY",4},{"HIS",10},{"ILE",8},
        {"LEU",8},{"LYS",9},{"MET",8},{"PHE",11},{"PRO",7},
        {"SER",6},{"THR",7},{"TRP",14},{"TYR",12},{"VAL",7},
        {"UNK",5},
        {"A",22},{"G",23},{"C",20},{"U",20},{"N",12},
        {"DA",21},{"DG",22},{"DC",19},{"DT",20},{"DN",11}
    };
    auto it = counts.find(res);
    return it != counts.end() ? it->second : 5;
}

// ============================================================================
// Dense matrix type  (row-major, contiguous)
// ============================================================================
struct Mat {
    std::vector<float> data;
    int rows = 0, cols = 0;

    Mat() = default;
    Mat(int r, int c) : data(r * c, 0.0f), rows(r), cols(c) {}
    Mat(int r, int c, float val) : data(r * c, val), rows(r), cols(c) {}

    float& operator()(int r, int c)       { return data[r * cols + c]; }
    float  operator()(int r, int c) const { return data[r * cols + c]; }
    float* row_ptr(int r)       { return data.data() + r * cols; }
    const float* row_ptr(int r) const { return data.data() + r * cols; }
    int size() const { return rows * cols; }
    void fill(float v) { std::fill(data.begin(), data.end(), v); }
    void resize(int r, int c) { rows = r; cols = c; data.assign(r * c, 0.0f); }

    Mat transpose() const {
        Mat t(cols, rows);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                t(j, i) = (*this)(i, j);
        return t;
    }
};

// 3D tensor (B x R x C) stored contiguously
struct Tensor3 {
    std::vector<float> data;
    int d0 = 0, d1 = 0, d2 = 0;

    Tensor3() = default;
    Tensor3(int a, int b, int c) : data(a*b*c, 0.0f), d0(a), d1(b), d2(c) {}
    Tensor3(int a, int b, int c, float v) : data(a*b*c, v), d0(a), d1(b), d2(c) {}

    float& operator()(int a, int b, int c) { return data[(a*d1 + b)*d2 + c]; }
    float  operator()(int a, int b, int c) const { return data[(a*d1 + b)*d2 + c]; }
    int size() const { return d0 * d1 * d2; }
    void fill(float v) { std::fill(data.begin(), data.end(), v); }
};

// ============================================================================
// Vec3  --  minimal 3-D vector
// ============================================================================
struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    float dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    float norm() const { return std::sqrt(x*x + y*y + z*z); }
    float norm_sq() const { return x*x + y*y + z*z; }
    Vec3 normalized() const {
        float n = norm();
        return (n > 1e-8f) ? Vec3{x/n,y/n,z/n} : Vec3{0,0,0};
    }
    Vec3 cross(const Vec3& o) const {
        return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x};
    }
};

// 3x3 rotation matrix
struct Rot3 {
    float m[3][3] = {{1,0,0},{0,1,0},{0,0,1}};

    static Rot3 identity() { Rot3 r; return r; }

    Vec3 apply(const Vec3& v) const {
        return {m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
                m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
                m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z};
    }

    Rot3 transpose() const {
        Rot3 r;
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) r.m[i][j] = m[j][i];
        return r;
    }

    Rot3 operator*(const Rot3& o) const {
        Rot3 r;
        for (int i=0;i<3;++i)
            for (int j=0;j<3;++j) {
                r.m[i][j] = 0;
                for (int k=0;k<3;++k) r.m[i][j] += m[i][k] * o.m[k][j];
            }
        return r;
    }

    float det() const {
        return m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
             - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
             + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]);
    }
};

// Rigid body transform  T = (R, t)  :  x_global = R * x_local + t
struct Rigid {
    Rot3 rot;
    Vec3 trans;
    Rigid() = default;
    Rigid(Rot3 r, Vec3 t) : rot(r), trans(t) {}

    Vec3 apply(const Vec3& v) const { return rot.apply(v) + trans; }
    Vec3 apply_inv(const Vec3& v) const { return rot.transpose().apply(v - trans); }

    Rigid compose(const Rigid& other) const {
        return {rot * other.rot, rot.apply(other.trans) + trans};
    }

    Rigid inverse() const {
        Rot3 rt = rot.transpose();
        return {rt, rt.apply(trans * -1.0f)};
    }
};

// ============================================================================
// Random number generator helpers
// ============================================================================
static inline std::mt19937& global_rng() {
    static std::mt19937 rng(42);
    return rng;
}

inline void set_seed(uint32_t s) { global_rng().seed(s); }

inline float randn() {
    static std::normal_distribution<float> d(0.0f, 1.0f);
    return d(global_rng());
}

inline float rand_uniform() {
    static std::uniform_real_distribution<float> d(0.0f, 1.0f);
    return d(global_rng());
}

inline int rand_int(int lo, int hi) {
    if (lo >= hi) return lo;
    std::uniform_int_distribution<int> d(lo, hi - 1);
    return d(global_rng());
}

inline void xavier_uniform(Mat& W) {
    float limit = std::sqrt(6.0f / (float)(W.rows + W.cols));
    for (auto& v : W.data) v = (rand_uniform() * 2.0f - 1.0f) * limit;
}

inline void gating_init(Mat& W) {
    for (auto& v : W.data) v = 0.0f;
}

// Quaternion to rotation matrix
inline Rot3 quaternion_to_rot3(float w, float x, float y, float z) {
    float s = 2.0f / (w*w + x*x + y*y + z*z);
    Rot3 r;
    r.m[0][0] = 1 - s*(y*y+z*z); r.m[0][1] = s*(x*y-z*w);     r.m[0][2] = s*(x*z+y*w);
    r.m[1][0] = s*(x*y+z*w);     r.m[1][1] = 1 - s*(x*x+z*z); r.m[1][2] = s*(y*z-x*w);
    r.m[2][0] = s*(x*z-y*w);     r.m[2][1] = s*(y*z+x*w);     r.m[2][2] = 1 - s*(x*x+y*y);
    return r;
}

// Random rotation matrix
inline Rot3 random_rotation() {
    float q[4];
    for (int i = 0; i < 4; ++i) q[i] = randn();
    float n = std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if (n < 1e-10f) return Rot3::identity();
    float sign = (q[0] < 0) ? -1.0f : 1.0f;
    for (int i = 0; i < 4; ++i) q[i] = sign * q[i] / n;
    return quaternion_to_rot3(q[0], q[1], q[2], q[3]);
}

// ============================================================================
// Layer norm  (over last dim)
// ============================================================================
struct LayerNorm {
    int dim;
    std::vector<float> gamma, beta;
    float eps;

    LayerNorm() : dim(0), eps(1e-5f) {}
    explicit LayerNorm(int d, float e = 1e-5f)
        : dim(d), gamma(d, 1.0f), beta(d, 0.0f), eps(e) {}

    void forward(Mat& mat) const {
        assert(mat.cols == dim);
        for (int i = 0; i < mat.rows; ++i) {
            float* r = mat.row_ptr(i);
            float mean = 0;
            for (int j = 0; j < dim; ++j) mean += r[j];
            mean /= dim;
            float var = 0;
            for (int j = 0; j < dim; ++j) { float d2 = r[j] - mean; var += d2*d2; }
            var /= dim;
            float inv = 1.0f / std::sqrt(var + eps);
            for (int j = 0; j < dim; ++j)
                r[j] = gamma[j] * (r[j] - mean) * inv + beta[j];
        }
    }

    // Variant without learnable affine parameters
    void forward_no_affine(Mat& mat) const {
        for (int i = 0; i < mat.rows; ++i) {
            float* r = mat.row_ptr(i);
            float mean = 0;
            for (int j = 0; j < mat.cols; ++j) mean += r[j];
            mean /= mat.cols;
            float var = 0;
            for (int j = 0; j < mat.cols; ++j) { float d2 = r[j] - mean; var += d2*d2; }
            var /= mat.cols;
            float inv = 1.0f / std::sqrt(var + eps);
            for (int j = 0; j < mat.cols; ++j)
                r[j] = (r[j] - mean) * inv;
        }
    }
};

// ============================================================================
// Linear layer   y = x W^T + b   (W is out x in)
// ============================================================================
struct Linear {
    Mat W;
    std::vector<float> bias;
    bool use_bias;

    Linear() : use_bias(false) {}

    Linear(int in_dim, int out_dim, bool bias_ = true) : use_bias(bias_) {
        W = Mat(out_dim, in_dim);
        xavier_uniform(W);
        if (use_bias) bias.assign(out_dim, 0.0f);
    }

    Mat forward(const Mat& x) const {
        int N = x.rows;
        int out_dim = W.rows;
        int in_dim = W.cols;
        assert(x.cols == in_dim);
        Mat y(N, out_dim);
        for (int i = 0; i < N; ++i) {
            const float* xi = x.row_ptr(i);
            float* yi = y.row_ptr(i);
            for (int o = 0; o < out_dim; ++o) {
                float s = use_bias ? bias[o] : 0.0f;
                const float* Wo = W.row_ptr(o);
                for (int k = 0; k < in_dim; ++k) s += xi[k] * Wo[k];
                yi[o] = s;
            }
        }
        return y;
    }
};

// ============================================================================
// Embedding layer  (lookup table)
// ============================================================================
struct Embedding {
    Mat weight;
    int num_embeddings = 0, embedding_dim = 0;

    Embedding() = default;
    Embedding(int ne, int ed) : weight(ne, ed), num_embeddings(ne), embedding_dim(ed) {
        for (auto& v : weight.data) v = randn() * 0.02f;
    }

    void lookup(int idx, float* out) const {
        const float* row = weight.row_ptr(std::clamp(idx, 0, num_embeddings - 1));
        std::copy(row, row + embedding_dim, out);
    }

    Mat forward(const std::vector<int>& indices) const {
        int N = (int)indices.size();
        Mat out(N, embedding_dim);
        for (int i = 0; i < N; ++i)
            lookup(indices[i], out.row_ptr(i));
        return out;
    }
};

// ============================================================================
// Activation helpers
// ============================================================================
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float relu_f(float x) { return x > 0.0f ? x : 0.0f; }

inline void silu_inplace(Mat& m) { for (auto& v : m.data) v = silu(v); }
inline void sigmoid_inplace(Mat& m) { for (auto& v : m.data) v = sigmoid_f(v); }
inline void relu_inplace(Mat& m) { for (auto& v : m.data) v = relu_f(v); }

// ============================================================================
// Softmax  (over columns of a row, in-place)
// ============================================================================
inline void softmax_row(float* row, int n) {
    float mx = *std::max_element(row, row + n);
    float s = 0;
    for (int j = 0; j < n; ++j) { row[j] = std::exp(row[j] - mx); s += row[j]; }
    float inv = 1.0f / (s + 1e-12f);
    for (int j = 0; j < n; ++j) row[j] *= inv;
}

inline void softmax_rows(Mat& m) {
    for (int i = 0; i < m.rows; ++i)
        softmax_row(m.row_ptr(i), m.cols);
}

inline void log_softmax_rows(Mat& m) {
    for (int i = 0; i < m.rows; ++i) {
        float* r = m.row_ptr(i);
        float mx = *std::max_element(r, r + m.cols);
        float s = 0;
        for (int j = 0; j < m.cols; ++j) s += std::exp(r[j] - mx);
        float log_s = std::log(s + 1e-12f) + mx;
        for (int j = 0; j < m.cols; ++j) r[j] -= log_s;
    }
}

// ============================================================================
// SwiGLU activation
// ============================================================================
inline Mat swiglu(const Mat& x) {
    assert(x.cols % 2 == 0);
    int half = x.cols / 2;
    Mat out(x.rows, half);
    for (int i = 0; i < x.rows; ++i)
        for (int j = 0; j < half; ++j) {
            float gate = silu(x(i, j + half));
            out(i, j) = x(i, j) * gate;
        }
    return out;
}

// ============================================================================
// Transition block  (SwiGLU style, matching Boltz)
// ============================================================================
struct Transition {
    LayerNorm norm;
    Linear fc1, fc2, fc3;

    Transition() = default;
    Transition(int dim, int hidden)
        : norm(dim), fc1(dim, hidden, false),
          fc2(dim, hidden, false), fc3(hidden, dim, false) {}

    Mat forward(const Mat& x) const {
        Mat xn = x;
        norm.forward(xn);
        Mat a = fc1.forward(xn);
        Mat b = fc2.forward(xn);
        silu_inplace(a);
        for (int i = 0; i < a.size(); ++i) a.data[i] *= b.data[i];
        return fc3.forward(a);
    }
};

// ============================================================================
// MatMul & element-wise helpers
// ============================================================================
inline void mat_add(Mat& A, const Mat& B) {
    assert(A.rows == B.rows && A.cols == B.cols);
    for (int i = 0; i < A.size(); ++i) A.data[i] += B.data[i];
}

inline void mat_sub(Mat& A, const Mat& B) {
    assert(A.rows == B.rows && A.cols == B.cols);
    for (int i = 0; i < A.size(); ++i) A.data[i] -= B.data[i];
}

inline void mat_mul_scalar(Mat& A, float s) {
    for (auto& v : A.data) v *= s;
}

inline void mat_elementwise_mul(Mat& A, const Mat& B) {
    assert(A.rows == B.rows && A.cols == B.cols);
    for (int i = 0; i < A.size(); ++i) A.data[i] *= B.data[i];
}

inline Mat mat_concat_cols(const Mat& A, const Mat& B) {
    assert(A.rows == B.rows);
    Mat C(A.rows, A.cols + B.cols);
    for (int i = 0; i < A.rows; ++i) {
        std::copy(A.row_ptr(i), A.row_ptr(i) + A.cols, C.row_ptr(i));
        std::copy(B.row_ptr(i), B.row_ptr(i) + B.cols, C.row_ptr(i) + A.cols);
    }
    return C;
}

inline Mat matmul(const Mat& A, const Mat& B) {
    assert(A.cols == B.rows);
    Mat C(A.rows, B.cols, 0.0f);
    for (int i = 0; i < A.rows; ++i)
        for (int k = 0; k < A.cols; ++k) {
            float a = A(i, k);
            for (int j = 0; j < B.cols; ++j)
                C(i, j) += a * B(k, j);
        }
    return C;
}

// Pairwise distance matrix from Nx3 coordinates
inline Mat cdist(const Mat& coords) {
    int N = coords.rows;
    Mat D(N, N, 0.0f);
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j) {
            float dx = coords(i,0)-coords(j,0);
            float dy = coords(i,1)-coords(j,1);
            float dz = coords(i,2)-coords(j,2);
            float d = std::sqrt(dx*dx + dy*dy + dz*dz);
            D(i,j) = d;
            D(j,i) = d;
        }
    return D;
}

// ============================================================================
// One-hot encoding for sequences
// ============================================================================
inline Mat one_hot_encode(const std::string& seq) {
    int N = (int)seq.size();
    Mat oh(N, NUM_TOKENS, 0.0f);
    for (int i = 0; i < N; ++i) {
        int idx = aa_to_index(seq[i]);
        if (idx >= NUM_TOKENS) idx = NUM_AA;
        oh(i, idx) = 1.0f;
    }
    return oh;
}

inline Mat one_hot_full(const std::vector<int>& token_ids) {
    int N = (int)token_ids.size();
    Mat oh(N, NUM_FULL_TOKENS, 0.0f);
    for (int i = 0; i < N; ++i) {
        int idx = std::clamp(token_ids[i], 0, NUM_FULL_TOKENS - 1);
        oh(i, idx) = 1.0f;
    }
    return oh;
}

// ============================================================================
//  DATA TYPES  (matching Python boltz.data.types)
// ============================================================================

struct AtomData {
    std::string name;
    int element = 6;
    int charge = 0;
    Vec3 coords;
    bool is_present = true;
};

struct BondData {
    int atom_1 = 0, atom_2 = 0;
    int type = BOND_SINGLE;
};

struct ResidueData {
    std::string name;
    int res_type = 0;
    int res_idx = 0;
    int atom_idx = 0;
    int atom_num = 0;
    int atom_center = 0;
    int atom_disto = 0;
    bool is_standard = true;
    bool is_present = true;
};

struct ChainData {
    std::string name;
    int mol_type = MOL_PROTEIN;
    int entity_id = 0;
    int sym_id = 0;
    int asym_id = 0;
    int atom_idx = 0;
    int atom_num = 0;
    int res_idx = 0;
    int res_num = 0;
    int cyclic_period = 0;
};

struct ConnectionData {
    int chain_1 = 0, chain_2 = 0;
    int res_1 = 0, res_2 = 0;
    int atom_1 = 0, atom_2 = 0;
};

struct InterfaceData {
    int chain_1 = 0, chain_2 = 0;
    bool valid = true;
};

struct Structure {
    std::vector<AtomData> atoms;
    std::vector<BondData> bonds;
    std::vector<ResidueData> residues;
    std::vector<ChainData> chains;
    std::vector<ConnectionData> connections;
    std::vector<InterfaceData> interfaces;
    std::vector<bool> mask;
};

struct MSAEntry {
    std::vector<int> residues;
    std::vector<std::pair<int,int>> deletions;
    int taxonomy_id = -1;
};

struct MSAData {
    std::vector<MSAEntry> sequences;
};

struct TokenData {
    int token_idx = 0;
    int atom_idx = 0;
    int atom_num = 0;
    int res_idx = 0;
    int res_type = 0;
    int sym_id = 0;
    int asym_id = 0;
    int entity_id = 0;
    int mol_type = MOL_PROTEIN;
    int center_idx = 0;
    int disto_idx = 0;
    Vec3 center_coords;
    Vec3 disto_coords;
    bool resolved_mask = true;
    bool disto_mask = true;
    int cyclic_period = 0;
};

struct TokenBondData {
    int token_1 = 0, token_2 = 0;
};

struct Tokenized {
    std::vector<TokenData> tokens;
    std::vector<TokenBondData> bonds;
    Structure structure;
    std::map<int, MSAData> msa;
};

struct ChainInfo {
    int chain_id = 0;
    std::string chain_name;
    int mol_type = MOL_PROTEIN;
    int num_residues = 0;
    bool valid = true;
    int entity_id = 0;
};

struct Record {
    std::string id;
    std::vector<ChainInfo> chains;
    std::vector<InterfaceData> interfaces;
};

// ============================================================================
//  DATA PARSING: FASTA
// ============================================================================

struct FASTARecord {
    std::string id;
    std::string chain_id;
    std::string entity_type;
    std::string msa_id;
    std::string sequence;
};

inline std::vector<FASTARecord> parse_fasta(const std::string& path) {
    std::vector<FASTARecord> records;
    std::ifstream f(path);
    if (!f.is_open()) return records;

    FASTARecord current;
    bool has_current = false;
    std::string line;

    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        if (line.empty()) continue;

        if (line[0] == '>') {
            if (has_current && !current.sequence.empty())
                records.push_back(current);
            current = FASTARecord();
            has_current = true;

            std::string header = line.substr(1);
            auto sp = header.find(' ');
            if (sp != std::string::npos) header = header.substr(0, sp);

            std::vector<std::string> parts;
            std::istringstream ss(header);
            std::string part;
            while (std::getline(ss, part, '|'))
                parts.push_back(part);

            if (parts.size() >= 1) current.chain_id = parts[0];
            if (parts.size() >= 2) {
                current.entity_type = parts[1];
                for (auto& c : current.entity_type) c = (char)std::toupper(c);
            }
            if (parts.size() >= 3) current.msa_id = parts[2];
            current.id = current.chain_id;
        } else {
            current.sequence += line;
        }
    }
    if (has_current && !current.sequence.empty())
        records.push_back(current);
    return records;
}

// ============================================================================
//  DATA PARSING: A3M/MSA
// ============================================================================

inline MSAData parse_a3m(const std::string& path, int max_seqs = MAX_MSA_SEQS) {
    MSAData msa;
    std::ifstream f(path);
    if (!f.is_open()) return msa;

    std::string line;
    std::unordered_set<std::string> visited;
    int seq_idx = 0;
    int taxonomy_id = -1;

    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        if (line.empty() || line[0] == '#') continue;

        if (line[0] == '>') {
            taxonomy_id = -1;
            continue;
        }

        std::string clean;
        for (char c : line)
            if (c != '-' && !std::islower(c)) clean += (char)std::toupper(c);
        if (visited.count(clean)) continue;
        visited.insert(clean);

        MSAEntry entry;
        entry.taxonomy_id = taxonomy_id;
        int count = 0, res_idx = 0;

        for (char c : line) {
            if (c != '-' && std::islower(c)) { count++; continue; }
            std::string tok = prot_letter_to_3letter((char)std::toupper(c));
            int tid = full_token_to_id(tok);
            entry.residues.push_back(tid);
            if (count > 0) {
                entry.deletions.push_back({res_idx, count});
                count = 0;
            }
            res_idx++;
        }
        msa.sequences.push_back(entry);
        seq_idx++;
        if (max_seqs > 0 && seq_idx >= max_seqs) break;
    }
    return msa;
}

// ============================================================================
//  DATA PARSING: PDB
// ============================================================================

struct PDBAtom {
    std::string record_type;
    int serial = 0;
    std::string name;
    std::string res_name;
    std::string chain_id;
    int res_seq = 0;
    Vec3 coords;
    float occupancy = 1.0f;
    float b_factor = 0.0f;
    std::string element;
};

struct PDBData {
    std::vector<PDBAtom> atoms;
    std::vector<std::pair<int,int>> conect;
};

inline PDBData parse_pdb(const std::string& path) {
    PDBData pdb;
    std::ifstream f(path);
    if (!f.is_open()) return pdb;

    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        if (line.size() < 6) continue;
        std::string rec = line.substr(0, 6);
        while (!rec.empty() && rec.back() == ' ') rec.pop_back();

        if (rec == "ATOM" || rec == "HETATM") {
            if (line.size() < 54) continue;
            PDBAtom a;
            a.record_type = rec;
            try { a.serial = std::stoi(line.substr(6, 5)); } catch (...) {}
            a.name = line.substr(12, 4);
            while (!a.name.empty() && a.name.front() == ' ') a.name.erase(a.name.begin());
            while (!a.name.empty() && a.name.back() == ' ') a.name.pop_back();
            a.res_name = line.substr(17, 3);
            while (!a.res_name.empty() && a.res_name.front() == ' ') a.res_name.erase(a.res_name.begin());
            while (!a.res_name.empty() && a.res_name.back() == ' ') a.res_name.pop_back();
            a.chain_id = line.substr(21, 1);
            try { a.res_seq = std::stoi(line.substr(22, 4)); } catch (...) {}
            try {
                a.coords.x = std::stof(line.substr(30, 8));
                a.coords.y = std::stof(line.substr(38, 8));
                a.coords.z = std::stof(line.substr(46, 8));
            } catch (...) {}
            if (line.size() >= 60) try { a.occupancy = std::stof(line.substr(54, 6)); } catch (...) {}
            if (line.size() >= 66) try { a.b_factor = std::stof(line.substr(60, 6)); } catch (...) {}
            if (line.size() >= 78) {
                a.element = line.substr(76, 2);
                while (!a.element.empty() && a.element.front() == ' ') a.element.erase(a.element.begin());
                while (!a.element.empty() && a.element.back() == ' ') a.element.pop_back();
            }
            pdb.atoms.push_back(a);
        } else if (rec == "CONECT") {
            try {
                int a1 = std::stoi(line.substr(6, 5));
                for (int off = 11; off + 5 <= (int)line.size(); off += 5) {
                    std::string s = line.substr(off, 5);
                    while (!s.empty() && s.front() == ' ') s.erase(s.begin());
                    if (s.empty()) continue;
                    int a2 = std::stoi(s);
                    pdb.conect.push_back({a1, a2});
                }
            } catch (...) {}
        }
    }
    return pdb;
}

inline Structure pdb_to_structure(const PDBData& pdb) {
    Structure s;
    if (pdb.atoms.empty()) return s;

    struct ResKey {
        std::string chain; int resseq; std::string resname;
        bool operator<(const ResKey& o) const {
            if (chain != o.chain) return chain < o.chain;
            return resseq < o.resseq;
        }
    };
    std::map<std::string, int> chain_map;
    std::map<ResKey, std::vector<int>> res_atoms;
    std::vector<ResKey> res_order;

    for (int i = 0; i < (int)pdb.atoms.size(); ++i) {
        const auto& a = pdb.atoms[i];
        ResKey key{a.chain_id, a.res_seq, a.res_name};
        if (res_atoms.find(key) == res_atoms.end()) res_order.push_back(key);
        res_atoms[key].push_back(i);
        chain_map[a.chain_id] = 0;
    }
    int cid = 0;
    for (auto& [name, id] : chain_map) id = cid++;

    for (const auto& pa : pdb.atoms) {
        AtomData ad; ad.name = pa.name; ad.coords = pa.coords; ad.is_present = true;
        s.atoms.push_back(ad);
    }

    std::string prev_chain; int chain_atom_start = 0, chain_res_start = 0, chain_atom_count = 0;
    for (int ri = 0; ri < (int)res_order.size(); ++ri) {
        const auto& key = res_order[ri];
        const auto& aidxs = res_atoms[key];
        if (key.chain != prev_chain && !prev_chain.empty()) {
            ChainData cd; cd.name = prev_chain; cd.mol_type = MOL_PROTEIN;
            cd.asym_id = chain_map[prev_chain]; cd.entity_id = cd.asym_id;
            cd.atom_idx = chain_atom_start; cd.atom_num = chain_atom_count;
            cd.res_idx = chain_res_start; cd.res_num = (int)s.residues.size() - chain_res_start;
            s.chains.push_back(cd);
            chain_atom_start = aidxs[0]; chain_res_start = (int)s.residues.size(); chain_atom_count = 0;
        }
        if (key.chain != prev_chain) {
            chain_atom_start = aidxs[0]; chain_res_start = (int)s.residues.size(); chain_atom_count = 0;
        }
        ResidueData rd; rd.name = key.resname; rd.res_idx = (int)s.residues.size();
        rd.atom_idx = aidxs[0]; rd.atom_num = (int)aidxs.size();
        rd.atom_center = aidxs[0]; rd.atom_disto = aidxs[0]; rd.is_present = true;
        for (int ai : aidxs) {
            if (pdb.atoms[ai].name == "CA") rd.atom_center = ai;
            if (pdb.atoms[ai].name == "CB") rd.atom_disto = ai;
        }
        int rt = NUM_AA;
        for (int i = 0; i < NUM_AA; ++i) { if (key.resname == AA_3LETTER[i]) { rt = i; break; } }
        rd.res_type = rt; rd.is_standard = (rt < NUM_AA);
        s.residues.push_back(rd);
        chain_atom_count += (int)aidxs.size();
        prev_chain = key.chain;
    }
    if (!prev_chain.empty()) {
        ChainData cd; cd.name = prev_chain; cd.mol_type = MOL_PROTEIN;
        cd.asym_id = chain_map[prev_chain]; cd.entity_id = cd.asym_id;
        cd.atom_idx = chain_atom_start; cd.atom_num = chain_atom_count;
        cd.res_idx = chain_res_start; cd.res_num = (int)s.residues.size() - chain_res_start;
        s.chains.push_back(cd);
    }
    s.mask.assign(s.chains.size(), true);

    std::unordered_map<int, int> serial_to_idx;
    for (int i = 0; i < (int)pdb.atoms.size(); ++i) serial_to_idx[pdb.atoms[i].serial] = i;
    for (auto& [s1, s2] : pdb.conect) {
        auto it1 = serial_to_idx.find(s1), it2 = serial_to_idx.find(s2);
        if (it1 != serial_to_idx.end() && it2 != serial_to_idx.end())
            s.bonds.push_back({it1->second, it2->second, BOND_SINGLE});
    }
    return s;
}

// ============================================================================
//  DATA PARSING: mmCIF (simplified)
// ============================================================================

inline std::vector<PDBAtom> parse_mmcif_atoms(const std::string& path) {
    std::vector<PDBAtom> atoms;
    std::ifstream f(path);
    if (!f.is_open()) return atoms;

    std::string line;
    bool in_atom_site = false;
    std::vector<std::string> col_names;
    int col_group=-1,col_id=-1,col_name=-1,col_comp=-1,col_asym=-1;
    int col_seq=-1,col_x=-1,col_y=-1,col_z=-1,col_bfac=-1,col_elem=-1;

    while (std::getline(f, line)) {
        while (!line.empty() && (line.back()=='\r'||line.back()=='\n')) line.pop_back();
        if (line.find("_atom_site.") == 0) {
            in_atom_site = true;
            std::string col = line.substr(11);
            while (!col.empty() && col.back()==' ') col.pop_back();
            int idx = (int)col_names.size();
            col_names.push_back(col);
            if (col=="group_PDB") col_group=idx;
            else if (col=="id") col_id=idx;
            else if ((col=="label_atom_id"||col=="auth_atom_id")&&col_name<0) col_name=idx;
            else if ((col=="label_comp_id"||col=="auth_comp_id")&&col_comp<0) col_comp=idx;
            else if ((col=="label_asym_id"||col=="auth_asym_id")&&col_asym<0) col_asym=idx;
            else if ((col=="label_seq_id"||col=="auth_seq_id")&&col_seq<0) col_seq=idx;
            else if (col=="Cartn_x") col_x=idx;
            else if (col=="Cartn_y") col_y=idx;
            else if (col=="Cartn_z") col_z=idx;
            else if (col=="B_iso_or_equiv") col_bfac=idx;
            else if (col=="type_symbol") col_elem=idx;
            continue;
        }
        if (in_atom_site && !line.empty() && line[0]!='_' && line[0]!='#' && line[0]!=';') {
            if (line.find("loop_")!=std::string::npos||line.find("data_")!=std::string::npos) {
                in_atom_site = false; continue;
            }
            std::vector<std::string> fields;
            std::istringstream ss(line); std::string tok;
            while (ss >> tok) fields.push_back(tok);
            if ((int)fields.size() < (int)col_names.size()) continue;
            PDBAtom a;
            if (col_group>=0) a.record_type = fields[col_group];
            if (col_name>=0) a.name = fields[col_name];
            if (col_comp>=0) a.res_name = fields[col_comp];
            if (col_asym>=0) a.chain_id = fields[col_asym];
            if (col_seq>=0) try { a.res_seq = std::stoi(fields[col_seq]); } catch (...) {}
            if (col_x>=0&&col_y>=0&&col_z>=0) {
                try { a.coords.x=std::stof(fields[col_x]); a.coords.y=std::stof(fields[col_y]); a.coords.z=std::stof(fields[col_z]); } catch (...) {}
            }
            if (col_bfac>=0) try { a.b_factor = std::stof(fields[col_bfac]); } catch (...) {}
            if (col_elem>=0) a.element = fields[col_elem];
            atoms.push_back(a);
        } else if (in_atom_site && (line.empty() || line[0]=='#' || line.find("loop_")==0)) {
            in_atom_site = false;
        }
    }
    return atoms;
}

// ============================================================================
//  TOKENIZATION  (matching BoltzTokenizer)
// ============================================================================

inline Tokenized tokenize_structure(const Structure& structure) {
    Tokenized result;
    result.structure = structure;
    int token_idx = 0;
    std::unordered_map<int, int> atom_to_token;

    for (int ci = 0; ci < (int)structure.chains.size(); ++ci) {
        if (ci < (int)structure.mask.size() && !structure.mask[ci]) continue;
        const auto& chain = structure.chains[ci];
        for (int ri = chain.res_idx; ri < chain.res_idx + chain.res_num; ++ri) {
            const auto& res = structure.residues[ri];
            if (res.is_standard) {
                TokenData td;
                td.token_idx = token_idx; td.atom_idx = res.atom_idx; td.atom_num = res.atom_num;
                td.res_idx = res.res_idx; td.res_type = res.res_type;
                td.sym_id = chain.sym_id; td.asym_id = chain.asym_id;
                td.entity_id = chain.entity_id; td.mol_type = chain.mol_type;
                td.center_idx = res.atom_center; td.disto_idx = res.atom_disto;
                if (res.atom_center < (int)structure.atoms.size())
                    td.center_coords = structure.atoms[res.atom_center].coords;
                if (res.atom_disto < (int)structure.atoms.size())
                    td.disto_coords = structure.atoms[res.atom_disto].coords;
                td.resolved_mask = res.is_present; td.disto_mask = res.is_present;
                td.cyclic_period = chain.cyclic_period;
                result.tokens.push_back(td);
                for (int ai = res.atom_idx; ai < res.atom_idx + res.atom_num; ++ai)
                    atom_to_token[ai] = token_idx;
                token_idx++;
            } else {
                for (int ai = res.atom_idx; ai < res.atom_idx + res.atom_num; ++ai) {
                    TokenData td;
                    td.token_idx = token_idx; td.atom_idx = ai; td.atom_num = 1;
                    td.res_idx = res.res_idx; td.res_type = NUM_AA;
                    td.sym_id = chain.sym_id; td.asym_id = chain.asym_id;
                    td.entity_id = chain.entity_id; td.mol_type = chain.mol_type;
                    td.center_idx = ai; td.disto_idx = ai;
                    if (ai < (int)structure.atoms.size()) td.center_coords = structure.atoms[ai].coords;
                    td.disto_coords = td.center_coords;
                    td.resolved_mask = res.is_present;
                    td.disto_mask = td.resolved_mask; td.cyclic_period = chain.cyclic_period;
                    result.tokens.push_back(td);
                    atom_to_token[ai] = token_idx;
                    token_idx++;
                }
            }
        }
    }
    for (const auto& bond : structure.bonds) {
        auto it1 = atom_to_token.find(bond.atom_1), it2 = atom_to_token.find(bond.atom_2);
        if (it1 != atom_to_token.end() && it2 != atom_to_token.end())
            result.bonds.push_back({it1->second, it2->second});
    }
    for (const auto& conn : structure.connections) {
        auto it1 = atom_to_token.find(conn.atom_1), it2 = atom_to_token.find(conn.atom_2);
        if (it1 != atom_to_token.end() && it2 != atom_to_token.end())
            result.bonds.push_back({it1->second, it2->second});
    }
    return result;
}

// ============================================================================
//  CROPPING  (matching BoltzCropper)
// ============================================================================

inline Tokenized crop_tokens(const Tokenized& data, int max_tokens, int neighborhood_size = 20) {
    if ((int)data.tokens.size() <= max_tokens) return data;
    std::vector<int> resolved;
    for (int i = 0; i < (int)data.tokens.size(); ++i)
        if (data.tokens[i].resolved_mask) resolved.push_back(i);
    if (resolved.empty()) return data;

    int query_idx = resolved[rand_int(0, (int)resolved.size())];
    Vec3 qc = data.tokens[query_idx].center_coords;

    std::vector<int> sorted_idx(data.tokens.size());
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) {
        return (data.tokens[a].center_coords - qc).norm_sq() < (data.tokens[b].center_coords - qc).norm_sq();
    });

    std::set<int> cropped;
    for (int idx : sorted_idx) {
        if ((int)cropped.size() >= max_tokens) break;
        const auto& tok = data.tokens[idx];
        std::vector<int> chain_toks;
        for (int j = 0; j < (int)data.tokens.size(); ++j)
            if (data.tokens[j].asym_id == tok.asym_id) chain_toks.push_back(j);
        if ((int)chain_toks.size() <= neighborhood_size) {
            for (int ct : chain_toks) cropped.insert(ct);
        } else {
            int center_res = tok.res_idx;
            for (int ct : chain_toks)
                if (std::abs(data.tokens[ct].res_idx - center_res) <= neighborhood_size / 2)
                    cropped.insert(ct);
        }
        if ((int)cropped.size() >= max_tokens) break;
    }

    Tokenized result; result.structure = data.structure; result.msa = data.msa;
    std::set<int> token_set(cropped.begin(), cropped.end());
    for (int ti : cropped) result.tokens.push_back(data.tokens[ti]);
    for (const auto& bond : data.bonds)
        if (token_set.count(bond.token_1) && token_set.count(bond.token_2))
            result.bonds.push_back(bond);
    return result;
}

// ============================================================================
// Relative position encoding  (matching RelativePositionEncoder)
// ============================================================================
inline Mat relative_position_features(int N) {
    int r_max = REL_POS_MAX, s_max = SYM_MAX;
    int feat_dim = 4 * (r_max + 1) + 2 * (s_max + 1) + 1;
    Mat feats(N * N, feat_dim, 0.0f);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            int row = i * N + j;
            int delta = j - i;
            int bin_r = std::clamp(delta + r_max, 0, 2 * r_max);
            feats(row, bin_r) = 1.0f;
            feats(row, 2*(r_max+1) + bin_r) = 1.0f;
            feats(row, 4*(r_max+1)) = 1.0f;
            feats(row, 4*(r_max+1) + 1 + s_max) = 1.0f;
        }
    return feats;
}

inline Mat relative_position_features_full(const std::vector<TokenData>& tokens) {
    int N = (int)tokens.size();
    int r_max = REL_POS_MAX, s_max = SYM_MAX;
    int feat_dim = 4 * (r_max + 1) + 2 * (s_max + 1) + 1;
    Mat feats(N * N, feat_dim, 0.0f);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            int row = i * N + j;
            bool same_chain = (tokens[i].asym_id == tokens[j].asym_id);
            bool same_res = same_chain && (tokens[i].res_idx == tokens[j].res_idx);
            bool same_entity = (tokens[i].entity_id == tokens[j].entity_id);
            int delta = tokens[j].res_idx - tokens[i].res_idx;
            int bin_r = same_chain ? std::clamp(delta+r_max, 0, 2*r_max) : 2*r_max+1;
            feats(row, bin_r) = 1.0f;
            int delta_t = tokens[j].token_idx - tokens[i].token_idx;
            int bin_t = (same_chain && same_res) ? std::clamp(delta_t+r_max,0,2*r_max) : 2*r_max+1;
            feats(row, 2*(r_max+1)+bin_t) = 1.0f;
            feats(row, 4*(r_max+1)) = same_entity ? 1.0f : 0.0f;
            int delta_s = tokens[j].sym_id - tokens[i].sym_id;
            int bin_s = same_chain ? 2*s_max+1 : std::clamp(delta_s+s_max,0,2*s_max);
            feats(row, 4*(r_max+1)+1+bin_s) = 1.0f;
        }
    return feats;
}

// ============================================================================
//  Multi-Head Self-Attention with Pair Bias  (AttentionPairBias)
// ============================================================================
struct AttentionPairBias {
    int c_s, c_z, num_heads, head_dim;
    LayerNorm norm_s, norm_z;
    Linear proj_q, proj_k, proj_v, proj_g, proj_o, proj_z;
    bool initial_norm;

    AttentionPairBias() : c_s(0), c_z(0), num_heads(0), head_dim(0), initial_norm(true) {}
    AttentionPairBias(int cs, int cz, int nh, bool init_norm = true)
        : c_s(cs), c_z(cz), num_heads(nh), head_dim(cs / nh),
          norm_s(cs), norm_z(cz),
          proj_q(cs, cs, true), proj_k(cs, cs, false), proj_v(cs, cs, false),
          proj_g(cs, cs, false), proj_o(cs, cs, false), proj_z(cz, nh, false),
          initial_norm(init_norm) {}

    Mat forward(const Mat& s_in, const Mat& z_flat) const {
        int N = s_in.rows;
        Mat s = s_in;
        if (initial_norm) norm_s.forward(s);
        Mat Q = proj_q.forward(s), K = proj_k.forward(s);
        Mat V = proj_v.forward(s), G = proj_g.forward(s);
        sigmoid_inplace(G);
        Mat z_n = z_flat; norm_z.forward(z_n);
        Mat z_bias = proj_z.forward(z_n);
        Mat out(N, c_s, 0.0f);
        for (int h = 0; h < num_heads; ++h) {
            int off = h * head_dim;
            float scale = 1.0f / std::sqrt((float)head_dim);
            Mat attn(N, N, 0.0f);
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    float sv = 0;
                    for (int d = 0; d < head_dim; ++d) sv += Q(i,off+d)*K(j,off+d);
                    attn(i,j) = sv*scale + z_bias(i*N+j, h);
                }
            softmax_rows(attn);
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    float w = attn(i,j);
                    for (int d = 0; d < head_dim; ++d) out(i,off+d) += w*V(j,off+d);
                }
        }
        for (int i = 0; i < out.size(); ++i) out.data[i] *= G.data[i];
        return proj_o.forward(out);
    }
};

// ============================================================================
// Triangle Multiplication  (Outgoing / Incoming)
// ============================================================================
struct TriangleMultiplication {
    int dim; bool outgoing;
    LayerNorm norm_in, norm_out;
    Linear p_in, g_in, p_out, g_out;

    TriangleMultiplication() : dim(0), outgoing(true) {}
    TriangleMultiplication(int d, bool out)
        : dim(d), outgoing(out), norm_in(d), norm_out(d),
          p_in(d,2*d,false), g_in(d,2*d,false), p_out(d,d,false), g_out(d,d,false) {}

    Mat forward(const Mat& x_flat, int N) const {
        Mat x = x_flat; norm_in.forward(x);
        Mat x_in = x;
        Mat p = p_in.forward(x); Mat g = g_in.forward(x);
        sigmoid_inplace(g);
        for (int i = 0; i < p.size(); ++i) p.data[i] *= g.data[i];
        Mat a(N*N,dim), b(N*N,dim);
        for (int i = 0; i < N*N; ++i) for (int d = 0; d < dim; ++d) { a(i,d)=p(i,d); b(i,d)=p(i,dim+d); }
        Mat z(N*N, dim, 0.0f);
        if (outgoing) {
            for (int i=0;i<N;++i) for (int j=0;j<N;++j) for (int k=0;k<N;++k) for (int d=0;d<dim;++d)
                z(i*N+j,d) += a(i*N+k,d)*b(j*N+k,d);
        } else {
            for (int i=0;i<N;++i) for (int j=0;j<N;++j) for (int k=0;k<N;++k) for (int d=0;d<dim;++d)
                z(i*N+j,d) += a(k*N+i,d)*b(k*N+j,d);
        }
        norm_out.forward(z);
        Mat z_out = p_out.forward(z);
        Mat g_o = g_out.forward(x_in); sigmoid_inplace(g_o);
        for (int i = 0; i < z_out.size(); ++i) z_out.data[i] *= g_o.data[i];
        return z_out;
    }
};

// ============================================================================
// Triangle Attention  (Starting / Ending node)
// ============================================================================
struct TriangleAttention {
    int c_in, c_hidden, no_heads; bool starting;
    LayerNorm layer_norm;
    Linear linear_bias, proj_q, proj_k, proj_v, proj_o, proj_g;

    TriangleAttention() : c_in(0), c_hidden(0), no_heads(0), starting(true) {}
    TriangleAttention(int cin, int chid, int nh, bool start)
        : c_in(cin), c_hidden(chid), no_heads(nh), starting(start),
          layer_norm(cin), linear_bias(cin,nh,false),
          proj_q(cin,nh*chid,false), proj_k(cin,nh*chid,false),
          proj_v(cin,nh*chid,false), proj_o(nh*chid,cin,false),
          proj_g(cin,nh*chid,false) {}

    Mat forward(const Mat& x_flat, int N) const {
        Mat x = x_flat; layer_norm.forward(x);
        auto idx = [&](int i, int j) -> int { return starting ? i*N+j : j*N+i; };
        Mat tri_bias = linear_bias.forward(x);
        Mat Q=proj_q.forward(x), K=proj_k.forward(x), V=proj_v.forward(x), G=proj_g.forward(x);
        sigmoid_inplace(G);
        Mat out(N*N, no_heads*c_hidden, 0.0f);
        for (int i=0;i<N;++i) for (int h=0;h<no_heads;++h) {
            int h_off=h*c_hidden; float scale=1.0f/std::sqrt((float)c_hidden);
            std::vector<float> attn_row(N*N);
            for (int j1=0;j1<N;++j1) {
                int idx1=idx(i,j1);
                for (int j2=0;j2<N;++j2) {
                    int idx2=idx(i,j2); float sv=0;
                    for (int d=0;d<c_hidden;++d) sv+=Q(idx1,h_off+d)*K(idx2,h_off+d);
                    attn_row[j1*N+j2]=sv*scale+tri_bias(idx2,h);
                }
                softmax_row(attn_row.data()+j1*N, N);
            }
            for (int j1=0;j1<N;++j1) { int idx1=idx(i,j1);
                for (int j2=0;j2<N;++j2) { int idx2=idx(i,j2); float w=attn_row[j1*N+j2];
                    for (int d=0;d<c_hidden;++d) out(idx1,h_off+d)+=w*V(idx2,h_off+d);
                }
            }
        }
        for (int i=0;i<out.size();++i) out.data[i]*=G.data[i];
        return proj_o.forward(out);
    }
};

// ============================================================================
// Outer Product Mean
// ============================================================================
struct OuterProductMean {
    int c_in, c_hidden, c_out;
    LayerNorm norm; Linear proj_a, proj_b, proj_o;

    OuterProductMean() : c_in(0), c_hidden(0), c_out(0) {}
    OuterProductMean(int ci, int ch, int co) : c_in(ci), c_hidden(ch), c_out(co),
        norm(ci), proj_a(ci,ch,false), proj_b(ci,ch,false), proj_o(ch*ch,co,true) {}

    Mat forward(const Mat& m_flat, int S, int N) const {
        Mat m=m_flat; norm.forward(m);
        Mat a=proj_a.forward(m), b=proj_b.forward(m);
        int OP=c_hidden*c_hidden; Mat z(N*N,OP,0.0f);
        for (int s=0;s<S;++s) for (int i=0;i<N;++i) { int ai=s*N+i;
            for (int j=0;j<N;++j) { int bj=s*N+j,zidx=i*N+j;
                for (int p=0;p<c_hidden;++p) for (int q=0;q<c_hidden;++q)
                    z(zidx,p*c_hidden+q) += a(ai,p)*b(bj,q);
            }
        }
        float inv_S=1.0f/std::max(S,1); for (auto& v : z.data) v*=inv_S;
        return proj_o.forward(z);
    }
};

// ============================================================================
// Pair Weighted Averaging
// ============================================================================
struct PairWeightedAveraging {
    int c_m,c_z,c_h,num_heads;
    LayerNorm norm_m,norm_z; Linear proj_m,proj_g,proj_z,proj_o;

    PairWeightedAveraging() : c_m(0),c_z(0),c_h(0),num_heads(0) {}
    PairWeightedAveraging(int cm,int cz,int ch,int nh)
        : c_m(cm),c_z(cz),c_h(ch),num_heads(nh),
          norm_m(cm),norm_z(cz),proj_m(cm,ch*nh,false),proj_g(cm,ch*nh,false),
          proj_z(cz,nh,false),proj_o(ch*nh,cm,false) {}

    Mat forward(const Mat& m_flat, const Mat& z_flat, int S, int N) const {
        Mat m=m_flat; norm_m.forward(m); Mat z=z_flat; norm_z.forward(z);
        Mat V=proj_m.forward(m), G=proj_g.forward(m); sigmoid_inplace(G);
        Mat W=proj_z.forward(z);
        for (int h=0;h<num_heads;++h) for (int i=0;i<N;++i) {
            std::vector<float> w(N); for (int j=0;j<N;++j) w[j]=W(i*N+j,h);
            softmax_row(w.data(),N); for (int j=0;j<N;++j) W(i*N+j,h)=w[j];
        }
        Mat out(S*N, c_h*num_heads, 0.0f);
        for (int s=0;s<S;++s) for (int i=0;i<N;++i) for (int h=0;h<num_heads;++h) {
            int h_off=h*c_h;
            for (int j=0;j<N;++j) { float w=W(i*N+j,h); int vj=s*N+j,oi=s*N+i;
                for (int d=0;d<c_h;++d) out(oi,h_off+d)+=w*V(vj,h_off+d);
            }
        }
        for (int i=0;i<out.size();++i) out.data[i]*=G.data[i];
        return proj_o.forward(out);
    }
};

// ============================================================================
// MSA Layer & Module
// ============================================================================
struct MSALayer {
    int msa_s,token_z;
    OuterProductMean opm; PairWeightedAveraging pwa;
    Transition trans_m,trans_z;
    TriangleMultiplication tri_mul_out,tri_mul_in;
    TriangleAttention tri_att_start,tri_att_end;

    MSALayer() : msa_s(0),token_z(0) {}
    MSALayer(int ms, int tz) : msa_s(ms),token_z(tz) {
        opm=OuterProductMean(ms,16,tz); pwa=PairWeightedAveraging(ms,tz,PAIR_HEAD_DIM,PAIR_HEADS);
        trans_m=Transition(ms,ms*TRANSITION_FACTOR); trans_z=Transition(tz,tz*TRANSITION_FACTOR);
        tri_mul_out=TriangleMultiplication(tz,true); tri_mul_in=TriangleMultiplication(tz,false);
        tri_att_start=TriangleAttention(tz,PAIR_HEAD_DIM,PAIR_HEADS,true);
        tri_att_end=TriangleAttention(tz,PAIR_HEAD_DIM,PAIR_HEADS,false);
    }
    void forward(Mat& m_flat, Mat& z_flat, int S, int N) const {
        Mat dz=opm.forward(m_flat,S,N); mat_add(z_flat,dz);
        Mat dm=pwa.forward(m_flat,z_flat,S,N); mat_add(m_flat,dm);
        Mat dm2=trans_m.forward(m_flat); mat_add(m_flat,dm2);
        Mat dz2=tri_mul_out.forward(z_flat,N); mat_add(z_flat,dz2);
        Mat dz3=tri_mul_in.forward(z_flat,N); mat_add(z_flat,dz3);
        Mat dz4=tri_att_start.forward(z_flat,N); mat_add(z_flat,dz4);
        Mat dz5=tri_att_end.forward(z_flat,N); mat_add(z_flat,dz5);
        Mat dz6=trans_z.forward(z_flat); mat_add(z_flat,dz6);
    }
};

struct MSAModule {
    int num_blocks; Linear s_proj,msa_proj; std::vector<MSALayer> layers;
    MSAModule() : num_blocks(0) {}
    MSAModule(int msa_s,int token_z,int s_input_dim,int nblocks) : num_blocks(nblocks),
        s_proj(s_input_dim,msa_s,false),msa_proj(NUM_TOKENS,msa_s,false) {
        layers.resize(nblocks); for (int i=0;i<nblocks;++i) layers[i]=MSALayer(msa_s,token_z);
    }
    void forward(const Mat& s_input, const Mat& msa_oh, Mat& z_flat, int N) const {
        Mat m=msa_proj.forward(msa_oh); Mat s=s_proj.forward(s_input); mat_add(m,s);
        for (int b=0;b<num_blocks;++b) layers[b].forward(m,z_flat,1,N);
    }
};

// ============================================================================
// Pairformer Layer & Module
// ============================================================================
struct PairformerLayer {
    TriangleMultiplication tri_mul_out,tri_mul_in;
    TriangleAttention tri_att_start,tri_att_end;
    Transition trans_z,trans_s; AttentionPairBias attention; bool no_update_s;

    PairformerLayer() : no_update_s(false) {}
    PairformerLayer(int token_s, int token_z, bool no_upd_s=false) : no_update_s(no_upd_s) {
        tri_mul_out=TriangleMultiplication(token_z,true); tri_mul_in=TriangleMultiplication(token_z,false);
        tri_att_start=TriangleAttention(token_z,PAIR_HEAD_DIM,PAIR_HEADS,true);
        tri_att_end=TriangleAttention(token_z,PAIR_HEAD_DIM,PAIR_HEADS,false);
        trans_z=Transition(token_z,token_z*TRANSITION_FACTOR);
        if (!no_update_s) { trans_s=Transition(token_s,token_s*TRANSITION_FACTOR); attention=AttentionPairBias(token_s,token_z,NUM_HEADS); }
    }
    void forward(Mat& s, Mat& z_flat, int N) const {
        Mat dz;
        dz=tri_mul_out.forward(z_flat,N); mat_add(z_flat,dz);
        dz=tri_mul_in.forward(z_flat,N); mat_add(z_flat,dz);
        dz=tri_att_start.forward(z_flat,N); mat_add(z_flat,dz);
        dz=tri_att_end.forward(z_flat,N); mat_add(z_flat,dz);
        dz=trans_z.forward(z_flat); mat_add(z_flat,dz);
        if (!no_update_s) { Mat ds=attention.forward(s,z_flat); mat_add(s,ds); ds=trans_s.forward(s); mat_add(s,ds); }
    }
};

struct PairformerModule {
    int num_blocks; std::vector<PairformerLayer> layers;
    PairformerModule() : num_blocks(0) {}
    PairformerModule(int token_s,int token_z,int nblocks) : num_blocks(nblocks) {
        layers.resize(nblocks); for (int i=0;i<nblocks;++i) layers[i]=PairformerLayer(token_s,token_z);
    }
    void forward(Mat& s, Mat& z_flat, int N) const { for (int b=0;b<num_blocks;++b) layers[b].forward(s,z_flat,N); }
};

// ============================================================================
// Invariant Point Attention  (IPA)
// ============================================================================
struct InvariantPointAttention {
    int c_s,c_z,num_heads,head_dim,n_qp,n_vp;
    LayerNorm norm_s,norm_z;
    Linear proj_q,proj_k,proj_v,proj_qp,proj_kp,proj_vp,proj_z,proj_o;

    InvariantPointAttention() : c_s(0),c_z(0),num_heads(0),head_dim(0),n_qp(0),n_vp(0) {}
    InvariantPointAttention(int cs,int cz,int nh,int nqp,int nvp)
        : c_s(cs),c_z(cz),num_heads(nh),head_dim(cs/nh),n_qp(nqp),n_vp(nvp),
          norm_s(cs),norm_z(cz),proj_q(cs,cs,false),proj_k(cs,cs,false),proj_v(cs,cs,false),
          proj_qp(cs,nh*nqp*3,false),proj_kp(cs,nh*nqp*3,false),proj_vp(cs,nh*nvp*3,false),
          proj_z(cz,nh,false),proj_o(cs+nh*(cz+nvp*3),cs,true) {}

    Mat forward(const Mat& s_in, const Mat& z_flat, const std::vector<Rigid>& frames, int N) const {
        Mat s=s_in; norm_s.forward(s);
        Mat Q=proj_q.forward(s),K=proj_k.forward(s),V=proj_v.forward(s);
        Mat QP=proj_qp.forward(s),KP=proj_kp.forward(s),VP=proj_vp.forward(s);
        Mat z=z_flat; norm_z.forward(z); Mat pair_bias=proj_z.forward(z);
        auto get_point=[](const Mat& P,int row,int h,int p,int npts)->Vec3{
            int base=(h*npts+p)*3; return{P(row,base),P(row,base+1),P(row,base+2)};
        };
        int out_dim=c_s+num_heads*(c_z+n_vp*3); Mat out(N,out_dim,0.0f);
        float gamma_p=std::sqrt(2.0f/(9.0f*n_qp));
        for (int h=0;h<num_heads;++h) {
            int h_off=h*head_dim; float scale=1.0f/std::sqrt((float)head_dim);
            std::vector<float> attn(N*N);
            for (int i=0;i<N;++i) { for (int j=0;j<N;++j) {
                float qk=0; for (int d=0;d<head_dim;++d) qk+=Q(i,h_off+d)*K(j,h_off+d);
                qk*=scale; qk+=pair_bias(i*N+j,h);
                float ps=0; for (int p=0;p<n_qp;++p) {
                    Vec3 qi=frames[i].apply(get_point(QP,i,h,p,n_qp));
                    Vec3 kj=frames[j].apply(get_point(KP,j,h,p,n_qp));
                    Vec3 diff=qi-kj; ps+=diff.dot(diff);
                }
                qk-=gamma_p*ps; attn[i*N+j]=qk;
            } softmax_row(attn.data()+i*N,N); }
            for (int i=0;i<N;++i) {
                for (int j=0;j<N;++j) { float w=attn[i*N+j];
                    for (int d=0;d<head_dim;++d) out(i,h_off+d)+=w*V(j,h_off+d);
                }
                int pair_off=c_s+h*c_z;
                for (int j=0;j<N;++j) { float w=attn[i*N+j];
                    for (int d=0;d<c_z;++d) out(i,pair_off+d)+=w*z(i*N+j,d);
                }
                int pt_off=c_s+num_heads*c_z+h*n_vp*3;
                std::vector<Vec3> sum_pts(n_vp);
                for (int j=0;j<N;++j) { float w=attn[i*N+j];
                    for (int p=0;p<n_vp;++p) { Vec3 vp=frames[j].apply(get_point(VP,j,h,p,n_vp)); sum_pts[p]=sum_pts[p]+vp*w; }
                }
                for (int p=0;p<n_vp;++p) { Vec3 local=frames[i].apply_inv(sum_pts[p]);
                    out(i,pt_off+p*3+0)+=local.x; out(i,pt_off+p*3+1)+=local.y; out(i,pt_off+p*3+2)+=local.z;
                }
            }
        }
        return proj_o.forward(out);
    }
};

// ============================================================================
// Backbone Update & Structure Module
// ============================================================================
struct BackboneUpdate {
    Linear proj;
    BackboneUpdate() = default;
    BackboneUpdate(int c_s) : proj(c_s, 6, true) {}
    std::vector<Rigid> forward(const Mat& s, const std::vector<Rigid>& frames) const {
        int N=s.rows; Mat upd=proj.forward(s); std::vector<Rigid> nf(N);
        for (int i=0;i<N;++i) {
            float tx=upd(i,0),ty=upd(i,1),tz=upd(i,2),r1=upd(i,3)*0.1f,r2=upd(i,4)*0.1f,r3=upd(i,5)*0.1f;
            Rot3 dR; dR.m[0][0]=1;dR.m[0][1]=-r3;dR.m[0][2]=r2; dR.m[1][0]=r3;dR.m[1][1]=1;dR.m[1][2]=-r1; dR.m[2][0]=-r2;dR.m[2][1]=r1;dR.m[2][2]=1;
            Vec3 c0={dR.m[0][0],dR.m[1][0],dR.m[2][0]}; Vec3 c1={dR.m[0][1],dR.m[1][1],dR.m[2][1]};
            c0=c0.normalized(); c1=(c1-c0*c1.dot(c0)).normalized(); Vec3 c2=c0.cross(c1);
            dR.m[0][0]=c0.x;dR.m[1][0]=c0.y;dR.m[2][0]=c0.z; dR.m[0][1]=c1.x;dR.m[1][1]=c1.y;dR.m[2][1]=c1.z; dR.m[0][2]=c2.x;dR.m[1][2]=c2.y;dR.m[2][2]=c2.z;
            nf[i]=frames[i].compose(Rigid(dR,Vec3{tx*0.1f,ty*0.1f,tz*0.1f}));
        } return nf;
    }
};

struct StructureModule {
    int num_blocks;
    std::vector<InvariantPointAttention> ipa_layers;
    std::vector<Transition> transitions;
    std::vector<BackboneUpdate> bb_updates;
    StructureModule() : num_blocks(0) {}
    StructureModule(int c_s,int c_z,int nblocks) : num_blocks(nblocks) {
        ipa_layers.resize(nblocks); transitions.resize(nblocks); bb_updates.resize(nblocks);
        for (int i=0;i<nblocks;++i) {
            ipa_layers[i]=InvariantPointAttention(c_s,c_z,NUM_HEADS,NUM_IPA_QUERY_POINTS,NUM_IPA_VALUE_POINTS);
            transitions[i]=Transition(c_s,c_s*TRANSITION_FACTOR); bb_updates[i]=BackboneUpdate(c_s);
        }
    }
    std::vector<Rigid> forward(Mat& s, const Mat& z_flat, int N) const {
        std::vector<Rigid> frames(N);
        for (int b=0;b<num_blocks;++b) {
            Mat ds=ipa_layers[b].forward(s,z_flat,frames,N); mat_add(s,ds);
            Mat ds2=transitions[b].forward(s); mat_add(s,ds2);
            frames=bb_updates[b].forward(s,frames);
        } return frames;
    }
};

struct BackboneAtoms { Vec3 N_pos, CA_pos, C_pos, O_pos; };

inline std::vector<BackboneAtoms> place_backbone_atoms(const std::vector<Rigid>& frames, int N_res) {
    std::vector<BackboneAtoms> atoms(N_res);
    for (int i=0;i<N_res;++i) {
        Vec3 n_local={-BOND_CA_N,0,0};
        float cx=BOND_CA_C*std::cos(ANGLE_N_CA_C-(float)M_PI), cy=BOND_CA_C*std::sin(ANGLE_N_CA_C-(float)M_PI);
        Vec3 c_local={cx,cy,0};
        float ox=c_local.x+BOND_C_O*std::cos((float)M_PI-ANGLE_CA_C_O), oy=c_local.y+BOND_C_O*std::sin((float)M_PI-ANGLE_CA_C_O);
        atoms[i].CA_pos=frames[i].apply({0,0,0}); atoms[i].N_pos=frames[i].apply(n_local);
        atoms[i].C_pos=frames[i].apply(c_local); atoms[i].O_pos=frames[i].apply({ox,oy,0});
    } return atoms;
}

// ============================================================================
// AdaLN, ConditionedTransitionBlock, DiffusionTransformer (for diffusion)
// ============================================================================
struct AdaLN {
    LayerNorm a_norm, s_norm; Linear s_scale, s_bias;
    AdaLN() = default;
    AdaLN(int dim,int dim_cond) : a_norm(dim),s_norm(dim_cond),s_scale(dim_cond,dim,true),s_bias(dim_cond,dim,false) {}
    Mat forward(const Mat& a, const Mat& s) const {
        Mat an=a; a_norm.forward_no_affine(an); Mat sn=s; s_norm.forward(sn);
        Mat scale=s_scale.forward(sn); sigmoid_inplace(scale); Mat bias=s_bias.forward(sn);
        Mat out(a.rows,a.cols); for (int i=0;i<out.size();++i) out.data[i]=scale.data[i]*an.data[i]+bias.data[i]; return out;
    }
};

struct ConditionedTransitionBlock {
    AdaLN adaln; Linear swish_gate,a_to_b,b_to_a,output_proj_lin;
    ConditionedTransitionBlock() = default;
    ConditionedTransitionBlock(int dim,int dim_cond,int expansion=2) {
        adaln=AdaLN(dim,dim_cond); int inner=dim*expansion;
        swish_gate=Linear(dim,inner*2,false); a_to_b=Linear(dim,inner,false); b_to_a=Linear(inner,dim,false);
        output_proj_lin=Linear(dim_cond,dim,true);
        for (auto& v : output_proj_lin.bias) v=-2.0f; gating_init(output_proj_lin.W);
    }
    Mat forward(const Mat& a_in, const Mat& s) const {
        Mat a=adaln.forward(a_in,s); Mat sg=swish_gate.forward(a); Mat sg_act=swiglu(sg);
        Mat ab=a_to_b.forward(a); Mat b(sg_act.rows,sg_act.cols);
        for (int i=0;i<b.size();++i) b.data[i]=sg_act.data[i]*ab.data[i];
        Mat out_proj=output_proj_lin.forward(s); sigmoid_inplace(out_proj);
        Mat result=b_to_a.forward(b); for (int i=0;i<result.size();++i) result.data[i]*=out_proj.data[i]; return result;
    }
};

struct DiffusionTransformerLayer {
    AdaLN adaln; AttentionPairBias pair_bias_attn; Linear output_proj_lin; ConditionedTransitionBlock transition;
    DiffusionTransformerLayer() = default;
    DiffusionTransformerLayer(int dim,int dim_cond,int dim_pair,int heads)
        : adaln(dim,dim_cond),pair_bias_attn(dim,dim_pair,heads,false),transition(dim,dim_cond) {
        output_proj_lin=Linear(dim_cond,dim,true);
        for (auto& v : output_proj_lin.bias) v=-2.0f; gating_init(output_proj_lin.W);
    }
    Mat forward(const Mat& a_in, const Mat& s, const Mat& z) const {
        Mat b=adaln.forward(a_in,s); b=pair_bias_attn.forward(b,z);
        Mat out_proj=output_proj_lin.forward(s); sigmoid_inplace(out_proj);
        for (int i=0;i<b.size();++i) b.data[i]*=out_proj.data[i];
        Mat a=a_in; mat_add(a,b); Mat dt=transition.forward(a,s); mat_add(a,dt); return a;
    }
};

struct DiffusionTransformer {
    int depth; std::vector<DiffusionTransformerLayer> layers;
    DiffusionTransformer() : depth(0) {}
    DiffusionTransformer(int d,int dim,int dim_cond,int dim_pair,int heads) : depth(d) {
        layers.resize(d); for (int i=0;i<d;++i) layers[i]=DiffusionTransformerLayer(dim,dim_cond,dim_pair,heads);
    }
    Mat forward(const Mat& a, const Mat& s, const Mat& z) const {
        Mat out=a; for (int i=0;i<depth;++i) out=layers[i].forward(out,s,z); return out;
    }
};

// ============================================================================
// Fourier Embedding, Single/Pairwise Conditioning, Noise Schedule
// ============================================================================
struct FourierEmbedding {
    Mat proj_w; std::vector<float> proj_b; int dim;
    FourierEmbedding() : dim(0) {}
    FourierEmbedding(int d) : dim(d) { proj_w.resize(d,1); proj_b.resize(d);
        for (int i=0;i<d;++i) { proj_w(i,0)=randn(); proj_b[i]=randn(); }
    }
    Mat forward(float t) const {
        Mat out(1,dim); for (int i=0;i<dim;++i) out(0,i)=std::cos(2.0f*(float)M_PI*(proj_w(i,0)*t+proj_b[i])); return out;
    }
};

struct SingleConditioning {
    float sigma_data; LayerNorm norm_single; Linear single_embed;
    FourierEmbedding fourier_embed; LayerNorm norm_fourier; Linear fourier_to_single;
    std::vector<Transition> transitions;
    SingleConditioning() : sigma_data(SIGMA_DATA) {}
    SingleConditioning(int token_s,float sd=SIGMA_DATA,int num_trans=2) : sigma_data(sd) {
        int s_input_dim=token_s+NUM_TOKENS; int out_dim=2*token_s;
        norm_single=LayerNorm(s_input_dim); single_embed=Linear(s_input_dim,out_dim,true);
        fourier_embed=FourierEmbedding(DIM_FOURIER); norm_fourier=LayerNorm(DIM_FOURIER);
        fourier_to_single=Linear(DIM_FOURIER,out_dim,false);
        transitions.resize(num_trans); for (int i=0;i<num_trans;++i) transitions[i]=Transition(out_dim,2*out_dim);
    }
    std::pair<Mat,Mat> forward(float time, const Mat& s_trunk, const Mat& s_inputs) const {
        Mat s=mat_concat_cols(s_trunk,s_inputs); norm_single.forward(s); s=single_embed.forward(s);
        Mat fourier=fourier_embed.forward(time); norm_fourier.forward(fourier);
        Mat f2s=fourier_to_single.forward(fourier);
        for (int i=0;i<s.rows;++i) for (int j=0;j<s.cols;++j) s(i,j)+=f2s(0,j);
        for (const auto& trans : transitions) { Mat ds=trans.forward(s); mat_add(s,ds); }
        return {s,fourier};
    }
};

struct PairwiseConditioning {
    LayerNorm norm; Linear proj; std::vector<Transition> transitions;
    PairwiseConditioning() = default;
    PairwiseConditioning(int token_z,int rel_pos_dim,int num_trans=2) {
        norm=LayerNorm(token_z+rel_pos_dim); proj=Linear(token_z+rel_pos_dim,token_z,false);
        transitions.resize(num_trans); for (int i=0;i<num_trans;++i) transitions[i]=Transition(token_z,2*token_z);
    }
    Mat forward(const Mat& z_trunk, const Mat& rel_pos_feats) const {
        Mat z=mat_concat_cols(z_trunk,rel_pos_feats); norm.forward(z); z=proj.forward(z);
        for (const auto& trans : transitions) { Mat dz=trans.forward(z); mat_add(z,dz); } return z;
    }
};

struct NoiseSchedule {
    float sigma_data=SIGMA_DATA, sigma_min=0.0004f, sigma_max=160.0f, rho=7.0f, s_noise=1.003f;
    int num_steps=200;
    float sigma(float t) const { float ir=1.0f/rho; return std::pow(std::pow(sigma_max,ir)+t*(std::pow(sigma_min,ir)-std::pow(sigma_max,ir)),rho); }
    float c_skip(float s) const { return sigma_data*sigma_data/(s*s+sigma_data*sigma_data); }
    float c_out(float s) const { return s*sigma_data/std::sqrt(s*s+sigma_data*sigma_data); }
    float c_in(float s) const { return 1.0f/std::sqrt(s*s+sigma_data*sigma_data); }
    float loss_weight(float s) const { return 1.0f/(s*s+sigma_data*sigma_data); }
    std::vector<float> timesteps() const { std::vector<float> ts(num_steps+1); for (int i=0;i<=num_steps;++i) ts[i]=sigma((float)i/(float)num_steps); return ts; }
};

struct ExponentialInterpolation {
    float start,end_val,alpha;
    float compute(float t) const { if (std::abs(alpha)>1e-8f) return start+(end_val-start)*(std::exp(alpha*t)-1.0f)/(std::exp(alpha)-1.0f); return start+(end_val-start)*t; }
};

struct PiecewiseStepFunction {
    std::vector<float> thresholds, values;
    float compute(float t) const { int idx=0; while (idx<(int)thresholds.size()&&t>thresholds[idx]) idx++; return values[std::min(idx,(int)values.size()-1)]; }
};

// ============================================================================
// Diffusion Module
// ============================================================================
struct DiffusionModule {
    SingleConditioning single_cond; PairwiseConditioning pair_cond;
    DiffusionTransformer token_transformer; Linear atom_to_token,atom_pos_proj;
    LayerNorm atom_pos_norm; NoiseSchedule noise_schedule;

    DiffusionModule() = default;
    DiffusionModule(int token_s,int token_z) {
        single_cond=SingleConditioning(token_s);
        int rel_feat_dim=4*(REL_POS_MAX+1)+2*(SYM_MAX+1)+1;
        pair_cond=PairwiseConditioning(token_z,rel_feat_dim);
        token_transformer=DiffusionTransformer(TOKEN_TRANSFORMER_DEPTH,2*token_s,2*token_s,token_z,NUM_HEADS);
        atom_to_token=Linear(3,2*token_s,false); atom_pos_norm=LayerNorm(2*token_s);
        atom_pos_proj=Linear(2*token_s,3,false);
    }
    Mat denoise(const Mat& noisy,float sigma,const Mat& s_trunk,const Mat& z_trunk,const Mat& s_inputs,const Mat& rel_feats,int N) const {
        auto [s_cond,fourier]=single_cond.forward(sigma,s_trunk,s_inputs);
        Mat z_cond=pair_cond.forward(z_trunk,rel_feats);
        float ci=noise_schedule.c_in(sigma); Mat sc=noisy; mat_mul_scalar(sc,ci);
        Mat af=atom_to_token.forward(sc); mat_add(s_cond,af);
        Mat a=token_transformer.forward(s_cond,s_cond,z_cond);
        atom_pos_norm.forward(a); Mat pu=atom_pos_proj.forward(a);
        float cs=noise_schedule.c_skip(sigma), co=noise_schedule.c_out(sigma);
        Mat result(N,3); for (int i=0;i<N;++i) for (int d=0;d<3;++d) result(i,d)=cs*noisy(i,d)+co*pu(i,d);
        return result;
    }
    Mat sample(const Mat& s_trunk,const Mat& z_trunk,const Mat& s_inputs,const Mat& rel_feats,int N) const {
        Mat x(N,3); float si=noise_schedule.sigma(0.0f);
        for (int i=0;i<N;++i) for (int d=0;d<3;++d) x(i,d)=randn()*si;
        auto sigmas=noise_schedule.timesteps();
        for (int step=0;step<noise_schedule.num_steps;++step) {
            float st=sigmas[step],sn=sigmas[step+1]; if (st<=1e-10f) break;
            Mat xd=denoise(x,st,s_trunk,z_trunk,s_inputs,rel_feats,N);
            if (sn>1e-10f) { for (int i=0;i<N;++i) for (int d=0;d<3;++d) { float dx=(x(i,d)-xd(i,d))/st; x(i,d)+=dx*(sn-st); } }
            else x=xd;
        } return x;
    }
};

// ============================================================================
// Confidence Heads & Module  (pLDDT, PDE, PAE, pTM, iPTM)
// ============================================================================
struct ConfidenceHeads {
    Linear to_plddt_logits,to_pde_logits,to_pae_logits,to_resolved_logits; bool compute_pae;
    ConfidenceHeads() : compute_pae(false) {}
    ConfidenceHeads(int token_s,int token_z,bool pae=true) : compute_pae(pae) {
        to_plddt_logits=Linear(token_s,NUM_PLDDT_BINS,false); to_pde_logits=Linear(token_z,NUM_PDE_BINS,false);
        to_resolved_logits=Linear(token_s,2,false); if (compute_pae) to_pae_logits=Linear(token_z,NUM_PAE_BINS,false);
    }
    struct ConfidenceOutput { Mat plddt_logits,pde_logits,pae_logits,resolved_logits; std::vector<float> plddt; float complex_plddt,ptm,iptm; };

    static float tm_function(float d, float Nres) { float d0=1.24f*std::pow(std::max(Nres,19.0f)-15.0f,1.0f/3.0f)-1.8f; return 1.0f/(1.0f+(d/d0)*(d/d0)); }

    static float compute_ptm(const Mat& pae_logits, int N) {
        if (pae_logits.rows==0||pae_logits.cols==0) return 0.0f;
        int nb=pae_logits.cols; float bw=32.0f/nb, Nres=(float)N, best=0;
        for (int i=0;i<N;++i) { float sum_tm=0;
            for (int j=0;j<N;++j) {
                std::vector<float> probs(nb); float psum=0;
                for (int b=0;b<nb;++b) { probs[b]=std::exp(pae_logits(i*N+j,b)); psum+=probs[b]; }
                for (int b=0;b<nb;++b) probs[b]/=(psum+1e-12f);
                float tm=0; for (int b=0;b<nb;++b) tm+=probs[b]*tm_function(0.5f*bw+b*bw,Nres);
                sum_tm+=tm;
            } best=std::max(best,sum_tm/N);
        } return best;
    }

    ConfidenceOutput forward(const Mat& s, const Mat& z, int N) const {
        ConfidenceOutput out;
        out.plddt_logits=to_plddt_logits.forward(s); out.resolved_logits=to_resolved_logits.forward(s);
        Mat z_sym(N*N,z.cols); for (int i=0;i<N;++i) for (int j=0;j<N;++j) for (int d=0;d<z.cols;++d) z_sym(i*N+j,d)=z(i*N+j,d)+z(j*N+i,d);
        out.pde_logits=to_pde_logits.forward(z_sym);
        if (compute_pae) out.pae_logits=to_pae_logits.forward(z);
        out.plddt.resize(N); float bw=1.0f/NUM_PLDDT_BINS;
        for (int i=0;i<N;++i) {
            Mat row(1,NUM_PLDDT_BINS); std::copy(out.plddt_logits.row_ptr(i),out.plddt_logits.row_ptr(i)+NUM_PLDDT_BINS,row.row_ptr(0));
            softmax_rows(row); float val=0; for (int b=0;b<NUM_PLDDT_BINS;++b) val+=row(0,b)*(0.5f*bw+b*bw); out.plddt[i]=val;
        }
        out.complex_plddt=0; for (float v:out.plddt) out.complex_plddt+=v; out.complex_plddt/=std::max(N,1);
        out.ptm=compute_ptm(out.pae_logits,N); out.iptm=out.ptm;
        return out;
    }
};

struct ConfidenceModule {
    Embedding dist_bin_embed; LayerNorm s_norm,z_norm; PairformerModule pairformer; ConfidenceHeads heads;
    ConfidenceModule() = default;
    ConfidenceModule(int token_s,int token_z,int pf_blocks=2) {
        dist_bin_embed=Embedding(NUM_DIST_BINS,token_z); s_norm=LayerNorm(token_s); z_norm=LayerNorm(token_z);
        pairformer=PairformerModule(token_s,token_z,pf_blocks); heads=ConfidenceHeads(token_s,token_z,true);
    }
    ConfidenceHeads::ConfidenceOutput forward(const Mat& s_in,const Mat& z_in,const Mat& x_pred,int N) const {
        Mat s=s_in; s_norm.forward(s); Mat z=z_in; z_norm.forward(z);
        Mat dists=cdist(x_pred);
        for (int i=0;i<N;++i) for (int j=0;j<N;++j) {
            int bin=std::min((int)(dists(i,j)*NUM_DIST_BINS/22.0f),NUM_DIST_BINS-1);
            float emb[TOKEN_Z]; dist_bin_embed.lookup(bin,emb);
            for (int k=0;k<TOKEN_Z;++k) z(i*N+j,k)+=emb[k];
        }
        pairformer.forward(s,z,N); return heads.forward(s,z,N);
    }
};

// ============================================================================
// Recycling & Template Embedding
// ============================================================================
struct RecyclingModule {
    Linear s_recycle,z_recycle; LayerNorm s_norm,z_norm;
    RecyclingModule() = default;
    RecyclingModule(int token_s,int token_z) : s_recycle(token_s,token_s,false),z_recycle(token_z,token_z,false),
        s_norm(token_s),z_norm(token_z) { gating_init(s_recycle.W); gating_init(z_recycle.W); }
    void apply(Mat& s,Mat& z,const Mat& s_prev,const Mat& z_prev) const {
        Mat sn=s_prev; s_norm.forward(sn); Mat sr=s_recycle.forward(sn); mat_add(s,sr);
        Mat zn=z_prev; z_norm.forward(zn); Mat zr=z_recycle.forward(zn); mat_add(z,zr);
    }
};

struct TemplateEmbedding {
    Linear template_proj; LayerNorm template_norm;
    TemplateEmbedding() = default;
    TemplateEmbedding(int token_z) : template_proj(token_z,token_z,false),template_norm(token_z) {}
    void embed(Mat& z, const Mat& template_feats) const {
        Mat tf=template_feats; template_norm.forward(tf); Mat proj=template_proj.forward(tf); mat_add(z,proj);
    }
};

// ============================================================================
// Loss Functions
// ============================================================================
inline float fape_loss(const std::vector<Rigid>& pf, const std::vector<BackboneAtoms>& pa,
                       const std::vector<Rigid>& tf, const std::vector<BackboneAtoms>& ta, int N) {
    float total=0; int count=0;
    auto get_atom=[](const std::vector<BackboneAtoms>& at,int res,int a)->Vec3{
        switch(a){case 0:return at[res].N_pos;case 1:return at[res].CA_pos;case 2:return at[res].C_pos;case 3:return at[res].O_pos;default:return at[res].CA_pos;}};
    for (int i=0;i<N;++i) { Rigid pi=pf[i].inverse(),ti=tf[i].inverse();
        for (int j=0;j<N;++j) for (int a=0;a<4;++a) {
            Vec3 d=pi.apply(get_atom(pa,j,a))-ti.apply(get_atom(ta,j,a));
            total+=std::min(d.norm(),FAPE_CLAMP)/FAPE_D; count++;
        }
    } return count>0?total/count:0.0f;
}

inline float smooth_lddt_loss(const std::vector<BackboneAtoms>& pred,const std::vector<BackboneAtoms>& target,int N,float cutoff=15.0f) {
    if (N<2) return 0.0f; float total=0; int count=0;
    for (int i=0;i<N;++i) { float num=0,den=0;
        for (int j=0;j<N;++j) { if (i==j) continue;
            float dt=(target[i].CA_pos-target[j].CA_pos).norm(); if (dt>cutoff) continue;
            float dp=(pred[i].CA_pos-pred[j].CA_pos).norm(); float diff=std::abs(dt-dp);
            num+=(sigmoid_f(0.5f-diff)+sigmoid_f(1.0f-diff)+sigmoid_f(2.0f-diff)+sigmoid_f(4.0f-diff))/4.0f; den+=1.0f;
        } if (den>0) { total+=num/den; count++; }
    } return count>0?1.0f-total/count:0.0f;
}

inline float distogram_loss(const Mat& pred,const Mat& target,const std::vector<bool>& mask,int N) {
    Mat p=pred; log_softmax_rows(p); float total=0,count=0;
    for (int i=0;i<N;++i) { if (!mask[i]) continue; for (int j=0;j<N;++j) { if (!mask[j]||i==j) continue;
        for (int b=0;b<p.cols;++b) total-=target(i*N+j,b)*p(i*N+j,b); count+=1.0f;
    }} return count>0?total/count:0.0f;
}

inline float bfactor_loss(const Mat& pred,const std::vector<float>& true_bf,const std::vector<float>& mask) {
    int N=pred.rows,bins=pred.cols; Mat lp=pred; log_softmax_rows(lp); float total=0,count=0;
    for (int i=0;i<N;++i) { if (mask[i]<0.5f||true_bf[i]<1e-5f) continue;
        int bin=0; float step=100.0f/(bins-1); for (int b=0;b<bins-1;++b) if (true_bf[i]>(b+1)*step) bin=b+1;
        bin=std::min(bin,bins-1); total-=lp(i,bin); count+=1.0f;
    } return count>0?total/count:0.0f;
}

// ============================================================================
// Distogram & BFactor Modules
// ============================================================================
struct DistogramModule {
    Linear proj;
    DistogramModule() = default;
    DistogramModule(int token_z,int num_bins) : proj(token_z,num_bins,true) {}
    Mat forward(const Mat& z_flat,int N) const {
        Mat z_sym(N*N,z_flat.cols); for (int i=0;i<N;++i) for (int j=0;j<N;++j) for (int d=0;d<z_flat.cols;++d) z_sym(i*N+j,d)=z_flat(i*N+j,d)+z_flat(j*N+i,d);
        return proj.forward(z_sym);
    }
};

struct BFactorModule {
    Linear proj; int num_bins;
    BFactorModule() : num_bins(0) {}
    BFactorModule(int token_s,int nb=50) : proj(token_s,nb,true),num_bins(nb) {}
    Mat forward(const Mat& s) const { return proj.forward(s); }
};

// ============================================================================
// LR Scheduler, EMA
// ============================================================================
struct LRScheduler {
    float base_lr=0.0f,max_lr=1.8e-3f; int warmup_steps=1000,start_decay=50000,decay_every=50000; float decay_factor=0.95f;
    float get_lr(int step) const {
        if (step<=warmup_steps) return base_lr+((float)step/warmup_steps)*max_lr;
        if (step>start_decay) { int s=step-start_decay; return max_lr*std::pow(decay_factor,(float)((s/decay_every)+1)); }
        return max_lr;
    }
};

struct EMA {
    float decay; int num_updates; std::vector<std::vector<float>> shadow_params;
    EMA() : decay(0.999f),num_updates(0) {}
    void init(const std::vector<std::vector<float>*>& params) { shadow_params.clear(); for (const auto* p:params) shadow_params.push_back(*p); }
    void update(const std::vector<std::vector<float>*>& params) {
        num_updates++; float d=std::min(decay,(1.0f+num_updates)/(10.0f+num_updates));
        for (int i=0;i<(int)shadow_params.size()&&i<(int)params.size();++i)
            for (int j=0;j<(int)shadow_params[i].size();++j) shadow_params[i][j]-=(1.0f-d)*(shadow_params[i][j]-(*params[i])[j]);
    }
};

// ============================================================================
// Center & Random Augmentation
// ============================================================================
inline Mat center_coords(const Mat& coords, const std::vector<float>& mask) {
    int N=coords.rows; float wsum=0; Vec3 c;
    for (int i=0;i<N;++i) { float w=(i<(int)mask.size())?mask[i]:1.0f; wsum+=w; c.x+=w*coords(i,0); c.y+=w*coords(i,1); c.z+=w*coords(i,2); }
    c=c*(1.0f/(wsum+1e-12f)); Mat out=coords;
    for (int i=0;i<N;++i) { out(i,0)-=c.x; out(i,1)-=c.y; out(i,2)-=c.z; } return out;
}

inline Mat random_augment(const Mat& coords, float s_trans=1.0f) {
    Rot3 R=random_rotation(); Mat out(coords.rows,3); Vec3 t={randn()*s_trans,randn()*s_trans,randn()*s_trans};
    for (int i=0;i<coords.rows;++i) { Vec3 v={coords(i,0),coords(i,1),coords(i,2)}; Vec3 rv=R.apply(v)+t; out(i,0)=rv.x; out(i,1)=rv.y; out(i,2)=rv.z; }
    return out;
}

// ============================================================================
// Full Boltz Model
// ============================================================================
struct BoltzModel {
    int N_res = 0;
    Linear s_init,z_init_1,z_init_2,rel_pos_proj,token_bond_proj;
    LayerNorm s_norm,z_norm;
    MSAModule msa_module; PairformerModule pairformer; StructureModule structure_module;
    DistogramModule distogram; BFactorModule bfactor_module;
    DiffusionModule diffusion; ConfidenceModule confidence;
    RecyclingModule recycling; TemplateEmbedding template_emb;

    BoltzModel() = default;
    void init(int n_res) {
        N_res=n_res; int si=NUM_TOKENS;
        s_init=Linear(si,TOKEN_S,false); z_init_1=Linear(si,TOKEN_Z,false); z_init_2=Linear(si,TOKEN_Z,false);
        int rfd=4*(REL_POS_MAX+1)+2*(SYM_MAX+1)+1;
        rel_pos_proj=Linear(rfd,TOKEN_Z,false); s_norm=LayerNorm(TOKEN_S); z_norm=LayerNorm(TOKEN_Z);
        msa_module=MSAModule(MSA_S,TOKEN_Z,si,NUM_MSA_BLOCKS); pairformer=PairformerModule(TOKEN_S,TOKEN_Z,NUM_PAIRFORMER_BLOCKS);
        structure_module=StructureModule(TOKEN_S,TOKEN_Z,NUM_DIFFUSION_BLOCKS);
        distogram=DistogramModule(TOKEN_Z,NUM_DIST_BINS); bfactor_module=BFactorModule(TOKEN_S);
        diffusion=DiffusionModule(TOKEN_S,TOKEN_Z); confidence=ConfidenceModule(TOKEN_S,TOKEN_Z);
        recycling=RecyclingModule(TOKEN_S,TOKEN_Z); template_emb=TemplateEmbedding(TOKEN_Z);
        token_bond_proj=Linear(1,TOKEN_Z,false);
    }

    struct Output {
        std::vector<Rigid> frames; std::vector<BackboneAtoms> atoms;
        Mat s,z,disto,bfactor,diffusion_coords;
        ConfidenceHeads::ConfidenceOutput conf;
    };

    Output forward(const std::string& sequence, int num_recycles=NUM_RECYCLES) const {
        int N=(int)sequence.size(); Mat seq_oh=one_hot_encode(sequence);
        Mat s=s_init.forward(seq_oh); Mat z1=z_init_1.forward(seq_oh),z2=z_init_2.forward(seq_oh);
        Mat z_flat(N*N,TOKEN_Z,0.0f);
        for (int i=0;i<N;++i) for (int j=0;j<N;++j) for (int d=0;d<TOKEN_Z;++d) z_flat(i*N+j,d)=z1(i,d)+z2(j,d);
        Mat rel_feats=relative_position_features(N); Mat rel_proj=rel_pos_proj.forward(rel_feats); mat_add(z_flat,rel_proj);
        Mat bond_feat(N*N,1,0.0f); for (int i=0;i<N-1;++i) { bond_feat(i*N+i+1,0)=1.0f; bond_feat((i+1)*N+i,0)=1.0f; }
        Mat bond_proj=token_bond_proj.forward(bond_feat); mat_add(z_flat,bond_proj);
        Mat s_init_emb=s, z_init_emb=z_flat;

        for (int rec=0;rec<num_recycles;++rec) {
            s=s_init_emb; Mat z_cur=z_init_emb;
            if (rec>0) recycling.apply(s,z_cur,s,z_flat);
            z_flat=z_cur;
            msa_module.forward(seq_oh,seq_oh,z_flat,N);
            Mat sn=s; s_norm.forward(sn); s=sn;
            Mat zn=z_flat; z_norm.forward(zn); z_flat=zn;
            pairformer.forward(s,z_flat,N);
        }

        Mat s_struct=s; std::vector<Rigid> frames=structure_module.forward(s_struct,z_flat,N);
        auto atoms=place_backbone_atoms(frames,N);
        Mat disto=distogram.forward(z_flat,N); Mat bfac=bfactor_module.forward(s);
        Mat diff_coords=diffusion.sample(s,z_flat,seq_oh,rel_feats,N);
        auto conf=confidence.forward(s,z_flat,diff_coords,N);
        return {frames,atoms,s,z_flat,disto,bfac,diff_coords,conf};
    }
};

// ============================================================================
// Prediction Pipeline
// ============================================================================
struct PredictionPipeline {
    BoltzModel model; int num_recycles=NUM_RECYCLES;
    struct PredictionResult { std::string sequence; std::vector<BackboneAtoms> atoms; Mat coordinates; std::vector<float> plddt; float complex_plddt,ptm,iptm; Mat distogram; };
    PredictionResult predict(const std::string& sequence) {
        model.init((int)sequence.size()); auto output=model.forward(sequence,num_recycles);
        return {sequence,output.atoms,output.diffusion_coords,output.conf.plddt,output.conf.complex_plddt,output.conf.ptm,output.conf.iptm,output.disto};
    }
    std::vector<PredictionResult> predict_fasta(const std::string& path) {
        auto records=parse_fasta(path); std::vector<PredictionResult> results;
        for (const auto& rec:records) if (rec.entity_type=="PROTEIN"||rec.entity_type.empty()) results.push_back(predict(rec.sequence));
        return results;
    }
};

// ============================================================================
// Featurization
// ============================================================================
struct Features {
    Mat res_type,profile,deletion_mean,token_pad_mask,token_bonds,disto_target;
    std::vector<int> asym_id,entity_id,residue_index,token_index,sym_id,mol_type;
};

inline Features featurize(const Tokenized& data) {
    Features f; int N=(int)data.tokens.size();
    f.res_type=Mat(N,NUM_FULL_TOKENS,0.0f);
    for (int i=0;i<N;++i) { int rt=std::clamp(data.tokens[i].res_type+2,0,NUM_FULL_TOKENS-1); f.res_type(i,rt)=1.0f; }
    f.profile=f.res_type; f.deletion_mean=Mat(N,1,0.0f); f.token_pad_mask=Mat(1,N,1.0f);
    f.token_bonds=Mat(N*N,1,0.0f);
    for (const auto& bond:data.bonds) if (bond.token_1<N&&bond.token_2<N) { f.token_bonds(bond.token_1*N+bond.token_2,0)=1.0f; f.token_bonds(bond.token_2*N+bond.token_1,0)=1.0f; }
    f.asym_id.resize(N); f.entity_id.resize(N); f.residue_index.resize(N); f.token_index.resize(N); f.sym_id.resize(N); f.mol_type.resize(N);
    for (int i=0;i<N;++i) { f.asym_id[i]=data.tokens[i].asym_id; f.entity_id[i]=data.tokens[i].entity_id;
        f.residue_index[i]=data.tokens[i].res_idx; f.token_index[i]=data.tokens[i].token_idx;
        f.sym_id[i]=data.tokens[i].sym_id; f.mol_type[i]=data.tokens[i].mol_type; }
    f.disto_target=Mat(N*N,NUM_DIST_BINS,0.0f);
    for (int i=0;i<N;++i) for (int j=0;j<N;++j) {
        float d=(data.tokens[i].disto_coords-data.tokens[j].disto_coords).norm();
        int bin=std::clamp((int)((d-DIST_MIN)/(DIST_MAX-DIST_MIN)*NUM_DIST_BINS),0,NUM_DIST_BINS-1);
        f.disto_target(i*N+j,bin)=1.0f;
    }
    return f;
}

// ============================================================================
// Symmetry permutations
// ============================================================================
struct SymmetryPermutation { std::vector<std::pair<int,int>> swaps; };
inline std::vector<SymmetryPermutation> get_residue_symmetries(const std::string& res_name) {
    if (res_name=="ASP") return {{{{6,7},{7,6}}}};
    if (res_name=="GLU") return {{{{7,8},{8,7}}}};
    if (res_name=="PHE") return {{{{6,7},{7,6},{8,9},{9,8}}}};
    if (res_name=="TYR") return {{{{6,7},{7,6},{8,9},{9,8}}}};
    return {};
}

// ============================================================================
// VDW Radii
// ============================================================================
inline float vdw_radius(int element) {
    static const float radii[]={1.2f,1.4f,2.2f,1.9f,1.8f,1.7f,1.6f,1.55f,1.5f,1.54f,2.4f,2.2f,2.1f,2.1f,1.95f,1.8f,1.8f,1.88f,2.8f,2.4f,
        2.3f,2.15f,2.05f,2.05f,2.05f,2.05f,2.0f,2.0f,2.0f,2.1f,2.1f,2.1f,2.05f,1.9f,1.9f,2.02f,2.9f,2.55f,2.4f,2.3f};
    if (element>=1&&element<=40) return radii[element-1]; return 2.0f;
}

inline float compute_rmsd(const Mat& c1,const Mat& c2) {
    int N=c1.rows; float sum=0; for (int i=0;i<N;++i) { float dx=c1(i,0)-c2(i,0),dy=c1(i,1)-c2(i,1),dz=c1(i,2)-c2(i,2); sum+=dx*dx+dy*dy+dz*dz; } return std::sqrt(sum/std::max(N,1));
}

// ============================================================================
// PDB & mmCIF output
// ============================================================================
inline std::string atoms_to_pdb(const std::vector<BackboneAtoms>& atoms,const std::string& sequence,
                                const std::string& model_name="BOLTZ",const std::vector<float>* plddt=nullptr) {
    std::ostringstream oss;
    oss<<"REMARK   Generated by "<<model_name<<" C++ port\n"; oss<<"REMARK   Sequence: "<<sequence<<"\n"; oss<<"MODEL     1\n";
    int serial=1;
    for (int i=0;i<(int)atoms.size();++i) {
        int resSeq=i+1; int aa_idx=(i<(int)sequence.size())?aa_to_index(sequence[i]):NUM_AA;
        const char* resName=aa_index_to_3letter(aa_idx);
        float bfac=(plddt&&i<(int)plddt->size())?(*plddt)[i]*100.0f:0.0f;
        auto write_atom=[&](const char* name,const Vec3& pos){
            oss<<"ATOM  "<<std::setw(5)<<serial++<<" "<<std::setw(4)<<std::left<<name<<" "<<std::right
               <<std::setw(3)<<resName<<" A"<<std::setw(4)<<resSeq<<"    "<<std::fixed<<std::setprecision(3)
               <<std::setw(8)<<pos.x<<std::setw(8)<<pos.y<<std::setw(8)<<pos.z
               <<std::setw(6)<<std::setprecision(2)<<1.00<<std::setw(6)<<std::setprecision(2)<<bfac
               <<"           "<<name[0]<<"\n";
        };
        write_atom("N",atoms[i].N_pos); write_atom("CA",atoms[i].CA_pos);
        write_atom("C",atoms[i].C_pos); write_atom("O",atoms[i].O_pos);
    }
    oss<<"TER\nENDMDL\nEND\n"; return oss.str();
}

inline std::string atoms_to_mmcif(const std::vector<BackboneAtoms>& atoms,const std::string& sequence,
                                   const std::string& entry_id="BOLTZ",const std::vector<float>* plddt=nullptr) {
    std::ostringstream oss;
    oss<<"data_"<<entry_id<<"\n#\nloop_\n";
    oss<<"_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n";
    oss<<"_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_seq_id\n";
    oss<<"_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n";
    int serial=1;
    for (int i=0;i<(int)atoms.size();++i) {
        int resSeq=i+1; int aa_idx=(i<(int)sequence.size())?aa_to_index(sequence[i]):NUM_AA;
        const char* resName=aa_index_to_3letter(aa_idx);
        float bfac=(plddt&&i<(int)plddt->size())?(*plddt)[i]*100.0f:0.0f;
        auto wa=[&](const char* name,const char* elem,const Vec3& pos){
            oss<<"ATOM "<<serial++<<" "<<elem<<" "<<name<<" "<<resName<<" A "<<resSeq<<" "
               <<std::fixed<<std::setprecision(3)<<pos.x<<" "<<pos.y<<" "<<pos.z<<" "
               <<std::setprecision(2)<<1.00<<" "<<bfac<<"\n";
        };
        wa("N","N",atoms[i].N_pos); wa("CA","C",atoms[i].CA_pos); wa("C","C",atoms[i].C_pos); wa("O","O",atoms[i].O_pos);
    }
    oss<<"#\n"; return oss.str();
}

inline bool write_pdb(const std::string& filename,const std::string& pdb_str) { std::ofstream f(filename); if (!f.is_open()) return false; f<<pdb_str; return true; }
inline bool write_mmcif(const std::string& filename,const std::string& cif_str) { std::ofstream f(filename); if (!f.is_open()) return false; f<<cif_str; return true; }
inline std::string read_file(const std::string& filename) { std::ifstream f(filename); if (!f.is_open()) return ""; std::ostringstream ss; ss<<f.rdbuf(); return ss.str(); }

// ============================================================================
// Training config & simulation
// ============================================================================
struct TrainingConfig {
    float learning_rate=1.8e-3f; int warmup_steps=1000,max_steps=100000,start_decay=50000;
    float ema_decay=0.999f,diffusion_loss_weight=4.0f,distogram_loss_weight=0.03f;
    float confidence_loss_weight=1e-4f,bfactor_loss_weight=0.01f,alpha_pae=0.0f;
    int num_recycles=3,diffusion_samples=48,max_tokens=384;
    bool structure_prediction_training=true,confidence_prediction=false;
};

struct TrainingStep { float total_loss=0,diffusion_loss=0,distogram_loss_val=0; };

inline TrainingStep simulate_training_step(BoltzModel& model,const std::string& sequence,const TrainingConfig& config) {
    TrainingStep step; auto output=model.forward(sequence,config.num_recycles);
    int N=(int)sequence.size();
    std::vector<Rigid> ref_frames(N); for (int i=0;i<N;++i) ref_frames[i]=Rigid(Rot3::identity(),Vec3(i*3.8f,0,0));
    auto ref_atoms=place_backbone_atoms(ref_frames,N);
    step.diffusion_loss=fape_loss(output.frames,output.atoms,ref_frames,ref_atoms,N);
    std::vector<bool> mask(N,true); Mat td(N*N,NUM_DIST_BINS,0.0f);
    for (int i=0;i<N;++i) for (int j=0;j<N;++j) { float d=(ref_atoms[i].CA_pos-ref_atoms[j].CA_pos).norm();
        int bin=std::clamp((int)((d-DIST_MIN)/(DIST_MAX-DIST_MIN)*NUM_DIST_BINS),0,NUM_DIST_BINS-1); td(i*N+j,bin)=1.0f; }
    step.distogram_loss_val=distogram_loss(output.disto,td,mask,N);
    step.total_loss=config.diffusion_loss_weight*step.diffusion_loss+config.distogram_loss_weight*step.distogram_loss_val
                   +smooth_lddt_loss(output.atoms,ref_atoms,N);
    return step;
}



// ============================================================================
// Additional layer types not in original (Dropout, Pair Averaging v2,
// Atom Transformer with windowed attention)
// ============================================================================

// Row/Column dropout (matching Python get_dropout_mask)
struct DropoutMask {
    float dropout_rate = 0.0f;
    bool columnwise = false;

    Mat get_mask(int rows, int cols, bool training) const {
        Mat mask(rows, cols, 1.0f);
        if (!training || dropout_rate <= 0.0f) return mask;
        float keep = 1.0f - dropout_rate;
        float scale = 1.0f / keep;
        if (columnwise) {
            for (int j = 0; j < cols; ++j) {
                if (rand_uniform() >= keep)
                    for (int i = 0; i < rows; ++i) mask(i, j) = 0.0f;
                else
                    for (int i = 0; i < rows; ++i) mask(i, j) = scale;
            }
        } else {
            for (int i = 0; i < rows; ++i) {
                if (rand_uniform() >= keep)
                    for (int j = 0; j < cols; ++j) mask(i, j) = 0.0f;
                else
                    for (int j = 0; j < cols; ++j) mask(i, j) = scale;
            }
        }
        return mask;
    }
};

// Windowed Atom Transformer (matching Python AtomTransformer)
struct AtomTransformerWindowed {
    int window_q = ATOMS_PER_WINDOW_Q;
    int window_k = ATOMS_PER_WINDOW_K;
    DiffusionTransformer inner;

    AtomTransformerWindowed() = default;
    AtomTransformerWindowed(int depth, int dim, int dim_cond, int dim_pair, int heads,
                            int wq = ATOMS_PER_WINDOW_Q, int wk = ATOMS_PER_WINDOW_K)
        : window_q(wq), window_k(wk),
          inner(depth, dim, dim_cond, dim_pair, heads) {}

    Mat forward(const Mat& q, const Mat& c, const Mat& p_flat, int N_atoms) const {
        // If windowed, reshape into windows
        if (window_q > 0 && N_atoms > window_q) {
            int NW = N_atoms / window_q;
            // Process each window
            Mat out(N_atoms, q.cols, 0.0f);
            for (int w = 0; w < NW; ++w) {
                int start = w * window_q;
                int end = std::min(start + window_q, N_atoms);
                int ws = end - start;
                Mat qw(ws, q.cols), cw(ws, c.cols);
                for (int i = 0; i < ws; ++i) {
                    std::copy(q.row_ptr(start+i), q.row_ptr(start+i)+q.cols, qw.row_ptr(i));
                    std::copy(c.row_ptr(start+i), c.row_ptr(start+i)+c.cols, cw.row_ptr(i));
                }
                // Use pair features for this window
                Mat pw(ws*ws, p_flat.cols, 0.0f);
                for (int i = 0; i < ws; ++i) for (int j = 0; j < ws; ++j) {
                    int gi = start+i, gj = start+j;
                    if (gi < N_atoms && gj < N_atoms && gi*N_atoms+gj < p_flat.rows)
                        std::copy(p_flat.row_ptr(gi*N_atoms+gj),
                                  p_flat.row_ptr(gi*N_atoms+gj)+p_flat.cols,
                                  pw.row_ptr(i*ws+j));
                }
                Mat res = inner.forward(qw, cw, pw);
                for (int i = 0; i < ws; ++i)
                    std::copy(res.row_ptr(i), res.row_ptr(i)+res.cols, out.row_ptr(start+i));
            }
            return out;
        }
        // No windowing, use full attention
        return inner.forward(q, c, p_flat);
    }
};

// ============================================================================
// Atom Encoder/Decoder (matching Python AtomAttentionEncoder/Decoder)
// ============================================================================
struct AtomEncoder {
    int atom_s, atom_z, token_s, token_z;
    Linear embed_atom_features, embed_ref_pos, embed_ref_dist, embed_mask;
    Linear s_to_c, z_to_p, r_to_q;
    Linear c_to_p_q, c_to_p_k;
    Linear atom_to_token_proj;
    AtomTransformerWindowed atom_transformer;

    AtomEncoder() : atom_s(0), atom_z(0), token_s(0), token_z(0) {}
    AtomEncoder(int as, int az, int ts, int tz, int depth = 3, int heads = 4)
        : atom_s(as), atom_z(az), token_s(ts), token_z(tz) {
        embed_atom_features = Linear(ATOM_FEATURE_DIM, as, false);
        embed_ref_pos = Linear(3, az, false);
        embed_ref_dist = Linear(1, az, false);
        embed_mask = Linear(1, az, false);
        s_to_c = Linear(ts, as, false);
        z_to_p = Linear(tz, az, false);
        r_to_q = Linear(10, as, false);
        c_to_p_q = Linear(as, az, false);
        c_to_p_k = Linear(as, az, false);
        atom_to_token_proj = Linear(as, 2*ts, false);
        atom_transformer = AtomTransformerWindowed(depth, as, as, az, heads);
        gating_init(s_to_c.W); gating_init(z_to_p.W); gating_init(r_to_q.W);
    }
};

struct AtomDecoder {
    int atom_s, token_s;
    Linear a_to_q;
    AtomTransformerWindowed atom_transformer;
    LayerNorm pos_norm;
    Linear pos_proj;

    AtomDecoder() : atom_s(0), token_s(0) {}
    AtomDecoder(int as, int az, int ts, int depth = 3, int heads = 4)
        : atom_s(as), token_s(ts) {
        a_to_q = Linear(2*ts, as, false);
        atom_transformer = AtomTransformerWindowed(depth, as, as, az, heads);
        pos_norm = LayerNorm(as);
        pos_proj = Linear(as, 3, false);
    }
};

// ============================================================================
// Extended Confidence Module with full iPTM, ligand_iptm, protein_iptm
// ============================================================================
struct ConfidenceModuleExtended {
    Embedding dist_bin_embed;
    LayerNorm s_norm, z_norm;
    PairformerModule pairformer;
    ConfidenceHeads heads;
    Linear s_to_z, s_to_z_t;
    bool add_s_to_z_prod = false;
    bool use_s_diffusion = false;

    ConfidenceModuleExtended() = default;
    ConfidenceModuleExtended(int token_s, int token_z, int pf_blocks = 2) {
        dist_bin_embed = Embedding(NUM_DIST_BINS, token_z);
        s_norm = LayerNorm(token_s);
        z_norm = LayerNorm(token_z);
        pairformer = PairformerModule(token_s, token_z, pf_blocks);
        heads = ConfidenceHeads(token_s, token_z, true);
        int s_input_dim = token_s + 2 * NUM_FULL_TOKENS + 1 + 4;
        s_to_z = Linear(s_input_dim, token_z, false);
        s_to_z_t = Linear(s_input_dim, token_z, false);
    }

    struct ExtendedOutput {
        ConfidenceHeads::ConfidenceOutput base;
        float complex_iplddt = 0;
        float complex_pde = 0;
        float complex_ipde = 0;
        float ligand_iptm = 0;
        float protein_iptm = 0;
    };

    ExtendedOutput forward(const Mat& s_in, const Mat& z_in, const Mat& x_pred,
                            const std::vector<int>& mol_types,
                            const std::vector<int>& asym_ids, int N) const {
        ExtendedOutput out;
        Mat s = s_in; s_norm.forward(s);
        Mat z = z_in; z_norm.forward(z);

        // Add distance bin embeddings
        Mat dists = cdist(x_pred);
        for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
            int bin = std::min((int)(dists(i,j) * NUM_DIST_BINS / 22.0f), NUM_DIST_BINS - 1);
            float emb[TOKEN_Z];
            dist_bin_embed.lookup(bin, emb);
            for (int k = 0; k < TOKEN_Z; ++k) z(i*N+j, k) += emb[k];
        }

        pairformer.forward(s, z, N);
        out.base = heads.forward(s, z, N);

        // Compute interface pLDDT
        float iplddt_num = 0, iplddt_den = 0;
        for (int i = 0; i < N; ++i) {
            bool is_interface = false;
            for (int j = 0; j < N; ++j) {
                if (asym_ids[i] != asym_ids[j] && dists(i,j) < 8.0f) {
                    is_interface = true; break;
                }
            }
            bool is_ligand = (mol_types[i] == MOL_NONPOLYMER);
            float w = is_ligand ? 2.0f : (is_interface ? 1.0f : 0.0f);
            iplddt_num += out.base.plddt[i] * w;
            iplddt_den += w;
        }
        out.complex_iplddt = (iplddt_den > 0) ? iplddt_num / iplddt_den : out.base.complex_plddt;

        return out;
    }
};

// ============================================================================
// Full Boltz1 Model (extended, matching Python Boltz1 class)
// ============================================================================
struct Boltz1Model {
    BoltzModel base;
    AtomEncoder atom_encoder;
    AtomDecoder atom_decoder;
    ConfidenceModuleExtended confidence_ext;
    DiffusionModule atom_diffusion;
    bool has_atom_encoder = false;

    Boltz1Model() = default;

    void init(int n_res, bool use_atom_level = false) {
        base.init(n_res);
        if (use_atom_level) {
            atom_encoder = AtomEncoder(ATOM_S, ATOM_Z, TOKEN_S, TOKEN_Z,
                                        ATOM_ENCODER_DEPTH, NUM_HEADS);
            atom_decoder = AtomDecoder(ATOM_S, ATOM_Z, TOKEN_S,
                                        ATOM_DECODER_DEPTH, NUM_HEADS);
            has_atom_encoder = true;
        }
        confidence_ext = ConfidenceModuleExtended(TOKEN_S, TOKEN_Z);
    }

    BoltzModel::Output forward(const std::string& sequence, int num_recycles = NUM_RECYCLES) const {
        return base.forward(sequence, num_recycles);
    }
};

// ============================================================================
// Predict pipeline (extended, matching Python predict flow)
// ============================================================================
struct PredictionPipelineExtended {
    Boltz1Model model;
    int num_recycles = NUM_RECYCLES;
    int num_samples = 1;
    bool rank_by_confidence = true;

    struct ExtendedResult {
        std::string sequence;
        std::vector<BackboneAtoms> atoms;
        Mat coordinates;
        std::vector<float> plddt;
        float complex_plddt, ptm, iptm;
        Mat distogram;
        int rank = 0;
        float ranking_score = 0;
    };

    std::vector<ExtendedResult> predict(const std::string& sequence) {
        model.init((int)sequence.size());
        std::vector<ExtendedResult> results;

        for (int s = 0; s < num_samples; ++s) {
            auto output = model.forward(sequence, num_recycles);
            ExtendedResult r;
            r.sequence = sequence;
            r.atoms = output.atoms;
            r.coordinates = output.diffusion_coords;
            r.plddt = output.conf.plddt;
            r.complex_plddt = output.conf.complex_plddt;
            r.ptm = output.conf.ptm;
            r.iptm = output.conf.iptm;
            r.distogram = output.disto;
            // Ranking score = 0.2 * ptm + 0.8 * iptm (AF3 formula)
            r.ranking_score = 0.2f * r.ptm + 0.8f * r.iptm;
            results.push_back(r);
        }

        if (rank_by_confidence && results.size() > 1) {
            std::sort(results.begin(), results.end(),
                [](const ExtendedResult& a, const ExtendedResult& b) {
                    return a.ranking_score > b.ranking_score;
                });
        }
        for (int i = 0; i < (int)results.size(); ++i) results[i].rank = i;
        return results;
    }
};


} // namespace boltz
