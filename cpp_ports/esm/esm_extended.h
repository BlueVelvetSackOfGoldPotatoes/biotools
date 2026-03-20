// esm_extended.h - Extended C++17 port of ESM (Evolutionary Scale Modeling)
//
// This header adds the remaining components not yet in esm.h:
//   - Triangle multiplicative updates (outgoing/incoming)
//   - Triangle attention (starting/ending node)
//   - IPA (Invariant Point Attention) and Structure Module
//   - Full ESMFold model class
//   - GVP graph convolution (simplified without torch_geometric)
//   - GVP graph embedding with edge features
//   - DihedralFeatures module (learnable)
//   - NormalizedResidualBlock
//   - CoordBatchConverter for inverse folding
//   - Multichain utilities
//   - PDB I/O utilities
//   - Alphabet tokenize() method
//   - MSA position embedding
//   - Incremental decoding state
//   - Dropout with dimension sharing
//   - Extended pretrained registry (MSA, ESM-IF, ESMFold)
//   - Rigid transforms and quaternion utilities for structure prediction

#ifndef ESM_EXTENDED_H
#define ESM_EXTENDED_H

#include "esm.h"

namespace esm {

// ===================================================================
// SECTION E1: Quaternion and Rigid Body Utilities
// ===================================================================

struct Quaternion {
    float w = 1.0f, x = 0.0f, y = 0.0f, z = 0.0f;

    Quaternion() = default;
    Quaternion(float w_, float x_, float y_, float z_) : w(w_), x(x_), y(y_), z(z_) {}

    Quaternion operator*(const Quaternion& q) const {
        return {
            w * q.w - x * q.x - y * q.y - z * q.z,
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w
        };
    }

    Quaternion conjugate() const { return {w, -x, -y, -z}; }
    float norm() const { return std::sqrt(w*w + x*x + y*y + z*z); }

    Quaternion normalized() const {
        float n = norm();
        if (n < 1e-8f) return {1, 0, 0, 0};
        float inv = 1.0f / n;
        return {w * inv, x * inv, y * inv, z * inv};
    }

    Vec3 rotate(const Vec3& v) const {
        Quaternion p(0, v.x, v.y, v.z);
        Quaternion result = (*this) * p * conjugate();
        return {result.x, result.y, result.z};
    }

    void to_rotation_matrix(float* R) const {
        float xx = x*x, yy = y*y, zz = z*z;
        float xy = x*y, xz = x*z, yz = y*z;
        float wx = w*x, wy = w*y, wz = w*z;
        R[0] = 1 - 2*(yy + zz); R[1] = 2*(xy - wz);     R[2] = 2*(xz + wy);
        R[3] = 2*(xy + wz);     R[4] = 1 - 2*(xx + zz); R[5] = 2*(yz - wx);
        R[6] = 2*(xz - wy);     R[7] = 2*(yz + wx);     R[8] = 1 - 2*(xx + yy);
    }

    static Quaternion from_rotation_matrix(const float* R) {
        float tr = R[0] + R[4] + R[8];
        Quaternion q;
        if (tr > 0) {
            float s = 0.5f / std::sqrt(tr + 1.0f);
            q.w = 0.25f / s;
            q.x = (R[7] - R[5]) * s;
            q.y = (R[2] - R[6]) * s;
            q.z = (R[3] - R[1]) * s;
        } else if (R[0] > R[4] && R[0] > R[8]) {
            float s = 2.0f * std::sqrt(1.0f + R[0] - R[4] - R[8]);
            q.w = (R[7] - R[5]) / s; q.x = 0.25f * s;
            q.y = (R[1] + R[3]) / s; q.z = (R[2] + R[6]) / s;
        } else if (R[4] > R[8]) {
            float s = 2.0f * std::sqrt(1.0f + R[4] - R[0] - R[8]);
            q.w = (R[2] - R[6]) / s; q.x = (R[1] + R[3]) / s;
            q.y = 0.25f * s; q.z = (R[5] + R[7]) / s;
        } else {
            float s = 2.0f * std::sqrt(1.0f + R[8] - R[0] - R[4]);
            q.w = (R[3] - R[1]) / s; q.x = (R[2] + R[6]) / s;
            q.y = (R[5] + R[7]) / s; q.z = 0.25f * s;
        }
        return q.normalized();
    }
};

struct Rigid {
    Quaternion rot;
    Vec3 trans;

    Rigid() = default;
    Rigid(const Quaternion& r, const Vec3& t) : rot(r), trans(t) {}
    static Rigid identity() { return Rigid(Quaternion(), Vec3()); }

    Vec3 apply(const Vec3& v) const { return rot.rotate(v) + trans; }

    Rigid compose(const Rigid& other) const {
        return Rigid(rot * other.rot, rot.rotate(other.trans) + trans);
    }

    Rigid inverse() const {
        Quaternion inv_rot = rot.conjugate();
        Vec3 inv_trans = inv_rot.rotate(trans * -1.0f);
        return Rigid(inv_rot, inv_trans);
    }

    static Rigid from_3_points(const Vec3& p1, const Vec3& p2, const Vec3& p3) {
        Vec3 e1 = (p3 - p2).normalized();
        Vec3 v2 = p1 - p2;
        Vec3 u2 = v2 - e1 * e1.dot(v2);
        Vec3 e2 = u2.normalized();
        Vec3 e3 = e1.cross(e2);
        float R[9] = {e1.x, e2.x, e3.x, e1.y, e2.y, e3.y, e1.z, e2.z, e3.z};
        return Rigid(Quaternion::from_rotation_matrix(R), p2);
    }
};


// ===================================================================
// SECTION E2: Triangle Multiplicative Update
// ===================================================================

struct TriangleMultiplicationOutgoing {
    int c_z = 128, c_hidden = 128;
    ESM1bLayerNorm layer_norm_in;
    Linear linear_a_p, linear_a_g, linear_b_p, linear_b_g;
    ESM1bLayerNorm layer_norm_out;
    Linear linear_z, linear_g;

    TriangleMultiplicationOutgoing() = default;
    TriangleMultiplicationOutgoing(int cz, int ch, std::mt19937& rng)
        : c_z(cz), c_hidden(ch) {
        layer_norm_in = ESM1bLayerNorm(cz);
        linear_a_p = Linear(cz, ch, true, rng);
        linear_a_g = Linear(cz, ch, true, rng);
        linear_b_p = Linear(cz, ch, true, rng);
        linear_b_g = Linear(cz, ch, true, rng);
        layer_norm_out = ESM1bLayerNorm(ch);
        linear_z = Linear(ch, cz, true, rng);
        linear_g = Linear(cz, cz, true, rng);
        linear_z.weight.zeros();
        for (auto& b : linear_z.bias) b = 0.0f;
    }

    void forward(float* pair_out, const float* pair_state, int L,
                 const std::vector<bool>& mask = {}) const {
        int cz = c_z, ch = c_hidden;
        int LL = L * L;
        std::vector<float> normed(LL * cz);
        math::vec_copy(normed.data(), pair_state, LL * cz);
        for (int i = 0; i < LL; ++i) layer_norm_in.forward(normed.data() + i * cz);

        std::vector<float> a(LL * ch), b(LL * ch), ga(LL * ch), gb(LL * ch);
        linear_a_p.forward(normed.data(), a.data(), LL);
        linear_a_g.forward(normed.data(), ga.data(), LL);
        linear_b_p.forward(normed.data(), b.data(), LL);
        linear_b_g.forward(normed.data(), gb.data(), LL);
        math::sigmoid_inplace(ga.data(), LL * ch);
        math::sigmoid_inplace(gb.data(), LL * ch);
        math::vec_mul(a.data(), ga.data(), LL * ch);
        math::vec_mul(b.data(), gb.data(), LL * ch);

        if (!mask.empty())
            for (int i = 0; i < L; ++i)
                for (int j = 0; j < L; ++j)
                    if (!mask[i * L + j]) {
                        math::vec_zero(a.data() + (i * L + j) * ch, ch);
                        math::vec_zero(b.data() + (i * L + j) * ch, ch);
                    }

        // Outgoing: z_ij = sum_k a_ik * b_jk
        std::vector<float> z_out(LL * ch, 0.0f);
        for (int i = 0; i < L; ++i)
            for (int j = 0; j < L; ++j) {
                float* dst = z_out.data() + (i * L + j) * ch;
                for (int k = 0; k < L; ++k) {
                    const float* a_ik = a.data() + (i * L + k) * ch;
                    const float* b_jk = b.data() + (j * L + k) * ch;
                    for (int d = 0; d < ch; ++d) dst[d] += a_ik[d] * b_jk[d];
                }
            }

        for (int i = 0; i < LL; ++i) layer_norm_out.forward(z_out.data() + i * ch);
        std::vector<float> projected(LL * cz);
        linear_z.forward(z_out.data(), projected.data(), LL);
        std::vector<float> gate(LL * cz);
        linear_g.forward(normed.data(), gate.data(), LL);
        math::sigmoid_inplace(gate.data(), LL * cz);
        math::vec_mul(projected.data(), gate.data(), LL * cz);
        math::vec_copy(pair_out, projected.data(), LL * cz);
    }
};

struct TriangleMultiplicationIncoming {
    int c_z = 128, c_hidden = 128;
    ESM1bLayerNorm layer_norm_in;
    Linear linear_a_p, linear_a_g, linear_b_p, linear_b_g;
    ESM1bLayerNorm layer_norm_out;
    Linear linear_z, linear_g;

    TriangleMultiplicationIncoming() = default;
    TriangleMultiplicationIncoming(int cz, int ch, std::mt19937& rng)
        : c_z(cz), c_hidden(ch) {
        layer_norm_in = ESM1bLayerNorm(cz);
        linear_a_p = Linear(cz, ch, true, rng);
        linear_a_g = Linear(cz, ch, true, rng);
        linear_b_p = Linear(cz, ch, true, rng);
        linear_b_g = Linear(cz, ch, true, rng);
        layer_norm_out = ESM1bLayerNorm(ch);
        linear_z = Linear(ch, cz, true, rng);
        linear_g = Linear(cz, cz, true, rng);
        linear_z.weight.zeros();
        for (auto& b : linear_z.bias) b = 0.0f;
    }

    void forward(float* pair_out, const float* pair_state, int L,
                 const std::vector<bool>& mask = {}) const {
        int cz = c_z, ch = c_hidden;
        int LL = L * L;
        std::vector<float> normed(LL * cz);
        math::vec_copy(normed.data(), pair_state, LL * cz);
        for (int i = 0; i < LL; ++i) layer_norm_in.forward(normed.data() + i * cz);

        std::vector<float> a(LL * ch), b(LL * ch), ga(LL * ch), gb(LL * ch);
        linear_a_p.forward(normed.data(), a.data(), LL);
        linear_a_g.forward(normed.data(), ga.data(), LL);
        linear_b_p.forward(normed.data(), b.data(), LL);
        linear_b_g.forward(normed.data(), gb.data(), LL);
        math::sigmoid_inplace(ga.data(), LL * ch);
        math::sigmoid_inplace(gb.data(), LL * ch);
        math::vec_mul(a.data(), ga.data(), LL * ch);
        math::vec_mul(b.data(), gb.data(), LL * ch);

        if (!mask.empty())
            for (int i = 0; i < L; ++i)
                for (int j = 0; j < L; ++j)
                    if (!mask[i * L + j]) {
                        math::vec_zero(a.data() + (i * L + j) * ch, ch);
                        math::vec_zero(b.data() + (i * L + j) * ch, ch);
                    }

        // Incoming: z_ij = sum_k a_ki * b_kj
        std::vector<float> z_out(LL * ch, 0.0f);
        for (int i = 0; i < L; ++i)
            for (int j = 0; j < L; ++j) {
                float* dst = z_out.data() + (i * L + j) * ch;
                for (int k = 0; k < L; ++k) {
                    const float* a_ki = a.data() + (k * L + i) * ch;
                    const float* b_kj = b.data() + (k * L + j) * ch;
                    for (int d = 0; d < ch; ++d) dst[d] += a_ki[d] * b_kj[d];
                }
            }

        for (int i = 0; i < LL; ++i) layer_norm_out.forward(z_out.data() + i * ch);
        std::vector<float> projected(LL * cz);
        linear_z.forward(z_out.data(), projected.data(), LL);
        std::vector<float> gate(LL * cz);
        linear_g.forward(normed.data(), gate.data(), LL);
        math::sigmoid_inplace(gate.data(), LL * cz);
        math::vec_mul(projected.data(), gate.data(), LL * cz);
        math::vec_copy(pair_out, projected.data(), LL * cz);
    }
};

// ===================================================================
// SECTION E3: Triangle Attention (Starting/Ending Node)
// ===================================================================

struct TriangleAttentionStartingNode {
    int c_z = 128, c_hidden = 32, num_heads = 4;
    float inf_val = 1e9f, scaling = 1.0f;
    ESM1bLayerNorm layer_norm;
    Linear linear_q, linear_k, linear_v, linear_b, linear_g, linear_o;

    TriangleAttentionStartingNode() = default;
    TriangleAttentionStartingNode(int cz, int ch, int nh, float inf, std::mt19937& rng)
        : c_z(cz), c_hidden(ch), num_heads(nh), inf_val(inf) {
        scaling = 1.0f / std::sqrt(static_cast<float>(ch));
        layer_norm = ESM1bLayerNorm(cz);
        int pd = ch * nh;
        linear_q = Linear(cz, pd, true, rng); linear_k = Linear(cz, pd, true, rng);
        linear_v = Linear(cz, pd, true, rng); linear_b = Linear(cz, nh, false, rng);
        linear_g = Linear(cz, pd, true, rng); linear_o = Linear(pd, cz, true, rng);
        linear_o.weight.zeros();
        for (auto& bv : linear_o.bias) bv = 0.0f;
    }

    void forward(float* output, const float* pair_state, int L,
                 const std::vector<bool>& mask = {}) const {
        int cz = c_z, ch = c_hidden, nh = num_heads, pd = ch * nh, LL = L * L;
        std::vector<float> normed(LL * cz);
        math::vec_copy(normed.data(), pair_state, LL * cz);
        for (int i = 0; i < LL; ++i) layer_norm.forward(normed.data() + i * cz);

        std::vector<float> Q(LL * pd), K(LL * pd), V(LL * pd);
        std::vector<float> bias(LL * nh), gate(LL * pd);
        linear_q.forward(normed.data(), Q.data(), LL);
        linear_k.forward(normed.data(), K.data(), LL);
        linear_v.forward(normed.data(), V.data(), LL);
        linear_b.forward(normed.data(), bias.data(), LL);
        linear_g.forward(normed.data(), gate.data(), LL);
        math::sigmoid_inplace(gate.data(), LL * pd);

        std::vector<float> attn_out(LL * pd, 0.0f);
        for (int i = 0; i < L; ++i)
            for (int h = 0; h < nh; ++h)
                for (int j = 0; j < L; ++j) {
                    std::vector<float> scores(L);
                    for (int k = 0; k < L; ++k) {
                        float dot = 0.0f;
                        for (int d = 0; d < ch; ++d)
                            dot += Q[(i*L+j)*pd + h*ch+d] * K[(i*L+k)*pd + h*ch+d];
                        dot *= scaling;
                        dot += bias[(i*L+k)*nh + h];
                        if (!mask.empty() && !mask[i*L+k]) dot = -inf_val;
                        scores[k] = dot;
                    }
                    math::softmax_row(scores.data(), L);
                    for (int k = 0; k < L; ++k) {
                        float w = scores[k];
                        for (int d = 0; d < ch; ++d)
                            attn_out[(i*L+j)*pd + h*ch+d] += w * V[(i*L+k)*pd + h*ch+d];
                    }
                }

        math::vec_mul(attn_out.data(), gate.data(), LL * pd);
        linear_o.forward(attn_out.data(), output, LL);
    }
};

struct TriangleAttentionEndingNode {
    int c_z = 128, c_hidden = 32, num_heads = 4;
    float inf_val = 1e9f, scaling = 1.0f;
    ESM1bLayerNorm layer_norm;
    Linear linear_q, linear_k, linear_v, linear_b, linear_g, linear_o;

    TriangleAttentionEndingNode() = default;
    TriangleAttentionEndingNode(int cz, int ch, int nh, float inf, std::mt19937& rng)
        : c_z(cz), c_hidden(ch), num_heads(nh), inf_val(inf) {
        scaling = 1.0f / std::sqrt(static_cast<float>(ch));
        layer_norm = ESM1bLayerNorm(cz);
        int pd = ch * nh;
        linear_q = Linear(cz, pd, true, rng); linear_k = Linear(cz, pd, true, rng);
        linear_v = Linear(cz, pd, true, rng); linear_b = Linear(cz, nh, false, rng);
        linear_g = Linear(cz, pd, true, rng); linear_o = Linear(pd, cz, true, rng);
        linear_o.weight.zeros();
        for (auto& bv : linear_o.bias) bv = 0.0f;
    }

    void forward(float* output, const float* pair_state, int L,
                 const std::vector<bool>& mask = {}) const {
        int cz = c_z, ch = c_hidden, nh = num_heads, pd = ch * nh, LL = L * L;
        std::vector<float> normed(LL * cz);
        math::vec_copy(normed.data(), pair_state, LL * cz);
        for (int i = 0; i < LL; ++i) layer_norm.forward(normed.data() + i * cz);

        std::vector<float> Q(LL * pd), K(LL * pd), V(LL * pd);
        std::vector<float> bias(LL * nh), gate(LL * pd);
        linear_q.forward(normed.data(), Q.data(), LL);
        linear_k.forward(normed.data(), K.data(), LL);
        linear_v.forward(normed.data(), V.data(), LL);
        linear_b.forward(normed.data(), bias.data(), LL);
        linear_g.forward(normed.data(), gate.data(), LL);
        math::sigmoid_inplace(gate.data(), LL * pd);

        std::vector<float> attn_out(LL * pd, 0.0f);
        for (int j = 0; j < L; ++j)
            for (int h = 0; h < nh; ++h)
                for (int i = 0; i < L; ++i) {
                    std::vector<float> scores(L);
                    for (int k = 0; k < L; ++k) {
                        float dot = 0.0f;
                        for (int d = 0; d < ch; ++d)
                            dot += Q[(i*L+j)*pd + h*ch+d] * K[(k*L+j)*pd + h*ch+d];
                        dot *= scaling;
                        dot += bias[(k*L+j)*nh + h];
                        if (!mask.empty() && !mask[k*L+j]) dot = -inf_val;
                        scores[k] = dot;
                    }
                    math::softmax_row(scores.data(), L);
                    for (int k = 0; k < L; ++k) {
                        float w = scores[k];
                        for (int d = 0; d < ch; ++d)
                            attn_out[(i*L+j)*pd + h*ch+d] += w * V[(k*L+j)*pd + h*ch+d];
                    }
                }

        math::vec_mul(attn_out.data(), gate.data(), LL * pd);
        linear_o.forward(attn_out.data(), output, LL);
    }
};

// ===================================================================
// SECTION E4: Full TriangularSelfAttentionBlock (with triangle ops)
// ===================================================================

struct FullTriangularSelfAttentionBlock {
    int sequence_state_dim = 0, pairwise_state_dim = 0;
    ESM1bLayerNorm layernorm_1;
    SequenceToPair seq_to_pair;
    PairToSequence pair_to_seq;
    GatedAttention seq_attention;
    TriangleMultiplicationOutgoing tri_mul_out;
    TriangleMultiplicationIncoming tri_mul_in;
    TriangleAttentionStartingNode tri_att_start;
    TriangleAttentionEndingNode tri_att_end;
    ResidueMLP mlp_seq, mlp_pair;

    FullTriangularSelfAttentionBlock() = default;
    FullTriangularSelfAttentionBlock(int sd, int pd, int shw, int phw,
                                     float /*dropout*/, std::mt19937& rng)
        : sequence_state_dim(sd), pairwise_state_dim(pd) {
        int sh = sd / shw, ph = pd / phw;
        layernorm_1 = ESM1bLayerNorm(sd);
        seq_to_pair = SequenceToPair(sd, pd / 2, pd, rng);
        pair_to_seq = PairToSequence(pd, sh, rng);
        seq_attention = GatedAttention(sd, sh, shw, true, rng);
        tri_mul_out = TriangleMultiplicationOutgoing(pd, pd, rng);
        tri_mul_in = TriangleMultiplicationIncoming(pd, pd, rng);
        tri_att_start = TriangleAttentionStartingNode(pd, phw, ph, 1e9f, rng);
        tri_att_end = TriangleAttentionEndingNode(pd, phw, ph, 1e9f, rng);
        mlp_seq = ResidueMLP(sd, 4 * sd, rng);
        mlp_pair = ResidueMLP(pd, 4 * pd, rng);
    }

    void forward(float* seq_state, float* pair_state, int L,
                 const std::vector<bool>& mask = {}) const {
        int sd = sequence_state_dim, pd = pairwise_state_dim;
        int nh = sd / 32;

        std::vector<float> bias(L * L * nh);
        pair_to_seq.forward(pair_state, L, bias.data());

        std::vector<float> normed(L * sd);
        math::vec_copy(normed.data(), seq_state, L * sd);
        layernorm_1.forward_batch(normed.data(), L);
        std::vector<float> ao(L * sd);
        seq_attention.forward(normed.data(), L, ao.data(), bias.data(), mask);
        math::vec_add(seq_state, ao.data(), L * sd);
        mlp_seq.forward(seq_state, L);

        std::vector<float> pu(L * L * pd);
        seq_to_pair.forward(seq_state, L, pu.data());
        math::vec_add(pair_state, pu.data(), L * L * pd);

        std::vector<bool> tri_mask;
        if (!mask.empty()) {
            tri_mask.resize(L * L, true);
            for (int i = 0; i < L; ++i)
                for (int j = 0; j < L; ++j)
                    tri_mask[i * L + j] = mask[i] && mask[j];
        }

        { std::vector<float> u(L*L*pd); tri_mul_out.forward(u.data(), pair_state, L, tri_mask); math::vec_add(pair_state, u.data(), L*L*pd); }
        { std::vector<float> u(L*L*pd); tri_mul_in.forward(u.data(), pair_state, L, tri_mask); math::vec_add(pair_state, u.data(), L*L*pd); }
        { std::vector<float> u(L*L*pd); tri_att_start.forward(u.data(), pair_state, L, tri_mask); math::vec_add(pair_state, u.data(), L*L*pd); }
        { std::vector<float> u(L*L*pd); tri_att_end.forward(u.data(), pair_state, L, tri_mask); math::vec_add(pair_state, u.data(), L*L*pd); }

        mlp_pair.forward(pair_state, L * L);
    }
};

// ===================================================================
// SECTION E5: Invariant Point Attention (IPA)
// ===================================================================

struct InvariantPointAttention {
    int c_s = 384, c_z = 128, c_hidden = 16, c_hidden_v = 16;
    int no_heads = 12, no_qk_points = 4, no_v_points = 8;
    Linear linear_q, linear_kv, linear_q_points, linear_kv_points, linear_b, linear_out;
    float head_scale = 1.0f, point_weight = 0.0f;

    InvariantPointAttention() = default;
    InvariantPointAttention(int cs, int cz, int ch, int chv,
                            int nh, int nqk, int nv, std::mt19937& rng)
        : c_s(cs), c_z(cz), c_hidden(ch), c_hidden_v(chv),
          no_heads(nh), no_qk_points(nqk), no_v_points(nv) {
        head_scale = 1.0f / std::sqrt(static_cast<float>(ch));
        point_weight = std::sqrt(2.0f / (9.0f * nqk));
        linear_q = Linear(cs, ch * nh, true, rng);
        linear_kv = Linear(cs, (ch + chv) * nh, true, rng);
        linear_q_points = Linear(cs, nh * nqk * 3, true, rng);
        linear_kv_points = Linear(cs, nh * (nqk + nv) * 3, true, rng);
        linear_b = Linear(cz, nh, true, rng);
        int out_in = nh * (chv + nv * 4);
        linear_out = Linear(out_in, cs, true, rng);
        linear_out.weight.zeros();
        for (auto& bv : linear_out.bias) bv = 0.0f;
    }

    void forward(const float* seq_state, const float* pair_state,
                 const std::vector<Rigid>& rigids, int L, float* output,
                 const std::vector<bool>& mask = {}) const {
        int nh = no_heads, ch = c_hidden, chv = c_hidden_v;
        int nqk = no_qk_points, nv = no_v_points;
        int pd_kv = (ch + chv) * nh;

        std::vector<float> Q(L * ch * nh), KV(L * pd_kv);
        linear_q.forward(seq_state, Q.data(), L);
        linear_kv.forward(seq_state, KV.data(), L);

        std::vector<float> qp(L * nh * nqk * 3), kvp(L * nh * (nqk + nv) * 3);
        linear_q_points.forward(seq_state, qp.data(), L);
        linear_kv_points.forward(seq_state, kvp.data(), L);

        std::vector<float> bias(L * L * nh);
        linear_b.forward(pair_state, bias.data(), L * L);

        int oph = chv + nv * 4;
        std::vector<float> ipa_out(L * nh * oph, 0.0f);

        for (int h = 0; h < nh; ++h)
            for (int qi = 0; qi < L; ++qi) {
                std::vector<float> scores(L);
                for (int ki = 0; ki < L; ++ki) {
                    float dot = 0.0f;
                    for (int d = 0; d < ch; ++d)
                        dot += Q[qi*ch*nh + h*ch+d] * KV[ki*pd_kv + h*(ch+chv)+d];
                    dot *= head_scale;
                    float ps = 0.0f;
                    for (int p = 0; p < nqk; ++p) {
                        int qo = qi*nh*nqk*3 + h*nqk*3 + p*3;
                        int ko = ki*nh*(nqk+nv)*3 + h*(nqk+nv)*3 + p*3;
                        Vec3 ql(qp[qo], qp[qo+1], qp[qo+2]);
                        Vec3 kl(kvp[ko], kvp[ko+1], kvp[ko+2]);
                        Vec3 diff = rigids[qi].apply(ql) - rigids[ki].apply(kl);
                        ps += diff.norm_sq();
                    }
                    dot -= point_weight * ps;
                    dot += bias[(qi*L+ki)*nh + h];
                    if (!mask.empty() && !mask[ki]) dot = -1e9f;
                    scores[ki] = dot;
                }
                math::softmax_row(scores.data(), L);
                float* op = ipa_out.data() + qi*nh*oph + h*oph;
                for (int ki = 0; ki < L; ++ki) {
                    float w = scores[ki];
                    for (int d = 0; d < chv; ++d)
                        op[d] += w * KV[ki*pd_kv + h*(ch+chv)+ch+d];
                    for (int p = 0; p < nv; ++p) {
                        int vo = ki*nh*(nqk+nv)*3 + h*(nqk+nv)*3 + (nqk+p)*3;
                        Vec3 vl(kvp[vo], kvp[vo+1], kvp[vo+2]);
                        Vec3 vg = rigids[ki].apply(vl);
                        Vec3 vq = rigids[qi].inverse().apply(vg);
                        op[chv+p*4+0] += w * vq.x;
                        op[chv+p*4+1] += w * vq.y;
                        op[chv+p*4+2] += w * vq.z;
                        op[chv+p*4+3] += w * vq.norm();
                    }
                }
            }
        linear_out.forward(ipa_out.data(), output, L);
    }
};

// ===================================================================
// SECTION E6: Structure Module
// ===================================================================

struct StructureModuleConfig {
    int c_s = 384, c_z = 128, c_ipa = 16, c_resnet = 128;
    int no_heads_ipa = 12, no_qk_points = 4, no_v_points = 8;
    float dropout_rate = 0.1f;
    int no_blocks = 8, no_transition_layers = 1, no_resnet_blocks = 2, no_angles = 7;
    float trans_scale_factor = 10.0f, epsilon = 1e-8f, inf = 1e5f;
};

struct AngleResnet {
    int c_in = 384, c_hidden = 128, no_blocks = 2, no_angles = 7;
    Linear linear_in, linear_initial, linear_out;
    std::vector<std::pair<Linear, Linear>> blocks;

    AngleResnet() = default;
    AngleResnet(int cin, int ch, int nb, int na, std::mt19937& rng)
        : c_in(cin), c_hidden(ch), no_blocks(nb), no_angles(na) {
        linear_in = Linear(cin, ch, true, rng);
        linear_initial = Linear(cin, ch, true, rng);
        blocks.reserve(nb);
        for (int i = 0; i < nb; ++i)
            blocks.emplace_back(Linear(ch, ch, true, rng), Linear(ch, ch, true, rng));
        linear_out = Linear(ch, na * 2, true, rng);
    }

    void forward(const float* s, const float* s_initial, float* out, int L) const {
        std::vector<float> x(L * c_hidden), init(L * c_hidden);
        linear_in.forward(s, x.data(), L);
        linear_initial.forward(s_initial, init.data(), L);
        math::vec_add(x.data(), init.data(), L * c_hidden);
        math::relu_inplace(x.data(), L * c_hidden);
        for (auto& [l1, l2] : blocks) {
            std::vector<float> res(L * c_hidden), tmp(L * c_hidden);
            math::vec_copy(res.data(), x.data(), L * c_hidden);
            l1.forward(x.data(), tmp.data(), L);
            math::relu_inplace(tmp.data(), L * c_hidden);
            l2.forward(tmp.data(), x.data(), L);
            math::vec_add(x.data(), res.data(), L * c_hidden);
            math::relu_inplace(x.data(), L * c_hidden);
        }
        linear_out.forward(x.data(), out, L);
        for (int i = 0; i < L; ++i)
            for (int a = 0; a < no_angles; ++a) {
                float& sx = out[i*no_angles*2 + a*2];
                float& sy = out[i*no_angles*2 + a*2 + 1];
                float n = std::sqrt(sx*sx + sy*sy + 1e-12f);
                sx /= n; sy /= n;
            }
    }
};

struct StructureModule {
    StructureModuleConfig cfg;
    ESM1bLayerNorm layer_norm_s, layer_norm_z;
    Linear linear_in;
    std::vector<InvariantPointAttention> ipa_layers;
    std::vector<ESM1bLayerNorm> ipa_norms;
    std::vector<Linear> transition_fc1, transition_fc2, transition_fc3;
    std::vector<ESM1bLayerNorm> transition_norms;
    Linear bb_update;
    AngleResnet angle_resnet;

    StructureModule() = default;
    StructureModule(const StructureModuleConfig& c, std::mt19937& rng) : cfg(c) {
        layer_norm_s = ESM1bLayerNorm(c.c_s); layer_norm_z = ESM1bLayerNorm(c.c_z);
        linear_in = Linear(c.c_s, c.c_s, true, rng);
        for (int i = 0; i < c.no_blocks; ++i) {
            ipa_layers.emplace_back(c.c_s, c.c_z, c.c_ipa, c.c_ipa,
                                    c.no_heads_ipa, c.no_qk_points, c.no_v_points, rng);
            ipa_norms.emplace_back(c.c_s);
            transition_fc1.emplace_back(c.c_s, c.c_s, true, rng);
            transition_fc2.emplace_back(c.c_s, c.c_s, true, rng);
            transition_fc3.emplace_back(c.c_s, c.c_s, true, rng);
            transition_norms.emplace_back(c.c_s);
        }
        bb_update = Linear(c.c_s, 6, true, rng);
        bb_update.weight.zeros(); for (auto& b : bb_update.bias) b = 0.0f;
        angle_resnet = AngleResnet(c.c_s, c.c_resnet, c.no_resnet_blocks, c.no_angles, rng);
    }

    struct Output {
        std::vector<std::vector<Rigid>> frames;
        std::vector<std::vector<float>> angles;
        std::vector<std::vector<Vec3>> positions;
        std::vector<float> final_s;
    };

    Output forward(const float* s_in, const float* z_in, int L,
                   const std::vector<bool>& mask = {}) const {
        int cs = cfg.c_s, cz = cfg.c_z;
        Output out;
        std::vector<float> s(L*cs), z(L*L*cz);
        math::vec_copy(s.data(), s_in, L*cs);
        math::vec_copy(z.data(), z_in, L*L*cz);
        layer_norm_s.forward_batch(s.data(), L);
        for (int i = 0; i < L*L; ++i) layer_norm_z.forward(z.data() + i*cz);
        { std::vector<float> sp(L*cs); linear_in.forward(s.data(), sp.data(), L); math::vec_copy(s.data(), sp.data(), L*cs); }

        std::vector<Rigid> rigids(L, Rigid::identity());
        std::vector<float> s_init(L*cs);
        math::vec_copy(s_init.data(), s.data(), L*cs);

        for (int blk = 0; blk < cfg.no_blocks; ++blk) {
            std::vector<float> ipa(L*cs);
            ipa_layers[blk].forward(s.data(), z.data(), rigids, L, ipa.data(), mask);
            math::vec_add(s.data(), ipa.data(), L*cs);
            ipa_norms[blk].forward_batch(s.data(), L);
            {
                std::vector<float> t1(L*cs), t2(L*cs), t3(L*cs);
                transition_fc1[blk].forward(s.data(), t1.data(), L);
                math::relu_inplace(t1.data(), L*cs);
                transition_fc2[blk].forward(t1.data(), t2.data(), L);
                math::relu_inplace(t2.data(), L*cs);
                transition_fc3[blk].forward(t2.data(), t3.data(), L);
                math::vec_add(s.data(), t3.data(), L*cs);
                transition_norms[blk].forward_batch(s.data(), L);
            }
            std::vector<float> bb(L*6);
            bb_update.forward(s.data(), bb.data(), L);
            for (int i = 0; i < L; ++i) {
                Vec3 dt(bb[i*6+3]*cfg.trans_scale_factor, bb[i*6+4]*cfg.trans_scale_factor, bb[i*6+5]*cfg.trans_scale_factor);
                float rx = bb[i*6], ry = bb[i*6+1], rz = bb[i*6+2];
                float angle = std::sqrt(rx*rx+ry*ry+rz*rz+1e-12f);
                float ha = angle*0.5f, sa = std::sin(ha)/angle;
                Quaternion dq(std::cos(ha), rx*sa, ry*sa, rz*sa);
                rigids[i] = rigids[i].compose(Rigid(dq.normalized(), dt));
            }
            out.frames.push_back(rigids);
            std::vector<float> angles(L*cfg.no_angles*2);
            angle_resnet.forward(s.data(), s_init.data(), angles.data(), L);
            out.angles.push_back(angles);
            std::vector<Vec3> pos(L*3);
            for (int i = 0; i < L; ++i) {
                pos[i*3+0] = rigids[i].apply(Vec3(-0.5272f, 1.3593f, 0.0f));
                pos[i*3+1] = rigids[i].apply(Vec3(0.0f, 0.0f, 0.0f));
                pos[i*3+2] = rigids[i].apply(Vec3(1.5233f, 0.0f, 0.0f));
            }
            out.positions.push_back(pos);
        }
        out.final_s.resize(L*cs);
        math::vec_copy(out.final_s.data(), s.data(), L*cs);
        return out;
    }
};

// ===================================================================
// SECTION E7: Full FoldingTrunk (with recycles and structure module)
// ===================================================================

struct FullFoldingTrunkConfig {
    int num_blocks = 48, sequence_state_dim = 1024, pairwise_state_dim = 128;
    int sequence_head_width = 32, pairwise_head_width = 32, position_bins = 32;
    int max_recycles = 4, recycle_bins = 15;
    StructureModuleConfig structure_module;
};

struct FullFoldingTrunk {
    FullFoldingTrunkConfig config;
    RelativePosition pairwise_pos_emb;
    std::vector<FullTriangularSelfAttentionBlock> blocks;
    ESM1bLayerNorm recycle_s_norm, recycle_z_norm;
    Embedding recycle_disto;
    Linear trunk2sm_s, trunk2sm_z;
    StructureModule structure_mod;

    FullFoldingTrunk() = default;
    FullFoldingTrunk(const FullFoldingTrunkConfig& cfg, std::mt19937& rng) : config(cfg) {
        int cs = cfg.sequence_state_dim, cz = cfg.pairwise_state_dim;
        pairwise_pos_emb = RelativePosition(cfg.position_bins, cz, rng);
        blocks.reserve(cfg.num_blocks);
        for (int i = 0; i < cfg.num_blocks; ++i)
            blocks.emplace_back(cs, cz, cfg.sequence_head_width, cfg.pairwise_head_width, 0.0f, rng);
        recycle_s_norm = ESM1bLayerNorm(cs); recycle_z_norm = ESM1bLayerNorm(cz);
        recycle_disto = Embedding(cfg.recycle_bins, cz, -1, rng);
        for (int d = 0; d < cz; ++d) recycle_disto.weight(0, d) = 0.0f;
        trunk2sm_s = Linear(cs, cfg.structure_module.c_s, true, rng);
        trunk2sm_z = Linear(cz, cfg.structure_module.c_z, true, rng);
        structure_mod = StructureModule(cfg.structure_module, rng);
    }

    struct TrunkOutput {
        std::vector<float> s_s, s_z;
        StructureModule::Output structure;
    };

    TrunkOutput forward(const float* seq_feats, const float* pair_feats,
                        const std::vector<int>& residx, int L,
                        const std::vector<bool>& mask = {}, int num_recycles = -1) const {
        int cs = config.sequence_state_dim, cz = config.pairwise_state_dim;
        int nr = (num_recycles < 0) ? config.max_recycles : num_recycles;
        nr = std::max(1, nr + 1);

        std::vector<float> s_s(L*cs), s_z(L*L*cz);
        std::vector<float> rec_s(L*cs, 0.0f), rec_z(L*L*cz, 0.0f);
        std::vector<int> rec_bins(L*L, 0);
        StructureModule::Output last_struct;

        for (int rec = 0; rec < nr; ++rec) {
            std::vector<float> rs(L*cs), rz(L*L*cz);
            math::vec_copy(rs.data(), rec_s.data(), L*cs);
            recycle_s_norm.forward_batch(rs.data(), L);
            math::vec_copy(rz.data(), rec_z.data(), L*L*cz);
            for (int i = 0; i < L*L; ++i) recycle_z_norm.forward(rz.data() + i*cz);
            for (int i = 0; i < L*L; ++i) {
                int bin = rec_bins[i];
                if (bin >= 0 && bin < config.recycle_bins)
                    math::vec_add(rz.data() + i*cz, recycle_disto.lookup(bin), cz);
            }

            math::vec_copy(s_s.data(), seq_feats, L*cs);
            math::vec_add(s_s.data(), rs.data(), L*cs);
            math::vec_copy(s_z.data(), pair_feats, L*L*cz);
            math::vec_add(s_z.data(), rz.data(), L*L*cz);

            std::vector<float> pe(L*L*cz);
            pairwise_pos_emb.forward(residx, L, pe.data(), cz);
            math::vec_add(s_z.data(), pe.data(), L*L*cz);

            for (auto& block : blocks)
                block.forward(s_s.data(), s_z.data(), L, mask);

            std::vector<float> sm_s(L*config.structure_module.c_s);
            std::vector<float> sm_z(L*L*config.structure_module.c_z);
            trunk2sm_s.forward(s_s.data(), sm_s.data(), L);
            trunk2sm_z.forward(s_z.data(), sm_z.data(), L*L);
            last_struct = structure_mod.forward(sm_s.data(), sm_z.data(), L, mask);

            math::vec_copy(rec_s.data(), s_s.data(), L*cs);
            math::vec_copy(rec_z.data(), s_z.data(), L*L*cz);

            if (!last_struct.positions.empty()) {
                auto& pos = last_struct.positions.back();
                for (int i = 0; i < L; ++i) {
                    Vec3 N=pos[i*3], CA=pos[i*3+1], C=pos[i*3+2];
                    Vec3 b=CA-N, c=C-CA, a=b.cross(c);
                    Vec3 CB_i = a*(-0.58273431f) + b*0.56802827f + c*(-0.54067466f) + CA;
                    for (int j = 0; j < L; ++j) {
                        Vec3 N2=pos[j*3], CA2=pos[j*3+1], C2=pos[j*3+2];
                        Vec3 b2=CA2-N2, c2=C2-CA2, a2=b2.cross(c2);
                        Vec3 CB_j = a2*(-0.58273431f) + b2*0.56802827f + c2*(-0.54067466f) + CA2;
                        float ds = (CB_i - CB_j).norm_sq();
                        int bin = 0;
                        for (int bk = 0; bk < config.recycle_bins-1; ++bk) {
                            float bnd = 3.375f + (21.375f-3.375f)*bk/(config.recycle_bins-2);
                            if (ds > bnd*bnd) bin++;
                        }
                        rec_bins[i*L+j] = bin;
                    }
                }
            }
        }
        return {std::move(s_s), std::move(s_z), std::move(last_struct)};
    }
};

// ===================================================================
// SECTION E8: ESMFold Model
// ===================================================================

struct FullESMFoldConfig {
    FullFoldingTrunkConfig trunk;
    int lddt_head_hid_dim = 128, distogram_bins = 64, lddt_bins = 50;
    std::string esm_type = "esm2_3B";
    bool use_esm_attn_map = true;
};

class ESMFoldModel {
public:
    FullESMFoldConfig config;
    ESMModel esm;
    FullFoldingTrunk trunk;
    std::mt19937 rng;
    std::vector<float> esm_s_combine;
    Linear esm_s_mlp_proj, esm_s_mlp_out;
    Linear esm_z_mlp_proj, esm_z_mlp_out;
    Embedding aa_embedding;
    Linear distogram_head, ptm_head, lm_head_proj;
    int n_tokens_embed = 23;

    ESMFoldModel() = default;
    ESMFoldModel(const FullESMFoldConfig& cfg, unsigned seed = 42)
        : config(cfg), rng(seed) {
        ESMConfig ec; ec.num_layers = 36; ec.embed_dim = 2560;
        ec.ffn_embed_dim = 10240; ec.attention_heads = 40;
        ec.model_type = ESMConfig::ModelType::ESM2;
        ec.use_rotary_embeddings = true; ec.token_dropout = true;
        esm = ESMModel(ec, seed);
        int cs = cfg.trunk.sequence_state_dim, cz = cfg.trunk.pairwise_state_dim;
        int ef = ec.embed_dim, ea = ec.num_layers * ec.attention_heads;
        esm_s_combine.assign(ec.num_layers + 1, 0.0f);
        esm_s_mlp_proj = Linear(ef, cs, true, rng);
        esm_s_mlp_out = Linear(cs, cs, true, rng);
        if (cfg.use_esm_attn_map) {
            esm_z_mlp_proj = Linear(ea, cz, true, rng);
            esm_z_mlp_out = Linear(cz, cz, true, rng);
        }
        aa_embedding = Embedding(n_tokens_embed, cs, 0, rng);
        trunk = FullFoldingTrunk(cfg.trunk, rng);
        distogram_head = Linear(cz, cfg.distogram_bins, true, rng);
        ptm_head = Linear(cz, cfg.distogram_bins, true, rng);
        lm_head_proj = Linear(cs, n_tokens_embed, true, rng);
    }

    void print_summary(std::ostream& os = std::cout) const {
        os << "ESMFold Model Summary\n"
           << "  ESM backbone: " << config.esm_type << "\n"
           << "  Trunk blocks: " << config.trunk.num_blocks << "\n"
           << "  Sequence dim: " << config.trunk.sequence_state_dim << "\n"
           << "  Pairwise dim: " << config.trunk.pairwise_state_dim << "\n"
           << "  SM blocks: " << config.trunk.structure_module.no_blocks << "\n"
           << "  Max recycles: " << config.trunk.max_recycles << "\n"
           << "  Distogram bins: " << config.distogram_bins << "\n";
    }
};

// ===================================================================
// SECTION E9: GVP Graph Convolution (simplified)
// ===================================================================

struct GVPConvExt {
    int si = 0, vi = 0, so = 0, vo = 0, se = 0, ve = 0;
    std::vector<GVPLayer> msg_layers;

    GVPConvExt() = default;
    GVPConvExt(int si_, int vi_, int so_, int vo_, int se_, int ve_,
               int n_layers, bool vg, std::mt19937& rng)
        : si(si_), vi(vi_), so(so_), vo(vo_), se(se_), ve(ve_) {
        int ms = 2*si+se, mv = 2*vi+ve;
        msg_layers.reserve(n_layers);
        if (n_layers == 1) {
            msg_layers.emplace_back(ms, mv, so, vo, false, false, false, rng);
        } else {
            msg_layers.emplace_back(ms, mv, so, vo, vg, true, true, rng);
            for (int i = 1; i < n_layers-1; ++i)
                msg_layers.emplace_back(so, vo, so, vo, vg, true, true, rng);
            msg_layers.emplace_back(so, vo, so, vo, false, false, false, rng);
        }
    }

    void forward(const float* ns, const float* nv, const float* es, const float* ev,
                 const int* src, const int* dst, int N, int E,
                 float* out_s, float* out_v) const {
        std::vector<float> ms_buf(E*so, 0.0f), mv_buf(E*vo*3, 0.0f);
        int msi = 2*si+se, mvi = 2*vi+ve;
        for (int e = 0; e < E; ++e) {
            int s_idx = src[e], d_idx = dst[e];
            if (s_idx < 0 || d_idx < 0) continue;
            std::vector<float> cs_buf(msi), cv_buf(mvi*3);
            math::vec_copy(cs_buf.data(), ns + s_idx*si, si);
            math::vec_copy(cs_buf.data()+si, es + e*se, se);
            math::vec_copy(cs_buf.data()+si+se, ns + d_idx*si, si);
            if (vi > 0 && nv) {
                math::vec_copy(cv_buf.data(), nv + s_idx*vi*3, vi*3);
                math::vec_copy(cv_buf.data()+vi*3, ev + e*ve*3, ve*3);
                math::vec_copy(cv_buf.data()+(vi+ve)*3, nv + d_idx*vi*3, vi*3);
            }
            std::vector<float> cur_s = cs_buf, cur_v = cv_buf;
            for (auto& layer : msg_layers) {
                std::vector<float> ns_tmp(layer.so), nv_tmp(layer.vo*3);
                layer.forward(cur_s.data(), cur_v.data(), ns_tmp.data(), nv_tmp.data(), 1);
                cur_s = ns_tmp; cur_v = nv_tmp;
            }
            math::vec_copy(ms_buf.data()+e*so, cur_s.data(), so);
            if (vo > 0) math::vec_copy(mv_buf.data()+e*vo*3, cur_v.data(), vo*3);
        }
        math::vec_zero(out_s, N*so);
        if (vo > 0) math::vec_zero(out_v, N*vo*3);
        std::vector<int> counts(N, 0);
        for (int e = 0; e < E; ++e) {
            int d = dst[e]; if (d < 0 || d >= N) continue;
            math::vec_add(out_s+d*so, ms_buf.data()+e*so, so);
            if (vo > 0) math::vec_add(out_v+d*vo*3, mv_buf.data()+e*vo*3, vo*3);
            counts[d]++;
        }
        for (int i = 0; i < N; ++i) if (counts[i] > 1) {
            float inv = 1.0f / counts[i];
            math::vec_scale(out_s+i*so, inv, so);
            if (vo > 0) math::vec_scale(out_v+i*vo*3, inv, vo*3);
        }
    }
};

struct GVPConvLayerExt {
    int ns = 0, nv = 0, es = 0, ev = 0;
    GVPConvExt conv;
    GVPLayerNorm norm1, norm2;
    GVPLayer ff1, ff2;

    GVPConvLayerExt() = default;
    GVPConvLayerExt(int ns_, int nv_, int es_, int ev_,
                    bool vg, int n_msg, std::mt19937& rng, float eps = 1e-4f)
        : ns(ns_), nv(nv_), es(es_), ev(ev_) {
        conv = GVPConvExt(ns, nv, ns, nv, es, ev, n_msg, vg, rng);
        norm1 = GVPLayerNorm(ns, nv, eps);
        norm2 = GVPLayerNorm(ns, nv, eps);
        ff1 = GVPLayer(ns, nv, 4*ns, 2*nv, vg, true, true, rng, eps);
        ff2 = GVPLayer(4*ns, 2*nv, ns, nv, false, false, false, rng, eps);
    }

    void forward(float* node_s, float* node_v, const float* edge_s, const float* edge_v,
                 const int* src, const int* dst, int N, int E) const {
        std::vector<float> dhs(N*ns), dhv(N*nv*3);
        conv.forward(node_s, node_v, edge_s, edge_v, src, dst, N, E, dhs.data(), dhv.data());
        math::vec_add(node_s, dhs.data(), N*ns);
        if (nv > 0 && node_v) math::vec_add(node_v, dhv.data(), N*nv*3);
        norm1.forward(node_s, node_v, N);
        std::vector<float> fs(N*ff1.so), fv(N*ff1.vo*3);
        ff1.forward(node_s, node_v, fs.data(), fv.data(), N);
        std::vector<float> fos(N*ns), fov(N*nv*3);
        ff2.forward(fs.data(), fv.data(), fos.data(), fov.data(), N);
        math::vec_add(node_s, fos.data(), N*ns);
        if (nv > 0 && node_v) math::vec_add(node_v, fov.data(), N*nv*3);
        norm2.forward(node_s, node_v, N);
    }
};

// ===================================================================
// SECTION E10: DihedralFeatures Module (Learnable)
// ===================================================================

struct DihedralFeaturesModule {
    Linear node_embedding;
    ESM1bLayerNorm norm;
    int embed_dim = 0;

    DihedralFeaturesModule() = default;
    DihedralFeaturesModule(int ed, std::mt19937& rng) : embed_dim(ed) {
        node_embedding = Linear(6, ed, true, rng);
        norm = ESM1bLayerNorm(ed);
    }

    void forward(const std::vector<std::array<Vec3, 3>>& coords, float* output) const {
        int L = static_cast<int>(coords.size());
        auto dih_features = features::compute_dihedral_features(coords);
        std::vector<float> flat(L * 6);
        for (int i = 0; i < L; ++i) for (int d = 0; d < 6; ++d)
            flat[i * 6 + d] = dih_features[i][d];
        node_embedding.forward(flat.data(), output, L);
        norm.forward_batch(output, L);
    }
};

// ===================================================================
// SECTION E11: NormalizedResidualBlock
// ===================================================================

struct NormalizedResidualBlock {
    ESM1bLayerNorm layer_norm;
    int embed_dim = 0;

    NormalizedResidualBlock() = default;
    explicit NormalizedResidualBlock(int ed) : embed_dim(ed), layer_norm(ed) {}

    void pre_norm(float* x, float* normed, int n) const {
        math::vec_copy(normed, x, n * embed_dim);
        layer_norm.forward_batch(normed, n);
    }
    void add_residual(float* x, const float* update, int n) const {
        math::vec_add(x, update, n * embed_dim);
    }
};

// ===================================================================
// SECTION E12: CoordBatchConverter
// ===================================================================

struct CoordBatchConverter {
    const Alphabet* alphabet = nullptr;

    CoordBatchConverter() = default;
    explicit CoordBatchConverter(const Alphabet* a) : alphabet(a) {}

    struct Result {
        std::vector<std::vector<std::array<Vec3, 3>>> coords;
        std::vector<std::vector<float>> confidence;
        std::vector<std::string> strs;
        std::vector<std::vector<int>> tokens;
        std::vector<std::vector<bool>> padding_mask;
    };

    Result convert(const std::vector<std::tuple<
            std::vector<std::array<Vec3, 3>>, std::vector<float>, std::string>>& batch) const {
        Result r;
        int bs = static_cast<int>(batch.size());
        r.coords.resize(bs); r.confidence.resize(bs);
        r.strs.resize(bs); r.tokens.resize(bs); r.padding_mask.resize(bs);
        int max_len = 0;
        for (auto& [c,cf,sq] : batch) max_len = std::max(max_len, static_cast<int>(c.size()));
        int tl = max_len + (alphabet->prepend_bos?1:0) + (alphabet->append_eos?1:0);
        int bos = alphabet->get_idx("<cath>");
        if (bos == alphabet->unk_idx) bos = alphabet->cls_idx;
        for (int b = 0; b < bs; ++b) {
            auto& [coords, conf, seq] = batch[b];
            int L = static_cast<int>(coords.size());
            r.coords[b] = coords;
            r.confidence[b] = conf.empty() ? std::vector<float>(L, 1.0f) : conf;
            std::string sq = seq.empty() ? std::string(L, 'X') : seq;
            r.strs[b] = sq;
            r.tokens[b].assign(tl, alphabet->pad_idx);
            int off = 0;
            if (alphabet->prepend_bos) { r.tokens[b][0] = bos; off = 1; }
            for (int i = 0; i < L; ++i)
                r.tokens[b][off+i] = alphabet->get_idx(std::string(1, sq[i]));
            if (alphabet->append_eos && L+off < tl)
                r.tokens[b][L+off] = alphabet->eos_idx;
            r.padding_mask[b].assign(tl, false);
            for (int i = L+off+(alphabet->append_eos?1:0); i < tl; ++i)
                r.padding_mask[b][i] = true;
        }
        return r;
    }
};

// ===================================================================
// SECTION E13: Multichain Utilities
// ===================================================================

namespace multichain {
using BackboneCoords = std::vector<std::array<Vec3, 3>>;
using ChainCoords = std::map<std::string, BackboneCoords>;

inline BackboneCoords concatenate_coords(const ChainCoords& coords,
        const std::string& target, int pad_len = 10) {
    std::array<Vec3,3> nan_a = {Vec3(NAN,NAN,NAN), Vec3(NAN,NAN,NAN), Vec3(NAN,NAN,NAN)};
    BackboneCoords pad(pad_len, nan_a), result;
    auto it = coords.find(target);
    if (it != coords.end())
        result.insert(result.end(), it->second.begin(), it->second.end());
    for (auto& [cid, cc] : coords) {
        if (cid == target) continue;
        result.insert(result.end(), pad.begin(), pad.end());
        result.insert(result.end(), cc.begin(), cc.end());
    }
    return result;
}
} // namespace multichain

// ===================================================================
// SECTION E14: PDB I/O
// ===================================================================

namespace pdb_io {
struct PDBAtom {
    std::string name; Vec3 coord;
    int residue_index = 0; std::string chain_id, residue_name;
    float b_factor = 0.0f;
};

inline std::vector<PDBAtom> parse_pdb(const std::string& path) {
    std::vector<PDBAtom> atoms;
    std::ifstream f(path); if (!f.is_open()) return atoms;
    std::string line;
    while (std::getline(f, line)) {
        if (line.size() < 54) continue;
        if (line.substr(0,4) != "ATOM" && line.substr(0,6) != "HETATM") continue;
        PDBAtom a;
        a.name = line.substr(12, 4);
        while (!a.name.empty() && a.name.front() == ' ') a.name.erase(a.name.begin());
        while (!a.name.empty() && a.name.back() == ' ') a.name.pop_back();
        a.residue_name = line.substr(17, 3);
        while (!a.residue_name.empty() && a.residue_name.front() == ' ') a.residue_name.erase(a.residue_name.begin());
        while (!a.residue_name.empty() && a.residue_name.back() == ' ') a.residue_name.pop_back();
        a.chain_id = line.substr(21, 1);
        try { a.residue_index = std::stoi(line.substr(22, 4)); } catch (...) { continue; }
        try {
            a.coord.x = std::stof(line.substr(30, 8));
            a.coord.y = std::stof(line.substr(38, 8));
            a.coord.z = std::stof(line.substr(46, 8));
        } catch (...) { continue; }
        if (line.size() >= 66) try { a.b_factor = std::stof(line.substr(60, 6)); } catch (...) {}
        atoms.push_back(a);
    }
    return atoms;
}

inline std::vector<std::array<Vec3, 3>> extract_backbone(const std::vector<PDBAtom>& atoms,
        const std::string& chain = "") {
    std::map<int, std::map<std::string, Vec3>> residues;
    for (auto& a : atoms) {
        if (!chain.empty() && a.chain_id != chain) continue;
        if (a.name == "N" || a.name == "CA" || a.name == "C")
            residues[a.residue_index][a.name] = a.coord;
    }
    std::vector<std::array<Vec3, 3>> coords;
    for (auto& [idx, am] : residues)
        if (am.count("N") && am.count("CA") && am.count("C"))
            coords.push_back({am["N"], am["CA"], am["C"]});
    return coords;
}

inline std::string extract_sequence(const std::vector<PDBAtom>& atoms, const std::string& chain = "") {
    static const std::unordered_map<std::string, char> t2o = {
        {"ALA",'A'},{"ARG",'R'},{"ASN",'N'},{"ASP",'D'},{"CYS",'C'},{"GLN",'Q'},
        {"GLU",'E'},{"GLY",'G'},{"HIS",'H'},{"ILE",'I'},{"LEU",'L'},{"LYS",'K'},
        {"MET",'M'},{"PHE",'F'},{"PRO",'P'},{"SER",'S'},{"THR",'T'},{"TRP",'W'},
        {"TYR",'Y'},{"VAL",'V'}};
    std::map<int, std::string> rn;
    for (auto& a : atoms) {
        if (!chain.empty() && a.chain_id != chain) continue;
        if (a.name == "CA") rn[a.residue_index] = a.residue_name;
    }
    std::string seq;
    for (auto& [i, n] : rn) { auto it = t2o.find(n); seq += (it != t2o.end()) ? it->second : 'X'; }
    return seq;
}

inline std::string write_pdb(const std::vector<Vec3>& pos, const std::string& seq,
                              const std::vector<float>& plddt = {}) {
    static const std::vector<std::string> anames = {"N", "CA", "C"};
    static const std::unordered_map<char, std::string> o2t = {
        {'A',"ALA"},{'R',"ARG"},{'N',"ASN"},{'D',"ASP"},{'C',"CYS"},{'Q',"GLN"},
        {'E',"GLU"},{'G',"GLY"},{'H',"HIS"},{'I',"ILE"},{'L',"LEU"},{'K',"LYS"},
        {'M',"MET"},{'F',"PHE"},{'P',"PRO"},{'S',"SER"},{'T',"THR"},{'W',"TRP"},
        {'Y',"TYR"},{'V',"VAL"},{'X',"UNK"}};
    std::ostringstream oss;
    int anum = 1, L = static_cast<int>(seq.size());
    for (int i = 0; i < L && i*3+2 < static_cast<int>(pos.size()); ++i) {
        auto it = o2t.find(seq[i]);
        std::string rn = (it != o2t.end()) ? it->second : "UNK";
        float bf = (!plddt.empty() && i < static_cast<int>(plddt.size())) ? plddt[i]*100.0f : 0.0f;
        for (int a = 0; a < 3; ++a) {
            char buf[120];
            std::snprintf(buf, sizeof(buf),
                "ATOM  %5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00%6.2f           %c\n",
                anum, anames[a].c_str(), rn.c_str(), i+1,
                pos[i*3+a].x, pos[i*3+a].y, pos[i*3+a].z, bf, anames[a][0]);
            oss << buf; anum++;
        }
    }
    oss << "END\n";
    return oss.str();
}
} // namespace pdb_io

// ===================================================================
// SECTION E15: Tokenization, Incremental State, MSA Position Embedding
// ===================================================================

namespace tokenization {
inline std::vector<std::string> tokenize(const Alphabet& a, const std::string& text) {
    std::vector<std::string> tokens;
    for (char c : text) tokens.push_back(std::string(1, c));
    return tokens;
}

inline std::vector<std::string> full_tokenize(const Alphabet& a, const std::string& text) {
    std::vector<std::string> result;
    std::string current;
    for (size_t i = 0; i < text.size(); ++i) {
        bool found = false;
        for (auto& tok : a.all_toks) {
            if (tok.size() > 1 && tok[0] == '<' && tok.back() == '>') {
                if (i + tok.size() <= text.size() && text.substr(i, tok.size()) == tok) {
                    for (char c : current) result.push_back(std::string(1, c));
                    current.clear();
                    result.push_back(tok);
                    i += tok.size() - 1; found = true; break;
                }
            }
        }
        if (!found) current += text[i];
    }
    for (char c : current) result.push_back(std::string(1, c));
    return result;
}
} // namespace tokenization

struct IncrementalState {
    struct LayerCache {
        std::vector<float> prev_key, prev_value;
        int cached_len = 0, embed_dim = 0;
        void append(const float* k, const float* v, int ed) {
            embed_dim = ed;
            prev_key.insert(prev_key.end(), k, k+ed);
            prev_value.insert(prev_value.end(), v, v+ed);
            cached_len++;
        }
        void clear() { prev_key.clear(); prev_value.clear(); cached_len = 0; }
    };
    std::map<int, LayerCache> self_attn_cache, encoder_attn_cache;
    void clear() { self_attn_cache.clear(); encoder_attn_cache.clear(); }
};

struct MSAPositionEmbedding {
    Matrix weight;
    int max_depth = 1024, embed_dim = 0;

    MSAPositionEmbedding() = default;
    MSAPositionEmbedding(int md, int ed, std::mt19937& rng) : max_depth(md), embed_dim(ed) {
        weight = Matrix(md, ed);
        std::normal_distribution<float> dist(0.0f, 0.01f);
        for (auto& v : weight.data) v = dist(rng);
    }
    const float* lookup(int idx) const {
        if (idx < 0 || idx >= max_depth) return nullptr;
        return weight.row_ptr(idx);
    }
};

struct SharedDropout {
    float rate = 0.0f; int share_dim = -1;
    SharedDropout() = default;
    SharedDropout(float r, int d) : rate(r), share_dim(d) {}
    void forward(float*, int) const {} // no-op in inference
};

// ===================================================================
// SECTION E16: Extended Pretrained Registry
// ===================================================================

namespace pretrained {

struct MSAModelInfo { std::string name; int nl, ed, ffn, nh; };
inline const std::vector<MSAModelInfo>& msa_model_registry() {
    static const std::vector<MSAModelInfo> r = {
        {"esm_msa1_t12_100M_UR50S", 12, 768, 3072, 12},
        {"esm_msa1b_t12_100M_UR50S", 12, 768, 3072, 12},
    };
    return r;
}

inline MSATransformerModel create_msa_model(const std::string& name, unsigned seed = 42) {
    for (auto& i : msa_model_registry())
        if (i.name == name) {
            MSATransformerConfig c; c.num_layers=i.nl; c.embed_dim=i.ed;
            c.ffn_embed_dim=i.ffn; c.attention_heads=i.nh;
            return MSATransformerModel(c, seed);
        }
    throw std::runtime_error("Unknown MSA model: " + name);
}

struct ESMIFInfo { std::string name; int gl, el, dl, eed, ded, effn, dffn, eah, dah; };
inline const std::vector<ESMIFInfo>& esm_if_registry() {
    static const std::vector<ESMIFInfo> r = {
        {"esm_if1_gvp4_t16_142M_UR50", 4, 8, 8, 512, 512, 2048, 2048, 8, 8},
    };
    return r;
}

inline GVPTransformerModel create_esm_if(const std::string& name, unsigned seed = 42) {
    for (auto& i : esm_if_registry())
        if (i.name == name) {
            std::mt19937 rng(seed);
            GVPTransformerConfig c;
            c.encoder_embed_dim=i.eed; c.encoder_ffn_embed_dim=i.effn;
            c.encoder_attention_heads=i.eah; c.encoder_layers=i.el;
            c.decoder_config.decoder_embed_dim=i.ded;
            c.decoder_config.decoder_ffn_embed_dim=i.dffn;
            c.decoder_config.decoder_attention_heads=i.dah;
            c.decoder_config.decoder_layers=i.dl;
            c.decoder_config.encoder_embed_dim=i.eed;
            c.gvp_config.num_encoder_layers=i.gl;
            return GVPTransformerModel(c, rng);
        }
    throw std::runtime_error("Unknown ESM-IF model: " + name);
}

struct ESMFoldInfo { std::string name; int tb; std::string esm; };
inline const std::vector<ESMFoldInfo>& esmfold_registry() {
    static const std::vector<ESMFoldInfo> r = {
        {"esmfold_v0", 48, "esm2_3B"}, {"esmfold_v1", 48, "esm2_3B"},
        {"esmfold_structure_module_only_8M", 0, "esm2_8M"},
        {"esmfold_structure_module_only_35M", 0, "esm2_35M"},
        {"esmfold_structure_module_only_150M", 0, "esm2_150M"},
        {"esmfold_structure_module_only_650M", 0, "esm2_650M"},
        {"esmfold_structure_module_only_3B", 0, "esm2_3B"},
        {"esmfold_structure_module_only_15B", 0, "esm2_15B"},
    };
    return r;
}

inline std::vector<std::string> list_all_models() {
    std::vector<std::string> n;
    for (auto& m : model_registry()) n.push_back(m.name);
    for (auto& m : msa_model_registry()) n.push_back(m.name);
    for (auto& m : esm_if_registry()) n.push_back(m.name);
    for (auto& m : esmfold_registry()) n.push_back(m.name);
    return n;
}
} // namespace pretrained

// ===================================================================
// SECTION E17: Extended Feature Extraction
// ===================================================================

namespace features {

inline Matrix attention_contact_map(const std::vector<Tensor3D>& attn,
                                     int start = 0, int end = -1) {
    if (attn.empty()) return Matrix();
    int sl = attn[0].d1; if (end < 0) end = sl;
    int L = end - start, nl = static_cast<int>(attn.size()), nh = attn[0].d0;
    Matrix contacts(L, L, 0.0f);
    for (int l = 0; l < nl; ++l)
        for (int h = 0; h < nh; ++h) {
            Matrix a(L, L);
            for (int i = 0; i < L; ++i) for (int j = 0; j < L; ++j)
                a(i, j) = attn[l](h, start+i, start+j);
            math::symmetrize(a.data.data(), L);
            math::apc(a.data.data(), L);
            for (int i = 0; i < L*L; ++i) contacts.data[i] += a.data[i];
        }
    float inv = 1.0f / (nl * nh);
    for (auto& v : contacts.data) v *= inv;
    return contacts;
}

inline std::vector<float> per_token_perplexity(const Matrix& logits, const std::vector<int>& tokens) {
    int sl = std::min(logits.rows, static_cast<int>(tokens.size())), vs = logits.cols;
    std::vector<float> ppl(sl);
    for (int i = 0; i < sl; ++i) {
        const float* r = logits.row_ptr(i);
        float mx = *std::max_element(r, r+vs), se = 0.0f;
        for (int j = 0; j < vs; ++j) se += std::exp(r[j]-mx);
        float lp = r[tokens[i]] - mx - std::log(se);
        ppl[i] = std::exp(-lp);
    }
    return ppl;
}

inline float pseudo_likelihood_score(const ESMModel& m, const std::string& seq) {
    auto tok = m.alphabet.encode(seq);
    int sl = static_cast<int>(tok.size());
    int s = m.alphabet.prepend_bos?1:0, e = sl-(m.alphabet.append_eos?1:0);
    float tll = 0.0f; int cnt = 0;
    for (int p = s; p < e; ++p) {
        auto mt = tok; mt[p] = m.alphabet.mask_idx;
        auto o = m.forward(mt);
        const float* lg = o.logits.row_ptr(p);
        int vs = o.logits.cols;
        float mx = *std::max_element(lg, lg+vs), se = 0.0f;
        for (int j = 0; j < vs; ++j) se += std::exp(lg[j]-mx);
        tll += lg[tok[p]] - mx - std::log(se); cnt++;
    }
    return cnt > 0 ? tll / cnt : 0.0f;
}

inline float mutation_effect(const ESMModel& m, const std::string& wt, int pos, char mut) {
    auto tok = m.alphabet.encode(wt);
    int tp = pos + (m.alphabet.prepend_bos?1:0);
    auto mt = tok; mt[tp] = m.alphabet.mask_idx;
    auto o = m.forward(mt);
    const float* lg = o.logits.row_ptr(tp);
    int vs = o.logits.cols;
    float mx = *std::max_element(lg, lg+vs), se = 0.0f;
    for (int j = 0; j < vs; ++j) se += std::exp(lg[j]-mx);
    float ls = mx + std::log(se);
    int wi = m.alphabet.get_idx(std::string(1, wt[pos]));
    int mi = m.alphabet.get_idx(std::string(1, mut));
    return (lg[mi] - ls) - (lg[wi] - ls);
}

} // namespace features

// ===================================================================
// SECTION E18: Residue Constants (from OpenFold)
// ===================================================================

namespace residue_constants {
inline const std::vector<std::string>& restypes() {
    static const std::vector<std::string> t = {
        "A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"};
    return t;
}
inline const std::vector<std::string>& restypes_with_x() {
    static const std::vector<std::string> t = {
        "A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V","X"};
    return t;
}
inline int restype_num() { return 20; }
inline const std::unordered_map<std::string, int>& restype_order() {
    static const std::unordered_map<std::string, int> o = {
        {"A",0},{"R",1},{"N",2},{"D",3},{"C",4},{"Q",5},{"E",6},{"G",7},
        {"H",8},{"I",9},{"L",10},{"K",11},{"M",12},{"F",13},{"P",14},
        {"S",15},{"T",16},{"W",17},{"Y",18},{"V",19}};
    return o;
}
inline const std::unordered_map<std::string, int>& restype_order_with_x() {
    static const std::unordered_map<std::string, int> o = {
        {"A",0},{"R",1},{"N",2},{"D",3},{"C",4},{"Q",5},{"E",6},{"G",7},
        {"H",8},{"I",9},{"L",10},{"K",11},{"M",12},{"F",13},{"P",14},
        {"S",15},{"T",16},{"W",17},{"Y",18},{"V",19},{"X",20}};
    return o;
}
inline float bond_length_n_ca() { return 1.458f; }
inline float bond_length_ca_c() { return 1.523f; }
inline float bond_length_c_n()  { return 1.329f; }
} // namespace residue_constants

} // namespace esm

#endif // ESM_EXTENDED_H
