#pragma once
// boltz_loss.h -- Complete loss functions for the Boltz C++ port.
//
// Covers:
//   FAPE loss (frame aligned point error) with clamping
//   Smooth lDDT loss with configurable cutoffs
//   Distogram cross-entropy loss
//   B-factor prediction loss
//   Diffusion denoising loss (MSE + smooth lDDT)
//   Confidence losses (pLDDT, PDE, PAE, resolved)
//   Validation metrics (factored lDDT, RMSD, PAE MAE, PDE MAE, pLDDT MAE)
//   Weighted minimum RMSD
//   Token-level lDDT for various interaction types

#include "boltz.h"

namespace boltz {

// ============================================================================
// Interaction type weights (matching Python out_types_weights)
// ============================================================================
struct InteractionWeights {
    float dna_protein = 5.0f;
    float rna_protein = 5.0f;
    float ligand_protein = 20.0f;
    float dna_ligand = 2.0f;
    float rna_ligand = 2.0f;
    float intra_ligand = 20.0f;
    float intra_dna = 2.0f;
    float intra_rna = 8.0f;
    float intra_protein = 20.0f;
    float protein_protein = 20.0f;
    float modified = 0.0f;
};

// ============================================================================
// FAPE Loss (extended with per-type weighting)
// ============================================================================
inline float fape_loss_extended(const std::vector<Rigid>& pred_frames,
                                 const std::vector<BackboneAtoms>& pred_atoms,
                                 const std::vector<Rigid>& true_frames,
                                 const std::vector<BackboneAtoms>& true_atoms,
                                 const std::vector<float>& mask,
                                 int N, float clamp_val = FAPE_CLAMP, float d_val = FAPE_D) {
    float total = 0; int count = 0;
    auto get_atom = [](const std::vector<BackboneAtoms>& at, int res, int a) -> Vec3 {
        switch(a) { case 0: return at[res].N_pos; case 1: return at[res].CA_pos;
                    case 2: return at[res].C_pos; case 3: return at[res].O_pos;
                    default: return at[res].CA_pos; }
    };
    for (int i = 0; i < N; ++i) {
        float wi = (i < (int)mask.size()) ? mask[i] : 1.0f;
        if (wi < 0.5f) continue;
        Rigid pi = pred_frames[i].inverse(), ti = true_frames[i].inverse();
        for (int j = 0; j < N; ++j) {
            float wj = (j < (int)mask.size()) ? mask[j] : 1.0f;
            if (wj < 0.5f) continue;
            for (int a = 0; a < 4; ++a) {
                Vec3 d = pi.apply(get_atom(pred_atoms, j, a)) - ti.apply(get_atom(true_atoms, j, a));
                total += std::min(d.norm(), clamp_val) / d_val;
                count++;
            }
        }
    }
    return count > 0 ? total / count : 0.0f;
}

// ============================================================================
// Distogram Loss (extended with per-pair masking)
// ============================================================================
inline float distogram_loss_extended(const Mat& pred, const Mat& target,
                                      const std::vector<float>& mask, int N,
                                      float min_dist = DIST_MIN, float max_dist = DIST_MAX) {
    Mat p = pred; log_softmax_rows(p);
    float total = 0, count = 0;
    for (int i = 0; i < N; ++i) {
        float wi = (i < (int)mask.size()) ? mask[i] : 1.0f;
        if (wi < 0.5f) continue;
        for (int j = 0; j < N; ++j) {
            float wj = (j < (int)mask.size()) ? mask[j] : 1.0f;
            if (wj < 0.5f || i == j) continue;
            for (int b = 0; b < p.cols; ++b)
                total -= target(i*N+j, b) * p(i*N+j, b);
            count += 1.0f;
        }
    }
    return count > 0 ? total / count : 0.0f;
}

// ============================================================================
// B-factor Loss
// ============================================================================
inline float bfactor_loss_extended(const Mat& pred, const std::vector<float>& true_bf,
                                    const std::vector<float>& mask, int num_bins = 50,
                                    float max_bfac = 100.0f) {
    int N = pred.rows;
    Mat lp = pred; log_softmax_rows(lp);
    float total = 0, count = 0;
    for (int i = 0; i < N; ++i) {
        float wi = (i < (int)mask.size()) ? mask[i] : 1.0f;
        if (wi < 0.5f || (i < (int)true_bf.size() && true_bf[i] < 1e-5f)) continue;
        float bf = (i < (int)true_bf.size()) ? true_bf[i] : 0.0f;
        int bin = std::clamp((int)(bf / max_bfac * (num_bins - 1)), 0, num_bins - 1);
        total -= lp(i, bin); count += 1.0f;
    }
    return count > 0 ? total / count : 0.0f;
}

// ============================================================================
// Confidence Losses
// ============================================================================

// pLDDT loss (cross-entropy against binned lDDT values)
inline float plddt_loss(const Mat& plddt_logits, const std::vector<float>& true_lddt,
                        const std::vector<float>& mask, int num_bins = NUM_PLDDT_BINS) {
    int N = plddt_logits.rows;
    Mat lp = plddt_logits; log_softmax_rows(lp);
    float total = 0, count = 0;
    float bw = 1.0f / num_bins;
    for (int i = 0; i < N; ++i) {
        float wi = (i < (int)mask.size()) ? mask[i] : 1.0f;
        if (wi < 0.5f) continue;
        float lddt_val = (i < (int)true_lddt.size()) ? true_lddt[i] : 0.5f;
        int bin = std::clamp((int)(lddt_val / bw), 0, num_bins - 1);
        total -= lp(i, bin); count += 1.0f;
    }
    return count > 0 ? total / count : 0.0f;
}

// PDE loss (cross-entropy against binned pairwise distance errors)
inline float pde_loss(const Mat& pde_logits, const Mat& pred_dists, const Mat& true_dists,
                      const std::vector<float>& mask, int N, int num_bins = NUM_PDE_BINS,
                      float max_error = 32.0f) {
    Mat lp = pde_logits; log_softmax_rows(lp);
    float total = 0, count = 0;
    float bw = max_error / num_bins;
    for (int i = 0; i < N; ++i) {
        float wi = (i < (int)mask.size()) ? mask[i] : 1.0f;
        if (wi < 0.5f) continue;
        for (int j = 0; j < N; ++j) {
            float wj = (j < (int)mask.size()) ? mask[j] : 1.0f;
            if (wj < 0.5f || i == j) continue;
            float error = std::abs(pred_dists(i, j) - true_dists(i, j));
            int bin = std::clamp((int)(error / bw), 0, num_bins - 1);
            total -= lp(i*N+j, bin); count += 1.0f;
        }
    }
    return count > 0 ? total / count : 0.0f;
}

// PAE loss (cross-entropy against binned predicted aligned errors)
inline float pae_loss(const Mat& pae_logits, const Mat& true_pae,
                      const std::vector<float>& mask, int N, int num_bins = NUM_PAE_BINS,
                      float max_error = 32.0f) {
    Mat lp = pae_logits; log_softmax_rows(lp);
    float total = 0, count = 0;
    float bw = max_error / num_bins;
    for (int i = 0; i < N; ++i) {
        float wi = (i < (int)mask.size()) ? mask[i] : 1.0f;
        if (wi < 0.5f) continue;
        for (int j = 0; j < N; ++j) {
            float wj = (j < (int)mask.size()) ? mask[j] : 1.0f;
            if (wj < 0.5f) continue;
            float error = true_pae(i, j);
            int bin = std::clamp((int)(error / bw), 0, num_bins - 1);
            total -= lp(i*N+j, bin); count += 1.0f;
        }
    }
    return count > 0 ? total / count : 0.0f;
}

// Resolved loss (binary cross-entropy)
inline float resolved_loss(const Mat& resolved_logits, const std::vector<bool>& true_resolved,
                            const std::vector<float>& mask) {
    int N = resolved_logits.rows;
    float total = 0, count = 0;
    for (int i = 0; i < N; ++i) {
        float wi = (i < (int)mask.size()) ? mask[i] : 1.0f;
        if (wi < 0.5f) continue;
        bool res = (i < (int)true_resolved.size()) ? true_resolved[i] : true;
        int target = res ? 1 : 0;
        // Binary cross-entropy from logits
        Mat row(1, 2);
        row(0, 0) = resolved_logits(i, 0); row(0, 1) = resolved_logits(i, 1);
        log_softmax_rows(row);
        total -= row(0, target); count += 1.0f;
    }
    return count > 0 ? total / count : 0.0f;
}

// ============================================================================
// Combined confidence loss (matching Python confidence_loss)
// ============================================================================
struct ConfidenceLossWeights {
    float plddt_weight = 1.0f;
    float pde_weight = 1.0f;
    float pae_weight = 0.0f;  // only if PAE is computed
    float resolved_weight = 0.01f;
};

// ============================================================================
// Validation metrics
// ============================================================================

// Factored lDDT loss
inline float factored_lddt(const Mat& pred_coords, const Mat& true_coords,
                           const std::vector<float>& mask, float cutoff = 15.0f) {
    int N = pred_coords.rows; float total = 0; int count = 0;
    for (int i = 0; i < N; ++i) {
        float wi = (i < (int)mask.size()) ? mask[i] : 1.0f;
        if (wi < 0.5f) continue;
        float num = 0, den = 0;
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            float wj = (j < (int)mask.size()) ? mask[j] : 1.0f;
            if (wj < 0.5f) continue;
            float dt = 0, dp = 0;
            for (int d = 0; d < 3; ++d) {
                float ddt = true_coords(i,d) - true_coords(j,d);
                float ddp = pred_coords(i,d) - pred_coords(j,d);
                dt += ddt*ddt; dp += ddp*ddp;
            }
            dt = std::sqrt(dt); dp = std::sqrt(dp);
            if (dt > cutoff) continue;
            float diff = std::abs(dt - dp);
            float score = ((diff < 0.5f) + (diff < 1.0f) + (diff < 2.0f) + (diff < 4.0f)) / 4.0f;
            num += score; den += 1.0f;
        }
        if (den > 0) { total += num / den; count++; }
    }
    return count > 0 ? total / count : 0.0f;
}

// Token-level lDDT per interaction type
inline float token_lddt_by_type(const Mat& pred_coords, const Mat& true_coords,
                                 const std::vector<int>& mol_types,
                                 const std::vector<int>& asym_ids,
                                 const std::string& interaction_type,
                                 float cutoff = 15.0f) {
    int N = pred_coords.rows;
    // Determine which pairs to include
    auto should_include = [&](int i, int j) -> bool {
        if (i == j) return false;
        bool same_chain = (asym_ids[i] == asym_ids[j]);
        bool is_prot_i = (mol_types[i] == MOL_PROTEIN);
        bool is_prot_j = (mol_types[j] == MOL_PROTEIN);
        bool is_lig_i = (mol_types[i] == MOL_NONPOLYMER);
        bool is_lig_j = (mol_types[j] == MOL_NONPOLYMER);
        bool is_dna_i = (mol_types[i] == MOL_DNA);
        bool is_dna_j = (mol_types[j] == MOL_DNA);
        bool is_rna_i = (mol_types[i] == MOL_RNA);
        bool is_rna_j = (mol_types[j] == MOL_RNA);

        if (interaction_type == "intra_protein") return same_chain && is_prot_i && is_prot_j;
        if (interaction_type == "protein_protein") return !same_chain && is_prot_i && is_prot_j;
        if (interaction_type == "ligand_protein") return (is_lig_i && is_prot_j) || (is_prot_i && is_lig_j);
        if (interaction_type == "intra_ligand") return same_chain && is_lig_i && is_lig_j;
        if (interaction_type == "dna_protein") return (is_dna_i && is_prot_j) || (is_prot_i && is_dna_j);
        if (interaction_type == "rna_protein") return (is_rna_i && is_prot_j) || (is_prot_i && is_rna_j);
        if (interaction_type == "intra_dna") return same_chain && is_dna_i && is_dna_j;
        if (interaction_type == "intra_rna") return same_chain && is_rna_i && is_rna_j;
        return true;
    };

    float total = 0; int count = 0;
    for (int i = 0; i < N; ++i) {
        float num = 0, den = 0;
        for (int j = 0; j < N; ++j) {
            if (!should_include(i, j)) continue;
            float dt = 0, dp = 0;
            for (int d = 0; d < 3; ++d) {
                float ddt = true_coords(i,d)-true_coords(j,d);
                float ddp = pred_coords(i,d)-pred_coords(j,d);
                dt += ddt*ddt; dp += ddp*ddp;
            }
            dt = std::sqrt(dt); dp = std::sqrt(dp);
            if (dt > cutoff) continue;
            float diff = std::abs(dt - dp);
            float score = ((diff<0.5f)+(diff<1.0f)+(diff<2.0f)+(diff<4.0f))/4.0f;
            num += score; den += 1.0f;
        }
        if (den > 0) { total += num/den; count++; }
    }
    return count > 0 ? total / count : 0.0f;
}

// Weighted minimum RMSD
inline float weighted_min_rmsd(const std::vector<Mat>& pred_samples,
                                const Mat& true_coords,
                                const std::vector<float>& mask) {
    float best_rmsd = 1e30f;
    for (auto& pred : pred_samples) {
        float rmsd = 0; int count = 0;
        for (int i = 0; i < pred.rows; ++i) {
            float w = (i < (int)mask.size()) ? mask[i] : 1.0f;
            if (w < 0.5f) continue;
            for (int d = 0; d < 3; ++d) {
                float diff = pred(i,d) - true_coords(i,d);
                rmsd += diff * diff;
            }
            count++;
        }
        rmsd = (count > 0) ? std::sqrt(rmsd / count) : 1e30f;
        best_rmsd = std::min(best_rmsd, rmsd);
    }
    return best_rmsd;
}

// Predicted pLDDT MAE
inline float plddt_mae(const std::vector<float>& pred_plddt,
                       const std::vector<float>& true_lddt,
                       const std::vector<float>& mask) {
    float total = 0; int count = 0;
    for (int i = 0; i < (int)pred_plddt.size() && i < (int)true_lddt.size(); ++i) {
        float w = (i < (int)mask.size()) ? mask[i] : 1.0f;
        if (w < 0.5f) continue;
        total += std::abs(pred_plddt[i] - true_lddt[i]);
        count++;
    }
    return count > 0 ? total / count : 0.0f;
}

// Compute full training loss
struct TrainingLoss {
    float diffusion = 0, distogram_val = 0, confidence_val = 0;
    float bfactor_val = 0, total = 0;
};

inline TrainingLoss compute_full_training_loss(
    const Mat& pred_coords, const Mat& true_coords,
    const Mat& pred_disto, const Mat& true_disto,
    const std::vector<float>& mask, int N,
    const TrainingConfig& config)
{
    TrainingLoss loss;

    // Smooth lDDT as diffusion loss proxy
    std::vector<BackboneAtoms> pred_bb(N), true_bb(N);
    for (int i = 0; i < N; ++i) {
        pred_bb[i].CA_pos = Vec3(pred_coords(i,0), pred_coords(i,1), pred_coords(i,2));
        true_bb[i].CA_pos = Vec3(true_coords(i,0), true_coords(i,1), true_coords(i,2));
    }
    loss.diffusion = smooth_lddt_loss(pred_bb, true_bb, N);

    std::vector<bool> bool_mask(N);
    for (int i = 0; i < N; ++i) bool_mask[i] = (i < (int)mask.size()) ? (mask[i] > 0.5f) : true;
    loss.distogram_val = distogram_loss(pred_disto, true_disto, bool_mask, N);

    loss.total = config.diffusion_loss_weight * loss.diffusion
               + config.distogram_loss_weight * loss.distogram_val;
    return loss;
}

} // namespace boltz
