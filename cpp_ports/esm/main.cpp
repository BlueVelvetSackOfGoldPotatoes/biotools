// main.cpp - Demo of ESM (Evolutionary Scale Modeling) C++ port
//
// Creates a small ESM model with random weights, runs inference on
// a short protein sequence, and prints per-residue embeddings,
// top MLM predictions, and attention statistics.

#include "esm.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// Helper: find top-k indices in a row of logits
static std::vector<int> topk_indices(const float* row, int n, int k) {
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [row](int a, int b) { return row[a] > row[b]; });
    indices.resize(k);
    return indices;
}

int main() {
    std::cout << "=== ESM (Evolutionary Scale Modeling) - C++ Port Demo ===\n\n";

    // ---------------------------------------------------------------
    // 1. Alphabet demo
    // ---------------------------------------------------------------
    esm::Alphabet alphabet;
    std::cout << "Alphabet\n";
    std::cout << "  Vocab size: " << alphabet.vocab_size() << "\n";
    std::cout << "  <cls> idx:  " << alphabet.cls_idx << "\n";
    std::cout << "  <pad> idx:  " << alphabet.pad_idx << "\n";
    std::cout << "  <eos> idx:  " << alphabet.eos_idx << "\n";
    std::cout << "  <unk> idx:  " << alphabet.unk_idx << "\n";
    std::cout << "  <mask> idx: " << alphabet.mask_idx << "\n";
    std::cout << "  Tokens:     ";
    for (int i = 0; i < alphabet.vocab_size(); ++i) {
        std::cout << alphabet.get_tok(i);
        if (i + 1 < alphabet.vocab_size()) std::cout << " ";
    }
    std::cout << "\n\n";

    // ---------------------------------------------------------------
    // 2. Tokenize a short protein sequence
    // ---------------------------------------------------------------
    // Human insulin B-chain fragment
    std::string sequence = "FVNQHLCGSHLVEAL";
    auto tokens = alphabet.encode(sequence);

    std::cout << "Input sequence: " << sequence << "\n";
    std::cout << "Token indices:  [";
    for (int i = 0; i < (int)tokens.size(); ++i) {
        std::cout << tokens[i];
        if (i + 1 < (int)tokens.size()) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "Decoded back:   " << alphabet.decode(tokens) << "\n\n";

    // ---------------------------------------------------------------
    // 3. Build a small ESM model
    // ---------------------------------------------------------------
    esm::ESMConfig config;
    config.num_layers      = 6;
    config.embed_dim       = 320;
    config.ffn_embed_dim   = 1280;
    config.attention_heads = 20;
    config.max_positions   = 1024;
    config.add_bias_kv     = false;  // ESM-2 style

    std::cout << "Building ESM model...\n";
    esm::ESMModel model(config, /*seed=*/42);
    model.print_summary();
    std::cout << "\n";

    // ---------------------------------------------------------------
    // 4. Run forward pass
    // ---------------------------------------------------------------
    // Request embeddings from layer 0 (after embed) and final layer
    std::vector<int> repr_layers = {0, config.num_layers};
    bool need_head_weights = true;

    std::cout << "Running forward pass on \"" << sequence << "\" ("
              << tokens.size() << " tokens incl. special)...\n";

    auto output = model.forward(tokens, repr_layers, need_head_weights);

    std::cout << "  Logits shape: (" << output.logits.rows << ", "
              << output.logits.cols << ")\n";

    // ---------------------------------------------------------------
    // 5. Print per-residue embeddings from the final layer
    // ---------------------------------------------------------------
    std::cout << "\nFinal-layer embeddings (first 8 dims per residue):\n";
    auto& final_repr = output.representations[config.num_layers];
    int ed = config.embed_dim;
    int show_dims = std::min(8, ed);

    // Skip <cls> (pos 0) and <eos> (last pos) to show residues only
    int start_pos = alphabet.prepend_bos ? 1 : 0;
    int end_pos = (int)tokens.size() - (alphabet.append_eos ? 1 : 0);

    for (int i = start_pos; i < end_pos; ++i) {
        char aa = sequence[i - start_pos];
        std::cout << "  " << aa << " (pos " << std::setw(2) << i << "): [";
        for (int d = 0; d < show_dims; ++d) {
            std::cout << std::fixed << std::setprecision(4) << std::setw(8)
                      << final_repr(i, d);
            if (d + 1 < show_dims) std::cout << ",";
        }
        std::cout << " ...]\n";
    }

    // ---------------------------------------------------------------
    // 6. Compute per-residue embedding norms
    // ---------------------------------------------------------------
    std::cout << "\nEmbedding L2 norms per residue:\n";
    for (int i = start_pos; i < end_pos; ++i) {
        float norm = 0.0f;
        for (int d = 0; d < ed; ++d) {
            float v = final_repr(i, d);
            norm += v * v;
        }
        norm = std::sqrt(norm);
        char aa = sequence[i - start_pos];
        std::cout << "  " << aa << ": " << std::fixed << std::setprecision(4) << norm << "\n";
    }

    // ---------------------------------------------------------------
    // 7. Top MLM predictions per position
    // ---------------------------------------------------------------
    std::cout << "\nTop-3 MLM predictions per residue position:\n";
    int top_k = 3;
    for (int i = start_pos; i < end_pos; ++i) {
        char aa = sequence[i - start_pos];
        const float* logit_row = output.logits.row_ptr(i);
        auto top_indices = topk_indices(logit_row, output.logits.cols, top_k);

        std::cout << "  " << aa << " -> ";
        for (int j = 0; j < top_k; ++j) {
            int idx = top_indices[j];
            std::cout << alphabet.get_tok(idx) << "("
                      << std::fixed << std::setprecision(2)
                      << logit_row[idx] << ")";
            if (j + 1 < top_k) std::cout << ", ";
        }
        std::cout << "\n";
    }

    // ---------------------------------------------------------------
    // 8. Attention statistics
    // ---------------------------------------------------------------
    std::cout << "\nAttention statistics (mean attention weight per layer):\n";
    for (auto& [layer_idx, attn] : output.attentions) {
        double sum = 0.0;
        int count = 0;
        for (int h = 0; h < attn.d0; ++h) {
            for (int qi = 0; qi < attn.d1; ++qi) {
                for (int ki = 0; ki < attn.d2; ++ki) {
                    sum += attn(h, qi, ki);
                    count++;
                }
            }
        }
        double mean = sum / std::max(count, 1);
        std::cout << "  Layer " << layer_idx << ": mean_attn=" << std::fixed
                  << std::setprecision(6) << mean
                  << "  (expected ~" << std::setprecision(4)
                  << 1.0 / attn.d2 << " for uniform)\n";
    }

    // ---------------------------------------------------------------
    // 9. Test with masked sequence
    // ---------------------------------------------------------------
    std::cout << "\n--- Masked prediction demo ---\n";
    std::string masked_seq = sequence;
    int mask_pos = 7; // mask the 8th residue (0-indexed)
    char original_aa = masked_seq[mask_pos];
    masked_seq[mask_pos] = 'X'; // We'll manually replace with <mask> token

    auto mask_tokens = alphabet.encode(masked_seq);
    // Replace the X token with <mask>
    int token_pos = mask_pos + (alphabet.prepend_bos ? 1 : 0);
    mask_tokens[token_pos] = alphabet.mask_idx;

    std::cout << "Original:  " << sequence << "\n";
    std::cout << "Masked at position " << mask_pos << " (was '" << original_aa << "')\n";

    auto mask_output = model.forward(mask_tokens);

    const float* mask_logits = mask_output.logits.row_ptr(token_pos);
    auto top5 = topk_indices(mask_logits, mask_output.logits.cols, 5);

    std::cout << "Top-5 predictions for masked position:\n";
    for (int j = 0; j < 5; ++j) {
        int idx = top5[j];
        std::cout << "  " << (j + 1) << ". " << alphabet.get_tok(idx)
                  << " (logit=" << std::fixed << std::setprecision(3)
                  << mask_logits[idx] << ")\n";
    }

    // ---------------------------------------------------------------
    // 10. Batch of sequences (processed individually)
    // ---------------------------------------------------------------
    std::cout << "\n--- Processing multiple sequences ---\n";
    std::vector<std::string> sequences = {
        "MKTLLILAVL",      // Short peptide
        "FVNQHLCGSHLVEAL", // Insulin B-chain fragment
        "GIVEQCCTSICSLYQ", // Another peptide
    };

    for (auto& seq : sequences) {
        auto seq_tokens = alphabet.encode(seq);
        auto seq_output = model.forward(seq_tokens, {config.num_layers});

        auto& repr = seq_output.representations[config.num_layers];
        // Compute mean embedding (excluding special tokens)
        std::vector<float> mean_emb(ed, 0.0f);
        int n_residues = 0;
        int s = alphabet.prepend_bos ? 1 : 0;
        int e = (int)seq_tokens.size() - (alphabet.append_eos ? 1 : 0);
        for (int i = s; i < e; ++i) {
            for (int d = 0; d < ed; ++d) {
                mean_emb[d] += repr(i, d);
            }
            n_residues++;
        }
        for (int d = 0; d < ed; ++d) mean_emb[d] /= n_residues;

        float norm = 0.0f;
        for (int d = 0; d < ed; ++d) norm += mean_emb[d] * mean_emb[d];
        norm = std::sqrt(norm);

        std::cout << "  " << std::setw(20) << std::left << seq
                  << " len=" << std::setw(3) << seq.size()
                  << " mean_emb_norm=" << std::fixed << std::setprecision(4)
                  << norm << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
