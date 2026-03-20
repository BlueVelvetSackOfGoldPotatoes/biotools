// main.cpp -- Boltz C++ port comprehensive demo
//
// Exercises the full pipeline: parsing, tokenization, featurization, recycling,
// diffusion, confidence prediction, and output in both PDB and mmCIF formats.

#include "boltz.h"

#include <chrono>
#include <iostream>

int main(int argc, char* argv[]) {
    using namespace boltz;
    using namespace std::chrono;

    // Default peptide sequence (8 residues)
    std::string sequence = "ACDEFGHI";
    std::string input_file;
    bool from_file = false;

    if (argc > 1) {
        std::string arg = argv[1];
        // Check if it's a file
        if (arg.size() > 4 && (arg.substr(arg.size()-6) == ".fasta" ||
                               arg.substr(arg.size()-4) == ".pdb" ||
                               arg.substr(arg.size()-3) == ".fa")) {
            input_file = arg;
            from_file = true;
        } else {
            sequence = arg;
        }
    }

    // If loading from FASTA file
    if (from_file) {
        auto records = parse_fasta(input_file);
        if (!records.empty()) {
            sequence = records[0].sequence;
            std::cout << "Loaded sequence from " << input_file << ": "
                      << records[0].chain_id << "|" << records[0].entity_type << "\n";
        } else {
            std::cerr << "Failed to parse input file: " << input_file << "\n";
            return 1;
        }
    }

    int N = (int)sequence.size();
    if (N < 2 || N > 20) {
        std::cerr << "Sequence length must be between 2 and 20 for this demo.\n";
        return 1;
    }

    // Validate sequence
    for (char c : sequence) {
        if (aa_to_index(c) >= NUM_AA) {
            std::cerr << "Warning: '" << c << "' is not a standard amino acid, "
                      << "will be treated as UNK.\n";
        }
    }

    std::cout << "=== Boltz C++ Port - Full Structure Prediction Pipeline ===\n";
    std::cout << "Sequence: " << sequence << " (" << N << " residues)\n";
    std::cout << "\nArchitecture:\n";
    std::cout << "  Single repr dim (TOKEN_S): " << TOKEN_S << "\n";
    std::cout << "  Pair repr dim (TOKEN_Z):   " << TOKEN_Z << "\n";
    std::cout << "  MSA embedding dim:         " << MSA_S << "\n";
    std::cout << "  Attention heads:           " << NUM_HEADS << "\n";
    std::cout << "  MSA blocks:                " << NUM_MSA_BLOCKS << "\n";
    std::cout << "  Pairformer blocks:         " << NUM_PAIRFORMER_BLOCKS << "\n";
    std::cout << "  Structure module blocks:   " << NUM_DIFFUSION_BLOCKS << "\n";
    std::cout << "  Diffusion transformer:     " << TOKEN_TRANSFORMER_DEPTH << " layers\n";
    std::cout << "  IPA query/value points:    " << NUM_IPA_QUERY_POINTS
              << "/" << NUM_IPA_VALUE_POINTS << "\n";
    std::cout << "  Distogram bins:            " << NUM_DIST_BINS << "\n";
    std::cout << "  Confidence (pLDDT bins):   " << NUM_PLDDT_BINS << "\n";
    std::cout << "  Confidence (PAE bins):     " << NUM_PAE_BINS << "\n";
    std::cout << "  Recycling iterations:      " << NUM_RECYCLES << "\n\n";

    // Set deterministic seed
    set_seed(42);

    // Build model
    std::cout << "Initializing full model (random weights)... " << std::flush;
    auto t0 = high_resolution_clock::now();
    BoltzModel model;
    model.init(N);
    auto t1 = high_resolution_clock::now();
    std::cout << "done ("
              << duration_cast<milliseconds>(t1 - t0).count() << " ms)\n";

    // Forward pass with recycling
    std::cout << "Running forward pass with " << NUM_RECYCLES << " recycling iterations... " << std::flush;
    auto t2 = high_resolution_clock::now();
    auto output = model.forward(sequence, NUM_RECYCLES);
    auto t3 = high_resolution_clock::now();
    std::cout << "done ("
              << duration_cast<milliseconds>(t3 - t2).count() << " ms)\n\n";

    // Print predicted atom coordinates
    std::cout << "Predicted backbone coordinates (N, CA, C, O):\n";
    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < N; ++i) {
        const auto& a = output.atoms[i];
        std::cout << "  Res " << (i+1) << " (" << sequence[i] << "):\n";
        std::cout << "    N:  (" << a.N_pos.x  << ", " << a.N_pos.y  << ", " << a.N_pos.z  << ")\n";
        std::cout << "    CA: (" << a.CA_pos.x << ", " << a.CA_pos.y << ", " << a.CA_pos.z << ")\n";
        std::cout << "    C:  (" << a.C_pos.x  << ", " << a.C_pos.y  << ", " << a.C_pos.z  << ")\n";
        std::cout << "    O:  (" << a.O_pos.x  << ", " << a.O_pos.y  << ", " << a.O_pos.z  << ")\n";
    }

    // Confidence predictions
    std::cout << "\n--- Confidence Predictions ---\n";
    std::cout << "Per-residue pLDDT:\n  ";
    for (int i = 0; i < N; ++i) {
        std::cout << std::setprecision(3) << output.conf.plddt[i];
        if (i < N-1) std::cout << ", ";
    }
    std::cout << "\n";
    std::cout << "Complex pLDDT: " << std::setprecision(4) << output.conf.complex_plddt << "\n";
    std::cout << "pTM:           " << std::setprecision(4) << output.conf.ptm << "\n";
    std::cout << "ipTM:          " << std::setprecision(4) << output.conf.iptm << "\n";

    // Construct a trivial "reference" structure: extended chain along x-axis
    std::vector<Rigid> ref_frames(N);
    for (int i = 0; i < N; ++i) {
        ref_frames[i] = Rigid(Rot3::identity(), Vec3(i * 3.8f, 0, 0));
    }
    auto ref_atoms = place_backbone_atoms(ref_frames, N);

    // Compute FAPE loss
    float fape = fape_loss(output.frames, output.atoms,
                           ref_frames, ref_atoms, N);
    std::cout << "\n--- Loss Metrics (vs extended chain reference) ---\n";
    std::cout << "FAPE loss:       " << std::setprecision(4) << fape << "\n";

    // Compute smooth lDDT loss
    float lddt = smooth_lddt_loss(output.atoms, ref_atoms, N);
    std::cout << "Smooth lDDT loss: " << std::setprecision(4) << lddt << "\n";

    // Distogram loss
    std::vector<bool> mask(N, true);
    Mat target_disto(N*N, NUM_DIST_BINS, 0.0f);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float d = (ref_atoms[i].CA_pos - ref_atoms[j].CA_pos).norm();
            int bin = std::clamp((int)((d - DIST_MIN)/(DIST_MAX-DIST_MIN)*NUM_DIST_BINS),
                                  0, NUM_DIST_BINS-1);
            target_disto(i*N+j, bin) = 1.0f;
        }
    float disto_l = distogram_loss(output.disto, target_disto, mask, N);
    std::cout << "Distogram loss:   " << std::setprecision(4) << disto_l << "\n";

    // Compute pairwise CA distances for the predicted structure
    std::cout << "\nPredicted CA-CA distances:\n";
    for (int i = 0; i < N; ++i) {
        std::cout << "  ";
        for (int j = 0; j < N; ++j) {
            float d = (output.atoms[i].CA_pos - output.atoms[j].CA_pos).norm();
            std::cout << std::setw(7) << std::setprecision(2) << d;
        }
        std::cout << "\n";
    }

    // Distogram: show argmax bin per pair for first few pairs
    std::cout << "\nDistogram (argmax bin for sequential pairs):\n";
    for (int i = 0; i < std::min(N-1, 5); ++i) {
        int row = i * N + (i + 1);
        int best_bin = 0;
        float best_val = output.disto(row, 0);
        for (int b = 1; b < NUM_DIST_BINS; ++b) {
            if (output.disto(row, b) > best_val) {
                best_val = output.disto(row, b);
                best_bin = b;
            }
        }
        float bin_center = DIST_MIN + (best_bin + 0.5f) *
                           (DIST_MAX - DIST_MIN) / NUM_DIST_BINS;
        std::cout << "  Pair (" << (i+1) << "," << (i+2)
                  << "): bin " << best_bin
                  << " ~ " << std::setprecision(1) << bin_center << " A\n";
    }

    // Noise schedule info
    std::cout << "\n--- Noise Schedule ---\n";
    NoiseSchedule ns;
    std::cout << "sigma_min: " << ns.sigma_min << ", sigma_max: " << ns.sigma_max << "\n";
    std::cout << "sigma(0.0): " << std::setprecision(3) << ns.sigma(0.0f) << "\n";
    std::cout << "sigma(0.5): " << std::setprecision(3) << ns.sigma(0.5f) << "\n";
    std::cout << "sigma(1.0): " << std::setprecision(4) << ns.sigma(1.0f) << "\n";

    // LR Schedule
    LRScheduler lr;
    std::cout << "\n--- LR Schedule ---\n";
    std::cout << "Step     0: " << lr.get_lr(0) << "\n";
    std::cout << "Step   500: " << lr.get_lr(500) << "\n";
    std::cout << "Step  1000: " << lr.get_lr(1000) << "\n";
    std::cout << "Step 25000: " << lr.get_lr(25000) << "\n";
    std::cout << "Step 75000: " << lr.get_lr(75000) << "\n";

    // Write PDB with pLDDT in B-factor column
    std::string pdb_file = "boltz_output.pdb";
    std::string pdb = atoms_to_pdb(output.atoms, sequence, "BOLTZ", &output.conf.plddt);
    if (write_pdb(pdb_file, pdb)) {
        std::cout << "\nPDB written to: " << pdb_file << "\n";
    }

    // Write mmCIF
    std::string cif_file = "boltz_output.cif";
    std::string cif = atoms_to_mmcif(output.atoms, sequence, "BOLTZ", &output.conf.plddt);
    if (write_mmcif(cif_file, cif)) {
        std::cout << "mmCIF written to: " << cif_file << "\n";
    }

    std::cout << "\n=== Done ===\n";
    return 0;
}
