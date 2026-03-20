// test_extended.cpp - Compile test for esm_extended.h
#include "esm_extended.h"
#include <iostream>

int main() {
    std::cout << "=== ESM Extended C++ Port - Compilation Test ===\n\n";

    // Test Quaternion and Rigid
    esm::Quaternion q(1, 0, 0, 0);
    esm::Vec3 v(1, 2, 3);
    esm::Vec3 rotated = q.rotate(v);
    std::cout << "Quaternion identity rotation: ("
              << rotated.x << ", " << rotated.y << ", " << rotated.z << ")\n";

    esm::Rigid r = esm::Rigid::identity();
    esm::Vec3 applied = r.apply(v);
    std::cout << "Rigid identity: (" << applied.x << ", " << applied.y << ", " << applied.z << ")\n";

    // Test TriangleMultiplication
    std::mt19937 rng(42);
    esm::TriangleMultiplicationOutgoing tri_out(16, 16, rng);
    esm::TriangleMultiplicationIncoming tri_in(16, 16, rng);
    std::cout << "TriangleMultiplication created\n";

    // Test TriangleAttention
    esm::TriangleAttentionStartingNode tri_att_s(16, 8, 2, 1e9f, rng);
    esm::TriangleAttentionEndingNode tri_att_e(16, 8, 2, 1e9f, rng);
    std::cout << "TriangleAttention created\n";

    // Test FullTriangularSelfAttentionBlock
    esm::FullTriangularSelfAttentionBlock full_block(64, 16, 32, 8, 0.0f, rng);
    std::cout << "FullTriangularSelfAttentionBlock created\n";

    // Test IPA
    esm::InvariantPointAttention ipa(32, 16, 8, 8, 4, 2, 2, rng);
    std::cout << "InvariantPointAttention created\n";

    // Test StructureModule
    esm::StructureModuleConfig sm_cfg;
    sm_cfg.c_s = 32; sm_cfg.c_z = 16; sm_cfg.c_ipa = 8; sm_cfg.c_resnet = 16;
    sm_cfg.no_heads_ipa = 2; sm_cfg.no_qk_points = 2; sm_cfg.no_v_points = 2;
    sm_cfg.no_blocks = 1; sm_cfg.no_resnet_blocks = 1;
    esm::StructureModule sm(sm_cfg, rng);
    std::cout << "StructureModule created\n";

    // Test running SM
    int L = 4;
    std::vector<float> s_in(L * sm_cfg.c_s, 0.1f), z_in(L * L * sm_cfg.c_z, 0.01f);
    auto sm_out = sm.forward(s_in.data(), z_in.data(), L);
    std::cout << "StructureModule forward: " << sm_out.positions.size() << " blocks\n";
    if (!sm_out.positions.empty()) {
        auto& pos = sm_out.positions.back();
        std::cout << "  First residue N:  (" << pos[0].x << ", " << pos[0].y << ", " << pos[0].z << ")\n";
        std::cout << "  First residue CA: (" << pos[1].x << ", " << pos[1].y << ", " << pos[1].z << ")\n";
        std::cout << "  First residue C:  (" << pos[2].x << ", " << pos[2].y << ", " << pos[2].z << ")\n";
    }

    // Test DihedralFeatures
    esm::DihedralFeaturesModule dih(32, rng);
    std::vector<std::array<esm::Vec3, 3>> coords = {
        {esm::Vec3(-0.5f, 1.3f, 0.0f), esm::Vec3(0,0,0), esm::Vec3(1.5f, 0, 0)},
        {esm::Vec3(1.0f, 2.3f, 0.0f), esm::Vec3(1.5f,1.0f,0), esm::Vec3(3.0f, 1, 0)},
        {esm::Vec3(2.5f, 2.3f, 0.0f), esm::Vec3(3.0f,1.0f,0), esm::Vec3(4.5f, 1, 0)},
    };
    std::vector<float> dih_out(3 * 32);
    dih.forward(coords, dih_out.data());
    std::cout << "DihedralFeatures: first 4 dims = ["
              << dih_out[0] << ", " << dih_out[1] << ", " << dih_out[2] << ", " << dih_out[3] << "]\n";

    // Test PDB I/O
    std::vector<esm::Vec3> positions;
    for (auto& c : coords) { positions.push_back(c[0]); positions.push_back(c[1]); positions.push_back(c[2]); }
    std::string pdb = esm::pdb_io::write_pdb(positions, "GGG");
    std::cout << "PDB output (" << pdb.size() << " chars):\n";
    std::cout << pdb.substr(0, std::min(size_t(200), pdb.size())) << "\n";

    // Test multichain
    esm::multichain::ChainCoords chain_coords;
    chain_coords["A"] = coords;
    chain_coords["B"] = {
        {esm::Vec3(10,0,0), esm::Vec3(11,0,0), esm::Vec3(12,0,0)},
    };
    auto concat = esm::multichain::concatenate_coords(chain_coords, "A", 5);
    std::cout << "Concatenated chains: " << concat.size() << " residues\n";

    // Test CoordBatchConverter
    esm::Alphabet alpha(esm::AlphabetArch::InvariantGVP);
    esm::CoordBatchConverter conv(&alpha);
    std::vector<std::tuple<std::vector<std::array<esm::Vec3,3>>, std::vector<float>, std::string>> batch;
    batch.push_back({coords, {}, "GGG"});
    auto cr = conv.convert(batch);
    std::cout << "CoordBatchConverter: " << cr.tokens[0].size() << " tokens\n";

    // Test tokenization
    auto toks = esm::tokenization::full_tokenize(alpha, "ACGT<mask>XY");
    std::cout << "Full tokenize: [";
    for (size_t i = 0; i < toks.size(); ++i) {
        std::cout << toks[i];
        if (i + 1 < toks.size()) std::cout << ", ";
    }
    std::cout << "]\n";

    // Test extended registry
    auto all_models = esm::pretrained::list_all_models();
    std::cout << "Total models in registry: " << all_models.size() << "\n";
    for (auto& n : all_models) std::cout << "  " << n << "\n";

    // Test feature extraction
    esm::ESMModel model = esm::create_esm2(2, 64, 4, 42);
    auto output = model.infer("ACDEF", {2});
    auto emb = esm::features::extract_residue_embeddings(output, 2, model.alphabet);
    auto mean = esm::features::mean_embedding(emb);
    std::cout << "Mean embedding norm: " << esm::math::vec_norm(mean.data(), static_cast<int>(mean.size())) << "\n";

    // Test residue constants
    std::cout << "Residue types: " << esm::residue_constants::restypes().size() << "\n";
    std::cout << "N-CA bond length: " << esm::residue_constants::bond_length_n_ca() << "\n";

    // Test GVP conv layer
    esm::GVPConvLayerExt gvp_cl(16, 4, 8, 1, true, 3, rng);
    std::cout << "GVPConvLayerExt created\n";

    std::cout << "\nAll tests passed!\n";
    return 0;
}
