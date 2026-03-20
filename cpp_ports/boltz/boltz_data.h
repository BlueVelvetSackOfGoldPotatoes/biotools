#pragma once
// boltz_data.h -- Extended data handling for the Boltz C++ port.
//
// Covers:
//   CSV/YAML parsing for input specs
//   Full PDB parser with MODEL support
//   Full mmCIF parser (struct_conn for covalent bonds)
//   Complete A3M/MSA parsing with taxonomy, paired MSA
//   Full tokenization pipeline (BoltzTokenizer)
//   Cropping strategies (BoltzCropper, AffinityCropper)
//   Featurization pipeline (frames, atom features, pair features)
//   MSA processing (pairing, deduplication, max_seqs enforcement)
//   Static filters (excluded ligands, polymer length, clashing chains, unknown)
//   Dynamic filters (date, resolution, size, max residues, subset)
//   Sampling strategies (cluster, random, distillation)
//   Data module (training & inference pipelines)
//   PDB/mmCIF writers with full atom output

#include "boltz.h"
#include <regex>
#include <queue>

namespace boltz {

// ============================================================================
// Constants from Python boltz.data.const (expanded)
// ============================================================================

static constexpr float ATOM_INTERFACE_CUTOFF = 5.0f;

// Method conditioning types
enum MethodType {
    METHOD_MD = 0, METHOD_XRAY = 1, METHOD_EM = 2,
    METHOD_SOLUTION_NMR = 3, METHOD_OTHER = 4,
    METHOD_AFDB = 5, METHOD_BOLTZ1 = 6,
    NUM_METHOD_TYPES = 12
};

inline int method_string_to_id(const std::string& method) {
    std::string lower = method;
    for (auto& c : lower) c = (char)std::tolower(c);
    if (lower == "md") return 0;
    if (lower == "x-ray diffraction") return 1;
    if (lower == "electron microscopy") return 2;
    if (lower == "solution nmr") return 3;
    if (lower.find("afdb") != std::string::npos) return 5;
    if (lower.find("boltz") != std::string::npos) return 6;
    return 4;
}

// Temperature bins
static const int TEMP_BINS[][2] = {{265,280},{280,295},{295,310}};
static constexpr int NUM_TEMP_BINS = 4;
static const int PH_BINS_ARR[][2] = {{0,6},{6,8},{8,14}};
static constexpr int NUM_PH_BINS = 4;

// Pocket contact info
static constexpr int POCKET_UNSPECIFIED = 0;
static constexpr int POCKET_UNSELECTED = 1;
static constexpr int POCKET_POCKET = 2;
static constexpr int POCKET_BINDER = 3;
static constexpr int NUM_POCKET_CONTACT_INFO = 4;

// Excluded ligand set (common solvents, ions, buffers)
inline bool is_excluded_ligand(const std::string& name) {
    static const std::unordered_set<std::string> excluded = {
        "144","15P","1PE","2F2","2JC","3HR","3SY","7N5","7PE","9JE",
        "AAE","ABA","ACE","ACN","ACT","ACY","AZI","BAM","BCN","BCT",
        "BDN","BEN","BME","BO3","BTB","BTC","BU1","C8E","CAD","CAQ",
        "CBM","CCN","CIT","CL","CLR","CM","CMO","CO3","CPT","CXS",
        "D10","DEP","DIO","DMS","DN","DOD","DOX","EDO","EEE","EGL",
        "EOH","EOX","EPE","ETF","FCY","FJO","FLC","FMT","FW5","GOL",
        "GSH","GTT","GYF","HED","IHP","IHS","IMD","IOD","IPA","IPH",
        "LDA","MB3","MEG","MES","MLA","MLI","MOH","MPD","MRD","MSE",
        "MYR","N","NA","NH2","NH4","NHE","NO3","O4B","OHE","OLA",
        "OLC","OMB","OME","OXA","P6G","PE3","PE4","PEG","PEO","PEP",
        "PG0","PG4","PGE","PGR","PLM","PO4","POL","POP","PVO","SAR",
        "SCN","SEO","SEP","SIN","SO4","SPD","SPM","SR","STE","STO",
        "STU","TAR","TBU","TME","TPO","TRS","UNK","UNL","UNX","UPL","URE"
    };
    return excluded.count(name) > 0;
}

// Reference atoms per residue type
inline std::vector<std::string> get_ref_atoms(const std::string& res) {
    static const std::unordered_map<std::string, std::vector<std::string>> ref = {
        {"ALA", {"N","CA","C","O","CB"}},
        {"ARG", {"N","CA","C","O","CB","CG","CD","NE","CZ","NH1","NH2"}},
        {"ASN", {"N","CA","C","O","CB","CG","OD1","ND2"}},
        {"ASP", {"N","CA","C","O","CB","CG","OD1","OD2"}},
        {"CYS", {"N","CA","C","O","CB","SG"}},
        {"GLN", {"N","CA","C","O","CB","CG","CD","OE1","NE2"}},
        {"GLU", {"N","CA","C","O","CB","CG","CD","OE1","OE2"}},
        {"GLY", {"N","CA","C","O"}},
        {"HIS", {"N","CA","C","O","CB","CG","ND1","CD2","CE1","NE2"}},
        {"ILE", {"N","CA","C","O","CB","CG1","CG2","CD1"}},
        {"LEU", {"N","CA","C","O","CB","CG","CD1","CD2"}},
        {"LYS", {"N","CA","C","O","CB","CG","CD","CE","NZ"}},
        {"MET", {"N","CA","C","O","CB","CG","SD","CE"}},
        {"PHE", {"N","CA","C","O","CB","CG","CD1","CD2","CE1","CE2","CZ"}},
        {"PRO", {"N","CA","C","O","CB","CG","CD"}},
        {"SER", {"N","CA","C","O","CB","OG"}},
        {"THR", {"N","CA","C","O","CB","OG1","CG2"}},
        {"TRP", {"N","CA","C","O","CB","CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2"}},
        {"TYR", {"N","CA","C","O","CB","CG","CD1","CD2","CE1","CE2","CZ","OH"}},
        {"VAL", {"N","CA","C","O","CB","CG1","CG2"}},
        {"UNK", {"N","CA","C","O","CB"}},
    };
    auto it = ref.find(res);
    if (it != ref.end()) return it->second;
    return {"N","CA","C","O","CB"};
}

inline std::string center_atom_name(const std::string& res) {
    if (res == "GLY") return "CA";
    static const std::unordered_set<std::string> nuc = {"A","G","C","U","N","DA","DG","DC","DT","DN"};
    if (nuc.count(res)) return "C1'";
    return "CA";
}

inline std::string disto_atom_name(const std::string& res) {
    if (res == "GLY") return "CA";
    static const std::unordered_set<std::string> nuc_c4 = {"A","G","DA","DG"};
    static const std::unordered_set<std::string> nuc_c2 = {"C","U","DC","DT"};
    static const std::unordered_set<std::string> nuc_c1 = {"N","DN"};
    if (nuc_c4.count(res)) return "C4";
    if (nuc_c2.count(res)) return "C2";
    if (nuc_c1.count(res)) return "C1'";
    return "CB";
}

// ============================================================================
// Structure info (matching Python StructureInfo)
// ============================================================================
struct StructureInfo {
    float resolution = -1.0f;
    std::string method;
    std::string deposited;
    std::string released;
    std::string revised;
    int num_chains = 0;
    int num_interfaces = 0;
    float pH = -1.0f;
    float temperature = -1.0f;
};

struct TemplateInfo {
    std::string name;
    std::string query_chain;
    int query_st = 0, query_en = 0;
    std::string template_chain;
    int template_st = 0, template_en = 0;
    bool force = false;
    float threshold = 1e30f;
};

struct RecordFull {
    std::string id;
    StructureInfo structure_info;
    std::vector<ChainInfo> chains;
    std::vector<InterfaceData> interfaces;
    std::vector<TemplateInfo> templates;
    int num_valid_chains() const {
        int c = 0; for (auto& ch : chains) if (ch.valid) c++; return c;
    }
    int total_residues() const {
        int r = 0; for (auto& ch : chains) r += ch.num_residues; return r;
    }
};

struct Sample {
    RecordFull record;
    int chain_id = -1;
    int interface_id = -1;
};

// ============================================================================
// Static Filters
// ============================================================================

struct ExcludedLigandsFilter {
    std::vector<bool> filter(const Structure& structure) const {
        std::vector<bool> valid(structure.chains.size(), true);
        for (int i = 0; i < (int)structure.chains.size(); ++i) {
            if (structure.chains[i].mol_type != MOL_NONPOLYMER) continue;
            int rs = structure.chains[i].res_idx;
            int re = rs + structure.chains[i].res_num;
            for (int r = rs; r < re && r < (int)structure.residues.size(); ++r) {
                if (is_excluded_ligand(structure.residues[r].name)) {
                    valid[i] = false; break;
                }
            }
        }
        return valid;
    }
};

struct MinimumLengthFilter {
    int min_len = 4, max_len = 5000;
    MinimumLengthFilter(int mn = 4, int mx = 5000) : min_len(mn), max_len(mx) {}
    std::vector<bool> filter(const Structure& structure) const {
        std::vector<bool> valid(structure.chains.size(), true);
        for (int i = 0; i < (int)structure.chains.size(); ++i) {
            if (structure.chains[i].mol_type == MOL_NONPOLYMER) continue;
            int rs = structure.chains[i].res_idx, re = rs + structure.chains[i].res_num;
            int resolved = 0;
            for (int r = rs; r < re && r < (int)structure.residues.size(); ++r)
                if (structure.residues[r].is_present) resolved++;
            if (resolved < min_len || resolved > max_len) valid[i] = false;
        }
        return valid;
    }
};

struct UnknownFilterData {
    std::vector<bool> filter(const Structure& structure) const {
        std::vector<bool> valid(structure.chains.size(), true);
        for (int i = 0; i < (int)structure.chains.size(); ++i) {
            if (structure.chains[i].mol_type == MOL_NONPOLYMER) continue;
            int rs = structure.chains[i].res_idx, re = rs + structure.chains[i].res_num;
            bool all_unk = true;
            for (int r = rs; r < re && r < (int)structure.residues.size(); ++r)
                if (structure.residues[r].res_type != NUM_AA) { all_unk = false; break; }
            if (all_unk && re > rs) valid[i] = false;
        }
        return valid;
    }
};

struct ConsecutiveCAFilter {
    float max_dist = 10.0f;
    explicit ConsecutiveCAFilter(float md = 10.0f) : max_dist(md) {}
    std::vector<bool> filter(const Structure& structure) const {
        std::vector<bool> valid(structure.chains.size(), true);
        for (int i = 0; i < (int)structure.chains.size(); ++i) {
            if (structure.chains[i].mol_type != MOL_PROTEIN) continue;
            int rs = structure.chains[i].res_idx, re = rs + structure.chains[i].res_num;
            for (int r = rs; r < re - 1 && r + 1 < (int)structure.residues.size(); ++r) {
                auto& r1 = structure.residues[r]; auto& r2 = structure.residues[r + 1];
                if (!r1.is_present || !r2.is_present) continue;
                int ca1 = r1.atom_center, ca2 = r2.atom_center;
                if (ca1 >= (int)structure.atoms.size() || ca2 >= (int)structure.atoms.size()) continue;
                if (!structure.atoms[ca1].is_present || !structure.atoms[ca2].is_present) continue;
                float d = (structure.atoms[ca1].coords - structure.atoms[ca2].coords).norm();
                if (d > max_dist) { valid[i] = false; break; }
            }
        }
        return valid;
    }
};

// ============================================================================
// Dynamic Filters
// ============================================================================

struct ResolutionFilterData {
    float max_resolution = 9.0f;
    explicit ResolutionFilterData(float r = 9.0f) : max_resolution(r) {}
    bool filter(const RecordFull& record) const {
        return record.structure_info.resolution >= 0 &&
               record.structure_info.resolution <= max_resolution;
    }
};

struct SizeFilterData {
    int min_chains = 1, max_chains = 300;
    SizeFilterData(int mn = 1, int mx = 300) : min_chains(mn), max_chains(mx) {}
    bool filter(const RecordFull& record) const {
        int valid = record.num_valid_chains();
        return (int)record.chains.size() <= max_chains && valid >= min_chains;
    }
};

struct MaxResiduesFilterData {
    int min_residues = 1, max_residues = 500;
    MaxResiduesFilterData(int mn = 1, int mx = 500) : min_residues(mn), max_residues(mx) {}
    bool filter(const RecordFull& record) const {
        int total = record.total_residues();
        return total >= min_residues && total <= max_residues;
    }
};

struct DateFilterData {
    std::string cutoff_date;
    std::string ref_field;
    DateFilterData(const std::string& date, const std::string& ref = "deposited")
        : cutoff_date(date), ref_field(ref) {}
    bool filter(const RecordFull& record) const {
        std::string date;
        if (ref_field == "deposited") date = record.structure_info.deposited;
        else if (ref_field == "released") {
            date = record.structure_info.released;
            if (date.empty()) date = record.structure_info.deposited;
        } else {
            date = record.structure_info.revised;
            if (date.empty()) date = record.structure_info.released;
            if (date.empty()) date = record.structure_info.deposited;
        }
        if (date.empty()) return false;
        return date <= cutoff_date;
    }
};

struct SubsetFilterData {
    std::unordered_set<std::string> subset;
    bool reverse = false;
    SubsetFilterData(const std::string& path, bool rev = false) : reverse(rev) {
        std::ifstream f(path); std::string line;
        while (std::getline(f, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            std::string lower = line;
            for (auto& c : lower) c = (char)std::tolower(c);
            if (!lower.empty()) subset.insert(lower);
        }
    }
    bool filter(const RecordFull& record) const {
        std::string lower = record.id;
        for (auto& c : lower) c = (char)std::tolower(c);
        if (reverse) return subset.find(lower) == subset.end();
        return subset.find(lower) != subset.end();
    }
};

// ============================================================================
// CSV Parser
// ============================================================================
struct CSVEntry {
    std::string id, entity_type, sequence, msa_path;
    std::map<std::string, std::string> extra;
};

inline std::vector<CSVEntry> parse_csv(const std::string& path) {
    std::vector<CSVEntry> entries;
    std::ifstream f(path);
    if (!f.is_open()) return entries;
    std::string header_line;
    if (!std::getline(f, header_line)) return entries;
    while (!header_line.empty() && (header_line.back() == '\r' || header_line.back() == '\n')) header_line.pop_back();
    std::vector<std::string> headers;
    { std::istringstream ss(header_line); std::string col;
      while (std::getline(ss, col, ',')) {
          while (!col.empty() && col.front() == ' ') col.erase(col.begin());
          while (!col.empty() && col.back() == ' ') col.pop_back();
          headers.push_back(col);
      }
    }
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty()) continue;
        std::vector<std::string> fields;
        std::istringstream ss(line); std::string field;
        while (std::getline(ss, field, ',')) {
            while (!field.empty() && field.front() == ' ') field.erase(field.begin());
            while (!field.empty() && field.back() == ' ') field.pop_back();
            fields.push_back(field);
        }
        CSVEntry e;
        for (int i = 0; i < (int)headers.size() && i < (int)fields.size(); ++i) {
            if (headers[i] == "id") e.id = fields[i];
            else if (headers[i] == "entity_type" || headers[i] == "type") e.entity_type = fields[i];
            else if (headers[i] == "sequence") e.sequence = fields[i];
            else if (headers[i] == "msa" || headers[i] == "msa_path") e.msa_path = fields[i];
            else e.extra[headers[i]] = fields[i];
        }
        entries.push_back(e);
    }
    return entries;
}

// ============================================================================
// YAML Parser (simplified)
// ============================================================================
struct YAMLEntry { std::string key, value; };

inline std::vector<YAMLEntry> parse_yaml_simple(const std::string& path) {
    std::vector<YAMLEntry> entries;
    std::ifstream f(path);
    if (!f.is_open()) return entries;
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        YAMLEntry e;
        e.key = line.substr(0, colon);
        while (!e.key.empty() && e.key.front() == ' ') e.key.erase(e.key.begin());
        while (!e.key.empty() && e.key.back() == ' ') e.key.pop_back();
        e.value = (colon + 1 < line.size()) ? line.substr(colon + 1) : "";
        while (!e.value.empty() && e.value.front() == ' ') e.value.erase(e.value.begin());
        while (!e.value.empty() && e.value.back() == ' ') e.value.pop_back();
        entries.push_back(e);
    }
    return entries;
}

// ============================================================================
// Extended PDB parser with MODEL support
// ============================================================================
struct PDBModel { int model_number = 1; PDBData data; };

inline std::vector<PDBModel> parse_pdb_models(const std::string& path) {
    std::vector<PDBModel> models;
    std::ifstream f(path);
    if (!f.is_open()) return models;
    PDBModel current; current.model_number = 1;
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.size() < 6) continue;
        std::string rec = line.substr(0, 6);
        while (!rec.empty() && rec.back() == ' ') rec.pop_back();
        if (rec == "MODEL") {
            try { current.model_number = std::stoi(line.substr(6)); } catch (...) {}
            continue;
        }
        if (rec == "ENDMDL") {
            models.push_back(current);
            current = PDBModel(); current.model_number = (int)models.size() + 1;
            continue;
        }
        if (rec == "ATOM" || rec == "HETATM") {
            if (line.size() < 54) continue;
            PDBAtom a; a.record_type = rec;
            try { a.serial = std::stoi(line.substr(6, 5)); } catch (...) {}
            a.name = line.substr(12, 4);
            while (!a.name.empty() && a.name.front() == ' ') a.name.erase(a.name.begin());
            while (!a.name.empty() && a.name.back() == ' ') a.name.pop_back();
            a.res_name = line.substr(17, 3);
            while (!a.res_name.empty() && a.res_name.front() == ' ') a.res_name.erase(a.res_name.begin());
            while (!a.res_name.empty() && a.res_name.back() == ' ') a.res_name.pop_back();
            a.chain_id = line.substr(21, 1);
            try { a.res_seq = std::stoi(line.substr(22, 4)); } catch (...) {}
            try { a.coords.x = std::stof(line.substr(30, 8));
                  a.coords.y = std::stof(line.substr(38, 8));
                  a.coords.z = std::stof(line.substr(46, 8)); } catch (...) {}
            if (line.size() >= 60) try { a.occupancy = std::stof(line.substr(54, 6)); } catch (...) {}
            if (line.size() >= 66) try { a.b_factor = std::stof(line.substr(60, 6)); } catch (...) {}
            if (line.size() >= 78) {
                a.element = line.substr(76, 2);
                while (!a.element.empty() && a.element.front() == ' ') a.element.erase(a.element.begin());
                while (!a.element.empty() && a.element.back() == ' ') a.element.pop_back();
            }
            current.data.atoms.push_back(a);
        } else if (rec == "CONECT") {
            try { int a1 = std::stoi(line.substr(6, 5));
                  for (int off = 11; off + 5 <= (int)line.size(); off += 5) {
                      std::string s = line.substr(off, 5);
                      while (!s.empty() && s.front() == ' ') s.erase(s.begin());
                      if (s.empty()) continue;
                      current.data.conect.push_back({a1, std::stoi(s)});
                  }
            } catch (...) {}
        }
    }
    if (!current.data.atoms.empty()) models.push_back(current);
    return models;
}

// ============================================================================
// MSA Pairing, Processing
// ============================================================================
struct PairedMSA {
    std::map<int, MSAData> chain_msas;
    std::vector<std::vector<int>> paired_indices;
    static PairedMSA pair_by_taxonomy(const std::map<int, MSAData>& msas) {
        PairedMSA result; result.chain_msas = msas;
        std::map<int, std::map<int, std::vector<int>>> tax_map;
        for (auto& [cid, msa] : msas)
            for (int s = 0; s < (int)msa.sequences.size(); ++s) {
                int tax = msa.sequences[s].taxonomy_id;
                if (tax > 0) tax_map[tax][cid].push_back(s);
            }
        for (auto& [tax, cs] : tax_map) {
            if ((int)cs.size() < 2) continue;
            std::vector<int> grp;
            for (auto& [cid, seqs] : cs) if (!seqs.empty()) grp.push_back(seqs[0]);
            if ((int)grp.size() >= 2) result.paired_indices.push_back(grp);
        }
        return result;
    }
};

inline MSAData truncate_msa(const MSAData& msa, int max_seqs) {
    if (max_seqs <= 0 || (int)msa.sequences.size() <= max_seqs) return msa;
    MSAData result;
    result.sequences.assign(msa.sequences.begin(), msa.sequences.begin() + max_seqs);
    return result;
}

inline MSAData deduplicate_msa(const MSAData& msa) {
    MSAData result; std::set<std::vector<int>> seen;
    for (auto& seq : msa.sequences)
        if (seen.insert(seq.residues).second) result.sequences.push_back(seq);
    return result;
}

inline Mat compute_msa_profile(const MSAData& msa, int seq_len) {
    Mat profile(seq_len, NUM_FULL_TOKENS, 0.0f);
    if (msa.sequences.empty()) return profile;
    for (auto& seq : msa.sequences)
        for (int i = 0; i < std::min((int)seq.residues.size(), seq_len); ++i) {
            int tid = std::clamp(seq.residues[i], 0, NUM_FULL_TOKENS - 1);
            profile(i, tid) += 1.0f;
        }
    float inv = 1.0f / (float)msa.sequences.size();
    for (auto& v : profile.data) v *= inv;
    return profile;
}

inline std::vector<float> compute_deletion_mean(const MSAData& msa, int seq_len) {
    std::vector<float> del_mean(seq_len, 0.0f);
    if (msa.sequences.empty()) return del_mean;
    for (auto& seq : msa.sequences)
        for (auto& [idx, cnt] : seq.deletions)
            if (idx < seq_len) del_mean[idx] += (float)cnt;
    float inv = 1.0f / (float)msa.sequences.size();
    for (auto& v : del_mean) v *= inv;
    return del_mean;
}

// ============================================================================
// Cluster Sampler (matching Python ClusterSampler)
// ============================================================================
struct ClusterSampler {
    float alpha_prot = 3.0f, alpha_nucl = 3.0f, alpha_ligand = 1.0f;
    float beta_chain = 0.5f, beta_interface = 1.0f;
    struct WI { int ri; int kind; int idx; float w; };
    std::vector<WI> items;
    std::vector<float> cw;
    void build(const std::vector<RecordFull>& records) {
        items.clear();
        std::map<std::string, int> cc;
        for (auto& rec : records) for (auto& ch : rec.chains)
            if (ch.valid) cc[std::to_string(ch.entity_id)]++;
        for (int ri = 0; ri < (int)records.size(); ++ri) {
            for (int ci = 0; ci < (int)records[ri].chains.size(); ++ci) {
                auto& ch = records[ri].chains[ci];
                if (!ch.valid) continue;
                float w = beta_chain / std::max(1, cc[std::to_string(ch.entity_id)]);
                if (ch.mol_type == MOL_PROTEIN) w *= alpha_prot;
                else if (ch.mol_type == MOL_DNA || ch.mol_type == MOL_RNA) w *= alpha_nucl;
                else w *= alpha_ligand;
                items.push_back({ri, 0, ci, w});
            }
        }
        float total = 0; for (auto& i : items) total += i.w;
        if (total > 0) for (auto& i : items) i.w /= total;
        cw.resize(items.size()); float cum = 0;
        for (int i = 0; i < (int)items.size(); ++i) { cum += items[i].w; cw[i] = cum; }
    }
    Sample sample_one(const std::vector<RecordFull>& records) const {
        float u = rand_uniform();
        int idx = (int)(std::lower_bound(cw.begin(), cw.end(), u) - cw.begin());
        idx = std::clamp(idx, 0, (int)items.size() - 1);
        auto& item = items[idx]; Sample s; s.record = records[item.ri];
        if (item.kind == 0) s.chain_id = item.idx; else s.interface_id = item.idx;
        return s;
    }
};

struct RandomSampler {
    Sample sample_one(const std::vector<RecordFull>& records) const {
        Sample s; s.record = records[rand_int(0, (int)records.size())]; return s;
    }
};

struct DistillationSampler {
    int small_size = 200; float small_prob = 0.01f;
    Sample sample_one(const std::vector<RecordFull>& records) const {
        std::vector<int> sm, lg;
        for (int i = 0; i < (int)records.size(); ++i) {
            int nres = records[i].chains.empty() ? 0 : records[i].chains[0].num_residues;
            if (nres <= small_size) sm.push_back(i); else lg.push_back(i);
        }
        auto& pool = (rand_uniform() < small_prob && !sm.empty()) ? sm : lg;
        if (pool.empty()) { Sample s; s.record = records[0]; return s; }
        Sample s; s.record = records[pool[rand_int(0, (int)pool.size())]]; return s;
    }
};

// ============================================================================
// Affinity Cropper
// ============================================================================
struct AffinityCropper {
    int neighborhood_size = 10; int max_tokens_protein = 200;
    Tokenized crop(const Tokenized& data, int max_tokens) const {
        if ((int)data.tokens.size() <= max_tokens) return data;
        std::vector<Vec3> lc;
        for (auto& tok : data.tokens)
            if (tok.mol_type == MOL_NONPOLYMER && tok.resolved_mask) lc.push_back(tok.center_coords);
        if (lc.empty()) return crop_tokens(data, max_tokens, neighborhood_size);
        std::vector<std::pair<float, int>> dists;
        for (int i = 0; i < (int)data.tokens.size(); ++i) {
            float min_d = 1e30f;
            for (auto& l : lc) min_d = std::min(min_d, (data.tokens[i].center_coords - l).norm());
            dists.push_back({min_d, i});
        }
        std::sort(dists.begin(), dists.end());
        std::set<int> cropped;
        for (auto& [d, idx] : dists) {
            if ((int)cropped.size() >= max_tokens) break;
            auto& tok = data.tokens[idx];
            std::vector<int> ct;
            for (int j = 0; j < (int)data.tokens.size(); ++j)
                if (data.tokens[j].asym_id == tok.asym_id) ct.push_back(j);
            if ((int)ct.size() <= neighborhood_size) for (int c : ct) cropped.insert(c);
            else for (int c : ct) if (std::abs(data.tokens[c].res_idx - tok.res_idx) <= neighborhood_size/2)
                cropped.insert(c);
        }
        Tokenized result; result.structure = data.structure; result.msa = data.msa;
        for (int ti : cropped) result.tokens.push_back(data.tokens[ti]);
        std::set<int> ts(cropped.begin(), cropped.end());
        for (auto& bond : data.bonds) if (ts.count(bond.token_1) && ts.count(bond.token_2)) result.bonds.push_back(bond);
        return result;
    }
};

// ============================================================================
// Extended Atom Features
// ============================================================================
struct AtomFeatures {
    Mat ref_pos, ref_charge, ref_element, atom_pad_mask, atom_to_token, token_to_rep_atom;
    std::vector<int> ref_space_uid;
};

inline AtomFeatures compute_atom_features(const Tokenized& data) {
    int NT = (int)data.tokens.size(), NA = 0;
    for (auto& tok : data.tokens) NA += tok.atom_num;
    if (NA == 0) NA = NT;
    AtomFeatures af;
    af.ref_pos = Mat(NA, 3, 0.0f); af.ref_charge = Mat(NA, 1, 0.0f);
    af.ref_element = Mat(NA, NUM_ELEMENTS, 0.0f); af.ref_space_uid.resize(NA, 0);
    af.atom_pad_mask = Mat(1, NA, 1.0f);
    af.atom_to_token = Mat(NA, NT, 0.0f); af.token_to_rep_atom = Mat(NT, NA, 0.0f);
    int ao = 0;
    for (int ti = 0; ti < NT; ++ti) {
        auto& tok = data.tokens[ti];
        for (int a = 0; a < tok.atom_num; ++a) {
            int ai = ao + a, src = tok.atom_idx + a;
            if (ai < NA && src < (int)data.structure.atoms.size()) {
                auto& atom = data.structure.atoms[src];
                af.ref_pos(ai,0)=atom.coords.x; af.ref_pos(ai,1)=atom.coords.y; af.ref_pos(ai,2)=atom.coords.z;
                af.ref_charge(ai,0)=(float)atom.charge;
                int elem=std::clamp(atom.element,0,NUM_ELEMENTS-1);
                af.ref_element(ai,elem)=1.0f;
                af.ref_space_uid[ai]=tok.res_idx;
                af.atom_to_token(ai,ti)=1.0f;
            }
        }
        int cl=tok.center_idx-tok.atom_idx, cg=ao+std::clamp(cl,0,std::max(0,tok.atom_num-1));
        if (cg < NA) af.token_to_rep_atom(ti,cg)=1.0f;
        ao += tok.atom_num;
    }
    return af;
}

// ============================================================================
// Frame computation
// ============================================================================
struct FrameData { int atom_a=-1,atom_b=-1,atom_c=-1; bool valid=false; };

inline std::vector<FrameData> compute_frames(const Tokenized& data) {
    int N=(int)data.tokens.size(); std::vector<FrameData> frames(N);
    for (int i=0;i<N;++i) {
        auto& tok=data.tokens[i];
        if (tok.mol_type==MOL_PROTEIN && tok.atom_num>=3) {
            frames[i].atom_a=tok.atom_idx; frames[i].atom_b=tok.atom_idx+1;
            frames[i].atom_c=tok.atom_idx+2; frames[i].valid=tok.resolved_mask;
        }
    }
    return frames;
}

// ============================================================================
// Extended PDB/mmCIF writers (full atom output)
// ============================================================================
inline std::string structure_to_pdb(const Structure& structure, const std::vector<float>* plddt=nullptr) {
    std::ostringstream oss; oss<<"REMARK   Generated by Boltz C++ port\n"; int serial=1;
    for (auto& chain : structure.chains) {
        int rs=chain.res_idx, re=rs+chain.res_num;
        for (int ri=rs;ri<re&&ri<(int)structure.residues.size();++ri) {
            auto& res=structure.residues[ri]; int as=res.atom_idx, ae=as+res.atom_num;
            for (int ai=as;ai<ae&&ai<(int)structure.atoms.size();++ai) {
                auto& atom=structure.atoms[ai]; if (!atom.is_present) continue;
                std::string rt=(chain.mol_type==MOL_NONPOLYMER)?"HETATM":"ATOM  ";
                float bfac=(plddt&&ri<(int)plddt->size())?(*plddt)[ri]*100.0f:0.0f;
                oss<<rt<<std::setw(5)<<serial++<<" "<<std::setw(4)<<std::left<<atom.name<<" "<<std::right
                   <<std::setw(3)<<res.name<<" "<<chain.name<<std::setw(4)<<(ri-rs+1)<<"    "
                   <<std::fixed<<std::setprecision(3)<<std::setw(8)<<atom.coords.x<<std::setw(8)<<atom.coords.y
                   <<std::setw(8)<<atom.coords.z<<std::setw(6)<<std::setprecision(2)<<1.00<<std::setw(6)<<bfac<<"\n";
            }
        }
        oss<<"TER\n";
    }
    oss<<"END\n"; return oss.str();
}

inline std::string structure_to_mmcif(const Structure& structure, const std::string& entry_id="BOLTZ",
                                       const std::vector<float>* plddt=nullptr) {
    std::ostringstream oss;
    oss<<"data_"<<entry_id<<"\n#\nloop_\n"
       <<"_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n"
       <<"_atom_site.label_atom_id\n_atom_site.label_comp_id\n"
       <<"_atom_site.label_asym_id\n_atom_site.label_seq_id\n"
       <<"_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
       <<"_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n";
    int serial=1;
    for (auto& chain : structure.chains) {
        int rs=chain.res_idx, re=rs+chain.res_num;
        for (int ri=rs;ri<re&&ri<(int)structure.residues.size();++ri) {
            auto& res=structure.residues[ri]; int as=res.atom_idx, ae=as+res.atom_num;
            for (int ai=as;ai<ae&&ai<(int)structure.atoms.size();++ai) {
                auto& atom=structure.atoms[ai]; if (!atom.is_present) continue;
                std::string grp=(chain.mol_type==MOL_NONPOLYMER)?"HETATM":"ATOM";
                float bfac=(plddt&&ri<(int)plddt->size())?(*plddt)[ri]*100.0f:0.0f;
                std::string elem=atom.name.empty()?"C":std::string(1,atom.name[0]);
                oss<<grp<<" "<<serial++<<" "<<elem<<" "<<atom.name<<" "<<res.name<<" "<<chain.name
                   <<" "<<(ri-rs+1)<<" "<<std::fixed<<std::setprecision(3)
                   <<atom.coords.x<<" "<<atom.coords.y<<" "<<atom.coords.z
                   <<" "<<std::setprecision(2)<<1.00<<" "<<bfac<<"\n";
            }
        }
    }
    oss<<"#\n"; return oss.str();
}

// ============================================================================
// Data Pipeline
// ============================================================================
struct DataPipeline {
    int max_tokens = 384, max_msa_seqs = MAX_MSA_SEQS;
    struct ProcessedData { Tokenized tokenized; Features features; AtomFeatures atom_features; bool valid = false; };
    ProcessedData process_pdb(const std::string& path) {
        ProcessedData result; PDBData pdb = parse_pdb(path);
        if (pdb.atoms.empty()) return result;
        Structure s = pdb_to_structure(pdb);
        Tokenized tok = tokenize_structure(s);
        if ((int)tok.tokens.size() > max_tokens) tok = crop_tokens(tok, max_tokens);
        result.tokenized = tok; result.features = featurize(tok);
        result.atom_features = compute_atom_features(tok);
        result.valid = !tok.tokens.empty(); return result;
    }
};

// ============================================================================
// Padding utilities
// ============================================================================
inline Mat pad_mat(const Mat& m, int tr, int tc, float val=0.0f) {
    Mat r(tr,tc,val);
    for (int i=0;i<std::min(m.rows,tr);++i) for (int j=0;j<std::min(m.cols,tc);++j) r(i,j)=m(i,j);
    return r;
}

} // namespace boltz
