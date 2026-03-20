#ifndef CELLENGINE_INTERNAL_TYPES_DEFINED
struct ODDIntervention {
    std::string key;
    std::string label;
    bool disable_electrical = false;
    bool disable_chemical = false;
    bool disable_diffusion = false;
    bool disable_self_chemical = false;
    bool disable_mechanical = false;
    bool freeze_sigma = false;
    bool freeze_gap_plasticity = false;
    bool disable_slow_drift = false;
    bool disable_teacher = false;
    int neutralize_gene_channel = -1;
    int freeze_gene_channel = -1;
    int perturb_gene_channel = -1;
    double perturb_gene_delta = 0.0;
    int ablate_force_cell = -1;
    int ablate_edge_src = -1;
    int ablate_edge_dst = -1;
};
#endif

#ifdef USE_CUDA
struct CudaTickParams {
    double tau_v_base = 0.0;
    double tau_v_gain = 0.0;
    double epsilon_base = 0.0;
    double epsilon_gain = 0.0;
    double fhn_a = 0.0;
    double fhn_b = 0.0;
    double tau_ca = 0.0;
    double alpha_ca = 0.0;
    double v_ca_thresh = 0.0;
    double chem_coeff = 0.0;
    double mech_base = 0.0;
    double chemical_balance = 0.0;
    double sigma_target = 0.0;
    double sigma_alpha = 0.0;
    double sigma_beta = 0.0;
    double basal_contractility = 0.0;
    double muscle_bias = 0.0;
    double energy_output_scale = 0.0;
    double energy_replenish = 0.0;
    double cost_vm = 0.0;
    double cost_ca = 0.0;
    double cost_mech = 0.0;
    double activity_decay = 0.0;
    double hebb_rate = 0.0;
    double decay_rate = 0.0;
    double teacher_hebb_boost = 0.0;
};

const char* kCellEngineCudaSource = R"CUDA(
extern "C" {
struct CudaTickParams {
    double tau_v_base;
    double tau_v_gain;
    double epsilon_base;
    double epsilon_gain;
    double fhn_a;
    double fhn_b;
    double tau_ca;
    double alpha_ca;
    double v_ca_thresh;
    double chem_coeff;
    double mech_base;
    double chemical_balance;
    double sigma_target;
    double sigma_alpha;
    double sigma_beta;
    double basal_contractility;
    double muscle_bias;
    double energy_output_scale;
    double energy_replenish;
    double cost_vm;
    double cost_ca;
    double cost_mech;
    double activity_decay;
    double hebb_rate;
    double decay_rate;
    double teacher_hebb_boost;
};

__device__ inline double clampd(const double v, const double lo, const double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ inline double sigmoid(const double x) {
    if (x >= 0.0) {
        const double z = exp(-x);
        return 1.0 / (1.0 + z);
    }
    const double z = exp(x);
    return z / (1.0 + z);
}

__global__ void cellengine_tick_kernel(
    int count,
    const int* active,
    const int* neighbor_offsets,
    const int* neighbor_flat,
    const int* cell_to_grid,
    const double* surface_feature,
    const double* stress,
    const double* V,
    const double* Wrec,
    const double* Ca,
    const double* sigma,
    const double* energy,
    const double* activity_avg,
    const double* chem_prev,
    const double* chem_field,
    const double* G,
    const double* gap,
    const double* gap_base,
    CudaTickParams params,
    double dt,
    int teaching_active,
    int disable_electrical,
    int disable_chemical,
    int disable_diffusion,
    int disable_self_chemical,
    int disable_mechanical,
    int freeze_sigma,
    int freeze_gap_plasticity,
    double* nextV,
    double* nextW,
    double* nextCa,
    double* nextSigma,
    double* nextEnergy,
    double* nextA,
    double* nextChem,
    double* nextActivity,
    double* nextGap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (active[i] == 0) {
        nextV[i] = 0.0;
        nextW[i] = 0.0;
        nextCa[i] = 0.0;
        nextSigma[i] = 0.1;
        nextEnergy[i] = 0.0;
        nextA[i] = 0.0;
        nextChem[i] = 0.0;
        nextActivity[i] = 0.0;
        for (int p = neighbor_offsets[i]; p < neighbor_offsets[i + 1]; ++p) nextGap[p] = 0.0;
        return;
    }

    const int gbase = i * 8;
    const double g0 = G[gbase + 0];
    const double g1 = G[gbase + 1];
    const double g2 = G[gbase + 2];
    const double g3 = G[gbase + 3];
    const double g4 = G[gbase + 4];
    const double g7 = G[gbase + 7];
    const double v = V[i];
    const double w = Wrec[i];
    const double energy_gate = energy[i] > 0.05 ? energy[i] : 0.05;
    const int grid_idx = cell_to_grid[i];

    double I_elec = 0.0;
    double I_chem = (grid_idx >= 0 && !disable_chemical && !disable_diffusion) ? chem_field[grid_idx] : 0.0;
    for (int p = neighbor_offsets[i]; p < neighbor_offsets[i + 1]; ++p) {
        const int j = neighbor_flat[p];
        if (active[j] == 0) continue;
        const double wgap = gap[p];
        if (!disable_electrical) {
            I_elec += wgap * (V[j] - v);
        }
        if (disable_diffusion && !disable_chemical) {
            I_chem += wgap * chem_prev[j];
        }
    }

    const double I_mech = disable_mechanical ? 0.0 : params.mech_base * (0.50 + 2.20 * g2) * stress[i];
    const double chem_self = (disable_chemical || disable_self_chemical) ? 0.0 : params.chemical_balance * chem_prev[i];
    const double I_total = sigma[i] * (I_elec + params.chem_coeff * I_chem + I_mech + chem_self) * energy_gate;

    const double tau_v = params.tau_v_base + params.tau_v_gain * (g0 - 0.5);
    const double tau_v_safe = tau_v > 0.35 ? tau_v : 0.35;
    const double eps = params.epsilon_base + params.epsilon_gain * (g0 - 0.5);
    const double eps_safe = eps > 0.015 ? eps : 0.015;
    const double dV = (v - (v * v * v) / 3.0 - w + I_total) / tau_v_safe;
    const double dW = eps_safe * (v + params.fhn_a - params.fhn_b * w);
    const double vn = clampd(v + dV * dt, -2.5, 2.5);
    const double wn = clampd(w + dW * dt, -2.5, 2.5);
    const double can = clampd(Ca[i] + dt * ((-Ca[i] / params.tau_ca) + params.alpha_ca * ((vn - params.v_ca_thresh) > 0.0 ? (vn - params.v_ca_thresh) : 0.0)), 0.0, 2.5);
    const double activity = params.activity_decay * activity_avg[i] + (1.0 - params.activity_decay) * fabs(vn);
    const double sigma_n = freeze_sigma
        ? sigma[i]
        : clampd(sigma[i] + dt * (params.sigma_alpha * (params.sigma_target - sigma[i]) +
                                  params.sigma_beta * (params.sigma_target - activity)),
                 0.10, 5.0);
    const double a_gate = 0.10 + 0.90 * sigmoid(2.5 * vn);
    const double mech_contract = ((stress[i] > 0.0) ? stress[i] : 0.0) * (0.15 + 1.10 * g2);
    const double a_mech = clampd(
        (0.20 + params.basal_contractility + 1.80 * g1 + params.muscle_bias * surface_feature[i]) *
            (tanh(2.2 * can) + 0.70 * mech_contract) * a_gate * energy_gate * params.energy_output_scale,
        -2.5, 2.5);
    const double chem = disable_chemical ? 0.0 : (0.15 + g3) * tanh(1.6 * can) * ((g4 >= 0.5) ? 1.0 : -1.0);
    const double energy_n = clampd(
        energy[i] + dt * (params.energy_replenish - params.cost_vm * fabs(vn) * fabs(vn) -
                          params.cost_ca * can - params.cost_mech * fabs(a_mech)),
        0.0, 1.0);

    nextV[i] = vn;
    nextW[i] = wn;
    nextCa[i] = can;
    nextSigma[i] = sigma_n;
    nextEnergy[i] = energy_n;
    nextA[i] = a_mech;
    nextChem[i] = chem;
    nextActivity[i] = activity;

    const double hebb_scale = params.hebb_rate * (0.35 + 1.2 * g7) * (teaching_active ? params.teacher_hebb_boost : 1.0);
    for (int p = neighbor_offsets[i]; p < neighbor_offsets[i + 1]; ++p) {
        const int j = neighbor_flat[p];
        if (active[j] == 0) {
            nextGap[p] = 0.0;
            continue;
        }
        const double corr_ij = clampd(V[i] * V[j], -1.0, 1.0);
        if (freeze_gap_plasticity) {
            nextGap[p] = gap[p];
        } else {
            const double gap_n = gap[p] + dt * (hebb_scale * corr_ij - params.decay_rate * (gap[p] - gap_base[p]));
            nextGap[p] = clampd(gap_n, 0.0, 1.0);
        }
    }
}

__global__ void cellengine_diffusion_kernel(
    int grid_count,
    const int* diffusion_offsets,
    const int* diffusion_flat,
    const int* grid_to_cell,
    const int* active,
    const double* chem_out,
    const double* field_in,
    double rate,
    double decay,
    double local_dt,
    double* field_out) {
    const int gi = blockIdx.x * blockDim.x + threadIdx.x;
    if (gi >= grid_count) return;
    const double self = field_in[gi];
    double neigh_sum = 0.0;
    int neigh_n = 0;
    for (int p = diffusion_offsets[gi]; p < diffusion_offsets[gi + 1]; ++p) {
        neigh_sum += field_in[diffusion_flat[p]];
        neigh_n += 1;
    }
    const double neigh_mean = neigh_n > 0 ? (neigh_sum / (double)neigh_n) : 0.0;
    const double lap = neigh_mean - self;
    const int cell_id = grid_to_cell[gi];
    const double source = (cell_id >= 0 && active[cell_id] != 0) ? chem_out[cell_id] : 0.0;
    const double next = self + local_dt * (rate * lap - decay * self + source);
    field_out[gi] = clampd(next, -6.0, 6.0);
}
}
)CUDA";

class CudaCellTickRuntime {
public:
    ~CudaCellTickRuntime() {
        if (module_) cuModuleUnload(module_);
        if (context_) cuDevicePrimaryCtxRelease(device_);
        free_device();
    }

    bool should_try(const int count, const int edge_count, const int grid_count) {
        const std::string mode = env_lower("CELLENGINE_USE_CUDA", "auto");
        if (mode == "0" || mode == "false" || mode == "off" || mode == "no") return false;
        if (!ready()) return false;
        if (mode == "1" || mode == "true" || mode == "on" || mode == "yes" || mode == "force") return true;
        const int work = count + edge_count + grid_count;
        return count >= 96 || work >= 2048;
    }

    bool tick(CellPopulation& pop,
              const BodyTemplate& body,
              const DecodedGenome& d,
              const CellEngineConfig& cfg,
              const double dt,
              const bool teaching_active,
              const ODDIntervention* intervention) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!ready_locked()) return false;
        if (cuCtxSetCurrent(context_) != CUDA_SUCCESS) return false;

        int count = pop.count;
        const int edge_count = static_cast<int>(body.neighbor_flat.size());
        int grid_count = std::max(1, body.grid_cell_count());
        if (!ensure_capacity(count, edge_count, grid_count)) return false;

        surface_feature_host_.assign(static_cast<std::size_t>(count), 0.0);
        for (int i = 0; i < count; ++i) {
            surface_feature_host_[static_cast<std::size_t>(i)] = body.cells[static_cast<std::size_t>(i)].features[3];
        }

        auto h2d = [](void* dst, const void* src, std::size_t bytes) {
            return bytes == 0 || cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
        };
        if (!h2d(d_active_, pop.active.data(), sizeof(int) * static_cast<std::size_t>(count))) return false;
        if (!h2d(d_neighbor_offsets_, body.neighbor_offsets.data(), sizeof(int) * body.neighbor_offsets.size())) return false;
        if (!h2d(d_neighbor_flat_, body.neighbor_flat.data(), sizeof(int) * body.neighbor_flat.size())) return false;
        if (!h2d(d_cell_to_grid_, body.cell_to_grid.data(), sizeof(int) * body.cell_to_grid.size())) return false;
        if (!h2d(d_grid_to_cell_, body.grid_to_cell.data(), sizeof(int) * body.grid_to_cell.size())) return false;
        if (!h2d(d_diffusion_offsets_, body.diffusion_offsets.data(), sizeof(int) * body.diffusion_offsets.size())) return false;
        if (!h2d(d_diffusion_flat_, body.diffusion_flat.data(), sizeof(int) * body.diffusion_flat.size())) return false;
        if (!h2d(d_surface_feature_, surface_feature_host_.data(), sizeof(double) * surface_feature_host_.size())) return false;
        if (!h2d(d_stress_, pop.stress.data(), sizeof(double) * static_cast<std::size_t>(count))) return false;
        if (!h2d(d_V_, pop.V.data(), sizeof(double) * static_cast<std::size_t>(count))) return false;
        if (!h2d(d_Wrec_, pop.Wrec.data(), sizeof(double) * static_cast<std::size_t>(count))) return false;
        if (!h2d(d_Ca_, pop.Ca.data(), sizeof(double) * static_cast<std::size_t>(count))) return false;
        if (!h2d(d_sigma_, pop.sigma.data(), sizeof(double) * static_cast<std::size_t>(count))) return false;
        if (!h2d(d_energy_, pop.energy.data(), sizeof(double) * static_cast<std::size_t>(count))) return false;
        if (!h2d(d_activity_, pop.activity_avg.data(), sizeof(double) * static_cast<std::size_t>(count))) return false;
        if (!h2d(d_chem_prev_, pop.chem_out.data(), sizeof(double) * static_cast<std::size_t>(count))) return false;
        if (!h2d(d_chem_field_, pop.chem_field.data(), sizeof(double) * static_cast<std::size_t>(grid_count))) return false;
        if (!h2d(d_G_, pop.G.data(), sizeof(double) * pop.G.size())) return false;
        if (!h2d(d_gap_, pop.gap.data(), sizeof(double) * pop.gap.size())) return false;
        if (!h2d(d_gap_base_, pop.gap_base.data(), sizeof(double) * pop.gap_base.size())) return false;

        CudaTickParams params{
            d.tau_v_base, d.tau_v_gain, d.epsilon_base, d.epsilon_gain, d.fhn_a, d.fhn_b,
            d.tau_ca, d.alpha_ca, d.v_ca_thresh, d.chem_coeff, d.mech_base, d.chemical_balance,
            d.sigma_target, d.sigma_alpha, d.sigma_beta, d.basal_contractility, d.muscle_bias,
            d.energy_output_scale, d.energy_replenish, d.cost_vm, d.cost_ca, d.cost_mech,
            d.activity_decay, d.hebb_rate, d.decay_rate, d.teacher_hebb_boost
        };
        double dt_arg = dt;
        int teaching_flag = teaching_active ? 1 : 0;
        int disable_electrical = (intervention && intervention->disable_electrical) ? 1 : 0;
        int disable_chemical = (intervention && intervention->disable_chemical) ? 1 : 0;
        int disable_diffusion = (intervention && intervention->disable_diffusion) ? 1 : 0;
        int disable_self_chemical = (intervention && intervention->disable_self_chemical) ? 1 : 0;
        int disable_mechanical = (intervention && intervention->disable_mechanical) ? 1 : 0;
        int freeze_sigma = (intervention && intervention->freeze_sigma) ? 1 : 0;
        int freeze_gap = (intervention && intervention->freeze_gap_plasticity) ? 1 : 0;

        void* tick_args[] = {
            &count, &d_active_, &d_neighbor_offsets_, &d_neighbor_flat_, &d_cell_to_grid_,
            &d_surface_feature_, &d_stress_, &d_V_, &d_Wrec_, &d_Ca_, &d_sigma_, &d_energy_,
            &d_activity_, &d_chem_prev_, &d_chem_field_, &d_G_, &d_gap_, &d_gap_base_, &params,
            &dt_arg, &teaching_flag, &disable_electrical, &disable_chemical,
            &disable_diffusion, &disable_self_chemical, &disable_mechanical, &freeze_sigma, &freeze_gap,
            &d_nextV_, &d_nextW_, &d_nextCa_, &d_nextSigma_, &d_nextEnergy_,
            &d_nextA_, &d_nextChem_, &d_nextActivity_, &d_nextGap_
        };
        const unsigned int block = 128;
        const unsigned int grid = static_cast<unsigned int>((count + static_cast<int>(block) - 1) / static_cast<int>(block));
        if (cuLaunchKernel(tick_kernel_, grid, 1, 1, block, 1, 1, 0, nullptr, tick_args, nullptr) != CUDA_SUCCESS) {
            return false;
        }

        if (disable_chemical || disable_diffusion) {
            if (cudaMemset(d_field_tmp_, 0, sizeof(double) * static_cast<std::size_t>(grid_count)) != cudaSuccess) return false;
            std::swap(d_chem_field_, d_field_tmp_);
        } else {
            const int diffusion_steps = std::max(1, cfg.chemical_diffusion_steps);
            double diff_rate = cfg.chemical_diffusion_rate;
            double diff_decay = cfg.chemical_decay;
            double local_dt = dt / static_cast<double>(diffusion_steps);
            const unsigned int diff_grid = static_cast<unsigned int>((grid_count + static_cast<int>(block) - 1) / static_cast<int>(block));
            for (int step = 0; step < diffusion_steps; ++step) {
                void* diff_args[] = {
                    &grid_count, &d_diffusion_offsets_, &d_diffusion_flat_, &d_grid_to_cell_, &d_active_,
                    &d_nextChem_, &d_chem_field_, &diff_rate, &diff_decay, &local_dt, &d_field_tmp_
                };
                if (cuLaunchKernel(diff_kernel_, diff_grid, 1, 1, block, 1, 1, 0, nullptr, diff_args, nullptr) != CUDA_SUCCESS) {
                    return false;
                }
                std::swap(d_chem_field_, d_field_tmp_);
            }
        }

        if (cudaDeviceSynchronize() != cudaSuccess) return false;

        auto d2h = [](void* dst, const void* src, std::size_t bytes) {
            return bytes == 0 || cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
        };
        if (!d2h(pop.V.data(), d_nextV_, sizeof(double) * static_cast<std::size_t>(count))) return false;
        if (!d2h(pop.Wrec.data(), d_nextW_, sizeof(double) * static_cast<std::size_t>(count))) return false;
        if (!d2h(pop.Ca.data(), d_nextCa_, sizeof(double) * static_cast<std::size_t>(count))) return false;
        if (!d2h(pop.sigma.data(), d_nextSigma_, sizeof(double) * static_cast<std::size_t>(count))) return false;
        if (!d2h(pop.energy.data(), d_nextEnergy_, sizeof(double) * static_cast<std::size_t>(count))) return false;
        if (!d2h(pop.a_mech.data(), d_nextA_, sizeof(double) * static_cast<std::size_t>(count))) return false;
        if (!d2h(pop.chem_out.data(), d_nextChem_, sizeof(double) * static_cast<std::size_t>(count))) return false;
        if (!d2h(pop.activity_avg.data(), d_nextActivity_, sizeof(double) * static_cast<std::size_t>(count))) return false;
        if (!d2h(pop.gap.data(), d_nextGap_, sizeof(double) * static_cast<std::size_t>(edge_count))) return false;
        if (!d2h(pop.chem_field.data(), d_chem_field_, sizeof(double) * static_cast<std::size_t>(grid_count))) return false;
        return true;
    }

private:
    bool ready() {
        std::lock_guard<std::mutex> lock(mu_);
        return ready_locked();
    }

    bool ready_locked() {
        if (initialized_) return available_;
        initialized_ = true;
        if (cuInit(0) != CUDA_SUCCESS) return false;
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) return false;
        if (cuDeviceGet(&device_, 0) != CUDA_SUCCESS) return false;
        if (cuDevicePrimaryCtxRetain(&context_, device_) != CUDA_SUCCESS) return false;
        if (cuCtxSetCurrent(context_) != CUDA_SUCCESS) return false;
        int major = 0;
        int minor = 0;
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_);
        const std::string arch = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);

        nvrtcProgram program;
        if (nvrtcCreateProgram(&program, kCellEngineCudaSource, "cellengine_tick.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS) {
            return false;
        }
        const char* options[] = {arch.c_str(), "--std=c++14"};
        const nvrtcResult compile_result = nvrtcCompileProgram(program, 2, options);
        std::size_t log_size = 0;
        nvrtcGetProgramLogSize(program, &log_size);
        if (log_size > 1) {
            compile_log_.assign(log_size, '\0');
            nvrtcGetProgramLog(program, compile_log_.data());
        }
        if (compile_result != NVRTC_SUCCESS) {
            nvrtcDestroyProgram(&program);
            return false;
        }
        std::size_t ptx_size = 0;
        nvrtcGetPTXSize(program, &ptx_size);
        std::string ptx(ptx_size, '\0');
        nvrtcGetPTX(program, ptx.data());
        nvrtcDestroyProgram(&program);
        if (cuModuleLoadDataEx(&module_, ptx.data(), 0, nullptr, nullptr) != CUDA_SUCCESS) return false;
        if (cuModuleGetFunction(&tick_kernel_, module_, "cellengine_tick_kernel") != CUDA_SUCCESS) return false;
        if (cuModuleGetFunction(&diff_kernel_, module_, "cellengine_diffusion_kernel") != CUDA_SUCCESS) return false;
        available_ = true;
        return true;
    }

    bool ensure_capacity(const int count, const int edge_count, const int grid_count) {
        return ensure_buffer(reinterpret_cast<void**>(&d_active_), sizeof(int) * static_cast<std::size_t>(count), cap_active_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_neighbor_offsets_), sizeof(int) * (static_cast<std::size_t>(count) + 1), cap_neighbor_offsets_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_neighbor_flat_), sizeof(int) * static_cast<std::size_t>(edge_count), cap_neighbor_flat_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_cell_to_grid_), sizeof(int) * static_cast<std::size_t>(count), cap_cell_to_grid_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_grid_to_cell_), sizeof(int) * static_cast<std::size_t>(grid_count), cap_grid_to_cell_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_diffusion_offsets_), sizeof(int) * (static_cast<std::size_t>(grid_count) + 1), cap_diffusion_offsets_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_diffusion_flat_), sizeof(int) * static_cast<std::size_t>(std::max(1, grid_count * 6)), cap_diffusion_flat_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_surface_feature_), sizeof(double) * static_cast<std::size_t>(count), cap_surface_feature_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_stress_), sizeof(double) * static_cast<std::size_t>(count), cap_stress_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_V_), sizeof(double) * static_cast<std::size_t>(count), cap_V_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_Wrec_), sizeof(double) * static_cast<std::size_t>(count), cap_Wrec_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_Ca_), sizeof(double) * static_cast<std::size_t>(count), cap_Ca_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_sigma_), sizeof(double) * static_cast<std::size_t>(count), cap_sigma_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_energy_), sizeof(double) * static_cast<std::size_t>(count), cap_energy_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_activity_), sizeof(double) * static_cast<std::size_t>(count), cap_activity_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_chem_prev_), sizeof(double) * static_cast<std::size_t>(count), cap_chem_prev_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_chem_field_), sizeof(double) * static_cast<std::size_t>(grid_count), cap_chem_field_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_field_tmp_), sizeof(double) * static_cast<std::size_t>(grid_count), cap_field_tmp_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_G_), sizeof(double) * static_cast<std::size_t>(count * kGeneExprDim), cap_G_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_gap_), sizeof(double) * static_cast<std::size_t>(edge_count), cap_gap_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_gap_base_), sizeof(double) * static_cast<std::size_t>(edge_count), cap_gap_base_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_nextV_), sizeof(double) * static_cast<std::size_t>(count), cap_nextV_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_nextW_), sizeof(double) * static_cast<std::size_t>(count), cap_nextW_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_nextCa_), sizeof(double) * static_cast<std::size_t>(count), cap_nextCa_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_nextSigma_), sizeof(double) * static_cast<std::size_t>(count), cap_nextSigma_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_nextEnergy_), sizeof(double) * static_cast<std::size_t>(count), cap_nextEnergy_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_nextA_), sizeof(double) * static_cast<std::size_t>(count), cap_nextA_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_nextChem_), sizeof(double) * static_cast<std::size_t>(count), cap_nextChem_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_nextActivity_), sizeof(double) * static_cast<std::size_t>(count), cap_nextActivity_) &&
               ensure_buffer(reinterpret_cast<void**>(&d_nextGap_), sizeof(double) * static_cast<std::size_t>(edge_count), cap_nextGap_);
    }

    bool ensure_buffer(void** ptr, const std::size_t bytes, std::size_t& cap) {
        if (bytes == 0) return true;
        if (bytes <= cap && *ptr != nullptr) return true;
        if (*ptr != nullptr) cudaFree(*ptr);
        *ptr = nullptr;
        if (cudaMalloc(ptr, bytes) != cudaSuccess) return false;
        cap = bytes;
        return true;
    }

    void free_device() {
        auto release = [](void*& ptr) {
            if (ptr) cudaFree(ptr);
            ptr = nullptr;
        };
        release(reinterpret_cast<void*&>(d_active_));
        release(reinterpret_cast<void*&>(d_neighbor_offsets_));
        release(reinterpret_cast<void*&>(d_neighbor_flat_));
        release(reinterpret_cast<void*&>(d_cell_to_grid_));
        release(reinterpret_cast<void*&>(d_grid_to_cell_));
        release(reinterpret_cast<void*&>(d_diffusion_offsets_));
        release(reinterpret_cast<void*&>(d_diffusion_flat_));
        release(reinterpret_cast<void*&>(d_surface_feature_));
        release(reinterpret_cast<void*&>(d_stress_));
        release(reinterpret_cast<void*&>(d_V_));
        release(reinterpret_cast<void*&>(d_Wrec_));
        release(reinterpret_cast<void*&>(d_Ca_));
        release(reinterpret_cast<void*&>(d_sigma_));
        release(reinterpret_cast<void*&>(d_energy_));
        release(reinterpret_cast<void*&>(d_activity_));
        release(reinterpret_cast<void*&>(d_chem_prev_));
        release(reinterpret_cast<void*&>(d_chem_field_));
        release(reinterpret_cast<void*&>(d_field_tmp_));
        release(reinterpret_cast<void*&>(d_G_));
        release(reinterpret_cast<void*&>(d_gap_));
        release(reinterpret_cast<void*&>(d_gap_base_));
        release(reinterpret_cast<void*&>(d_nextV_));
        release(reinterpret_cast<void*&>(d_nextW_));
        release(reinterpret_cast<void*&>(d_nextCa_));
        release(reinterpret_cast<void*&>(d_nextSigma_));
        release(reinterpret_cast<void*&>(d_nextEnergy_));
        release(reinterpret_cast<void*&>(d_nextA_));
        release(reinterpret_cast<void*&>(d_nextChem_));
        release(reinterpret_cast<void*&>(d_nextActivity_));
        release(reinterpret_cast<void*&>(d_nextGap_));
    }

    std::mutex mu_;
    bool initialized_ = false;
    bool available_ = false;
    std::string compile_log_;
    CUdevice device_{};
    CUcontext context_ = nullptr;
    CUmodule module_ = nullptr;
    CUfunction tick_kernel_ = nullptr;
    CUfunction diff_kernel_ = nullptr;

    std::vector<double> surface_feature_host_;

    int* d_active_ = nullptr;
    int* d_neighbor_offsets_ = nullptr;
    int* d_neighbor_flat_ = nullptr;
    int* d_cell_to_grid_ = nullptr;
    int* d_grid_to_cell_ = nullptr;
    int* d_diffusion_offsets_ = nullptr;
    int* d_diffusion_flat_ = nullptr;
    double* d_surface_feature_ = nullptr;
    double* d_stress_ = nullptr;
    double* d_V_ = nullptr;
    double* d_Wrec_ = nullptr;
    double* d_Ca_ = nullptr;
    double* d_sigma_ = nullptr;
    double* d_energy_ = nullptr;
    double* d_activity_ = nullptr;
    double* d_chem_prev_ = nullptr;
    double* d_chem_field_ = nullptr;
    double* d_field_tmp_ = nullptr;
    double* d_G_ = nullptr;
    double* d_gap_ = nullptr;
    double* d_gap_base_ = nullptr;
    double* d_nextV_ = nullptr;
    double* d_nextW_ = nullptr;
    double* d_nextCa_ = nullptr;
    double* d_nextSigma_ = nullptr;
    double* d_nextEnergy_ = nullptr;
    double* d_nextA_ = nullptr;
    double* d_nextChem_ = nullptr;
    double* d_nextActivity_ = nullptr;
    double* d_nextGap_ = nullptr;

    std::size_t cap_active_ = 0;
    std::size_t cap_neighbor_offsets_ = 0;
    std::size_t cap_neighbor_flat_ = 0;
    std::size_t cap_cell_to_grid_ = 0;
    std::size_t cap_grid_to_cell_ = 0;
    std::size_t cap_diffusion_offsets_ = 0;
    std::size_t cap_diffusion_flat_ = 0;
    std::size_t cap_surface_feature_ = 0;
    std::size_t cap_stress_ = 0;
    std::size_t cap_V_ = 0;
    std::size_t cap_Wrec_ = 0;
    std::size_t cap_Ca_ = 0;
    std::size_t cap_sigma_ = 0;
    std::size_t cap_energy_ = 0;
    std::size_t cap_activity_ = 0;
    std::size_t cap_chem_prev_ = 0;
    std::size_t cap_chem_field_ = 0;
    std::size_t cap_field_tmp_ = 0;
    std::size_t cap_G_ = 0;
    std::size_t cap_gap_ = 0;
    std::size_t cap_gap_base_ = 0;
    std::size_t cap_nextV_ = 0;
    std::size_t cap_nextW_ = 0;
    std::size_t cap_nextCa_ = 0;
    std::size_t cap_nextSigma_ = 0;
    std::size_t cap_nextEnergy_ = 0;
    std::size_t cap_nextA_ = 0;
    std::size_t cap_nextChem_ = 0;
    std::size_t cap_nextActivity_ = 0;
    std::size_t cap_nextGap_ = 0;
};

CudaCellTickRuntime& cuda_cell_tick_runtime() {
    static CudaCellTickRuntime runtime;
    return runtime;
}
#endif

#ifndef CELLENGINE_INTERNAL_TYPES_DEFINED
struct TrialStats {
    int ticks = 0;
    double mean_abs_force = 0.0;
    double mean_energy = 0.0;
    double mean_abs_theta = 0.0;
    double task_primary = 0.0;
    double task_secondary = 0.0;
    bool success = false;
};

struct ReplayFrame {
    int tick = 0;
    double x = 0.0;
    double theta = 0.0;
    double x_dot = 0.0;
    double theta_dot = 0.0;
    double x_ddot = 0.0;
    double theta_ddot = 0.0;
    double organism_force = 0.0;
    double teacher_force = 0.0;
    double total_force = 0.0;
    double mean_energy = 0.0;
    double mean_stress = 0.0;
    double mean_activity = 0.0;
    double mean_calcium = 0.0;
    double active_fraction = 0.0;
    double task_aux_a = 0.0;
    double task_aux_b = 0.0;
    double task_primary = 0.0;
    double task_secondary = 0.0;
    int task_counter = 0;
    bool task_success = false;
    bool damage_event = false;
    bool terminal = false;
};

struct ReplayCellState {
    int tick = 0;
    int cell_id = 0;
    int active = 0;
    double V = 0.0;
    double Wrec = 0.0;
    double Ca = 0.0;
    double sigma = 0.0;
    double energy = 0.0;
    double a_mech = 0.0;
    double chem_out = 0.0;
    double stress = 0.0;
    double activity = 0.0;
    double I_elec = 0.0;
    double I_chem = 0.0;
    double I_mech = 0.0;
    double I_self_chem = 0.0;
    double I_total = 0.0;
    double stress_hinge = 0.0;
    double stress_lateral = 0.0;
    double stress_gravity = 0.0;
    double stress_drift = 0.0;
    double stress_neighbor = 0.0;
    double stress_final = 0.0;
    double mech_a_gate = 0.0;
    double mech_contract = 0.0;
    double mech_prefactor = 0.0;
    double mech_force_contrib = 0.0;
};

struct ReplayEdgeState {
    int tick = 0;
    int src_cell_id = 0;
    int dst_cell_id = 0;
    int src_active = 0;
    int dst_active = 0;
    double gap_forward = 0.0;
    double gap_reverse = 0.0;
    double gap_mean = 0.0;
    double corr_forward = 0.0;
    double corr_reverse = 0.0;
    double hebb_delta_forward = 0.0;
    double hebb_delta_reverse = 0.0;
    double decay_delta_forward = 0.0;
    double decay_delta_reverse = 0.0;
    double old_gap_forward = 0.0;
    double old_gap_reverse = 0.0;
    double new_gap_forward = 0.0;
    double new_gap_reverse = 0.0;
};

struct RLReplayResult {
    std::string algorithm;
    EvaluationSummary summary;
    std::vector<ReplayFrame> frames;
    std::vector<int> actions;
};

struct CellEvalResult {
    EvaluationSummary summary;
    double score = 0.0;
};

struct EvolutionResult {
    Genome best_genome;
    double best_search_score = -1.0;
    int best_generation = 0;
};
#endif
