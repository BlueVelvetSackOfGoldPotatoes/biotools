#include "core/tensor/cuda_backend.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#endif

namespace {

#ifdef USE_CUDA
bool env_flag(const char* key, bool fallback) {
    const char* v = std::getenv(key);
    if (!v) return fallback;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
    if (s == "0" || s == "false" || s == "no" || s == "off") return false;
    return fallback;
}

double env_double(const char* key, double fallback) {
    const char* v = std::getenv(key);
    if (!v) return fallback;
    char* end = nullptr;
    const double parsed = std::strtod(v, &end);
    if (end == v) return fallback;
    return parsed;
}
#endif

#ifdef USE_CUDA

class CudaMatmulState {
public:
    CudaMatmulState() = default;
    ~CudaMatmulState() {
        if (d_a_) cudaFree(d_a_);
        if (d_b_) cudaFree(d_b_);
        if (d_c_) cudaFree(d_c_);
        if (handle_) cublasDestroy(handle_);
    }

    bool ready() {
        std::lock_guard<std::mutex> lock(mu_);
        if (initialized_) return available_;
        initialized_ = true;

        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
            available_ = false;
            return false;
        }
        if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
            handle_ = nullptr;
            available_ = false;
            return false;
        }

        available_ = true;
        return true;
    }

    bool matmul(const double* a,
                const double* b,
                double* c,
                std::size_t m,
                std::size_t n,
                std::size_t k) {
        if (m == 0 || n == 0 || k == 0) return true;
        if (m > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            k > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            return false;
        }

        std::lock_guard<std::mutex> lock(mu_);
        if (!available_ || !handle_) return false;

        const std::size_t bytes_a = m * k * sizeof(double);
        const std::size_t bytes_b = k * n * sizeof(double);
        const std::size_t bytes_c = m * n * sizeof(double);

        if (!ensure_capacity(bytes_a, bytes_b, bytes_c)) return false;

        if (cudaMemcpy(d_a_, a, bytes_a, cudaMemcpyHostToDevice) != cudaSuccess) return false;
        if (cudaMemcpy(d_b_, b, bytes_b, cudaMemcpyHostToDevice) != cudaSuccess) return false;

        // Row-major trick:
        // C_row(m,n) = A_row(m,k) * B_row(k,n)
        // Equivalent in column-major memory with no transpose:
        // C_col(n,m) = B_col(n,k) * A_col(k,m)
        const int mm = static_cast<int>(m);
        const int nn = static_cast<int>(n);
        const int kk = static_cast<int>(k);
        const double alpha = 1.0;
        const double beta = 0.0;
        const cublasStatus_t st = cublasDgemm(
            handle_,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            nn,
            mm,
            kk,
            &alpha,
            d_b_,
            nn,
            d_a_,
            kk,
            &beta,
            d_c_,
            nn
        );
        if (st != CUBLAS_STATUS_SUCCESS) return false;

        if (cudaMemcpy(c, d_c_, bytes_c, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        return true;
    }

private:
    bool ensure_capacity(std::size_t bytes_a, std::size_t bytes_b, std::size_t bytes_c) {
        if (bytes_a > cap_a_) {
            if (d_a_) cudaFree(d_a_);
            d_a_ = nullptr;
            if (cudaMalloc(reinterpret_cast<void**>(&d_a_), bytes_a) != cudaSuccess) return false;
            cap_a_ = bytes_a;
        }
        if (bytes_b > cap_b_) {
            if (d_b_) cudaFree(d_b_);
            d_b_ = nullptr;
            if (cudaMalloc(reinterpret_cast<void**>(&d_b_), bytes_b) != cudaSuccess) return false;
            cap_b_ = bytes_b;
        }
        if (bytes_c > cap_c_) {
            if (d_c_) cudaFree(d_c_);
            d_c_ = nullptr;
            if (cudaMalloc(reinterpret_cast<void**>(&d_c_), bytes_c) != cudaSuccess) return false;
            cap_c_ = bytes_c;
        }
        return true;
    }

    std::mutex mu_;
    bool initialized_ = false;
    bool available_ = false;

    cublasHandle_t handle_ = nullptr;
    double* d_a_ = nullptr;
    double* d_b_ = nullptr;
    double* d_c_ = nullptr;
    std::size_t cap_a_ = 0;
    std::size_t cap_b_ = 0;
    std::size_t cap_c_ = 0;
};

CudaMatmulState& cuda_state() {
    static CudaMatmulState state;
    return state;
}

#endif

} // namespace

namespace tensor_cuda {

bool should_try_cuda_for_matmul(std::size_t m, std::size_t n, std::size_t k) {
#ifdef USE_CUDA
    if (!env_flag("TENSOR_USE_CUDA", true)) return false;
    if (!cuda_state().ready()) return false;

    if (env_flag("TENSOR_FORCE_CUDA", false)) return true;

    // 2 * m*n*k floating-point ops for GEMM.
    const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const double min_flops = env_double("TENSOR_CUDA_MIN_FLOPS", 2.0e7);
    return flops >= min_flops;
#else
    (void)m;
    (void)n;
    (void)k;
    return false;
#endif
}

bool matmul_host(const double* a,
                 const double* b,
                 double* c,
                 std::size_t m,
                 std::size_t n,
                 std::size_t k) {
#ifdef USE_CUDA
    return cuda_state().matmul(a, b, c, m, n, k);
#else
    (void)a;
    (void)b;
    (void)c;
    (void)m;
    (void)n;
    (void)k;
    return false;
#endif
}

} // namespace tensor_cuda
