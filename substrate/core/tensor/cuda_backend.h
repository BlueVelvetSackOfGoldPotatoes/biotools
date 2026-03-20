#pragma once

#include <cstddef>

namespace tensor_cuda {

// Runtime policy for whether a matmul should be offloaded.
bool should_try_cuda_for_matmul(std::size_t m, std::size_t n, std::size_t k);

// Host-pointer matmul using CUDA/cuBLAS.
// Returns true on success, false if backend unavailable or an error occurs.
bool matmul_host(const double* a,
                 const double* b,
                 double* c,
                 std::size_t m,
                 std::size_t n,
                 std::size_t k);

} // namespace tensor_cuda

