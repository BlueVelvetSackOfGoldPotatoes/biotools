#pragma once

#include "core/bio/bio_runtime.h"
#include "core/tensor/tensor.h"

namespace bio {

inline Tensor affine(const Tensor& input,
                     Tensor& weight,
                     const Tensor* bias = nullptr,
                     bool weight_transposed = false) {
    Tensor effective_input = input;
    if (auto* rt = active_runtime()) {
        rt->apply_linear_input_transport(&weight, input, effective_input);
    }

    Tensor out = weight_transposed ? effective_input.matmul(weight.transpose())
                                   : effective_input.matmul(weight);
    if (bias) {
        out = out.add_row_broadcast(*bias);
    }

    if (auto* rt = active_runtime()) {
        rt->record_linear_activity(&weight, effective_input, out);
    }
    return out;
}

} // namespace bio

