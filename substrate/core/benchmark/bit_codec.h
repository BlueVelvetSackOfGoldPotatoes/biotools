#pragma once

#include "core/tensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace benchmark::bitcodec {

void append_one_hot_bits(std::vector<uint8_t>& out, int value, int cardinality);

std::vector<uint8_t> encode_categorical_vector_one_hot(const std::vector<int>& values, int cardinality);

Tensor bits_to_tensor_row(const std::vector<uint8_t>& bits);

} // namespace benchmark::bitcodec

