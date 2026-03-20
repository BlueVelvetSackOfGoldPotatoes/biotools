#include "core/benchmark/bit_codec.h"

#include <stdexcept>

namespace benchmark::bitcodec {

void append_one_hot_bits(std::vector<uint8_t>& out, int value, int cardinality) {
    if (cardinality <= 0) {
        throw std::runtime_error("append_one_hot_bits: cardinality must be > 0");
    }
    if (value < 0 || value >= cardinality) {
        throw std::runtime_error("append_one_hot_bits: value out of range");
    }

    out.reserve(out.size() + static_cast<std::size_t>(cardinality));
    for (int i = 0; i < cardinality; ++i) {
        out.push_back(static_cast<uint8_t>(i == value ? 1 : 0));
    }
}

std::vector<uint8_t> encode_categorical_vector_one_hot(const std::vector<int>& values, int cardinality) {
    std::vector<uint8_t> bits;
    bits.reserve(values.size() * static_cast<std::size_t>(cardinality));
    for (const int value : values) append_one_hot_bits(bits, value, cardinality);
    return bits;
}

Tensor bits_to_tensor_row(const std::vector<uint8_t>& bits) {
    Tensor out(1, bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        out(0, i) = bits[i] ? 1.0 : 0.0;
    }
    return out;
}

} // namespace benchmark::bitcodec

