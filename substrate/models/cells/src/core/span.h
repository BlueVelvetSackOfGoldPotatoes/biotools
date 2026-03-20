#pragma once

#include <cstddef>
#include <vector>

namespace cells {

template <typename T>
class Span {
public:
    Span() : ptr_(nullptr), size_(0) {}
    Span(T* ptr, std::size_t size) : ptr_(ptr), size_(size) {}

    template <typename U>
    Span(std::vector<U>& v) : ptr_(v.data()), size_(v.size()) {}

    template <typename U>
    Span(const std::vector<U>& v) : ptr_(v.data()), size_(v.size()) {}

    T* data() const { return ptr_; }
    std::size_t size() const { return size_; }

    T& operator[](std::size_t i) const { return ptr_[i]; }

    T* begin() const { return ptr_; }
    T* end() const { return ptr_ + size_; }

private:
    T* ptr_;
    std::size_t size_;
};

} // namespace cells

