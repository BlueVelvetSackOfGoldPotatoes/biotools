#include "tensor.h"
#include "core/tensor/cuda_backend.h"

#ifdef _OPENMP
#define TENSOR_OMP_PAR_FOR _Pragma("omp parallel for schedule(static)")
#define TENSOR_OMP_PAR_FOR_COLLAPSE2 _Pragma("omp parallel for collapse(2) schedule(static)")
#define TENSOR_OMP_PAR_FOR_REDUCE_SUM _Pragma("omp parallel for reduction(+ : s) schedule(static)")
#define TENSOR_OMP_PAR_FOR_REDUCE_MAX _Pragma("omp parallel for reduction(max : mx) schedule(static)")
#define TENSOR_OMP_PAR_FOR_REDUCE_MIN _Pragma("omp parallel for reduction(min : mn) schedule(static)")
#else
#define TENSOR_OMP_PAR_FOR
#define TENSOR_OMP_PAR_FOR_COLLAPSE2
#define TENSOR_OMP_PAR_FOR_REDUCE_SUM
#define TENSOR_OMP_PAR_FOR_REDUCE_MAX
#define TENSOR_OMP_PAR_FOR_REDUCE_MIN
#endif

// --- Constructors ---
Tensor::Tensor() : rows(0), cols(0) {}
Tensor::Tensor(size_t r, size_t c) : data(r * c, 0.0), rows(r), cols(c) {}
Tensor::Tensor(size_t r, size_t c, double val) : data(r * c, val), rows(r), cols(c) {}
Tensor::Tensor(size_t r, size_t c, const std::vector<double>& d) : data(d), rows(r), cols(c) {
    TENSOR_CHECK(d.size() == r * c, "data size mismatch in constructor");
}

// --- Element access ---
double& Tensor::operator()(size_t i, size_t j) { return data[i * cols + j]; }
double Tensor::operator()(size_t i, size_t j) const { return data[i * cols + j]; }
double& Tensor::at(size_t i) {
    TENSOR_CHECK(i < data.size(), "flat index out of bounds");
    return data[i];
}
double Tensor::at(size_t i) const {
    TENSOR_CHECK(i < data.size(), "flat index out of bounds");
    return data[i];
}

// --- Shape ---
Tensor Tensor::reshape(size_t nr, size_t nc) const {
    TENSOR_CHECK(nr * nc == rows * cols, "reshape size mismatch");
    return Tensor(nr, nc, data);
}

Tensor Tensor::flatten() const { return reshape(1, rows * cols); }

Tensor Tensor::slice_rows(size_t start, size_t count) const {
    TENSOR_CHECK(start + count <= rows, "slice_rows out of bounds");
    Tensor r(count, cols);
    std::copy(data.begin() + start * cols, data.begin() + (start + count) * cols, r.data.begin());
    return r;
}

Tensor Tensor::slice_cols(size_t start, size_t count) const {
    TENSOR_CHECK(start + count <= cols, "slice_cols out of bounds");
    Tensor r(rows, count);
    TENSOR_OMP_PAR_FOR_COLLAPSE2
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < count; ++j)
            r(i, j) = (*this)(i, start + j);
    return r;
}

Tensor Tensor::row(size_t i) const { return slice_rows(i, 1); }
Tensor Tensor::col(size_t j) const { return slice_cols(j, 1); }

void Tensor::set_row(size_t i, const Tensor& row_data) {
    TENSOR_CHECK(i < rows, "set_row: row index out of bounds");
    TENSOR_CHECK(row_data.cols == cols && row_data.rows == 1, "set_row: shape mismatch");
    std::copy(row_data.data.begin(), row_data.data.end(), data.begin() + i * cols);
}

// --- Matrix ops ---
Tensor Tensor::matmul(const Tensor& other) const {
    TENSOR_CHECK(cols == other.rows, "matmul dimension mismatch");
    Tensor result(rows, other.cols);
    if (tensor_cuda::should_try_cuda_for_matmul(rows, other.cols, cols)) {
        if (tensor_cuda::matmul_host(
                data.data(),
                other.data.data(),
                result.data.data(),
                rows,
                other.cols,
                cols
            )) {
            return result;
        }
    }

    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < rows; ++i)
        for (size_t k = 0; k < cols; ++k) {
            double a = data[i * cols + k];
            for (size_t j = 0; j < other.cols; ++j)
                result.data[i * other.cols + j] += a * other.data[k * other.cols + j];
        }
    return result;
}

Tensor Tensor::transpose() const {
    Tensor r(cols, rows);
    TENSOR_OMP_PAR_FOR_COLLAPSE2
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            r(j, i) = (*this)(i, j);
    return r;
}

// --- Element-wise ---
Tensor Tensor::operator+(const Tensor& o) const {
    TENSOR_CHECK(same_shape(o), "operator+ shape mismatch");
    Tensor r(rows, cols);
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < data.size(); ++i) r.data[i] = data[i] + o.data[i];
    return r;
}
Tensor Tensor::operator-(const Tensor& o) const {
    TENSOR_CHECK(same_shape(o), "operator- shape mismatch");
    Tensor r(rows, cols);
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < data.size(); ++i) r.data[i] = data[i] - o.data[i];
    return r;
}
Tensor Tensor::operator*(const Tensor& o) const {
    TENSOR_CHECK(same_shape(o), "operator* shape mismatch");
    Tensor r(rows, cols);
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < data.size(); ++i) r.data[i] = data[i] * o.data[i];
    return r;
}
Tensor Tensor::operator/(const Tensor& o) const {
    TENSOR_CHECK(same_shape(o), "operator/ shape mismatch");
    Tensor r(rows, cols);
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < data.size(); ++i) r.data[i] = data[i] / o.data[i];
    return r;
}
Tensor Tensor::operator*(double s) const {
    Tensor r(rows, cols);
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < data.size(); ++i) r.data[i] = data[i] * s;
    return r;
}
Tensor Tensor::operator/(double s) const {
    Tensor r(rows, cols);
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < data.size(); ++i) r.data[i] = data[i] / s;
    return r;
}
Tensor Tensor::operator+(double s) const {
    Tensor r(rows, cols);
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < data.size(); ++i) r.data[i] = data[i] + s;
    return r;
}
Tensor Tensor::operator-(double s) const {
    Tensor r(rows, cols);
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < data.size(); ++i) r.data[i] = data[i] - s;
    return r;
}
Tensor Tensor::operator-() const {
    Tensor r(rows, cols);
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < data.size(); ++i) r.data[i] = -data[i];
    return r;
}

Tensor& Tensor::operator+=(const Tensor& o) {
    TENSOR_CHECK(same_shape(o), "operator+= shape mismatch");
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < data.size(); ++i) data[i] += o.data[i];
    return *this;
}
Tensor& Tensor::operator-=(const Tensor& o) {
    TENSOR_CHECK(same_shape(o), "operator-= shape mismatch");
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < data.size(); ++i) data[i] -= o.data[i];
    return *this;
}
Tensor& Tensor::operator*=(const Tensor& o) {
    TENSOR_CHECK(same_shape(o), "operator*= shape mismatch");
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < data.size(); ++i) data[i] *= o.data[i];
    return *this;
}
Tensor& Tensor::operator*=(double s) {
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < data.size(); ++i) data[i] *= s;
    return *this;
}
Tensor& Tensor::operator+=(double s) {
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < data.size(); ++i) data[i] += s;
    return *this;
}
Tensor& Tensor::operator-=(double s) {
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < data.size(); ++i) data[i] -= s;
    return *this;
}

// --- Reductions ---
Tensor Tensor::sum_rows() const {
    Tensor r(1, cols);
    TENSOR_OMP_PAR_FOR
    for (size_t j = 0; j < cols; ++j) {
        double col_sum = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            col_sum += (*this)(i, j);
        }
        r(0, j) = col_sum;
    }
    return r;
}

Tensor Tensor::sum_cols() const {
    Tensor r(rows, 1);
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            r(i, 0) += (*this)(i, j);
    return r;
}

double Tensor::sum() const {
    double s = 0;
    TENSOR_OMP_PAR_FOR_REDUCE_SUM
    for (auto v : data) s += v;
    return s;
}

double Tensor::mean() const {
    TENSOR_CHECK(!data.empty(), "mean() on empty tensor");
    return sum() / static_cast<double>(data.size());
}

double Tensor::max_val() const {
    TENSOR_CHECK(!data.empty(), "max_val() on empty tensor");
    double mx = data[0];
    TENSOR_OMP_PAR_FOR_REDUCE_MAX
    for (std::size_t i = 0; i < data.size(); ++i) {
        mx = std::max(mx, data[i]);
    }
    return mx;
}

double Tensor::min_val() const {
    TENSOR_CHECK(!data.empty(), "min_val() on empty tensor");
    double mn = data[0];
    TENSOR_OMP_PAR_FOR_REDUCE_MIN
    for (std::size_t i = 0; i < data.size(); ++i) {
        mn = std::min(mn, data[i]);
    }
    return mn;
}

size_t Tensor::argmax() const {
    TENSOR_CHECK(!data.empty(), "argmax() on empty tensor");
    return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
}

Tensor Tensor::argmax_rows() const {
    TENSOR_CHECK(cols > 0, "argmax_rows() on zero-column tensor");
    Tensor r(rows, 1);
    for (size_t i = 0; i < rows; ++i) {
        double best = (*this)(i, 0);
        size_t idx = 0;
        for (size_t j = 1; j < cols; ++j) {
            if ((*this)(i, j) > best) { best = (*this)(i, j); idx = j; }
        }
        r(i, 0) = static_cast<double>(idx);
    }
    return r;
}

Tensor Tensor::max_rows() const {
    TENSOR_CHECK(cols > 0, "max_rows() on zero-column tensor");
    Tensor r(rows, 1);
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < rows; ++i) {
        double best = (*this)(i, 0);
        for (size_t j = 1; j < cols; ++j)
            if ((*this)(i, j) > best) best = (*this)(i, j);
        r(i, 0) = best;
    }
    return r;
}

// --- Math ---
Tensor Tensor::apply(const std::function<double(double)>& fn) const {
    Tensor r(rows, cols);
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < data.size(); ++i) r.data[i] = fn(data[i]);
    return r;
}

Tensor Tensor::square() const { return apply([](double x) { return x * x; }); }
Tensor Tensor::sqrt_t() const { return apply([](double x) { return std::sqrt(x); }); }
Tensor Tensor::exp_t() const { return apply([](double x) { return std::exp(x); }); }
Tensor Tensor::log_t() const { return apply([](double x) { return std::log(x); }); }
Tensor Tensor::abs_t() const { return apply([](double x) { return std::abs(x); }); }
Tensor Tensor::sign() const { return apply([](double x) { return x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0); }); }
Tensor Tensor::pow_t(double p) const { return apply([p](double x) { return std::pow(x, p); }); }

Tensor Tensor::clamp(double lo, double hi) const {
    return apply([lo, hi](double x) { return std::max(lo, std::min(hi, x)); });
}

Tensor Tensor::add_row_broadcast(const Tensor& row_vec) const {
    TENSOR_CHECK(row_vec.rows == 1 && row_vec.cols == cols, "add_row_broadcast shape mismatch");
    Tensor r(rows, cols);
    TENSOR_OMP_PAR_FOR_COLLAPSE2
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            r(i, j) = (*this)(i, j) + row_vec(0, j);
    return r;
}

// --- Initialization ---
void Tensor::fill_zeros() { std::fill(data.begin(), data.end(), 0.0); }
void Tensor::fill_ones() { std::fill(data.begin(), data.end(), 1.0); }
void Tensor::fill_val(double v) { std::fill(data.begin(), data.end(), v); }

void Tensor::fill_random_normal(double mean, double stddev, std::mt19937& rng) {
    std::normal_distribution<double> dist(mean, stddev);
    for (auto& v : data) v = dist(rng);
}

void Tensor::fill_random_uniform(double lo, double hi, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(lo, hi);
    for (auto& v : data) v = dist(rng);
}

void Tensor::fill_xavier(size_t fan_in, size_t fan_out, std::mt19937& rng) {
    double stddev = std::sqrt(2.0 / (fan_in + fan_out));
    fill_random_normal(0.0, stddev, rng);
}

void Tensor::fill_he(size_t fan_in, std::mt19937& rng) {
    double stddev = std::sqrt(2.0 / fan_in);
    fill_random_normal(0.0, stddev, rng);
}

void Tensor::fill_identity() {
    fill_zeros();
    size_t n = std::min(rows, cols);
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < n; ++i) (*this)(i, i) = 1.0;
}

// --- I/O ---
void Tensor::save_csv(const std::string& filepath) const {
    std::ofstream f(filepath);
    TENSOR_CHECK(f.is_open(), "save_csv: cannot open " + filepath);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (j > 0) f << ",";
            f << std::setprecision(8) << (*this)(i, j);
        }
        f << "\n";
    }
}

Tensor Tensor::load_csv(const std::string& filepath) {
    std::ifstream f(filepath);
    TENSOR_CHECK(f.is_open(), "load_csv: cannot open " + filepath);
    std::vector<std::vector<double>> all_rows;
    std::string line;
    while (std::getline(f, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
        if (!row.empty()) all_rows.push_back(row);
    }
    if (all_rows.empty()) return Tensor();
    size_t r = all_rows.size(), c = all_rows[0].size();
    // Validate consistent column counts
    for (size_t i = 1; i < r; ++i)
        TENSOR_CHECK(all_rows[i].size() == c, "load_csv: ragged rows");
    Tensor t(r, c);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            t(i, j) = all_rows[i][j];
    return t;
}

void Tensor::save_binary(const std::string& filepath) const {
    std::ofstream f(filepath, std::ios::binary);
    TENSOR_CHECK(f.is_open(), "save_binary: cannot open " + filepath);
    f.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    f.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
}

Tensor Tensor::load_binary(const std::string& filepath) {
    std::ifstream f(filepath, std::ios::binary);
    TENSOR_CHECK(f.is_open(), "load_binary: cannot open " + filepath);
    size_t r, c;
    f.read(reinterpret_cast<char*>(&r), sizeof(r));
    f.read(reinterpret_cast<char*>(&c), sizeof(c));
    TENSOR_CHECK(f.good(), "load_binary: failed to read header");
    Tensor t(r, c);
    f.read(reinterpret_cast<char*>(t.data.data()), t.data.size() * sizeof(double));
    TENSOR_CHECK(f.good(), "load_binary: failed to read data");
    return t;
}

void Tensor::print(std::ostream& os, int precision) const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j)
            os << std::setw(precision + 6) << std::fixed << std::setprecision(precision) << (*this)(i, j);
        os << "\n";
    }
}

// --- Static constructors ---
Tensor Tensor::zeros(size_t r, size_t c) { return Tensor(r, c, 0.0); }
Tensor Tensor::ones(size_t r, size_t c) { return Tensor(r, c, 1.0); }
Tensor Tensor::eye(size_t n) {
    Tensor t(n, n);
    for (size_t i = 0; i < n; ++i) t(i, i) = 1.0;
    return t;
}
Tensor Tensor::randn(size_t r, size_t c, std::mt19937& rng) {
    Tensor t(r, c);
    t.fill_random_normal(0, 1, rng);
    return t;
}
Tensor Tensor::rand(size_t r, size_t c, std::mt19937& rng) {
    Tensor t(r, c);
    t.fill_random_uniform(0, 1, rng);
    return t;
}
Tensor Tensor::from_vector(const std::vector<double>& v) {
    return Tensor(v.size(), 1, v);
}
Tensor Tensor::hstack(const Tensor& a, const Tensor& b) {
    TENSOR_CHECK(a.rows == b.rows, "hstack: row count mismatch");
    Tensor r(a.rows, a.cols + b.cols);
    TENSOR_OMP_PAR_FOR
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) r(i, j) = a(i, j);
        for (size_t j = 0; j < b.cols; ++j) r(i, a.cols + j) = b(i, j);
    }
    return r;
}
Tensor Tensor::vstack(const Tensor& a, const Tensor& b) {
    TENSOR_CHECK(a.cols == b.cols, "vstack: column count mismatch");
    Tensor r(a.rows + b.rows, a.cols);
    std::copy(a.data.begin(), a.data.end(), r.data.begin());
    std::copy(b.data.begin(), b.data.end(), r.data.begin() + a.data.size());
    return r;
}

// --- Comparison ---
bool Tensor::same_shape(const Tensor& o) const { return rows == o.rows && cols == o.cols; }
double Tensor::norm() const {
    double s = 0;
    TENSOR_OMP_PAR_FOR_REDUCE_SUM
    for (auto v : data) s += v * v;
    return std::sqrt(s);
}
double Tensor::frobenius() const { return norm(); }
