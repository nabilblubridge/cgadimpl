// =====================
// file: include/ag/tensor.hpp (declarations only)
// =====================
#pragma once
#include <cstddef>
#include <utility>
#include <vector>


namespace ag {


class Tensor {
public:
// ctors
Tensor();
Tensor(int rows, int cols);


// factories
static Tensor zeros(int r, int c);
static Tensor ones (int r, int c);
static Tensor randn(int r, int c, unsigned seed=42);
static Tensor zeros_like(const Tensor& x);
static Tensor ones_like (const Tensor& x);


// shape/info
int rows() const;
int cols() const;
std::pair<int,int> shape() const;
std::size_t size() const;


// element access
float& operator()(int i, int j);
const float& operator()(int i, int j) const;


// grad accumulation utility
Tensor& add_(const Tensor& g);


// reductions
float sum_scalar() const;
static Tensor sum_all(const Tensor& X);


// pointwise / matrix ops
friend Tensor operator+(const Tensor& a, const Tensor& b);
friend Tensor operator-(const Tensor& a, const Tensor& b);
friend Tensor operator*(const Tensor& a, const Tensor& b); // Hadamard
friend Tensor operator-(const Tensor& x); // unary negation
friend Tensor operator*(const Tensor& a, float s); // scalar scale
friend Tensor operator*(float s, const Tensor& a); // scalar scale


static Tensor relu (const Tensor& x);
static Tensor relu_mask(const Tensor& x); // 1 where x>0 else 0
static Tensor transpose(const Tensor& x);
static Tensor matmul (const Tensor& A, const Tensor& B);


private:
int r{0}, c{0};
std::vector<float> d; // private storage
};


} // namespace ag