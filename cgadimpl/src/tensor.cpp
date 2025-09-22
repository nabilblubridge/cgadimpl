// =====================
// file: src/tensor.cpp (implementations)
// =====================
#include <random>
#include <algorithm>
#include <stdexcept>
#include "ad/tensor.hpp"


namespace ag {


Tensor::Tensor() = default;
Tensor::Tensor(int rows, int cols) : r(rows), c(cols), d(static_cast<std::size_t>(rows)*cols, 0.f) {}


Tensor Tensor::zeros(int r, int c){ return Tensor(r,c); }
Tensor Tensor::ones (int r, int c){ Tensor t(r,c); std::fill(t.d.begin(), t.d.end(), 1.f); return t; }
Tensor Tensor::randn(int r, int c, unsigned seed){ Tensor t(r,c); std::mt19937 gen(seed); std::normal_distribution<float> N(0.f,1.f); for(auto &x: t.d) x = N(gen); return t; }
Tensor Tensor::zeros_like(const Tensor& x){ return zeros(x.r, x.c); }
Tensor Tensor::ones_like (const Tensor& x){ return ones (x.r, x.c); }


int Tensor::rows() const { return r; }
int Tensor::cols() const { return c; }
std::pair<int,int> Tensor::shape() const { return {r,c}; }
std::size_t Tensor::size() const { return d.size(); }


float& Tensor::operator()(int i, int j){ return d[static_cast<std::size_t>(i)*c + j]; }
const float& Tensor::operator()(int i, int j) const { return d[static_cast<std::size_t>(i)*c + j]; }


Tensor& Tensor::add_(const Tensor& g){
if(r!=g.r || c!=g.c) throw std::runtime_error("add_: shape mismatch");
for(std::size_t i=0;i<d.size();++i) d[i]+=g.d[i];
return *this;
}


float Tensor::sum_scalar() const { float s=0.f; for(float x: d) s+=x; return s; }
Tensor Tensor::sum_all(const Tensor& X){ Tensor y(1,1); y(0,0) = X.sum_scalar(); return y; }


Tensor operator+(const Tensor& a, const Tensor& b){ if(a.r!=b.r || a.c!=b.c) throw std::runtime_error("+: shape mismatch"); Tensor y(a.r,a.c); for(std::size_t i=0;i<a.d.size();++i) y.d[i]=a.d[i]+b.d[i]; return y; }
Tensor operator-(const Tensor& a, const Tensor& b){ if(a.r!=b.r || a.c!=b.c) throw std::runtime_error("-: shape mismatch"); Tensor y(a.r,a.c); for(std::size_t i=0;i<a.d.size();++i) y.d[i]=a.d[i]-b.d[i]; return y; }
Tensor operator*(const Tensor& a, const Tensor& b){ if(a.r!=b.r || a.c!=b.c) throw std::runtime_error("*: shape mismatch"); Tensor y(a.r,a.c); for(std::size_t i=0;i<a.d.size();++i) y.d[i]=a.d[i]*b.d[i]; return y; }
Tensor operator-(const Tensor& x){ Tensor y(x.r,x.c); for(std::size_t i=0;i<x.d.size();++i) y.d[i] = -x.d[i]; return y; }
Tensor operator*(const Tensor& a, float s){ Tensor y(a.r,a.c); for(std::size_t i=0;i<a.d.size();++i) y.d[i]=a.d[i]*s; return y; }
Tensor operator*(float s, const Tensor& a){ return a*s; }


Tensor Tensor::relu(const Tensor& x){ Tensor y(x.r,x.c); for(std::size_t i=0;i<x.d.size();++i) y.d[i] = x.d[i] > 0.f ? x.d[i] : 0.f; return y; }
Tensor Tensor::relu_mask(const Tensor& x){ Tensor m(x.r,x.c); for(std::size_t i=0;i<x.d.size();++i) m.d[i] = x.d[i] > 0.f ? 1.f : 0.f; return m; }



Tensor Tensor::transpose(const Tensor& x){ Tensor y(x.c, x.r); for(int i=0;i<x.r;++i) for(int j=0;j<x.c;++j) y(j,i)=x(i,j); return y; }


Tensor Tensor::matmul(const Tensor& A, const Tensor& B){ if(A.c!=B.r) throw std::runtime_error("matmul: inner dim mismatch"); Tensor Y(A.r, B.c);
for(int i=0;i<A.r;++i){ for(int k=0;k<A.c;++k){ float aik=A(i,k); for(int j=0;j<B.c;++j){ Y(i,j) += aik * B(k,j); } } }
return Y; }


} // namespace ag