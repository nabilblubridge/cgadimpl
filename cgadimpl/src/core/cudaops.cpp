// =====================
// file: src/cudaops.cpp
// =====================
#include "ad/cudaops.hpp"
#include "ad/cudarray.hpp"
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>


namespace ag {
namespace detail {
// =====================================================
// ADD
// =====================================================
std::shared_ptr<Node> add_cudaops(const std::shared_ptr<Node>& a,
                                  const std::shared_ptr<Node>& b)
{
    auto A = a->d_array;
    auto B = b->d_array;

    auto [M, K]  = a->value.shape();
    auto [K2, N] = b->value.shape();

    auto* fn = run_cuda_add;
    if (!fn)
        throw std::runtime_error("No CUDA Add kernel registered");

    Tensor C({M, N});
    auto n = std::make_shared<Node>(C, (a->requires_grad || b->requires_grad),
                                    Op::Add, "+", true);

    fn(A, B, n->d_array, M * K);
    n->siz = M*K;

    // cudaMemcpy(n->value.data(), n->d_array, M * N * sizeof(float),
    //            cudaMemcpyDeviceToHost);

    // // Debug print
    // std::cout << "[CUDA ADD output preview]: ";
    // for (int i = 0; i < std::min(10, M * N); ++i)
    //     std::cout << n->value.data()[i] << " ";
    // std::cout << "\n";

    n->inputs = {a, b};
    return n;
}


// =====================================================
// SUB
// =====================================================
std::shared_ptr<Node> sub_cudaops(const std::shared_ptr<Node>& a,
                                  const std::shared_ptr<Node>& b)
{
    auto A = a->d_array;
    auto B = b->d_array;

    auto [M, K]  = a->value.shape();
    auto [K2, N] = b->value.shape();

    auto* fn = run_cuda_sub;
    if (!fn)
        throw std::runtime_error("No CUDA Sub kernel registered");

    Tensor C({M, N});
    auto n = std::make_shared<Node>(C, (a->requires_grad || b->requires_grad),
                                    Op::Sub, "-", true);

    fn(A, B, n->d_array, M * K);

    cudaMemcpy(n->value.data(), n->d_array, M * N * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << "[CUDA SUB output preview]: ";
    for (int i = 0; i < std::min(10, M * N); ++i)
        std::cout << n->value.data()[i] << " ";
    std::cout << "\n";

    n->inputs = {a, b};
    return n;
}


// =====================================================
// MUL (Hadamard)
// =====================================================
std::shared_ptr<Node> mul_cudaops(const std::shared_ptr<Node>& a,
                                     const std::shared_ptr<Node>& b)
{
    auto A = a->d_array;
    auto B = b->d_array;

    auto [M, K]  = a->value.shape();
    auto [K2, N] = b->value.shape();

    auto* fn = run_cuda_hadmul;
    if (!fn)
        throw std::runtime_error("No CUDA Hadamard kernel registered");

    Tensor C({M, N});
    auto n = std::make_shared<Node>(C, (a->requires_grad || b->requires_grad),
                                    Op::Mul, "⨀", true);

    fn(A, B, n->d_array, M * K);

    cudaMemcpy(n->value.data(), n->d_array, M * N * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << "[CUDA MUL output preview]: ";
    for (int i = 0; i < std::min(10, M * N); ++i)
        std::cout << n->value.data()[i] << " ";
    std::cout << "\n";

    n->inputs = {a, b};
    return n;
}


// =====================================================
// DIV
// =====================================================
std::shared_ptr<Node> div_cudaops(const std::shared_ptr<Node>& a,
                                  const std::shared_ptr<Node>& b)
{
    auto A = a->d_array;
    auto B = b->d_array;

    auto [M, K]  = a->value.shape();
    auto [K2, N] = b->value.shape();

    auto* fn = run_cuda_div;
    if (!fn)
        throw std::runtime_error("No CUDA Div kernel registered");

    Tensor C({M, N});
    auto n = std::make_shared<Node>(C, (a->requires_grad || b->requires_grad),
                                    Op::Div, "/", true);

    fn(A, B, n->d_array, M * K);

    cudaMemcpy(n->value.data(), n->d_array, M * N * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << "[CUDA DIV output preview]: ";
    for (int i = 0; i < std::min(10, M * N); ++i)
        std::cout << n->value.data()[i] << " ";
    std::cout << "\n";

    n->inputs = {a, b};
    return n;
}






   
    std::shared_ptr<Node> sigmoid_cudaops(const std::shared_ptr<Node>& x){ 
                      const Tensor& xin = x->value;
        auto X = x->d_array;

    auto [M, K]  = x->value.shape();

    auto* fn = run_cuda_sigmoid;
    if (!fn)
        throw std::runtime_error("No CUDA Div kernel registered");

    Tensor C({M, K});
    auto n = std::make_shared<Node>(C, (x->requires_grad),
                                    Op::Sigmoid, "/", true);

    fn(X, n->d_array, M * K);

    cudaMemcpy(n->value.data(), n->d_array, M * K * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << "[CUDA Sigmoid output preview]: ";
    for (int i = 0; i < std::min(10, M * K); ++i)
        std::cout << n->value.data()[i] << " ";
    std::cout << "\n";

    n->inputs = {x};
    return n;
    }





















        std::shared_ptr<Node> sigmoidiff_cudaops(const std::shared_ptr<Node>& x){ 
                      const Tensor& xin = x->value;
        auto X = x->d_array;

    auto [M, K]  = x->value.shape();

    auto* fn = run_cuda_sigmoidiff;
    if (!fn)
        throw std::runtime_error("No CUDA Sigmoidiff kernel registered");

    Tensor C({M, K});
    auto n = std::make_shared<Node>(C, (x->requires_grad),
                                    Op::Sigmoidiff, "/", true);

    fn(X, n->d_array, M * K);

    cudaMemcpy(n->value.data(), n->d_array, M * K * sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << "[CUDA Sigmoidiff output preview]: ";
    for (int i = 0; i < std::min(10, M * K); ++i)
        std::cout << n->value.data()[i] << " ";
    std::cout << "\n";

    n->inputs = {x};
    return n;
    }






}






}