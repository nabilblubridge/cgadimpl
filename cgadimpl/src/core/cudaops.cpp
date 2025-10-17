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

std::shared_ptr<Node> add_cudaops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ 

     auto A = a->d_array;
         auto B = b->d_array;

         auto [M,K]  = a->value.shape();
         auto [K2,N] = b->value.shape();


         auto* fn =run_cuda_add;
         if (!fn) 
         {
         
        //  Tensor y = a->value + b->value; 
        // auto n = std::make_shared<Node>(y, a->requires_grad || b->requires_grad, Op::Add, "+"); 
        // n->inputs = {a, b}; 
        // ag::debug::on_node_created(n); 
                  throw std::runtime_error("No CPU Add kernel registered");

        // return n; 

         }
                  Tensor C({M,N});
                  auto n = std::make_shared<Node>(C,
             (a->requires_grad || b->requires_grad),
             Op::Add, "+", true);

         fn(A, B, n->d_array, M*K);
                          cudaMemcpy(n->value.data(), n->d_array, M * N * sizeof(float), cudaMemcpyDeviceToHost);

                        for (int i = 0; i < 10; ++i)
    std::cout << n->value.data()[i] << " ";
std::cout << std::endl;

         
         n->inputs = { a, b };
         return n;




    }


}
}