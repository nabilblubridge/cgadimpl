// =====================
// file: include/ag/detail/cudaops.hpp
// =====================
#pragma once

#include "ad/graph.hpp"
#include "ad/checkpoint.hpp"
#include "ad/kernels_api.hpp"
#include "ad/debug.hpp"
#include <iostream>
#include <math.h>
#include <iterator>
#include <memory>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>


namespace ag {
namespace detail {

std::shared_ptr<Node> add_cudaops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> sub_cudaops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> mul_cudaops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> div_cudaops(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);


}
}