// =====================
// file: src/schema.cpp
// =====================
#include "ad/schema.hpp"

namespace ag {

const char* op_name(Op op){
switch(op){
case Op::Leaf: return "leaf"; 
case Op::Add: return "+"; 
case Op::Sub: return "-"; 
case Op::Mul: return "*";
case Op::Relu: return "relu"; 
case Op::MatMul: return "matmul"; 
case Op::Sum: return "sum";
case Op::Exp: return "exp"; 
case Op::Log: return "log";
case Op::Tanh: return "tanh"; 
case Op::Sigmoid: return "sigmoid";
case Op::Softplus: return "softplus"; 
case Op::SiLU: return "silu";
case Op::GELU: return "gelu"; 
case Op::LeakyRelu: return "leakyrelu";
case Op::RowSum: return "rowsum"; 
case Op::RowMax: return "rowmax";
case Op::MeanAll: return "meanall"; 
case Op::SoftmaxRow: return "softmax_row";
case Op::LogSumExpRow: return "logsumexp_row"; 
case Op::CeWithLogits: return "ce_with_logits";
default: return "?";
}
}


int op_arity(Op op){
switch(op){
case Op::Leaf: return 0; 
case Op::Relu: return 1; 
case Op::Sum: return 1;
case Op::Add: return 2; 
case Op::Sub: return 2; 
case Op::Mul: return 2; 
case Op::MatMul: return 2;
case Op::Exp: case Op::Log: case Op::Tanh: case Op::Sigmoid: case Op::Softplus: case Op::SiLU: case Op::GELU: return 1;
case Op::LeakyRelu: return 2; // (x, alpha[1x1])
// rowwise reductions
case Op::RowSum: case Op::RowMax: return 1;
// misc
case Op::MeanAll: return 1;
case Op::SoftmaxRow: case Op::LogSumExpRow: return 1;
case Op::CeWithLogits: return 2; // (logits, targets)
default: return -1;
}
}

} // namespace ag