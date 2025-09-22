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
default: return -1;
}
}


} // namespace ag

