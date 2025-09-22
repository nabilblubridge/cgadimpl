// =====================
// file: include/ag/schema.hpp (declarations only)
// =====================

#pragma once


namespace ag {


enum class Op : unsigned char { Leaf=0, Add, Sub, Mul, Relu, MatMul, Sum };


const char* op_name(Op op);
int op_arity(Op op);


} // namespace ag

