// =====================
// file: include/ag/schema.hpp (declarations only)
// =====================

#pragma once


namespace ag {


// enum class Op : unsigned char { Leaf=0, Add, Sub, Mul, Relu, MatMul, Sum };
enum class Op : unsigned char {
Leaf=0, Add, Sub, Mul, Relu, MatMul, Sum,
// unary elementwise
Exp, Log, Tanh, Sigmoid, Softplus, SiLU, GELU, LeakyRelu,
// rowwise reductions
RowSum, RowMax,
// misc reductions
MeanAll,
// softmax family
SoftmaxRow, LogSumExpRow,
// composite losses
CeWithLogits
};


const char* op_name(Op op);
int op_arity(Op op);


} // namespace ag

