// =====================
// file: src/ops.cpp
// =====================
#include "ad/ops.hpp"


namespace ag {


Value add(const Value& a, const Value& b){ Tensor y = a.val() + b.val(); auto n = std::make_shared<Node>(y, a.node->requires_grad || b.node->requires_grad, Op::Add, "+"); n->inputs = {a.node, b.node}; return Value(n); }
Value sub(const Value& a, const Value& b){ Tensor y = a.val() - b.val(); auto n = std::make_shared<Node>(y, a.node->requires_grad || b.node->requires_grad, Op::Sub, "-"); n->inputs = {a.node, b.node}; return Value(n); }
Value mul(const Value& a, const Value& b){ Tensor y = a.val() * b.val(); auto n = std::make_shared<Node>(y, a.node->requires_grad || b.node->requires_grad, Op::Mul, "*"); n->inputs = {a.node, b.node}; return Value(n); }
Value relu(const Value& x){ Tensor y = Tensor::relu(x.val()); auto n = std::make_shared<Node>(y, x.node->requires_grad, Op::Relu, "relu"); n->inputs = {x.node}; return Value(n); }
Value matmul(const Value& a, const Value& b){ Tensor y = Tensor::matmul(a.val(), b.val()); auto n = std::make_shared<Node>(y, a.node->requires_grad || b.node->requires_grad, Op::MatMul, "matmul"); n->inputs = {a.node, b.node}; return Value(n); }
Value sum(const Value& x){ Tensor y = Tensor::sum_all(x.val()); auto n = std::make_shared<Node>(y, x.node->requires_grad, Op::Sum, "sum"); n->inputs = {x.node}; return Value(n); }


} // namespace ag

