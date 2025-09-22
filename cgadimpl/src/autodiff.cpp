// =====================
// file: src/autodiff.cpp
// =====================
#include <unordered_map>
#include "ad/autodiff.hpp"


namespace ag {


void zero_grad(const Value& root){
auto order = topo_from(root.node.get());
for(Node* n : order) if(n->requires_grad) n->grad = Tensor::zeros_like(n->value);
}


void backward(const Value& root, const Tensor* grad_seed){
auto order = topo_from(root.node.get());
// seed root
if(root.node->requires_grad){
if(grad_seed) root.node->grad = *grad_seed;
else root.node->grad = (root.node->value.size()==1 ? Tensor::ones(1,1) : Tensor::ones_like(root.node->value));
}
// propagate
for(auto it = order.rbegin(); it!=order.rend(); ++it){
Node* n = *it; if(!n->requires_grad) continue; const Tensor gy = n->grad;
switch(n->op){
case Op::Add: {
Node* A=n->inputs[0].get(), *B=n->inputs[1].get(); if(A->requires_grad) A->grad.add_(gy); if(B->requires_grad) B->grad.add_(gy); break; }
case Op::Sub: {
Node* A=n->inputs[0].get(), *B=n->inputs[1].get(); if(A->requires_grad) A->grad.add_(gy); if(B->requires_grad) B->grad.add_(-gy); break; }
case Op::Mul: {
Node* A=n->inputs[0].get(), *B=n->inputs[1].get(); if(A->requires_grad) A->grad.add_( gy * B->value ); if(B->requires_grad) B->grad.add_( gy * A->value ); break; }
case Op::Relu: {
Node* X=n->inputs[0].get(); if(X->requires_grad){ Tensor g = gy * Tensor::relu_mask(n->value); X->grad.add_(g);} break; }
case Op::MatMul: {
Node* A=n->inputs[0].get(), *B=n->inputs[1].get(); if(A->requires_grad) A->grad.add_( Tensor::matmul(gy, Tensor::transpose(B->value)) ); if(B->requires_grad) B->grad.add_( Tensor::matmul( Tensor::transpose(A->value), gy ) ); break; }
case Op::Sum: {
Node* X=n->inputs[0].get(); if(X->requires_grad){ float s = gy(0,0); Tensor g = Tensor::ones_like(X->value) * s; X->grad.add_(g);} break; }
case Op::Leaf: default: break;
}
}
}


Tensor jvp(const Value& root, const std::unordered_map<Node*, Tensor>& seed){
auto order = topo_from(root.node.get());
std::unordered_map<Node*, Tensor> T; T.reserve(order.size());


auto zero_like = [](const Tensor& x){ return Tensor::zeros_like(x); };


for(Node* n : order){
Tensor t = seed.count(n) ? seed.at(n) : zero_like(n->value);
switch(n->op){
case Op::Leaf: break; // t already the seed
case Op::Add: { Node* A=n->inputs[0].get(), *B=n->inputs[1].get(); t = T[A] + T[B]; break; }
case Op::Sub: { Node* A=n->inputs[0].get(), *B=n->inputs[1].get(); t = T[A] + (-T[B]); break; }
case Op::Mul: { Node* A=n->inputs[0].get(), *B=n->inputs[1].get(); t = (T[A] * B->value) + (A->value * T[B]); break; }
case Op::Relu:{ Node* X=n->inputs[0].get(); t = T[X] * Tensor::relu_mask(n->value); break; }
case Op::MatMul:{ Node* A=n->inputs[0].get(), *B=n->inputs[1].get(); t = Tensor::matmul(T[A], B->value); t = t + Tensor::matmul(A->value, T[B]); break; }
case Op::Sum: { Node* X=n->inputs[0].get(); float s = T[X].sum_scalar(); t = Tensor(1,1); t(0,0)=s; break; }
}
T[n] = t;
}
return T[root.node.get()];
}


} // namespace ag