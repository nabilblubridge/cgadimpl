// =====================
// file: include/ag/nodeops.hpp (declarations only)
// =====================
#pragma once
#include "ad/ops.hpp"
#include "ad/graph.hpp"
#include "ad/debug.hpp"
#include "ad/kernels_api.hpp"
#include "ad/graph.hpp"
#include "ad/checkpoint.hpp"


namespace ag {

struct CheckpointOptions;

//std::shared_ptr<Node> checkpoint(const std::shared_ptr<Node> &v, const CheckpointOptions &opts);

const std::shared_ptr<Node>& add (const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> sub (const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> mul (const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> relu (const std::shared_ptr<Node>& x);
std::shared_ptr<Node> matmul(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b);
std::shared_ptr<Node> sum (const std::shared_ptr<Node>& x);
std::shared_ptr<Node> flomul (const std::shared_ptr<Node>& a, float b);

inline std::shared_ptr<Node> operator+(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ return add(a,b);}
inline std::shared_ptr<Node> operator-(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ return sub(a,b);}
inline std::shared_ptr<Node> operator*(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b){ return mul(a,b);}
inline std::shared_ptr<Node> operator*(const std::shared_ptr<Node>& a, float& b){ return flomul(a,b);}

// unary elementwise
std::shared_ptr<Node> exp (const std::shared_ptr<Node>& x);
std::shared_ptr<Node> log (const std::shared_ptr<Node>& x);
std::shared_ptr<Node> tanh (const std::shared_ptr<Node>& x);
std::shared_ptr<Node> gcu (const std::shared_ptr<Node>& x);
std::shared_ptr<Node> mish (const std::shared_ptr<Node>& x);
std::shared_ptr<Node> gaus (const std::shared_ptr<Node>& x);
std::shared_ptr<Node> parcon(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> sigmoid(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> softplus(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> gelu (const std::shared_ptr<Node>& x); // tanh approx
std::shared_ptr<Node> silu (const std::shared_ptr<Node>& x); // x * sigmoid(x)
std::shared_ptr<Node> leaky_relu(const std::shared_ptr<Node>& x, float alpha=0.01f); // alpha via const input
std::shared_ptr<Node> lisht(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> transpose(const std::shared_ptr<Node>& x);
std::shared_ptr<Node> swiglu(const std::shared_ptr<Node>& x, const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d);
std::shared_ptr<Node> deconval(const std::shared_ptr<Node>& x, float& g); // alpha via const input
std::shared_ptr<Node> rms(const std::shared_ptr<Node>& x); // root mean square normalization
std::shared_ptr<Node> realrms(const std::shared_ptr<Node>& x, float& g); // with learned scale
std::shared_ptr<Node> dyntanh(const std::shared_ptr<Node>& x, float& a, float& b, float& g); // dynamic tanh via mean_all
std::shared_ptr<Node> relaynor(const std::shared_ptr<Node>& x, float& b, float& g); // with learned scale and bias
std::shared_ptr<Node> mambassm(const std::shared_ptr<Node>& z, const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d); // state space model


// rowwise reductions / softmax family
std::shared_ptr<Node> rowsum (const std::shared_ptr<Node>& x); // [B,C] -> [B,1]
std::shared_ptr<Node> rowmax (const std::shared_ptr<Node>& x); // [B,C] -> [B,1]
std::shared_ptr<Node> mean_all(const std::shared_ptr<Node>& x); // scalar
std::shared_ptr<Node> softmax_row(const std::shared_ptr<Node>& z); // [B,C] -> [B,C]
std::shared_ptr<Node> logsumexp_row(const std::shared_ptr<Node>& z); // [B,C] -> [B,1]
std::shared_ptr<Node> laynor(const std::shared_ptr<Node>& x);
const std::shared_ptr<Node>& alibiatt(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d, float& m); // m = max seq len

// composite loss (one-hot targets)
std::shared_ptr<Node> cross_entropy_with_logits(std::shared_ptr<Node> logits, std::shared_ptr<Node> onehot);
std::shared_ptr<Node> kldivergence(std::shared_ptr<Node> logits, std::shared_ptr<Node> onehot);
std::shared_ptr<Node> fmab(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c); // fused multiply-add a@b + c
std::shared_ptr<Node> attention(const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b, const std::shared_ptr<Node>& c, const std::shared_ptr<Node>& d);
std::shared_ptr<Node> mse_loss(const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target);
std::shared_ptr<Node> mae_loss(const std::shared_ptr<Node>& pred, const std::shared_ptr<Node>& target);


// Tensor forward_eval_node(Node* node);


} // namespace ag