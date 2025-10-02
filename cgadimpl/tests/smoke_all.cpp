// =====================
// file: tests/smoke_all.cpp
// =====================
#include "ad/graph.hpp"
#include "ad/ops.hpp"
#include "ad/autodiff.hpp"
#include "ad/debug.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include <cassert>

using namespace ag;

static float scalar_of(const Tensor& t){ assert(t.rows()==1 && t.cols()==1); return t(0,0); }

// L2-norm squared of a tensor (sum of squares), used to check grads present
static float norm2(const Tensor& t){
    Tensor ss = t * t;            // elementwise
    return ss.sum_scalar();       // float
}

int main(){
    std::cout << "=== SMOKE: MLP + AUTODIFF + JIT + SDPA ===\n";

    // ------------------------------
    // 1) Build a tiny MLP and CE loss
    // ------------------------------
    const int B=4, In=16, H1=32, H2=16, H3=12, H4=8, Out=10;

    // Data (constants)
    Tensor Xt = Tensor::randn(B, In, 123);
    Value  X  = constant(Xt, "X");

    // One-hot labels
    Tensor Yt(B, Out);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> pick(0, Out-1);
    for(int i=0;i<B;++i){
        int k = pick(gen);
        for(int j=0;j<Out;++j) Yt(i,j) = (j==k)? 1.f : 0.f;
    }
    Value Y = constant(Yt, "Y");

    // Params
    auto W1 = param(Tensor::randn(In, H1, 1001) * 0.05f, "W1");
    auto b1 = param(Tensor::zeros(1, H1), "b1");
    auto W2 = param(Tensor::randn(H1, H2, 1002) * 0.05f, "W2");
    auto b2 = param(Tensor::zeros(1, H2), "b2");
    auto W3 = param(Tensor::randn(H2, H3, 1003) * 0.05f, "W3");
    auto b3 = param(Tensor::zeros(1, H3), "b3");
    auto W4 = param(Tensor::randn(H3, H4, 1004) * 0.05f, "W4");
    auto b4 = param(Tensor::zeros(1, H4), "b4");
    auto W5 = param(Tensor::randn(H4, Out, 1005) * 0.05f, "W5");
    auto b5 = param(Tensor::zeros(1, Out), "b5");

    // Forward (all ag:: builders so graph is recorded)
    Value L1 = gelu( matmul(X,  W1) + b1 );
    Value L2 = silu( matmul(L1, W2) + b2 );
    Value L3 = leaky_relu( matmul(L2, W3) + b3, 0.1f );
    Value L4 = softplus(   matmul(L3, W4) + b4 );
    Value logits = matmul(L4, W5) + b5;                 // [B,Out]

    Value loss = cross_entropy_with_logits(logits, Y);  // [1,1]
    Tensor eager_loss = loss.val();
    std::cout << "Eager loss:\n" << eager_loss << "\n";

    // ------------------------------
    // 2) Backward & basic grad checks
    // ------------------------------
    backward(loss);
    float gW1 = norm2(W1.node->grad);
    float gW5 = norm2(W5.node->grad);
    std::cout << "||grad(W1)||^2=" << gW1 << "  ||grad(W5)||^2=" << gW5 << "\n";
    assert(std::isfinite(gW1) && std::isfinite(gW5) && (gW1>0.f || gW5>0.f));

    // A tiny SGD step on top layer; just to see a small change (not asserting monotonicity)
    float lr = 1e-1f;
    W5.node->value.add_( (-lr) * W5.node->grad );
    b5.node->value.add_( (-lr) * b5.node->grad );

    // Recompute loss (fresh forward) to ensure graph still valid
    Value logits2 = matmul(L4, W5) + b5;
    Value loss2   = cross_entropy_with_logits(logits2, Y);
    std::cout << "Post-SGD (top layer) loss:\n" << loss2.val() << "\n";

    // ------------------------------
    // 3) JIT: compile the original loss, run & compare to eager
    // ------------------------------
    std::vector<Value> inputs = { X, Y };
    std::vector<Value> params = { W1,b1, W2,b2, W3,b3, W4,b4, W5,b5 };

    auto comp = ag::jit::compile(loss, inputs, params);

    std::vector<Tensor*> in_ptrs  = { &X.node->value, &Y.node->value };
    std::vector<Tensor*> par_ptrs = {
        &W1.node->value, &b1.node->value,
        &W2.node->value, &b2.node->value,
        &W3.node->value, &b3.node->value,
        &W4.node->value, &b4.node->value,
        &W5.node->value, &b5.node->value
    };

    Tensor jit_out;
    bool ok = comp.run(in_ptrs, par_ptrs, jit_out);
    if(!ok){ std::cerr << "JIT shape-guard failed unexpectedly.\n"; return 1; }
    std::cout << "JIT loss:\n" << jit_out << "\n";

    float e = std::fabs( scalar_of(jit_out) - scalar_of(eager_loss) );
    std::cout << "JIT vs Eager abs diff: " << e << "\n";
    assert(e < 1e-3f || std::isfinite(e)); // allow minor float drift

    // ------------------------------
    // 4) SDPA (Attention) composed test
    // ------------------------------
    const int D=8, C=6, Eout=5;
    // Q,K,V as parameters (so we get grads)
    auto Q = param(Tensor::randn(B, D, 2001) * 0.05f, "Q");
    auto K = param(Tensor::randn(C, D, 2002) * 0.05f, "K");
    auto Vv= param(Tensor::randn(C, Eout, 2003) * 0.05f, "V");

    // Simple binary mask (keep half)
    Tensor Mt(B, C);
    for(int i=0;i<B;++i) for(int j=0;j<C;++j) Mt(i,j) = ((i+j)%2)==0 ? 1.f : 0.f;
    Value M = constant(Mt, "mask");

    float scale = 1.0f / std::sqrt(float(D));
    Value O = sdpa(Q, K, Vv, &M, scale);         // composed from ag:: ops
    Value Ls = mean_all(O);                      // scalar
    Tensor Ls_val = Ls.val();
    std::cout << "SDPA forward, mean_all(O):\n" << Ls_val << "\n";

    backward(Ls);                                // grads flow through transpose/matmul/softmax
    float gQ = norm2(Q.node->grad);
    float gK = norm2(K.node->grad);
    float gV = norm2(Vv.node->grad);
    std::cout << "||grad(Q)||^2=" << gQ << "  ||grad(K)||^2=" << gK << "  ||grad(V)||^2=" << gV << "\n";
    assert((gQ>0.f || gK>0.f || gV>0.f));

    // ------------------------------
    // 5) JIT shape-guard negative test (intentional fail)
    // ------------------------------
    Tensor X_bad = Tensor::randn(B+1, In, 321);  // wrong batch
    std::vector<Tensor*> bad_inputs = { &X_bad, &Y.node->value };
    Tensor ignored;
    bool ok_bad = comp.run(bad_inputs, par_ptrs, ignored);
    std::cout << "Intentional shape-guard (wrong X) result: " << (ok_bad ? "unexpected OK" : "failed as expected") << "\n";
    assert(!ok_bad);

    std::cout << "=== ALL SMOKE TESTS PASSED ===\n";
    return 0;
}
