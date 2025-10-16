#include "ad/detail/autodiff_ops.hpp"

#include <cmath>
#include "ad/nodeops.hpp"

namespace ag {
namespace detail{

// helper: reduce a gradient to a parent's shape (broadcast-aware)
inline Tensor rt(const Tensor& g, const Tensor& like){ return Tensor::reduce_to(g, like); }

// ----- elementwise binary -----
void hvjp_Add(Node* n, const std::shared_ptr<Node>& gy){
    auto A = n->inputs[0]; auto B = n->inputs[1];


    auto H = gy*1.0 ;
    if (A->requires_grad) A->grad.add_( rt(H->value, A->value) );
    if (B->requires_grad) B->grad.add_( rt(H->value, B->value) );
}
void hvjp_Sub(Node* n, const std::shared_ptr<Node>& gy){
    auto A = n->inputs[0]; auto B = n->inputs[1];
      auto H = gy*1.0 ;
    if (A->requires_grad) A->grad.add_( rt(H->value, A->value) );
    if (B->requires_grad) B->grad.add_( rt(-(H->value), B->value) );
}
void hvjp_Mul(Node* n, const std::shared_ptr<Node>& gy){
    auto A = n->inputs[0]; auto B = n->inputs[1];


    if (A->requires_grad) A->grad.add_( rt((gy*B)->value, A->value) );
    if (B->requires_grad) B->grad.add_( rt((gy*A)->value, B->value) );
 }

void hvjp_MSELoss(Node* n, const std::shared_ptr<Node>& gy /*unused: scalar gy*/){
    auto Z = n->inputs[0];
    auto Y = n->inputs[1];
    int B = Z->value.rows(), C = Z->value.cols();
     auto diff = Z - Y;
     auto gZ = diff * (2.0f / float(B * C));
     auto gY = -1.0*diff * (2.0f / float(B * C));
    if (Z->requires_grad) Z->grad.add_(gZ->value);
    if (Y->requires_grad) Y->grad.add_(gY->value);
}



// ----- elementwise trinary -----
void hvjp_FMA(Node* n, const std::shared_ptr<Node>& gy){
    auto A = n->inputs[0];
    auto B = n->inputs[1];
    auto C = n->inputs[2];

    // External kernel (if plugin loaded), else fallback to  matmul_nodeops
    auto* mm = ag::kernels::cpu().matmul;

    // Shapes
    const std::shared_ptr<Node>& At = A;
    const std::shared_ptr<Node>& Bt = B;
    auto [M, K]  = At->value.shape();
    auto [K2, N] = Bt->value.shape();
    (void)K2; // assume forward already checked

    if (A->requires_grad){
        auto BT = transpose_nodeops(n->inputs[1]); // (N x K)
         Tensor dA(M, K);                   // temp buffer

        if (mm) {
            // dA = gy (MxN) * BT (NxK)
            mm(gy->value.data(), BT->value.data(), dA.data(), M, N, K);
                    auto c = std::make_shared<Node>(dA, false, Op::MatMul, "matmul");

            A->grad.add_(c->value);
        } else {
             dA = Tensor::matmul(gy->value, BT->value);

                auto c = std::make_shared<Node>(dA, false, Op::MatMul, "matmul");

            A->grad.add_(c->value);
        }
        
    }

    if (B->requires_grad){
        auto AT = transpose_nodeops(n->inputs[0]); // (K x M)
         Tensor dB(K, N);                   // temp buffer

        if (mm) {
            // dB = AT (KxM) * gy (MxN)
            mm(AT->value.data(), gy->value.data(), dB.data(), K, M, N);
                              auto c = std::make_shared<Node>(dB, false, Op::MatMul, "matmul");

            B->grad.add_(c->value);
        } else {
             dB = Tensor::matmul(AT->value, gy->value);

                auto c = std::make_shared<Node>(dB, false, Op::MatMul, "matmul");

            B->grad.add_(c->value);
        }
       
    }
    if (C->requires_grad) C->grad.add_( rt(gy->value, C->value) );
}


void hvjp_Sqrt(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0];
    if (X->requires_grad) X->grad.add_( rt((0.5f * gy * sqrt_nodeops(  reci_nodeops(X)))->value, X->value) );
}

void hvjp_LayerNorm(Node* n, const std::shared_ptr<Node>& gy){

    auto x = n->inputs[0];
     int N = x->value.cols(); // normalize over last dim (row-wise)

   //  debug::print_ auto("gy",gy);
     
    
    // stddev [2x1] - float
     auto std =  sqrt_nodeops((n->tapenode[0]) +0.01);
  //  debug::print_ auto("std",std);

    // x - mean [2x1]
     auto xmu = x - (n->tapenode[1]);
 //   debug::print_ auto("xmu",xmu);

    // sum of grad_out along row
     auto grad_sum =  rowsum_nodeops(gy);
  //  debug::print_ auto("grad_sum",grad_sum);

    // dot(grad_out, x - mean) along row
     auto grad_dot_xmu =  rowsum_nodeops(gy * xmu);
   // debug::print_ auto("grad_dot_xmu",grad_dot_xmu);

    // term: N * grad_out
     auto term1 = gy * float(N);
 //   debug::print_ auto("term1",term1);

    // term: subtract sum of grad_out
     auto term2 = term1 - grad_sum;
   // debug::print_ auto("term2",term2);

    // term: subtract (x - mean) * (grad_dot_xmu / (var + eps))
     auto term3 = term2 - (xmu * (grad_dot_xmu / ((n->tapenode[0]) + 0.01)));
  //  debug::print_ auto("term3",term3);

    // scale: divide by (N * std)
     auto dx = term3 / (std * float(N));
  //  debug::print_ auto("dx",dx);

    if (x->requires_grad) x->grad.add_( dx->value );

}


void hvjp_RealLayerNorm(Node* n, const std::shared_ptr<Node>& gy){

    auto x = n->inputs[0];
    auto b = n->inputs[1];
    auto g = n->inputs[2];
     int N = x->value.cols(); // normalize over last dim (row-wise)

   //  debug::print_ auto("gy",gy);
     
    
    // stddev [2x1] - float
     auto std =  sqrt_nodeops((n->tapenode[0]) +0.01);
  //  debug::print_ auto("std",std);

    // x - mean [2x1]
     auto xmu = x - (n->tapenode[1]);
 //   debug::print_ auto("xmu",xmu);

    // sum of grad_out along row
     auto grad_sum =  rowsum_nodeops(gy);
  //  debug::print_ auto("grad_sum",grad_sum);

    // dot(grad_out, x - mean) along row
     auto grad_dot_xmu =  rowsum_nodeops(gy * xmu);
   // debug::print_ auto("grad_dot_xmu",grad_dot_xmu);


    // term: N * grad_out
     auto term1 = gy * float(N);
 //   debug::print_ auto("term1",term1);

    // term: subtract sum of grad_out
     auto term2 = term1 - grad_sum;
   // debug::print_ auto("term2",term2);

    // term: subtract (x - mean) * (grad_dot_xmu / (var + eps))
     auto term3 = term2 - (xmu * (grad_dot_xmu / ((n->tapenode[0]) + 0.01)));
  //  debug::print_ auto("term3",term3);

    // scale: divide by (N * std)
     auto dx = term3 / (std * float(N));
 //debug::print_ auto("dx",term3);
// debug::print_ auto("g",g->value);

    if (x->requires_grad) x->grad.add_( dx->value);
if (b->requires_grad) b->grad.add_(  rowsum_nodeops(gy)->value );   // db = sum over batch
if (g->requires_grad) g->grad.add_(  rowsum_nodeops(gy * ((n->tapenode[2])))->value );

}


void hvjp_RMSNorm(Node* n, const std::shared_ptr<Node>& gy){

    auto x = n->inputs[0];
    auto rms = n->tapenode[0];
    auto y   = n->tapenode[1];   // normalized x

    // upstream dot
    auto dot = rowsum_nodeops(gy*y);  // [batch x 1]

    auto grad_x = (gy/rms) - (y * dot / (rms*1.2f));

    if (x->requires_grad) x->grad.add_(grad_x->value);


}


void hvjp_Div(Node* n, const std::shared_ptr<Node>& gy){
    auto A = n->inputs[0]; auto B = n->inputs[1];
    if (A->requires_grad) A->grad.add_( rt( (gy * (reci_nodeops(B)))->value, A->value) );
    if (B->requires_grad) B->grad.add_( rt( (-1.0*gy * (reci_nodeops(B)) * (reci_nodeops(B)) * A)->value, B->value) );
}
void hvjp_Reciprocal(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0]; if (!X->requires_grad) return;
    X->grad.add_( rt( (-1.0*gy * (reci_nodeops(X)) * (reci_nodeops(X)))->value, X->value) );
}


void hvjp_Sign(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0]; if (!X->requires_grad) return;

    auto zeros_node = std::make_shared<Node>(
        Tensor::zeros_like(X->value),
        X->requires_grad, Op::Leaf, "ones_like"
    );
    X->grad.add_( rt( (gy *zeros_node)->value, X->value) );
}


void hvjp_Relumask(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0]; if (!X->requires_grad) return;

    auto zeros_node = std::make_shared<Node>(
        Tensor::zeros_like(X->value),
        X->requires_grad, Op::Leaf, "ones_like"
    );
    X->grad.add_( rt( (gy *zeros_node)->value, X->value) );
}


void hvjp_RealRMSNorm(Node* n, const std::shared_ptr<Node>& gy){

    auto x = n->inputs[0];
    auto g = n->inputs[1];
    auto rms = n->tapenode[0];
    auto y   = n->tapenode[1];   // normalized x


    // upstream dot
    auto dot = rowsum_nodeops(gy * y);  // [batch x 1]

    auto grad_x = g*((gy / rms) - (y * dot / (rms*x->value.cols())));
    auto m = gy * (x / rms);


    if (x->requires_grad) x->grad.add_(grad_x->value);
    if (g->requires_grad) g->grad.add_( m->value);


}













// // ----- elementwise quarternary -----
void hvjp_RELUAtt(Node* n, const std::shared_ptr<Node>& gy){
    auto A = n->inputs[0];
    auto B = n->inputs[1];
    auto C = n->inputs[2];
    auto D = n->inputs[3];
    
     auto q = n->tapenode[0] ;
     auto k = n->tapenode[1] ;
     auto v = n->tapenode[2] ;
    float scale = 1.0f / std::sqrt(float(k->value.cols()));
     auto s = n->tapenode[3] ;

    // ---- Backprop chain ----

    // y = s v
     auto dL_ds =  matmul_nodeops(gy,  transpose_nodeops(v));   // [B x B]
     auto dL_dv =  matmul_nodeops( transpose_nodeops(s), gy);   // [A x D]

    // s = softmax(g)
    
         auto dot =  relumask_nodeops(s )* dL_ds;
      auto  dL_dg = dot;
    

    // g = q k^T
     auto dL_dq =  matmul_nodeops(dL_dg, k);
     auto dL_dk =  matmul_nodeops( transpose_nodeops(dL_dg), q);

    // q = A B
     auto dL_dA_q =  matmul_nodeops(dL_dq,  transpose_nodeops(B));
     auto dL_dB   =  matmul_nodeops( transpose_nodeops(A), dL_dq)* scale;;

    // k = A C
     auto dL_dA_k =  matmul_nodeops(dL_dk,  transpose_nodeops(C));
     auto dL_dC   =  matmul_nodeops( transpose_nodeops(A), dL_dk)* scale;

    // v = A D
     auto dL_dA_v =  matmul_nodeops(dL_dv,  transpose_nodeops(D));
     auto dL_dD   =  matmul_nodeops( transpose_nodeops(A), dL_dv);

    // combine A contributions
     auto dL_dA = dL_dA_q + dL_dA_k + dL_dA_v;

    // ---- Accumulate ----
    if (A->requires_grad) A->grad.add_(dL_dA->value);
    if (B->requires_grad) B->grad.add_(dL_dB->value);
    if (C->requires_grad) C->grad.add_(dL_dC->value);
    if (D->requires_grad) D->grad.add_(dL_dD->value);

}




// // ----- elementwise quarternary -----
void hvjp_SigAtt(Node* n, const std::shared_ptr<Node>& gy){
    auto A = n->inputs[0];
    auto B = n->inputs[1];
    auto C = n->inputs[2];
    auto D = n->inputs[3];
    
     auto q = n->tapenode[0] ;
     auto k = n->tapenode[1] ;
     auto v = n->tapenode[2] ;
    float scale = 1.0f / std::sqrt(float(k->value.cols()));
     auto s = n->tapenode[3] ;

    // ---- Backprop chain ----

    // y = s v
     auto dL_ds =  matmul_nodeops(gy,  transpose_nodeops(v));   // [B x B]
     auto dL_dv =  matmul_nodeops( transpose_nodeops(s), gy);   // [A x D]

    // s = softmax(g)
    auto ones_node = std::make_shared<Node>(
        Tensor::ones_like(s->value),
        s->requires_grad, Op::Leaf, "ones_like"
    );
         auto dot =  ( s * (ones_node-s))* dL_ds;
      auto  dL_dg = dot;
    

    // g = q k^T
     auto dL_dq =  matmul_nodeops(dL_dg, k);
     auto dL_dk =  matmul_nodeops( transpose_nodeops(dL_dg), q);

    // q = A B
     auto dL_dA_q =  matmul_nodeops(dL_dq,  transpose_nodeops(B));
     auto dL_dB   =  matmul_nodeops( transpose_nodeops(A), dL_dq)* scale;;

    // k = A C
     auto dL_dA_k =  matmul_nodeops(dL_dk,  transpose_nodeops(C));
     auto dL_dC   =  matmul_nodeops( transpose_nodeops(A), dL_dk)* scale;

    // v = A D
     auto dL_dA_v =  matmul_nodeops(dL_dv,  transpose_nodeops(D));
     auto dL_dD   =  matmul_nodeops( transpose_nodeops(A), dL_dv);

    // combine A contributions
     auto dL_dA = dL_dA_q + dL_dA_k + dL_dA_v;

    // ---- Accumulate ----
    if (A->requires_grad) A->grad.add_(dL_dA->value);
    if (B->requires_grad) B->grad.add_(dL_dB->value);
    if (C->requires_grad) C->grad.add_(dL_dC->value);
    if (D->requires_grad) D->grad.add_(dL_dD->value);

}


















// // ----- elementwise quarternary -----
void hvjp_Attention(Node* n, const std::shared_ptr<Node>& gy){
    auto A = n->inputs[0];
    auto B = n->inputs[1];
    auto C = n->inputs[2];
    auto D = n->inputs[3];
    
     auto q = n->tapenode[0] ;
     auto k = n->tapenode[1] ;
     auto v = n->tapenode[2] ;
    float scale = 1.0f / std::sqrt(float(k->value.cols()));
        auto g = matmul_nodeops(q, (transpose_nodeops(k)*scale)) ;
    auto s = softmax_row_nodeops(g);

    // ---- Backprop chain ----

    // y = s v
     auto dL_ds =  matmul_nodeops(gy,  transpose_nodeops(v));   // [B x B]
     auto dL_dv =  matmul_nodeops( transpose_nodeops(s), gy);   // [A x D]

    // s = softmax(g)
    
         auto dot =  rowsum_nodeops(s * dL_ds);
      auto  dL_dg = s * (dL_ds - dot);
    

    // g = q k^T
     auto dL_dq =  matmul_nodeops(dL_dg, k);
     auto dL_dk =  matmul_nodeops( transpose_nodeops(dL_dg), q);

    // q = A B
     auto dL_dA_q =  matmul_nodeops(dL_dq,  transpose_nodeops(B));
     auto dL_dB   =  matmul_nodeops( transpose_nodeops(A), dL_dq)* scale;;

    // k = A C
     auto dL_dA_k =  matmul_nodeops(dL_dk,  transpose_nodeops(C));
     auto dL_dC   =  matmul_nodeops( transpose_nodeops(A), dL_dk)* scale;

    // v = A D
     auto dL_dA_v =  matmul_nodeops(dL_dv,  transpose_nodeops(D));
     auto dL_dD   =  matmul_nodeops( transpose_nodeops(A), dL_dv);

    // combine A contributions
     auto dL_dA = dL_dA_q + dL_dA_k + dL_dA_v;

    // ---- Accumulate ----
    if (A->requires_grad) A->grad.add_(dL_dA->value);
    if (B->requires_grad) B->grad.add_(dL_dB->value);
    if (C->requires_grad) C->grad.add_(dL_dC->value);
    if (D->requires_grad) D->grad.add_(dL_dD->value);

}


void hvjp_AlibiAttention(Node* n, const std::shared_ptr<Node>& gy){
    auto A = n->inputs[0];
    auto B = n->inputs[1];
    auto C = n->inputs[2];
    auto D = n->inputs[3];
    
     auto q = n->tapenode[0] ;
     auto k = n->tapenode[1] ;
     auto v = n->tapenode[2] ;
    float scale = 1.0f / std::sqrt(float(k->value.cols()));
     auto s = n->tapenode[3] ;

    // ---- Backprop chain ----

    // y = s v
     auto dL_ds =  matmul_nodeops(gy,  transpose_nodeops(v));   // [B x B]
     auto dL_dv =  matmul_nodeops( transpose_nodeops(s), gy);   // [A x D]

    // s = softmax(g)
    
         auto dot =  rowsum_nodeops(s * dL_ds);
      auto  dL_dg = s * (dL_ds - dot);
    

    // g = q k^T
     auto dL_dq =  matmul_nodeops(dL_dg, k);
     auto dL_dk =  matmul_nodeops( transpose_nodeops(dL_dg), q);

    // q = A B
     auto dL_dA_q =  matmul_nodeops(dL_dq,  transpose_nodeops(B));
     auto dL_dB   =  matmul_nodeops( transpose_nodeops(A), dL_dq)* scale;;

    // k = A C
     auto dL_dA_k =  matmul_nodeops(dL_dk,  transpose_nodeops(C));
     auto dL_dC   =  matmul_nodeops( transpose_nodeops(A), dL_dk)* scale;

    // v = A D
     auto dL_dA_v =  matmul_nodeops(dL_dv,  transpose_nodeops(D));
     auto dL_dD   =  matmul_nodeops( transpose_nodeops(A), dL_dv);

    // combine A contributions
     auto dL_dA = dL_dA_q + dL_dA_k + dL_dA_v;

    // ---- Accumulate ----
    if (A->requires_grad) A->grad.add_(dL_dA->value);
    if (B->requires_grad) B->grad.add_(dL_dB->value);
    if (C->requires_grad) C->grad.add_(dL_dC->value);
    if (D->requires_grad) D->grad.add_(dL_dD->value);

}


void hvjp_SWIGLU(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0];
    auto A = n->inputs[1];
    auto B = n->inputs[2];
    auto C = n->inputs[3];
    auto D = n->inputs[4];

     auto y =  matmul_nodeops(X,  transpose_nodeops(A)) + B;
     auto q = y *  sigmoid_nodeops(y);
     auto h =  matmul_nodeops(X,  transpose_nodeops(C)) + D;
     auto w = q * h;

    // derivatives



                 auto ones_node = std::make_shared<Node>(
        Tensor::ones_like(y->value),
        X->requires_grad, Op::Leaf, "ones_like"
    );
     auto Swishdif =  sigmoid_nodeops(y) + y * ( sigmoid_nodeops(y) * ( ones_node -  sigmoid_nodeops(y)));

     auto dL_dB = Swishdif * h * gy;
     auto dL_dA =  matmul_nodeops( transpose_nodeops(Swishdif * h * gy), X);

     auto dL_dD = q*gy;
     auto dL_dC =  matmul_nodeops( transpose_nodeops(q * gy), X);

     auto dL_dX =  matmul_nodeops(Swishdif * h * gy, A)
                 +  matmul_nodeops(q * gy, C);

    // accumulate grads
    if (X->requires_grad) X->grad.add_(dL_dX->value);
    if (A->requires_grad) A->grad.add_(dL_dA->value);
    if (B->requires_grad) B->grad.add_(dL_dB->value);
    if (C->requires_grad) C->grad.add_(dL_dC->value);
    if (D->requires_grad) D->grad.add_(dL_dD->value);

}


// // ----- unary activations -----
void hvjp_Relu(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0];
    if (!X->requires_grad) return;

        auto* mm = ag::kernels::cpu().relumask;
            auto [K2, N] = (X->value).shape();

                 Tensor dA(K2, N);                   // temp buffer

        if(mm)
        {
            mm((X->value).data(), dA.data(), dA.numel());
            auto p = std::make_shared<Node>(dA, false, Op::Relumask, "relumask");
                X->grad.add_( rt(( gy * p)->value, X->value) );

        }
        else{
           dA = Tensor::relu_mask(X->value);
            auto p = std::make_shared<Node>(dA, false, Op::Relumask, "relumask");
                X->grad.add_( rt(( gy * p)->value, X->value) );


        }


}


void hvjp_MOE(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0];
    auto W = n->inputs[1];
    auto B = n->inputs[2];

    auto y = matmul_nodeops(X, transpose_nodeops(W)) + B; 

    auto dL_dB = gy;
    auto dL_dW = matmul_nodeops(transpose_nodeops(gy), X);
    auto dL_dX = matmul_nodeops(gy, W);

    if (X->requires_grad) X->grad.add_(dL_dX->value);
    if (W->requires_grad) W->grad.add_(dL_dW->value);
    if (B->requires_grad) B->grad.add_(dL_dB->value);

}


void hvjp_Exp(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0];
    
    if (X->requires_grad) X->grad.add_( rt( (gy *  n->tapenode[0])->value, X->value) );
}


void hvjp_Log(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0];
    std::cout<<"trfhnjy";
    if (X->requires_grad) X->grad.add_( rt( (gy / X)->value, X->value) );
}

void hvjp_GCU(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0];
    if (X->requires_grad) X->grad.add_( rt( (gy * ( cos_nodeops(X)-(X* sin_nodeops(X))))->value, X->value) );
 }

void hvjp_Mish(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0];
    if (X->requires_grad) X->grad.add_( rt( (gy * ( tanh_nodeops(  softplus_nodeops(X) )-(  (X* sigmoid_nodeops(X))  / ( cosh_nodeops(  softplus_nodeops(X)* cosh_nodeops(  softplus_nodeops(X) ))    )            )))->value, X->value) );
}

void hvjp_Cosh(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0];
    if (X->requires_grad) X->grad.add_( rt( (gy * (sinh_nodeops(X)))->value, X->value) );
}


void hvjp_Sinh(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0];
    if (X->requires_grad) X->grad.add_( rt( (gy * (cosh_nodeops(X)))->value, X->value) );
}


void hvjp_Cos(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0];
    if (X->requires_grad) X->grad.add_( rt( -1.0*(gy * (sin_nodeops(X)))->value, X->value) );
}


void hvjp_Sin(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0];
    if (X->requires_grad) X->grad.add_( rt( (gy * (cos_nodeops(X)))->value, X->value) );
}


void hvjp_Tanh(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0];
    if (!X->requires_grad) return;
     auto th = n->shared_from_this();


                 auto ones_node = std::make_shared<Node>(
        Tensor::ones_like(X->value),
        X->requires_grad, Op::Leaf, "ones_like"
    );
    
    X->grad.add_( rt( (gy * (ones_node - th*th))->value, X->value) );
}



void hvjp_Sigmoid(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0]; if (!X->requires_grad) return;
     auto s =  sigmoid_nodeops(X);



                 auto ones_node = std::make_shared<Node>(
        Tensor::ones_like(X->value),
        X->requires_grad, Op::Leaf, "ones_like"
    );
    std::cout<<"YAY!";


    X->grad.add_( rt( (gy * ( s * ( ones_node-s) ))->value, X->value) );
}




void hvjp_Softplus(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0]; if (!X->requires_grad) return;
    X->grad.add_( rt( (gy * sigmoid_nodeops(X))->value, X->value) );
}

void hvjp_Gaus(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0]; if (!X->requires_grad) return;
    X->grad.add_( rt( (gy * -2*X* exp_nodeops(-1*X*X))->value, X->value) );
}

void hvjp_Transpose(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0]; if (!X->requires_grad) return;

    X->grad.add_( rt(  (transpose_nodeops(gy) )->value, X->value) );
}





void hvjp_Linear(Node* n, const std::shared_ptr<Node>& gy){
    auto A = n->inputs[0];
    auto B = n->inputs[1];
    auto C = n->inputs[2];

    // External kernel (if plugin loaded), else fallback to Tensor::matmul
    auto* mm = ag::kernels::cpu().matmul;

    // Shapes
    auto At = A;
    auto Bt = B;
    auto [M, K]  = A->value.shape();
    auto [K2, N] = B->value.shape();
    (void)K2; // assume forward already checked

    if (A->requires_grad){
        auto BT = transpose_nodeops(B); // (N x K)
        Tensor dA(M, K);                   // temp buffer

        if (mm) {
            // dA = gy (MxN) * BT (NxK)
            mm(gy->value.data(), BT->value.data(), dA.data(), M, N, K);

            auto c = std::make_shared<Node>(dA, false, Op::MatMul, "matmul");

            A->grad.add_(c->value);
        } else {
            dA = Tensor::matmul(gy->value, BT->value);

            auto c = std::make_shared<Node>(dA, false, Op::MatMul, "matmul");

            A->grad.add_(c->value);
        }
        A->grad.add_(dA);
    }

    if (B->requires_grad){
        auto AT = transpose_nodeops(At); // (K x M)
        Tensor dB(K, N);                   // temp buffer

        if (mm) {
            // dB = AT (KxM) * gy (MxN)
            mm(AT->value.data(), gy->value.data(), dB.data(), K, M, N);

            auto c = std::make_shared<Node>(dB, false, Op::MatMul, "matmul");

            A->grad.add_(c->value);
        } else {
            dB = Tensor::matmul(AT->value, gy->value);
              auto c = std::make_shared<Node>(dB, false, Op::MatMul, "matmul");

            A->grad.add_(c->value);
        }
        B->grad.add_(dB);
    }
    if (C->requires_grad) C->grad.add_( rt(gy->value, C->value) );
}







void hvjp_SiLU(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0]; if (!X->requires_grad) return;
     auto s =  sigmoid_nodeops(X);


auto ones_node = std::make_shared<Node>(
        Tensor::ones_like(s->value),
        X->requires_grad, Op::Leaf, "ones_like"
    );
    X->grad.add_( rt( (gy * ( s + X * ( s * ( ones_node-s) ) ))->value, X->value) );
}

void hvjp_Parcon(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0]; if (!X->requires_grad) return;
    auto ones_node = std::make_shared<Node>(
        Tensor::ones_like(X->value),
        X->requires_grad, Op::Leaf, "ones_like"
    );
    X->grad.add_( rt( (gy * ( 2 * ones_node - 2*X  ))->value, X->value) );
}

void hvjp_LiSHT(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0]; if (!X->requires_grad) return;
    X->grad.add_( rt( (gy * (  tanh_nodeops(X)+ ( reci_nodeops(cosh_nodeops(X)* cosh_nodeops(X))*X ) ))->value, X->value) );
}




void hvjp_GELU(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0];
    if (!X->requires_grad) return;

    constexpr float c = 0.7978845608028654f; // sqrt(2/pi)
    int R = X->value.rows(), C = X->value.cols();
    Tensor g(R, C);

    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            float x = X->value(i, j);

            // GELU(x) = 0.5 * x * (1 + tanh(c*(x + 0.044715x^3)))
            float u = c * (x + 0.044715f * x * x * x);
            float t = std::tanh(u);
            float dudx = c * (1.f + 3.f * 0.044715f * x * x);

            // derivative of GELU wrt x:
            float dgelu = 0.5f * (1.f + t) + 0.5f * x * (1.f - t * t) * dudx;

            g(i, j) = gy->value(i, j) * dgelu;
        }
    }

    X->grad.add_(g);

}



void hvjp_LeakyRelu(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0]; auto A = n->inputs[1];
    if (!X->requires_grad) return;
    float a = A->value(0,0);
    int R=X->value.rows(), C=X->value.cols();
     Tensor g(R,C);
    for(int i=0;i<R;++i) for(int j=0;j<C;++j){
        float z=X->value(i,j);
        g(i,j)= gy->value(i,j) * (z>0.f ? 1.f : a);
    }
    X->grad.add_( g );
}

// // ----- matmul -----
void hvjp_MatMul(Node* n, const std::shared_ptr<Node>& gy){
   auto A = n->inputs[0];
    auto B = n->inputs[1];

    // External kernel (if plugin loaded), else fallback to  matmul_nodeops
    auto* mm = ag::kernels::cpu().matmul;


    auto [M, K]  = A->value.shape();
    auto [K2, N] = B->value.shape();
    (void)K2; // assume forward already checked

    if (A->requires_grad){
        auto BT = transpose_nodeops(n->inputs[1]); // (N x K)
         Tensor dA(M, K);                   // temp buffer

        if (mm) {
            // dA = gy (MxN) * BT (NxK)
            mm(gy->value.data(), BT->value.data(), dA.data(), M, N, K);
                    auto c = std::make_shared<Node>(dA, false, Op::MatMul, "matmul");

            A->grad.add_(c->value);
        } else {
             dA = Tensor::matmul(gy->value, BT->value);

                auto c = std::make_shared<Node>(dA, false, Op::MatMul, "matmul");

            A->grad.add_(c->value);
        }
        
    }

    if (B->requires_grad){
        auto AT = transpose_nodeops(n->inputs[0]); // (K x M)
         Tensor dB(K, N);                   // temp buffer

        if (mm) {
            // dB = AT (KxM) * gy (MxN)
            mm(AT->value.data(), gy->value.data(), dB.data(), K, M, N);
                              auto c = std::make_shared<Node>(dB, false, Op::MatMul, "matmul");

            B->grad.add_(c->value);
        } else {
             dB = Tensor::matmul(AT->value, gy->value);

                auto c = std::make_shared<Node>(dB, false, Op::MatMul, "matmul");

            B->grad.add_(c->value);
        }
       
    }
}

 void hvjp_Dyntanh(Node* n, const std::shared_ptr<Node>& gy){
    auto X = n->inputs[0]; 
    auto A = n->inputs[1]; 
    auto B = n->inputs[2]; 
    auto G = n->inputs[3];



        if (X->requires_grad) X->grad.add_((gy* reci_nodeops(cosh_nodeops(X*A)* cosh_nodeops(X*A))*A*G)->value); 
    if (A->requires_grad) A->grad.add_((gy* reci_nodeops(cosh_nodeops(X*A)* cosh_nodeops(X*A))*X*G)->value);
    if (B->requires_grad) B->grad.add_(gy->value);
    if (G->requires_grad) G->grad.add_((gy* tanh_nodeops((n->tapenode.back()))  )->value );
 }




// // ----- reductions -----
void hvjp_Sum(Node* n, const std::shared_ptr<Node>& gy){
    auto X=n->inputs[0]; if(!X->requires_grad) return;
    float s = gy->value(0,0);


            auto ones_node = std::make_shared<Node>(
        Tensor::ones_like(X->value),
        X->requires_grad, Op::Leaf, "ones_like"
    );
    X->grad.add_( (ones_node * s )->value );
}



void hvjp_RowSum(Node* n, const std::shared_ptr<Node>& gy){
    auto X=n->inputs[0]; if(!X->requires_grad) return;
    // gy [B,1] broadcast across columns
        auto ones_node = std::make_shared<Node>(
        Tensor::ones_like(X->value),
        X->requires_grad, Op::Leaf, "ones_like"
    );
     Tensor g = Tensor::ones_like(X->value) * gy->value; // explicit broadcast
auto n_g = std::make_shared<Node>(g, false, Op::Leaf, "broadcast_grad");
    X->grad.add_( n_g->value );
}




void hvjp_RowMax(Node* n, const std::shared_ptr<Node>& gy){
    auto X=n->inputs[0]; if(!X->requires_grad) return;
    int R=X->value.rows(), C=X->value.cols();
     auto m =  rowmax_nodeops(X);
     Tensor g(R,C);
    for(int i=0;i<R;++i) for(int j=0;j<C;++j)
        g(i,j) = (X->value(i,j)==m->value(i,0)) ? gy->value(i,0) : 0.f;
    X->grad.add_( g );
}




void hvjp_MeanAll(Node* n, const std::shared_ptr<Node>& gy){
    auto X=n->inputs[0]; if(!X->requires_grad) return;


    // Get scale as a node (constant node)
    float inv_size = 1.0f / float(X->value.rows() * X->value.cols());
    auto scale_node = std::make_shared<Node>(
        inv_size * Tensor::ones_like(gy->value),
        false, Op::Leaf, "const_scale"
    );

    // ones_like(X) as node
    auto ones_node = std::make_shared<Node>(
        Tensor::ones_like(X->value),
        X->requires_grad, Op::Leaf, "ones_like"
    );

    // Compute y = ones_like(X) * gy * scale (all as nodes)
    auto temp = mul_nodeops(ones_node, gy);       // gy * ones
    auto scaled = mul_nodeops(temp, scale_node);  // * scale

    // Accumulate back into gradient
    X->grad.add_( scaled->value );
}

// // ----- softmax / losses -----
void hvjp_SoftmaxRow(Node* n, const std::shared_ptr<Node>& gy){
    auto Z = n->inputs[0]; if(!Z->requires_grad) return;
     auto y = n->shared_from_this(); // softmax(Z)
     auto dot =  rowsum_nodeops( y * gy ); // [B,1]
     auto g = y * (gy - dot);
    Z->grad.add_( g->value );
}



void hvjp_LogSumExpRow(Node* n, const std::shared_ptr<Node>& gy){
    auto Z = n->inputs[0]; if(!Z->requires_grad) return;
     auto y =  softmax_row_nodeops(Z);
    Z->grad.add_( (y * gy)->value ); // gy [B,1] broadcast
}



void hvjp_CeWithLogits(Node* n, const std::shared_ptr<Node>& gy /*unused: scalar gy*/){
    auto Z = n->inputs[0];
    auto Y = n->inputs[1];
    int B = Z->value.rows();
     auto sm =  softmax_row_nodeops(Z);
     auto gZ = (sm - Y) * (1.0f / float(B));
    if (Z->requires_grad) Z->grad.add_( gZ->value );
    if (Y->requires_grad) {
         auto lse =  logsumexp_row_nodeops(Z);
         auto lsm = Z - lse;
         auto gY  = lsm * (-1.0f / float(B));
        Y->grad.add_( gY->value );
    }
}

void hvjp_MAELoss(Node* n, const std::shared_ptr<Node>& gy /*unused: scalar gy*/){
    auto Z = n->inputs[0];
    auto Y = n->inputs[1];
    int B = Z->value.rows(), C = Z->value.cols();
     auto diff =  sign_nodeops(Z - Y);
     auto gZ = diff * (1.0f / float(B * C));
     auto gY = -1.0*diff * (1.0f / float(B * C));
    if (Z->requires_grad) Z->grad.add_(gZ->value);
    if (Y->requires_grad) Y->grad.add_(gY->value);

}


void hvjp_KLDivergence(Node* n, const std::shared_ptr<Node>& gy /*unused: scalar gy*/){
    auto Z = n->inputs[0];
    auto Y = n->inputs[1];
    int B = Z->value.rows();
     auto sm =  softmax_row_nodeops(Z);
     auto gZ = (sm - Y) * (1.0f / float(B));
    if (Z->requires_grad) Z->grad.add_( gZ->value );
    if (Y->requires_grad) {
         auto lse =  logsumexp_row_nodeops(Z);
         auto lsm = Z - lse;


 auto ones_node = std::make_shared<Node>(
        Tensor::ones_like(Y->value),
        Y->requires_grad, Op::Leaf, "ones_like"
    );

         auto gY = ( log_nodeops(Y) +  ones_node - lsm) * (1.0f / float(B));
        Y->grad.add_( gY->value );
    }



}







void hvjp_Leaf(Node*, const std::shared_ptr<Node>&){ /* no-op */ }

}
 // anon


HVjpFn hvjp_lookup(Op op){
    switch(op){
#define OP(name, arity, str) case Op::name: return &detail::hvjp_##name;
#include "ad/detail/ops.def"
#undef OP
        default: return nullptr;
    }
}

} // namespace ag
