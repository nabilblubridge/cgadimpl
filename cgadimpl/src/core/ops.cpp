// =====================
// file: src/ops.cpp
// =====================

#include <iostream>
#include "ad/ops.hpp"
#include "ad/graph.hpp"
#include "ad/debug.hpp"
#include "ad/kernels_api.hpp"
#include "ad/nodeops.hpp"
#include <math.h>
#include <iterator>


namespace ag {


    Value add(const Value& a, const Value& b){ 
        return Value(add(a.node, b.node)); 
    }

    Value sub(const Value& a, const Value& b){ 
        
        return Value(sub(a.node, b.node)); 
    }



    Value mul(const Value& a, const Value& b){ 
        return Value(mul(a.node, b.node)); 
    }

    Value flomul(const Value& a, float b){ 
        return Value(flomul(a.node, b));
    }

    Value relu(const Value& x){ 
      
        return Value(relu(x.node));
    }





    Value matmul(const Value& a, const Value& b){ 
         return Value(matmul(a.node, b.node)); 
    }

    Value fmab(const Value& a, const Value& b, const Value& c){ 
        return Value(fmab(a.node, b.node, c.node)); 
    }


    Value attention(const Value& a, const Value& b, const Value& c, const Value& d){ 
    return Value(attention(a.node, b.node, c.node, d.node));
    }


    Value alibiatt(const Value& a, const Value& b, const Value& c, const Value& d, float m) { 
    return Value(alibiatt(a.node, b.node, c.node, d.node, m));
}



    Value swiglu(const Value& x, const Value& a, const Value& b, const Value& c, const Value& d){ 
    return Value(swiglu(x.node, a.node, b.node, c.node, d.node));
    }







    Value sum(const Value& x){ 
        return Value(ag::sum(x.node));
    }

    Value transpose(const Value& x){ 
        return Value(transpose(x.node));
    }

    Value exp(const Value& x){ 
        return Value(exp(x.node));
    }
    
    Value log(const Value& x){ 
        return Value(log(x.node));
    }


    Value mish(const Value& x){ 
        return Value(mish(x.node));
    }
    
    Value tanh(const Value& x){ 
        return Value(tanh(x.node));
    }
    
    Value sigmoid(const Value& x){ 
        return Value(sigmoid(x.node));
    }
    
    Value softplus(const Value& x){ 
        return Value(softplus(x.node));
    }

    Value gaus(const Value& x){ 
        return Value(gaus(x.node));
    }
    
    Value gelu(const Value& x){ 
        return Value(gelu(x.node));
    }



    Value gcu(const Value& x){ 
        return Value(gcu(x.node));
    }
    
    Value silu(const Value& x){ 
        return Value(silu(x.node));
    }

    Value parcon(const Value& x){ 
        return Value(parcon(x.node));
    }

    Value lisht(const Value& x){ 
        return Value(lisht(x.node));
    }
    
    Value leaky_relu(const Value& x, float alpha){ 
        return Value(leaky_relu(x.node, alpha));
    }


    Value rowsum(const Value& x){ 
        return Value(rowsum(x.node));
    }
    
    Value rowmax(const Value& x){ 
        return Value(rowmax(x.node));
    }

    Value rms(const Value& x){ 
return Value(rms(x.node));
    }

    Value realrms(const Value& x, float g){ 
return Value(realrms(x.node, g));
    }

    Value laynor(const Value& x){ 
        return Value(laynor(x.node));
    }

    Value relaynor(const Value& x, float b, float g){ 
        return Value(relaynor(x.node, b, g));
    }
    
    Value mean_all(const Value& x){ 
        return Value(mean_all(x.node));
    }

    Value dyntanh(const Value& x, float a, float b, float g){ 
        return Value(dyntanh(x.node, a, b, g));
    }
    
    Value softmax_row(const Value& z){ 
        return Value(softmax_row(z.node));
    }
    
    Value logsumexp_row(const Value& z){ 
        return Value(logsumexp_row(z.node));
    }


    Value mambassm(const Value& z, const Value& a, const Value& b, const Value& c, const Value& d){ 

        return Value(mambassm(z.node, a.node, b.node, c.node, d.node));

        
    }


    Value cross_entropy_with_logits(const Value& logits, const Value& onehot){
    // Stable CE = mean( -sum(onehot * (logits - logsumexp_row(logits))) )
        return Value(cross_entropy_with_logits(logits.node, onehot.node));
    }


    Value kldivergence(const Value& logits, const Value& onehot){
        return Value(kldivergence(logits.node, onehot.node));
    }

    Value mse_loss(const Value& pred, const Value& target) {
    return Value(mse_loss(pred.node, target.node));
}


    Value mae_loss(const Value& pred, const Value& target) {
    return Value(mae_loss(pred.node, target.node));
}

Tensor forward_eval_node(const std::shared_ptr<Node> &node) {
    if (!node) throw std::runtime_error("forward_eval_node: null node");

    switch (node->op) {
        case Op::Add: {
            const Tensor &A = node->inputs[0]->value;
            const Tensor &B = node->inputs[1]->value;
            return A + B;
        }
        case Op::Sub: {
            const Tensor &A = node->inputs[0]->value;
            const Tensor &B = node->inputs[1]->value;
            return A - B;
        }
        case Op::Mul: {
            const Tensor &A = node->inputs[0]->value;
            const Tensor &B = node->inputs[1]->value;
            return A * B;
        }
        case Op::MatMul: {
            const Tensor &A = node->inputs[0]->value;
            const Tensor &B = node->inputs[1]->value;
            return Tensor::matmul(A, B);
        }
        case Op::Relu: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::relu(X);
        }
        case Op::Sigmoid: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::sigmoid(X);
        }
        case Op::Tanh: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::tanh(X);
        }
        case Op::Exp: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::exp(X);
        }
        case Op::Log: {
            const Tensor &X = node->inputs[0]->value;
            return Tensor::log(X);
        }
        case Op::AlibiAttention: {
            const Tensor &a = node->inputs[0]->value;
            const Tensor &b = node->inputs[1]->value;
            const Tensor &c = node->inputs[2]->value;
            const Tensor &d = node->inputs[3]->value;

            Tensor q = Tensor::matmul(a, b);
            Tensor k = Tensor::matmul(a, c);
            Tensor v = Tensor::matmul(a, d);

            Tensor logits = Tensor::matmul(q, Tensor::transpose(k) * (1.f / sqrt(float(k.cols()))));
            Tensor bias   = Tensor::alibi(logits.rows(), logits.cols(), /*m*/128);
            Tensor g      = logits + bias;
            Tensor s      = Tensor::softmax_row(g);
            Tensor y      = Tensor::matmul(s, v);
            return y;
        }
        case Op::Leaf:
            return node->value;
        default:
            if (!node->tape.empty()) {
                return *(node->tape.back());
            }
            throw std::runtime_error("forward_eval_node: unsupported op for recompute");
    }
}

// ------------------------------------------------------------
// Small adapter so checkpoint.cpp (which uses Node*) can link.
// ------------------------------------------------------------
Tensor forward_eval_node(Node* node) {
    // Non-owning shared_ptr wrapper (no deletion)
    return forward_eval_node(std::shared_ptr<Node>(node, [](Node*){}));
}

// ------------------------------------------------------------
// checkpoint() — mark a node for checkpointing
// ------------------------------------------------------------
Value checkpoint(const Value &v, const CheckpointOptions &opts) {
    if (!v.node) return v;
    ag::checkpoint_impl::mark_node_checkpoint(v.node, opts);
    return v;
}



} // namespace ag