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
        return Value(add_nodeops(a.node, b.node)); 
    }

    Value sub(const Value& a, const Value& b){ 
        
        return Value(sub_nodeops(a.node, b.node)); 
    }



    Value mul(const Value& a, const Value& b){ 
        return Value(mul_nodeops(a.node, b.node)); 
    }

    Value flomul(const Value& a, float b){ 
        return Value(flomul_nodeops(a.node, b));
    }

    Value relu(const Value& x){ 
      
        return Value(relu_nodeops(x.node));
    }





    Value matmul(const Value& a, const Value& b){ 
         return Value(matmul_nodeops(a.node, b.node)); 
    }

    Value fmab(const Value& a, const Value& b, const Value& c){ 
        return Value(fmab_nodeops(a.node, b.node, c.node)); 
    }


    Value attention(const Value& a, const Value& b, const Value& c, const Value& d){ 
    return Value(attention_nodeops(a.node, b.node, c.node, d.node));
    }


    Value alibiatt(const Value& a, const Value& b, const Value& c, const Value& d, float m) { 
    return Value(alibiatt_nodeops(a.node, b.node, c.node, d.node, m));
}



    Value swiglu(const Value& x, const Value& a, const Value& b, const Value& c, const Value& d){ 
    return Value(swiglu_nodeops(x.node, a.node, b.node, c.node, d.node));
    }







    Value sum(const Value& x){ 
        return Value(ag::sum_nodeops(x.node));
    }

    Value transpose(const Value& x){ 
        return Value(transpose_nodeops(x.node));
    }

    Value exp(const Value& x){ 
        return Value(exp_nodeops(x.node));
    }
    
    Value log(const Value& x){ 
        return Value(log_nodeops(x.node));
    }


    Value mish(const Value& x){ 
        return Value(mish_nodeops(x.node));
    }
    
    Value tanh(const Value& x){ 
        return Value(tanh_nodeops(x.node));
    }
    
    Value sigmoid(const Value& x){ 
        return Value(sigmoid_nodeops(x.node));
    }
    
    Value softplus(const Value& x){ 
        return Value(softplus_nodeops(x.node));
    }

    Value gaus(const Value& x){ 
        return Value(gaus_nodeops(x.node));
    }
    
    Value gelu(const Value& x){ 
        return Value(gelu_nodeops(x.node));
    }



    Value gcu(const Value& x){ 
        return Value(gcu_nodeops(x.node));
    }
    
    Value silu(const Value& x){ 
        return Value(silu_nodeops(x.node));
    }

    Value parcon(const Value& x){ 
        return Value(parcon_nodeops(x.node));
    }

    Value lisht(const Value& x){ 
        return Value(lisht_nodeops(x.node));
    }
    
    Value leaky_relu(const Value& x, float alpha){ 
        return Value(leaky_relu_nodeops(x.node, alpha));
    }


    Value rowsum(const Value& x){ 
        return Value(rowsum_nodeops(x.node));
    }
    
    Value rowmax(const Value& x){ 
        return Value(rowmax_nodeops(x.node));
    }

    Value rms(const Value& x){ 
return Value(rms_nodeops(x.node));
    }

    Value realrms(const Value& x, float g){ 
return Value(realrms_nodeops(x.node, g));
    }

    Value laynor(const Value& x){ 
        return Value(laynor_nodeops(x.node));
    }

    Value relaynor(const Value& x, float b, float g){ 
        return Value(relaynor_nodeops(x.node, b, g));
    }
    
    Value mean_all(const Value& x){ 
        return Value(mean_all_nodeops(x.node));
    }

    Value dyntanh(const Value& x, float a, float b, float g){ 
        return Value(dyntanh_nodeops(x.node, a, b, g));
    }
    
    Value softmax_row(const Value& z){ 
        return Value(softmax_row_nodeops(z.node));
    }
    
    Value logsumexp_row(const Value& z){ 
        return Value(logsumexp_row_nodeops(z.node));
    }


    Value mambassm(const Value& z, const Value& a, const Value& b, const Value& c, const Value& d){ 

        return Value(mambassm_nodeops(z.node, a.node, b.node, c.node, d.node));

        
    }


    Value cross_entropy_with_logits(const Value& logits, const Value& onehot){
    // Stable CE = mean( -sum(onehot * _nodeops(logits - logsumexp_row_nodeops(logits))) )
        return Value(cross_entropy_with_logits_nodeops(logits.node, onehot.node));
    }


    Value kldivergence(const Value& logits, const Value& onehot){
        return Value(kldivergence_nodeops(logits.node, onehot.node));
    }

    Value mse_loss(const Value& pred, const Value& target) {
    return Value(mse_loss_nodeops(pred.node, target.node));
}


    Value mae_loss(const Value& pred, const Value& target) {
    return Value(mae_loss_nodeops(pred.node, target.node));
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