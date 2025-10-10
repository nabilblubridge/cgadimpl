// =====================
// file: src/ops.cpp
// =====================
#include "ad/ops.hpp"
<<<<<<< HEAD
#include "ad/graph.hpp"
#include "ad/debug.hpp"
#include "ad/kernels_api.hpp"
#include "ad/nodeops.hpp"
#include <math.h>
#include <iterator>

=======
#include "ad/nodeops.hpp" // Include the new node-level declarations
>>>>>>> newbr/HEAD

namespace ag {

    Value add(const Value& a, const Value& b){ 
<<<<<<< HEAD
        return Value(add_nodeops(a.node, b.node)); 
=======
        return Value(detail::add_nodeops(a.node, b.node)); 
>>>>>>> newbr/HEAD
    }

    Value sub(const Value& a, const Value& b){ 
        
<<<<<<< HEAD
        return Value(sub_nodeops(a.node, b.node)); 
=======
        return Value(detail::sub_nodeops(a.node, b.node)); 
>>>>>>> newbr/HEAD
    }



    Value mul(const Value& a, const Value& b){ 
<<<<<<< HEAD
        return Value(mul_nodeops(a.node, b.node)); 
    }

    Value flomul(const Value& a, float b){ 
        return Value(flomul_nodeops(a.node, b));
=======
        return Value(detail::mul_nodeops(a.node, b.node)); 
    }

    Value flomul(const Value& a, float b){ 
        return Value(detail::flomul_nodeops(a.node, b));
>>>>>>> newbr/HEAD
    }

    Value relu(const Value& x){ 
      
<<<<<<< HEAD
        return Value(relu_nodeops(x.node));
=======
        return Value(detail::relu_nodeops(x.node));
>>>>>>> newbr/HEAD
    }





    Value matmul(const Value& a, const Value& b){ 
<<<<<<< HEAD
         return Value(matmul_nodeops(a.node, b.node)); 
    }

    Value fmab(const Value& a, const Value& b, const Value& c){ 
        return Value(fmab_nodeops(a.node, b.node, c.node)); 
=======
         return Value(detail::matmul_nodeops(a.node, b.node)); 
    }

    Value fmab(const Value& a, const Value& b, const Value& c){ 
        return Value(detail::fmab_nodeops(a.node, b.node, c.node)); 
>>>>>>> newbr/HEAD
    }


    Value attention(const Value& a, const Value& b, const Value& c, const Value& d){ 
<<<<<<< HEAD
    return Value(attention_nodeops(a.node, b.node, c.node, d.node));
=======
    return Value(detail::attention_nodeops(a.node, b.node, c.node, d.node));
>>>>>>> newbr/HEAD
    }


    Value alibiatt(const Value& a, const Value& b, const Value& c, const Value& d, float m) { 
<<<<<<< HEAD
    return Value(alibiatt_nodeops(a.node, b.node, c.node, d.node, m));
=======
    return Value(detail::alibiatt_nodeops(a.node, b.node, c.node, d.node, m));
>>>>>>> newbr/HEAD
}



    Value swiglu(const Value& x, const Value& a, const Value& b, const Value& c, const Value& d){ 
<<<<<<< HEAD
    return Value(swiglu_nodeops(x.node, a.node, b.node, c.node, d.node));
=======
    return Value(detail::swiglu_nodeops(x.node, a.node, b.node, c.node, d.node));
>>>>>>> newbr/HEAD
    }


    Value sum(const Value& x){ 
<<<<<<< HEAD
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
=======
        return Value(detail::sum_nodeops(x.node));
    }

    Value transpose(const Value& x){ 
        return Value(detail::transpose_nodeops(x.node));
    }

    Value exp(const Value& x){ 
        return Value(detail::exp_nodeops(x.node));
    }
    
    Value log(const Value& x){ 
        return Value(detail::exp_nodeops(x.node));
>>>>>>> newbr/HEAD
    }


    Value mish(const Value& x){ 
<<<<<<< HEAD
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
=======
        return Value(detail::mish_nodeops(x.node));
    }
    
    Value tanh(const Value& x){ 
        return Value(detail::tanh_nodeops(x.node));
    }
    
    Value sigmoid(const Value& x){ 
        return Value(detail::sigmoid_nodeops(x.node));
    }
    
    Value softplus(const Value& x){ 
        return Value(detail::softplus_nodeops(x.node));
    }

    Value gaus(const Value& x){ 
        return Value(detail::gaus_nodeops(x.node));
    }
    
    Value gelu(const Value& x){ 
        return Value(detail::gelu_nodeops(x.node));
>>>>>>> newbr/HEAD
    }



    Value gcu(const Value& x){ 
<<<<<<< HEAD
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
=======
        return Value(detail::gcu_nodeops(x.node));
    }
    
    Value silu(const Value& x){ 
        return Value(detail::silu_nodeops(x.node));
    }

    Value parcon(const Value& x){ 
        return Value(detail::parcon_nodeops(x.node));
    }

    Value lisht(const Value& x){ 
        return Value(detail::lisht_nodeops(x.node));
    }
    
    Value leaky_relu(const Value& x, float alpha){ 
        return Value(detail::leaky_relu_nodeops(x.node, alpha));
>>>>>>> newbr/HEAD
    }


    Value rowsum(const Value& x){ 
<<<<<<< HEAD
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
=======
        return Value(detail::rowsum_nodeops(x.node));
    }
    
    Value rowmax(const Value& x){ 
        return Value(detail::rowmax_nodeops(x.node));
    }

    Value rms(const Value& x){ 
return Value(detail::rms_nodeops(x.node));
    }

    Value realrms(const Value& x, float g){ 
return Value(detail::realrms_nodeops(x.node, g));
    }

    Value laynor(const Value& x){ 
        return Value(detail::laynor_nodeops(x.node));
    }

    Value relaynor(const Value& x, float b, float g){ 
        return Value(detail::relaynor_nodeops(x.node, b, g));
    }
    
    Value mean_all(const Value& x){ 
        return Value(detail::mean_all_nodeops(x.node));
    }

    Value dyntanh(const Value& x, float a, float b, float g){ 
        return Value(detail::dyntanh_nodeops(x.node, a, b, g));
    }
    
    Value softmax_row(const Value& z){ 
        return Value(detail::softmax_row_nodeops(z.node));
    }
    
    Value logsumexp_row(const Value& z){ 
        return Value(detail::logsumexp_row_nodeops(z.node));
>>>>>>> newbr/HEAD
    }


    Value mambassm(const Value& z, const Value& a, const Value& b, const Value& c, const Value& d){ 

<<<<<<< HEAD
        return Value(mambassm_nodeops(z.node, a.node, b.node, c.node, d.node));
=======
        return Value(detail::mambassm_nodeops(z.node, a.node, b.node, c.node, d.node));
>>>>>>> newbr/HEAD

        
    }


    Value cross_entropy_with_logits(const Value& logits, const Value& onehot){
    // Stable CE = mean( -sum(onehot * _nodeops(logits - logsumexp_row_nodeops(logits))) )
<<<<<<< HEAD
        return Value(cross_entropy_with_logits_nodeops(logits.node, onehot.node));
=======
        return Value(detail::cross_entropy_with_logits_nodeops(logits.node, onehot.node));
>>>>>>> newbr/HEAD
    }


    Value kldivergence(const Value& logits, const Value& onehot){
<<<<<<< HEAD
        return Value(kldivergence_nodeops(logits.node, onehot.node));
    }

    Value mse_loss(const Value& pred, const Value& target) {
    return Value(mse_loss_nodeops(pred.node, target.node));
=======
        return Value(detail::kldivergence_nodeops(logits.node, onehot.node));
    }

    Value mse_loss(const Value& pred, const Value& target) {
    return Value(detail::mse_loss_nodeops(pred.node, target.node));
>>>>>>> newbr/HEAD
}


    Value mae_loss(const Value& pred, const Value& target) {
<<<<<<< HEAD
    return Value(mae_loss_nodeops(pred.node, target.node));
=======
    return Value(detail::mae_loss_nodeops(pred.node, target.node));
>>>>>>> newbr/HEAD
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