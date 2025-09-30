// =====================
// file: src/optim.cpp
// =====================
#include "ad/ops.hpp"
#include "ad/debug.hpp"
#include "ad/optim.hpp"
#include "ad/autodiff.hpp"
#include "ad/autodiff_ops.hpp"
#include "ad/tensor.hpp"
#include "ad/debug.hpp"
#include <math.h>


namespace ag {






void SGD(const Value& root, const Tensor* grad_seed, int learning_rate) {
    auto order = topo_from(root.node.get());

    // seed
    if (root.node->requires_grad) {
        root.node->grad = grad_seed ? *grad_seed
                                    : (root.node->value.size()==1 ? Tensor::ones(1,1)
                                                                  : Tensor::ones_like(root.node->value));
    }

    // reverse topo
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node* n = *it;
        if (!n->requires_grad) continue;
        const Tensor& gy = n->grad;
        ag::debug::on_backprop_step(n, gy); // (optional) prints one line per node

        VjpFn fn = vjp_lookup(n->op);
        if (fn) {
            for(int i=0;i<n->inputs.size();++i)
                if (n->inputs[i]->requires_grad) {
                    n->inputs[i]->value.add_(-1*learning_rate*n->inputs[i]->grad );
                }
        }
        // handler accumulates into parents
    }
}

}