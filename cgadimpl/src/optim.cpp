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
    for (auto it = order.begin(); it != order.end(); ++it) {
        Node* n = *it;
        if (n->requires_grad ) {
            n->value.add_(-learning_rate * n->grad);
        }
    }

}

}