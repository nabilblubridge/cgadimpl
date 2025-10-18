// // =====================
// // file: tests/test_ag.cpp
// // =====================
// #include <iostream>
// #include "ad/ag_all.hpp"
// #include <iomanip>
// #include <iostream>

// static void printTensor(const char* name,
//                         const ag::Tensor& T,
//                         int max_r = -1, int max_c = -1,
//                         int width = 9, int prec = 4) {
//     using std::cout;
//     using std::fixed;
//     using std::setw;
//     using std::setprecision;

//     const int r = T.rows(), c = T.cols();
//     if (max_r < 0) max_r = r;
//     if (max_c < 0) max_c = c;

//     cout << name << " [" << r << "x" << c << "]";
//     if (r == 1 && c == 1) { // scalar fast path
//         cout << " = " << fixed << setprecision(6) << T(0,0) << "\n";
//         return;
//     }
//     cout << "\n";

//     const int rr = std::min(r, max_r);
//     const int cc = std::min(c, max_c);
//     for (int i = 0; i < rr; ++i) {
//         cout << "  ";
//         for (int j = 0; j < cc; ++j) {
//             cout << setw(width) << fixed << setprecision(prec) << T(i,j);
//         }
//         if (cc < c) cout << " ...";
//         cout << "\n";
//     }
//     if (rr < r) cout << "  ...\n";
// }

// using namespace std;

// int main(){
// using namespace ag;
// Tensor A = Tensor::randn(2,3);
// Tensor B = Tensor::randn(3,2);
// auto a = param(A, "A");
// auto b = param(B, "B");


// auto y = sum(relu(matmul(a,b))); // scalar


// zero_grad(y);
// backward(y);
// std::cout << "y = " << y.val().sum_scalar() << endl;
// std::cout << "dL/dA[0,0] = " << a.grad()(0,0) << ", dL/dB[0,0] = " << b.grad()(0,0) << endl;


// // JVP: along dA=ones, dB=zeros
// std::unordered_map<Node*, Tensor> seed; seed[a.node.get()] = Tensor::ones_like(a.val());
// Tensor jy = jvp(y, seed);
// std::cout << "JVP dy(dA,0) = " << jy(0,0) << endl;

// printTensor("A", a.val());
// printTensor("B", b.val());
// ag::Tensor Z = ag::Tensor::matmul(a.val(), b.val());
// printTensor("Z = A*B", Z);
// printTensor("ReLU mask", ag::Tensor::relu_mask(Z));
// printTensor("grad A", a.grad());
// printTensor("grad B", b.grad());
// printTensor("JVP dy(dA,0)", jy);  // jy is 1x1, prints as scalar

// cout << "Numerically verified! \nTest successful!\n";
// return 0;
// }
#include <iostream>
#include "ad/ag_all.hpp"
#include "optim.hpp"
#include <random>
#include <iomanip>
using namespace ag;


int main(){
using namespace std;
using namespace ag;
Tensor A = Tensor::randn(1,7);
Tensor B = Tensor::randn(7,2);
auto a = param(A, "A");
auto b = param(B, "B");
Tensor C = Tensor::randn(8,8);
Tensor D = Tensor::randn(8,8);
auto c = param(C, "C");
auto d = param(D, "D");

Tensor Yt(8, 8);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> pick(0, 2 - 1);
    for (int i = 0; i < 8; ++i) {
        int k = pick(gen);
        for (int j = 0; j < 8; ++j) Yt(i, j) = (j == k) ? 1.f : 0.f;
    }
    Value W = constant(Yt, "Y");


auto bias = param(Tensor::zeros(8,8), "bias");

    auto y =   matmul(a,b); // [2,2]
    auto q=y;
std::cout << "y = " << y.val()
<<","<< endl<< "A = " << a.val()
<<","<< endl<< "B = " << b.val()<<","<< endl
<< "bias = " << bias.val() << endl<< "q = " << q.val() << endl;
std::cout << "y grad " << y.grad() << endl;
std::cout << "dL/dA[0,0] = " << a.grad()
<<","<< endl<< "dL/dB[0,0] = " << b.grad()<<","<< endl
<< "dL/dbias[0,0] = " << bias.grad() << endl<< "dL/dq = " << q.grad() << endl;
zero_grad(y);
backward(y);

std::cout << "y = " << y.val()
<<","<< endl<< "A = " << a.val()
<<","<< endl<< "B = " << b.val()<<","<< endl
<< "bias = " << bias.val() << endl<< "q = " << q.val() << endl;
std::cout << "y grad " << y.grad() << endl;
std::cout << "dL/dA[0,0] = " << a.grad()
<<","<< endl<< "dL/dB[0,0] = " << b.grad()<<","<< endl
<< "dL/dC = " << c.grad() << endl<< "dL/dD = " << d.grad() << endl;



}