#include <iostream>
#include <random>
#include "ad/ag_all.hpp"
#include "ad/export_hlo.hpp"
#include "ad/optim.hpp"

#include <iomanip>
#include <fstream>
#include <filesystem>
// ==========================================================================
// CSV dump utilities
// ==========================================================================
// 
// static void write_csv_tensor(const ag::Tensor& T,
//                              const std::string& filepath,
//                              int precision = 6)
// {
//     std::filesystem::create_directories(std::filesystem::path(filepath).parent_path());
//     std::ofstream out(filepath);
//     if (!out) { std::cerr << "Failed to open " << filepath << "\n"; return; }
//     out << std::fixed << std::setprecision(precision);
//     for (int i = 0; i < T.rows(); ++i) {
//         for (int j = 0; j < T.cols(); ++j) {
//             if (j) out << ',';
//             out << T(i, j);
//         }
//         out << '\n';
//     }
//     out.close();
//     std::cout << "Wrote " << T.rows() << "x" << T.cols()
//               << " CSV to: " << filepath << "\n";
// }

// // Convenience wrappers for ag::Value
// static void dump_csv_val (const std::string& label, const ag::Value& v,
//                           const std::string& dir = "build/dumps")
// {
//     write_csv_tensor(v.val(),  dir + "/" + label + ".csv");
// }
// static void dump_csv_grad(const std::string& label,  ag::Value& v,
//                           const std::string& dir = "build/dumps")
// {
//     write_csv_tensor(v.grad(), dir + "/" + label + "_grad.csv");
// }


// ==========================================================================
// Pretty-print utilities
// ==========================================================================

static void print_tensor(const std::string& label,
                         const ag::Tensor& T,
                         int max_r = 6, int max_c = 8,
                         int width = 10, int precision = 4) {
    std::cout << label << " [" << T.rows() << "x" << T.cols() << "]\n";
    const int R = std::min(T.rows(), max_r);
    const int C = std::min(T.cols(), max_c);
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            std::cout << std::setw(width) << std::fixed
                      << std::setprecision(precision) << T(i,j);
        }
        if (C < T.cols()) std::cout << "  ...";
        std::cout << "\n";
    }
    if (R < T.rows()) std::cout << "...\n";
    std::cout << std::endl;
}

static void print_value(const std::string& label,
                        const ag::Value& v,
                        int max_r = 6, int max_c = 8) {
    print_tensor(label, v.val(), max_r, max_c);
}
static void print_grad(const std::string& label, ag::Value& v,  int max_r = 6, int max_c = 8) {
    print_tensor(label + ".grad", v.grad(), max_r, max_c);
}

// ==========================================================================
// MLP utilities
// ==========================================================================

using namespace ag;



// Mean-squared error over all elements (batch & classes)
static inline ag::Value mse_loss(const ag::Value& pred, const ag::Value& target) {
    ag::Value diff = pred - target;
    ag::Value sq   = diff * diff;               // elementwise
    ag::Value s    = sum(sq);                   // scalar [1,1]
    int B = pred.shape().first, C = pred.shape().second;
    ag::Tensor scale = ag::Tensor::ones(1,1);
    scale(0,0) = 1.0f / float(B * C);
    return s * constant(scale);                 // broadcast scalar
}



int main(){
using namespace std;
using namespace ag;
Tensor A = Tensor::randn(2,2);
Tensor B = Tensor::randn(2,2);
Tensor C = Tensor::randn(2,2);
Tensor D = Tensor::randn(2,2);
Tensor E = Tensor::randn(2,2);
Tensor F = Tensor::randn(2,2);
Tensor G = Tensor::randn(2,2);
Tensor H = Tensor::randn(2,2);
Tensor I = Tensor::randn(2,2);
Tensor J = Tensor::randn(2,2);
Tensor K = Tensor::randn(2,1);
Tensor L = Tensor::randn(2,2);

auto a = param(A, "A");
auto b = param(B, "B");
auto c = param(C, "C");
auto d = param(D, "D");
auto e = param(E, "E");
auto f = param(F, "F");
auto g = param(G, "G");
auto h = param(H, "H");
auto i = param(I, "I");
auto j = param(J, "J");
auto k = param(K, "K");
auto l = param(L, "L");


auto q = alibiatt(rms(a), b, c, d, 0.125) + a; // scalar, tests broadcasting [B,2] + [1,2]
auto p = swiglu(rms(q), e, f, g, h) + q;
auto y = sum(softmax_row(fmab(rms(p), i, k)));

// --- BEFORE backward ---
std::cout << "Before backward:" << std::endl;

std::cout << "y = " << y.val() << std::endl;
std::cout << "q = " << q.val() << std::endl;
std::cout << "p = " << p.val() << std::endl;

std::cout << "dL/dA = " << a.grad() << std::endl;
std::cout << "dL/dB = " << b.grad() << std::endl;
std::cout << "dL/dC = " << c.grad() << std::endl;
std::cout << "dL/dD = " << d.grad() << std::endl;
std::cout << "dL/dE = " << e.grad() << std::endl;
std::cout << "dL/dF = " << f.grad() << std::endl;
std::cout << "dL/dG = " << g.grad() << std::endl;
std::cout << "dL/dH = " << h.grad() << std::endl;
std::cout << "dL/dI = " << i.grad() << std::endl;
std::cout << "dL/dK = " << k.grad() << std::endl;

std::cout << "A = " << a.val() << std::endl;
std::cout << "B = " << b.val() << std::endl;
std::cout << "C = " << c.val() << std::endl;
std::cout << "D = " << d.val() << std::endl;
std::cout << "E = " << e.val() << std::endl;
std::cout << "F = " << f.val() << std::endl;
std::cout << "G = " << g.val() << std::endl;
std::cout << "H = " << h.val() << std::endl;
std::cout << "I = " << i.val() << std::endl;
std::cout << "K = " << k.val() << std::endl;

zero_grad(y);
backward(y);

// --- AFTER backward ---
std::cout << "\nAfter backward:" << std::endl;

std::cout << "y = " << y.val() << std::endl;
std::cout << "q = " << q.val() << std::endl;
std::cout << "p = " << p.val() << std::endl;

std::cout << "dL/dA = " << a.grad() << std::endl;
std::cout << "dL/dB = " << b.grad() << std::endl;
std::cout << "dL/dC = " << c.grad() << std::endl;
std::cout << "dL/dD = " << d.grad() << std::endl;
std::cout << "dL/dE = " << e.grad() << std::endl;
std::cout << "dL/dF = " << f.grad() << std::endl;
std::cout << "dL/dG = " << g.grad() << std::endl;
std::cout << "dL/dH = " << h.grad() << std::endl;
std::cout << "dL/dI = " << i.grad() << std::endl;
std::cout << "dL/dK = " << k.grad() << std::endl;

std::cout << "A = " << a.val() << std::endl;
std::cout << "B = " << b.val() << std::endl;
std::cout << "C = " << c.val() << std::endl;
std::cout << "D = " << d.val() << std::endl;
std::cout << "E = " << e.val() << std::endl;
std::cout << "F = " << f.val() << std::endl;
std::cout << "G = " << g.val() << std::endl;
std::cout << "H = " << h.val() << std::endl;
std::cout << "I = " << i.val() << std::endl;
std::cout << "K = " << k.val() << std::endl;

std::cout << "\n\n--- Gradients zeroed for next pass ---\n\n";


std::cout << " \n \n \n";



}