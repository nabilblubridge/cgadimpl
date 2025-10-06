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
Tensor A = Tensor::randn(5,2);
Tensor B = Tensor::randn(3,2);
Tensor C = Tensor::randn(3,2);
Tensor D = Tensor::randn(3,2);
Tensor E = Tensor::randn(2,2);
Tensor F = Tensor::randn(2,2);
Tensor G = Tensor::randn(2,2);
Tensor H = Tensor::randn(2,2);
Tensor I = Tensor::randn(2,2);
Tensor J = Tensor::randn(2,2);
Tensor K = Tensor::randn(2,1);
Tensor X = Tensor::randn(2,2);


auto x = param(X, "X"); 
auto j = param(J, "J");
auto k = param(K, "K"); 
auto a = param(A, "A");
auto b = param(B, "B");
auto c = param(C, "C");
auto d = param(D, "D");
auto e = param(E, "E");
auto f = param(F, "F");
auto g = param(G, "G");
auto h = param(H, "H");
auto i = param(I, "I");
// auto j = param(J, "J");
// auto k = param(K, "K");





auto m = f + g;  // simple add
auto loss = sum(m);
backward(loss);
std::cout << f.grad() << ", " << g.grad() << std::endl;  // should be non-zero


}