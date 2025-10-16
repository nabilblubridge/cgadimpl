// // kernels/cpu/src/agkernels_cpu.cpp
// #include "ad/kernels_api.hpp"
// #include <cstdint>

// extern "C" {

// // ---------------- reference implementations ----------------
//  void relu_impl(const float* x, float* y, int64_t n){
//   for (int64_t i=0;i<n;++i) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
// }

//  void matmul_impl(const float* A, const float* B, float* C,
//                         int M, int K, int N){
//   // C(MxN) = A(MxK) * B(KxN)
//   for (int i=0;i<M;++i){
//     for (int j=0;j<N;++j){
//       float acc = 0.f;
//       const float* Ai = A + i*K;
//       for (int k=0;k<K;++k) acc += Ai[k] * B[k*N + j];  
//       C[i*N + j] = acc;
//     }
//   }
// }

// // ---------------- required export ----------------
// AG_EXPORT int ag_get_cpu_kernels_v1(struct ag_cpu_v1* out){
//   if (!out) return -1;
//   out->abi_version = AG_KERNELS_ABI_V1;
//   out->relu   = &relu_impl;
//   out->matmul = &matmul_impl;
//   return 0;
// }

// } // extern "C"
//===================================================================================================================
#include "ad/kernels_api.hpp"
#include <cstdint>

// Headers for CPU intrinsics (AVX/FMA) and OpenMP
#include <immintrin.h>
#include <omp.h>
#include <iostream>

#include "matker.cuh"


extern "C" {

// ---------------- Optimized Implementations ----------------

/**
 * Optimized ReLU using AVX2 and OpenMP.
 * Processes 8 floats at a time and parallelizes across all cores.
 */
void relu_impl_optimized(const float* x, float* y, int64_t n) {
    const __m256 zeros = _mm256_setzero_ps(); // A vector of 8 zeros

    #pragma omp parallel for
    for (int64_t i = 0; i < n; i += 8) {
        // Ensure we don't read past the end of the array
        if (i + 8 <= n) {
            __m256 x_vec = _mm256_loadu_ps(x + i);      // Load 8 floats from x
            __m256 y_vec = _mm256_max_ps(x_vec, zeros); // Compute max(x, 0) for all 8 floats
            _mm256_storeu_ps(y + i, y_vec);            // Store 8 results back to y
        } else {
            // Handle the remaining elements one by one
            for (int64_t j = i; j < n; ++j) {
                y[j] = x[j] > 0.0f ? x[j] : 0.0f;
            }
        }
    }
}


void relu_cuda(const float* x, float* y, int64_t n) {
    run_cuda_relu(x, y, n);
}

void exp_cuda(const float* x, float* y, int64_t n) {
    run_cuda_exp(x, y, n);
}

void relumask_cuda(const float* x, float* y, int64_t n) {
    run_cuda_relumask(x, y, n);
}

void sigmoid_cuda(const float* x, float* y, int64_t n) {
    run_cuda_sigmoid(x, y, n);
}

void sigmoidiff_cuda(const float* x, float* y, int64_t n) {
    run_cuda_sigmoidiff(x, y, n);
}


void gemm_impl_optimized(const float* A, const float* B,  const float* C, float*E, int M, int K, int N) {
    int q = N;
    int p = K;
    int s = p+q;
    std::cout<<"sddf";
    if(s)
    {
    run_cuda_gemm(A, B, C, E, M, K, N);
    }
}


void matmul_impl_naive(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}


/**
 * Optimized MatMul using AVX2, FMA, OpenMP, and cache blocking.
 * C(MxN) = A(MxK) * B(KxN)
 */
void matmul_impl_optimized(const float* A, const float* B, float* C, int M, int K, int N) {
    // We use tiling to improve cache locality.
    // Tile sizes are chosen to fit in L1/L2 cache.
    // These values may need tuning for specific CPU architectures.
    const int TILE_M = 16;
    const int TILE_N = 16;
    const int TILE_K = 16;

    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < M; i0 += TILE_M) {
        for (int j0 = 0; j0 < N; j0 += TILE_N) {
            // Zero out the C tile we are about to compute
            for (int i = i0; i < i0 + TILE_M && i < M; ++i) {
                for (int j = j0; j < j0 + TILE_N && j < N; ++j) {
                    C[i * N + j] = 0.0f;
                }
            }

            // Accumulate into the C tile
            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                for (int i = i0; i < i0 + TILE_M && i < M; ++i) {
                    for (int k = k0; k < k0 + TILE_K && k < K; ++k) {
                        const float A_val = A[i * K + k];
                        // Broadcast A_val into a vector of 8 identical floats
                        const __m256 a_vec = _mm256_set1_ps(A_val);

                        // Process 8 elements of B and C at a time
                        for (int j = j0; j < j0 + TILE_N && j < N; j += 8) {
                            if (j + 8 <= j0 + TILE_N && j + 8 <= N) {
                                // Load 8 floats from B and C
                                __m256 b_vec = _mm256_loadu_ps(B + k * N + j);
                                __m256 c_vec = _mm256_loadu_ps(C + i * N + j);

                                // Fused Multiply-Add: c_vec = (a_vec * b_vec) + c_vec
                                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);

                                // Store the result back to C
                                _mm256_storeu_ps(C + i * N + j, c_vec);
                            } else {
                                // Handle the remainder (less than 8 elements)
                                for(int j_rem = j; j_rem < j0 + TILE_N && j_rem < N; ++j_rem) {
                                    C[i * N + j_rem] += A_val * B[k * N + j_rem];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void add_cuda(const float* x, const float* z, float* y, int64_t n) {
    run_cuda_add(x, z, y, n);
}

void sub_cuda(const float* x, const float* z, float* y, int64_t n) {
    run_cuda_sub(x, z, y, n);
}

void hadmul_cuda(const float* x, const float* z, float* y, int64_t n) {
    run_cuda_hadmul(x, z, y, n);
}


void flashattention_cuda(const float* Q, const float* K, const float* V,
    float* O, int B, int nh, int N, int d) {
    run_flash_forward(Q, K, V, O, B, nh, N, d);
}

void matmul_impl_cudatile(const float* A, const float* B, float* C, int M, int K, int N) {
    // This is a placeholder for a CUDA-tiled implementation.
    // In a real scenario, this function would offload computation to a GPU.
    // For now, we will just call the naive implementation as a stub.
    int q = N;
    int p = K;
    int s = p+q;
    if(s)
    {
    run_cuda_matrix(A, B, C,  M,  K,  N);
    }
}


// ---------------- required export ----------------
// This part exports the new optimized functions.
AG_EXPORT int ag_get_cpu_kernels_v1(struct ag_cpu_v1* out){
  if (!out) return -1;
  out->abi_version = AG_KERNELS_ABI_V1;
  out->relu   = &relu_cuda;
  out->relumask   = &relumask_cuda;
  out->matmul = &matmul_impl_cudatile;
  out->fmab = &gemm_impl_optimized;
  out->exp = &exp_cuda;
  out->sigmoid = &sigmoid_cuda;
  out->sigmoidiff = &sigmoidiff_cuda;
  out->add = &add_cuda;
  out->sub = &sub_cuda;
  out->hadmul = &hadmul_cuda;
  out->flasha = &flashattention_cuda;
  return 0;
}

} // extern "C"