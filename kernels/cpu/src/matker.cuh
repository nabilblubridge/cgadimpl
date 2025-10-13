#pragma once
void run_cuda_matrix(const float* A, const float* B, float* C, int M, int K, int N);
void run_cuda_gemm(const float* A, const float* B,  const float* C, float* E, int M, int K, int N);
void run_cuda_relu(const float* A, float* B, int N);
void run_cuda_relumask(const float* A, float* B, int N);