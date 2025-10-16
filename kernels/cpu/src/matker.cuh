#pragma once
void run_cuda_matrix(const float* A, const float* B, float* C, int M, int K, int N);
void run_cuda_gemm(const float* A, const float* B,  const float* C, float* E, int M, int K, int N);
void run_cuda_relu(const float* A, float* B, int N);
void run_cuda_relumask(const float* A, float* B, int N);
void run_cuda_exp(const float* A, float* B, int N);
void run_cuda_sigmoid(const float* A, float* B, int N);
void run_cuda_sigmoidiff(const float* A, float* B, int N);
void run_cuda_add(const float* A, const float* B, float* C, int width);
void run_cuda_sub(const float* A, const float* B, float* C, int width);
void run_cuda_hadmul(const float* A, const float* B, float* C, int width);
void run_flash_forward(const float* Q, const float* K, const float* V, float* O, int B, int nh, int N, int d);
