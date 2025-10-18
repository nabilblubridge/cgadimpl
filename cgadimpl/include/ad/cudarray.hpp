// cudarray.hpp
#pragma once
void run_cuda_add(const float* a, const float* b, float* c, int n);
void run_cuda_sub(const float* a, const float* b, float* c, int n);
void run_cuda_hadmul(const float* a, const float* b, float* c, int n);
void run_cuda_div(const float* a, const float* b, float* c, int n);
void run_cuda_sigmoidiff(const float* A, float* B, int width);
void run_cuda_sigmoid(const float* A, float* B, int width);


