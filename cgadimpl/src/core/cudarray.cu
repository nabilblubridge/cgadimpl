#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

// =====================================================
// Elementwise Addition
// =====================================================


__global__ void adding_cuda(const float* A, const float* B, float* C, int width)
{
   int bx = blockIdx.x;
    int tx = threadIdx.x;

    int row = bx * blockDim.x + tx;




    // Accumulate into existing C value instead of overwriting
    if(row<width)
                C[row] = A[row] + B[row];
}














void run_cuda_add(const float* A, const float* B, float* C, int width)
{
    int threads = 256;
    int blocks = (width + threads - 1) / threads;

    adding_cuda<<<blocks, threads>>>(A, B, C, width);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: "
                  << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
}



// =====================================================
// Elementwise Subtraction
// =====================================================
__global__ void subbing_cuda(const float* A, const float* B, float* C, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width)
        C[idx] = A[idx] - B[idx];
}

 void run_cuda_sub(const float* A, const float* B, float* C, int width)
{
    int threads = 256;
    int blocks = (width + threads - 1) / threads;

    subbing_cuda<<<blocks, threads>>>(A, B, C, width);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA SUB kernel launch error: "
                  << cudaGetErrorString(err) << std::endl;

    cudaDeviceSynchronize();
}


// =====================================================
// Elementwise Division
// =====================================================
__global__ void diving_cuda(const float* A, const float* B, float* C, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width)
        C[idx] = B[idx] != 0.0f ? A[idx] / B[idx] : 0.0f; // safe divide
}

 void run_cuda_div(const float* A, const float* B, float* C, int width)
{
    int threads = 256;
    int blocks = (width + threads - 1) / threads;

    diving_cuda<<<blocks, threads>>>(A, B, C, width);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA DIV kernel launch error: "
                  << cudaGetErrorString(err) << std::endl;

    cudaDeviceSynchronize();
}


// =====================================================
// Elementwise Multiplication (Hadamard Product)
// =====================================================
__global__ void muling_cuda(const float* A, const float* B, float* C, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width)
        C[idx] = A[idx] * B[idx];
}

 void run_cuda_hadmul(const float* A, const float* B, float* C, int width)
{
    int threads = 256;
    int blocks = (width + threads - 1) / threads;

    muling_cuda<<<blocks, threads>>>(A, B, C, width);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA HADMUL kernel launch error: "
                  << cudaGetErrorString(err) << std::endl;

    cudaDeviceSynchronize();
}





__global__ void sigmoid_thread(const float* A, float* B, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width)
        B[idx] = (1.0f + tanhf(A[idx] / 2.0f)) / 2.0f;
}

void run_cuda_sigmoid(const float* A, float* B, int width)
{
    int threads = 256;
    int blocks = (width + threads - 1) / threads;

    sigmoid_thread<<<blocks, threads>>>(A, B, width);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA sigmoid launch error: "
                  << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
}





__global__ void sigmoidiff_thread(const float* A, float* B, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width) {
        float t = tanhf(A[idx] * 0.5f);
        B[idx] = 0.25f * (1.0f - (t * t));
    }
}

void run_cuda_sigmoidiff(const float* A, float* B, int width)
{
    int threads = 256;
    int blocks = (width + threads - 1) / threads;

    sigmoidiff_thread<<<blocks, threads>>>(A, B, width);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA sigmoid diff launch error: "
                  << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();
}
