#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>




__global__ void adding_cuda(const float* A, const float* B, float* C, int width)
{
   int bx = blockIdx.x;
    int tx = threadIdx.x;

    int row = bx * blockDim.x + tx;

    float acc = 0.0f;



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



