#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>


#define TILE 8

__global__ void tile_matrix_multiply(float* A, float* B, float* C, int M, int N, int K)
{
    __shared__ float shareA[TILE][TILE];
    __shared__ float shareB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    // Loop over tiles of K dimension
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        // Load tiles into shared memory (using read-only cache intrinsics)
        shareA[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? __ldg(&A[row * K + a_col]) : 0.0f;

        shareB[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? __ldg(&B[b_row * N + col]) : 0.0f;

        __syncthreads();

        // Multiply the tiles (fmaf intrinsic for fused multiply-add)
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc = fmaf(shareA[threadIdx.y][k], shareB[k][threadIdx.x], acc);

        __syncthreads();
    }

    // Write result only inside bounds
    if (row < M && col < N)
        C[row * N + col] = acc;
}



void run_cuda_matrix(const float* A, const float* B, float* C, int M, int K, int N)
{
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, N*K*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));

    cudaMemcpy(d_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*K*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE, TILE);
    dim3 numBlocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    tile_matrix_multiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}



__global__ void tile_gemm(float* A, float* B, float* C, int M, int K, int N)
{
   __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc = fmaf(As[ty][k], Bs[k][tx], acc);

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = fmaf(1.0f, acc, C[row * N + col]);



}


void run_cuda_gemm(const float* A, const float* B,  const float* C, float* E, int M, int K, int N)
{
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE, TILE);
    dim3 numBlocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    tile_gemm<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(E, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}


__global__ void relu_thread(const float* A, float* B, int width)
{

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int row = bx * blockDim.x + tx;

    float acc = 0.0f;

    

    // Accumulate into existing C value instead of overwriting
    if(row<width)
                B[row] = A[row] > 0.0f ? A[row] : 0.0f;
}



__global__ void exp_thread(const float* A, float* B, int width)
{

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int row = bx * blockDim.x + tx;

    float acc = 0.0f;

    

    // Accumulate into existing C value instead of overwriting
    if(row<width)
                B[row] =  __expf( A[row]);
}



void run_cuda_exp(const float* A, float* B, int width)
{
    float *d_A, *d_B;
    int size = width * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    int threads = 1024;

    dim3 threadsPerBlock(threads);
    dim3 numBlocks((width + threads - 1) / threads);

    exp_thread<<<numBlocks, threadsPerBlock>>>(d_A, d_B, width);
    cudaDeviceSynchronize();

        cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);



    cudaFree(d_A);
    cudaFree(d_B);
}


void run_cuda_relu(const float* A, float* B, int width)
{
    float *d_A, *d_B;
    int size = width * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    int threads = 1024;

    dim3 threadsPerBlock(threads);
    dim3 numBlocks((width + threads - 1) / threads);

    relu_thread<<<numBlocks, threadsPerBlock>>>(d_A, d_B, width);
    cudaDeviceSynchronize();

        cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);



    cudaFree(d_A);
    cudaFree(d_B);
}


__global__ void relumask_thread(const float* A, float* B, int width)
{

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int row = bx * blockDim.x + tx;

    float acc = 0.0f;

    

    // Accumulate into existing C value instead of overwriting
    if(row<width)
               { B[row] = A[row] > 0.0f ? 1.0f : 0.0f;
                printf("Block %d Thread %d active for row %d\n", blockIdx.x, threadIdx.x, row);}
}


void run_cuda_relumask(const float* A, float* B, int width)
{
    float *d_A, *d_B;
    int size = width * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    int threads = 1024;

    dim3 threadsPerBlock(threads);
    dim3 numBlocks((width + threads - 1) / threads);

    relumask_thread<<<numBlocks, threadsPerBlock>>>(d_A, d_B, width);
    cudaDeviceSynchronize();

        cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);



    cudaFree(d_A);
    cudaFree(d_B);
}




// int main()
// {
//     const int width = 4;
//     const int size = width * width * sizeof(float);

//     float h_A[width * width] = {
//         1, 2, 3, 4,
//         5, 6, 7, 8,
//         9, 10, 11, 12,
//         13, 14, 15, 16
//     };

//     float h_B[width * width] = {
//         1, 0, 0, 0,
//         0, 1, 0, 0,
//         0, 0, 1, 0,
//         0, 0, 0, 1
//     };

//     float h_C[width * width] = {0};

//     float *d_A, *d_B, *d_C;
//     cudaMalloc(&d_A, size);
//     cudaMalloc(&d_B, size);
//     cudaMalloc(&d_C, size);

//     cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

//     dim3 threadsPerBlock(TILE, TILE);           // 4×4 threads per block
//     dim3 numBlocks(width / TILE, width / TILE); // 1×1 for 4×4 input

//     tile_matrix_multiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, width);
//     cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

//     cudaDeviceSynchronize();


//     std::cout << "Result matrix C:\n";
//     for (int i = 0; i < width; ++i) {
//         for (int j = 0; j < width; ++j)
//             std::cout << h_C[i * width + j] << " ";
//         std::cout << "\n";
//     }

//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);

//     return 0;
// }
