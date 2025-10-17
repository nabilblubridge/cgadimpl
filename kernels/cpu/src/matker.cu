#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <math_functions.h>

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

__global__ void sigmoid_thread(const float* A, float* B, int width)
{

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int row = bx * blockDim.x + tx;

    float acc = 0.0f;

    

    // Accumulate into existing C value instead of overwriting
    if(row<width)
             //  { 
                B[row] = (1.0f + tanhf(A[row]/ 2.0f)) / 2.0f;
              //  printf("Block %d Thread %d active for row %d\n", blockIdx.x, threadIdx.x, row);}
}

void run_cuda_sigmoid(const float* A, float* B, int width)
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

    sigmoid_thread<<<numBlocks, threadsPerBlock>>>(d_A, d_B, width);
    cudaDeviceSynchronize();

        cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);



    cudaFree(d_A);
    cudaFree(d_B);
}


__global__ void sigmoidiff_thread(const float* A, float* B, int width)
{

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    int row = bx * blockDim.x + tx;

    float acc = 0.0f;

    

    // Accumulate into existing C value instead of overwriting
    if(row<width)
       { 
            float t = tanhf(A[row] * 0.5f);
B[row] = 0.25f * (1.0f -( t * t));
              //  printf("Block %d Thread %d active for row %d\n", blockIdx.x, threadIdx.x, row);
              
            }
}


void run_cuda_sigmoidiff(const float* A, float* B, int width)
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

    sigmoidiff_thread<<<numBlocks, threadsPerBlock>>>(d_A, d_B, width);
    cudaDeviceSynchronize();

        cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);



    cudaFree(d_A);
    cudaFree(d_B);
}



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
    float *d_A, *d_B, *d_C;
    int size = width * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    int threads = 1024;

    dim3 threadsPerBlock(threads);
    dim3 numBlocks((width + threads - 1) / threads);

    adding_cuda<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, width);
    cudaDeviceSynchronize();

        cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);



    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}





__global__ void subbing_cuda(const float* A, const float* B, float* C, int width)
{
   int bx = blockIdx.x;
    int tx = threadIdx.x;

    int row = bx * blockDim.x + tx;

    float acc = 0.0f;



    // Accumulate into existing C value instead of overwriting
    if(row<width)
                C[row] = A[row] - B[row];
}
















void run_cuda_sub(const float* A, const float* B, float* C, int width)
{
    float *d_A, *d_B, *d_C;
    int size = width * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    int threads = 1024;

    dim3 threadsPerBlock(threads);
    dim3 numBlocks((width + threads - 1) / threads);

    subbing_cuda<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, width);
    cudaDeviceSynchronize();

        cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);



    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}






__global__ void muling_cuda(const float* A, const float* B, float* C, int width)
{
   int bx = blockIdx.x;
    int tx = threadIdx.x;

    int row = bx * blockDim.x + tx;

    float acc = 0.0f;



    // Accumulate into existing C value instead of overwriting
    if(row<width)
                C[row] = A[row] * B[row];
}
















void run_cuda_hadmul(const float* A, const float* B, float* C, int width)
{
    float *d_A, *d_B, *d_C;
    int size = width * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    int threads = 1024;

    dim3 threadsPerBlock(threads);
    dim3 numBlocks((width + threads - 1) / threads);

    muling_cuda<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, width);
    cudaDeviceSynchronize();

        cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);



    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}



// --------------------------------------------
// Flash Attention Forward Kernel (Fixed Logic)
// --------------------------------------------
__global__ void flash_forward_kernel(
    const float* Q, const float* K, const float* V, float* O,
    const int N, const int d,
    const int Tc, const int Tr, const int Bc, const int Br,
    float* l, float* m, const float softmax_scale)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y; // batch and head index
    int i = blockIdx.z;
    int j = threadIdx.y;

    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d); // per batch+head
    int lm_offset  = (bx * gridDim.y * N) + (by * N);

    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S  = &sram[tile_size * 3]; // S has shape [Br x Bc]

        int row_idx = i * Br + tx;
        if (row_idx < N) {

        // load Qi
        for (int x = 0; x < d; x++) {
            Qi[tx * d + x] = Q[qkv_offset + row_idx * d + x];
        }

        float row_m_prev = -INFINITY;
        float row_l_prev = 0.0f;
        if (m && l && i < Tr) {
            // if we already processed previous tiles
            row_m_prev = m[lm_offset + row_idx];
            row_l_prev = l[lm_offset + row_idx];
        }

        float row_m_new = row_m_prev;
        float row_l_new = row_l_prev;

        // process each K/V tile
        if (j < Tc) {
            int col_base = j * Bc;
            __syncthreads();

            // Load K/V to SRAM
            if (col_base + tx < N) {
                for (int x = 0; x < d; x++) {
                    Kj[tx * d + x] = K[qkv_offset + (col_base + tx) * d + x];
                    Vj[tx * d + x] = V[qkv_offset + (col_base + tx) * d + x];
                }
            } else {
                for (int x = 0; x < d; x++) {
                    Kj[tx * d + x] = 0.0f;
                    Vj[tx * d + x] = 0.0f;
                }
            }

            __syncthreads();

            // compute attention scores S = QK^T
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0.0f;
                for (int x = 0; x < d; x++) {
                    sum += Qi[tx * d + x] * Kj[y * d + x];
                }
                sum *= softmax_scale;
                S[tx * Bc + y] = sum;
                if (sum > row_m) row_m = sum;
            }

            // compute P = exp(S - row_m)
            float row_l = 0.0f;
            for (int y = 0; y < Bc; y++) {
                float val = __expf(S[tx * Bc + y] - row_m);
                S[tx * Bc + y] = val;
                row_l += val;
            }

            // combine with running max/sum
            float new_m = fmaxf(row_m_prev, row_m);
            float new_l = __expf(row_m_prev - new_m) * row_l_prev +
                          __expf(row_m - new_m) * row_l;

            // accumulate O
            for (int x = 0; x < d; x++) {
                float pv = 0.0f;
                for (int y = 0; y < Bc; y++) {
                    pv += S[tx * Bc + y] * Vj[y * d + x];
                }

                float old_O = (row_l_prev > 0)
                    ? O[qkv_offset + row_idx * d + x]
                    : 0.0f;

                float new_O = (1.0f / new_l) *
                              ( __expf(row_m_prev - new_m) * row_l_prev * old_O +
                                __expf(row_m - new_m) * pv );

                O[qkv_offset + row_idx * d + x] = new_O;
            }

            row_m_prev = new_m;
            row_l_prev = new_l;
            row_m_new  = new_m;
            row_l_new  = new_l;
        }

        // store back updated m and l
        m[lm_offset + row_idx] = row_m_new;
        l[lm_offset + row_idx] = row_l_new;
        __syncthreads();
    }
}

// --------------------------------------------
// Host Launcher
// --------------------------------------------
void run_flash_forward(const float* Q, const float* K, const float* V, float* O,
                       int B, int nh, int N, int d) {
    const int Bc = 32, Br = 32;
    const int Tc = (N + Bc - 1) / Bc;
    const int Tr = (N + Br - 1) / Br;
    const float softmax_scale = 1.0f / sqrtf((float)d);

    size_t qkv_size = B * nh * N * d * sizeof(float);
    size_t lm_size  = B * nh * N * sizeof(float);

    float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m;
    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMalloc(&d_O, qkv_size);
    cudaMalloc(&d_l, lm_size);
    cudaMalloc(&d_m, lm_size);

    cudaMemcpy(d_Q, Q, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, qkv_size, cudaMemcpyHostToDevice);
    cudaMemset(d_O, 0, qkv_size);
    cudaMemset(d_l, 0, lm_size);
    cudaMemset(d_m, 0, lm_size); // init to -inf handled inside kernel

    dim3 grid_dim(B, nh, Tr);
    dim3 block_dim(Br, Tc);
    int shared_mem = (3 * Bc * d + Br * Bc) * sizeof(float);

    flash_forward_kernel<<<grid_dim, block_dim, shared_mem>>>(
        d_Q, d_K, d_V, d_O,
        N, d, Tc, Tr, Bc, Br, d_l, d_m, softmax_scale
    );
    cudaDeviceSynchronize();

    cudaMemcpy(O, d_O, qkv_size, cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_l);
    cudaFree(d_m);
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
