import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#define TILE_WIDTH 32

__global__ void matmul_optimized_kernel(float* C, const float* A, const float* B, int M, int N, int K) {
    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float shared_B[TILE_WIDTH + 1][TILE_WIDTH]; 

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_row = by * TILE_WIDTH;
    int block_col = bx * TILE_WIDTH;
    int row = block_row + ty;
    int col = block_col + tx;

    float sum = 0.0f;

    for (int m = 0; m < (K + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
        int a_col = m * TILE_WIDTH + tx;
        int a_row = row;
        bool a_valid = (a_row < M) && (a_col < K);
        shared_A[ty][tx] = a_valid ? __ldg(&A[a_row * K + a_col]) : 0.0f;

        int b_row = m * TILE_WIDTH + ty;
        int b_col = block_col + tx;
        bool b_valid = (b_row < K) && (b_col < N);
        shared_B[ty][tx] = b_valid ? __ldg(&B[b_row * N + b_col]) : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k += 8) {
            int k0 = k, k1 = k+1, k2 = k+2, k3 = k+3;
            int k4 = k+4, k5 = k+5, k6 = k+6, k7 = k+7;

            sum = __fma_rn(shared_A[ty][k0], shared_B[k0][tx], sum);
            sum = __fma_rn(shared_A[ty][k1], shared_B[k1][tx], sum);
            sum = __fma_rn(shared_A[ty][k2], shared_B[k2][tx], sum);
            sum = __fma_rn(shared_A[ty][k3], shared_B[k3][tx], sum);
            if (k4 < TILE_WIDTH) {
                sum = __fma_rn(shared_A[ty][k4], shared_B[k4][tx], sum);
                sum = __fma_rn(shared_A[ty][k5], shared_B[k5][tx], sum);
                sum = __fma_rn(shared_A[ty][k6], shared_B[k6][tx], sum);
                sum = __fma_rn(shared_A[ty][k7], shared_B[k7][tx], sum);
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
"""

cpp_src = """
#include <cuda_runtime.h>
#include <torch/extension.h>

at::Tensor matmul_optimized(const at::Tensor A, const at::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    at::Tensor C = at::empty({M, N}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1)/TILE_WIDTH, (M + TILE_WIDTH - 1)/TILE_WIDTH);

    matmul_optimized_kernel<<<blocks, threads>>>(
        C.data_ptr<float>(), 
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        M, N, K
    );

    return C;
}
"""

module = load_inline(
    name='matmul_optimized',
    cpp_sources=cpp_src,
    cuda_sources=source,
    extra_cuda_cflags=['-lineinfo', '-std=c++14', '-arch=sm_75'],
    with_cuda=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, A, B):
        return module.matmul_optimized(A, B)
