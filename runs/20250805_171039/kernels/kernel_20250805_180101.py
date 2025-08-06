import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_WIDTH 32

__global__ void optimized_matmul_kernel(float* C, const float* A, const float* B, int M, int N, int K) {
    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

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
        shared_A[ty][tx] = a_valid ? A[a_row * K + a_col] : 0.0f;

        int b_row = m * TILE_WIDTH + ty;
        int b_col = col;
        bool b_valid = (b_row < K) && (b_col < N);
        shared_B[tx][ty] = b_valid ? B[b_row * N + b_col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k += 4) {
            int k0 = k;
            int k1 = k + 1;
            int k2 = k + 2;
            int k3 = k + 3;

            sum = __fma__(shared_A[ty][k0], shared_B[tx][k0], sum);
            sum = __fma__(shared_A[ty][k1], shared_B[tx][k1], sum);
            sum = __fma__(shared_A[ty][k2], shared_B[tx][k2], sum);
            sum = __fma__(shared_A[ty][k3], shared_B[tx][k3], sum);
        }
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_optimized_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K_A = A.size(1);
    int K_B = B.size(0);
    int N = B.size(1);

    if (K_A != K_B) {
        throw std::invalid_argument("Incompatible matrix dimensions");
    }

    auto C = torch::zeros({M, N}, A.options());
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1)/TILE_WIDTH, (M + TILE_WIDTH - 1)/TILE_WIDTH);

    optimized_matmul_kernel<<<grid, threads>>>(C.data_ptr<float>(),
                                              A.data_ptr<float>(),
                                              B.data_ptr<float>(),
                                              M, N, K_A);
    return C;
}
"""

cpp_src = (
    "torch::Tensor matmul_optimized_cuda(torch::Tensor A, torch::Tensor B);"
)

matmul_optimized = load_inline(
    name="matmul_optimized",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["matmul_optimized_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-use_fast_math"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_optimized = matmul_optimized

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_optimized.matmul_optimized_cuda(A, B)
