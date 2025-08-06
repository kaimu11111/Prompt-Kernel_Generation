import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matmul_optimized_kernel(float* C, const float* A, const float* B, int M, int N, int K) {
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
        int b_row = m * TILE_WIDTH + ty;

        if (row < M && a_col < K) {
            shared_A[ty][tx] = A[row * K + a_col];
        } else {
            shared_A[ty][tx] = 0.0f;
        }

        if (b_row < K && col < N) {
            shared_B[ty][tx] = B[b_row * N + col];
        } else {
            shared_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute the partial sum
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }

        // Removed the __syncthreads() after the loop
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

    auto C = torch::empty({M, N}, A.options());
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1)/TILE_WIDTH, (M + TILE_WIDTH - 1)/TILE_WIDTH);

    matmul_optimized_kernel<<<grid, threads>>>(
        C.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        M, N, K_A
    );

    return C;
}
"""

cpp_src = (
    "torch::Tensor matmul_optimized_cuda(torch::Tensor A, torch::Tensor B);"
)

# Compile the inline CUDA code
matmul_optimized = load_inline(
    name="matmul_optimized",
    cuda_sources=source,
    cpp_sources=cpp_src,
    functions=["matmul_optimized_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matmul_optimized = matmul_optimized  # The loaded module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_optimized.matmul_optimized_cuda(A, B)
