import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matmul_kernel(float* C, const float* A, const float* B, int M, int N, int K) {
    __shared__ float shared_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_B[TILE_WIDTH][TILE_WIDTH];

    int block_row = blockIdx.y * TILE_WIDTH;
    int block_col = blockIdx.x * TILE_WIDTH;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

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

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }

        __syncthreads(); // Missing syncthreads after shared memory access
    }

    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], sum);
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    if (A.size(1) != B.size(0)) {
        throw std::invalid_argument("Incompatible matrix dimensions");
    }

    auto options = A.options();
    auto C = torch::zeros({M, N}, options);

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1)/TILE_WIDTH, (M + TILE_WIDTH - 1)/TILE_WIDTH);

    matmul_kernel<<<grid, threads>>>(
        C.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        M, N, K
    );

    return C;
}
"""

cpp_src = """
#include <torch/extension.h>
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cuda = load_inline(
    name="matmul_cuda",
    cuda_sources=source,
    cpp_sources=cpp_src,
    functions="matmul_cuda",
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return matmul_cuda.matmul_cuda(A, B)
