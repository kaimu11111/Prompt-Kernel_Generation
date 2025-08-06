import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

#define TB 32
#define TW 64  // Tile size must be multiple of TB for efficient loading

template <typename T>
__global__ void matmul_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int N, int K
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_row = blockIdx.y * TB;
    int block_col = blockIdx.x * TB;

    int row = block_row + ty;
    int col = block_col + tx;

    T value = 0.0;

    for (int tile = 0; tile < (K + TW - 1) / TW; ++tile) {
        int k_start = tile * TW;
        int k_end = std::min(k_start + TW, K);

        __shared__ T sA[TB][TW];  // Tile A: rows from current block_row, cols from k_start
        __shared__ T sB[TW][TB];  // Tile B: rows from k_start, cols from current block_col

        // Load sA: each thread handles TW/TB columns (e.g., 2 for TW=64/TB=32)
        for (int i = 0; i < (TW / TB); ++i) {
            int k_col = tx * (TW / TB) + i;
            if (k_col < TW) {
                int A_row = block_row + ty;
                int A_col = k_start + k_col;
                sA[ty][k_col] = (A_row < M && A_col < K) ? A[A_row * K + A_col] : T(0);
            }
        }

        // Load sB: each thread handles TW/TB rows
        for (int i = 0; i < (TW / TB); ++i) {
            int k_row = ty * (TW / TB) + i;
            if (k_row < TW) {
                int B_row = k_start + k_row;
                int B_col = block_col + tx;
                sB[k_row][tx] = (B_row < K && B_col < N) ? B[B_row * N + B_col] : T(0);
            }
        }

        __syncthreads();

        // Process current tile: iterate only over valid k elements (k_start to k_end)
        for (int k = 0; k < (k_end - k_start); ++k) {
            value += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();  // Not needed here but kept for potential compiler optimizations
    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "2D tensors required");
    int M = A.size(0);
    int K_A = A.size(1);
    int K_B = B.size(0);
    int N = B.size(1);
    TORCH_CHECK(K_A == K_B, "Incompatible matrix dimensions");
    const int K = K_A;

    auto C = torch::empty({M, N}, A.options());

    dim3 threadsPerBlock(TB, TB);
    int blocks_x = (N + TB - 1) / TB;
    int blocks_y = (M + TB - 1) / TB;
    dim3 blocksPerGrid(blocks_x, blocks_y);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_cuda", ([&] {
        matmul_kernel<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}
"""

cpp_src = (
    "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"
)

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["matmul_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_cuda = matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda.matmul_cuda(A, B)
