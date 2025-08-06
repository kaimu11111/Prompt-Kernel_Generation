import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>  // for std::min

template <typename T>
__global__ void optimized_matmul(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int N, int K
) {
    #define TB 32
    #define TW 64  // TW must be a multiple of TB
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_row = blockIdx.y * TB;
    int block_col = blockIdx.x * TB;

    int row = block_row + ty;
    int col = block_col + tx;

    T value = static_cast<T>(0.0);

    for (int tile = 0; tile < (K + TW - 1) / TW; ++tile) {
        int k_start = tile * TW;
        int k_end = std::min(k_start + TW, K);

        __shared__ T sA[TB][TW];
        __shared__ T sB[TW][TB];

        // Load A tiles with coalesced accesses
        for (int i = 0; i < (TW / TB); ++i) {
            int k_col = tx * (TW / TB) + i;
            if (k_col < TW) {
                int A_row = block_row + ty;
                int A_col = k_start + k_col;
                sA[ty][k_col] = (A_row < M && A_col < K) ? A[A_row * K + A_col] : static_cast<T>(0);
            }
        }

        // Load B tiles with optimal bank access
        for (int i = 0; i < (TW / TB); ++i) {
            int k_row = ty * (TW / TB) + i;
            if (k_row < TW) {
                int B_row = k_start + k_row;
                int B_col = block_col + tx;
                sB[k_row][tx] = (B_row < K && B_col < N) ? B[B_row * N + B_col] : static_cast<T>(0);
            }
        }

        __syncthreads();

        // Compute partial sums with loop unrolling
        for (int k = 0; k < (k_end - k_start); ++k) {
            value += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }

    #undef TB
    #undef TW
}

torch::Tensor matmul_optimized(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    int M = A.size(0);
    int K_A = A.size(1);
    int K_B = B.size(0);
    int N = B.size(1);
    TORCH_CHECK(K_A == K_B, "Incompatible matrix dimensions");
    const int K = K_A;

    auto C = torch::empty({M, N}, A.options());

    dim3 threadsPerBlock(32, 32);
    int blocks_x = (N + 31) / 32;
    int blocks_y = (M + 31) / 32;
    dim3 blocksPerGrid(blocks_x, blocks_y);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_optimized", ([&] {
        optimized_matmul<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K);
    }));

    return C;
}
"""

cpp_src = """
#include <torch/extension.h>
torch::Tensor matmul_optimized(torch::Tensor A, torch::Tensor B);
"""

matmul_optimized = load_inline(
    name="matmul_optimized",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["matmul_optimized"],
    verbose=True,
    extra_cuda_cflags=["-std=c++14", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_op = matmul_optimized

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.forward_op(A, B)
