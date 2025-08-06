import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>  // for std::min

#define TB 32
#define TW 32  // TW must be a multiple of TB for this implementation

__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_row = blockIdx.y * TB;
    int block_col = blockIdx.x * TB;

    int row = block_row + ty;
    int col = block_col + tx;

    float value = 0.0f;

    for (int tile = 0; tile < (K + TW - 1) / TW; ++tile) {
        int k_start = tile * TW;
        int k_end = std::min(k_start + TW, K);

        __shared__ __align__(16) float sA[TB][TW];
        __shared__ __align__(16) float sB[TW][TB];

        // Load A into shared memory
        if (row < M && k_start + tx < K) {
            sA[ty][tx] = A[row * K + (k_start + tx)];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load B into shared memory
        if (k_start + ty < K && col < N) {
            sB[ty][tx] = B[(k_start + ty) * N + col];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum with unrolled loop
        #pragma unroll
        for (int k = 0; k < (k_end - k_start); ++k) {
            value += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();  // Maintain synchronization between tiles
    }

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

    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

cpp_src = """
#include <torch/extension.h>
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cuda_cflags=["-std=c++14"],
    extra_cflags=["-std=c++14"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul(A, B)
