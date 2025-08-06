import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

#define TB 16
#define TW 32  // TW must be a multiple of TB for this implementation

template <typename T>
__global__ void matmul_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int M, int N, int K
) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int block_row = blockIdx.y * TB;
    int block_col = blockIdx.x * TB;

    T value = 0.0;

    for (int tile = 0; tile < (K + TW - 1) / TW; ++tile) {
        int k_start = tile * TW;
        int k_end = std::min(k_start + TW, K);

        __shared__ __align__(16) T sA[TB][TW];
        __shared__ __align__(16) T sB[TW][TB];

        // Load A tile into shared memory
        for (int i = 0; i < (TW / TB); ++i) {
            int k_col = tx * (TW / TB) + i;
            if (k_col < TW) {
                int A_row = block_row + ty;
                int A_col = k_start + k_col;
                sA[ty][k_col] = 
                    (A_row < M && A_col < K) ? 
                    A[A_row * K + A_col] : T(0);
            }
        }

        // Load B tile into shared memory with coalesced accesses
        for (int i = 0; i < (TW / TB); ++i) {
            int k_row = tx * (TW / TB) + i;
            if (k_row < TW) {
                int B_row = k_start + k_row;
                int B_col = block_col + tx;
                sB[k_row][tx] = 
                    (B_row < K && B_col < N) ? 
                    B[B_row * N + B_col] : T(0);
            }
        }

        __syncthreads();

        // Compute partial sum with loop unrolling
        #pragma unroll
        for (int k = 0; k < (k_end - k_start); ++k) {
            value += sA[ty][k] * sB[k][tx];
        }

        // __syncthreads();  // Removed unnecessary synchronization here
    }

    // Write result to global memory
    int row = block_row + ty;
    int col = block_col + tx;
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "2D tensors required");
    int M = A.size(0);   // Removed .toInt()
    int K_A = A.size(1); // Removed .toInt()
    int K_B = B.size(0); // Removed .toInt()
    int N = B.size(1);   // Removed .toInt()
    TORCH_CHECK(K_A == K_B, "Incompatible matrix dimensions");
    const int K = K_A;

    auto C = torch::empty({M, N}, A.options());

    dim3 threadsPerBlock(TB, TB);
    int blocks_x = (N + TB - 1) / TB;
    int blocks_y = (M + TB - 1) / TB;
    dim3 blocksPerGrid(blocks_x, blocks_y);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_cuda", ([&] (auto dummy) {
        matmul_kernel<scalar_t><<<blocksPerGrid, threadsPerBlock>>>(
            A.data_ptr<scalar_t>(), 
            B.data_ptr<scalar_t>(), 
            C.data_ptr<scalar_t>(),
            M, N, K);
    }));

    return C;
}
"""

cpp_src = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul = load_inline(
    name="matmul",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = matmul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul.matmul_cuda(A, B)
