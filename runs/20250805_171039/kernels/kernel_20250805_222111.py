import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matmul_custom_kernel(float* C, const float* A, const float* B, int M, int N, int K) {
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
        // Load A into shared memory with row-major using __ldg
        int a_col = m * TILE_WIDTH + tx;
        int a_row = row;
        bool a_valid = (a_row < M) && (a_col < K);
        shared_A[ty][tx] = a_valid ? __ldg(&A[a_row * K + a_col]) : 0.0f;

        // Transpose B to column-major in shared memory
        int b_row = m * TILE_WIDTH + ty;
        int b_col = col;
        bool b_valid = (b_row < K) && (b_col < N);
        shared_B[tx][ty] = b_valid ? __ldg(&B[b_row * N + b_col]) : 0.0f;

        __syncthreads();

        // 4-way unrolled FMA computation using __fma_rn
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k += 4) {
            int k0 = k, k1 = k+1, k2 = k+2, k3 = k+3;
            sum = __fma_rn(shared_A[ty][k0], shared_B[k0][tx], sum);
            if (k1 < TILE_WIDTH) sum = __fma_rn(shared_A[ty][k1], shared_B[k1][tx], sum);
            if (k2 < TILE_WIDTH) sum = __fma_rn(shared_A[ty][k2], shared_B[k2][tx], sum);
            if (k3 < TILE_WIDTH) sum = __fma_rn(shared_A[ty][k3], shared_B[k3][tx], sum);
        }
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_custom_cuda(torch::Tensor A, torch::Tensor B) {
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

    matmul_custom_kernel<<<grid, threads>>>(
        C.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        M, N, K_A
    );

    return C;
}
"""

cpp_src = (
    "torch::Tensor matmul_custom_cuda(torch::Tensor A, torch::Tensor B);"
)

matmul_custom = load_inline(
    name="matmul_custom",
    cuda_sources=source,
    cpp_sources=cpp_src,
    functions=["matmul_custom_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3", "--use_fast_math", "--expt-relaxed-constexpr", "--maxrregcount=255"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_custom = matmul_custom

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_custom.matmul_custom_cuda(A, B)
