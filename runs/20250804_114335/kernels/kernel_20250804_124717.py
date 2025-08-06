import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matmul_optimized(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int blockRow = blockIdx.x * TILE_WIDTH;
    int blockCol = blockIdx.y * TILE_WIDTH;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    for (int kStart = 0; kStart < K; kStart += TILE_WIDTH) {
        // Load tiles with coalesced access and transposed B for bank conflict reduction
        int aRow = blockRow + tx;
        int aCol = kStart + ty;
        sA[tx][ty] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;

        int bRow = kStart + tx; // Corrected tx instead of ty
        int bCol = blockCol + ty; // Corrected ty instead of tx
        sB[ty][tx] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;

        __syncthreads();

        // Compute with unrolled loop for better ILP
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += sA[tx][i] * sB[i][ty];
        }

        // No redundant syncthreads() here
        __syncthreads(); // Added syncthreads after loop iteration
    }

    // Write with boundary check
    int cRow = blockRow + tx;
    int cCol = blockCol + ty;
    if (cRow < M && cCol < N) {
        C[cRow * N + cCol] = sum;
    }
}

// Optimized host wrapper with tile size 32 and ECC-aware grid setup
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions mismatch.");

    auto C = torch::empty({M, N}, A.options());

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(
        (M + TILE_WIDTH - 1) / TILE_WIDTH,
        (N + TILE_WIDTH - 1) / TILE_WIDTH
    );

    matmul_optimized<<<blocksPerGrid, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    return C;
}
"""

cpp_src = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_op = load_inline(
    name="matmul_optimized",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cuda_cflags=["-std=c++17", "--use_fast_math"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.cuda_matmul = matmul_op.matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.cuda_matmul(A, B)
