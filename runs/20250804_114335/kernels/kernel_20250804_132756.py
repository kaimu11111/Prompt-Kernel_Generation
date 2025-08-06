import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matmul_optimized(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float sA0[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sA1[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB0[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB1[TILE_WIDTH][TILE_WIDTH];
    __shared__ bool swap_flag;

    int blockRow = blockIdx.x * TILE_WIDTH;
    int blockCol = blockIdx.y * TILE_WIDTH;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    // Initialize swap_flag once
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        swap_flag = false;
    }
    __syncthreads();

    for (int kStart = 0; kStart < K; kStart += TILE_WIDTH) {
        // Determine current and next shared memory buffers
        float (*current_sA)[TILE_WIDTH] = swap_flag ? sA1 : sA0;
        float (*current_sB)[TILE_WIDTH] = swap_flag ? sB1 : sB0;
        float (*next_sA)[TILE_WIDTH] = swap_flag ? sA0 : sA1;
        float (*next_sB)[TILE_WIDTH] = swap_flag ? sB0 : sB1;

        // Load current tile into current buffers with coalesced access
        int aRow = blockRow + tx;
        int aCol = kStart + ty;
        current_sA[tx][ty] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;

        int bRow = kStart + ty;
        int bCol = blockCol + tx;
        current_sB[ty][tx] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;

        // Preload next tile into next buffers if applicable
        int next_kStart = kStart + TILE_WIDTH;
        if (next_kStart < K) {
            aRow = blockRow + tx;
            aCol = next_kStart + ty;
            next_sA[tx][ty] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;

            bRow = next_kStart + ty;
            bCol = blockCol + tx;
            next_sB[ty][tx] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        }

        __syncthreads();

        // Unrolled computation of partial sums
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += current_sA[tx][i] * current_sB[i][ty];
        }

        // Toggle buffer and synchronize before next iteration
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            swap_flag = !swap_flag;
        }
        __syncthreads();
    }

    // Write result with boundary check
    int cRow = blockRow + tx;
    int cCol = blockCol + ty;
    if (cRow < M && cCol < N) {
        C[cRow * N + cCol] = sum;
    }
}

// Host wrapper with optimized grid configuration
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible matrix dimensions.");

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
    name="matmul",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cuda_cflags=["-std=c++17", "-Xptxas=-v"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = matmul_op.matmul_cuda  # Access the function from module

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul(A, B)
