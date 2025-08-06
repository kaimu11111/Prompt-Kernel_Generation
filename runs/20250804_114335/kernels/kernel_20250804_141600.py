import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matmul_optimized(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float sA0[TILE_WIDTH][TILE_WIDTH] __align__(16);
    __shared__ float sA1[TILE_WIDTH][TILE_WIDTH] __align__(16);
    __shared__ float sB0[TILE_WIDTH][TILE_WIDTH] __align__(16);
    __shared__ float sB1[TILE_WIDTH][TILE_WIDTH] __align__(16);
    __shared__ bool swap_flag;

    int blockRow = blockIdx.x * TILE_WIDTH;
    int blockCol = blockIdx.y * TILE_WIDTH;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        swap_flag = false;
    }
    __syncthreads();

    for (int kStart = 0; kStart < K; kStart += TILE_WIDTH) {
        float (*current_sA)[TILE_WIDTH] = swap_flag ? sA1 : sA0;
        float (*current_sB)[TILE_WIDTH] = swap_flag ? sB1 : sB0;
        float (*next_sA)[TILE_WIDTH] = swap_flag ? sA0 : sA1;
        float (*next_sB)[TILE_WIDTH] = swap_flag ? sB0 : sB1;

        // Load current tiles
        int aRow = blockRow + tx;
        int aCol = kStart + ty;
        current_sA[tx][ty] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;

        int bRow = kStart + tx;
        int bCol = blockCol + ty;
        current_sB[tx][ty] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;

        // Preload next tiles if possible
        if (kStart + TILE_WIDTH < K) {
            aRow = blockRow + tx;
            aCol = kStart + TILE_WIDTH + ty;
            next_sA[tx][ty] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;

            bRow = kStart + TILE_WIDTH + tx;
            bCol = blockCol + ty;
            next_sB[tx][ty] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        }

        __syncthreads();

        // Unrolled computation
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += current_sA[tx][i] * current_sB[i][ty];
        }

        // Toggle buffer
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            swap_flag = !swap_flag;
        }
        __syncthreads();
    }

    int cRow = blockRow + tx;
    int cCol = blockCol + ty;
    if (cRow < M && cCol < N) {
        C[cRow * N + cCol] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(A.size(1) == B.size(0), "Matrix dimensions mismatch.");

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks(
        (M + TILE_WIDTH - 1) / TILE_WIDTH,
        (N + TILE_WIDTH - 1) / TILE_WIDTH
    );

    matmul_optimized<<<blocks, threads>>>(
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
    extra_cuda_cflags=["-std=c++17"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = matmul_op.matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul(A, B)
