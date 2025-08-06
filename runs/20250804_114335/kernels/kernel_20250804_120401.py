import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH][TILE_WIDTH];

    int blockRow = blockIdx.x * TILE_WIDTH;
    int blockCol = blockIdx.y * TILE_WIDTH;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    for (int kStart = 0; kStart < K; kStart += TILE_WIDTH) {
        // Load tiles into shared memory with coalesced access patterns
        int aRow = blockRow + tx;
        int aCol = kStart + ty;
        sA[tx][ty] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;

        // Transpose sB for column-major storage to reduce bank conflicts
        int bRow = kStart + ty;
        int bCol = blockCol + tx;
        sB[ty][tx] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;

        __syncthreads();

        // Unroll inner loop for better instruction-level parallelism
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += sA[tx][i] * sB[ty][i];
        }
    }

    // Write result to global memory with bounds check
    int cRow = blockRow + tx;
    int cCol = blockCol + ty;
    if (cRow < M && cCol < N) {
        C[cRow * N + cCol] = sum;
    }
}

// Host wrapper function with corrected thread/block configuration
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

    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(
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
        # Correctly bind the CUDA function
        self.matmul = matmul_op.matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul(A, B)
