import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BS 16

__global__ void matrixMultiply(float* C, const float* A, const float* B, int N) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BS + ty;
    int col = bx * BS + tx;

    float Cvalue = 0.0f;

    for (int tile = 0; tile < (N + BS - 1)/BS; ++tile) {
        __shared__ float As[BS][BS];
        __shared__ float Bs[BS][BS];

        // Load A into shared memory
        int aRow = by * BS + ty;
        int aCol = tile * BS + tx;
        if (aRow < N && aCol < N) {
            As[ty][tx] = A[aRow * N + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load B into shared memory
        int bRow = tile * BS + tx;
        int bCol = bx * BS + ty;
        if (bRow < N && bCol < N) {
            Bs[tx][ty] = B[bRow * N + bCol];
        } else {
            Bs[tx][ty] = 0.0f;
        }

        __syncthreads();

        // Compute the dot product for this tile
        for (int k = 0; k < BS; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = Cvalue;
    }
}

torch::Tensor matrix_multiply_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 threads(BS, BS);
    dim3 blocks( (N + BS - 1)/BS, (N + BS - 1)/BS );

    matrixMultiply<<<blocks, threads>>>(C.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), N);

    return C;
}
"""

cpp_src = (
    "torch::Tensor matrix_multiply_cuda(torch::Tensor A, torch::Tensor B);"
)

matrix_multiply_cuda = load_inline(
    name="matrix_multiply",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["matrix_multiply_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix_multiply = matrix_multiply_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matrix_multiply(A, B)
