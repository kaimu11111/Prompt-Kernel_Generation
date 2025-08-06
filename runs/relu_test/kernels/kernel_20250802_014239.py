import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void optimized_relu(const float* x, float* y, int n) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    int block_elements = block_size * 4;
    int block_start = bid * block_elements;
    int block_end = min(block_start + block_elements, n);

    // Load contiguous chunk into shared memory
    for (int i = tid; i < block_elements; i += block_size) {
        int pos = block_start + i;
        if (pos < n) {
            shared[i] = x[pos];
        } else {
            shared[i] = 0.0f;
        }
    }
    __syncthreads();

    // Process in shared memory (coalesced access)
    for (int i = tid; i < block_elements; i += block_size) {
        if (block_start + i < n) {
            shared[i] = fmaxf(shared[i], 0.0f);
        }
    }
    __syncthreads();

    // Write back to global memory
    for (int i = tid; i < block_elements; i += block_size) {
        int pos = block_start + i;
        if (pos < n) {
            y[pos] = shared[i];
        }
    }
}

torch::Tensor optimized_relu_cuda(torch::Tensor x) {
    int n = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    int block_elements = block_size * 4;
    int grid_size = (n + block_elements - 1) / block_elements;
    int shared_size = block_elements * sizeof(float);

    optimized_relu<<<grid_size, block_size, shared_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n
    );

    return y;
}
"""

cpp_src = "torch::Tensor optimized_relu_cuda(torch::Tensor x);"

optimized_relu_mod = load_inline(
    name="optimized_relu_mod",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["optimized_relu_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_cuda = optimized_relu_mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_cuda.optimized_relu_cuda(x)
