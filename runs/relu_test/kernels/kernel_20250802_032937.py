import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void optimized_relu(const float* x, float* y, int n) {
    int block_start = blockIdx.x * blockDim.x * 4;
    int block_end = min(block_start + blockDim.x * 4, n);
    int num_elements = block_end - block_start;
    int chunk_count = num_elements / 4;

    // Process full chunks of 4 elements using vectorized operations
    for (int i = threadIdx.x; i < chunk_count; i += blockDim.x) {
        int base = block_start + i * 4;
        float4 val = ((float4*)x)[base / 4];
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        ((float4*)y)[base / 4] = val;
    }

    // Handle residual elements within the block
    int residual = num_elements % 4;
    if (residual > 0) {
        for (int i = threadIdx.x; i < residual; i += blockDim.x) {
            int pos = block_start + (chunk_count * 4) + i;
            y[pos] = fmaxf(x[pos], 0.0f);
        }
    }
}

torch::Tensor optimized_relu_cuda(torch::Tensor x) {
    int n = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int grid_size = (n + (block_size * 4 - 1)) / (block_size * 4);

    optimized_relu<<<grid_size, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}
'''

cpp_src = "torch::Tensor optimized_relu_cuda(torch::Tensor x);"

optimized_relu = load_inline(
    name="optimized_relu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["optimized_relu_cuda"],
    verbose=False,
    extra_cuda_cflags=['-arch=sm_75']
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_op = optimized_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_op.optimized_relu_cuda(x)
