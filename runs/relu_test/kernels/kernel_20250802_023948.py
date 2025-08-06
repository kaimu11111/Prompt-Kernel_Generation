import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void new_relu(const float* x, float* y, int n) {
    int block_start = blockIdx.x * blockDim.x * 4;
    int block_end = min(block_start + blockDim.x * 4, n);
    int chunk_size = block_end - block_start;

    // Process full 4-element chunks with float4
    for (int i = threadIdx.x; i < (chunk_size / 4); i += blockDim.x) {
        int base = block_start + i * 4;
        float4 val = ((float4*)x)[base / 4];
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        ((float4*)y)[base / 4] = val;
    }

    // Handle residual elements (0-3)
    int residual = chunk_size % 4;
    if (residual > 0) {
        for (int i = threadIdx.x; i < residual; i += blockDim.x) {
            int pos = block_start + (chunk_size / 4) * 4 + i;
            y[pos] = fmaxf(x[pos], 0.0f);
        }
    }
}

torch::Tensor new_relu_cuda(torch::Tensor x) {
    int n = x.numel();
    auto y = torch::empty_like(x);
    const int block_size = 256;
    const int grid_size = (n + (block_size * 4 - 1)) / (block_size * 4);
    new_relu<<<grid_size, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""

cpp_src = "torch::Tensor new_relu_cuda(torch::Tensor x);"

new_relu_mod = load_inline(
    name="new_relu_mod",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["new_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_cuda = new_relu_mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_cuda.new_relu_cuda(x)
