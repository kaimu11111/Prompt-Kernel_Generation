import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void optimized_relu(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process main float4 chunks with coalesced access
    int num_float4 = n / 4;
    for (; idx < num_float4; idx += stride) {
        int base = idx * 4;
        float4 val = ((float4*)x)[idx];
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        ((float4*)y)[idx] = val;
    }

    // Process residual elements (non-divergent approach)
    int residual_start = num_float4 * 4;
    int residual = n - residual_start;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < residual; i += stride) {
        int pos = residual_start + i;
        y[pos] = fmaxf(x[pos], 0.0f);
    }
}

torch::Tensor optimized_relu_cuda(torch::Tensor x) {
    int n = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    int num_float4 = n / 4;
    const int grid_size = (num_float4 + block_size - 1) / block_size;

    optimized_relu<<<grid_size, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""

cpp_src = "torch::Tensor optimized_relu_cuda(torch::Tensor x);"

optimized_relu = load_inline(
    name="optimized_relu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["optimized_relu_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_op = optimized_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_op.optimized_relu_cuda(x)
