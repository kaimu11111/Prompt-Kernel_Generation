import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void optimized_relu(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; idx < (n + 3) / 4; idx += stride) {
        int base = idx * 4;
        if (base + 4 <= n) {
            float4 val = ((float4*)x)[idx];
            val.x = fmaxf(val.x, 0.0f);
            val.y = fmaxf(val.y, 0.0f);
            val.z = fmaxf(val.z, 0.0f);
            val.w = fmaxf(val.w, 0.0f);
            ((float4*)y)[idx] = val;
        } else {
            int rem = n - base;
            if (rem <= 0) continue;
            y[base] = fmaxf(x[base], 0.0f);
            if (rem >= 2) {
                y[base + 1] = fmaxf(x[base + 1], 0.0f);
                if (rem >= 3) {
                    y[base + 2] = fmaxf(x[base + 2], 0.0f);
                }
            }
        }
    }
}

torch::Tensor optimized_relu_cuda(torch::Tensor x) {
    int n = x.numel();
    auto y = torch::empty_like(x);
    int num_float4 = (n + 3) / 4;
    const int block_size = 256;
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
