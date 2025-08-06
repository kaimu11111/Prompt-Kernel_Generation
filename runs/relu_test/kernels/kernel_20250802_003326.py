import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void optimized_relu_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int pos = idx; pos < n; pos += stride) {
        int base = pos;
        y[base] = fmaxf(x[base], 0.0f);
        if (base + 1 < n) y[base + 1] = fmaxf(x[base + 1], 0.0f);
        if (base + 2 < n) y[base + 2] = fmaxf(x[base + 2], 0.0f);
        if (base + 3 < n) y[base + 3] = fmaxf(x[base + 3], 0.0f);
    }
}

torch::Tensor optimized_relu_cuda(torch::Tensor x) {
    auto n = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    optimized_relu_kernel<<<grid_size, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}
"""

cpp_src = (
    "torch::Tensor optimized_relu_cuda(torch::Tensor x);"
)

optimized_relu = load_inline(
    name="optimized_relu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["optimized_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_cuda = optimized_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_cuda.optimized_relu_cuda(x)
