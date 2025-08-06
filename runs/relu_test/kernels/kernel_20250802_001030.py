import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void new_relu(const float* x, float* y, int n) {
    int base = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    for (int i = 0; i < 4; ++i) {
        int idx = base + i * blockDim.x;
        if (idx < n) {
            y[idx] = fmaxf(x[idx], 0.0f);
        }
    }
}

torch::Tensor new_relu_cuda(torch::Tensor x) {
    int n = x.numel();
    auto y = torch::empty_like(x);
    const int block_size = 512;
    const int grid_size = (n + (block_size * 4 - 1)) / (block_size * 4);
    new_relu<<<grid_size, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""

cpp_src = (
    "torch::Tensor new_relu_cuda(torch::Tensor x);"
)

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
