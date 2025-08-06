import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void new_relu(const float* x, float* y, int n) {
    int base = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    int stride = blockDim.x * gridDim.x * 4;
    for (int idx = base; idx < n; idx += stride) {
        {
            int pos0 = idx + 0 * blockDim.x;
            if (pos0 < n) y[pos0] = fmaxf(x[pos0], 0.0f);
            int pos1 = idx + 1 * blockDim.x;
            if (pos1 < n) y[pos1] = fmaxf(x[pos1], 0.0f);
            int pos2 = idx + 2 * blockDim.x;
            if (pos2 < n) y[pos2] = fmaxf(x[pos2], 0.0f);
            int pos3 = idx + 3 * blockDim.x;
            if (pos3 < n) y[pos3] = fmaxf(x[pos3], 0.0f);
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
