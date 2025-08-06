import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void optimized_relu(const float* __restrict__ x, float* __restrict__ y, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        y[i] = fmaxf(x[i], 0.0f);
    }
}

torch::Tensor optimized_relu_cuda(torch::Tensor x) {
    auto x_contig = x.contiguous();
    int n = x_contig.numel();
    auto y = torch::empty_like(x_contig);
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    optimized_relu<<<grid_size, block_size>>>(x_contig.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""

cpp_src = "torch::Tensor optimized_relu_cuda(torch::Tensor x);"

optimized_relu_mod = load_inline(
    name="optimized_relu_mod",
    cuda_sources=source,
    cpp_sources=cpp_src,
    functions=["optimized_relu_cuda"],
    verbose=False,
    extra_cuda_cflags=['-arch=sm_75', '-use_fast_math']
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_op = optimized_relu_mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_op.optimized_relu_cuda(x)
