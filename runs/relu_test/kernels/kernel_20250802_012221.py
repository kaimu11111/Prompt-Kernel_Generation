import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void new_relu(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process main float4 elements
    for (; idx < (n /4); idx += stride) {
        int base = idx *4;
        float4 val = ((float4*)x)[idx];
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        ((float4*)y)[idx] = val;
    }
    
    // Process residual elements
    int residual_start = (n /4)*4;
    for (int residual_idx = blockIdx.x * blockDim.x + threadIdx.x;
         residual_idx < (n - residual_start);
         residual_idx += stride) {
        int pos = residual_start + residual_idx;
        y[pos] = fmaxf(x[pos], 0.0f);
    }
}

torch::Tensor new_relu_cuda(torch::Tensor x) {
    int n = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 512;
    const int grid_size = (n + block_size - 1) / block_size;

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
