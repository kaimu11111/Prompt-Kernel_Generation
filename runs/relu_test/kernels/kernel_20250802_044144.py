import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void optimized_relu(const float* x, float* y, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_full = n /4;
    int residual = n %4;

    // Process full chunks with vectorized loads using __ldg
    for (int i = tid; i < num_full; i += blockDim.x * gridDim.x) {
        float4 val = __ldg(reinterpret_cast<const float4*>(x) + i);
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        reinterpret_cast<float4*>(y)[i] = val;
    }

    // Handle residual elements with grid-strided loop
    if (residual > 0) {
        int start = num_full *4;
        for (int i = tid; i < residual; i += blockDim.x * gridDim.x) {
            y[start + i] = fmaxf(x[start + i], 0.0f);
        }
    }
}

torch::Tensor optimized_relu_cuda(torch::Tensor x) {
    int n = x.numel();
    auto y = torch::empty_like(x);
    const int block_size = 256;
    const int num_full = n /4;
    const int grid_size = (num_full + block_size - 1) / block_size;

    optimized_relu<<<grid_size, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
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
