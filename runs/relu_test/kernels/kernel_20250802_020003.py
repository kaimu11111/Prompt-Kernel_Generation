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

    // Process full 4-element chunks using vectorized loads/stores
    if (tid < num_full) {
        int base = tid *4;
        float4 val = ((float4*)x)[tid];
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        ((float4*)y)[tid] = val;
    }

    // Process residual elements with coalesced thread loops
    int start = num_full *4;
    for (int i = tid; i < residual; i += blockDim.x * gridDim.x) {
        int pos = start + i;
        y[pos] = fmaxf(x[pos], 0.0f);
    }
}

torch::Tensor optimized_relu_cuda(torch::Tensor x) {
    int n = x.numel();
    auto y = torch::empty_like(x);

    int num_full = n /4;
    const int block_size = 256;
    const int grid_size = (num_full + block_size - 1) / block_size;

    optimized_relu<<<grid_size, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}
"""

cpp_src = "torch::Tensor optimized_relu_cuda(torch::Tensor x);"

optimized_relu_mod = load_inline(
    name="optimized_relu_mod",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["optimized_relu_cuda"],
    verbose=True,
    extra_cflags=["-arch=sm_75"],
    extra_ldflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_cuda = optimized_relu_mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_cuda.optimized_relu_cuda(x)
