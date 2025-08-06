import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector_types.h>

__global__ void optimized_relu(const float* x, float* y, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_full = n /4;
    int residual = n %4;

    // Process full chunks using vectorized loads/stores
    for (int i = tid; i < num_full; i += blockDim.x * gridDim.x) {
        float4 val = ((float4*)x)[i];
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        ((float4*)y)[i] = val;
    }

    // Process residual elements with contiguous thread assignment
    if (residual > 0) {
        int start = num_full *4;
        for (int i = tid; i < residual; i += blockDim.x * gridDim.x) {
            y[start + i] = fmaxf(x[start + i], 0.0f);
        }
    }
}
'''

cpp_src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" {
    // Declare the kernel without __global__
    void optimized_relu(const float*, float*, int);

    void optimized_relu_forward(const at::Tensor x, at::Tensor y) {
        int n = x.numel();
        int num_full = n /4;
        const int block_size = 128;
        const int grid_size = (num_full + block_size - 1) / block_size;
        optimized_relu<<<grid_size, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
        cudaDeviceSynchronize();
    }
}
'''

optimized_relu_mod = load_inline(
    name="optimized_relu_mod",
    cuda_sources=source,
    cpp_sources=cpp_src,
    functions=["optimized_relu_forward"],
    verbose=False,
    extra_cuda_cflags=['-arch=sm_75']
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.empty_like(x)
        optimized_relu_mod.optimized_relu_forward(x, y)
        return y
