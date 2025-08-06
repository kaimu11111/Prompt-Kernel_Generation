import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void new_relu(const float* x, float* y, int n) {
    extern __shared__ float4 shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    int num_full = n /4;
    int residual = n %4;
    int residual_start = num_full *4;

    // Process full chunks with shared memory
    int block_start = bid * block_size;
    int block_end = min(block_start + block_size, num_full);

    // Load into shared memory
    for (int i = tid; i < (block_end - block_start); i += block_size) {
        int global_idx = block_start + i;
        shared[tid] = ((float4*)x)[global_idx];
    }
    __syncthreads();

    // Process in shared memory
    for (int i = tid; i < (block_end - block_start); i += block_size) {
        float4 val = shared[i];
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        shared[i] = val;
    }
    __syncthreads();

    // Store back
    for (int i = tid; i < (block_end - block_start); i += block_size) {
        int global_idx = block_start + i;
        ((float4*)y)[global_idx] = shared[i];
    }

    // Process residual elements
    if (residual > 0) {
        for (int i = tid; i < residual; i += blockDim.x * gridDim.x) {
            int pos = residual_start + i;
            y[pos] = fmaxf(x[pos], 0.0f);
        }
    }
}

torch::Tensor new_relu_cuda(torch::Tensor x) {
    int n = x.numel();
    auto y = torch::empty_like(x);
    int num_full = n /4;  // Added missing variable definition
    const int block_size = 256;
    int grid_size = (num_full + block_size - 1) / block_size;
    int shared_size = block_size * sizeof(float4);

    new_relu<<<grid_size, block_size, shared_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
'''

cpp_src = "torch::Tensor new_relu_cuda(torch::Tensor x);"

new_relu_mod = load_inline(
    name="new_relu_mod",
    cuda_sources=source,
    cpp_sources=cpp_src,
    functions=["new_relu_cuda"],
    verbose=False,
    extra_cuda_cflags=['-arch=sm_75']
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return new_relu_mod.new_relu_cuda(x)
