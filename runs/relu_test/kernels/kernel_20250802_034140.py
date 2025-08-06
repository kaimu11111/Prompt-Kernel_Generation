import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void optimized_relu(const float* x, float* y, int n) {
    extern __shared__ float4 shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    int num_full = n /4;
    int residual = n %4;

    int block_start = bid * block_size;
    int block_end = min(block_start + block_size, num_full);

    if (tid < (block_end - block_start)) {
        shared[tid] = ((float4*)x)[block_start + tid];
    }
    __syncthreads();

    if (tid < (block_end - block_start)) {
        float4 val = shared[tid];
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        shared[tid] = val;
    }
    __syncthreads();

    if (tid < (block_end - block_start)) {
        ((float4*)y)[block_start + tid] = shared[tid];
    }

    if (residual > 0) {
        int residual_start = num_full *4;
        for (int i = tid; i < residual; i += blockDim.x * gridDim.x) {
            y[residual_start + i] = fmaxf(x[residual_start + i], 0.0f);
        }
    }
}

extern "C" {
    void optimized_relu_forward(const at::Tensor x, at::Tensor y) {
        int n = x.numel();
        int num_full = n /4;
        const int block_size = 256;
        const int grid_size = (num_full + block_size - 1) / block_size;
        const int shared_size = block_size * sizeof(float4);

        optimized_relu<<<grid_size, block_size, shared_size>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            n
        );
        cudaDeviceSynchronize();
    }
}
'''

cpp_src = r'''
#include <torch/extension.h>

extern "C" {
    void optimized_relu_forward(const at::Tensor, at::Tensor);
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
