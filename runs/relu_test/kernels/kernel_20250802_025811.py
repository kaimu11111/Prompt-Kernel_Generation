import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void optimized_relu(const float* x, float* y, int n) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    int block_elements = block_size * 4;
    int block_start = bid * block_elements;
    int block_end = min(block_start + block_elements, n);

    // Load into shared memory
    for (int i = tid; i < block_elements; i += block_size) {
        int pos = block_start + i;
        if (pos < n) {
            shared[i] = x[pos];
        } else {
            shared[i] = 0.0f;
        }
    }
    __syncthreads();

    // Process in shared memory
    for (int i = tid; i < block_elements; i += block_size) {
        shared[i] = fmaxf(shared[i], 0.0f);
    }
    __syncthreads();

    // Store back to global memory
    for (int i = tid; i < block_elements; i += block_size) {
        int pos = block_start + i;
        if (pos < n) {
            y[pos] = shared[i];
        }
    }
}
'''

cpp_src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

extern "C" {
    __global__ void optimized_relu(const float* x, float* y, int n);
    
    void optimized_relu_forward(const at::Tensor x, at::Tensor y) {
        const int block_size = 256;
        const int block_elements = block_size * 4;
        const int grid_size = (x.numel() + block_elements - 1) / block_elements;
        const int shared_size = block_elements * sizeof(float);

        optimized_relu<<<grid_size, block_size, shared_size>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            x.numel()
        );
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
