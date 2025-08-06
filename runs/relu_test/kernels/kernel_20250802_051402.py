import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void optimized_relu(const float* x, float* y, int n) {
    extern __shared__ float4 shared[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    int num_full = n / 4;
    int residual = n % 4;

    int block_start = bid * block_size;
    int block_end = min(block_start + block_size, num_full);

    // Load vectorized chunks into shared memory with coalesced access
    for (int i = tid; i < (block_end - block_start); i += block_size) {
        int global_idx = block_start + i;
        int offset = global_idx * 4;
        shared[i].x = x[offset];
        shared[i].y = x[offset + 1];
        shared[i].z = x[offset + 2];
        shared[i].w = x[offset + 3];
    }
    __syncthreads();

    // Process in shared memory to hide latency
    for (int i = tid; i < (block_end - block_start); i += block_size) {
        float4 val = shared[i];
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        shared[i] = val;
    }
    __syncthreads();

    // Store results back using coalesced writes
    for (int i = tid; i < (block_end - block_start); i += block_size) {
        int offset = (block_start + i) * 4;
        float4 val = shared[i];
        y[offset] = val.x;
        y[offset + 1] = val.y;
        y[offset + 2] = val.z;
        y[offset + 3] = val.w;
    }

    // Handle residual elements with grid-strided loop
    if (residual > 0) {
        int residual_start = num_full * 4;
        for (int i = tid; i < residual; i += blockDim.x * gridDim.x) {
            int pos = residual_start + i;
            y[pos] = fmaxf(x[pos], 0.0f);
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
"""

cpp_src = r"""
#include <torch/extension.h>

extern "C" {
    void optimized_relu_forward(const at::Tensor, at::Tensor);
}
"""

optimized_relu_mod = load_inline(
    name="optimized_relu_mod",
    cuda_sources=source,
    cpp_sources=cpp_src,
    functions=["optimized_relu_forward"],
    verbose=False,
    extra_cuda_cflags=['-arch=sm_75', '-use_fast_math']
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_contig = x.contiguous()
        y = torch.empty_like(x_contig)
        optimized_relu_mod.optimized_relu_forward(x_contig, y)
        return y
