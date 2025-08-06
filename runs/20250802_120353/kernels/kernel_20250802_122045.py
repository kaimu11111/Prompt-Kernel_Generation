import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int batch_size, int dim) {
    int row = blockIdx.x;
    extern __shared__ float shared_mem[];

    const int warp_size = 32;
    const int warp_count = blockDim.x / warp_size;

    // Compute local max
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = input[row * dim + i];
        if (val > local_max) {
            local_max = val;
        }
    }

    // Warp-level max reduction
    for (int mask = 16; mask > 0; mask >>=1) {
        local_max = max(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, mask));
    }

    // Store the warp's max in shared memory
    if (threadIdx.x % warp_size == 0) {
        shared_mem[threadIdx.x / warp_size] = local_max;
    }
    __syncthreads();

    // Reduce to global_max
    if (threadIdx.x < warp_count) {
        for (int s = warp_count / 2; s > 0; s >>=1) {
            if (threadIdx.x < s) {
                shared_mem[threadIdx.x] = max(shared_mem[threadIdx.x], shared_mem[threadIdx.x + s]);
            }
            __syncthreads();
        }
    }
    __syncthreads();
    float global_max = shared_mem[0];

    // Compute exponents and local sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = input[row * dim + i] - global_max;
        float exp_val = __expf(val);
        output[row * dim + i] = exp_val;
        local_sum += exp_val;
    }

    // Warp-level sum reduction
    for (int mask = 16; mask > 0; mask >>=1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, mask);
    }

    // Store the warp's sum
    if (threadIdx.x % warp_size ==0) {
        shared_mem[threadIdx.x / warp_size] = local_sum;
    }
    __syncthreads();

    // Reduce to total_sum
    if (threadIdx.x < warp_count) {
        for (int s = warp_count / 2; s > 0; s >>=1) {
            if (threadIdx.x < s) {
                shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
            }
            __syncthreads();
        }
    }
    __syncthreads();
    float total_sum = shared_mem[0];

    // Normalize
    float inv_total = 1.0f / total_sum;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        output[row * dim + i] *= inv_total;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int batch_size = input.size(0);
    int dim = input.size(1);

    const int block_size = 256;
    dim3 grid(batch_size);
    dim3 block(block_size);
    int warp_count = block_size / 32;
    int smem_size = warp_count * sizeof(float);

    softmax_kernel<<<grid, block, smem_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);

    return output;
}
"""

cpp_src = (
    "torch::Tensor softmax_cuda(torch::Tensor input);"
)

softmax_cuda = load_inline(
    name="softmax_cuda",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_cflags=["-std=c++14"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_cuda = softmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_cuda.softmax_cuda(x)
