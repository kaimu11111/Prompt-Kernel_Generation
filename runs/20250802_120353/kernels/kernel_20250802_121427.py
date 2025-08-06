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

    // Compute per-thread max
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = input[row * dim + i];
        if (val > local_max) local_max = val;
    }
    shared_mem[threadIdx.x] = local_max;
    __syncthreads();

    // Block max reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared_mem[threadIdx.x + s] > shared_mem[threadIdx.x]) {
                shared_mem[threadIdx.x] = shared_mem[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    float global_max = shared_mem[0];
    __syncthreads();

    // Compute exponentials and partial sums
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __expf(input[row * dim + i] - global_max);
        output[row * dim + i] = val;
        local_sum += val;
    }
    shared_mem[threadIdx.x] = local_sum;
    __syncthreads();

    // Block sum reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + s];
        }
        __syncthreads();
    }
    float total_sum = shared_mem[0];
    __syncthreads();

    // Normalize results
    float inv_sum = 1.0f / total_sum;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        output[row * dim + i] *= inv_sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int batch_size = input.size(0);
    int dim = input.size(1);

    const int block_size = 512;
    dim3 grid(batch_size);
    dim3 block(block_size);
    int smem_size = block.x * sizeof(float);

    softmax_kernel<<<grid, block, smem_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), batch_size, dim);
    return output;
}
"""

cpp_src = "torch::Tensor softmax_cuda(torch::Tensor input);"

softmax_cuda_mod = load_inline(
    name="softmax_cuda",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_cuda = softmax_cuda_mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_cuda.softmax_cuda(x)
