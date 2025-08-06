import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_kernel(const float* in_data, float* out_data, int batch_size, int features) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int chunk_size = (features + num_threads - 1) / num_threads;
    int start = tid * chunk_size;
    int end = min(start + chunk_size, features);

    // Compute local max
    float local_max = -FLT_MAX;
    for (int i = start; i < end; i++) {
        float val = in_data[row * features + i];
        if (val > local_max) {
            local_max = val;
        }
    }

    // Reduction for max
    __shared__ float shared_max[256];
    shared_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid] < shared_max[tid + s]) {
                shared_max[tid] = shared_max[tid + s];
            }
        }
        __syncthreads();
    }
    float global_max = shared_max[0];

    // Compute exponentials and local sum
    __syncthreads();
    float local_sum = 0.0f;
    for (int i = start; i < end; i++) {
        float val = in_data[row * features + i] - global_max;
        float exp_val = expf(val);
        out_data[row * features + i] = exp_val;
        local_sum += exp_val;
    }

    // Reduction for sum
    __shared__ float shared_sum[256];
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    float total_sum = shared_sum[0];

    // Normalize
    __syncthreads();
    for (int i = start; i < end; i++) {
        out_data[row * features + i] /= total_sum;
    }
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    auto batch_size = x.size(0);
    auto features = x.size(1);
    auto out = torch::empty_like(x);

    const int block_size = 256;
    dim3 blocks(batch_size);
    dim3 threads(block_size);

    softmax_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), batch_size, features);

    return out;
}
"""

cpp_src = (
    "torch::Tensor softmax_cuda(torch::Tensor x);"
)

softmax = load_inline(
    name="softmax",
    cuda_sources=source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax.softmax_cuda(x)
