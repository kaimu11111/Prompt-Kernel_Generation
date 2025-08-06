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
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    __shared__ float s_data[256]; // Shared memory for warp results

    const int chunk_size = (features + blockDim.x - 1) / blockDim.x;
    const int start = tid * chunk_size;
    const int end = min(start + chunk_size, features);

    // Step 1: Compute local max for each thread's chunk
    float local_max = -FLT_MAX;
    for (int i = start; i < end; i++) {
        float val = in_data[row * features + i];
        if (val > local_max) {
            local_max = val;
        }
    }

    // Warp reduction for max using shuffle
    float warp_max = local_max;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, warp_max, offset);
        warp_max = fmaxf(warp_max, other);
    }

    if (lane_id == 0) {
        s_data[warp_id] = warp_max;
    }
    __syncthreads();

    // Compute global max from all warp maxes
    float global_max = -FLT_MAX;
    const int num_warps = blockDim.x / 32;
    for (int i = 0; i < num_warps; i++) {
        float candidate = s_data[i];
        if (candidate > global_max) {
            global_max = candidate;
        }
    }

    // Step 2: Compute exponentials and local sum
    float local_sum = 0.0f;
    for (int i = start; i < end; i++) {
        float val = in_data[row * features + i] - global_max;
        float exp_val = expf(val);
        out_data[row * features + i] = exp_val;
        local_sum += exp_val;
    }

    // Warp reduction for sum using shuffle
    float warp_sum = local_sum;
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, warp_sum, offset);
        warp_sum += other;
    }

    if (lane_id == 0) {
        s_data[warp_id] = warp_sum;
    }
    __syncthreads();

    // Compute total_sum from all warp sums
    float total_sum = 0.0f;
    for (int i = 0; i < num_warps; i++) {
        total_sum += s_data[i];
    }

    // Step 3: Normalize
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

softmax_op = load_inline(
    name="softmax_op",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["softmax_cuda"],
    verbose=True,
    extra_cflags=["-std=c++14"],
    extra_cuda_flags=["-arch=sm_75"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.softmax_cuda = softmax_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_cuda.softmax_cuda(x)
