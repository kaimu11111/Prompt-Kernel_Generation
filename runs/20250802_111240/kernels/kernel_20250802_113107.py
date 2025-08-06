import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename T>
__device__ T warp_reduce_max(T val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        T temp = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, temp);
    }
    return val;
}

template <typename T>
__device__ T warp_reduce_sum(T val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        T temp = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val += temp;
    }
    return val;
}

__global__ void softmax_kernel(const float* in_data, float* out_data, int batch_size, int features) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    __shared__ float s_max[32];
    __shared__ float s_sum[32];

    const int block_size = blockDim.x;
    const int num_warps = (blockDim.x + 31) / 32;

    // Compute chunk for this thread
    const int chunk_size = (features + block_size - 1) / block_size;
    const int start = tid * chunk_size;
    const int end = min(start + chunk_size, features);

    // Step 1: Compute local max and exponential terms
    float local_max = -FLT_MAX;
    for (int i = start; i < end; i++) {
        float val = in_data[row * features + i];
        if (val > local_max) local_max = val;
    }

    // Step 2: Warp reduction for max
    float warp_max = warp_reduce_max<float>(local_max);
    if (lane_id == 0) s_max[warp_id] = warp_max;
    __syncthreads();

    // Step 3: Compute global max using first warp's reduction
    float global_max = -FLT_MAX;
    if (tid < 32) {
        int entry = tid;
        float val = (entry < num_warps) ? s_max[entry] : -FLT_MAX;
        float warp_max_global = warp_reduce_max<float>(val);
        if (lane_id == 0) s_max[0] = warp_max_global;
    }
    __syncthreads();
    global_max = s_max[0];

    // Step 4: Compute exponentials and local sum
    float local_sum = 0.0f;
    for (int i = start; i < end; i++) {
        float val = in_data[row * features + i] - global_max;
        float exp_val = expf(val);
        out_data[row * features + i] = exp_val;
        local_sum += exp_val;
    }

    // Step 5: Warp reduction for sum
    float warp_sum = warp_reduce_sum<float>(local_sum);
    if (lane_id == 0) s_sum[warp_id] = warp_sum;
    __syncthreads();

    // Step 6: Compute total sum using first warp's reduction
    float total_sum = 0.0f;
    if (tid < 32) {
        int entry = tid;
        float val = (entry < num_warps) ? s_sum[entry] : 0.0f;
        float warp_sum_total = warp_reduce_sum<float>(val);
        if (lane_id == 0) s_sum[0] = warp_sum_total;
    }
    __syncthreads();
    total_sum = s_sum[0];

    // Step 7: Normalize
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

softmax_cuda = load_inline(
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
        self.softmax_cuda = softmax_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_cuda.softmax_cuda(x)
