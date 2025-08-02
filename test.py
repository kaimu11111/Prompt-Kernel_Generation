import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for ReLU activation
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* x, float* out, int size) {
    __shared__ float block_data[256 * 4];  // 256 threads, each handling 4 elements
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    
    // Load 4 elements per thread into shared memory
    for (int i = 0; i < 4; i++) {
        int global_idx = block_idx * 256 * 4 + thread_idx * 4 + i;
        if (global_idx < size) {
            block_data[thread_idx * 4 + i] = x[global_idx];
        }
    }
    __syncthreads();
    
    // Apply ReLU to 4 elements per thread
    for (int i = 0; i < 4; i++) {
        int global_idx = block_idx * 256 * 4 + thread_idx * 4 + i;
        if (global_idx < size) {
            out[global_idx] = fmaxf(0.0f, block_data[thread_idx * 4 + i]);
        }
    }
    __syncthreads();
}

torch::Tensor relu_cuda(torch::Tensor x) {
    auto size = x.numel();
    auto out = torch::zeros_like(x);
    
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    relu_kernel<<<grid_size, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    
    return out;
}
"""

cpp_src = (
    "torch::Tensor relu_cuda(torch::Tensor x);"
)

# Compile the inline CUDA code for ReLU
relu = load_inline(
    name="relu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = relu

    def forward(self, x):
        return self.relu.relu_cuda(x)

# ──────────────────────────────────────────────────────────────
# Simple functional test & benchmark
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create random input
    batch_size = 4096
    dim = 200000

    def get_inputs():
        x = torch.rand(batch_size, dim)
        return x
    # reference output
    x = get_inputs()
    y_ref = torch.relu(x)

    # custom kernel output
    model = ModelNew().to(device)
    y_out = model(x)

    # correctness check
    max_err = (y_out - y_ref).abs().max().item()
    print(f"Max absolute error: {max_err:.3e}")


    # simple timing
    def bench(fn, iters=20, warmup=5):
        # warm-up
        for _ in range(warmup):
            _ = fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            _ = fn()
        torch.cuda.synchronize()
        return (time.time() - t0) * 1e3 / iters  # ms / iter

    t_ref  = bench(lambda: torch.relu(x))
    t_new  = bench(lambda: model(x))
    print(f"torch.relu avg latency: {t_ref:.2f} ms")
    print(f"custom ReLU avg latency: {t_new:.2f} ms")
