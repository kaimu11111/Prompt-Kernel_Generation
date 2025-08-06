import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void relu_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t size) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_start = thread_id * 4;

    if (chunk_start >= size) return;

    const float4* x4 = reinterpret_cast<const float4*>(x + chunk_start);
    float4 val = __ldg(x4);
    float* vals = reinterpret_cast<float*>(&val);

    for (int i = 0; i < 4; ++i) {
        int pos = chunk_start + i;
        if (pos >= size) continue;
        out[pos] = fmaxf(vals[i], 0.f);
    }
}

torch::Tensor custom_relu_cuda(torch::Tensor x) {
    int64_t size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    int num_threads_needed = (size + 3) / 4;
    int num_blocks = (num_threads_needed + block_size - 1) / block_size;

    relu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

cpp_src = "torch::Tensor custom_relu_cuda(torch::Tensor x);"

custom_relu = load_inline(
    name="custom_relu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["custom_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom_relu = custom_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_relu.custom_relu_cuda(x)
