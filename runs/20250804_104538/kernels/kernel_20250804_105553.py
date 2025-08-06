# <corrected, optimized Python script>
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx * 4;
    if (base < size) {
        int pos0 = base;
        int pos1 = base + 1;
        int pos2 = base + 2;
        int pos3 = base + 3;

        if (pos0 < size) {
            float x0 = __ldg(x + pos0);
            out[pos0] = (x0 > 0.f) ? x0 : 0.f;
        }
        if (pos1 < size) {
            float x1 = __ldg(x + pos1);
            out[pos1] = (x1 > 0.f) ? x1 : 0.f;
        }
        if (pos2 < size) {
            float x2 = __ldg(x + pos2);
            out[pos2] = (x2 > 0.f) ? x2 : 0.f;
        }
        if (pos3 < size) {
            float x3 = __ldg(x + pos3);
            out[pos3] = (x3 > 0.f) ? x3 : 0.f;
        }
    }
}

torch::Tensor custom_relu_cuda(torch::Tensor x) {
    int64_t size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 512;
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
    extra_cflags=["-O3", "-use_fast_math"],
    extra_ldflags=["-O3", "-use_fast_math"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.custom_relu = custom_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_relu.custom_relu_cuda(x)
