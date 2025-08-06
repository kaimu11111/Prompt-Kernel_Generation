import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (idx < size) {
        int pos0 = idx;
        int pos1 = pos0 + 1;
        int pos2 = pos0 + 2;
        int pos3 = pos0 + 3;

        if (pos0 < size) {
            out[pos0] = max(0.f, __ldg(x + pos0));
        }
        if (pos1 < size) {
            out[pos1] = max(0.f, __ldg(x + pos1));
        }
        if (pos2 < size) {
            out[pos2] = max(0.f, __ldg(x + pos2));
        }
        if (pos3 < size) {
            out[pos3] = max(0.f, __ldg(x + pos3));
        }
        idx += stride;
    }
}

torch::Tensor custom_relu_cuda(torch::Tensor x) {
    int64_t size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 512;
    int num_blocks = (size + block_size - 1) / block_size;

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
