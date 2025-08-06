import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vec_relu_kernel(const float* x, float* y, int num_float4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; idx < num_float4; idx += stride) {
        float4 val = ((float4*)x)[idx];
        val.x = fmaxf(val.x, 0.0f);
        val.y = fmaxf(val.y, 0.0f);
        val.z = fmaxf(val.z, 0.0f);
        val.w = fmaxf(val.w, 0.0f);
        ((float4*)y)[idx] = val;
    }
}

torch::Tensor vec_relu_cuda(torch::Tensor x) {
    int n = x.numel();
    int num_float4 = (n + 3) / 4;
    auto y = torch::empty_like(x);

    const int block_size = 256;
    const int grid_size = (num_float4 + block_size - 1) / block_size;

    vec_relu_kernel<<<grid_size, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_float4);

    return y;
}
"""

cpp_src = "torch::Tensor vec_relu_cuda(torch::Tensor x);"

vec_relu = load_inline(
    name="vec_relu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["vec_relu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.vec_relu = vec_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vec_relu.vec_relu_cuda(x)
