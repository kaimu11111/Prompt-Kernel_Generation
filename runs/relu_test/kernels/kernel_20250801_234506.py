import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void grid_strided_relu(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < n; idx += blockDim.x * gridDim.x) {
        y[idx] = fmaxf(x[idx], 0.0f);
    }
}

torch::Tensor grid_strided_relu_cuda(torch::Tensor x) {
    auto n = x.numel();
    auto y = torch::empty_like(x);

    const int block_size = 512; // Different block size than existing kernel
    const int grid_size = (n + block_size - 1) / block_size;

    grid_strided_relu<<<grid_size, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}
"""

cpp_src = "torch::Tensor grid_strided_relu_cuda(torch::Tensor x);"

grid_strided_relu = load_inline(
    name="grid_strided_relu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["grid_strided_relu_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_op = grid_strided_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu_op.grid_strided_relu_cuda(x)
