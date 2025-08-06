source = r'''
extern "C" {
    __global__ void optimized_relu(const float* x, float* y, int n) {
        int block_start = blockIdx.x * blockDim.x * 4;
        int block_end = std::min(block_start + blockDim.x * 4, n);
        for (int i = threadIdx.x; i < (block_end - block_start); i += blockDim.x) {
            int idx = block_start + i;
            y[idx] = fmaxf(x[idx], 0.0f);
        }
    }
}
'''

cpp_src = r'''
#include <torch/extension.h>
#include <vector>

extern "C" void optimized_relu_forward(const at::Tensor x, at::Tensor y) {
    const int threads = 256;
    const int num_elements = x.numel();
    const int blocks = (num_elements + threads * 4 - 1) / (threads * 4);
    optimized_relu<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements);
    cudaDeviceSynchronize();
}

extern "C" void optimized_relu_backward(const at::Tensor grad_y, const at::Tensor y, at::Tensor grad_x) {
    optimized_relu_forward(grad_y, grad_x); // ReLU gradient uses y's mask
}

py::module define_module(py::module_& m) {
    m.def("optimized_relu_forward", &optimized_relu_forward, "Optimized ReLU forward");
    m.def("optimized_relu_backward", &optimized_relu_backward, "Optimized ReLU backward");
    return m;
}
'''

import torch
from torch import nn
import pybind11
from torch.utils.cpp_extension import load_inline

optimized_relu = load_inline(name='optimized_relu', 
                            cuda=True,
                            cpp_sources=cpp_src,
                            cuda_sources=source,
                            extra_include_paths=[pybind11.get_include(),],
                            extra_cuda_cflags=['-arch=sm_75'])

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return optimized_relu.optimized_relu_forward(x)
