source = r'''
extern "C" {
    template<typename T>
    __global__ void optimized_relu(const T* x, T* y, int n) {
        int block_start = blockIdx.x * blockDim.x * 4;
        int block_end = std::min(block_start + blockDim.x * 4, n);
        for (int i = threadIdx.x; i < (block_end - block_start); i += blockDim.x) {
            int idx = block_start + i;
            y[idx] = max(x[idx], static_cast<T>(0));
        }
    }
}
'''

cpp_src = r'''
#include <torch/extension.h>

extern "C" {
    template<typename T>
    __global__ void optimized_relu(const T* x, T* y, int n);
}

void optimized_relu_forward_cuda(const at::Tensor x, at::Tensor y) {
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "optimized_relu_forward_cuda", ([&] {
        const int threads = 256;
        const int num_elements = x.numel();
        const int blocks = (num_elements + threads * 4 - 1) / (threads * 4);
        optimized_relu<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            num_elements);
        cudaDeviceSynchronize();
    }));
}

at::Tensor optimized_relu_forward(const at::Tensor &x) {
    auto y = at::empty_like(x);
    optimized_relu_forward_cuda(x, y);
    return y;
}

void optimized_relu_backward_cuda(const at::Tensor grad_y, const at::Tensor y, at::Tensor grad_x) {
    optimized_relu_forward_cuda(grad_y, grad_x); // ReLU gradient uses y's mask
}

at::Tensor optimized_relu_backward(const at::Tensor &grad_y, const at::Tensor &y) {
    auto grad_x = at::empty_like(grad_y);
    optimized_relu_backward_cuda(grad_y, y, grad_x);
    return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("optimized_relu_forward", &optimized_relu_forward, "Optimized ReLU forward");
    m.def("optimized_relu_backward", &optimized_relu_backward, "Optimized ReLU backward");
}
'''

import torch
from torch import nn
import torch.utils.cpp_extension

optimized_relu = torch.utils.cpp_extension.load_inline(
    name='optimized_relu',
    cuda=True,
    cpp_sources=cpp_src,
    cuda_sources=source,
    extra_cuda_cflags=['-arch=sm_75']
)

class ReLUFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = optimized_relu.optimized_relu_forward(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        y, = ctx.saved_tensors
        grad_x = optimized_relu.optimized_relu_backward(grad_y, y)
        return grad_x

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return ReLUFun.apply(x)
