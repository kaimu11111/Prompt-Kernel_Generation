import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
extern "C" {
    __global__ void my_kernel(float* A, float* shared_A, int a_row, int a_col, int K) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        bool a_valid = (a_row < K) && (a_col < K);
        shared_A[ty * K + tx] = a_valid ? __ldg(&A[a_row * K + a_col]) : 0.0f;
    }
}
"""

cpp_src = """
#include <cuda_runtime.h>

extern __global__ void my_kernel(float*, float*, int, int, int);

torch::Tensor my_op(torch::Tensor A) {
    int K = A.size(0);
    float* A_data = A.data_ptr<float>();
    float* shared_A_data;
    cudaMalloc(&shared_A_data, K*K * sizeof(float));
    dim3 threads(K, K);
    my_kernel<<<1, threads>>>(A_data, shared_A_data, 0, 0, K);
    cudaDeviceSynchronize();
    torch::Tensor output = torch::from_blob(shared_A_data, {K, K}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    return output;
}
"""

my_module = load_inline(name='my_module', 
                       cuda_src=source,
                       extra_cuda_cflags=['-arch=sm_75'],
                       cpp_sources=[cpp_src],
                       is_python_module=True,
                       with_cuda=True)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
    def forward(self, x):
        return my_module.my_op(x)
