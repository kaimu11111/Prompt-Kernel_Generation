import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 12
#define TILE_HEIGHT 12

template <typename T>
__global__ void conv_relu_add_bias_kernel(
    const T* input,
    const T* weight,
    const T* bias,
    T* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int kernel_size,
    int output_height,
    int output_width) {

    const int kernel_size_sq = kernel_size * kernel_size;
    const int input_region_width = TILE_WIDTH + kernel_size - 1;
    const int input_region_height = TILE_HEIGHT + kernel_size - 1;
    const int input_region_size = input_region_width * input_region_height;

    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int c_out = blockIdx.z % out_channels;
    int n = blockIdx.z / out_channels;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x_in_tile = tx;
    int y_in_tile = ty;

    int x_out = tile_x * TILE_WIDTH + x_in_tile;
    int y_out = tile_y * TILE_HEIGHT + y_in_tile;

    if (x_out >= output_width || y_out >= output_height) return;

    extern __shared__ T shared[];
    T* input_shared = shared;
    T* weight_shared = input_shared + in_channels * input_region_size;

    int input_x_base = tile_x * TILE_WIDTH - 1;
    int input_y_base = tile_y * TILE_HEIGHT - 1;
    int tid = ty * blockDim.x + tx;

    // Load input region into shared memory
    for (int c_in = tid; c_in < in_channels; c_in += blockDim.x * blockDim.y) {
        for (int dx = 0; dx < input_region_width; dx++) {
            for (int dy = 0; dy < input_region_height; dy++) {
                int global_x = input_x_base + dx;
                int global_y = input_y_base + dy;
                int input_idx = n * in_channels * input_height * input_width
                    + c_in * input_height * input_width
                    + global_x * input_width + global_y;
                int shared_idx = c_in * input_region_size + dx * input_region_height + dy;
                input_shared[shared_idx] = (global_x >= 0 && global_x < input_width &&
                                           global_y >= 0 && global_y < input_height) ?
                                          input[input_idx] : 0;
            }
        }
    }

    // Load weights for current output channel
    int weight_base = c_out * in_channels * kernel_size_sq;
    for (int i = tid; i < in_channels * kernel_size_sq; i += blockDim.x * blockDim.y) {
        weight_shared[i] = weight[weight_base + i];
    }

    __syncthreads();

    T sum = 0.0;
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kx = 0; kx < kernel_size; kx++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                int dx = x_in_tile + kx;
                int dy = y_in_tile + ky;
                int input_idx = c_in * input_region_size + dx * input_region_height + dy;
                int weight_idx = c_in * kernel_size_sq + kx * kernel_size + ky;
                sum += input_shared[input_idx] * weight_shared[weight_idx];
            }
        }
    }
    sum += bias[c_out];
    T out_val = sum > 0.0 ? sum : 0.0;

    // Write output
    int output_offset = n * out_channels * output_height * output_width
        + c_out * output_height * output_width
        + y_out * output_width + x_out;
    output[output_offset] = out_val;
}

torch::Tensor conv_relu_add_cuda(torch::Tensor input,
                                torch::Tensor weight,
                                torch::Tensor bias) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int input_height = input.size(2);
    const int input_width = input.size(3);
    const int output_height = input_height - kernel_size + 1;
    const int output_width = input_width - kernel_size + 1;

    auto output = torch::empty({batch_size, out_channels, output_height, output_width},
                              input.options());

    dim3 threads(TILE_WIDTH, TILE_HEIGHT);
    dim3 blocks(
        (output_width + TILE_WIDTH - 1) / TILE_WIDTH,
        (output_height + TILE_HEIGHT - 1) / TILE_HEIGHT,
        batch_size * out_channels
    );

    const int input_region_size_val = (TILE_WIDTH + kernel_size - 1) 
        * (TILE_HEIGHT + kernel_size - 1);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_relu_add_cuda", ([&] {
        const int shared_size = (input_region_size_val * in_channels 
            + in_channels * kernel_size * kernel_size) * sizeof(scalar_t);
        conv_relu_add_bias_kernel<scalar_t><<<blocks, threads, shared_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            input_height,
            input_width,
            kernel_size,
            output_height,
            output_width
        );
    }));

    return output;
}
"""

cpp_src = (
    "torch::Tensor conv_relu_add_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);"
)

conv_relu_add = load_inline(
    name="conv_relu_add",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["conv_relu_add_cuda"],
    verbose=True,
    extra_cuda_cflags=["-Wno-deprecated-gpu-targets"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.conv_relu_add = conv_relu_add

    def forward(self, x):
        weight = self.conv.weight
        bias = self.bias
        return self.conv_relu_add.conv_relu_add_cuda(x, weight, bias)
