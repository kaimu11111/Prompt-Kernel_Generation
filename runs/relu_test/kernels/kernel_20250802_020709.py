cuda
__global__ void optimized_relu(const float* x, float* y, int n) {
    int block_start = blockIdx.x * blockDim.x * 4;
    int block_end = min(block_start + blockDim.x * 4, n);
    for (int i = threadIdx.x; i < (block_end - block_start); i += blockDim.x) {
        int idx = block_start + i;
        y[idx] = fmaxf(x[idx], 0.0f);
    }
}
