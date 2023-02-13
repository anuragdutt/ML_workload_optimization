import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule

# Define CUDA kernel
kernel = """
__global__ void matrix_mul(float *a, float *b, float *c, int num_rows_a, int num_cols_b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= num_rows_a || idy >= num_cols_b) return;

    float value = 0.0f;
    for (int i = 0; i < num_cols_b; i++)
    {
        value += a[idx * num_cols_b + i] * b[i * num_cols_b + idy];
    }
    c[idx * num_cols_b + idy] = value;
}
"""

def matrix_multiplication_cuda(a, b):
    num_rows_a = a.shape[0]
    num_cols_b = b.shape[1]
    num_cols_a = a.shape[1]
    
    # Copy data to the GPU
    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)

    # Allocate memory on the GPU for result
    c_gpu = gpuarray.empty((num_rows_a, num_cols_b), np.float32)

    # Compile the CUDA kernel
    mod = SourceModule(kernel)
    func = mod.get_function("matrix_mul")

    # Call the CUDA kernel
    grid_dims = (num_rows_a, num_cols_b)
    block_dims = (16, 16)
    func(a_gpu, b_gpu, c_gpu, np.int32(num_rows_a), np.int32(num_cols_b), grid=grid_dims, block=block_dims)

    return c_gpu.get()

if __name__ == "__main__":
    # Test the code with two random matrices
    a = np.random.randn(1024, 1024).astype(np.float32)
    b = np.random.randn(1024, 1024).astype(np.float32)

    result = matrix_multiplication_cuda(a, b)