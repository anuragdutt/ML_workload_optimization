import os
import sys
import numpy as np
import numba as nb
from numba import cuda, float32
import math
import time

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16

@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp



if __name__ == "__main__":


    mat_size = int(sys.argv[1])
    # Initialize the data arrays
    A = np.full((mat_size, mat_size), 3, np.float) # matrix containing all 3's
    B = np.full((mat_size, mat_size), 4, np.float) # matrix containing all 4's

    # Copy the arrays to the device
    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)

    # Allocate memory on the device for the result
    C_global_mem = cuda.device_array((mat_size, mat_size))

    # Configure the blocks
    threadsperblock = (16, 16)
    # blockspergrid_x = int(256)
    # blockspergrid_y = int(256)
    blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[1]))

    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    print("TimePreModel --"+ str(time.time()))
    time.sleep(1)
    print("TimePreModelSleep --"+ str(time.time()))

    fast_matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
    
    print("TimeLazyLoad --"+ str(time.time()))
    time.sleep(1)
    print("TimeLazyLoadSleep --"+ str(time.time()))
    
    # Start the kernel 
    for i in range(0,50):
        fast_matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
