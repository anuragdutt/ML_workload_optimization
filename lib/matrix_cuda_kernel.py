import sys
import time
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import math
import matplotlib.pyplot as plt
from sys import getsizeof

# -- initialize the device
import pycuda.autoinit

# -----------------------------------------------------
# CUDA parameters
kernel_code_template = """
__global__  void MatProd(float* C, float* A, float* B, int dimAx, int dimBx, int dimCx, int dimCy)
{
  int row = blockDim.y*blockIdx.y+threadIdx.y;
  int col = blockDim.x*blockIdx.x+threadIdx.x;

	double Result = 0;

	if (row<=dimCy-1 && col<=dimCx-1)
	{
		for (int k = 0; k < dimAx; k++)
		{
			Result += A[k + dimAx*row] * B[col + dimBx*k];
		}

		C[col + row*dimCx] = Result;
	}
}
"""


if __name__ == "__main__":
    # get the kernel code from the template 
    kernel_code=kernel_code_template
    # compile the kernel code 
    mod = compiler.SourceModule(kernel_code)
    # get the kernel function from the compiled module
    MatProd = mod.get_function("MatProd")
    warp_size=16 # Warp size on the GPU.
    # --------------------------------------------------------------------
    # --------------------BEGIN of INITIALISATION-------------------------
    # --------------------------------------------------------------------

    # We create the python matrices for the computation C=A*B
    # This part is supposed as an input, so we don't take in account any computation
    # time here.

    mat_size = int(sys.argv[1])

    nb_columnsA = mat_size
    nb_linesA = mat_size

    nb_columnsB = mat_size
    nb_linesB = mat_size

    a_cpu = np.full((mat_size, mat_size), 3, np.float) # matrix containing all 3's
    b_cpu = np.full((mat_size, mat_size), 3, np.float) # matrix containing all 3's

    threadPerBlockx=warp_size 
    threadPerBlocky=warp_size
    BlockPerGridx = (int) (1 + (nb_columnsB - 1) // threadPerBlockx);
    BlockPerGridy = (int) (1 + (nb_linesA - 1) // threadPerBlockx);

    a_gpu = gpuarray.to_gpu(a_cpu)
    b_gpu=gpuarray.to_gpu(b_cpu)
    c_gpu = gpuarray.empty((nb_linesA, nb_columnsB), np.float32)
    
    print("TimePreModel --"+ str(time.time()))
    time.sleep(1)
    print("TimePreModelSleep --"+ str(time.time()))

    MatProd(
        # output
        c_gpu, 
        # inputs
        a_gpu, b_gpu,
        np.int32(nb_columnsA),np.int32(nb_columnsB),np.int32(nb_columnsB),np.int32(nb_linesA),
        # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block = (threadPerBlockx, threadPerBlocky, 1), grid=(BlockPerGridx,BlockPerGridy)
        )


    c_gpu_result=c_gpu.get()

    print("TimeLazyLoad --"+ str(time.time()))
    time.sleep(1)
    print("TimeLazyLoadSleep --"+ str(time.time()))

    # Start the kernel 
    for i in range(0,50):
        MatProd(
            # output
            c_gpu, 
            # inputs
            a_gpu, b_gpu,
            np.int32(nb_columnsA),np.int32(nb_columnsB),np.int32(nb_columnsB),np.int32(nb_linesA),
            # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
            block = (threadPerBlockx, threadPerBlocky, 1), grid=(BlockPerGridx,BlockPerGridy)
            )

        c_gpu_result=c_gpu.get()
