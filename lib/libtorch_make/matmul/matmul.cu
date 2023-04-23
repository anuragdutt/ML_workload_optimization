#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " N" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);

    // Create matrices                                                                                                                                                                                      
    auto A = torch::full({N, N}, 3.0f).to(torch::kCUDA);
    auto B = torch::full({N, N}, 4.0f).to(torch::kCUDA);
    auto C = torch::zeros({N, N}).to(torch::kCUDA);

    // Perform matrix multiplication on GPU 1000 times

    std::cout << "Processed Images" << std::endl;


    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "TimePreModel --" << std::to_string(std::time(nullptr)) << std::endl;

    for (int i = 0; i < 1000; ++i) {
        matmul_kernel<<<numBlocks, threadsPerBlock>>>(A.data_ptr<float>(),
                                                       B.data_ptr<float>(),
                                                       C.data_ptr<float>(),
                                                       N);
        // Wait for kernel to finish                                                                                                                                                                            
        cudaDeviceSynchronize();

	// Print result                                                                                                                                                                                         
	//	std::cout << "Result:" << std::endl << C << std::endl;

    }


    return 0;
}
