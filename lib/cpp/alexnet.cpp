#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include <unistd.h>

inline void printCurrentTime() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::cout <<std::fixed<< static_cast<double>(now_c);
}
int main(int argc, char** argv) {
    int N = 2048;
    int batch_size = std::stoi(argv[1]);
    torch::Tensor A = torch::rand({batch_size, N, N}, torch::kCUDA);
    torch::Tensor B = torch::rand({batch_size, N, N}, torch::kCUDA);
    torch::Tensor C = torch::zeros({batch_size, N, N}, torch::kCUDA);

    C = torch::matmul(A, B);

    std::cout<<"TimePreModel --"; printCurrentTime(); std::cout << std::endl;
    sleep(1);
    std::cout<<"TimePreModelSleep --"; printCurrentTime(); std::cout << std::endl;

    C = torch::matmul(A, B);

    std::cout<< "TimeLazyLoad --"; printCurrentTime(); std::cout << std::endl;
    sleep(1);
    std::cout<< "TimeLazyLoadSleep --"; printCurrentTime(); std::cout << std::endl;

    for(int i = 0; i < 1000; i++) {
	C = torch::matmul(A, B);
    }

    return 0;
}

