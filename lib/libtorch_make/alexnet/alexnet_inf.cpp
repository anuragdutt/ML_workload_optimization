#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>  // Added vector header
#include <chrono>

int main(int argc, char** argv) {
    int batch_size = std::stoi(argv[1]);  // Parse batch size argument

    // Load AlexNet model
    torch::jit::script::Module module = torch::jit::load("/home/pace/execution/ML_workload_optimization/models/alexnet.pt");
    module.to(at::kCUDA);  // Move model to GPU

    // Generate random image tensor of size (224,224,3)
    auto image = torch::randn({batch_size, 3, 224, 224}).to(at::kCUDA);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(image);

    std::cout << "Processed Images" << std::endl;

    std::cout << "TimePreModel --" << std::to_string(std::time(nullptr)) << std::endl;
    // Perform inference
    // auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 64/batch_size; i++) {
        for (int j = 0; j < 50; j++) {
            module.forward(inputs);
	    // std::cout << "Output: " << output << std::endl;
        }
    }
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Print duration
    // std::cout << "Inference duration: " << duration << " microseconds." << std::endl;

    return 0;
}
