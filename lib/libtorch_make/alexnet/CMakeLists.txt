cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(alexnet_inf LANGUAGES CUDA CXX)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})  # <-- add this line                                                                                                                                             
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_50)

cuda_add_executable(alexnet_inf alexnet_inf.cu)
target_link_libraries(alexnet_inf "${TORCH_LIBRARIES}")
