cmake_minimum_required( VERSION 2.8.0 )


project( try_thrust_man_vec LANGUAGES CXX CUDA )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA QUIET REQUIRED)

# Selects GPU arch flags for nvcc based on target_CUDA_architectures
# Xavier is compute capability 7.2, Tegra X2 is 6.1
#cuda_select_nvcc_arch_flags(ARCH_FLAGS 7.2)
cuda_select_nvcc_arch_flags(ARCH_FLAGS 6.1)

list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
LIST(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

set(CUDA_VERBOSE_BUILD ON)

set(EXEC_NAME "${CMAKE_PROJECT_NAME}")

set(SRCS
   "src/utils.cpp"
   "src/try_thrust_man_vec.cu"
   )

include_directories( "include" )

cuda_add_executable( "${EXEC_NAME}" "${SRCS}" )
target_link_libraries( "${EXEC_NAME}" m )


