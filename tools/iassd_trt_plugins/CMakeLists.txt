cmake_minimum_required(VERSION 3.16)

project(iassd_trt_plugins LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    string(REPLACE "." "" CMAKE_CUDA_ARCHITECTURES "$ENV{TORCH_CUDA_ARCH_LIST}")
endif()

find_library(CUDART_LIBRARY cudart
    HINTS
        "$ENV{CUDA_HOME}/lib64"
        "$ENV{CUDA_PATH}/lib64"
        "/usr/local/cuda/lib64"
)

find_library(NVINFER_LIBRARY nvinfer
    HINTS
        "/usr/lib/aarch64-linux-gnu"
        "/usr/local/tensorrt/lib"
)

find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin
    HINTS
        "/usr/lib/aarch64-linux-gnu"
        "/usr/local/tensorrt/lib"
)

if(NOT CUDART_LIBRARY)
    message(FATAL_ERROR "CUDA runtime library cudart를 찾을 수 없습니다.")
endif()

if(NOT NVINFER_LIBRARY)
    message(FATAL_ERROR "TensorRT library nvinfer를 찾을 수 없습니다.")
endif()

if(NOT NVINFER_PLUGIN_LIBRARY)
    message(FATAL_ERROR "TensorRT plugin library nvinfer_plugin을 찾을 수 없습니다.")
endif()

add_library(iassd_trt_plugins SHARED
    farthest_point_sampling_plugin.cu
    ../../pcdet/ops/pointnet2/pointnet2_batch/src/sampling_gpu.cu
)

target_include_directories(iassd_trt_plugins PRIVATE
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    "/usr/include/aarch64-linux-gnu"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../pcdet/ops/pointnet2/pointnet2_batch/src"
)

target_link_libraries(iassd_trt_plugins PRIVATE
    "${CUDART_LIBRARY}"
    "${NVINFER_LIBRARY}"
    "${NVINFER_PLUGIN_LIBRARY}"
)

set_target_properties(iassd_trt_plugins PROPERTIES
    OUTPUT_NAME "iassd_trt_plugins"
    CUDA_SEPARABLE_COMPILATION ON
)
