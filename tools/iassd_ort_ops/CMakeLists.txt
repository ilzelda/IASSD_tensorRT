cmake_minimum_required(VERSION 3.16)

project(iassd_ort_ops LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    string(REPLACE "." "" CMAKE_CUDA_ARCHITECTURES "$ENV{TORCH_CUDA_ARCH_LIST}")
endif()

if(NOT DEFINED ONNXRUNTIME_ROOT)
    execute_process(
        COMMAND python3 -c "import pathlib, onnxruntime; print(pathlib.Path(onnxruntime.__file__).resolve().parent)"
        OUTPUT_VARIABLE ONNXRUNTIME_ROOT
        RESULT_VARIABLE ONNXRUNTIME_ROOT_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT ONNXRUNTIME_ROOT_RESULT EQUAL 0)
        message(FATAL_ERROR "Python에서 onnxruntime 패키지를 import할 수 없습니다. ONNXRUNTIME_ROOT를 직접 지정하거나 onnxruntime-gpu를 설치하세요.")
    endif()
endif()

if(NOT DEFINED ONNXRUNTIME_INCLUDE_DIR)
    if(EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
        set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
    elseif(EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session/onnxruntime_cxx_api.h")
        set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session")
    else()
        set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
    endif()
endif()

if(NOT DEFINED ONNXRUNTIME_LIB_DIR)
    set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/capi")
endif()

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "onnxruntime_cxx_api.h를 찾을 수 없습니다: ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

find_library(CUDART_LIBRARY cudart
    HINTS
        "$ENV{CUDA_HOME}/lib64"
        "$ENV{CUDA_PATH}/lib64"
        "/usr/local/cuda/lib64"
)

if(NOT CUDART_LIBRARY)
    message(FATAL_ERROR "CUDA runtime library cudart를 찾을 수 없습니다.")
endif()

add_library(iassd_ort_ops SHARED
    farthest_point_sampling_op.cc
    ../../pcdet/ops/pointnet2/pointnet2_batch/src/sampling_gpu.cu
    ../../pcdet/ops/pointnet2/pointnet2_batch/src/ball_query_gpu.cu
    ../../pcdet/ops/pointnet2/pointnet2_batch/src/group_points_gpu.cu
)

target_include_directories(iassd_ort_ops PRIVATE
    "${ONNXRUNTIME_INCLUDE_DIR}"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../pcdet/ops/pointnet2/pointnet2_batch/src"
)

target_link_libraries(iassd_ort_ops PRIVATE
    "${CUDART_LIBRARY}"
)

find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS "${ONNXRUNTIME_LIB_DIR}" NO_DEFAULT_PATH)
if(ONNXRUNTIME_LIBRARY)
    target_link_libraries(iassd_ort_ops PRIVATE "${ONNXRUNTIME_LIBRARY}")
else()
    message(STATUS "libonnxruntime.so를 찾지 못해 명시 링크를 생략합니다. custom op는 ORT 프로세스가 전달하는 OrtApi로 동작합니다.")
endif()

set_target_properties(iassd_ort_ops PROPERTIES
    OUTPUT_NAME "iassd_ort_ops"
    CUDA_SEPARABLE_COMPILATION ON
)
