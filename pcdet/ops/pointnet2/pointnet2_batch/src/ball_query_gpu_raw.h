#ifndef _BALL_QUERY_GPU_RAW_H
#define _BALL_QUERY_GPU_RAW_H

#include <cuda_runtime_api.h>

// PyTorch, ONNX Runtime, TensorRT 경로가 함께 참조할 수 있는 포인터 기반 BallQuery CUDA launcher 선언이다.
void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample,
    const float *new_xyz, const float *xyz, int *idx, cudaStream_t stream = 0);

void ball_query_dilated_kernel_launcher_fast(int b, int n, int m, float max_radius, float min_radius, int nsample,
    const float *new_xyz, const float *xyz, int *idx, cudaStream_t stream = 0);

#endif
