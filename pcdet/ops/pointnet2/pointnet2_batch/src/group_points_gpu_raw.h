#ifndef _GROUP_POINTS_GPU_RAW_H
#define _GROUP_POINTS_GPU_RAW_H

#include <cuda_runtime_api.h>

// PyTorch, ONNX Runtime, TensorRT 경로가 함께 참조할 수 있는 포인터 기반 GroupPoints CUDA launcher 선언이다.
void group_points_kernel_launcher_fast(int b, int c, int n, int npoints, int nsample,
    const float *points, const int *idx, float *out, cudaStream_t stream = 0);

void group_points_grad_kernel_launcher_fast(int b, int c, int n, int npoints, int nsample,
    const float *grad_out, const int *idx, float *grad_points, cudaStream_t stream = 0);

#endif
