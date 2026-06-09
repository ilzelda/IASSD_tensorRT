#ifndef _SAMPLING_GPU_RAW_H
#define _SAMPLING_GPU_RAW_H

#include <cuda_runtime_api.h>

// PyTorch, ONNX Runtime, TensorRT 경로가 함께 참조할 수 있는 포인터 기반 CUDA launcher 선언이다.
void gather_points_kernel_launcher_fast(int b, int c, int n, int npoints,
    const float *points, const int *idx, float *out, cudaStream_t stream = 0);

void gather_points_grad_kernel_launcher_fast(int b, int c, int n, int npoints,
    const float *grad_out, const int *idx, float *grad_points, cudaStream_t stream = 0);

void farthest_point_sampling_kernel_launcher(int b, int n, int m,
    const float *dataset, float *temp, int *idxs, cudaStream_t stream = 0);

void furthest_point_sampling_with_dist_kernel_launcher(int b, int n, int m,
    const float *dataset, float *temp, int *idxs, cudaStream_t stream = 0);

#endif
