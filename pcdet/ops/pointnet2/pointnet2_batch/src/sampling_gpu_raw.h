#ifndef _SAMPLING_GPU_RAW_H
#define _SAMPLING_GPU_RAW_H

// PyTorch, ONNX Runtime, TensorRT 경로가 함께 참조할 수 있는 포인터 기반 CUDA launcher 선언이다.
void gather_points_kernel_launcher_fast(int b, int c, int n, int npoints,
    const float *points, const int *idx, float *out);

void gather_points_grad_kernel_launcher_fast(int b, int c, int n, int npoints,
    const float *grad_out, const int *idx, float *grad_points);

void farthest_point_sampling_kernel_launcher(int b, int n, int m,
    const float *dataset, float *temp, int *idxs);

void furthest_point_sampling_with_dist_kernel_launcher(int b, int n, int m,
    const float *dataset, float *temp, int *idxs);

#endif
