#include <cuda_runtime_api.h>
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "sampling_gpu_raw.h"
#include "ball_query_gpu_raw.h"
#include "group_points_gpu_raw.h"

namespace {

int64_t ReadIntAttr(const Ort::ConstKernelInfo& info, const char* primary_name, const char* fallback_name) {
    try {
        return info.GetAttribute<int64_t>(primary_name);
    } catch (const Ort::Exception&) {
        return info.GetAttribute<int64_t>(fallback_name);
    }
}

float ReadFloatAttr(const Ort::ConstKernelInfo& info, const char* primary_name, const char* fallback_name) {
    try {
        return info.GetAttribute<float>(primary_name);
    } catch (const Ort::Exception&) {
        return info.GetAttribute<float>(fallback_name);
    }
}

void CheckCuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

bool IsDebugRangeCheckEnabled() {
    const char* value = std::getenv("IASSD_DEBUG_RANGE_CHECK");
    return value != nullptr && std::string(value) != "0";
}

void CheckIndexRangeOnHost(
    const int32_t* host_idx,
    size_t count,
    int64_t upper_bound,
    const char* op_name,
    const std::vector<int64_t>& reference_shape,
    const std::vector<int64_t>& idx_shape
) {
    if (!IsDebugRangeCheckEnabled()) {
        return;
    }

    int32_t min_idx = count == 0 ? 0 : host_idx[0];
    int32_t max_idx = count == 0 ? 0 : host_idx[0];
    for (size_t i = 0; i < count; ++i) {
        const int32_t value = host_idx[i];
        min_idx = std::min(min_idx, value);
        max_idx = std::max(max_idx, value);
    }

    if (min_idx < 0 || max_idx >= upper_bound) {
        throw std::runtime_error(
            std::string(op_name) +
            " idx range 오류: min_idx=" + std::to_string(min_idx) +
            ", max_idx=" + std::to_string(max_idx) +
            ", upper_bound=" + std::to_string(upper_bound) +
            ", reference_shape=" + std::to_string(reference_shape.size() > 0 ? reference_shape[0] : -1) +
            "/" + std::to_string(reference_shape.size() > 1 ? reference_shape[1] : -1) +
            "/" + std::to_string(reference_shape.size() > 2 ? reference_shape[2] : -1) +
            ", idx_shape=" + std::to_string(idx_shape.size() > 0 ? idx_shape[0] : -1) +
            "/" + std::to_string(idx_shape.size() > 1 ? idx_shape[1] : -1) +
            "/" + std::to_string(idx_shape.size() > 2 ? idx_shape[2] : -1)
        );
    }
}

int32_t* CopyHostInt32IndexToDevice(const int32_t* host_idx, size_t count) {
    int32_t* device_i32 = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_i32), count * sizeof(int32_t)), "int32 idx cudaMalloc 실패");
    try {
        CheckCuda(
            cudaMemcpy(device_i32, host_idx, count * sizeof(int32_t), cudaMemcpyHostToDevice),
            "int32 idx cudaMemcpy 실패"
        );
    } catch (...) {
        cudaFree(device_i32);
        throw;
    }
    return device_i32;
}

int32_t* CopyHostInt64IndexToInt32Device(const int64_t* host_i64, size_t count, std::vector<int32_t>* host_i32_out) {
    std::vector<int32_t> host_i32(count);
    for (size_t i = 0; i < count; ++i) {
        host_i32[i] = static_cast<int32_t>(host_i64[i]);
    }

    if (host_i32_out != nullptr) {
        *host_i32_out = host_i32;
    }

    int32_t* device_i32 = nullptr;
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_i32), count * sizeof(int32_t)), "int32 idx cudaMalloc 실패");
    try {
        CheckCuda(
            cudaMemcpy(device_i32, host_i32.data(), count * sizeof(int32_t), cudaMemcpyHostToDevice),
            "int32 idx cudaMemcpy 실패"
        );
    } catch (...) {
        cudaFree(device_i32);
        throw;
    }
    return device_i32;
}

struct FarthestPointSamplingKernel {
    FarthestPointSamplingKernel(const OrtApi& api, const OrtKernelInfo* info)
        : npoint_(ReadIntAttr(Ort::ConstKernelInfo(info), "npoint_i", "npoint")) {
        (void)api;
        if (npoint_ <= 0) {
            throw std::runtime_error("FarthestPointSampling의 npoint는 0보다 커야 합니다.");
        }
    }

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);
        const Ort::ConstValue xyz = ctx.GetInput(0);
        const Ort::TensorTypeAndShapeInfo xyz_info = xyz.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> xyz_shape = xyz_info.GetShape();

        if (xyz_shape.size() != 3 || xyz_shape[2] != 3) {
            throw std::runtime_error("FarthestPointSampling 입력 xyz shape는 [B, N, 3]이어야 합니다.");
        }

        const int64_t batch_size_i64 = xyz_shape[0];
        const int64_t num_points_i64 = xyz_shape[1];
        if (batch_size_i64 <= 0 || num_points_i64 <= 0) {
            throw std::runtime_error("FarthestPointSampling 입력 xyz의 B와 N은 0보다 커야 합니다.");
        }
        if (npoint_ > num_points_i64) {
            throw std::runtime_error("FarthestPointSampling의 npoint는 입력 포인트 수보다 클 수 없습니다.");
        }

        const std::vector<int64_t> output_shape = {batch_size_i64, npoint_};
        Ort::UnownedValue output = ctx.GetOutput(0, output_shape);

        const int batch_size = static_cast<int>(batch_size_i64);
        const int num_points = static_cast<int>(num_points_i64);
        const int npoint = static_cast<int>(npoint_);
        const float* xyz_data = xyz.GetTensorData<float>();
        int32_t* output_data = output.GetTensorMutableData<int32_t>();
        static_assert(sizeof(int32_t) == sizeof(int), "FPS CUDA launcher는 32-bit int index를 사용합니다.");

        float* temp = nullptr;
        const size_t temp_bytes = static_cast<size_t>(batch_size) * static_cast<size_t>(num_points) * sizeof(float);
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&temp), temp_bytes), "FPS temp cudaMalloc 실패");

        try {
            // 기존 PyTorch op와 동일하게 temp를 1e10으로 초기화한다.
            std::vector<float> temp_init(static_cast<size_t>(batch_size) * static_cast<size_t>(num_points), 1.0e10f);
            CheckCuda(cudaMemcpy(temp, temp_init.data(), temp_bytes, cudaMemcpyHostToDevice), "FPS temp 초기값 복사 실패");
            farthest_point_sampling_kernel_launcher(
                batch_size,
                num_points,
                npoint,
                xyz_data,
                temp,
                reinterpret_cast<int*>(output_data)
            );
            CheckCuda(cudaDeviceSynchronize(), "FPS kernel 동기화 실패");
        } catch (...) {
            cudaFree(temp);
            throw;
        }

        CheckCuda(cudaFree(temp), "FPS temp cudaFree 실패");
    }

    int64_t npoint_;
};

struct FarthestPointSamplingOp final : Ort::CustomOpBase<FarthestPointSamplingOp, FarthestPointSamplingKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new FarthestPointSamplingKernel(api, info);
    }

    const char* GetName() const {
        return "FarthestPointSampling";
    }

    const char* GetExecutionProviderType() const {
        return "CUDAExecutionProvider";
    }

    size_t GetInputTypeCount() const {
        return 1;
    }

    ONNXTensorElementDataType GetInputType(size_t index) const {
        if (index != 0) {
            throw std::runtime_error("FarthestPointSampling 입력 index가 범위를 벗어났습니다.");
        }
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }

    size_t GetOutputTypeCount() const {
        return 1;
    }

    ONNXTensorElementDataType GetOutputType(size_t index) const {
        if (index != 0) {
            throw std::runtime_error("FarthestPointSampling 출력 index가 범위를 벗어났습니다.");
        }
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    }
};

struct GatherPointsKernel {
    GatherPointsKernel(const OrtApi& api, const OrtKernelInfo* info) {
        (void)api;
        (void)info;
    }

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);
        const Ort::ConstValue features = ctx.GetInput(0);
        const Ort::ConstValue idx = ctx.GetInput(1);
        const Ort::TensorTypeAndShapeInfo features_info = features.GetTensorTypeAndShapeInfo();
        const Ort::TensorTypeAndShapeInfo idx_info = idx.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> features_shape = features_info.GetShape();
        const std::vector<int64_t> idx_shape = idx_info.GetShape();

        if (features_shape.size() != 3) {
            throw std::runtime_error("GatherPoints 입력 features shape는 [B, C, N]이어야 합니다.");
        }
        if (idx_shape.size() != 2) {
            throw std::runtime_error("GatherPoints 입력 idx shape는 [B, npoint]이어야 합니다.");
        }

        const int64_t batch_size_i64 = features_shape[0];
        const int64_t channel_i64 = features_shape[1];
        const int64_t num_points_i64 = features_shape[2];
        const int64_t idx_batch_size_i64 = idx_shape[0];
        const int64_t npoint_i64 = idx_shape[1];

        if (batch_size_i64 <= 0 || channel_i64 <= 0 || num_points_i64 <= 0 || npoint_i64 <= 0) {
            throw std::runtime_error("GatherPoints 입력 shape의 모든 차원은 0보다 커야 합니다.");
        }
        if (idx_batch_size_i64 != batch_size_i64) {
            throw std::runtime_error("GatherPoints features와 idx의 batch 크기가 일치해야 합니다.");
        }

        const std::vector<int64_t> output_shape = {batch_size_i64, channel_i64, npoint_i64};
        Ort::UnownedValue output = ctx.GetOutput(0, output_shape);

        const int batch_size = static_cast<int>(batch_size_i64);
        const int channel = static_cast<int>(channel_i64);
        const int num_points = static_cast<int>(num_points_i64);
        const int npoint = static_cast<int>(npoint_i64);
        const size_t idx_count = static_cast<size_t>(batch_size) * static_cast<size_t>(npoint);
        const float* features_data = features.GetTensorData<float>();
        float* output_data = output.GetTensorMutableData<float>();
        static_assert(sizeof(int32_t) == sizeof(int), "GatherPoints CUDA launcher는 32-bit int index를 사용합니다.");

        int32_t* device_idx = nullptr;
        const int32_t* host_idx = nullptr;
        std::vector<int32_t> converted_host_idx;
        const ONNXTensorElementDataType idx_type = idx_info.GetElementType();
        if (idx_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            host_idx = idx.GetTensorData<int32_t>();
            device_idx = CopyHostInt32IndexToDevice(host_idx, idx_count);
        } else if (idx_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            device_idx = CopyHostInt64IndexToInt32Device(idx.GetTensorData<int64_t>(), idx_count, &converted_host_idx);
            host_idx = converted_host_idx.data();
        } else {
            throw std::runtime_error("GatherPoints idx dtype은 int32 또는 int64여야 합니다.");
        }

        CheckIndexRangeOnHost(
            host_idx,
            idx_count,
            num_points_i64,
            "GatherPoints",
            features_shape,
            idx_shape
        );

        try {
            gather_points_kernel_launcher_fast(
                batch_size,
                channel,
                num_points,
                npoint,
                features_data,
                reinterpret_cast<const int*>(device_idx),
                output_data
            );
            CheckCuda(cudaDeviceSynchronize(), "GatherPoints kernel 동기화 실패");
        } catch (...) {
            if (device_idx != nullptr) {
                cudaFree(device_idx);
            }
            throw;
        }
        if (device_idx != nullptr) {
            CheckCuda(cudaFree(device_idx), "GatherPoints device_idx cudaFree 실패");
        }
    }
};

struct GatherPointsOp final : Ort::CustomOpBase<GatherPointsOp, GatherPointsKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new GatherPointsKernel(api, info);
    }

    const char* GetName() const {
        return "GatherPoints";
    }

    const char* GetExecutionProviderType() const {
        return "CUDAExecutionProvider";
    }

    size_t GetInputTypeCount() const {
        return 2;
    }

    ONNXTensorElementDataType GetInputType(size_t index) const {
        if (index == 0) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        }
        if (index == 1) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        }
        throw std::runtime_error("GatherPoints 입력 index가 범위를 벗어났습니다.");
    }

    OrtMemType GetInputMemoryType(size_t index) const {
        if (index == 1) {
            return OrtMemTypeCPUInput;
        }
        return OrtMemTypeDefault;
    }

    size_t GetOutputTypeCount() const {
        return 1;
    }

    ONNXTensorElementDataType GetOutputType(size_t index) const {
        if (index != 0) {
            throw std::runtime_error("GatherPoints 출력 index가 범위를 벗어났습니다.");
        }
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

struct BallQueryKernel {
    BallQueryKernel(const OrtApi& api, const OrtKernelInfo* info)
        : radius_(ReadFloatAttr(Ort::ConstKernelInfo(info), "radius_f", "radius")),
          nsample_(ReadIntAttr(Ort::ConstKernelInfo(info), "nsample_i", "nsample")) {
        (void)api;
        if (radius_ <= 0.0f) {
            throw std::runtime_error("BallQuery의 radius는 0보다 커야 합니다.");
        }
        if (nsample_ <= 0) {
            throw std::runtime_error("BallQuery의 nsample은 0보다 커야 합니다.");
        }
    }

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);
        const Ort::ConstValue xyz = ctx.GetInput(0);
        const Ort::ConstValue new_xyz = ctx.GetInput(1);
        const Ort::TensorTypeAndShapeInfo xyz_info = xyz.GetTensorTypeAndShapeInfo();
        const Ort::TensorTypeAndShapeInfo new_xyz_info = new_xyz.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> xyz_shape = xyz_info.GetShape();
        const std::vector<int64_t> new_xyz_shape = new_xyz_info.GetShape();

        if (xyz_shape.size() != 3 || xyz_shape[2] != 3) {
            throw std::runtime_error("BallQuery 입력 xyz shape는 [B, N, 3]이어야 합니다.");
        }
        if (new_xyz_shape.size() != 3 || new_xyz_shape[2] != 3) {
            throw std::runtime_error("BallQuery 입력 new_xyz shape는 [B, npoint, 3]이어야 합니다.");
        }

        const int64_t batch_size_i64 = xyz_shape[0];
        const int64_t num_points_i64 = xyz_shape[1];
        const int64_t new_batch_size_i64 = new_xyz_shape[0];
        const int64_t npoint_i64 = new_xyz_shape[1];

        if (batch_size_i64 <= 0 || num_points_i64 <= 0 || npoint_i64 <= 0) {
            throw std::runtime_error("BallQuery 입력 shape의 B, N, npoint는 0보다 커야 합니다.");
        }
        if (new_batch_size_i64 != batch_size_i64) {
            throw std::runtime_error("BallQuery xyz와 new_xyz의 batch 크기가 일치해야 합니다.");
        }

        const std::vector<int64_t> output_shape = {batch_size_i64, npoint_i64, nsample_};
        Ort::UnownedValue output = ctx.GetOutput(0, output_shape);

        const int batch_size = static_cast<int>(batch_size_i64);
        const int num_points = static_cast<int>(num_points_i64);
        const int npoint = static_cast<int>(npoint_i64);
        const int nsample = static_cast<int>(nsample_);
        const float* xyz_data = xyz.GetTensorData<float>();
        const float* new_xyz_data = new_xyz.GetTensorData<float>();
        int32_t* output_data = output.GetTensorMutableData<int32_t>();
        static_assert(sizeof(int32_t) == sizeof(int), "BallQuery CUDA launcher는 32-bit int index를 사용합니다.");

        const size_t output_bytes =
            static_cast<size_t>(batch_size) * static_cast<size_t>(npoint) * static_cast<size_t>(nsample) * sizeof(int32_t);
        CheckCuda(cudaMemset(output_data, 0x00, output_bytes), "BallQuery output cudaMemset 실패");

        ball_query_kernel_launcher_fast(
            batch_size,
            num_points,
            npoint,
            radius_,
            nsample,
            new_xyz_data,
            xyz_data,
            reinterpret_cast<int*>(output_data)
        );
        CheckCuda(cudaDeviceSynchronize(), "BallQuery kernel 동기화 실패");
    }

    float radius_;
    int64_t nsample_;
};

struct BallQueryOp final : Ort::CustomOpBase<BallQueryOp, BallQueryKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new BallQueryKernel(api, info);
    }

    const char* GetName() const {
        return "BallQuery";
    }

    const char* GetExecutionProviderType() const {
        return "CUDAExecutionProvider";
    }

    size_t GetInputTypeCount() const {
        return 2;
    }

    ONNXTensorElementDataType GetInputType(size_t index) const {
        if (index == 0 || index == 1) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        }
        throw std::runtime_error("BallQuery 입력 index가 범위를 벗어났습니다.");
    }

    size_t GetOutputTypeCount() const {
        return 1;
    }

    ONNXTensorElementDataType GetOutputType(size_t index) const {
        if (index != 0) {
            throw std::runtime_error("BallQuery 출력 index가 범위를 벗어났습니다.");
        }
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    }
};

struct GroupPointsKernel {
    GroupPointsKernel(const OrtApi& api, const OrtKernelInfo* info) {
        (void)api;
        (void)info;
    }

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);
        const Ort::ConstValue features = ctx.GetInput(0);
        const Ort::ConstValue idx = ctx.GetInput(1);
        const Ort::TensorTypeAndShapeInfo features_info = features.GetTensorTypeAndShapeInfo();
        const Ort::TensorTypeAndShapeInfo idx_info = idx.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> features_shape = features_info.GetShape();
        const std::vector<int64_t> idx_shape = idx_info.GetShape();

        if (features_shape.size() != 3) {
            throw std::runtime_error("GroupPoints 입력 features shape는 [B, C, N]이어야 합니다.");
        }
        if (idx_shape.size() != 3) {
            throw std::runtime_error("GroupPoints 입력 idx shape는 [B, npoint, nsample]이어야 합니다.");
        }

        const int64_t batch_size_i64 = features_shape[0];
        const int64_t channel_i64 = features_shape[1];
        const int64_t num_points_i64 = features_shape[2];
        const int64_t idx_batch_size_i64 = idx_shape[0];
        const int64_t npoint_i64 = idx_shape[1];
        const int64_t nsample_i64 = idx_shape[2];

        if (batch_size_i64 <= 0 || channel_i64 <= 0 || num_points_i64 <= 0 || npoint_i64 <= 0 || nsample_i64 <= 0) {
            throw std::runtime_error("GroupPoints 입력 shape의 모든 차원은 0보다 커야 합니다.");
        }
        if (idx_batch_size_i64 != batch_size_i64) {
            throw std::runtime_error("GroupPoints features와 idx의 batch 크기가 일치해야 합니다.");
        }

        const std::vector<int64_t> output_shape = {batch_size_i64, channel_i64, npoint_i64, nsample_i64};
        Ort::UnownedValue output = ctx.GetOutput(0, output_shape);

        const int batch_size = static_cast<int>(batch_size_i64);
        const int channel = static_cast<int>(channel_i64);
        const int num_points = static_cast<int>(num_points_i64);
        const int npoint = static_cast<int>(npoint_i64);
        const int nsample = static_cast<int>(nsample_i64);
        const size_t idx_count =
            static_cast<size_t>(batch_size) * static_cast<size_t>(npoint) * static_cast<size_t>(nsample);
        const float* features_data = features.GetTensorData<float>();
        float* output_data = output.GetTensorMutableData<float>();
        static_assert(sizeof(int32_t) == sizeof(int), "GroupPoints CUDA launcher는 32-bit int index를 사용합니다.");

        int32_t* device_idx = nullptr;
        const int32_t* host_idx = nullptr;
        std::vector<int32_t> converted_host_idx;
        const ONNXTensorElementDataType idx_type = idx_info.GetElementType();
        if (idx_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
            host_idx = idx.GetTensorData<int32_t>();
            device_idx = CopyHostInt32IndexToDevice(host_idx, idx_count);
        } else if (idx_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            device_idx = CopyHostInt64IndexToInt32Device(idx.GetTensorData<int64_t>(), idx_count, &converted_host_idx);
            host_idx = converted_host_idx.data();
        } else {
            throw std::runtime_error("GroupPoints idx dtype은 int32 또는 int64여야 합니다.");
        }

        CheckIndexRangeOnHost(
            host_idx,
            idx_count,
            num_points_i64,
            "GroupPoints",
            features_shape,
            idx_shape
        );

        try {
            group_points_kernel_launcher_fast(
                batch_size,
                channel,
                num_points,
                npoint,
                nsample,
                features_data,
                reinterpret_cast<const int*>(device_idx),
                output_data
            );
            CheckCuda(cudaDeviceSynchronize(), "GroupPoints kernel 동기화 실패");
        } catch (...) {
            if (device_idx != nullptr) {
                cudaFree(device_idx);
            }
            throw;
        }
        if (device_idx != nullptr) {
            CheckCuda(cudaFree(device_idx), "GroupPoints device_idx cudaFree 실패");
        }
    }
};

struct GroupPointsOp final : Ort::CustomOpBase<GroupPointsOp, GroupPointsKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new GroupPointsKernel(api, info);
    }

    const char* GetName() const {
        return "GroupPoints";
    }

    const char* GetExecutionProviderType() const {
        return "CUDAExecutionProvider";
    }

    size_t GetInputTypeCount() const {
        return 2;
    }

    ONNXTensorElementDataType GetInputType(size_t index) const {
        if (index == 0) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        }
        if (index == 1) {
            return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        }
        throw std::runtime_error("GroupPoints 입력 index가 범위를 벗어났습니다.");
    }

    OrtMemType GetInputMemoryType(size_t index) const {
        if (index == 1) {
            return OrtMemTypeCPUInput;
        }
        return OrtMemTypeDefault;
    }

    size_t GetOutputTypeCount() const {
        return 1;
    }

    ONNXTensorElementDataType GetOutputType(size_t index) const {
        if (index != 0) {
            throw std::runtime_error("GroupPoints 출력 index가 범위를 벗어났습니다.");
        }
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

FarthestPointSamplingOp g_farthest_point_sampling_op;
GatherPointsOp g_gather_points_op;
BallQueryOp g_ball_query_op;
GroupPointsOp g_group_points_op;

}

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
    const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
    Ort::InitApi(api);
    OrtCustomOpDomain* domain = nullptr;

    OrtStatus* status = api->CreateCustomOpDomain("IASSD", &domain);
    if (status != nullptr) {
        return status;
    }

    status = api->CustomOpDomain_Add(domain, &g_farthest_point_sampling_op);
    if (status != nullptr) {
        api->ReleaseCustomOpDomain(domain);
        return status;
    }

    status = api->CustomOpDomain_Add(domain, &g_gather_points_op);
    if (status != nullptr) {
        api->ReleaseCustomOpDomain(domain);
        return status;
    }

    status = api->CustomOpDomain_Add(domain, &g_ball_query_op);
    if (status != nullptr) {
        api->ReleaseCustomOpDomain(domain);
        return status;
    }

    status = api->CustomOpDomain_Add(domain, &g_group_points_op);
    if (status != nullptr) {
        api->ReleaseCustomOpDomain(domain);
        return status;
    }

    status = api->AddCustomOpDomain(options, domain);
    if (status != nullptr) {
        api->ReleaseCustomOpDomain(domain);
        return status;
    }

    return nullptr;
}
