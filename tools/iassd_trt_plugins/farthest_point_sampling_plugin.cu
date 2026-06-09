#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <string>
#include <vector>

#include "sampling_gpu_raw.h"

namespace {

constexpr char const* kPluginName = "FarthestPointSampling";
constexpr char const* kPluginVersion = "1";

void WriteInt32(char*& buffer, int32_t value) {
    std::memcpy(buffer, &value, sizeof(value));
    buffer += sizeof(value);
}

int32_t ReadInt32(char const*& buffer) {
    int32_t value = 0;
    std::memcpy(&value, buffer, sizeof(value));
    buffer += sizeof(value);
    return value;
}

__global__ void FillFloatKernel(float* output, int64_t count, float value) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = value;
    }
}

void FillFloatAsync(float* output, int64_t count, float value, cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    FillFloatKernel<<<blocks, threads, 0, stream>>>(output, count, value);
}

class FarthestPointSamplingPlugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    explicit FarthestPointSamplingPlugin(int32_t npoint)
        : npoint_(npoint) {
    }

    FarthestPointSamplingPlugin(void const* serial_data, size_t serial_length) {
        if (serial_data == nullptr || serial_length != sizeof(int32_t)) {
            npoint_ = 0;
            return;
        }
        auto const* ptr = static_cast<char const*>(serial_data);
        npoint_ = ReadInt32(ptr);
    }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        auto* plugin = new FarthestPointSamplingPlugin(npoint_);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t output_index,
        nvinfer1::DimsExprs const* inputs,
        int32_t nb_inputs,
        nvinfer1::IExprBuilder& expr_builder) noexcept override {
        (void)output_index;
        (void)nb_inputs;
        nvinfer1::DimsExprs output;
        output.nbDims = 2;
        output.d[0] = inputs[0].d[0];
        output.d[1] = expr_builder.constant(npoint_);
        return output;
    }

    bool supportsFormatCombination(
        int32_t pos,
        nvinfer1::PluginTensorDesc const* in_out,
        int32_t nb_inputs,
        int32_t nb_outputs) noexcept override {
        (void)nb_inputs;
        (void)nb_outputs;
        if (in_out[pos].format != nvinfer1::TensorFormat::kLINEAR) {
            return false;
        }
        if (pos == 0) {
            return in_out[pos].type == nvinfer1::DataType::kFLOAT;
        }
        if (pos == 1) {
            return in_out[pos].type == nvinfer1::DataType::kINT32;
        }
        return false;
    }

    void configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* in,
        int32_t nb_inputs,
        nvinfer1::DynamicPluginTensorDesc const* out,
        int32_t nb_outputs) noexcept override {
        (void)in;
        (void)nb_inputs;
        (void)out;
        (void)nb_outputs;
    }

    size_t getWorkspaceSize(
        nvinfer1::PluginTensorDesc const* inputs,
        int32_t nb_inputs,
        nvinfer1::PluginTensorDesc const* outputs,
        int32_t nb_outputs) const noexcept override {
        (void)nb_inputs;
        (void)outputs;
        (void)nb_outputs;
        if (inputs[0].dims.nbDims != 3) {
            return 0;
        }
        const int32_t batch_size = inputs[0].dims.d[0];
        const int32_t num_points = inputs[0].dims.d[1];
        return static_cast<size_t>(batch_size) * static_cast<size_t>(num_points) * sizeof(float);
    }

    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* input_desc,
        nvinfer1::PluginTensorDesc const* output_desc,
        void const* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override {
        (void)output_desc;
        if (input_desc[0].dims.nbDims != 3 || input_desc[0].dims.d[2] != 3 || workspace == nullptr) {
            return 1;
        }

        const int32_t batch_size = input_desc[0].dims.d[0];
        const int32_t num_points = input_desc[0].dims.d[1];
        float const* xyz = static_cast<float const*>(inputs[0]);
        float* temp = static_cast<float*>(workspace);
        int32_t* idx = static_cast<int32_t*>(outputs[0]);

        // 기존 IA-SSD/PyTorch FPS와 동일한 초기 거리 값을 사용한다.
        FillFloatAsync(
            temp,
            static_cast<int64_t>(batch_size) * static_cast<int64_t>(num_points),
            1.0e10f,
            stream);
        farthest_point_sampling_kernel_launcher(
            batch_size,
            num_points,
            npoint_,
            xyz,
            temp,
            reinterpret_cast<int*>(idx),
            stream);
        return cudaGetLastError() == cudaSuccess ? 0 : 1;
    }

    nvinfer1::DataType getOutputDataType(
        int32_t index,
        nvinfer1::DataType const* input_types,
        int32_t nb_inputs) const noexcept override {
        (void)index;
        (void)input_types;
        (void)nb_inputs;
        return nvinfer1::DataType::kINT32;
    }

    char const* getPluginType() const noexcept override {
        return kPluginName;
    }

    char const* getPluginVersion() const noexcept override {
        return kPluginVersion;
    }

    int32_t getNbOutputs() const noexcept override {
        return 1;
    }

    int32_t initialize() noexcept override {
        return npoint_ > 0 ? 0 : 1;
    }

    void terminate() noexcept override {
    }

    size_t getSerializationSize() const noexcept override {
        return sizeof(int32_t);
    }

    void serialize(void* buffer) const noexcept override {
        auto* ptr = static_cast<char*>(buffer);
        WriteInt32(ptr, npoint_);
    }

    void destroy() noexcept override {
        delete this;
    }

    void setPluginNamespace(char const* plugin_namespace) noexcept override {
        namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
    }

    char const* getPluginNamespace() const noexcept override {
        return namespace_.c_str();
    }

    void attachToContext(
        cudnnContext* cudnn_context,
        cublasContext* cublas_context,
        nvinfer1::IGpuAllocator* gpu_allocator) noexcept override {
        (void)cudnn_context;
        (void)cublas_context;
        (void)gpu_allocator;
    }

    void detachFromContext() noexcept override {
    }

private:
    int32_t npoint_{0};
    std::string namespace_;
};

class FarthestPointSamplingPluginCreator final : public nvinfer1::IPluginCreator {
public:
    FarthestPointSamplingPluginCreator() {
        fields_.emplace_back("npoint", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
        fields_.emplace_back("npoint_i", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
        field_collection_.nbFields = static_cast<int32_t>(fields_.size());
        field_collection_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override {
        return kPluginName;
    }

    char const* getPluginVersion() const noexcept override {
        return kPluginVersion;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        return &field_collection_;
    }

    nvinfer1::IPluginV2* createPlugin(
        char const* name,
        nvinfer1::PluginFieldCollection const* field_collection) noexcept override {
        (void)name;
        int32_t npoint = 0;
        if (field_collection != nullptr) {
            for (int32_t i = 0; i < field_collection->nbFields; ++i) {
                auto const& field = field_collection->fields[i];
                if (field.data == nullptr || field.length <= 0) {
                    continue;
                }
                if (std::strcmp(field.name, "npoint") == 0 || std::strcmp(field.name, "npoint_i") == 0) {
                    if (field.type == nvinfer1::PluginFieldType::kINT32) {
                        npoint = *static_cast<int32_t const*>(field.data);
                    }
                }
            }
        }

        if (npoint <= 0) {
            return nullptr;
        }
        auto* plugin = new FarthestPointSamplingPlugin(npoint);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name,
        void const* serial_data,
        size_t serial_length) noexcept override {
        (void)name;
        auto* plugin = new FarthestPointSamplingPlugin(serial_data, serial_length);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    void setPluginNamespace(char const* plugin_namespace) noexcept override {
        namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
    }

    char const* getPluginNamespace() const noexcept override {
        return namespace_.c_str();
    }

private:
    std::string namespace_;
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection field_collection_{};
};

} // namespace

REGISTER_TENSORRT_PLUGIN(FarthestPointSamplingPluginCreator);
