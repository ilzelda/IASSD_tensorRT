#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <string>
#include <vector>

#include "sampling_gpu_raw.h"

namespace {

constexpr char const* kPluginName = "GatherPoints";
constexpr char const* kPluginVersion = "1";

class GatherPointsPlugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    GatherPointsPlugin() = default;

    GatherPointsPlugin(void const* serial_data, size_t serial_length) {
        (void)serial_data;
        (void)serial_length;
    }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        auto* plugin = new GatherPointsPlugin();
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t output_index,
        nvinfer1::DimsExprs const* inputs,
        int32_t nb_inputs,
        nvinfer1::IExprBuilder& expr_builder) noexcept override {
        (void)expr_builder;
        nvinfer1::DimsExprs output{};
        if (output_index != 0 || nb_inputs != 2) {
            return output;
        }
        output.nbDims = 3;
        output.d[0] = inputs[0].d[0];
        output.d[1] = inputs[0].d[1];
        output.d[2] = inputs[1].d[1];
        return output;
    }

    bool supportsFormatCombination(
        int32_t pos,
        nvinfer1::PluginTensorDesc const* in_out,
        int32_t nb_inputs,
        int32_t nb_outputs) noexcept override {
        if (nb_inputs != 2 || nb_outputs != 1) {
            return false;
        }
        if (in_out[pos].format != nvinfer1::TensorFormat::kLINEAR) {
            return false;
        }
        if (pos == 0) {
            return in_out[pos].type == nvinfer1::DataType::kFLOAT;
        }
        if (pos == 1) {
            return in_out[pos].type == nvinfer1::DataType::kINT32;
        }
        if (pos == 2) {
            return in_out[pos].type == nvinfer1::DataType::kFLOAT;
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
        (void)inputs;
        (void)nb_inputs;
        (void)outputs;
        (void)nb_outputs;
        return 0;
    }

    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* input_desc,
        nvinfer1::PluginTensorDesc const* output_desc,
        void const* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream) noexcept override {
        (void)output_desc;
        (void)workspace;
        const int batch_size = input_desc[0].dims.d[0];
        const int channel = input_desc[0].dims.d[1];
        const int num_points = input_desc[0].dims.d[2];
        const int npoint = input_desc[1].dims.d[1];
        const auto* features = static_cast<float const*>(inputs[0]);
        const auto* idx = static_cast<int const*>(inputs[1]);
        auto* output = static_cast<float*>(outputs[0]);
        gather_points_kernel_launcher_fast(
            batch_size,
            channel,
            num_points,
            npoint,
            features,
            idx,
            output,
            stream);
        return 0;
    }

    nvinfer1::DataType getOutputDataType(
        int32_t index,
        nvinfer1::DataType const* input_types,
        int32_t nb_inputs) const noexcept override {
        (void)index;
        (void)nb_inputs;
        return input_types[0];
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
        return 0;
    }

    void terminate() noexcept override {}

    size_t getSerializationSize() const noexcept override {
        return 0;
    }

    void serialize(void* buffer) const noexcept override {
        (void)buffer;
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

private:
    std::string namespace_;
};

class GatherPointsPluginCreator final : public nvinfer1::IPluginCreator {
public:
    GatherPointsPluginCreator() {
        field_collection_.nbFields = 0;
        field_collection_.fields = nullptr;
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
        (void)field_collection;
        auto* plugin = new GatherPointsPlugin();
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name,
        void const* serial_data,
        size_t serial_length) noexcept override {
        (void)name;
        auto* plugin = new GatherPointsPlugin(serial_data, serial_length);
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
    nvinfer1::PluginFieldCollection field_collection_{};
};

}  // namespace

REGISTER_TENSORRT_PLUGIN(GatherPointsPluginCreator);
