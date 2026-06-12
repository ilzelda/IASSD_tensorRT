#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <cstring>
#include <string>
#include <vector>

#include "ball_query_gpu_raw.h"

namespace {

constexpr char const* kPluginName = "BallQuery";
constexpr char const* kPluginVersion = "1";

template <typename T>
void WriteValue(char*& buffer, T value) {
    std::memcpy(buffer, &value, sizeof(T));
    buffer += sizeof(T);
}

template <typename T>
T ReadValue(char const*& buffer) {
    T value{};
    std::memcpy(&value, buffer, sizeof(T));
    buffer += sizeof(T);
    return value;
}

class BallQueryPlugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    BallQueryPlugin(float radius, int32_t nsample)
        : radius_(radius), nsample_(nsample) {}

    BallQueryPlugin(void const* serial_data, size_t serial_length) {
        (void)serial_length;
        auto const* buffer = static_cast<char const*>(serial_data);
        radius_ = ReadValue<float>(buffer);
        nsample_ = ReadValue<int32_t>(buffer);
    }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        auto* plugin = new BallQueryPlugin(radius_, nsample_);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t output_index,
        nvinfer1::DimsExprs const* inputs,
        int32_t nb_inputs,
        nvinfer1::IExprBuilder& expr_builder) noexcept override {
        nvinfer1::DimsExprs output{};
        if (output_index != 0 || nb_inputs != 2) {
            return output;
        }
        output.nbDims = 3;
        output.d[0] = inputs[0].d[0];
        output.d[1] = inputs[1].d[1];
        output.d[2] = expr_builder.constant(nsample_);
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
        if (pos == 0 || pos == 1) {
            return in_out[pos].type == nvinfer1::DataType::kFLOAT;
        }
        if (pos == 2) {
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
        const int num_points = input_desc[0].dims.d[1];
        const int npoint = input_desc[1].dims.d[1];
        const auto* xyz = static_cast<float const*>(inputs[0]);
        const auto* new_xyz = static_cast<float const*>(inputs[1]);
        auto* output = static_cast<int*>(outputs[0]);
        const size_t output_bytes =
            static_cast<size_t>(batch_size) * static_cast<size_t>(npoint) * static_cast<size_t>(nsample_) * sizeof(int);
        cudaMemsetAsync(output, 0x00, output_bytes, stream);
        ball_query_kernel_launcher_fast(
            batch_size,
            num_points,
            npoint,
            radius_,
            nsample_,
            new_xyz,
            xyz,
            output,
            stream);
        return 0;
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
        return 0;
    }

    void terminate() noexcept override {}

    size_t getSerializationSize() const noexcept override {
        return sizeof(float) + sizeof(int32_t);
    }

    void serialize(void* buffer) const noexcept override {
        auto* data = static_cast<char*>(buffer);
        WriteValue(data, radius_);
        WriteValue(data, nsample_);
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
    float radius_{0.0F};
    int32_t nsample_{0};
    std::string namespace_;
};

class BallQueryPluginCreator final : public nvinfer1::IPluginCreator {
public:
    BallQueryPluginCreator() {
        plugin_attributes_.emplace_back(nvinfer1::PluginField("radius", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
        plugin_attributes_.emplace_back(nvinfer1::PluginField("radius_f", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
        plugin_attributes_.emplace_back(nvinfer1::PluginField("nsample", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        plugin_attributes_.emplace_back(nvinfer1::PluginField("nsample_i", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
        field_collection_.nbFields = static_cast<int32_t>(plugin_attributes_.size());
        field_collection_.fields = plugin_attributes_.data();
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
        float radius = 0.0F;
        int32_t nsample = 0;
        for (int32_t index = 0; index < field_collection->nbFields; ++index) {
            nvinfer1::PluginField const& field = field_collection->fields[index];
            if (std::strcmp(field.name, "radius") == 0 || std::strcmp(field.name, "radius_f") == 0) {
                radius = *static_cast<float const*>(field.data);
            } else if (std::strcmp(field.name, "nsample") == 0 || std::strcmp(field.name, "nsample_i") == 0) {
                nsample = *static_cast<int32_t const*>(field.data);
            }
        }
        auto* plugin = new BallQueryPlugin(radius, nsample);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    nvinfer1::IPluginV2* deserializePlugin(
        char const* name,
        void const* serial_data,
        size_t serial_length) noexcept override {
        (void)name;
        auto* plugin = new BallQueryPlugin(serial_data, serial_length);
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
    std::vector<nvinfer1::PluginField> plugin_attributes_;
    nvinfer1::PluginFieldCollection field_collection_{};
};

}  // namespace

REGISTER_TENSORRT_PLUGIN(BallQueryPluginCreator);
