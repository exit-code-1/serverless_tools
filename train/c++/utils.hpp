#include <iostream>
#include <vector>
#include <numeric>
#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <unordered_map>
#include <cmath>

std::vector<std::vector<double>> instanceNormalize(
    const std::vector<std::vector<double>>& data, double epsilon = 1e-5) {

    size_t num_instances = data.size();
    size_t num_channels = data[0].size();

    std::vector<std::vector<double>> normalized_data = data;

    for (size_t i = 0; i < num_instances; ++i) {
        std::vector<double> instance = data[i];

        // Compute mean and variance for the current instance
        double mean = std::accumulate(instance.begin(), instance.end(), 0.0) / num_channels;
        double variance = 0.0;
        for (double val : instance) {
            variance += (val - mean) * (val - mean);
        }
        variance /= num_channels;

        // Compute instance normalization
        for (size_t j = 0; j < num_channels; ++j) {
            normalized_data[i][j] = (instance[j] - mean) / std::sqrt(variance + epsilon);
        }
    }

    return normalized_data;
}

// 加载查询涉及的算子及其输入数据
std::unordered_map<std::string, std::vector<std::vector<float>>> LoadQueryOperators() {
    // 返回示例数据，每个算子包含对应的输入特征
    return {
        {"CStore Index Scan", {{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}}},
        {"Seq Scan", {{9.0, 10.0, 11.0, 12.0}}}
    };
}

// 加载 ONNX 模型
Ort::Session LoadONNXModel(const Ort::Env& env, const std::string& model_path) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
    return Ort::Session(env, model_path.c_str(), session_options);
}

// 准备输入张量
Ort::Value PrepareInputTensor(const std::vector<std::vector<float>>& input_data, Ort::AllocatorWithDefaultOptions& allocator) {
    // 设置输入形状
    std::vector<int64_t> input_shape = {static_cast<int64_t>(input_data.size()), 4};

    // 将输入数据展平为一维向量
    std::vector<float> flattened_input_data;
    for (const auto& row : input_data) {
        flattened_input_data.insert(flattened_input_data.end(), row.begin(), row.end());
    }

    // 创建输入张量
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(
        memory_info, flattened_input_data.data(), flattened_input_data.size(),
        input_shape.data(), input_shape.size());
}

