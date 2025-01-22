#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "utils.hpp"


// 针对整个查询运行推理
void RunInferenceForQuery(const std::string& query_id, const std::string& model_path_prefix) {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Inference");

        // 加载查询涉及的算子及其输入数据
        auto query_operators = LoadQueryOperators();

        // 遍历每个算子
        for (const auto& [operator_type, input_data] : query_operators) {
            InferForOperator(operator_type, input_data, model_path_prefix, env);
        }
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
    }
}


// 运行推理并获取输出
std::vector<float> RunInference(Ort::Session& session, Ort::Value& input_tensor, const char* input_name, const char* output_name) {
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        &input_name, &input_tensor, 1,  // 输入名和张量
        &output_name, 1  // 输出名和数量
    );

    // 获取输出数据
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    return std::vector<float>(output_data, output_data + output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount());
}

// 对单个算子进行推理
void InferForOperator(const std::string& operator_type, 
                      const std::vector<std::vector<float>>& input_data, 
                      const std::string& model_path_prefix, 
                      const Ort::Env& env) {
    std::cout << "\n===== Operator: " << operator_type << " =====" << std::endl;

    // 构造 ONNX 模型路径
    std::string exec_model_path = model_path_prefix + "xgboost_exec_" + operator_type + ".onnx";
    std::string mem_model_path = model_path_prefix + "xgboost_mem_" + operator_type + ".onnx";

    // 加载 ONNX 模型
    Ort::Session exec_session = LoadONNXModel(env, exec_model_path);
    Ort::Session mem_session = LoadONNXModel(env, mem_model_path);

    // 准备输入张量
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Value input_tensor = PrepareInputTensor(input_data, allocator);

    // 获取输入和输出名称
    Ort::AllocatedStringPtr exec_input_name_ptr = exec_session.GetInputNameAllocated(0, allocator);
    const char* exec_input_name = exec_input_name_ptr.get();
    Ort::AllocatedStringPtr exec_output_name_ptr = exec_session.GetOutputNameAllocated(0, allocator);
    const char* exec_output_name = exec_output_name_ptr.get();

    Ort::AllocatedStringPtr mem_input_name_ptr = mem_session.GetInputNameAllocated(0, allocator);
    const char* mem_input_name = mem_input_name_ptr.get();
    Ort::AllocatedStringPtr mem_output_name_ptr = mem_session.GetOutputNameAllocated(0, allocator);
    const char* mem_output_name = mem_output_name_ptr.get();

    // 推理开始时间
    auto start_time = std::chrono::high_resolution_clock::now();

    // 执行推理
    auto exec_output = RunInference(exec_session, input_tensor, exec_input_name, exec_output_name);
    auto mem_output = RunInference(mem_session, input_tensor, mem_input_name, mem_output_name);

    // 推理结束时间
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inference_duration = end_time - start_time;

    // 输出推理时间
    std::cout << "Inference time: " << inference_duration.count() << " seconds" << std::endl;

    // 输出结果
    for (size_t i = 0; i < exec_output.size(); ++i) {
        std::cout << "Sample " << i + 1 << ":\n";
        std::cout << "Predicted Execution Time: " << exec_output[i] << std::endl;
        std::cout << "Predicted Peak Mem: " << mem_output[i] << std::endl;
    }
}

int main() {
    // 假设我们处理的是查询 ID "Q1"
    const std::string model_path_prefix = "/home/zhy/opengauss/tools/serverless_tools/train/";
    RunInferenceForQuery("Q1", model_path_prefix);
    return 0;
}
