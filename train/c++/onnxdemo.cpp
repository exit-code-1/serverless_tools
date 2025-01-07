#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

// Helper function to read binary ONNX model
std::vector<char> ReadModel(const std::string& model_path) {
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

// Function to load data (removing first column)
std::vector<std::vector<float>> LoadData() {
    return {
    {-0.571373f, -0.580132f, 1.732039f, -0.580534f},
    {0.415578f, -0.761755f, 1.435449f, -1.089271f},
    {0.415578f, -0.761755f, 1.435449f, -1.089271f},
    {-0.571225f, -0.580414f, 1.732039f, -0.580400f}
    };
}

int main() {
    try {
        // Initialize ONNX Runtime environment
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Inference");

        // Paths to the ONNX models
        const std::string exec_model_path = "/home/zhy/opengauss/tools/serverless_tools/train/xgboost_exec_CStore_Index_Scan.onnx";
        const std::string mem_model_path = "/home/zhy/opengauss/tools/serverless_tools/train/xgboost_mem_CStore_Index_Scan.onnx";

        // Create session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);  // Set number of threads
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        // Create inference sessions for both models
        Ort::Session exec_session(env, exec_model_path.c_str(), session_options);
        Ort::Session mem_session(env, mem_model_path.c_str(), session_options);

        // Load input data
        auto input_data = LoadData();

        // Input tensor shape: 4 samples, 6 features
        std::vector<int64_t> input_shape = {static_cast<int64_t>(input_data.size()), 4};

        // Flatten the input data into a contiguous vector
        std::vector<float> flattened_input_data;
        for (const auto& row : input_data) {
            flattened_input_data.insert(flattened_input_data.end(), row.begin(), row.end());
        }

        // Allocate memory for input tensor
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Create input tensor with flattened data
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, flattened_input_data.data(), flattened_input_data.size(), input_shape.data(), input_shape.size());

        // Get input and output names for execution time model
        Ort::AllocatedStringPtr exec_input_name_ptr = exec_session.GetInputNameAllocated(0, allocator);
        const char* exec_input_name = exec_input_name_ptr.get();
        Ort::AllocatedStringPtr exec_output_name_ptr = exec_session.GetOutputNameAllocated(0, allocator);
        const char* exec_output_name = exec_output_name_ptr.get();

        // Get input and output names for peak memory model
        Ort::AllocatedStringPtr mem_input_name_ptr = mem_session.GetInputNameAllocated(0, allocator);
        const char* mem_input_name = mem_input_name_ptr.get();
        Ort::AllocatedStringPtr mem_output_name_ptr = mem_session.GetOutputNameAllocated(0, allocator);
        const char* mem_output_name = mem_output_name_ptr.get();
// 获取当前时间
auto start_time = std::chrono::high_resolution_clock::now();
        // Run inference for execution time model
        auto exec_output_tensors = exec_session.Run(
            Ort::RunOptions{nullptr},
            &exec_input_name, &input_tensor, 1,  // Input name and tensor
            &exec_output_name, 1  // Output name and count
        );

        // Run inference for peak memory model
        auto mem_output_tensors = mem_session.Run(
            Ort::RunOptions{nullptr},
            &mem_input_name, &input_tensor, 1,  // Input name and tensor
            &mem_output_name, 1  // Output name and count
        );
// 获取推理结束时间
auto end_time = std::chrono::high_resolution_clock::now();
// 计算推理时间
std::chrono::duration<double> inference_duration = end_time - start_time;
std::cout << "Inference time: " << inference_duration.count() << " seconds" << std::endl;

        // Get output data for execution time model
        float* exec_output_data = exec_output_tensors[0].GetTensorMutableData<float>();

        // Get output data for peak memory model
        float* mem_output_data = mem_output_tensors[0].GetTensorMutableData<float>();

// Output the predicted results for each sample
for (size_t i = 0; i < input_data.size(); ++i) {
    std::cout << "Sample " << i + 1 << ":\n";
    std::cout << "Predicted Execution Time: " << exec_output_data[i] << std::endl;
    std::cout << "Predicted Peak Mem: " << mem_output_data[i] << std::endl;
}

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}