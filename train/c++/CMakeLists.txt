cmake_minimum_required(VERSION 3.12)
project(onnx_inference)

set(CMAKE_CXX_STANDARD 17)

# 设置 ONNX Runtime 路径
set(ONNX_RUNTIME_DIR /home/zhy/opengauss/openGauss-Scale/openGauss-third_party_binarylibs_Centos7.6_x86_64/kernel/dependency/onnxruntime)

# 包含头文件和库路径
include_directories(${ONNX_RUNTIME_DIR}/include)
link_directories(${ONNX_RUNTIME_DIR}/lib)

# 可执行文件
add_executable(onnx_inference onnxdemo.cpp)

# 链接 ONNX Runtime 库
target_link_libraries(onnx_inference onnxruntime)
