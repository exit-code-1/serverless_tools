cmake_minimum_required(VERSION 3.12)
project(onnx_inference)

set(CMAKE_CXX_STANDARD 17)
set(GCC_PATH "/home/zhy/opengauss/openGauss-Scale/openGauss-third_party_binarylibs_Centos7.6_x86_64/buildtools/gcc10.3/gcc")  # 更新为您实际的路径
set(CMAKE_C_COMPILER "${GCC_PATH}/bin/gcc")
set(CMAKE_CXX_COMPILER "${GCC_PATH}/bin/g++")
set(LD_LIBRARY_PATH "$GAUSSHOME/lib:$GCC_PATH/gcc/lib64:$GCC_PATH/isl/lib:$GCC_PATH/mpc/lib/:$GCC_PATH/mpfr/lib/:$GCC_PATH/gmp/lib/:$LD_LIBRARY_PATH")
# 设置 ONNX Runtime 路径
set(ONNX_RUNTIME_DIR /home/zhy/opengauss/openGauss-Scale/openGauss-third_party_binarylibs_Centos7.6_x86_64/kernel/dependency/onnxruntime)

# 包含头文件和库路径
include_directories(${ONNX_RUNTIME_DIR}/include)
link_directories(${ONNX_RUNTIME_DIR}/lib)

# 可执行文件
add_executable(onnx_inference onnxdemo.cpp)

# 链接 ONNX Runtime 库
target_link_libraries(onnx_inference onnxruntime)
