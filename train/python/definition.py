import os
import time
import numpy as np
import math
import onnxruntime
import torch
import utils
from structure import no_dop_operator_features, no_dop_operators_exec, no_dop_operators_mem, dop_operators_exec, dop_operators_mem

class ONNXModelManager:
    def __init__(self):
        self.no_dop_model_dir = "/home/zhy/opengauss/tools/serverless_tools/train/model/no_dop"
        self.dop_model_dir = "/home/zhy/opengauss/tools/serverless_tools/train/model/dop"
        self.exec_sessions = {}  # 存储执行时间模型的会话
        self.mem_sessions = {}   # 存储内存模型的会话
        self.load_models()

    def load_models(self):
        dop_operators = set()
        # 遍历模型目录，加载所有 ONNX 模型
        for operator_type in os.listdir(self.dop_model_dir):
            operator_path = os.path.join(self.dop_model_dir, operator_type)
            if os.path.isdir(operator_path):
                dop_operators.add(operator_type)
                # 将 operator_type 中的空格替换为下划线
                operator_name = operator_type.replace(' ', '_')
                
                # 加载执行时间模型
                exec_model_path = os.path.join(operator_path, f"exec_{operator_name}.onnx")
                if os.path.exists(exec_model_path):
                    self.exec_sessions[operator_type] = onnxruntime.InferenceSession(exec_model_path)
                
                # 加载内存模型
                mem_model_path = os.path.join(operator_path, f"mem_{operator_name}.onnx")
                if os.path.exists(mem_model_path):
                    self.mem_sessions[operator_type] = onnxruntime.InferenceSession(mem_model_path)
        for operator_type in os.listdir(self.no_dop_model_dir):
            if operator_type in dop_operators:
                continue
            operator_path = os.path.join(self.no_dop_model_dir, operator_type)
            if os.path.isdir(operator_path):
                # 将 operator_type 中的空格替换为下划线
                operator_name = operator_type.replace(' ', '_')
                
                # 加载执行时间模型
                exec_model_path = os.path.join(operator_path, f"exec_{operator_name}.onnx")
                if os.path.exists(exec_model_path):
                    self.exec_sessions[operator_type] = onnxruntime.InferenceSession(exec_model_path)
                
                # 加载内存模型
                mem_model_path = os.path.join(operator_path, f"mem_{operator_name}.onnx")
                if os.path.exists(mem_model_path):
                    self.mem_sessions[operator_type] = onnxruntime.InferenceSession(mem_model_path)

    def infer_exec(self, operator_type, feature_data):
        # 将 operator_type 中的空格替换为下划线以匹配模型文件名
        
        if operator_type not in self.exec_sessions:
            raise ValueError(f"No execution model found for operator type: {operator_type}")
        
        session = self.exec_sessions[operator_type]
        feature_array = np.array(feature_data).reshape(1, -1).astype(np.float32)
        inputs = {session.get_inputs()[0].name: feature_array}
        exec_pred = session.run(None, inputs)
        exec_pred[0][0]
        return exec_pred[0][0]

    def infer_mem(self, operator_type, feature_data):
        # 将 operator_type 中的空格替换为下划线以匹配模型文件名
        if operator_type not in self.mem_sessions:
            raise ValueError(f"No memory model found for operator type: {operator_type}")
        
        session = self.mem_sessions[operator_type]
        feature_array = np.array(feature_data).reshape(1, -1).astype(np.float32)
        inputs = {session.get_inputs()[0].name: feature_array}
        mem_pred = session.run(None, inputs)
        return mem_pred[0][0]

# 在 PlanNode 中使用 ONNXModelManager
class PlanNode:
    def __init__(self, plan_data, onnx_manager):
        self.visit = False
        self.plan_id = plan_data['plan_id']
        self.query_id = plan_data['query_id']
        self.dop = plan_data['dop']
        self.operator_type = plan_data['operator_type']
        self.updop =  plan_data['up_dop']
        self.downdop =  plan_data['down_dop']
        self.execution_time = plan_data['execution_time']
        self.pred_execution_time = 0
        self.pred_mem = 0
        self.peak_mem = plan_data['peak_mem']
        self.width = plan_data['width']
        self.exec_feature_data = None 
        self.mem_feature_data = None
        self.child_plans = []  # 用于存储子计划节点
        self.materialized = 'hash' in self.operator_type.lower() or 'aggregate' in self.operator_type.lower() or 'sort' in self.operator_type.lower() or 'materialize' in self.operator_type.lower()  # 判断是否是物化算子
        self.parent_node = None  # 父节点
        self.thread_execution_time = 0
        self.thread_complete_time = 0
        self.local_data_transfer_start_time = 0
        self.up_data_transfer_start_time = 0     
        self.pred_exec_time=0
        self.pred_mem_time=0 
        
        self.onnx_manager = onnx_manager  # 传入 ONNXModelManager 实例
        
        self.get_feature_data(plan_data)
        self.infer_exec_with_onnx()
        self.infer_mem_with_onnx()
    
    def add_child(self, child_node):
        self.child_plans.append(child_node)
        child_node.parent_node = self
        child_node.visit = False  # 重置子节点的访问标记
        
    def get_feature_data(self, plan_data):
        self.exec_feature_data, self.mem_feature_data = utils.prepare_inference_data(plan_data, plan_data['operator_type'])
        
    def infer_exec_with_onnx(self):
        """
        使用 ONNX 模型推理执行时间。
        """
        start_time = time.time()
        if self.exec_feature_data is None:
            self.pred_execution_time = 0.05
            return
        if self.operator_type in dop_operators_exec:
            pred_params = self.onnx_manager.infer_exec(self.operator_type, self.exec_feature_data)
            pred_exec = pred_params[1] / self.dop  + pred_params[2] * self.dop**pred_params[0] + pred_params[3]
            self.pred_execution_time = max(pred_exec, 1e-1)
        else:
            self.pred_execution_time = max(self.onnx_manager.infer_exec(self.operator_type, self.exec_feature_data), 1e-1)
        end_time = time.time()
        self.pred_exec_time += end_time - start_time

    def infer_mem_with_onnx(self):
        """
        使用 ONNX 模型推理内存。
        """
        start_time = time.time()
        if self.mem_feature_data is None:
            self.pred_mem = 500 * self.dop
            return
        if self.operator_type in dop_operators_mem:
            pred_params = self.onnx_manager.infer_mem(self.operator_type, self.mem_feature_data)
            pred_mem = pred_params[1] * (self.dop ** pred_params[0]) + pred_params[2]
            self.pred_mem = max(pred_mem, 1e-1)
        else:
            self.pred_mem = max(self.onnx_manager.infer_mem(self.operator_type, self.mem_feature_data), 1e-1)
        end_time = time.time()
        self.pred_mem_time += end_time - start_time