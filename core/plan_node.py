import os
import time
import numpy as np
import pandas as pd
import math
import onnxruntime
import torch
# import utils
from utils.feature_engineering import prepare_inference_data
# from structure import no_dop_operator_features, no_dop_operators_exec, no_dop_operators_mem, dop_operators_exec, dop_operators_mem, parallel_op
# --- 修改导入：确保导入了 no_dop_operators_mem ---
from config.structure import parallel_op, dop_sets, dop_operators_exec, dop_operators_mem, no_dop_operators_exec, no_dop_operators_mem
# --- 结束修改 ---
from core.onnx_manager import ONNXModelManager

class PlanNode:
    def __init__(self, plan_data, onnx_manager, use_estimates=False):
        self.use_estimates = use_estimates  # 模式开关
        self.visit = False
        self.plan_id = plan_data['plan_id']
        self.query_id = plan_data['query_id']
        self.dop = plan_data['dop']
        self.operator_type = plan_data['operator_type']
        
        # --- 存储真实值 ---
        self.actual_rows = plan_data['actual_rows']
        self.l_input_rows_actual = plan_data.get('l_input_rows') # 使用 .get 避免 Key Error
        self.r_input_rows_actual = plan_data.get('r_input_rows')
        
        # --- 存储估计值 ---
        self.estimate_rows = plan_data['estimate_rows']
        # l_input_rows 和 r_input_rows 的估计值需要后续计算，先初始化
        self.l_input_rows_estimate = None
        self.r_input_rows_estimate = None

        self.updop =  plan_data['up_dop']
        self.downdop =  plan_data['down_dop']
        self.send_time = plan_data['stream_data_send_time'] - plan_data['stream_quota_time']
        self.execution_time = plan_data['execution_time'] + self.send_time
        self.estimate_costs = plan_data['estimate_costs']
        self.build_time = plan_data['build_time']
        self.pred_execution_time = 0
        self.pred_mem = 0
        self.best_dop = 0
        self.thread_id = 0
        self.peak_mem = plan_data['peak_mem']
        self.width = plan_data['width']
        self.exec_feature_data = None 
        self.mem_feature_data = None
        self.child_plans = []  # 用于存储子计划节点
        self.materialized = 'hash' in self.operator_type.lower() or 'aggregate' in self.operator_type.lower() or 'sort' in self.operator_type.lower() or 'materialize' in self.operator_type.lower()  # 判断是否是物化算子
        self.parent_node = None  # 父节点
        self.is_parallel = (self.operator_type in parallel_op)
        self.thread_execution_time = 0
        self.thread_complete_time = 0
        self.local_data_transfer_start_time = 0
        self.up_data_transfer_start_time = 0     
        self.pred_exec_time=0
        self.pred_mem_time=0 
        self.pred_params = None
        
        self.onnx_manager = onnx_manager  # 传入 ONNXModelManager 实例
        
        # **新增存储不同 dop 的执行信息**
        self.exec_time_map = {self.dop: self.execution_time}  # 直接初始化
        self.matched_dops = {self.dop}  # 记录匹配的 dop
        # **存储不同 dop 的预测执行时间**
        self.pred_dop_exec_map = {}
        self.send_dop_exec_map = {}
        self.build_dop_exec_map = {}
        self.true_dop_exec_map = {self.dop: self.execution_time}

        self.get_feature_data(plan_data)
        self.infer_exec_with_onnx()
        self.infer_mem_with_onnx()

    def add_child(self, child_node):
        self.child_plans.append(child_node)
        child_node.parent_node = self
        child_node.visit = False  
        
    def update_estimated_inputs(self):
        """根据子节点的估计输出，更新本节点的估计输入。"""
        if not self.use_estimates or not self.child_plans:
            return

        if len(self.child_plans) > 0:
            # 左子节点的估计输出，成为本节点的左输入估计
            self.l_input_rows_estimate = self.child_plans[0].estimate_rows
        
        if len(self.child_plans) > 1:
            # 右子节点的估计输出，成为本节点的右输入估计
            self.r_input_rows_estimate = self.child_plans[1].estimate_rows

    def get_feature_data(self, plan_data):
        """
        根据 use_estimates 模式，准备用于特征工程的字典。
        """
        data_for_features = plan_data.copy()

        if self.use_estimates:
            # 替换 'actual_rows'
            data_for_features['actual_rows'] = self.estimate_rows
            
            # 替换 'l_input_rows' 和 'r_input_rows'
            # 如果此时 self.l_input_rows_estimate 还没被更新，则用原始值或0
            data_for_features['l_input_rows'] = self.l_input_rows_estimate if self.l_input_rows_estimate is not None else data_for_features.get('l_input_rows', 0)
            data_for_features['r_input_rows'] = self.r_input_rows_estimate if self.r_input_rows_estimate is not None else data_for_features.get('r_input_rows', 0)
            
        self.exec_feature_data, self.mem_feature_data = prepare_inference_data(
            data_for_features, 
            data_for_features['operator_type']
        )
            
    def update_real_exec_time(self, dop, execution_time):
        """ 记录不同 dop 对应的执行时间和内存 """
        self.exec_time_map[dop] = execution_time
        self.true_dop_exec_map[dop] = execution_time
        self.matched_dops.add(dop)
        
    def update_send_time(self, dop, send_time, build_time):
        """ 记录不同 dop 对应的执行时间和内存 """
        self.send_dop_exec_map[dop] = send_time
        self.build_dop_exec_map[dop] = build_time
        
    def infer_mem_with_onnx(self):
        """
        使用 ONNX 模型推理内存 (如果算子在列表中，否则使用默认值)。
        (移除额外健壮性检查，保留核心逻辑和 should_predict_mem 判断)
        """
        start_time = time.time()
        # 基础默认值 (如果不支持预测或特征缺失时使用)
        # 原始代码在特征缺失时用 1024 * self.dop
        # 这里我们需要统一一个基础值，让 PlanNode 外部或内部决定是否乘 dop
        default_mem_base = 1024.0

        # --- 检查是否应该预测内存 ---
        should_predict_mem = False
        if self.operator_type:
            should_predict_mem = (self.operator_type in dop_operators_mem or
                                  self.operator_type in no_dop_operators_mem)

        if not should_predict_mem:
            # 对于不支持预测的算子，使用默认值
            # 原始代码在特征缺失时会乘以 dop，这里保持一致？
            current_dop = self.dop if self.dop is not None and self.dop > 0 else 1
            self.pred_mem = default_mem_base * current_dop
            self.pred_mem_time = time.time() - start_time # 记录基础时间
            return # 直接返回
        # --- 结束检查 ---

        # --- 只有应该预测时才执行以下代码 ---
        if self.mem_feature_data is None:
            # 特征缺失时，使用原始逻辑的默认值
            current_dop = self.dop if self.dop is not None and self.dop > 0 else 1
            self.pred_mem = default_mem_base * current_dop
            self.pred_mem_time = time.time() - start_time # 记录基础时间
            return

        # --- 调用 ONNX Manager (假设它会处理 feature_data 类型并可能抛出异常) ---
        feature_values = self.mem_feature_data # 直接传递

        if self.operator_type in dop_operators_mem:
            # 调用 infer_mem 获取参数 (如果模型不存在，onnx_manager 会 raise ValueError)
            pred_params = self.onnx_manager.infer_mem(self.operator_type, feature_values)
            # 原始代码直接使用参数，不做 None 检查
            current_dop = self.dop if self.dop is not None and self.dop > 0 else 1
            # 原始计算公式
            pred_mem = max(pred_params[1] * (current_dop ** pred_params[0]) + pred_params[2], pred_params[3])
            self.pred_mem = max(pred_mem, 1e-1) # 原始下限

        elif self.operator_type in no_dop_operators_mem: # 确保使用 elif
             # 调用 infer_mem 获取预测值 (如果模型不存在，onnx_manager 会 raise ValueError)
             pred_mem_val = self.onnx_manager.infer_mem(self.operator_type, feature_values)
             self.pred_mem = max(pred_mem_val, 1e-1) # 原始下限

        else:
             # 理论上因为 should_predict_mem 判断，不应到达此处
             # 如果到达，说明配置或逻辑有误，按原始行为可能没有处理，这里赋基础默认值
             current_dop = self.dop if self.dop is not None and self.dop > 0 else 1
             self.pred_mem = default_mem_base * current_dop


        end_time = time.time()
        # 原始代码是在函数末尾累加，保持不变
        self.pred_mem_time += end_time - start_time # 累加耗时

    # --- 对 infer_exec_with_onnx 也做类似修改 ---
    def infer_exec_with_onnx(self):
        """
        使用 ONNX 模型推理执行时间 (如果算子在列表中，否则使用默认值)。
        (移除额外健壮性检查，保留核心逻辑和 should_predict_exec 判断)
        """
        start_time = time.time()
        default_exec = 0.05  + self.send_time # 基础默认值

        # --- 检查是否应该预测执行时间 ---
        should_predict_exec = False
        if self.operator_type:
             should_predict_exec = (self.operator_type in dop_operators_exec or
                                    self.operator_type in no_dop_operators_exec)

        if not should_predict_exec:
            self.pred_execution_time = default_exec
            self.pred_exec_time = time.time() - start_time
            return # 直接返回
        # --- 结束检查 ---

        # --- 只有应该预测时才执行以下代码 ---
        if self.exec_feature_data is None:
            self.pred_execution_time = default_exec
            self.pred_exec_time = time.time() - start_time
            return

        feature_values = self.exec_feature_data

        if self.operator_type in dop_operators_exec:
            # 调用 infer_exec 获取参数 (如果模型不存在，onnx_manager 会 raise ValueError)
            self.pred_params = self.onnx_manager.infer_exec(self.operator_type, feature_values)
            # 原始代码直接使用参数
            current_dop = self.dop if self.dop is not None and self.dop > 0 else 1
            # 原始计算公式
            pred_exec = self.pred_params[1] / (current_dop ** self.pred_params[0]) + self.pred_params[2] * (current_dop**self.pred_params[3]) + self.pred_params[4]
            self.pred_execution_time = max(pred_exec, 1e-1) # 原始下限

        elif self.operator_type in no_dop_operators_exec:
             # 调用 infer_exec 获取预测值 (如果模型不存在，onnx_manager 会 raise ValueError)
             pred_exec = self.onnx_manager.infer_exec(self.operator_type, feature_values)
             self.pred_execution_time = max(pred_exec, 1e-1) # 原始下限

        else:
             # 理论上不应到达
             self.pred_execution_time = default_exec
        self.pred_execution_time = float(self.pred_execution_time)
        if self.operator_type == 'CStore Scan' and self.exec_feature_data[-1] > 2000:
            self.pred_execution_time = self.pred_execution_time * math.log(self.exec_feature_data[-1])
        end_time = time.time()
        self.pred_exec_time += end_time - start_time
        
    def compute_pred_exec(self, dop):
        if self.pred_params is None:
            return 0
        pred_exec = (self.pred_params[1] / (dop ** self.pred_params[0])
                     + self.pred_params[2] * (dop ** self.pred_params[3])
                     + self.pred_params[4])
        return max(pred_exec, 1e-1)
    
    def interpolate_true_dop(self, target_dop):
        keys = sorted(self.true_dop_exec_map.keys())
        
        # 当真实执行时间数据不足时，直接使用预测执行时间作为替代
        if len(keys) < 2:
            return self.compute_pred_exec(target_dop)

        # 处理外插（左边界）
        if target_dop < keys[0]:
            left, right = keys[0], keys[1]
            left_val, right_val = self.true_dop_exec_map[left], self.true_dop_exec_map[right]
            weight = (target_dop - left) / (right - left)
            return left_val + weight * (right_val - left_val)

        # 处理外插（右边界）
        if target_dop > keys[-1]:
            left, right = keys[-2], keys[-1]
            left_val, right_val = self.true_dop_exec_map[left], self.true_dop_exec_map[right]
            weight = (target_dop - left) / (right - left)
            return left_val + weight * (right_val - left_val)

        # 在相邻两点间线性插值
        for i in range(1, len(keys)):
            if keys[i] >= target_dop:
                left, right = keys[i - 1], keys[i]
                left_val, right_val = self.true_dop_exec_map[left], self.true_dop_exec_map[right]
                weight = (target_dop - left) / (right - left)
                return left_val + weight * (right_val - left_val)

        return 0  # 这个分支理论上不会走到

    def compute_parallel_dop_predictions(self):
        """ 计算所有 dop 的预测执行时间(累加子节点的执行时间) """
        # 先递归计算所有子节点的预测执行时间
        for child in self.child_plans:
            child.compute_parallel_dop_predictions()

        # 如果当前节点支持并行且已经得到预测参数,则计算当前节点自身的预测执行时间
        if self.is_parallel and self.pred_params is not None:
            for dop in dop_sets:
                pred_exec = (self.pred_params[1] / (dop ** self.pred_params[0])
                            + self.pred_params[2] * (dop ** self.pred_params[3])
                            + self.pred_params[4])
                # 累加当前节点的预测执行时间到已有的子节点累加结果上
                self.pred_dop_exec_map[dop] = max(pred_exec, 1e-1)
                if dop not in self.matched_dops:
                    self.true_dop_exec_map[dop] = self.interpolate_true_dop(dop)
                    self.exec_time_map[dop] =  self.true_dop_exec_map[dop]        
                    self.matched_dops.add(dop)    
        else:
            for dop in dop_sets:
                self.pred_dop_exec_map[dop] = self.pred_execution_time
                self.true_dop_exec_map[dop] = self.execution_time
                self.exec_time_map[dop] =  self.execution_time   
                self.matched_dops.add(dop)
            # 对于真实执行时间，若某个候选 dop 缺失，则进行补全
            # 定义一个辅助函数，根据 pred_params 计算预测执行时间
        # 累加子节点的真实执行时间到当前节点的 dop_exec_time_map
        for child in self.child_plans:
            if getattr(child, "thread_id", self.thread_id) == self.thread_id:
                for dop, child_time in child.true_dop_exec_map.items():
                    self.true_dop_exec_map[dop] = self.true_dop_exec_map[dop] + child_time

        # 累加子节点的预测执行时间到当前节点的 pred_dop_exec_map
        for child in self.child_plans:
            if getattr(child, "thread_id", self.thread_id) == self.thread_id:
                for dop, child_pred in child.pred_dop_exec_map.items():
                    self.pred_dop_exec_map[dop] = self.pred_dop_exec_map[dop] + child_pred