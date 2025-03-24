import os
import time
import numpy as np
import math
import onnxruntime
import torch
import utils
from structure import no_dop_operator_features, no_dop_operators_exec, no_dop_operators_mem, dop_operators_exec, dop_operators_mem

default_dop = 8

class ONNXModelManager:
    def __init__(self):
        self.no_dop_model_dir = "/home/zhy/opengauss/tools/serverless_tools/train/model/no_dop"
        self.dop_model_dir = "/home/zhy/opengauss/tools/serverless_tools/train/model/dop"
        self.exec_sessions = {}  # 存储执行时间模型的会话
        self.mem_sessions = {}   # 存储内存模型的会话
        self.load_models()

    def load_models(self):
        dop_operators = set()
        # 遍历模型目录,加载所有 ONNX 模型
        for operator_type in os.listdir(self.dop_model_dir):
            operator_path = os.path.join(self.dop_model_dir, operator_type)
            if os.path.isdir(operator_path):
                dop_operators.add(operator_type)
                # 将 operator_type 中的空格替换为下划线
                operator_name = operator_type.replace(' ', '_')
                if operator_type in dop_operators_exec:
                    # 加载执行时间模型
                    exec_model_path = os.path.join(operator_path, f"exec_{operator_name}.onnx")
                    if os.path.exists(exec_model_path):
                        self.exec_sessions[operator_type] = onnxruntime.InferenceSession(exec_model_path)
                if operator_type in dop_operators_mem:
                    # 加载内存模型
                    mem_model_path = os.path.join(operator_path, f"mem_{operator_name}.onnx")
                    if os.path.exists(mem_model_path):
                        self.mem_sessions[operator_type] = onnxruntime.InferenceSession(mem_model_path)
        for operator_type in os.listdir(self.no_dop_model_dir):
            operator_path = os.path.join(self.no_dop_model_dir, operator_type)
            if os.path.isdir(operator_path):
                # 将 operator_type 中的空格替换为下划线
                operator_name = operator_type.replace(' ', '_')
                if operator_type in no_dop_operators_exec:
                    # 加载执行时间模型
                    exec_model_path = os.path.join(operator_path, f"exec_{operator_name}.onnx")
                    if os.path.exists(exec_model_path):
                        self.exec_sessions[operator_type] = onnxruntime.InferenceSession(exec_model_path)
                if operator_type in no_dop_operators_mem:
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
        self.best_dop = 0
        self.thread_id = 0
        self.peak_mem = plan_data['peak_mem']
        self.is_parallel = (self.operator_type in dop_operators_exec)
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
        self.pred_params = None
        
        self.onnx_manager = onnx_manager  # 传入 ONNXModelManager 实例
        
        # **新增存储不同 dop 的执行信息**
        self.dop_exec_time_map = {self.dop: self.execution_time}  # 直接初始化
        self.matched_dops = {self.dop}  # 记录匹配的 dop
        # **存储不同 dop 的预测执行时间**
        self.pred_dop_exec_map = {}

        self.get_feature_data(plan_data)
        self.infer_exec_with_onnx()
        self.infer_mem_with_onnx()

    def add_child(self, child_node):
        self.child_plans.append(child_node)
        child_node.parent_node = self
        child_node.visit = False  
        
    def get_feature_data(self, plan_data):
        self.exec_feature_data, self.mem_feature_data = utils.prepare_inference_data(plan_data, plan_data['operator_type'])
        
    def update_real_exec_time(self, dop, execution_time):
        """ 记录不同 dop 对应的执行时间和内存 """
        self.dop_exec_time_map[dop] = execution_time
        self.matched_dops.add(dop)
        
    def infer_exec_with_onnx(self):
        """
        使用 ONNX 模型推理执行时间。
        """
        start_time = time.time()
        if self.exec_feature_data is None:
            self.pred_execution_time = 0.05
            return
        if self.operator_type in dop_operators_exec:
            self.pred_params = self.onnx_manager.infer_exec(self.operator_type, self.exec_feature_data)
            pred_exec = self.pred_params[1] / (self.dop ** self.pred_params[0]) + self.pred_params[2] * (self.dop**self.pred_params[3]) + self.pred_params[4]
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
            pred_mem = max(pred_params[1] * (self.dop ** pred_params[0]) + pred_params[2], pred_params[3])
            self.pred_mem = max(pred_mem, 1e-1)
        else:
            self.pred_mem = max(self.onnx_manager.infer_mem(self.operator_type, self.mem_feature_data), 1e-1)
        end_time = time.time()
        self.pred_mem_time += end_time - start_time
    
    def compute_parallel_dop_predictions(self):
        """ 计算所有 dop 的预测执行时间(累加子节点的执行时间) """
        # 先递归计算所有子节点的预测执行时间
        for child in self.child_plans:
            child.compute_parallel_dop_predictions()

        # 累加子节点的真实执行时间到当前节点的 dop_exec_time_map
        for child in self.child_plans:
            if getattr(child, "thread_id", self.thread_id) == self.thread_id:
                for dop, child_time in child.dop_exec_time_map.items():
                    self.dop_exec_time_map[dop] = self.dop_exec_time_map.get(dop, 0) + child_time

        # 累加子节点的预测执行时间到当前节点的 pred_dop_exec_map
        for child in self.child_plans:
            if getattr(child, "thread_id", self.thread_id) == self.thread_id:
                for dop, child_pred in child.pred_dop_exec_map.items():
                    self.pred_dop_exec_map[dop] = self.pred_dop_exec_map.get(dop, 0) + child_pred

        # 如果当前节点支持并行且已经得到预测参数,则计算当前节点自身的预测执行时间
        if self.is_parallel and self.pred_params is not None:
            for dop in self.dop_exec_time_map.keys():
                pred_exec = (self.pred_params[1] / (dop ** self.pred_params[0])
                            + self.pred_params[2] * (dop ** self.pred_params[3])
                            + self.pred_params[4])
                # 累加当前节点的预测执行时间到已有的子节点累加结果上
                self.pred_dop_exec_map[dop] = self.pred_dop_exec_map.get(dop, 0) + max(pred_exec, 1e-1)
    
    def select_best_dop(cand_map):
        best_dop, best_time = min(cand_map.items(), key=lambda x: x[1])
        return best_dop, best_time
    
class ThreadBlock:
    def __init__(self, thread_id, nodes):
        """
        :param thread_id: 线程块标识
        :param nodes: 属于该线程块的所有 PlanNode
        """
        self.thread_id = thread_id
        self.nodes = nodes
        self.candidate_dops = set()  # 可后续设置
        self.real_dop_exec_time = {}     
        self.pred_dop_exec_time = {}  
        self.thread_execution_time = 0
        self.thread_complete_time = 0
        self.local_data_transfer_start_time = 0
        self.up_data_transfer_start_time = 0
        self.child_thread_ids = set()  # 用于记录子线程块的 thread_id
        self.child_max_execution_time = 0  # 用于记录子线程内的最大执行时间
        self.optimal_dop = None        # 用于记录最终最优 dop
        self.blocking_interval = {}

    def aggregate_metrics(self):
        """
        聚合线程块内所有节点的指标:
          - thread_execution_time:累加所有节点的执行时间
          - thread_complete_time:所有节点中最大的完成时间
          - local/up_data_transfer_start_time:取所有节点中的最大值(仅作参考)
          - candidate_dops:如果未设置,则取所有阻塞节点(materialized==True)的 matched_dops 的交集
          - blocking_interval:在 candidate_dops 下,
              从所有阻塞节点中选取 thread_execution_time 最大的那个作为“最上层阻塞节点”,
              然后直接使用该节点的 pred_dop_exec_map 作为阻塞时间
          - 同时,从该节点的 dop_exec_time_map 取得真实的累加执行时间
        """
        if not self.nodes:
            return

        # 聚合基本指标
        self.thread_execution_time = sum(node.execution_time for node in self.nodes)
        self.thread_complete_time = max(node.thread_complete_time for node in self.nodes)
        self.local_data_transfer_start_time = max(node.local_data_transfer_start_time for node in self.nodes)
        self.up_data_transfer_start_time = max(node.up_data_transfer_start_time for node in self.nodes)
        
        # 如果没有 candidate_dops,则取所有阻塞节点 matched_dops 的交集
        if not self.candidate_dops:
            candidate_dops = None
            for node in self.nodes:
                if node.materialized:
                    if candidate_dops is None:
                        candidate_dops = set(node.matched_dops)
                    else:
                        candidate_dops = candidate_dops.intersection(node.matched_dops)
            self.candidate_dops = candidate_dops if candidate_dops is not None else set()

        # 在线程块内选出“最上层阻塞节点”:定义为 materialized==True 且 thread_execution_time 最大的那个节点
        top_blocking = None
        max_exec = -1
        for node in self.nodes:
            if node.materialized and node.up_data_transfer_start_time > max_exec:
                max_exec = node.up_data_transfer_start_time
                top_blocking = node

        # 如果存在阻塞节点,就直接从该节点获取累加的真实和预测执行时间
        if top_blocking:
            self.real_dop_exec_time = top_blocking.dop_exec_time_map.copy()
            self.pred_dop_exec_time = top_blocking.pred_dop_exec_map.copy()
            # 同时,blocking_interval 直接取该节点的预测执行时间作为阻塞时间(各 dop 下)
            self.blocking_interval = top_blocking.pred_dop_exec_map.copy()
        else:
            # 如果没有阻塞节点,则各 dop 对应时间均为 0
            self.real_dop_exec_time = {dop: 0 for dop in self.candidate_dops}
            self.pred_dop_exec_time = {dop: 0 for dop in self.candidate_dops}
            self.blocking_interval = {dop: 0 for dop in self.candidate_dops}
            
    def choose_optimal_dop(self, child_max_execution_time, 
                        min_improvement_ratio=0.2, 
                        block_threshold=100,  # 阈值单位：毫秒
                        max_block_growth=1.2):
        """
        选择最优 dop 的函数：

        逻辑：
        1. 将 candidate_dops 按 dop 从大到小排列，例如 [64, 32, 16, 8, 4]。
        2. 对于相邻候选解，计算改善率：
                improvement_ratio = (T(smaller_dop) - T(larger_dop)) / T(smaller_dop)
            同时计算绝对差值 diff = (larger_dop - smaller_dop)，然后调整改善率：
                adjusted_improvement = improvement_ratio / (1 + absolute_diff_scale * sqrt(diff))
            如果 adjusted_improvement 低于 min_improvement_ratio,则认为较大 dop 的优势不明显，
            直接丢弃较大 dop,保留较小 dop。
        3. 过滤后得到 filtered_dops(顺序仍为从大到小,对应的预测执行时间单调递增）。
        4. 最终选择阶段：
            - 令 candidate = filtered_dops[-1]（资源最少的解）,其执行时间 candidate_exec 和阻塞时间 candidate_block;
            - 如果 candidate_exec ≤ 0.8 x child_max_execution_time,
                则进一步判断阻塞时间：
                    * 如果 candidate_block ≤ block_threshold,则直接选 candidate;
                    * 否则,比较 candidate_block 与基准阻塞时间(filtered_dops[0])的阻塞增长率,
                    并根据绝对线程差（采用平方根方式）调整阈值：
                        adjusted_max_block_growth = max_block_growth * (1 + absolute_diff_scale * sqrt(filtered_dops[0] - candidate))
                    如果 candidate 的阻塞增长率 ≤ adjusted_max_block_growth,则 candidate 可接受,
                    否则返回 filtered_dops[0]。
            - 如果 candidate_exec > 0.8 x child_max_execution_time,则直接返回 filtered_dops[0]。

        返回 (optimal_dop, pred_dop_exec_time[optimal_dop])
        """


        # 如果候选集为空或只有一个，直接返回
        if not self.candidate_dops:
            return None, None
        if len(self.candidate_dops) == 1:
            single_dop = next(iter(self.candidate_dops))
            return single_dop, self.pred_dop_exec_time.get(single_dop, float('inf'))
        
        # 1. 按 dop 从大到小排列（例如：[64, 32, 16, 8, 4]）
        sorted_dops = sorted(self.candidate_dops, reverse=True)
        
        # 2. 初步过滤：反向比较相邻候选解，使用平方根因子调整改善率
        filtered_dops = [sorted_dops[0]]  # 初始保留最大的 dop
        for dop in sorted_dops[1:]:
            larger_dop = filtered_dops[-1]
            time_larger = self.pred_dop_exec_time.get(larger_dop, float('inf'))
            time_smaller = self.pred_dop_exec_time.get(dop, float('inf'))
            # 计算标准改善率（以较小 dop 为基准）
            improvement_ratio = (time_smaller - time_larger) / time_smaller if time_smaller > 0 else 0
            # 计算绝对差值（线程数差）
            diff = larger_dop - dop
            factor = 1 + math.log2(diff)
            adjusted_improvement = improvement_ratio / factor
            if adjusted_improvement < min_improvement_ratio:
                # 改善不明显，则丢弃较大 dop，用较小 dop替换
                filtered_dops[-1] = dop
            else:
                filtered_dops.append(dop)
        # 如果过滤后只剩一个，直接返回
        if len(filtered_dops) == 1:
            final_dop = filtered_dops[0]
            return final_dop, self.pred_dop_exec_time.get(final_dop, float('inf'))

        # 3. 最终选择阶段：
        # filtered_dops[-1] 为资源最少的候选解，filtered_dops[0] 为资源最多的候选解
        candidate = filtered_dops[-1]
        candidate_exec = self.pred_dop_exec_time.get(candidate, float('inf'))
        candidate_block = self.blocking_interval.get(candidate, float('inf'))
        
        base = filtered_dops[0]
        base_block = self.blocking_interval.get(base, float('inf'))
        
        # 调整阻塞增长率阈值：采用平方根因子调整绝对线程差
        abs_thread_diff = base - candidate
        adjusted_max_block_growth = max_block_growth * (1 + math.log2(abs_thread_diff))
        
        if candidate_exec <= 0.8 * child_max_execution_time:
            if candidate_block <= block_threshold:
                optimal_dop = candidate
            else:
                block_growth = candidate_block / base_block if base_block > 0 else float('inf')
                if block_growth <= adjusted_max_block_growth:
                    optimal_dop = candidate
                else:
                    optimal_dop = base
        else:
            optimal_dop = base
        
        return optimal_dop, self.pred_dop_exec_time.get(optimal_dop, float('inf'))



    
