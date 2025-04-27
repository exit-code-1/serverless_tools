import os
import time
import numpy as np
import math
import onnxruntime
import torch
import utils
from structure import no_dop_operator_features, no_dop_operators_exec, no_dop_operators_mem, dop_operators_exec, dop_operators_mem, parallel_op

default_dop = 8
thread_cost = 6
thread_mem = 8194
dop_sets = {1,8,16,32,64,96,128}

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
        self.estimate_costs = plan_data['estimate_costs']
        self.pred_execution_time = 0
        self.pred_mem = 0
        self.best_dop = 0
        self.thread_id = 0
        self.peak_mem = plan_data['peak_mem']
        self.is_parallel = (self.operator_type in parallel_op)
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
        self.exec_time_map = {self.dop: self.execution_time}  # 直接初始化
        self.matched_dops = {self.dop}  # 记录匹配的 dop
        # **存储不同 dop 的预测执行时间**
        self.pred_dop_exec_map = {}
        self.true_dop_exec_map = {self.dop: self.execution_time}

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
        if dop in dop_sets:
            self.exec_time_map[dop] = execution_time
            self.true_dop_exec_map[dop] = execution_time
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
            self.pred_mem = 1024 * self.dop
            return
        if self.operator_type in dop_operators_mem:
            pred_params = self.onnx_manager.infer_mem(self.operator_type, self.mem_feature_data)
            pred_mem = max(pred_params[1] * (self.dop ** pred_params[0]) + pred_params[2], pred_params[3])
            self.pred_mem = max(pred_mem, 1e-1)
        else:
            self.pred_mem = max(self.onnx_manager.infer_mem(self.operator_type, self.mem_feature_data), 1e-1)
        end_time = time.time()
        self.pred_mem_time += end_time - start_time
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
        self.pred_time = 0

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
        
        # 如果没有 candidate_dops,则取所有节点 matched_dops 的交集
        if not self.candidate_dops:
            candidate_dops = None
            for node in self.nodes:
                if not node.is_parallel:
                    candidate_dops = {1}
                    break
                if candidate_dops is None:
                    candidate_dops = set(node.matched_dops)
                else:
                    candidate_dops = candidate_dops.intersection(node.matched_dops)
            self.candidate_dops = candidate_dops if candidate_dops is not None else set()
            
        # 在线程块内选出“最上层阻塞节点”:定义为 materialized==True 且 thread_execution_time 最大的那个节点
        top_blocking = None
        top_node = self.nodes[0]
        max_block_exec = -1
        max_exec = -1
        for node in self.nodes:
            if node.materialized and node.up_data_transfer_start_time > max_block_exec:
                max_block_exec = node.up_data_transfer_start_time
                top_blocking = node
            if node.thread_execution_time > max_exec:
                max_exec = node.thread_execution_time
                top_node = node

        self.real_dop_exec_time = top_node.true_dop_exec_map.copy()
        self.pred_dop_exec_time = top_node.pred_dop_exec_map.copy()
        # 如果存在阻塞节点,就直接从该节点获取累加的真实和预测执行时间
        if top_blocking:
            self.blocking_interval = top_blocking.pred_dop_exec_map.copy()
        else:
            self.blocking_interval = {dop: 0 for dop in self.candidate_dops}
            
    def choose_optimal_dop(self, child_max_execution_time, 
                        min_improvement_ratio=0.2, 
                        min_reduction_threshold=200,  # 绝对减少时间阈值（单位：ms）
                        block_threshold=200,  
                        max_block_growth=1.2):
        """
        选择最优 dop 的函数，改进逻辑：
        1. 先对 candidate_dops 按 dop 从小到大排序（假设 dop 数值越大，预测执行时间越低）
        2. 过滤阶段：遍历排序后的 candidate_dops，计算改善率和绝对减少量，
        其中改善率 = (T(前一个dop) - T(当前dop)) / T(当前dop)
        绝对减少量 = T(前一个dop) - T(当前dop)
        同时引入调整因子 factor = 1 + log2(当前dop - 前一个dop)
        如果调整后的改善率不足（或减少量不足）则用当前 dop 替换前一个；
        否则，将当前 dop 添加到过滤列表中。
        3. 最终选择阶段：遍历过滤后的 dop（按从小到大顺序），
        对每个 candidate：
            - 首先判断 candidate 的预测执行时间 candidate_exec，
            如果 candidate_exec > 0.8 * child_max_execution_time，则跳过该 candidate；
            - 否则，再判断 candidate 的阻塞时间 candidate_block：
                * 如果 candidate_block <= block_threshold，则直接选择该 candidate；
                * 否则，计算相对于基准 dop（过滤列表中最小的 dop）的阻塞增长率：
                    block_growth = candidate_block / base_block，
                    其中 base_block 为基准 dop 的阻塞时间；
                    同时计算调整后的最大阻塞增长率：adjusted_max_block_growth = max_block_growth * (1 + log2(candidate - base_dop))
                * 如果 block_growth <= adjusted_max_block_growth 且 (candidate_block - base_block) < min_reduction_threshold，则选择 candidate。
        如果遍历完都没有 candidate 满足条件，则返回过滤列表中最小的 dop（即预测执行时间最低的）。
        """
        if not self.candidate_dops:
            return None
        if len(self.candidate_dops) == 1:
            single_dop = next(iter(self.candidate_dops))
            self.optimal_dop = single_dop
            self.pred_time = self.pred_dop_exec_time[next(iter(self.pred_dop_exec_time))]
            return single_dop

        # 1. 按 dop 从小到大排序
        sorted_dops = sorted(self.candidate_dops)  # 假定 dop 为数字，升序排列
        # 2. 过滤阶段：结合改善率和减少量
        filtered_dops = [sorted_dops[0]]
        for dop in sorted_dops[1:]:
            prev_dop = filtered_dops[-1]
            time_prev = self.pred_dop_exec_time.get(prev_dop, float('1e-1'))
            time_curr = self.pred_dop_exec_time.get(dop, float('1e-1'))
            if time_curr <= 0:
                continue

            # 计算改善率和减少量
            improvement_ratio = (time_prev - time_curr) / time_prev
            absolute_reduction = time_prev - time_curr
            diff = dop - prev_dop  # 必然为正
            factor = 1 + math.log2(diff) if diff > 0 else 1
            adjusted_improvement = improvement_ratio / factor

            # 根据 absolute_reduction 动态调整最小改善率要求
            if absolute_reduction < min_reduction_threshold:
                continue
            else:
                # 当减少量大时，阈值降低，允许较低的改善率
                dynamic_min_improvement = min_improvement_ratio / math.log10(absolute_reduction)
            # 仅当调整后的改善率大于或等于动态阈值，并且减少量达到要求时，保留当前 dop
            if adjusted_improvement >= dynamic_min_improvement :
                filtered_dops.append(dop)

        # 若过滤后只有一个候选，则直接返回
        if len(filtered_dops) == 1:
            final_dop = filtered_dops[0]
            self.optimal_dop = final_dop
            self.pred_time = self.pred_dop_exec_time.get(final_dop, float('inf'))
            return final_dop

        # 3. 最终选择阶段：遍历过滤后的 candidate，从小到大
        base_dop = filtered_dops[0]  # 初始化为最小的 DOP
        base_block = self.blocking_interval.get(base_dop, float('inf'))

        for candidate in filtered_dops[1:]:  # 从小到大遍历
            candidate_exec = self.pred_dop_exec_time.get(candidate, float('inf'))

            # 先判断执行时间条件，不满足则更新 base_dop 并跳过
            if candidate_exec > 0.8 * child_max_execution_time:
                base_dop = candidate
                base_block = self.blocking_interval.get(base_dop, float('inf'))
                continue

            candidate_block = self.blocking_interval.get(candidate, float('inf'))
            # 如果阻塞时间低于阈值，直接选择该 candidate
            if candidate_block <= block_threshold:
                self.optimal_dop = candidate
                self.pred_time = candidate_exec
                break
            else:
                # 计算阻塞增长率：前一个 DOP 阻塞时间 / 当前 DOP 阻塞时间
                block_growth = base_block / candidate_block if candidate_block > 0 else float('inf')
                diff = candidate - base_dop
                adjusted_max_block_growth = max_block_growth * (1 + math.log2(diff)) if diff > 0 else max_block_growth

                # 如果增长率在可接受范围内，直接选择当前 DOP
                if block_growth <= adjusted_max_block_growth:
                    self.optimal_dop = candidate
                    self.pred_time = candidate_exec
                    break
                else:
                    # 增长率过大，更新 base_dop 和 base_block
                    base_dop = candidate
                    base_block = candidate_block

        # 使用最后更新的 base_dop
        self.optimal_dop = base_dop
        self.pred_time = self.pred_dop_exec_time.get(base_dop, float('inf'))

        return self.optimal_dop



    
