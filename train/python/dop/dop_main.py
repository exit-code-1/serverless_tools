import glob
import os
import sys
import pandas as pd
import dop_operator_train
# 将项目根目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import utils
# Load and merge data
# csv_path_pattern='/home/zhy/opengauss/data_file/tpch_10_output/plan_info.csv'
# csv_files = glob.glob(csv_path_pattern, recursive=True)
# data_list = [pd.read_csv(file, delimiter=';', encoding='utf-8') for file in csv_files]
# data = pd.concat(data_list, ignore_index=True)
data = pd.read_csv('/home/zhy/opengauss/data_file/tpch_10g_output_500/plan_info.csv', delimiter=';', encoding='utf-8')
dop_operator_train.train_all_operators(data, total_queries=500, train_ratio=0.8)