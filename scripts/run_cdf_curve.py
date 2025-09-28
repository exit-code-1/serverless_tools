# 文件路径: scripts/run_cdf_curve.py (增强版)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASETS = ['tpch', 'tpcds']
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output/cdf", f"exp2_predict_accuracy.pdf")

METHOD_ORDER = {
    'Auto-DOP': 0, 'PPM': 1, 'POO': 2,
}

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for row, data_set in enumerate(DATASETS):
    INPUT_DIR = os.path.join(PROJECT_ROOT, "input/cdf", data_set)
    if not os.path.exists(INPUT_DIR):
        print(f"警告: 目录 {INPUT_DIR} 不存在，跳过数据集 {data_set}")
        continue
        
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    def get_sort_key(file_path):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        return METHOD_ORDER.get(filename, 999)
    
    csv_files.sort(key=get_sort_key)

    # --- 绘图循环 ---
    for col_idx, (metric_name, ax) in enumerate(zip(['Execution Time Q-error', 'Memory Q-error'], [axs[row, 0], axs[row, 1]])):
        for file_path in csv_files:
            label = os.path.splitext(os.path.basename(file_path))[0]
            try:
                # 尝试用分号读取
                df = pd.read_csv(file_path, sep=';')
                # 如果找不到列，尝试用逗号读取
                if metric_name not in df.columns:
                    df = pd.read_csv(file_path, sep=',')
                
                # 再次检查列是否存在
                if metric_name not in df.columns:
                    print(f"警告: 文件 '{file_path}' 中找不到列 '{metric_name}'，已跳过此文件。")
                    print(f"      可用列为: {df.columns.tolist()}")
                    continue

                df_filtered = df[df[metric_name].le(2000)]
                qerrors = df_filtered[metric_name].dropna()
                qerrors = qerrors[qerrors > 0].sort_values()
                
                if not qerrors.empty:
                    y = np.linspace(0, 1, len(qerrors))
                    ax.plot(qerrors, y, label=label)

            except Exception as e:
                print(f"错误: 处理文件 '{file_path}' 时发生异常: {e}")
                continue

        ax.set_xscale('log')
        ax.set_title(metric_name.replace(' Q-error', '')) # 简化标题
        ax.set_xlabel('Q-Error')
        ax.set_ylabel('CDF')
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=13)

# ... (后面的布局和保存代码保持不变) ...
plt.subplots_adjust(
    hspace=0.35,
    top=0.93,
    bottom=0.09
)

row0_left  = axs[0, 0].get_position().x0
row0_right = axs[0, 1].get_position().x1
row1_left  = axs[1, 0].get_position().x0
row1_right = axs[1, 1].get_position().x1

row0_mid_x = (row0_left + row0_right) / 2
row1_mid_x = (row1_left + row1_right) / 2

fig.text(row0_mid_x, 0.5, '(a) TPC-H', fontsize=18, ha='center')
fig.text(row1_mid_x, 0.015, '(b) TPC-DS', fontsize=18, ha='center')

plt.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight')
print(f"✅ 合并 CDF 图已保存为 PDF：{OUTPUT_PATH}")