import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

plt.rcParams['font.family'] = 'Times New Roman'

# 路径配置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

INPUT_CSV = os.path.join(PROJECT_ROOT, "output/evaluations/optimization_comparison_report.csv")  # 替换为你的实际路径
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output/overall", f"exp1_overall_performance.pdf")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 读取数据
df = pd.read_csv(INPUT_CSV, sep=';')
df['query_id'] = df['query_id'].astype(str)
query_ids = df['query_id'].values
x = np.arange(len(query_ids))
bar_width = 0.15

# 数据
default_time = np.ones_like(x)
auto_dop_time = df['auto_dop_time_ratio'].values
PPM_time = df['PPM_time_ratio'].values
pio_time = df['PIO_time_ratio'].values
pipeline_time = df['POO_time_ratio'].values

default_cost = np.ones_like(x)
auto_dop_cost = df['auto_dop_cost_ratio'].values
PPM_cost = df['PPM_cost_ratio'].values
pio_cost = df['PIO_cost_ratio'].values
pipeline_cost = df['POO_cost_ratio'].values
# 图设置
fig, axs = plt.subplots(2, 1, figsize=(14, 8))

# --- Execution Time Ratio 图 ---
axs[0].bar(x - 1.5 * bar_width, default_time, width=bar_width, label='Default', color='gray', edgecolor='black')
axs[0].bar(x - 0.5 * bar_width, auto_dop_time, width=bar_width, label='Auto-DOP', color='steelblue', edgecolor='black')
axs[0].bar(x + 0.5 * bar_width, PPM_time, width=bar_width, label='PPM', color='red', edgecolor='black')
axs[0].bar(x + 1.5 * bar_width, pio_time, width=bar_width, label='PIO', color='darkgreen', edgecolor='black')
axs[0].bar(x + 2.5 * bar_width, pipeline_time, width=bar_width, label='POO', color='orange', edgecolor='black')
axs[0].set_xlabel('Query ID') 
axs[0].set_ylabel('Execution Time Ratio')
axs[0].set_xticks(x)
axs[0].set_xticklabels(query_ids)
axs[0].grid(True, axis='y', linestyle='--', linewidth=0.5)
axs[0].legend(fontsize=10)
axs[0].set_xlim([-0.5, len(x) - 0.5])  # 👈 消除左右空白

# --- Cost Ratio 图 ---
axs[1].bar(x - 1.5 * bar_width, default_cost, width=bar_width, label='Default', color='gray', edgecolor='black')
axs[1].bar(x - 0.5 * bar_width, auto_dop_cost, width=bar_width, label='Auto-DOP', color='steelblue', edgecolor='black')
axs[1].bar(x + 0.5 * bar_width, PPM_cost, width=bar_width, label='PPM', color='red', edgecolor='black')
axs[1].bar(x + 1.5 * bar_width, pio_cost, width=bar_width, label='PIO', color='darkgreen', edgecolor='black')
axs[1].bar(x + 2.5 * bar_width, pipeline_cost, width=bar_width, label='POO', color='orange', edgecolor='black')

axs[1].set_ylabel('Cost Ratio')
axs[1].set_xlabel('Query ID')
axs[1].set_xticks(x)
axs[1].set_xticklabels(query_ids)
axs[1].grid(True, axis='y', linestyle='--', linewidth=0.5)
axs[1].legend(fontsize=10)
axs[1].set_xlim([-0.5, len(x) - 0.5])  # 👈 消除左右空白

# 去掉子图的x轴刻度线
axs[0].tick_params(axis='x', length=0)  # length=0 表示刻度线长度为0，也就是不显示刻度线
axs[1].tick_params(axis='x', length=0)

# 间距 & 标签美化
plt.subplots_adjust(hspace=0.35, top=0.91, bottom=0.13)

# 自定义居中标签标题
fig.text(0.5, 0.50, '(a) Execution Time Ratio', ha='center', fontsize=14)
fig.text(0.5, 0.03, '(b) Cost Ratio', ha='center', fontsize=14)

# 保存
plt.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight')
print(f"✅ 成功保存 PDF：{OUTPUT_PATH}")