import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

# 路径配置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASETS = ['tpch', 'tpcds']
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output/tru_cdf", f"exp3_execution_time_only.pdf")

fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 👈 两列一行

for col, data_set in enumerate(DATASETS):
    INPUT_DIR = os.path.join(PROJECT_ROOT, "input/tru_cdf", data_set)
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

    ax = axs[col]
    for file_path in csv_files:
        df = pd.read_csv(file_path, sep=';')
        df = df[df['Execution Time Q-error'].le(2000)]
        qerrors = df['Execution Time Q-error'].dropna()
        qerrors = qerrors[qerrors > 0].sort_values()
        y = np.linspace(0, 1, len(qerrors))
        label = os.path.splitext(os.path.basename(file_path))[0]
        ax.plot(qerrors, y, label=label)
    ax.set_xscale('log')
    ax.set_title('Execution Time Q-Error')
    ax.set_xlabel('Q-Error')
    ax.set_ylabel('CDF')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=13)

# 设置标签 (a)、(b)
axs[0].text(0.5, -0.25, '(a) TPC-H', transform=axs[0].transAxes, fontsize=18, ha='center')
axs[1].text(0.5, -0.25, '(b) TPC-DS', transform=axs[1].transAxes, fontsize=18, ha='center')

plt.subplots_adjust(
    wspace=0.25,
    top=0.9,
    bottom=0.2
)

# 保存 PDF
plt.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight')
print(f"✅ 执行时间 Q-error 的 CDF 图已保存为 PDF：{OUTPUT_PATH}")
