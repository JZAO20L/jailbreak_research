"""
生成跨模型×跨数据集实验结果热力图
作为论文的主要统计图（main figure）
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 实验数据（从 all_exp_results.md 提取）
# 格式：[model][dataset][method] = ASR
data = {
    'Qwen3-0.6B\n(same family)': {
        'default': {'no_rewrite': 42.6, 'PAIR': 77.3, 'AutoDAN': 88.0, 'DeepInception': 47.2, 'Persona': 46.5, 'PAIR+SESS': 85.5, 'DAN+SESS': 100.0},
        'advbench': {'no_rewrite': 46.0, 'PAIR': 88.1, 'AutoDAN': 92.1, 'DeepInception': 79.2, 'Persona': 81.4, 'PAIR+SESS': 98.5, 'DAN+SESS': 100.0},
        'hb_ctx': {'no_rewrite': 86.0, 'PAIR': 86.0, 'AutoDAN': 95.0, 'DeepInception': 85.0, 'Persona': 85.0, 'PAIR+SESS': 95.0, 'DAN+SESS': 100.0},
        'hb_std': {'no_rewrite': 63.5, 'PAIR': 90.5, 'AutoDAN': 91.0, 'DeepInception': 85.0, 'Persona': 88.0, 'PAIR+SESS': 98.5, 'DAN+SESS': 100.0},
        'jbBench': {'no_rewrite': 52.0, 'PAIR': 90.0, 'AutoDAN': 85.0, 'DeepInception': 76.0, 'Persona': 75.0, 'PAIR+SESS': 98.0, 'DAN+SESS': 100.0},
    },
    'Qwen3-4B\n(same family)': {
        'default': {'no_rewrite': 30.3, 'PAIR': 55.5, 'AutoDAN': 86.8, 'DeepInception': 40.4, 'Persona': 32.3, 'PAIR+SESS': 79.0, 'DAN+SESS': 100.0},
        'advbench': {'no_rewrite': 3.3, 'PAIR': 77.3, 'AutoDAN': 75.4, 'DeepInception': 41.0, 'Persona': 16.7, 'PAIR+SESS': 81.0, 'DAN+SESS': 100.0},
        'hb_ctx': {'no_rewrite': 79.0, 'PAIR': 81.0, 'AutoDAN': 87.0, 'DeepInception': 80.0, 'Persona': 67.0, 'PAIR+SESS': 98.0, 'DAN+SESS': 100.0},
        'hb_std': {'no_rewrite': 21.5, 'PAIR': 75.5, 'AutoDAN': 75.5, 'DeepInception': 47.5, 'Persona': 36.0, 'PAIR+SESS': 91.5, 'DAN+SESS': 100.0},
        'jbBench': {'no_rewrite': 7.0, 'PAIR': 72.0, 'AutoDAN': 80.0, 'DeepInception': 38.0, 'Persona': 20.0, 'PAIR+SESS': 88.0, 'DAN+SESS': 100.0},
    },
    'Qwen3-14B\n(same family)': {
        'default': {'no_rewrite': 22.6, 'PAIR': 58.5, 'AutoDAN': 80.9, 'DeepInception': 32.0, 'Persona': 26.2, 'PAIR+SESS': 77.0, 'DAN+SESS': 99.7},
        'advbench': {'no_rewrite': 2.3, 'PAIR': 77.9, 'AutoDAN': 90.8, 'DeepInception': 40.4, 'Persona': 14.2, 'PAIR+SESS': 84.6, 'DAN+SESS': 99.4},
        'hb_ctx': {'no_rewrite': 66.0, 'PAIR': 76.0, 'AutoDAN': 83.0, 'DeepInception': 65.0, 'Persona': 60.0, 'PAIR+SESS': 90.0, 'DAN+SESS': 100.0},
        'hb_std': {'no_rewrite': 23.5, 'PAIR': 71.5, 'AutoDAN': 84.0, 'DeepInception': 36.0, 'Persona': 21.5, 'PAIR+SESS': 91.0, 'DAN+SESS': 99.0},
        'jbBench': {'no_rewrite': 7.0, 'PAIR': 72.0, 'AutoDAN': 89.0, 'DeepInception': 37.0, 'Persona': 17.0, 'PAIR+SESS': 87.0, 'DAN+SESS': 99.0},
    },
    'GPT-OSS-20B\n(cross family)': {
        'default': {'no_rewrite': 0.7, 'PAIR': 0.2, 'AutoDAN': 12.5, 'DeepInception': 0.4, 'Persona': 1.5, 'PAIR+SESS': 12.9, 'DAN+SESS': 30.0},
        'advbench': {'no_rewrite': 0.0, 'PAIR': 0.2, 'AutoDAN': 3.3, 'DeepInception': 0.0, 'Persona': 0.0, 'PAIR+SESS': 8.3, 'DAN+SESS': 28.1},
        'hb_ctx': {'no_rewrite': 0.0, 'PAIR': 0.0, 'AutoDAN': 11.0, 'DeepInception': 1.0, 'Persona': 0.0, 'PAIR+SESS': 12.0, 'DAN+SESS': 29.0},
        'hb_std': {'no_rewrite': 0.0, 'PAIR': 0.0, 'AutoDAN': 1.5, 'DeepInception': 0.0, 'Persona': 1.0, 'PAIR+SESS': 13.0, 'DAN+SESS': 36.5},
        'jbBench': {'no_rewrite': 0.0, 'PAIR': 0.0, 'AutoDAN': 2.0, 'DeepInception': 0.0, 'Persona': 1.0, 'PAIR+SESS': 10.0, 'DAN+SESS': 39.0},
    },
}

# 准备数据矩阵
models = list(data.keys())
datasets = ['default', 'advbench', 'hb_ctx', 'hb_std', 'jbBench']
# 调整顺序：baseline在前，SESS增强在后，PAIR+SESS倒数第二
methods = ['no_rewrite', 'DeepInception', 'Persona', 'PAIR', 'AutoDAN', 'PAIR+SESS', 'DAN+SESS']

# 创建 4 个子图（每个模型一个）
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

# 颜色映射
cmap = 'RdYlGn'  # 红-黄-绿，红色=低ASR，绿色=高ASR
vmin, vmax = 0, 100

for idx, model in enumerate(models):
    ax = axes[idx]
    
    # 构建热力图数据矩阵
    matrix = []
    for method in methods:
        row = []
        for dataset in datasets:
            row.append(data[model][dataset][method])
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # 绘制热力图（使用 matplotlib imshow）
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('ASR (%)', fontsize=9)
    
    # 添加数值标注
    for i in range(len(methods)):
        for j in range(len(datasets)):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    # 设置刻度和标签
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel('Dataset', fontsize=10)
    ax.set_ylabel('Method', fontsize=10)
    
    # 设置标题
    ax.set_title(model, fontsize=12, fontweight='bold', pad=10)
    
    # 设置标签
    ax.set_xlabel('Dataset', fontsize=10)
    ax.set_ylabel('Method', fontsize=10)
    
    # 添加分隔线区分同族/跨族
    if 'cross' in model:
        ax.set_facecolor('#FFF5F5')  # 淡红色背景
    else:
        ax.set_facecolor('#F5FFF5')  # 淡绿色背景

# 添加总标题
fig.suptitle('Cross-Model and Cross-Dataset Transfer Results\n(Same Family vs. Cross Family)', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()

# 保存
plt.savefig('main_figure_cross_model_dataset.png', dpi=300, bbox_inches='tight')
plt.savefig('main_figure_cross_model_dataset.pdf', bbox_inches='tight')
print('已生成: main_figure_cross_model_dataset.png/pdf')

# ============================================================================
# 第二个图：简化版汇总图（仅展示关键对比）
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(12, 8))

# 计算每个模型的平均 ASR
summary_data = {}
for model in models:
    summary_data[model] = {
        'PAIR': np.mean([data[model][d]['PAIR'] for d in datasets]),
        'AutoDAN': np.mean([data[model][d]['AutoDAN'] for d in datasets]),
        'PAIR+SESS': np.mean([data[model][d]['PAIR+SESS'] for d in datasets]),
        'DAN+SESS': np.mean([data[model][d]['DAN+SESS'] for d in datasets]),
    }

# 绘制分组柱状图
x = np.arange(len(models))
width = 0.2

bars1 = ax2.bar(x - 1.5*width, [summary_data[m]['PAIR'] for m in models], width, label='PAIR', color='#FFB6C1')
bars2 = ax2.bar(x - 0.5*width, [summary_data[m]['AutoDAN'] for m in models], width, label='AutoDAN', color='#DDA0DD')
bars3 = ax2.bar(x + 0.5*width, [summary_data[m]['PAIR+SESS'] for m in models], width, label='PAIR+SESS', color='#87CEEB')
bars4 = ax2.bar(x + 1.5*width, [summary_data[m]['DAN+SESS'] for m in models], width, label='DAN+SESS', color='#90EE90')

# 添加数值标签
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=8)

ax2.set_xlabel('Target Model', fontsize=12)
ax2.set_ylabel('Average ASR (%)', fontsize=12)
ax2.set_title('Cross-Model Transfer Performance (Average over 5 Datasets)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()
ax2.set_ylim(0, 110)
ax2.grid(axis='y', alpha=0.3)

# 添加分隔线
ax2.axvline(x=2.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax2.text(2.5, 105, 'Cross Family', ha='center', fontsize=10, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('summary_figure_cross_model.png', dpi=300, bbox_inches='tight')
plt.savefig('summary_figure_cross_model.pdf', bbox_inches='tight')
print('已生成: summary_figure_cross_model.png/pdf')

plt.close('all')
