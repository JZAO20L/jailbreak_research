"""
主实验结果图：蓝色系分组柱状图
展示 SESS 对 PAIR 和 DAN 的提升，以及跨模型迁移效果
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 从热力图提取核心数据（default 数据集）
models = ['Qwen3-0.6B', 'Qwen3-4B', 'Qwen3-14B', 'GPT-OSS-20B']
data = {
    'PAIR': [77.3, 55.5, 58.5, 0.2],
    'PAIR+SESS': [85.5, 79.0, 77.0, 12.9],
    'AutoDAN': [88.0, 86.8, 80.9, 12.5],
    'DAN+SESS': [100.0, 100.0, 99.7, 30.0],
}

# 蓝色系配色（从浅到深）
colors_blue = ['#E6F2FF', '#99CCFF', '#3399FF', '#0066CC']  # 从浅到深的蓝色

fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(models))
width = 0.2

# 绘制分组柱状图（顺序调整：baseline在前，SESS增强在后）
bars1 = ax.bar(x - 1.5*width, data['PAIR'], width, label='PAIR', color=colors_blue[0], edgecolor='#333333', linewidth=0.5)
bars2 = ax.bar(x - 0.5*width, data['AutoDAN'], width, label='AutoDAN', color=colors_blue[1], edgecolor='#333333', linewidth=0.5)
bars3 = ax.bar(x + 0.5*width, data['PAIR+SESS'], width, label='PAIR+SESS', color=colors_blue[2], edgecolor='#333333', linewidth=0.5)
bars4 = ax.bar(x + 1.5*width, data['DAN+SESS'], width, label='DAN+SESS', color=colors_blue[3], edgecolor='#333333', linewidth=0.5)

# 添加数值标签
def add_labels(bars, fontsize=9):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=fontsize, rotation=0)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# 设置标签
ax.set_xlabel('Target Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Cross-Model Transfer Performance (Default Dataset)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

# 设置 Y 轴范围
ax.set_ylim(0, 115)

# 添加网格线
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# 添加分隔线，区分同族和跨族
ax.axvline(x=2.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.text(2.5, 110, 'Cross Family', ha='center', fontsize=10, color='red', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))

# 添加提升标注（在柱子上方）
# 同族模型平均提升
avg_improvement_pair = np.mean([data['PAIR+SESS'][i] - data['PAIR'][i] for i in range(3)])
avg_improvement_dan = np.mean([data['DAN+SESS'][i] - data['AutoDAN'][i] for i in range(3)])

# 在图底部添加说明
textstr = f'Same Family Avg Improvement:\nPAIR → PAIR+SESS: +{avg_improvement_pair:.1f}pp\nAutoDAN → DAN+SESS: +{avg_improvement_dan:.1f}pp'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray')
ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', bbox=props)

plt.tight_layout()

# 保存
plt.savefig('main_bar_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('main_bar_chart.pdf', bbox_inches='tight')
print('已生成: main_bar_chart.png/pdf')

plt.close()
