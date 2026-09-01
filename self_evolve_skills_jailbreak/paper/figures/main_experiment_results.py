#!/usr/bin/env python3
"""
主实验结果图：多模型×多数据集分组柱状图
风格与 SESS_04 保持一致（蓝色系配色）
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# Global Style Configuration (与 SESS_04 保持一致)
# =============================================================================

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11

# Unified Blue Color Palette
COLORS = {
    'primary': '#2563EB',      # Deep blue (main bars)
    'secondary': '#3B82F6',    # Standard blue
    'light': '#60A5FA',        # Light blue
    'lighter': '#93C5FD',      # Lighter blue
    'lightest': '#BFDBFE',     # Lightest blue
    'baseline': '#9CA3AF',     # Gray (baseline/reference)
    'accent': '#10B981',       # Green (positive improvement)
    'warning': '#F59E0B',      # Orange (neutral)
    'danger': '#EF4444',       # Red (negative/cross-family)
    'dark': '#1E3A5F',         # Dark navy
}

OUTPUT_DIR = '/home/tiger/jailbreak_research/self_evolve_skills_jailbreak/paper/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_figure(fig, filename):
    """Save figure with consistent settings (PNG + PDF)"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    # Save PNG
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ {filename}")
    # Save PDF (矢量图)
    pdf_filename = filename.replace('.png', '.pdf')
    pdf_filepath = os.path.join(OUTPUT_DIR, pdf_filename)
    fig.savefig(pdf_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  ✓ {pdf_filename}")
    plt.close(fig)


def set_grid_off(ax):
    """Remove top and right spines"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_y_grid(ax, alpha=0.15):
    """Add subtle y-axis grid"""
    ax.grid(True, axis='y', alpha=alpha, linestyle='-', linewidth=0.5)


# =============================================================================
# 主实验结果图：多模型×多数据集
# =============================================================================

def main_experiment_results():
    """主实验结果图：4 模型 × 5 数据集 × 4 方法"""
    
    # 从热力图提取数据
    models = ['Qwen3-0.6B', 'Qwen3-4B', 'Qwen3-14B', 'GPT-OSS-20B']
    datasets = ['default', 'advbench', 'hb_ctx', 'hb_std', 'jbBench']
    
    # 数据：[model][dataset][method]
    data = {
        'PAIR': {
            'Qwen3-0.6B': [77.3, 88.1, 86.0, 90.5, 90.0],
            'Qwen3-4B': [55.5, 77.3, 81.0, 75.5, 72.0],
            'Qwen3-14B': [58.5, 77.9, 76.0, 71.5, 72.0],
            'GPT-OSS-20B': [0.2, 0.2, 0.0, 0.0, 0.0],
        },
        'PAIR+SESS': {
            'Qwen3-0.6B': [85.5, 98.5, 95.0, 98.5, 98.0],
            'Qwen3-4B': [79.0, 81.0, 98.0, 91.5, 88.0],
            'Qwen3-14B': [77.0, 84.6, 90.0, 91.0, 87.0],
            'GPT-OSS-20B': [12.9, 8.3, 12.0, 13.0, 10.0],
        },
        'AutoDAN': {
            'Qwen3-0.6B': [88.0, 92.1, 95.0, 91.0, 85.0],
            'Qwen3-4B': [86.8, 75.4, 87.0, 75.5, 80.0],
            'Qwen3-14B': [80.9, 90.8, 83.0, 84.0, 89.0],
            'GPT-OSS-20B': [12.5, 3.3, 11.0, 1.5, 2.0],
        },
        'DAN+SESS': {
            'Qwen3-0.6B': [100.0, 100.0, 100.0, 100.0, 100.0],
            'Qwen3-4B': [100.0, 100.0, 100.0, 100.0, 100.0],
            'Qwen3-14B': [99.7, 99.4, 100.0, 99.0, 99.0],
            'GPT-OSS-20B': [30.0, 28.1, 29.0, 36.5, 39.0],
        },
    }

    # 创建 4 个子图（每个模型一个）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]

        # 准备数据
        pair_vals = data['PAIR'][model]
        pair_sess_vals = data['PAIR+SESS'][model]
        autodan_vals = data['AutoDAN'][model]
        dan_sess_vals = data['DAN+SESS'][model]

        x = np.arange(len(datasets))
        width = 0.2

        # 绘制分组柱状图（顺序调整：baseline在前，SESS增强在后）
        bars1 = ax.bar(x - width * 1.5, pair_vals, width, label='PAIR',
                       color=COLORS['lightest'], edgecolor='#CBD5E1', linewidth=1)
        bars2 = ax.bar(x - width * 0.5, autodan_vals, width, label='AutoDAN',
                       color=COLORS['lighter'], edgecolor='#CBD5E1', linewidth=1)
        bars3 = ax.bar(x + width * 0.5, pair_sess_vals, width, label='PAIR+SESS',
                       color=COLORS['secondary'], edgecolor='#CBD5E1', linewidth=1)
        bars4 = ax.bar(x + width * 1.5, dan_sess_vals, width, label='DAN+SESS',
                       color=COLORS['primary'], edgecolor='#CBD5E1', linewidth=1)
        
        # 添加数值标签（GPT-OSS 低值也标注，其余仅显示 > 50 的值）
        is_gpt = (model == 'GPT-OSS-20B')
        for bar, val in zip(bars1, pair_vals):
            if is_gpt or val > 50:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.1f}', ha='center', fontsize=7, fontweight='bold', color=COLORS['dark'])

        for bar, val in zip(bars2, autodan_vals):
            if is_gpt or val > 50:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.1f}', ha='center', fontsize=7, fontweight='bold', color=COLORS['dark'])

        for bar, val in zip(bars3, pair_sess_vals):
            if is_gpt or val > 50:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.1f}', ha='center', fontsize=7, fontweight='bold', color=COLORS['dark'])

        for bar, val in zip(bars4, dan_sess_vals):
            if is_gpt or val > 50:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.1f}', ha='center', fontsize=7, fontweight='bold', color=COLORS['dark'])
        
        # 设置标签
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=9, rotation=15, ha='right')
        ax.set_ylabel('Attack Success Rate (%)', fontsize=10, fontweight='bold')
        ax.set_title(model, fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim(0, 115)
        
        # 添加网格
        add_y_grid(ax)
        set_grid_off(ax)
        
        # 添加图例（仅最后一个子图，右上角）
        if idx == 3:
            ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    
    # 添加总标题
    fig.suptitle('Main Experiment Results: Cross-Model and Cross-Dataset Transfer', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, 'main_experiment_results.png')


# =============================================================================
# 主实验结果图：简化版（仅 default 数据集）
# =============================================================================

def main_experiment_results_simple():
    """主实验结果图：仅 default 数据集，4 模型对比"""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 从热力图提取 default 数据集数据
    models = ['Qwen3-0.6B', 'Qwen3-4B', 'Qwen3-14B', 'GPT-OSS-20B']
    
    pair = [77.3, 55.5, 58.5, 0.2]
    pair_w_sess = [85.5, 79.0, 77.0, 12.9]
    autodan = [88.0, 86.8, 80.9, 12.5]
    autodan_w_sess = [100.0, 100.0, 99.7, 30.0]
    
    x = np.arange(len(models))
    width = 0.22

    # 绘制分组柱状图（顺序调整：baseline在前，SESS增强在后）
    bars1 = ax.bar(x - width * 1.5, pair, width, label='PAIR',
                   color=COLORS['lightest'], edgecolor='#CBD5E1', linewidth=1)
    bars2 = ax.bar(x - width * 0.5, autodan, width, label='AutoDAN',
                   color=COLORS['lighter'], edgecolor='#CBD5E1', linewidth=1)
    bars3 = ax.bar(x + width * 0.5, pair_w_sess, width, label='PAIR+SESS',
                   color=COLORS['secondary'], edgecolor='#CBD5E1', linewidth=1)
    bars4 = ax.bar(x + width * 1.5, autodan_w_sess, width, label='DAN+SESS',
                   color=COLORS['primary'], edgecolor='#CBD5E1', linewidth=1)
    
    # 添加数值标签
    for bar, val in zip(bars1, pair):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=8, fontweight='bold', color=COLORS['dark'])

    for bar, val in zip(bars2, autodan):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=8, fontweight='bold', color=COLORS['dark'])

    for bar, val in zip(bars3, pair_w_sess):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=8, fontweight='bold', color=COLORS['dark'])

    for bar, val in zip(bars4, autodan_w_sess):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=8, fontweight='bold', color=COLORS['dark'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Main Experiment Results (Default Dataset)', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 115)
    
    add_y_grid(ax)
    set_grid_off(ax)
    
    # 图例
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    save_figure(fig, 'main_experiment_results_simple.png')


# =============================================================================
# 主函数
# =============================================================================

if __name__ == '__main__':
    print("生成主实验结果图...")
    main_experiment_results()
    main_experiment_results_simple()
    print("完成！")
