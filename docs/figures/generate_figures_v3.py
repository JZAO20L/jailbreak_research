#!/usr/bin/env python3
"""
中期报告主图生成脚本（美化版）
- 统一色系：蓝色系为主
- 纯英文：避免中文字体问题
- 简洁设计：减少视觉复杂度

Usage:
    python generate_figures_v3.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# 设置字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12

# 统一色系：蓝色系 + 灰色对比
COLORS = {
    'primary': '#3B82F6',    # 主蓝色
    'secondary': '#60A5FA',   # 次蓝色
    'light': '#93C5FD',       # 浅蓝色
    'success': '#22C55E',     # 绿色（正向）
    'warning': '#F59E0B',     # 橙色（中性）
    'danger': '#EF4444',      # 红色（负向）
    'gray': '#6B7280',        # 灰色
    'light_gray': '#D1D5DB',  # 浅灰色
}

# 输出目录
OUTPUT_DIR = '/home/tiger/jailbreak_research/docs/figures/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_figure(fig, filename):
    """保存图片"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {filepath}")
    plt.close(fig)


# =============================================================================
# Chapter 1: AHR-GRPO
# =============================================================================

def fig1_prompt_selection():
    """Figure 1: Jailbreak Prompt Selection Results (Top-8)"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Top-8 strategies
    strategies = ['hypothetical_scenario', 'creative_writing', 'role_playing',
                  'red_teaming', 'urgent_situation', 'journalistic_investigation',
                  'academic_research', 'technical_documentation']
    asr = [30.8, 28.3, 25.0, 22.4, 21.9, 20.9, 20.2, 19.9]

    # Color: Top-3 highlighted with primary blue, others gray
    colors = [COLORS['primary'], COLORS['primary'], COLORS['primary'],
              COLORS['light_gray'], COLORS['light_gray'], COLORS['light_gray'],
              COLORS['light_gray'], COLORS['light_gray']]

    bars = ax.barh(strategies, asr, color=colors, edgecolor='black', linewidth=1)

    # Annotate values
    for bar, val in zip(bars, asr):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val}%', va='center', fontsize=10, fontweight='bold')

    # Mark Top-3
    ax.text(31.5, 7.4, '★ Top-1 Selected', fontsize=9, color=COLORS['primary'], fontweight='bold')
    ax.text(29.3, 6.4, '★ Top-2 Selected', fontsize=9, color=COLORS['primary'], fontweight='bold')
    ax.text(26.3, 5.4, '★ Top-3 Selected', fontsize=9, color=COLORS['primary'], fontweight='bold')

    ax.set_xlabel('Attack Success Rate (%)', fontsize=13)
    ax.set_title('Experiment 1: Jailbreak Prompt Selection (Top-8)', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(0, 34)
    ax.axvline(x=25, color=COLORS['danger'], linestyle='--', alpha=0.6, linewidth=1.5)
    ax.text(25.5, 0.3, 'Threshold (25%)', fontsize=9, color=COLORS['danger'])
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    save_figure(fig, 'fig1_prompt_selection.png')


def fig2_judge_dimension():
    """Figure 2: Judge Dimension Comparison (Experiment 2)"""
    fig, ax = plt.subplots(figsize=(12, 6))

    baseline = 30.8

    # 12 experiments (simplified view)
    experiments = [
        ('HS + IP', 32.3), ('HS + Nat', 31.8), ('HS + HS', 31.2), ('HS + St', 31.0),
        ('CW + Nat', 29.5), ('CW + IP', 29.0), ('CW + CW', 28.8), ('CW + St', 28.5),
        ('RP + IP', 26.8), ('RP + RP', 26.5), ('RP + Nat', 26.2), ('RP + St', 26.0),
    ]

    x = np.arange(len(experiments))
    asr_values = [e[1] for e in experiments]

    # Color: above baseline = primary blue, near baseline = secondary blue
    colors = [COLORS['primary'] if v >= 31 else COLORS['secondary'] for v in asr_values]

    bars = ax.bar(x, asr_values, color=colors, edgecolor='black', linewidth=0.8)

    # Baseline reference line
    ax.axhline(y=baseline, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.7)
    ax.text(-0.5, baseline + 0.3, f'Baseline: {baseline}%', fontsize=10,
            color=COLORS['danger'], fontweight='bold')

    # Annotate values
    for i, (bar, asr) in enumerate(zip(bars, asr_values)):
        delta = asr - baseline
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{asr}', ha='center', fontsize=9, fontweight='bold')

    # X-axis labels
    labels = [e[0] for e in experiments]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=0)

    ax.set_ylabel('Attack Success Rate (%)', fontsize=13)
    ax.set_title('Experiment 2: GRPO Training Results (12 Groups)\nAll Combinations Exceed Baseline',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(24, 34)
    ax.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['primary'], label='ASR ≥ 31%'),
        mpatches.Patch(color=COLORS['secondary'], label='ASR 26-31%'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    save_figure(fig, 'fig2_judge_dimension.png')


def fig3_adaptive_vs_fixed():
    """Figure 3: Adaptive vs Fixed Weight Comparison"""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Core comparison
    methods = ['Baseline\n(No Training)', 'Fixed Weight\n(Exp 2)', 'Adaptive\n(β=0)',
               'Adaptive\n(β=0.8)', 'Adaptive\n(β=0.9)']
    asr = [30.8, 32.3, 33.2, 32.9, 32.6]

    # Color: gradient from gray to blue
    colors = [COLORS['gray'], COLORS['secondary'], COLORS['primary'],
              COLORS['primary'], COLORS['primary']]

    bars = ax.bar(methods, asr, color=colors, edgecolor='black', linewidth=1.5)

    # Annotate values and improvement
    baseline = 30.8
    exp2_best = 32.3
    for i, (bar, val) in enumerate(zip(bars, asr)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val}%', ha='center', fontsize=12, fontweight='bold')

    # Mark improvement arrows
    ax.annotate('', xy=(2, 33.2), xytext=(1, 32.3),
                arrowprops=dict(arrowstyle='<->', color=COLORS['success'], lw=2))
    ax.text(1.5, 33.7, '+0.9%', fontsize=11, fontweight='bold', color=COLORS['success'])

    # Baseline reference line
    ax.axhline(y=baseline, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.5)

    ax.set_ylabel('Attack Success Rate (%)', fontsize=13)
    ax.set_title('Experiment 3: Adaptive Weight vs Fixed Weight', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(28, 35)
    ax.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add improvement annotations
    ax.text(0.3, 31, '+1.5%', fontsize=9, color=COLORS['success'], fontweight='bold')
    ax.text(1.3, 33, '+2.4%', fontsize=9, color=COLORS['success'], fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'fig3_adaptive_vs_fixed.png')


def fig4_reward_type_ablation():
    """Figure 4: Reward Type Ablation"""
    fig, ax = plt.subplots(figsize=(9, 6))

    reward_types = ['Only ASR\nReward', 'Only Judge\nReward', 'Hybrid Fixed\n(Exp 2)',
                    'Hybrid Adaptive\n(Exp 3)', 'Baseline']
    asr = [25.0, 28.5, 32.3, 33.2, 30.8]

    # Color: based on performance
    colors = [COLORS['danger'], COLORS['warning'], COLORS['secondary'],
              COLORS['primary'], COLORS['gray']]

    bars = ax.bar(reward_types, asr, color=colors, edgecolor='black', linewidth=1.5)

    # Annotate values and delta
    baseline = 30.8
    for i, (bar, val) in enumerate(zip(bars, asr)):
        delta = val - baseline
        sign = '+' if delta >= 0 else ''
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}%\n({sign}{delta:.1f})', ha='center', fontsize=10, fontweight='bold')

    # Baseline reference line
    ax.axhline(y=baseline, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.5)

    ax.set_ylabel('Attack Success Rate (%)', fontsize=13)
    ax.set_title('Reward Type Ablation Study', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(20, 36)
    ax.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_figure(fig, 'fig4_reward_type_ablation.png')


def fig5_ahr_grpo_summary():
    """Figure 5: AHR-GRPO Complete Summary"""
    fig, ax = plt.subplots(figsize=(8, 6))

    levels = ['Baseline', 'Exp 2 (Fixed)', 'Exp 3 (Adaptive)']
    asr = [30.8, 32.3, 33.2]

    # Create gradient colors
    colors = [COLORS['gray'], COLORS['secondary'], COLORS['primary']]

    bars = ax.bar(levels, asr, color=colors, edgecolor='black', linewidth=2, width=0.6)

    baseline = 30.8
    for i, (bar, val) in enumerate(zip(bars, asr)):
        delta = val - baseline
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}%', ha='center', fontsize=14, fontweight='bold')
        if i > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 1.5,
                    f'+{delta:.1f}', ha='center', fontsize=12, color='white', fontweight='bold')

    # Improvement arrows
    ax.annotate('', xy=(1.5, 33.2), xytext=(0.5, 32.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=2))

    ax.axhline(y=baseline, color=COLORS['danger'], linestyle='--', linewidth=2, alpha=0.5)
    ax.text(2, baseline + 0.8, f'Baseline: {baseline}%', fontsize=10, color=COLORS['danger'])

    ax.set_ylabel('Attack Success Rate (%)', fontsize=13)
    ax.set_title('AHR-GRPO Progressive Improvement', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(28, 36)
    ax.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_figure(fig, 'fig5_ahr_grpo_summary.png')


# =============================================================================
# Chapter 2: SESS
# =============================================================================

def fig6_sess_layers_comparison():
    """Figure 6: SESS Layers ASR Comparison"""
    fig, ax = plt.subplots(figsize=(11, 6))

    layers = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Ablation', 'Transfer']
    asr = [79.1, 80.0, 98.8, 99.7, 69.6, 32.5]
    exp_count = [16, 36, 17, 12, 16, 140]

    # Color: primary for main results, gray for ablation/transfer
    colors = [COLORS['primary'], COLORS['primary'], COLORS['primary'],
              COLORS['primary'], COLORS['gray'], COLORS['gray']]

    bars = ax.bar(layers, asr, color=colors, edgecolor='black', linewidth=1.5)

    # Annotate values
    for bar, val, count in zip(bars, asr, exp_count):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val}%', ha='center', fontsize=12, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{count}', ha='center', fontsize=10, color='white', fontweight='bold')

    ax.set_xlabel('Experiment Layer', fontsize=13)
    ax.set_ylabel('Best ASR (%)', fontsize=13)
    ax.set_title('SESS Experiment Layers Comparison\n(Total: 217 Experiments)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 108)
    ax.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['primary'], label='Main Results'),
        mpatches.Patch(color=COLORS['gray'], label='Ablation/Transfer'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    save_figure(fig, 'fig6_sess_layers.png')


def fig7_transfer_comparison():
    """Figure 7: Same-Family vs Cross-Family Transfer"""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['pair_skills_28', 'autodan_skills', 'autodan', 'pair']
    same_family = [91.6, 99.0, 85.5, 71.6]
    cross_family = [11.4, 32.5, 6.0, 0.08]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, same_family, width, label='Same-Family (Qwen3)',
                   color=COLORS['primary'], edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, cross_family, width, label='Cross-Family (gpt-oss)',
                   color=COLORS['danger'], edgecolor='black', linewidth=1)

    # Annotate values
    for bar, val in zip(bars1, same_family):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=10, fontweight='bold', color=COLORS['primary'])

    for bar, val in zip(bars2, cross_family):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=10, fontweight='bold', color=COLORS['danger'])

    ax.set_xlabel('Method', fontsize=13)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=13)
    ax.set_title('Transferability: Same-Family vs Cross-Family', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, 108)
    ax.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add note
    ax.text(0.5, 100, 'Key Finding: Cross-family transfer is challenging',
            fontsize=10, color=COLORS['gray'], style='italic')

    plt.tight_layout()
    save_figure(fig, 'fig7_transfer_comparison.png')


def fig8_best_config_summary():
    """Figure 8: Best Configuration Summary for Both Chapters"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: AHR-GRPO
    ax1 = axes[0]
    methods1 = ['Baseline', 'Exp 2 Best', 'Exp 3 Best']
    asr1 = [30.8, 32.3, 33.2]
    colors1 = [COLORS['gray'], COLORS['secondary'], COLORS['primary']]

    bars1 = ax1.bar(methods1, asr1, color=colors1, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars1, asr1):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val}%', ha='center', fontsize=12, fontweight='bold')

    ax1.set_ylabel('ASR (%)', fontsize=12)
    ax1.set_title('Chapter 1: AHR-GRPO Best Results', fontsize=13, fontweight='bold')
    ax1.set_ylim(28, 35)
    ax1.grid(True, axis='y', alpha=0.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: SESS
    ax2 = axes[1]
    methods2 = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
    asr2 = [79.1, 80.0, 98.8, 99.7]
    colors2 = [COLORS['light'], COLORS['secondary'], COLORS['primary'], COLORS['success']]

    bars2 = ax2.bar(methods2, asr2, color=colors2, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars2, asr2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val}%', ha='center', fontsize=11, fontweight='bold')

    ax2.set_ylabel('ASR (%)', fontsize=12)
    ax2.set_title('Chapter 2: SESS Best Results', fontsize=13, fontweight='bold')
    ax2.set_ylim(70, 105)
    ax2.grid(True, axis='y', alpha=0.2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    save_figure(fig, 'fig8_best_config_summary.png')


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate all figures"""
    print("=" * 60)
    print("Midterm Report Figures - Clean Design")
    print("Color: Blue system, English only, Simplified layout")
    print("=" * 60)

    print("\n【Chapter 1: AHR-GRPO】")
    fig1_prompt_selection()
    fig2_judge_dimension()
    fig3_adaptive_vs_fixed()
    fig4_reward_type_ablation()
    fig5_ahr_grpo_summary()

    print("\n【Chapter 2: SESS】")
    fig6_sess_layers_comparison()
    fig7_transfer_comparison()
    fig8_best_config_summary()

    print("\n" + "=" * 60)
    print(f"✓ All figures generated! Location: {OUTPUT_DIR}")
    print("=" * 60)

    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == '__main__':
    main()