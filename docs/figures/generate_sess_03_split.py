#!/usr/bin/env python3
"""
Updated SESS_03 Transfer Figures - Split into two separate plots
- Left: Cross-Model Transfer with all baselines on Qwen3-4B
- Right: Cross-Dataset Transfer with all methods
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11

COLORS = {
    'primary': '#2563EB',
    'secondary': '#3B82F6',
    'light': '#60A5FA',
    'lighter': '#93C5FD',
    'lightest': '#BFDBFE',
    'baseline': '#9CA3AF',
    'accent': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'dark': '#1E3A5F',
}

OUTPUT_DIR = '/home/tiger/jailbreak_research/docs/figures/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_figure(fig, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {filename}")


def set_grid_off(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_y_grid(ax, alpha=0.15):
    ax.grid(True, axis='y', alpha=alpha, linestyle='-', linewidth=0.5)


def sess_03_cross_model():
    """Figure SESS_03a: Cross-Model Transfer on Qwen3-4B with all baselines"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # All methods including baselines
    methods = ['PAIR', 'PAIR_w_SESS', 'AutoDAN', 'AutoDAN_w_SESS', 
               'DeepInception', 'Multilingual', 'Genetic']
    # Qwen3-4B target model results
    qwen3_4b_asr = [65.3, 86.7, 72.5, 89.2, 37.3, 2.1, 47.9]

    x = np.arange(len(methods))
    
    # Color scheme: SESS enhanced in dark blue, base methods in medium blue, other baselines in light gray-blue
    bar_colors = [
        COLORS['light'],      # PAIR base
        COLORS['primary'],    # PAIR_w_SESS enhanced
        COLORS['secondary'],  # AutoDAN base
        COLORS['primary'],    # AutoDAN_w_SESS enhanced
        COLORS['lightest'],   # DeepInception
        COLORS['lightest'],   # Multilingual
        COLORS['lightest'],   # Genetic
    ]

    bars = ax.bar(x, qwen3_4b_asr, color=bar_colors, edgecolor='#CBD5E1', linewidth=1.2, width=0.6)

    # Value annotations
    for bar, val in zip(bars, qwen3_4b_asr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val}%', ha='center', fontsize=9, fontweight='bold', color=COLORS['dark'])

    # Add improvement arrows for SESS methods
    # PAIR -> PAIR_w_SESS
    ax.annotate('', xy=(1, 86.7), xytext=(0, 65.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))
    ax.text(0.5, 77, f'+21.4%', fontsize=9, color=COLORS['accent'], fontweight='bold', ha='center')
    
    # AutoDAN -> AutoDAN_w_SESS
    ax.annotate('', xy=(3, 89.2), xytext=(2, 72.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))
    ax.text(2.5, 81, f'+16.7%', fontsize=9, color=COLORS['accent'], fontweight='bold', ha='center')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8, rotation=20, ha='right')
    ax.set_ylabel('Attack Success Rate (%) on Qwen3-4B', fontsize=11, fontweight='bold')
    ax.set_title('Cross-Model Transfer: SESS Enhancement on Qwen3-4B', fontsize=12, fontweight='bold', pad=15)
    ax.set_ylim(0, 100)
    add_y_grid(ax)
    set_grid_off(ax)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['primary'], label='SESS Enhanced (Ours)'),
        mpatches.Patch(color=COLORS['secondary'], label='AutoDAN Base'),
        mpatches.Patch(color=COLORS['light'], label='PAIR Base'),
        mpatches.Patch(color=COLORS['lightest'], label='Other Baselines'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    save_figure(fig, 'SESS_03a_cross_model_transfer.png')


def sess_03_cross_dataset():
    """Figure SESS_03b: Cross-Dataset Transfer with all methods"""
    fig, ax = plt.subplots(figsize=(11, 6))

    datasets = ['AdvBench', 'Malicious\nInstruct', 'HarmBench', 'XSTest', 'SafeBench']
    
    # All methods including DeepInception baseline
    pair = [78.2, 75.8, 72.1, 65.3, 69.5]
    pair_w_sess = [89.5, 86.2, 83.4, 74.8, 80.1]
    autodan = [82.1, 78.5, 75.3, 68.9, 71.2]
    autodan_w_sess = [94.5, 91.2, 88.7, 76.3, 82.4]
    deepinception = [37.3, 35.8, 34.2, 31.5, 36.1]

    x = np.arange(len(datasets))
    width = 0.18

    bars1 = ax.bar(x - width * 2, pair, width, label='PAIR',
                   color=COLORS['lightest'], edgecolor='#CBD5E1', linewidth=1)
    bars2 = ax.bar(x - width, pair_w_sess, width, label='PAIR_w_SESS',
                   color=COLORS['light'], edgecolor='#CBD5E1', linewidth=1)
    bars3 = ax.bar(x, autodan, width, label='AutoDAN',
                   color=COLORS['secondary'], edgecolor='#CBD5E1', linewidth=1)
    bars4 = ax.bar(x + width, autodan_w_sess, width, label='AutoDAN_w_SESS',
                   color=COLORS['primary'], edgecolor='#CBD5E1', linewidth=1)
    bars5 = ax.bar(x + width * 2, deepinception, width, label='DeepInception',
                   color=COLORS['lightest'], edgecolor='#CBD5E1', linewidth=1)

    # Value annotations for SESS enhanced methods
    for bar, val in zip(bars2, pair_w_sess):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=7, fontweight='bold', color=COLORS['dark'])

    for bar, val in zip(bars4, autodan_w_sess):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=7, fontweight='bold', color=COLORS['dark'])

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=8)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Cross-Dataset Transfer: SESS Generalization', fontsize=12, fontweight='bold', pad=15)
    ax.set_ylim(0, 108)
    add_y_grid(ax)
    set_grid_off(ax)
    ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.0, 1.15), framealpha=0.9)

    plt.tight_layout()
    save_figure(fig, 'SESS_03b_cross_dataset_transfer.png')


def main():
    print("=" * 60)
    print("SESS Transfer Figures - Split Version")
    print("=" * 60)

    print("\nGenerating Cross-Model Transfer figure...")
    sess_03_cross_model()

    print("\nGenerating Cross-Dataset Transfer figure...")
    sess_03_cross_dataset()

    print("\n" + "=" * 60)
    print(f"Figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
