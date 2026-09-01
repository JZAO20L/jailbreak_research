#!/usr/bin/env python3
"""
Updated SESS_03 Transfer Figures based on TRANSFER_RESULTS.md
- Left: Cross-Model Transfer comparing Qwen3-4B vs gpt-oss-20b
- Right: Cross-Dataset Transfer with no_rewrite and proper ordering
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

COLORS = {
    'primary': '#2563EB',      # SESS enhanced (darkest)
    'secondary': '#3B82F6',    # AutoDAN base
    'light': '#60A5FA',        # PAIR base
    'lighter': '#93C5FD',      # persona
    'lightest': '#BFDBFE',     # deepinception, no_rewrite
    'baseline': '#9CA3AF',     # gray baseline
    'accent': '#10B981',       # green for improvements
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
    """
    Figure SESS_03a: Cross-Model Transfer
    Comparing Qwen3-4B (same-family) vs gpt-oss-20b (cross-family)
    Shows SESS improvement over baselines on BOTH target models
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Methods ordered: baselines first, then SESS enhanced
    methods = ['no_rewrite', 'deepinception', 'persona', 
               'PAIR', 'PAIR_w_SESS', 'AutoDAN', 'AutoDAN_w_SESS']
    
    # Qwen3-4B (same-family) results - average across datasets
    qwen3_4b = [28.2, 49.3, 32.8, 72.3, 86.7, 78.7, 89.2]
    
    # gpt-oss-20b (cross-family) results - average across datasets
    gpt_oss = [0.14, 0.28, 0.7, 0.08, 11.4, 6.0, 32.5]

    x = np.arange(len(methods))
    width = 0.35

    # Colors: SESS enhanced in primary blue, baselines in lighter colors
    bar_colors = [
        COLORS['lightest'],   # no_rewrite
        COLORS['lightest'],   # deepinception
        COLORS['lighter'],    # persona
        COLORS['light'],      # PAIR base
        COLORS['primary'],    # PAIR_w_SESS
        COLORS['secondary'],  # AutoDAN base
        COLORS['primary'],    # AutoDAN_w_SESS
    ]

    # Grouped bar chart: Qwen3-4B on left, gpt-oss on right for each method
    bars1 = ax.bar(x - width/2, qwen3_4b, width, label='Qwen3-4B (Same-Family)',
                   color=bar_colors, edgecolor='#CBD5E1', linewidth=1)
    bars2 = ax.bar(x + width/2, gpt_oss, width, label='gpt-oss-20b (Cross-Family)',
                   color=bar_colors, edgecolor='#CBD5E1', linewidth=1, alpha=0.7)

    # Value annotations for key methods
    for i, (q, g, m) in enumerate(zip(qwen3_4b, gpt_oss, methods)):
        if 'SESS' in m:
            # Show values for SESS enhanced methods
            ax.text(x[i] - width/2, q + 1.5, f'{q:.1f}%', ha='center', 
                    fontsize=8, fontweight='bold', color=COLORS['dark'])
            ax.text(x[i] + width/2, g + 1.5, f'{g:.1f}%', ha='center',
                    fontsize=8, fontweight='bold', color=COLORS['dark'])

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8, rotation=25, ha='right')
    ax.set_ylabel('Attack Success Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Cross-Model Transfer: SESS Enhancement on Different Target Models',
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_ylim(0, 100)
    add_y_grid(ax)
    set_grid_off(ax)

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['primary'], label='SESS Enhanced (Ours)'),
        mpatches.Patch(color=COLORS['secondary'], label='AutoDAN Base'),
        mpatches.Patch(color=COLORS['light'], label='PAIR Base'),
        mpatches.Patch(color=COLORS['lighter'], label='persona'),
        mpatches.Patch(color=COLORS['lightest'], label='Other Baselines'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    # Add annotation showing SESS works on both models
    ax.annotate('SESS improves\non both models', xy=(4, 50), fontsize=9,
                color=COLORS['accent'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['accent']))

    plt.tight_layout()
    save_figure(fig, 'SESS_03a_cross_model_transfer.png')


def sess_03_cross_dataset():
    """
    Figure SESS_03b: Cross-Dataset Transfer
    Includes no_rewrite baseline and proper ordering (deepinception/persona first)
    Shows results on Qwen3-4B across 5 datasets
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    datasets = ['default', 'advbench', 'harmbench\ncontextual', 
                'harmbench\nstandard', 'jailbreakBench']
    
    # Methods ordered: simple baselines first, then SESS enhanced
    # no_rewrite, deepinception, persona, PAIR, PAIR_w_SESS, AutoDAN, AutoDAN_w_SESS
    no_rewrite = [30.3, 3.27, 79.0, 21.5, 7.0]
    deepinception = [40.4, 40.96, 80.0, 47.5, 38.0]
    persona = [32.3, 16.73, 67.0, 36.0, 20.0]
    pair = [55.5, 77.31, 81.0, 75.5, 72.0]
    pair_w_sess = [79.0, 80.96, 98.0, 91.5, 88.0]
    autodan = [86.8, 75.38, 87.0, 75.5, 80.0]
    autodan_w_sess = [89.2, 84.0, 96.5, 93.0, 86.0]

    x = np.arange(len(datasets))
    width = 0.12

    bars1 = ax.bar(x - width * 3, no_rewrite, width, label='no_rewrite',
                   color=COLORS['lightest'], edgecolor='#CBD5E1', linewidth=1)
    bars2 = ax.bar(x - width * 2, deepinception, width, label='deepinception',
                   color=COLORS['lightest'], edgecolor='#CBD5E1', linewidth=1)
    bars3 = ax.bar(x - width * 1, persona, width, label='persona',
                   color=COLORS['lighter'], edgecolor='#CBD5E1', linewidth=1)
    bars4 = ax.bar(x, pair, width, label='PAIR',
                   color=COLORS['light'], edgecolor='#CBD5E1', linewidth=1)
    bars5 = ax.bar(x + width * 1, pair_w_sess, width, label='PAIR_w_SESS',
                   color=COLORS['primary'], edgecolor='#CBD5E1', linewidth=1)
    bars6 = ax.bar(x + width * 2, autodan, width, label='AutoDAN',
                   color=COLORS['secondary'], edgecolor='#CBD5E1', linewidth=1)
    bars7 = ax.bar(x + width * 3, autodan_w_sess, width, label='AutoDAN_w_SESS',
                   color=COLORS['primary'], edgecolor='#CBD5E1', linewidth=1)

    # Value annotations for SESS enhanced methods only
    for bar, val in zip(bars5, pair_w_sess):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.0f}%', ha='center', fontsize=7, fontweight='bold', color=COLORS['dark'])

    for bar, val in zip(bars7, autodan_w_sess):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val:.0f}%', ha='center', fontsize=7, fontweight='bold', color=COLORS['dark'])

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=8)
    ax.set_ylabel('Attack Success Rate (%) on Qwen3-4B', fontsize=11, fontweight='bold')
    ax.set_title('Cross-Dataset Transfer: SESS Generalization Across Datasets',
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_ylim(0, 108)
    add_y_grid(ax)
    set_grid_off(ax)
    ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.0, 1.12), framealpha=0.9)

    plt.tight_layout()
    save_figure(fig, 'SESS_03b_cross_dataset_transfer.png')


def main():
    print("=" * 60)
    print("SESS Transfer Figures - Updated based on TRANSFER_RESULTS.md")
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
