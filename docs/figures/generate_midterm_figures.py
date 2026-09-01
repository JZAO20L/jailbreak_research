#!/usr/bin/env python3
"""
Midterm Report - Unified Figure Generation Script
- Consistent blue color scheme across all figures
- English-only labels and annotations
- Clean, publication-ready design
- All data sourced from figure_data_tables.md

Usage:
    python generate_midterm_figures.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# =============================================================================
# Global Style Configuration
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

OUTPUT_DIR = '/home/tiger/jailbreak_research/docs/figures/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_figure(fig, filename):
    """Save figure with consistent settings"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {filename}")


def set_grid_off(ax):
    """Remove top and right spines"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_y_grid(ax, alpha=0.15):
    """Add subtle y-axis grid"""
    ax.grid(True, axis='y', alpha=alpha, linestyle='-', linewidth=0.5)


# =============================================================================
# Chapter 1: AHR-GRPO Figures
# =============================================================================

def fig1_prompt_selection():
    """Figure 1: Jailbreak Prompt Selection - Top 8 Strategies (Horizontal Bar)"""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    strategies = [
        'hypothetical_scenario', 'creative_writing', 'role_playing',
        'red_teaming', 'urgent_situation', 'journalistic_investigation',
        'academic_research', 'technical_documentation'
    ]
    asr = [30.8, 28.3, 25.0, 22.4, 21.9, 20.9, 20.2, 19.9]

    # Top-3 in primary blue, rest in light gray
    colors = [COLORS['primary'], COLORS['primary'], COLORS['primary'],
              COLORS['lightest'], COLORS['lightest'], COLORS['lightest'],
              COLORS['lightest'], COLORS['lightest']]

    y_pos = np.arange(len(strategies))
    bars = ax.barh(y_pos, asr, color=colors, edgecolor='#CBD5E1', linewidth=1.2, height=0.6)

    # Value annotations
    for bar, val in zip(bars, asr):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f'{val}%', va='center', fontsize=10, fontweight='bold', color=COLORS['dark'])

    # Top-3 badges - positioned next to each selected bar (after invert_yaxis, y=0 is at top)
    ax.text(31.5, 0, '★ Selected', fontsize=9, color=COLORS['primary'], fontweight='bold', va='center')
    ax.text(29.0, 1, '★ Selected', fontsize=9, color=COLORS['primary'], fontweight='bold', va='center')
    ax.text(26.0, 2, '★ Selected', fontsize=9, color=COLORS['primary'], fontweight='bold', va='center')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(strategies, fontsize=10)
    ax.set_xlabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Attack Prompt Selection', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 35)
    ax.invert_yaxis()
    add_y_grid(ax)
    set_grid_off(ax)

    # Legend - positioned at bottom right to avoid blocking bars
    legend_elements = [
        mpatches.Patch(color=COLORS['primary'], label='Selected (ASR ≥ 25%)'),
        mpatches.Patch(color=COLORS['lightest'], label='Reference (ASR < 25%)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

    save_figure(fig, 'AHR_GRPO_01_attack_prompt_selection.png')


def fig2_judge_dimension():
    """Figure 2: Judge Dimension GRPO Results - 12 Experiment Groups"""
    fig, ax = plt.subplots(figsize=(12, 5.5))

    baseline = 30.8

    experiments = [
        ('HS+IP', 32.3), ('HS+Nat', 31.8), ('HS+HS', 31.2), ('HS+St', 31.0),
        ('CW+Nat', 29.5), ('CW+IP', 29.0), ('CW+CW', 28.8), ('CW+St', 28.5),
        ('RP+IP', 26.8), ('RP+RP', 26.5), ('RP+Nat', 26.2), ('RP+St', 26.0),
    ]

    x = np.arange(len(experiments))
    asr_values = [e[1] for e in experiments]

    # Color by performance level
    colors = []
    for v in asr_values:
        if v >= 31:
            colors.append(COLORS['primary'])
        elif v >= 28:
            colors.append(COLORS['secondary'])
        else:
            colors.append(COLORS['light'])

    bars = ax.bar(x, asr_values, color=colors, edgecolor='#CBD5E1', linewidth=1, width=0.7)

    # Baseline reference - black dashed line with label in middle-right
    ax.axhline(y=baseline, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(7.5, baseline + 0.25, f'Baseline: {baseline}%', fontsize=9,
            color='black', fontweight='bold', ha='center')

    # Value annotations
    for bar, val in zip(bars, asr_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{val}', ha='center', fontsize=9, fontweight='bold', color=COLORS['dark'])

    ax.set_xticks(x)
    ax.set_xticklabels([e[0] for e in experiments], fontsize=9)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Experiment 2: GRPO Training with Different Judge Dimensions',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(24, 34)
    add_y_grid(ax)
    set_grid_off(ax)

    legend_elements = [
        mpatches.Patch(color=COLORS['primary'], label='ASR ≥ 31%'),
        mpatches.Patch(color=COLORS['secondary'], label='ASR 28-31%'),
        mpatches.Patch(color=COLORS['light'], label='ASR < 28%'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    save_figure(fig, 'AHR_GRPO_02_judge_prompt_selection.png')


def fig3_adaptive_vs_fixed():
    """Figure 3: Adaptive Weight vs Fixed Weight Comparison"""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    methods = ['Baseline\n(No Training)', 'Fixed Weight\n(Exp 2 Best)', 'Adaptive\n(β=0.0)',
               'Adaptive\n(β=0.8)', 'Adaptive\n(β=0.9)']
    asr = [30.8, 32.3, 33.2, 32.9, 32.6]

    # Gradient from gray to deep blue
    colors = [COLORS['baseline'], COLORS['light'], COLORS['primary'],
              COLORS['secondary'], COLORS['lighter']]

    bars = ax.bar(methods, asr, color=colors, edgecolor='#CBD5E1', linewidth=1.2, width=0.6)

    baseline = 30.8

    # Value annotations
    for bar, val in zip(bars, asr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{val}%', ha='center', fontsize=11, fontweight='bold', color=COLORS['dark'])

    # Baseline reference - black dashed line
    ax.axhline(y=baseline, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Experiment 3: Adaptive Weight vs Fixed Weight',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(28, 36)
    add_y_grid(ax)
    set_grid_off(ax)

    plt.tight_layout()
    save_figure(fig, 'AHR_GRPO_03_adaptive_weight.png')


def fig4_reward_type_ablation():
    """Figure 4: Reward Type Ablation Study"""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    reward_types = ['Only ASR\nReward', 'Only Judge\nReward', 'Hybrid\nFixed (1:1)',
                    'Hybrid\nAdaptive', 'Baseline']
    asr = [25.0, 28.5, 32.3, 33.2, 30.8]

    # All blue system colors - gradient based on performance
    colors = [COLORS['lightest'], COLORS['light'], COLORS['secondary'],
              COLORS['primary'], COLORS['baseline']]

    bars = ax.bar(reward_types, asr, color=colors, edgecolor='#CBD5E1', linewidth=1.2, width=0.6)

    baseline = 30.8
    for bar, val in zip(bars, asr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val}%', ha='center', fontsize=11, fontweight='bold', color=COLORS['dark'])

    # Baseline reference - black dashed line
    ax.axhline(y=baseline, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Reward Type Ablation Study', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(20, 37)
    add_y_grid(ax)
    set_grid_off(ax)

    plt.tight_layout()
    save_figure(fig, 'AHR_GRPO_04_reward_ablation.png')


def fig5_ahr_grpo_summary():
    """Figure 5: AHR-GRPO Progressive Improvement Summary"""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    levels = ['Baseline', 'Experiment 2\n(Fixed Weight)', 'Experiment 3\n(Adaptive)']
    asr = [30.8, 32.3, 33.2]

    # Progressive blue gradient
    colors = [COLORS['baseline'], COLORS['lighter'], COLORS['primary']]

    bars = ax.bar(levels, asr, color=colors, edgecolor='#CBD5E1', linewidth=1.5, width=0.5)

    baseline = 30.8
    for i, (bar, val) in enumerate(zip(bars, asr)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val}%', ha='center', fontsize=13, fontweight='bold', color=COLORS['dark'])

    # Baseline reference - black dashed line
    ax.axhline(y=baseline, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('AHR-GRPO: Progressive Improvement Across Experiments',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(28, 36)
    add_y_grid(ax)
    set_grid_off(ax)

    plt.tight_layout()
    save_figure(fig, 'AHR_GRPO_05_summary.png')


# =============================================================================
# Chapter 2: SESS Figures
# =============================================================================

def fig6_sess_main_results():
    """Figure 6: SESS Main Results - Strategy and Data Distribution Effects"""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Method & Data Strategy (Layer 1-4 + Ablation best configs)
    configs = [
        'Layer 1\n(Method)',
        'Layer 2\n(Data)',
        'Layer 3\n(DAN)',
        'Layer 4\n(DAN Data)',
        'No Evolution'
    ]
    asr = [79.1, 80.6, 98.8, 99.7, 75.6]
    exp_count = [16, 36, 17, 12, 16]

    # Color gradient showing improvement
    colors = [COLORS['lightest'], COLORS['light'], COLORS['secondary'],
              COLORS['primary'], COLORS['baseline']]

    bars = ax.bar(configs, asr, color=colors, edgecolor='#CBD5E1', linewidth=1.2, width=0.6)

    for bar, val, count in zip(bars, asr, exp_count):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val}%', ha='center', fontsize=10, fontweight='bold', color=COLORS['dark'])
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                 f'n={count}', ha='center', fontsize=9, color='white', fontweight='bold')

    ax.set_ylabel('Best ASR (%)', fontsize=12, fontweight='bold')
    ax.set_title('SESS: Strategy & Data Distribution Effects', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 108)
    add_y_grid(ax)
    set_grid_off(ax)

    plt.tight_layout()
    save_figure(fig, 'SESS_01_sess_main_results.png')


def fig7_transfer_comparison():
    """Figure 7: Same-Family vs Cross-Family Transfer"""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Order: Baselines first (PAIR, AutoDAN), then our methods (PAIR_w_SESS, AutoDAN_w_SESS)
    methods = ['PAIR', 'AutoDAN', 'PAIR_w_SESS', 'AutoDAN_w_SESS']
    same_family = [71.6, 85.5, 91.6, 99.0]
    cross_family = [0.08, 6.0, 11.4, 32.5]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width / 2, same_family, width, label='Same-Family (Qwen3)',
                   color=COLORS['primary'], edgecolor='#CBD5E1', linewidth=1)
    bars2 = ax.bar(x + width / 2, cross_family, width, label='Cross-Family (gpt-oss)',
                   color=COLORS['lightest'], edgecolor='#CBD5E1', linewidth=1)

    # Value annotations - same family
    for bar, val in zip(bars1, same_family):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=9, fontweight='bold', color=COLORS['primary'])

    # Value annotations - cross family
    for bar, val in zip(bars2, cross_family):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=9, fontweight='bold', color=COLORS['dark'])

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Transferability: Same-Family vs Cross-Family Models',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 108)
    add_y_grid(ax)
    set_grid_off(ax)
    # Legend in upper left to avoid blocking bars
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    save_figure(fig, 'SESS_02_transfer_comparison.png')


def fig8_transfer_comprehensive():
    """Figure 8: Comprehensive Transfer Comparison - Cross-Model and Cross-Dataset"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Cross-Model Transfer (Same-Family vs Cross-Family)
    ax1 = axes[0]

    # Order: Baselines first (PAIR, AutoDAN), then our methods (PAIR_w_SESS, AutoDAN_w_SESS)
    methods = ['PAIR', 'AutoDAN', 'PAIR_w_SESS', 'AutoDAN_w_SESS']
    same_family = [71.6, 85.5, 91.6, 99.0]
    cross_family = [0.08, 6.0, 11.4, 32.5]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, same_family, width, label='Same-Family (Qwen3)',
                    color=COLORS['primary'], edgecolor='#CBD5E1', linewidth=1)
    bars2 = ax1.bar(x + width / 2, cross_family, width, label='Cross-Family (gpt-oss)',
                    color=COLORS['lightest'], edgecolor='#CBD5E1', linewidth=1)

    for bar, val in zip(bars1, same_family):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val}%', ha='center', fontsize=9, fontweight='bold', color=COLORS['primary'])

    for bar, val in zip(bars2, cross_family):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val}%', ha='center', fontsize=9, fontweight='bold', color=COLORS['dark'])

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylabel('Attack Success Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Cross-Model Transfer', fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylim(0, 108)
    add_y_grid(ax1)
    set_grid_off(ax1)
    ax1.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, 1.15), framealpha=0.9)

    # Right: Cross-Dataset Transfer with PAIR and PAIR_w_SESS for each dataset
    ax2 = axes[1]

    datasets = ['AdvBench', 'Malicious\nInstruct', 'HarmBench', 'XSTest', 'SafeBench']
    
    # PAIR baseline
    pair = [78.2, 75.8, 72.1, 65.3, 69.5]
    # PAIR with SESS
    pair_w_sess = [89.5, 86.2, 83.4, 74.8, 80.1]
    # AutoDAN baseline
    autodan = [82.1, 78.5, 75.3, 68.9, 71.2]
    # AutoDAN with SESS (our best)
    autodan_w_sess = [94.5, 91.2, 88.7, 76.3, 82.4]

    x = np.arange(len(datasets))
    width = 0.22  # 4 bars per group

    bars1 = ax2.bar(x - width * 1.5, pair, width, label='PAIR',
                    color=COLORS['lightest'], edgecolor='#CBD5E1', linewidth=1)
    bars2 = ax2.bar(x - width * 0.5, pair_w_sess, width, label='PAIR_w_SESS',
                    color=COLORS['light'], edgecolor='#CBD5E1', linewidth=1)
    bars3 = ax2.bar(x + width * 0.5, autodan, width, label='AutoDAN',
                    color=COLORS['secondary'], edgecolor='#CBD5E1', linewidth=1)
    bars4 = ax2.bar(x + width * 1.5, autodan_w_sess, width, label='AutoDAN_w_SESS (Ours)',
                    color=COLORS['primary'], edgecolor='#CBD5E1', linewidth=1)

    # Value annotations - show for PAIR_w_SESS and AutoDAN_w_SESS
    for bar, val in zip(bars1, pair):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f'{val}%', ha='center', fontsize=8, fontweight='bold', color=COLORS['dark'])

    for bar, val in zip(bars2, pair_w_sess):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f'{val}%', ha='center', fontsize=8, fontweight='bold', color=COLORS['dark'])

    for bar, val in zip(bars4, autodan_w_sess):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                 f'{val}%', ha='center', fontsize=8, fontweight='bold', color=COLORS['dark'])

    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=9)
    ax2.set_ylabel('Attack Success Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Cross-Dataset Transfer', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylim(0, 108)
    add_y_grid(ax2)
    set_grid_off(ax2)
    ax2.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.0, 1.18), framealpha=0.9)

    plt.suptitle('SESS: Comprehensive Transferability Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'SESS_03_transfer_comprehensive.png')


# =============================================================================
# Additional Analysis Figures
# =============================================================================

def fig9_judge_dimension_summary():
    """Figure 9: Judge Dimension Type Performance Summary"""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    dimensions = ['idea_preservation\n(Generic)', 'naturalness\n(Generic)',
                  'stealthiness\n(Generic)', 'Strategy-specific\n(Dedicated)']
    avg_delta = [1.2, 1.0, 0.8, 0.6]
    best_delta = [1.8, 1.2, 1.0, 1.0]

    x = np.arange(len(dimensions))
    width = 0.35

    bars1 = ax.bar(x - width / 2, avg_delta, width, label='Average Improvement',
                   color=COLORS['primary'], edgecolor='#CBD5E1', linewidth=1)
    bars2 = ax.bar(x + width / 2, best_delta, width, label='Best Improvement',
                   color=COLORS['light'], edgecolor='#CBD5E1', linewidth=1)

    # Value annotations
    for bar, val in zip(bars1, avg_delta):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'+{val:.1f}', ha='center', fontsize=9, fontweight='bold', color=COLORS['dark'])

    for bar, val in zip(bars2, best_delta):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'+{val:.1f}', ha='center', fontsize=9, fontweight='bold', color=COLORS['dark'])

    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, fontsize=9)
    ax.set_ylabel('Improvement vs Baseline (%)', fontsize=11, fontweight='bold')
    ax.set_title('Judge Dimension Type Performance Comparison',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 2.3)
    add_y_grid(ax)
    set_grid_off(ax)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    save_figure(fig, 'AHR_GRPO_06_judge_dimension.png')


def fig10_same_family_transfer():
    """Figure 10: Same-Family Transfer Results (Qwen Series) - All Methods"""
    fig, ax = plt.subplots(figsize=(11, 5.5))

    models = ['Qwen3-0.6B', 'Qwen3-4B', 'Qwen3-14B']
    
    # PAIR baseline
    pair = [78.5, 72.3, 82.1]
    # PAIR with SESS
    pair_w_sess = [95.1, 86.7, 86.5]
    # AutoDAN baseline
    autodan = [80.2, 78.7, 85.5]
    # AutoDAN with SESS
    autodan_w_sess = [96.8, 89.2, 90.3]

    x = np.arange(len(models))
    width = 0.22  # 4 bars per group

    bars1 = ax.bar(x - width * 1.5, pair, width, label='PAIR',
                   color=COLORS['lightest'], edgecolor='#CBD5E1', linewidth=1)
    bars2 = ax.bar(x - width * 0.5, pair_w_sess, width, label='PAIR_w_SESS',
                   color=COLORS['light'], edgecolor='#CBD5E1', linewidth=1)
    bars3 = ax.bar(x + width * 0.5, autodan, width, label='AutoDAN',
                   color=COLORS['secondary'], edgecolor='#CBD5E1', linewidth=1)
    bars4 = ax.bar(x + width * 1.5, autodan_w_sess, width, label='AutoDAN_w_SESS',
                   color=COLORS['primary'], edgecolor='#CBD5E1', linewidth=1)

    # Value annotations for all methods
    for bar, val in zip(bars1, pair):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=8, fontweight='bold', color=COLORS['dark'])

    for bar, val in zip(bars2, pair_w_sess):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=8, fontweight='bold', color=COLORS['dark'])

    for bar, val in zip(bars3, autodan):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=8, fontweight='bold', color=COLORS['dark'])

    for bar, val in zip(bars4, autodan_w_sess):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=8, fontweight='bold', color=COLORS['dark'])

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Same-Family Transfer: Qwen Series Models',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 108)
    add_y_grid(ax)
    set_grid_off(ax)
    # Legend in upper right, outside the plot area
    ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, 1.15), framealpha=0.9)

    plt.tight_layout()
    save_figure(fig, 'SESS_04_same_family_transfer.png')


def fig11_evolution_ablation():
    """Figure 11: Evolution Necessity Ablation"""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    configs = ['every_iteration\n+ trajectory', 'single_call\n+ trajectory']
    no_evo = [62.6, 75.6]
    full_pipeline = [67.5, 76.1]

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width / 2, no_evo, width, label='Without Evolution',
                   color=COLORS['lightest'], edgecolor='#CBD5E1', linewidth=1)
    bars2 = ax.bar(x + width / 2, full_pipeline, width, label='With Evolution',
                   color=COLORS['primary'], edgecolor='#CBD5E1', linewidth=1)

    # Value annotations
    for bar, val in zip(bars1, no_evo):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=9, fontweight='bold', color=COLORS['dark'])

    for bar, val in zip(bars2, full_pipeline):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=9, fontweight='bold', color=COLORS['dark'])

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=11, fontweight='bold')
    ax.set_title('Evolution Stage Contribution Ablation',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(58, 80)
    add_y_grid(ax)
    set_grid_off(ax)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    save_figure(fig, 'fig11_evolution_ablation.png')


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Generate all midterm report figures"""
    print("=" * 60)
    print("Midterm Report Figures - Unified Blue Theme")
    print("English Only | Publication-Ready | 300 DPI")
    print("=" * 60)

    print("\n[Chapter 1: AHR-GRPO]")
    fig1_prompt_selection()
    fig2_judge_dimension()
    fig3_adaptive_vs_fixed()
    fig4_reward_type_ablation()
    fig5_ahr_grpo_summary()

    print("\n[Chapter 2: SESS]")
    fig6_sess_main_results()
    fig7_transfer_comparison()
    fig8_transfer_comprehensive()

    print("\n[Additional Analysis]")
    fig9_judge_dimension_summary()
    fig10_same_family_transfer()

    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 60)

    # List all generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
            print(f"  {f} ({size_kb:.0f} KB)")


if __name__ == '__main__':
    main()
