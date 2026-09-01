"""Generate publication-quality figures for SESS paper."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette (colorblind-friendly)
COLORS = {
    'sess': '#2171B5',      # blue
    'pair': '#6BAED6',      # light blue
    'autodan': '#FD8D3C',   # orange
    'deep': '#A1D99B',      # green
    'persona': '#BCBDDC',   # purple
    'norewrite': '#FC9272', # red
    'baseline': '#969696',  # gray
}


def fig_layer1_ablation():
    """Layer 1: Method combination ablation bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # A: Skill Call Mode
    ax = axes[0]
    modes = ['single_call', 'every_iter']
    asrs = [74.7, 66.4]
    bars = ax.bar(modes, asrs, color=[COLORS['sess'], COLORS['baseline']])
    ax.set_ylabel('Average ASR (%)')
    ax.set_title('A: Skill Call Mode')
    ax.set_ylim(60, 80)
    for bar, val in zip(bars, asrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}%', ha='center', va='bottom', fontsize=9)

    # B: Extraction Mode
    ax = axes[1]
    modes = ['trajectory', 'final_prompt']
    asrs = [71.8, 69.3]
    bars = ax.bar(modes, asrs, color=[COLORS['sess'], COLORS['baseline']])
    ax.set_ylabel('Average ASR (%)')
    ax.set_title('B: Extraction Mode')
    ax.set_ylim(65, 75)
    for bar, val in zip(bars, asrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val}%', ha='center', va='bottom', fontsize=9)

    # C: Update Strategy
    ax = axes[2]
    strategies = ['statistical', 'success_only', 'failure_only', 'both']
    asrs = [69.7, 73.6, 69.5, 69.4]
    colors = [COLORS['sess'], COLORS['pair'], COLORS['baseline'], COLORS['norewrite']]
    bars = ax.bar(strategies, asrs, color=colors)
    ax.set_ylabel('Average ASR (%)')
    ax.set_title('C: Update Strategy')
    ax.set_ylim(65, 78)
    for bar, val in zip(bars, asrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'layer1_ablation.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'layer1_ablation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: layer1_ablation.pdf/png")


def fig_layer3_dan():
    """Layer 3: DAN template vs non-DAN comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    configs = ['full_evolve\n(0% CS)', 'balanced\n(20% CS)', 'early\n(30% CS)', 'evo\n(10% CS)']
    asrs = [98.8, 92.7, 89.9, 86.5]
    colors = [COLORS['sess'], COLORS['pair'], COLORS['autodan'], COLORS['baseline']]

    bars = ax.bar(configs, asrs, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Test ASR (%)')
    ax.set_title('Layer 3: DAN Template + Different Evolution Modes')
    ax.set_ylim(80, 102)

    for bar, val in zip(bars, asrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(y=98.8, color=COLORS['sess'], linestyle='--', alpha=0.3, label='Best: 98.8%')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'layer3_dan.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'layer3_dan.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: layer3_dan.pdf/png")


def fig_transfer_models():
    """Cross-model transfer comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = ['Qwen3-0.6B', 'Qwen3-4B', 'Qwen3-14B', 'GPT-OSS-20B\n(cross-family)']
    x = np.arange(len(models))
    width = 0.12

    methods = {
        'no_rewrite': [57.8, 28.2, 24.4, 0.1],
        'pair': [86.4, 72.3, 71.6, 0.1],
        'autodan': [90.2, 78.7, 85.5, 6.0],
        'deepinception': [75.5, 49.3, 42.5, 0.3],
        'persona': [71.9, 32.8, 27.8, 0.7],
        'pair_skills_28': [95.1, 86.7, 86.5, 11.4],
    }

    colors_list = [COLORS['norewrite'], COLORS['pair'], COLORS['autodan'],
                   COLORS['deep'], COLORS['persona'], COLORS['sess']]

    for i, (method, values) in enumerate(methods.items()):
        offset = (i - len(methods)/2 + 0.5) * width
        label = method.replace('_', '\_') if method == 'pair_skills_28' else method
        if method == 'pair_skills_28':
            label = 'pair_skills_28 (ours)'
        bars = ax.bar(x + offset, values, width, label=label,
                     color=colors_list[i], edgecolor='white', linewidth=0.5)

    ax.set_ylabel('ASR (%)')
    ax.set_title('Cross-Model Transfer Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'transfer_models.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'transfer_models.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: transfer_models.pdf/png")


def fig_transfer_heatmap():
    """Cross-dataset transfer heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Qwen3-4B cross-dataset
    ax = axes[0]
    datasets = ['default', 'advbench', 'hb_ctx', 'hb_std', 'jbBench']
    methods = ['no_rewrite', 'pair', 'autodan', 'deepinception', 'persona', 'pair_skills_28']
    data = np.array([
        [30.3, 3.3, 79.0, 21.5, 7.0],
        [55.5, 77.3, 81.0, 75.5, 72.0],
        [86.8, 75.4, 87.0, 75.5, 80.0],
        [40.4, 41.0, 80.0, 47.5, 38.0],
        [32.3, 16.7, 67.0, 36.0, 20.0],
        [79.0, 81.0, 98.0, 91.5, 88.0],
    ])

    im = ax.imshow(data, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha='right')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title('Qwen3-4B: Cross-Dataset ASR')

    for i in range(len(methods)):
        for j in range(len(datasets)):
            val = data[i, j]
            color = 'white' if val > 60 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                   fontsize=8, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)

    # Right: Cross-model summary
    ax = axes[1]
    models = ['0.6B', '4B', '14B', '20B*']
    methods_short = ['pair', 'autodan', 'pair_skills']
    data2 = np.array([
        [86.4, 72.3, 71.6, 0.1],
        [90.2, 78.7, 85.5, 6.0],
        [95.1, 86.7, 86.5, 11.4],
    ])

    im2 = ax.imshow(data2, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_yticks(range(len(methods_short)))
    ax.set_yticklabels(methods_short)
    ax.set_title('Cross-Model ASR (* = cross-family)')

    for i in range(len(methods_short)):
        for j in range(len(models)):
            val = data2[i, j]
            color = 'white' if val > 60 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                   fontsize=9, fontweight='bold', color=color)

    plt.colorbar(im2, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'transfer_heatmap.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'transfer_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: transfer_heatmap.pdf/png")


def fig_evolution_ablation():
    """Evolution necessity ablation."""
    fig, ax = plt.subplots(figsize=(6, 4))

    configs = ['every_iter + traj', 'single_call + traj', 'Average']
    no_evo = [62.6, 75.6, 69.6]
    full = [67.5, 76.1, 70.6]

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax.bar(x - width/2, no_evo, width, label='No Evolution', color=COLORS['baseline'])
    bars2 = ax.bar(x + width/2, full, width, label='Full Pipeline', color=COLORS['sess'])

    ax.set_ylabel('ASR (%)')
    ax.set_title('Evolution Stage Contribution')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.set_ylim(55, 82)

    for bar, val in zip(bars1, no_evo):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val}%', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, full):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val}%', ha='center', va='bottom', fontsize=9)

    # Add delta annotations
    for i in range(len(configs)):
        delta = full[i] - no_evo[i]
        ax.annotate(f'+{delta}%', xy=(x[i], max(no_evo[i], full[i]) + 2),
                   fontsize=9, ha='center', color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'evolution_ablation.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'evolution_ablation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: evolution_ablation.pdf/png")


if __name__ == '__main__':
    fig_layer1_ablation()
    fig_layer3_dan()
    fig_transfer_models()
    fig_transfer_heatmap()
    fig_evolution_ablation()
    print(f"\nAll figures saved to {OUTPUT_DIR}/")
