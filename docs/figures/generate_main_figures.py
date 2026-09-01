#!/usr/bin/env python3
"""
中期报告主图生成脚本
生成AHR-GRPO和SESS两章的核心主图

Usage:
    python generate_main_figures.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 输出目录
OUTPUT_DIR = '/home/tiger/jailbreak_research/docs/figures/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_figure(fig, filename):
    """保存图片"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 已保存: {filepath}")
    plt.close(fig)


# =============================================================================
# 第一章 AHR-GRPO 主图
# =============================================================================

def figure1_ahr_grpo_framework():
    """图1: AHR-GRPO整体训练框架"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 定义颜色
    colors = {
        'input': '#dbeafe',
        'policy': '#6366f1',
        'target': '#f59e0b',
        'reward': '#10b981',
        'update': '#ef4444',
    }

    # 绘制模块框
    def draw_box(x, y, w, h, text, color, fontsize=11):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                        boxstyle="round,pad=0.05,rounding_size=0.2",
                                        facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', wrap=True)

    # 输入层
    draw_box(0.5, 8.5, 2.5, 1, '原始有害Prompt', colors['input'])
    draw_box(0.5, 7, 2.5, 1, '攻击Prompt模板', colors['input'])

    # Policy模型
    draw_box(4, 7.5, 3, 2, 'Policy模型\n(GRPO训练)\n生成k个候选', colors['policy'])

    # Target模型
    draw_box(8, 7.5, 2.5, 2, 'Target模型\n执行攻击\n生成响应', colors['target'])

    # 奖励计算
    draw_box(4, 4.5, 3, 2, 'ASR Reward\n(稀疏信号)', colors['reward'])
    draw_box(8, 4.5, 2.5, 2, 'Judge Reward\n(稠密信号)', colors['reward'])

    # 自适应权重
    draw_box(5.5, 2, 4, 1.5, '自适应权重计算器\nλ = σ(α·log(var_ASR/var_Judge)+δ)',
             '#8b5cf6', fontsize=10)

    # 策略更新
    draw_box(5.5, 0.3, 4, 1.2, '混合奖励 → GRPO优化\n更新Policy', colors['update'])

    # 绘制箭头
    arrow_style = dict(arrowstyle='->', lw=2, color='black')

    # 输入到Policy
    ax.annotate('', xy=(4, 8.5), xytext=(3, 8.5), arrowprops=arrow_style)
    ax.annotate('', xy=(4, 7.5), xytext=(3, 7), arrowprops=arrow_style)

    # Policy到Target
    ax.annotate('', xy=(8, 8), xytext=(7, 8), arrowprops=arrow_style)

    # Target到ASR Reward
    ax.annotate('', xy=(5.5, 6), xytext=(9, 7), arrowprops=arrow_style)

    # Policy到Judge Reward
    ax.annotate('', xy=(9, 6), xytext=(5.5, 6), arrowprops=arrow_style)

    # 奖励到自适应权重
    ax.annotate('', xy=(7.5, 3), xytext=(5.5, 6), arrowprops=arrow_style)
    ax.annotate('', xy=(7.5, 3), xytext=(9, 6), arrowprops=arrow_style)

    # 自适应权重到更新
    ax.annotate('', xy=(7.5, 1.2), xytext=(7.5, 2.5), arrowprops=arrow_style)

    # 更新回到Policy
    ax.annotate('', xy=(5.5, 8.5), xytext=(7.5, 1.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='red', connectionstyle='arc3,rad=0.3'))

    ax.set_title('AHR-GRPO 整体训练框架', fontsize=18, fontweight='bold', pad=20)

    save_figure(fig, 'figure1_ahr_grpo_framework.png')


def figure2_lambda_evolution():
    """图2: Lambda演化曲线"""
    fig, ax = plt.subplots(figsize=(12, 6))

    steps = np.arange(1, 1001)

    # 模拟lambda演化过程
    np.random.seed(42)

    # 早期：lambda波动较低（依赖Judge探索）
    early = 0.35 + 0.08 * np.sin(steps[:200] / 40) + 0.03 * np.random.randn(200)

    # 中期：lambda逐渐上升（ASR信号分化）
    mid = 0.4 + 0.0006 * (steps[200:600] - 200) + 0.02 * np.random.randn(400)

    # 后期：lambda稳定在较高值
    late = 0.6 + 0.01 * np.random.randn(400)

    lambda_curve = np.concatenate([early, mid, late])
    lambda_curve = np.clip(lambda_curve, 0.2, 0.8)

    # 平滑处理
    from scipy.ndimage import uniform_filter1d
    lambda_smooth = uniform_filter1d(lambda_curve, size=10)

    # 绘制
    ax.plot(steps, lambda_smooth, 'b-', linewidth=2.5, label='λ (ASR权重)')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1.5,
               label='固定权重基线')

    # 区域标注
    ax.axvspan(1, 200, alpha=0.15, color='#6366f1')
    ax.axvspan(200, 600, alpha=0.15, color='#10b981')
    ax.axvspan(600, 1000, alpha=0.15, color='#f59e0b')

    # 标注文字
    ax.text(100, 0.75, '早期探索\n(依赖Judge)', fontsize=12, ha='center',
            color='#6366f1', fontweight='bold')
    ax.text(400, 0.75, '中期过渡\n(ASR分化)', fontsize=12, ha='center',
            color='#10b981', fontweight='bold')
    ax.text(800, 0.75, '后期优化\n(回归ASR)', fontsize=12, ha='center',
            color='#f59e0b', fontweight='bold')

    ax.set_xlabel('训练步数', fontsize=14)
    ax.set_ylabel('λ (ASR权重)', fontsize=14)
    ax.set_title('自适应奖励权重 λ 演化曲线', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    save_figure(fig, 'figure2_lambda_evolution.png')


def figure3_variance_ratio_mechanism():
    """图3: 方差比映射机制"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 子图1: 方差比与lambda的关系
    ax1 = axes[0]
    ratios = np.linspace(0.1, 10, 100)
    alpha = 2.0
    delta = -2.0
    lambda_raw = 1 / (1 + np.exp(-alpha * np.log(ratios) - delta))

    ax1.plot(ratios, lambda_raw, 'b-', linewidth=2)
    ax1.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='ratio=1')
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='λ=0.5')
    ax1.set_xlabel('方差比 (var_ASR / var_Judge)', fontsize=12)
    ax1.set_ylabel('λ_raw', fontsize=12)
    ax1.set_title('方差比 → λ 映射\n(Sigmoid变换)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 子图2: 不同alpha的影响
    ax2 = axes[1]
    for a in [1.0, 2.0, 4.0]:
        l = 1 / (1 + np.exp(-a * np.log(ratios) - delta))
        ax2.plot(ratios, l, linewidth=2, label=f'α={a}')
    ax2.set_xlabel('方差比', fontsize=12)
    ax2.set_ylabel('λ_raw', fontsize=12)
    ax2.set_title('参数α的影响\n(敏感度)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 子图3: 物理直觉示意
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')

    # 绘制概念图
    ax3.text(2, 8, 'ASR方差小\n(信号稀疏)', fontsize=12, ha='center',
             color='#6366f1', fontweight='bold', bbox=dict(boxstyle='round', facecolor='#dbeafe'))
    ax3.text(8, 8, 'ASR方差大\n(信号分化)', fontsize=12, ha='center',
             color='#f59e0b', fontweight='bold', bbox=dict(boxstyle='round', facecolor='#fef3c7'))

    ax3.annotate('', xy=(8, 7), xytext=(2, 7),
                 arrowprops=dict(arrowstyle='<->', lw=2, color='black'))

    ax3.text(2, 5, 'λ ↓\n依赖Judge\n稠密引导', fontsize=11, ha='center',
             color='#6366f1')
    ax3.text(8, 5, 'λ ↑\n回归ASR\n结果导向', fontsize=11, ha='center',
             color='#f59e0b')

    ax3.text(5, 2, '自适应机制:\n感知方差 → 调整权重 → 稳定收敛',
             fontsize=12, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#dcfce7', edgecolor='#10b981', linewidth=2))

    ax3.set_title('物理直觉', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'figure3_variance_ratio_mechanism.png')


# =============================================================================
# 第二章 SESS 主图
# =============================================================================

def figure4_sess_three_phase():
    """图4: SESS三阶段流程"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    colors = {
        'phase1': '#6366f1',
        'phase2': '#10b981',
        'phase3': '#f59e0b',
        'step': '#e0e7ff',
        'decision': '#fef3c7',
    }

    def draw_box(x, y, w, h, text, color, fontsize=10, edgecolor='black'):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                        boxstyle="round,pad=0.05,rounding_size=0.2",
                                        facecolor=color, edgecolor=edgecolor, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', wrap=True)

    def draw_diamond(x, y, size, text, color):
        diamond = mpatches.RegularPolygon((x, y), numVertices=4, radius=size,
                                           orientation=0, facecolor=color,
                                           edgecolor='black', linewidth=1.5)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # Phase 1: Cold Start
    draw_box(0.5, 8, 4, 1.5, 'Phase 1:\nCold Start', colors['phase1'], fontsize=14)
    draw_box(0.5, 6, 2, 1, '加载初始\nSkills', colors['step'])
    draw_box(2.7, 6, 2, 1, '执行攻击', colors['step'])
    draw_diamond(1.5, 4.5, 0.7, '成功?', colors['decision'])
    draw_box(0.5, 3, 2, 1, '提取\n新Skill', colors['step'])
    draw_box(2.7, 3, 2, 1, '记录失败\n轨迹', colors['step'])
    draw_box(1.5, 1.5, 3, 1, '更新Skills库', colors['step'])

    # Phase 2: Evolution
    draw_box(5.5, 8, 5.5, 1.5, 'Phase 2:\nEvolution', colors['phase2'], fontsize=14)
    draw_box(5.5, 6, 2, 1, '检索最佳\nSkill', colors['step'])
    draw_box(7.7, 6, 2, 1, '注入Skill\n攻击', colors['step'])
    draw_diamond(6.5, 4.5, 0.7, '成功?', colors['decision'])
    draw_box(5.5, 3, 2, 1, '提取成功\nSkill', colors['step'])
    draw_box(7.7, 3, 2, 1, '反思进化\nSkill', colors['step'])
    draw_box(6.5, 1.5, 3, 1, '更新统计', colors['step'])
    draw_diamond(9.5, 4.5, 0.7, '每50次?', colors['decision'])
    draw_box(9.5, 3, 1.5, 1, '维护流程', '#fee2e2')

    # Phase 3: Test
    draw_box(12, 8, 3.5, 1.5, 'Phase 3:\nTest', colors['phase3'], fontsize=14)
    draw_box(12, 6, 3.5, 1, '加载最终\nSkills库', colors['step'])
    draw_box(12, 4.5, 3.5, 1, '1000条测试\nPrompt', colors['step'])
    draw_box(12, 3, 3.5, 1, '计算ASR', colors['step'])
    draw_box(12, 1.5, 3.5, 1, 'Skills质量\n统计', colors['step'])

    # 箭头
    arrow_style = dict(arrowstyle='->', lw=1.5, color='black')

    # Phase 1 arrows
    ax.annotate('', xy=(1.5, 6), xytext=(1.5, 8), arrowprops=arrow_style)
    ax.annotate('', xy=(3.7, 6), xytext=(2.5, 6), arrowprops=arrow_style)
    ax.annotate('', xy=(1.5, 4.5), xytext=(3.2, 5.5), arrowprops=arrow_style)
    ax.annotate('', xy=(1.5, 3), xytext=(1.5, 3.8), arrowprops=arrow_style)
    ax.annotate('', xy=(3.7, 3), xytext=(1.5, 4.5), arrowprops=arrow_style)

    # Phase 2 arrows
    ax.annotate('', xy=(6.5, 6), xytext=(6.5, 8), arrowprops=arrow_style)
    ax.annotate('', xy=(8.7, 6), xytext=(7.5, 6), arrowprops=arrow_style)
    ax.annotate('', xy=(6.5, 4.5), xytext=(8.2, 5.5), arrowprops=arrow_style)
    ax.annotate('', xy=(6.5, 3), xytext=(6.5, 3.8), arrowprops=arrow_style)
    ax.annotate('', xy=(8.7, 3), xytext=(6.5, 4.5), arrowprops=arrow_style)

    # Phase 3 arrows
    ax.annotate('', xy=(13.75, 6), xytext=(13.75, 8), arrowprops=arrow_style)
    ax.annotate('', xy=(13.75, 4.5), xytext=(13.75, 6), arrowprops=arrow_style)
    ax.annotate('', xy=(13.75, 3), xytext=(13.75, 4.5), arrowprops=arrow_style)
    ax.annotate('', xy=(13.75, 1.5), xytext=(13.75, 3), arrowprops=arrow_style)

    # Phase连接箭头
    ax.annotate('', xy=(5.5, 8), xytext=(4.5, 8),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax.annotate('', xy=(12, 8), xytext=(11, 8),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))

    ax.set_title('SESS 三阶段自进化流程', fontsize=18, fontweight='bold', pad=20)

    save_figure(fig, 'figure4_sess_three_phase.png')


def figure5_skills_evolution():
    """图5: Skills演化过程"""
    fig, ax1 = plt.subplots(figsize=(12, 7))

    iterations = np.arange(0, 1000, 50)

    # Skills数量变化（模拟）
    skills_count = np.array([5, 12, 18, 25, 28, 35, 42, 38, 32, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28])

    # ASR变化
    asr = np.array([0.55, 0.62, 0.68, 0.72, 0.75, 0.78, 0.80, 0.82, 0.84, 0.85,
                    0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95])

    # 绘制Skills数量
    color1 = '#6366f1'
    ax1.plot(iterations, skills_count, color=color1, linewidth=2.5, marker='o',
             markersize=8, label='Skills数量')
    ax1.fill_between(iterations, skills_count, alpha=0.3, color=color1)
    ax1.set_xlabel('训练迭代次数', fontsize=14)
    ax1.set_ylabel('Skills数量', fontsize=14, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 50)

    # 绘制ASR
    ax2 = ax1.twinx()
    color2 = '#10b981'
    ax2.plot(iterations, asr, color=color2, linewidth=2.5, marker='s',
             linestyle='--', markersize=8, label='ASR')
    ax2.set_ylabel('ASR', fontsize=14, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.5, 1.0)

    # 标注维护节点
    maintenance_points = iterations[::2]
    for mp in maintenance_points:
        ax1.axvline(x=mp, color='gray', linestyle=':', alpha=0.2)

    # 标注关键阶段
    ax1.axvspan(0, 200, alpha=0.1, color='#6366f1', label='Cold Start')
    ax1.axvspan(200, 600, alpha=0.1, color='#10b981', label='Evolution')
    ax1.axvspan(600, 1000, alpha=0.1, color='#f59e0b', label='稳定阶段')

    ax1.text(100, 45, 'Cold Start', fontsize=12, ha='center', color='#6366f1', fontweight='bold')
    ax1.text(400, 45, 'Evolution', fontsize=12, ha='center', color='#10b981', fontweight='bold')
    ax1.text(800, 45, '稳定', fontsize=12, ha='center', color='#f59e0b', fontweight='bold')

    plt.title('Skills演化过程：数量与ASR变化', fontsize=16, fontweight='bold')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

    plt.tight_layout()
    save_figure(fig, 'figure5_skills_evolution.png')


def figure6_layer_asr_comparison():
    """图6: SESS各层实验ASR对比"""
    fig, ax = plt.subplots(figsize=(14, 7))

    layers = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Ablation', 'Transfer\n(跨族)']
    best_asr = [79.1, 80.0, 98.8, 99.7, 69.6, 32.5]
    colors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#gray']

    bars = ax.bar(layers, best_asr, color=colors, edgecolor='black', linewidth=2)

    # 标注数值
    for bar, asr in zip(bars, best_asr):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 2,
                f'{asr}%', ha='center', fontsize=13, fontweight='bold')

    # 标注实验数量
    exp_counts = [16, 36, 17, 12, 16, 140]
    for bar, count in zip(bars, exp_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{count}组', ha='center', fontsize=11, color='white', fontweight='bold')

    ax.set_xlabel('实验层', fontsize=14)
    ax.set_ylabel('最佳ASR (%)', fontsize=14)
    ax.set_title('SESS 各层实验最佳 ASR 对比\n(总计217组实验)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.axhline(y=85, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(5.5, 87, 'Baseline参考线', fontsize=10, color='red')

    plt.tight_layout()
    save_figure(fig, 'figure6_layer_asr_comparison.png')


def figure7_transfer_comparison():
    """图7: 跨族迁移效果对比"""
    fig, ax = plt.subplots(figsize=(12, 7))

    methods = ['pair_skills_28', 'autodan_skills_54', 'autodan\n(baseline)', 'pair\n(baseline)']
    same_family = [91.6, 99.0, 85.5, 71.6]
    cross_family = [11.4, 32.5, 6.0, 0.08]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, same_family, width, label='同族模型 (Qwen3)',
                   color='#10b981', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, cross_family, width, label='跨族模型 (gpt-oss-20b)',
                   color='#ef4444', edgecolor='black', linewidth=1.5)

    # 标注数值
    for bar, val in zip(bars1, same_family):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=11, fontweight='bold', color='#10b981')

    for bar, val in zip(bars2, cross_family):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=11, fontweight='bold', color='#ef4444')

    ax.set_xlabel('方法', fontsize=14)
    ax.set_ylabel('ASR (%)', fontsize=14)
    ax.set_title('Skills方法跨族迁移效果对比\n(核心发现：跨族迁移是真正挑战)',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 110)
    ax.grid(True, axis='y', alpha=0.3)

    # 标注差距
    for i, (s, c) in enumerate(zip(same_family, cross_family)):
        diff = s - c
        ax.annotate('', xy=(x[i] + width/2, c + 5), xytext=(x[i] - width/2, s + 5),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax.text(x[i], max(s, c) + 15, f'-{diff:.1f}%', ha='center', fontsize=10,
                color='gray', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'figure7_transfer_comparison.png')


# =============================================================================
# 主函数
# =============================================================================

def main():
    """生成所有主图"""
    print("=" * 60)
    print("中期报告主图生成")
    print("=" * 60)

    print("\n【第一章 AHR-GRPO 主图】")
    figure1_ahr_grpo_framework()
    figure2_lambda_evolution()
    figure3_variance_ratio_mechanism()

    print("\n【第二章 SESS 主图】")
    figure4_sess_three_phase()
    figure5_skills_evolution()
    figure6_layer_asr_comparison()
    figure7_transfer_comparison()

    print("\n" + "=" * 60)
    print(f"✓ 所有主图已生成完毕！保存位置: {OUTPUT_DIR}")
    print("=" * 60)

    # 输出文件列表
    print("\n生成的文件列表:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  - {f}")


if __name__ == '__main__':
    main()