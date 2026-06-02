"""
Layer 3 报告生成脚本

生成Markdown格式的实验报告
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def generate_layer3_report(summary: dict) -> str:
    """
    生成 Layer 3 Markdown 报告
    """
    report = []

    report.append("# Layer 3 AutoDAN起点Skills实验报告")
    report.append("")
    report.append(f"**日期**: {datetime.now().strftime('%Y-%m-%d')}")
    report.append(f"**实验总数**: {summary['total_experiments']}")
    report.append(f"**成功**: {summary['successful']}")
    report.append(f"**失败**: {summary['failed']}")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 一、实验设计")
    report.append("")
    report.append("Layer 3 基于AutoDAN的DAN模板作为初始Skills，探索数据策略对Skills进化的影响。")
    report.append("")
    report.append("| 固定配置 | 值 | 说明 |")
    report.append("|----------|-----|------|")
    report.append("| Skills来源 | DAN模板 (6个) | AutoDAN验证有效 |")
    report.append("| 更新策略 | statistical | Layer 1/2验证最优 |")
    report.append("| 检索模式 | single_call | Layer 1/2验证最优 |")
    report.append("| Extraction | trajectory | Layer 1/2验证最优 |")
    report.append("")
    report.append("### 消融变量")
    report.append("")
    report.append("| 消融点 | 变量 | 取值 |")
    report.append("|--------|------|------|")
    report.append("| **F: 数据量** | train_limit | small(300) / medium(500) / large(1000) |")
    report.append("| **G: 配比** | cs_ratio | full_evolve(0%) / early(30%) / balanced(20%) / evo(10%) |")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 二、基准对比")
    report.append("")
    report.append("### 与AutoDAN对比")
    report.append("")
    comparison = summary.get("comparison", {})
    report.append("| 方法 | ASR | 说明 |")
    report.append("|------|-----|------|")
    report.append(f"| **AutoDAN基准** | 86.3% | 无知识积累 |")
    report.append(f"| **Layer 3最佳** | {comparison.get('best_layer3', 0)*100:.1f}% | DAN模板Skills + 积累 |")
    report.append(f"| **差距** | {comparison.get('gap_to_autodan', 0)*100:+.1f}% | |")
    report.append("")
    report.append("### full_evove vs 有冷启动")
    report.append("")
    report.append(f"| 模式 | 平均ASR | 说明 |")
    report.append(f"|------|---------|------|")
    report.append(f"| **full_evove** | {comparison.get('full_evolve_avg', 0)*100:.1f}% | 无冷启动，直接进化 |")
    report.append(f"| **有冷启动** | {comparison.get('with_cs_avg', 0)*100:.1f}% | early/balanced/evo平均 |")
    report.append(f"| **差距** | {comparison.get('full_evolve_advantage', 0)*100:+.1f}% | |")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 三、最佳配置")
    report.append("")
    best = summary.get("best_config", {})
    report.append(f"| 指标 | 值 |")
    report.append(f"|------|-----|")
    report.append(f"| **最佳ASR** | {best.get('asr', 0)*100:.1f}% |")
    report.append(f"| **数据量** | {best.get('data_size', 'N/A')} |")
    report.append(f"| **配比** | {best.get('ratio', 'N/A')} |")
    report.append(f"| **Skills数量** | {best.get('skills', 'N/A')} |")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 四、消融分析")
    report.append("")
    report.append("### 消融点 F: 数据量")
    report.append("")
    avg_by_data = summary.get("avg_by_data_size", {})
    report.append("| 数据量 | 平均ASR | 说明 |")
    report.append("|--------|---------|------|")
    for size, asr in sorted(avg_by_data.items()):
        report.append(f"| {size} | {asr}% | |")
    report.append("")
    report.append("### 消融点 G: 配比")
    report.append("")
    avg_by_ratio = summary.get("avg_by_ratio", {})
    report.append("| 配比 | 平均ASR | 说明 |")
    report.append("|------|---------|------|")
    for ratio, asr in sorted(avg_by_ratio.items()):
        if ratio == "full_evolve":
            report.append(f"| **{ratio}** | {asr}% | 无冷启动 |")
        else:
            report.append(f"| {ratio} | {asr}% | 有冷启动 |")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 五、完整结果表")
    report.append("")
    report.append("| exp_id | 数据量 | 配比 | CS | Evo | ASR | Skills | 耗时(s) |")
    report.append("|--------|--------|------|-----|-----|-----|--------|---------|")
    for r in summary.get("results_table", []):
        report.append(f"| {r['exp_id']} | {r['data_size']} | {r['ratio']} | {r['cold_start_size']} | {r['evolution_size']} | {r['asr']}% | {r['final_skills']} | {r['elapsed_time']} |")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 六、关键发现")
    report.append("")
    findings = summary.get("key_findings", {})
    report.append(f"1. **最佳ASR**: {findings.get('best_asr', 0)*100:.1f}% ({findings.get('best_data_size', 'N/A')}, {findings.get('best_ratio', 'N/A')})")
    report.append(f"2. **full_evove效果**: 相比有冷启动 {findings.get('full_evolve_effect', 0)*100:+.1f}%")
    report.append(f"3. **数据量影响**: {findings.get('data_size_effect', 0)*100:.1f}% 范围")
    report.append(f"4. **配比影响**: {findings.get('ratio_effect', 0)*100:.1f}% 范围")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 七、结论")
    report.append("")
    report.append("### 核心结论")
    report.append("")
    if comparison.get('gap_to_autodan', 0) >= 0:
        report.append("1. **DAN模板Skills有效**: 达到/超越AutoDAN基准")
    else:
        report.append("1. **DAN模板Skills差距**: 未达到AutoDAN基准，需分析原因")
    report.append("")
    if comparison.get('full_evolve_advantage', 0) >= 0:
        report.append("2. **full_evove优势**: 强初始Skills下，冷启动非必要")
    else:
        report.append("2. **有冷启动优势**: 即使强初始Skills，冷启动仍有价值")
    report.append("")
    report.append("### 建议")
    report.append("")
    report.append("基于实验结果，推荐配置:")
    report.append(f"- 数据量: {best.get('data_size', 'medium')}")
    report.append(f"- 配比: {best.get('ratio', 'balanced')}")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 附录: 实验环境")
    report.append("")
    report.append("| 配置 | 值 |")
    report.append("|------|-----|")
    report.append("| Guard Model | Qwen3Guard-Gen-4B |")
    report.append("| Target Model | Qwen3-4B |")
    report.append("| Test Size | 1000条 |")
    report.append("| 判定标准 | Strict (仅Unsafe=Success) |")
    report.append("")
    report.append("---")
    report.append("")
    report.append(f"*Generated at {datetime.now().isoformat()}*")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Layer 3 Report Generator")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    with open(args.input) as f:
        summary = json.load(f)

    report = generate_layer3_report(summary)

    with open(args.output, "w") as f:
        f.write(report)

    print(f"\n报告生成完成: {args.output}")


if __name__ == "__main__":
    main()