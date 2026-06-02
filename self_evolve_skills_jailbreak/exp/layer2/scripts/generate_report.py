"""
Layer 2 报告生成脚本
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def generate_report(input_file: str, output_file: str):
    """生成 Layer 2 Markdown 报告"""
    with open(input_file) as f:
        summary = json.load(f)

    experiments = summary.get("experiments", [])
    successful = [e for e in experiments if e.get("success") and "asr" in e]

    # 按实验 ID 排序
    experiments.sort(key=lambda x: x.get("experiment_id", 0))

    # 生成 Markdown
    md = f"""# Layer 2 Data Ablation Experiment Report

**日期**: {datetime.now().strftime('%Y-%m-%d')}
**实验总数**: {len(experiments)}
**成功**: {len(successful)}
**失败**: {len(experiments) - len(successful)}

---

## 一、实验结果汇总

按 ASR 降序排列：

| 排名 | 方法 | 数据量 | 配比 | ASR | Avg Iter | Skills |
|------|------|--------|------|-----|----------|--------|
"""

    # 按 ASR 排序
    sorted_by_asr = sorted(successful, key=lambda x: x["asr"], reverse=True)

    for i, exp in enumerate(sorted_by_asr, 1):
        method = exp.get("method", {})
        method_str = f"{method.get('skill_extraction_mode', '')}_{method.get('update_strategy', '')}"
        data = exp.get("data_size", "")
        ratio = exp.get("ratio", "")
        asr = exp.get("asr", 0) * 100
        avg_iter = exp.get("avg_iterations", 0)
        skills = exp.get("evolution_skills", "N/A")

        md += f"| {i} | {method_str} | {data} | {ratio} | {asr:.1f}% | {avg_iter:.2f} | {skills} |\n"

    md += """
---

## 二、消融分析

"""

    # 按数据量分析
    if "by_data_size" in summary:
        md += """### 消融点 D: 数据量

| 数据量 | 平均 ASR | 平均迭代 | 实验数 |
|--------|----------|----------|--------|
"""
        for size, stats in summary["by_data_size"].items():
            md += f"| {size} | {stats['avg_asr']*100:.1f}% | {stats['avg_iter']:.2f} | {stats['count']} |\n"

        # 找出最佳数据量
        best_size = max(summary["by_data_size"].items(), key=lambda x: x[1]["avg_asr"])
        md += f"""
**结论**: **{best_size[0]}** 效果最好 (ASR: {best_size[1]['avg_asr']*100:.1f}%)

"""

    # 按配比分析
    if "by_ratio" in summary:
        md += """### 消融点 E: Cold Start / Evolution 配比

| 配比 | 平均 ASR | 平均迭代 | 实验数 |
|------|----------|----------|--------|
"""
        for ratio, stats in summary["by_ratio"].items():
            md += f"| {ratio} | {stats['avg_asr']*100:.1f}% | {stats['avg_iter']:.2f} | {stats['count']} |\n"

        # 找出最佳配比
        best_ratio = max(summary["by_ratio"].items(), key=lambda x: x[1]["avg_asr"])
        md += f"""
**结论**: **{best_ratio[0]}** 效果最好 (ASR: {best_ratio[1]['avg_asr']*100:.1f}%)

"""

    # 按方法分析
    if "by_method" in summary:
        md += """### 方法组合对比

| 方法组合 | 平均 ASR | 平均迭代 | 实验数 |
|----------|----------|----------|--------|
"""
        for method, stats in summary["by_method"].items():
            md += f"| {method} | {stats['avg_asr']*100:.1f}% | {stats['avg_iter']:.2f} | {stats['count']} |\n"

        md += "\n"

    # 最佳组合
    if summary.get("best_combination"):
        best = summary["best_combination"]
        method = best.get("method", {})
        md += f"""---

## 三、最佳组合

**{method.get('skill_call_mode', '')} + {method.get('skill_extraction_mode', '')} + {method.get('update_strategy', '')}**

| 配置 | 值 |
|------|-----|
| 数据量 | {best.get('data_size')} ({best.get('data_size_value')} 条) |
| 配比 | {best.get('ratio')} ({best.get('ratio_value')*100:.0f}%) |
| ASR | **{best.get('asr', 0)*100:.1f}%** |
| 平均迭代 | {best.get('avg_iterations', 0):.2f} |
| Skills数量 | {best.get('evolution_skills', 'N/A')} |

"""

    md += """---

## 四、关键发现

"""

    if "by_data_size" in summary and "by_ratio" in summary:
        sizes = sorted(summary["by_data_size"].items(), key=lambda x: x[1]["avg_asr"], reverse=True)
        ratios = sorted(summary["by_ratio"].items(), key=lambda x: x[1]["avg_asr"], reverse=True)

        md += f"""1. **数据量影响**:
   - {sizes[0][0]} 效果最好 ({sizes[0][1]['avg_asr']*100:.1f}%)
   - {sizes[-1][0]} 效果最差 ({sizes[-1][1]['avg_asr']*100:.1f}%)
   - 差距: {(sizes[0][1]['avg_asr'] - sizes[-1][1]['avg_asr'])*100:.1f}%

2. **配比影响**:
   - {ratios[0][0]} 效果最好 ({ratios[0][1]['avg_asr']*100:.1f}%)
   - {ratios[-1][0]} 效果最差 ({ratios[-1][1]['avg_asr']*100:.1f}%)
   - 差距: {(ratios[0][1]['avg_asr'] - ratios[-1][1]['avg_asr'])*100:.1f}%

"""

    md += """---

## 五、完整实验表

| exp_id | method | data_size | ratio | ASR | Avg Iter | Skills |
|--------|--------|-----------|-------|-----|----------|--------|
"""

    for exp in experiments:
        method = exp.get("method", {})
        method_str = f"{method.get('skill_extraction_mode', '')}_{method.get('update_strategy', '')}"
        data = exp.get("data_size", "")
        ratio = exp.get("ratio", "")
        asr = exp.get("asr", "N/A")
        if isinstance(asr, float):
            asr = f"{asr*100:.1f}%"
        avg_iter = exp.get("avg_iterations", "N/A")
        if isinstance(avg_iter, float):
            avg_iter = f"{avg_iter:.2f}"
        skills = exp.get("evolution_skills", "N/A")

        md += f"| {exp.get('experiment_id', '')} | {method_str} | {data} | {ratio} | {asr} | {avg_iter} | {skills} |\n"

    md += """
---

## 六、后续建议

1. 使用最佳数据配置运行完整实验 (3 epochs)
2. 与 Layer 1 结果对比验证
3. 与 Baselines 对比验证最终效果
"""

    # 保存报告
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate Layer 2 report")
    parser.add_argument("--input", required=True, help="Input summary JSON file")
    parser.add_argument("--output", required=True, help="Output Markdown report file")

    args = parser.parse_args()
    generate_report(args.input, args.output)


if __name__ == "__main__":
    main()