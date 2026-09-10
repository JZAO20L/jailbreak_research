#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""重建第三章的数据资产 (split A/B/C)。

为什么需要这个脚本: output/ 不入库, 换机器后三个种子文件全丢, 只能从
data/dataset/processed/10k 重推。历史上一度把 RFT 种子误用成 test 集造成
test 泄漏 (见 docs/LOG.md 08-26), 所以这里把隔离关系写成断言而不是注释。

    python agentic_jailbreak/scripts/build_splits.py            # 全量
    python agentic_jailbreak/scripts/build_splits.py --dry-run  # 只校验不落盘

产出 (均在 agentic_jailbreak/output/):
    rft_seed_train1000.json   A 段 = train[1000:2000], RFT 采集种子 (json 数组)
    grpo_data.jsonl           B 段 = train[0:1000],    GRPO 训练
    test_prompts.json         C 段 = test.jsonl 全量,   评估
"""

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DATA = REPO / "data/dataset/processed/10k"
OUT = REPO / "agentic_jailbreak/output"


def read_jsonl(path, field="prompt"):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            p = (obj.get(field) or "").strip()
            if p:
                prompts.append(p)
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    train = read_jsonl(DATA / "train.jsonl")
    test = read_jsonl(DATA / "test.jsonl")
    if len(train) < 2000:
        raise SystemExit(f"train.jsonl 只有 {len(train)} 条, 不够切 A[1000:2000]+B[0:1000]")

    b_seg = train[0:1000]        # GRPO 训练
    a_seg = train[1000:2000]     # RFT 采集种子
    c_seg = test                 # 评估

    # 三段两两不重叠是 08-26 定下的隔离口径, 破了就别跑实验
    sa, sb, sc = set(a_seg), set(b_seg), set(c_seg)
    assert not (sa & sb), f"A∩B 重叠 {len(sa & sb)} 条"
    assert not (sa & sc), f"A∩C 重叠 {len(sa & sc)} 条 (test 泄漏)"
    assert not (sb & sc), f"B∩C 重叠 {len(sb & sc)} 条 (test 泄漏)"
    print(f"隔离校验通过: |A|={len(sa)} |B|={len(sb)} |C|={len(sc)} "
          f"(train={len(train)}, test={len(test)})")

    if args.dry_run:
        return

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "rft_seed_train1000.json", "w", encoding="utf-8") as f:
        json.dump(a_seg, f, ensure_ascii=False)

    # plugin.reset 优先读 data_dict['prompt'], 但 swift 预处理会把自定义列剥掉,
    # 回落到"最后一条 user message" —— 两个都写, 换 swift 版本不会被静默打断
    with open(OUT / "grpo_data.jsonl", "w", encoding="utf-8") as f:
        for p in b_seg:
            f.write(json.dumps({"prompt": p,
                                "messages": [{"role": "user", "content": p}]},
                               ensure_ascii=False) + "\n")

    with open(OUT / "test_prompts.json", "w", encoding="utf-8") as f:
        json.dump(c_seg, f, ensure_ascii=False)

    print(f"已写入 {OUT}: rft_seed_train1000.json({len(a_seg)}) "
          f"grpo_data.jsonl({len(b_seg)}) test_prompts.json({len(c_seg)})")


if __name__ == "__main__":
    main()
