# SESS · ICLR 2027 投稿包

论文 *Beyond One-Shot Jailbreaking: Cross-Instance Experience Accumulation via
Reusable Skills* 的 **ICLR 2027 投稿完整包**：ICLR 格式论文 + 最小可执行代码 + 实验数据集。

```
SESS_iclr2027/
├── README.md                     # 本文档
├── SESS_ICLR2027.tex             # 论文（ICLR 2027 格式，匿名投稿版）
├── references.bib                # 参考文献
├── Figures/                      # 图表（SESS.drawio.pdf / main_experiment_results.pdf）
├── Tables/                       # 表格（Layer 1-4 细节表，供引用）
├── *.sty / *.bst / math_commands.tex   # 官方 ICLR 2027 样式文件
├── code/                         # 最小可执行 E2E 代码（过程式，见 code/README.md）
│   ├── sess.py · llm_client.py · start_services.sh · requirements.txt
└── data/                         # 论文使用的全部数据集
    ├── cold_start_prompts.json   # 200 条（Phase 1 冷启动）
    ├── evolution_prompts.json    # 800 条（Phase 2 进化）
    ├── test_prompts.json         # 1000 条（Phase 3 测试，独立抽样）
    ├── train_prompts.json        # 1000 条（Layer 2 数据策略消融）
    ├── seed_prompts.json         # 10 条（早期调试）
    ├── data_meta.json            # 数据溯源（WildTeaming 派生，seed 42）
    ├── dan_templates.json        # 6 个 DAN 强先验模板（Layer 3/4）
    └── benchmark/                # 迁移实验：AdvBench / HarmBench×2 / JailbreakBench
```

> ⚠️ 按投稿要求，**训练好的 skill 库不随附件提供**（已移除）；技能库由 `code/`
> 流水线经 cold start + evolution 阶段自行构建（可复现论文报告的全部数字）。

## 1. 编译论文

需要 LaTeX（pdflatex）+ natbib。样式文件已随包提供，无需安装 texlive 扩展：

```bash
cd SESS_iclr2027
pdflatex SESS_ICLR2027
bibtex  SESS_ICLR2027
pdflatex SESS_ICLR2027
pdflatex SESS_ICLR2027
```

- **匿名投稿**：作者已留空（`\author{Anonymous Submission}`）。
- **9 页限制**：ICLR 2027 主文 ≤ 9 页（参考文献不计）。本包为**逐字移植** AAAI 版
  内容；该版曾因超 9 页被 AAAI desk reject，转 ICLR 后**很可能仍超页**，需按
  `../SESS/rebuttal_plan.md` / `../REVISION_PLAN.md` 压缩（删减 Related Work、
  合并消融表、压缩 Efficiency 小节等）后方可提交。⚠️ 请在编译后确认页数。
- AI use statement / Reproducibility statement 已附于文末（不计页数）。

## 2. 运行代码

见 [`code/README.md`](code/README.md)。一句话：

```bash
bash code/start_services.sh          # 起 guard/target/attacker 三服务
python code/sess.py --mode full      # cold start -> evolution -> test
```

## 3. 数据集说明

- 核心数据源为 WildTeaming 风格 harm prompts（`data_meta.json` 记录溯源；
  上游 36,281 条去重池 → 10k 抽样 → 8000/1000/1000 划分，seed 42）。
- 论文口径：cold start=200 / evolution=800（训练、建库）；test=1000（独立抽样，
  与训练两两不重叠，已按关键词去重校验）；迁移集为 `benchmark/` 外部基准。
- 训练好的主 skill 库（463 skills）**不随附件提供**（匿名投稿合规）；技能库由
  流水线构建，`--load_library` 仅用于加载你自己之前的运行结果。

## 4. 从 AAAI 版转换的改动清单

| 项 | AAAI 版 | ICLR 版 |
|---|---|---|
| 文档类 | `aaai2027.sty` | `iclr2027_conference.sty` |
| 作者 | 留空 | `Anonymous Submission` |
| 引用 | `aaai2027.bst` | `iclr2027_conference.bst` |
| 附录 | `\section*{Appendix}` + A/B/C | `\appendix` + 自动字母编号 |
| 声明 | 无 | AI use statement（必填）+ Reproducibility statement（建议） |
| 内容 | 逐字 | 逐字（未删改，仅格式外壳） |

> 注意：正文内容尚未针对 9 页压缩，属于"格式已转、内容待压"状态。
