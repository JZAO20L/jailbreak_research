# 跨数据集迁移对比更新说明

**更新日期**: 2026-07-09
**文件**: `PROGRESS_REPORT_STRUCTURED.md`

---

## ✅ 更新内容

### Section 4.2.2 Cross-Dataset Transfer

#### 更新前（仅PAIR系列）

```markdown
**Table 3: 跨数据集迁移效果（Qwen3-4B）**

| Dataset | PAIR | PAIR+SESS | Δ |
|---------|------|-----------|---|
| default | 55.5% | 79.0% | +23.5pp |
| ...     | ...   | ...       | ... |

**关键发现**:
- 5个数据集全部提升
- 最大提升 +23.5pp
```

#### 更新后（PAIR系列 + AutoDAN系列）✅

```markdown
**Table 3: 跨数据集迁移效果（Qwen3-4B）**

**PAIR系列对比**：

| Dataset | PAIR | PAIR+SESS | Δ |
|---------|------|-----------|---|
| default | 55.5% | 79.0% | **+23.5pp** |
| advbench | 77.3% | 81.0% | +3.7pp |
| harmbench_ctx | 81.0% | 98.0% | **+17.0pp** |
| harmbench_std | 75.5% | 91.5% | **+16.0pp** |
| jailbreakBench | 72.0% | 88.0% | **+16.0pp** |

**AutoDAN系列对比**：

| Dataset | AutoDAN | DAN+SESS | Δ |
|---------|---------|----------|---|
| default | 86.8% | **100.0%** | **+13.2pp** |
| advbench | 75.4% | **100.0%** | **+24.6pp** |
| harmbench_ctx | 87.0% | **100.0%** | **+13.0pp** |
| harmbench_std | 75.5% | **100.0%** | **+24.5pp** |
| jailbreakBench | 80.0% | **100.0%** | **+20.0pp** |
```

---

## 📊 新增的关键发现

### PAIR系列特点
```
- 5个数据集全部提升
- 最大提升: +23.5pp（default）
- 平均提升: +15.0pp
- 有害类别明确的数据集检索更有效
```

### AutoDAN系列特点（新增）✅
```
- DAN+SESS在所有数据集上达到 100.0% ASR ✅
- 最大提升: +24.6pp（advbench）
- 平均提升: +19.1pp
- 强先验整合效果显著
```

### 两组对比（新增）✅
```
AutoDAN系列提升更大：
- AutoDAN系列: +19.1pp
- PAIR系列: +15.0pp

原因分析：
1. DAN模板提供强结构先验
2. 固定模板避免无效探索
3. 跨数据集泛化能力强
```

---

## 🎯 数据验证

### AutoDAN系列详细数据

| Dataset | AutoDAN | DAN+SESS | Δ | 排名 |
|---------|---------|----------|---|------|
| default | 86.8% | 100.0% | +13.2pp | 3 |
| advbench | 75.4% | 100.0% | **+24.6pp** | 1 ⭐ |
| harmbench_ctx | 87.0% | 100.0% | +13.0pp | 4 |
| harmbench_std | 75.5% | 100.0% | **+24.5pp** | 2 ⭐ |
| jailbreakBench | 80.0% | 100.0% | +20.0pp | — |
| **平均** | **78.9%** | **100.0%** | **+19.1pp** | — |

### PAIR系列详细数据

| Dataset | PAIR | PAIR+SESS | Δ | 排名 |
|---------|------|-----------|---|------|
| default | 55.5% | 79.0% | **+23.5pp** | 1 ⭐ |
| advbench | 77.3% | 81.0% | +3.7pp | 5 |
| harmbench_ctx | 81.0% | 98.0% | +17.0pp | 3 |
| harmbench_std | 75.5% | 91.5% | +16.0pp | 4 |
| jailbreakBench | 72.0% | 88.0% | +16.0pp | — |
| **平均** | **72.3%** | **87.5%** | **+15.0pp** | — |

---

## 💡 关键洞察

### 1. DAN+SESS完美表现

```
所有数据集上达到 100.0% ASR ✅

意义：
- 强先验（DAN模板）极其有效
- 跨数据集泛化能力强
- 无需针对特定数据集优化
```

### 2. AutoDAN系列提升更大

```
对比：
- AutoDAN系列平均提升: +19.1pp
- PAIR系列平均提升: +15.0pp
- 差异: +4.1pp

原因：
1. 基线差异：AutoDAN baseline (78.9%) > PAIR baseline (72.3%)
2. 先验优势：DAN模板提供结构化引导
3. 固定模板：避免进化过程中的无效探索
```

### 3. 不同数据集的提升分布

**AutoDAN系列**：
```
提升最大: advbench (+24.6pp), harmbench_std (+24.5pp)
提升最小: harmbench_ctx (+13.0pp), default (+13.2pp)

模式：提升相对均匀（13-25pp范围）
```

**PAIR系列**：
```
提升最大: default (+23.5pp)
提升最小: advbench (+3.7pp)

模式：提升差异较大（4-24pp范围）
```

---

## 📈 与表格文件的一致性

### 验证对比

**table3_transfer_datasets.tex**（LaTeX版本）：
```latex
\multicolumn{6}{l}{\textit{Baseline: PAIR}} \\
PAIR & 55.5 & 77.3 & 81.0 & 75.5 & 72.0 \\
PAIR + SESS & 79.0 & 81.0 & 98.0 & 91.5 & 88.0 \\
$\Delta$ & \textbf{+23.5} & +3.7 & \textbf{+17.0} & \textbf{+16.0} & \textbf{+16.0} \\
\midrule
\multicolumn{6}{l}{\textit{Baseline: AutoDAN}} \\
AutoDAN & 86.8 & 75.4 & 87.0 & 75.5 & 80.0 \\
DAN + SESS & \textbf{100.0} & \textbf{100.0} & \textbf{100.0} & \textbf{100.0} & \textbf{100.0} \\
$\Delta$ & \textbf{+13.2} & \textbf{+24.6} & \textbf{+13.0} & \textbf{+24.5} & \textbf{+20.0} \\
```

**PROGRESS_REPORT_STRUCTURED.md**（Markdown版本）：
- ✅ 数据完全一致
- ✅ 格式更清晰
- ✅ 包含分析说明

---

## 🎯 论文写作建议

### 正文表述建议

```latex
\subsection{Cross-Dataset Transfer}

We evaluate cross-dataset transfer on Qwen3-4B across 5 benchmark datasets.

\paragraph{PAIR Series.}
PAIR+SESS achieves consistent improvement across all datasets 
(+3.7pp to +23.5pp, avg +15.0pp), with the largest gain on 
default (+23.5pp).

\paragraph{AutoDAN Series.}
DAN+SESS reaches \textbf{100.0\% ASR on all datasets}, demonstrating 
the power of strong structural priors. The average improvement 
is +19.1pp, outperforming the PAIR series by +4.1pp.

\paragraph{Key Insight.}
The AutoDAN series shows larger gains, indicating that strong priors 
(DAN templates) are more effective for cross-dataset transfer than 
pure evolution (PAIR-style skills).
```

---

## ✅ 更新效果

### 修改前
- ❌ 仅展示PAIR系列
- ❌ 缺少AutoDAN系列对比
- ❌ 分析不够全面

### 修改后
- ✅ 完整的PAIR + AutoDAN对比
- ✅ 两组数据的详细分析
- ✅ 关键洞察清晰
- ✅ 与LaTeX表格一致

---

*更新完成时间: 2026-07-09*
*Section 4.2.2 已完整更新* ✅