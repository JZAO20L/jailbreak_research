# 条形图修改说明

**修改日期**: 2026-07-09
**修改文件**: `main_bar_chart.py`

---

## 修改内容

### 1. 表述修正
- ✅ 将所有"AutoDAN+SESS"改为"DAN+SESS"
- ✅ 明确使用固定DAN模板，而非AutoDAN进化方法

### 2. 顺序调整

#### 修改前顺序
```
位置1: PAIR          (baseline)
位置2: PAIR+SESS     (SESS增强)
位置3: AutoDAN       (baseline)
位置4: DAN+SESS      (SESS增强)
```

#### 修改后顺序 ✅
```
位置1: PAIR          (baseline)
位置2: AutoDAN       (baseline)
位置3: PAIR+SESS     (SESS增强) ← 倒数第二
位置4: DAN+SESS      (SESS增强) ← 最后
```

### 3. 修改原因
- **逻辑更清晰**：先展示baseline方法（PAIR, AutoDAN），再展示SESS增强方法（PAIR+SESS, DAN+SESS）
- **对比更直观**：baseline在前，改进方法在后，便于观察提升效果
- **配色协调**：蓝色从浅到深，与改进幅度对应

---

## 图表特点

### 数据来源
- 数据集：default (WildJailbreak测试集)
- 模型：Qwen3-0.6B, Qwen3-4B, Qwen3-14B, GPT-OSS-20B
- 指标：Attack Success Rate (ASR)

### 视觉设计
- ✅ 蓝色系配色（从浅到深）：#E6F2FF → #0066CC
- ✅ 红色分隔线：区分同族模型（Qwen系列）和跨族模型（GPT-OSS-20B）
- ✅ 数值标签：每个柱子上方显示精确ASR值
- ✅ 提升标注：左下角显示平均提升幅度

### 关键信息
- 同族模型平均提升：
  - PAIR → PAIR+SESS: +15.3pp
  - AutoDAN → DAN+SESS: +14.2pp
- 跨族模型提升显著：
  - DAN+SESS: 30.0%（vs AutoDAN 12.5%）

---

## 文件输出

### 生成的文件
```
/home/tiger/jailbreak_research/self_evolve_skills_jailbreak/paper/figures/
├── main_bar_chart.png  (214KB, 3569x2069, 高分辨率)
└── main_bar_chart.pdf  (30KB, 矢量图，适合论文)
```

### 使用建议
- **论文正文**：使用PDF格式（矢量图，清晰度高）
- **演示文稿**：使用PNG格式（高分辨率，适合投影）
- **文件大小**：PDF更适合论文投稿（文件小、质量高）

---

## 技术细节

### 绘制参数
```python
width = 0.2  # 柱子宽度
figsize = (12, 7)  # 图表尺寸
dpi = 300  # 分辨率
ylim = (0, 115)  # Y轴范围
```

### 配色方案
```python
colors_blue = [
    '#E6F2FF',  # PAIR (最浅蓝)
    '#99CCFF',  # AutoDAN (浅蓝)
    '#3399FF',  # PAIR+SESS (中蓝)
    '#0066CC'   # DAN+SESS (深蓝)
]
```

---

## 后续建议

1. **LaTeX引用**
   ```latex
   \begin{figure}[t]
   \centering
   \includegraphics[width=0.9\textwidth]{figures/main_bar_chart.pdf}
   \caption{Cross-model transfer performance comparison.}
   \label{fig:main_bar_chart}
   \end{figure}
   ```

2. **论文中的说明**
   - 强调baseline方法在前，SESS增强在后
   - 指出DAN+SESS使用固定模板，而非进化方法
   - 突出跨族迁移的显著提升

3. **图表编号**
   - 如果在正文：Figure 2 或 Figure 3
   - 如果在附录：Figure A1

---

*生成时间: 2026-07-09 17:41*
*脚本位置: paper/figures/main_bar_chart.py*