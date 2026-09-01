# 图表重绘总结

**修改日期**: 2026-07-09 17:47
**修改脚本**: 3个Python脚本

---

## ✅ 已完成的修改

### 1. 表述修正
- ✅ 所有"AutoDAN+SESS"改为"DAN+SESS"
- ✅ 明确使用固定DAN模板，而非AutoDAN进化方法

### 2. 顺序调整 ✅

所有图表都调整为统一顺序：

```
位置1: PAIR          (baseline方法)
位置2: AutoDAN       (baseline方法)
位置3: PAIR+SESS     (SESS增强，倒数第二) ← 
位置4: DAN+SESS      (SESS增强，最后)
```

**调整优势**：
- ✅ Baseline方法在前（PAIR, AutoDAN）
- ✅ SESS增强方法在后（PAIR+SESS, DAN+SESS）
- ✅ 对比更直观，逻辑更清晰
- ✅ 所有图表顺序统一

---

## 📊 生成的图表

### 1. main_experiment_results.png
- **尺寸**: 4760 x 3538 (高分辨率)
- **大小**: 415KB
- **内容**: 4模型 × 5数据集 × 4方法 分组柱状图
- **特点**: 每个模型一个子图，展示跨数据集性能

### 2. main_experiment_results_simple.png
- **尺寸**: 3561 x 2061
- **大小**: 165KB
- **内容**: 4模型 × default数据集 分组柱状图
- **特点**: 简化版，仅展示主要对比

### 3. main_figure_cross_model_dataset.png
- **尺寸**: 4696 x 4127 (高分辨率)
- **大小**: 677KB
- **内容**: 4模型 × 5数据集热力图
- **特点**: 完整的跨模型×跨数据集结果矩阵

---

## 🎨 配色方案

### 统一蓝色系
```python
位置1 (PAIR):         #BFDBFE (最浅蓝)
位置2 (AutoDAN):      #93C5FD (浅蓝)
位置3 (PAIR+SESS):    #3B82F6 (中蓝)
位置4 (DAN+SESS):     #2563EB (深蓝)
```

### 热力图颜色
- **映射**: RdYlGn (红-黄-绿)
- **含义**: 红色=低ASR，绿色=高ASR
- **范围**: 0-100%

---

## 📋 修改详情

### main_experiment_results.py

#### 函数1: main_experiment_results()
```python
# 修改前
bars1 = PAIR (位置1)
bars2 = PAIR+SESS (位置2)
bars3 = AutoDAN (位置3)
bars4 = DAN+SESS (位置4)

# 修改后 ✅
bars1 = PAIR (位置1)
bars2 = AutoDAN (位置2)
bars3 = PAIR+SESS (位置3) ← 倒数第二
bars4 = DAN+SESS (位置4)
```

#### 函数2: main_experiment_results_simple()
- ✅ 同样的顺序调整
- ✅ 标签从"AutoDAN+SESS"改为"DAN+SESS"

### main_figure_cross_model_dataset.py

#### methods列表顺序
```python
# 修改前
methods = ['no_rewrite', 'DeepInception', 'Persona', 
           'PAIR', 'PAIR+SESS', 'AutoDAN', 'DAN+SESS']

# 修改后 ✅
methods = ['no_rewrite', 'DeepInception', 'Persona', 
           'PAIR', 'AutoDAN', 'PAIR+SESS', 'DAN+SESS']
```

**热力图行顺序**（从上到下）：
1. no_rewrite
2. DeepInception
3. Persona
4. PAIR
5. AutoDAN
6. PAIR+SESS ← 倒数第二行
7. DAN+SESS ← 最后

---

## 🎯 关键改进

### 1. 表述清晰
- ❌ 修改前：AutoDAN+SESS（混淆进化方法和固定模板）
- ✅ 修改后：DAN+SESS（明确使用固定模板）

### 2. 逻辑一致
- ✅ 所有图表顺序统一
- ✅ Baseline在前，改进在后
- ✅ 便于观察提升效果

### 3. 视觉协调
- ✅ 蓝色系配色从浅到深
- ✅ 与改进幅度对应
- ✅ 热力图使用标准色板

---

## 📂 文件位置

```
/home/tiger/jailbreak_research/self_evolve_skills_jailbreak/paper/figures/
├── main_experiment_results.png          (415KB)
├── main_experiment_results.pdf          (矢量图)
├── main_experiment_results_simple.png   (165KB)
├── main_experiment_results_simple.pdf   (矢量图)
├── main_figure_cross_model_dataset.png  (677KB)
├── main_figure_cross_model_dataset.pdf  (矢量图)
├── summary_figure_cross_model.png       (额外生成)
└── summary_figure_cross_model.pdf       (矢量图)
```

---

## 📝 论文使用建议

### LaTeX引用示例

#### 主实验结果图（完整版）
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=0.95\textwidth]{figures/main_experiment_results.pdf}
\caption{Cross-model and cross-dataset transfer results. 
Baseline methods (PAIR, AutoDAN) shown first, 
followed by SESS-enhanced variants (PAIR+SESS, DAN+SESS).}
\label{fig:main_results}
\end{figure*}
```

#### 简化版（推荐用于正文）
```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.9\textwidth]{figures/main_experiment_results_simple.pdf}
\caption{Main experiment results on the default dataset.}
\label{fig:main_results_simple}
\end{figure}
```

#### 热力图（附录或补充材料）
```latex
\begin{figure*}[p]
\centering
\includegraphics[width=0.95\textwidth]{figures/main_figure_cross_model_dataset.pdf}
\caption{Complete transfer results matrix across all models and datasets.}
\label{fig:heatmap}
\end{figure*}
```

---

## ✅ 验证清单

- [x] 表述统一（DAN+SESS）
- [x] 顺序调整完成
- [x] 配色方案统一
- [x] 高分辨率PNG生成
- [x] 矢量图PDF生成
- [x] 文件格式验证通过
- [x] 文件大小合理
- [ ] LaTeX编译验证（待执行）
- [ ] 论文插入（待执行）

---

## 📊 数据一致性验证

### 关键数据点
```
Qwen3-4B, default数据集:
- PAIR: 55.5%
- AutoDAN: 86.8%
- PAIR+SESS: 79.0%  (+23.5pp vs PAIR)
- DAN+SESS: 100.0%  (+13.2pp vs AutoDAN)

跨族迁移（GPT-OSS-20B, default）:
- PAIR: 0.2%
- AutoDAN: 12.5%
- PAIR+SESS: 12.9%  (+12.7pp vs PAIR)
- DAN+SESS: 30.0%   (+17.5pp vs AutoDAN)
```

✅ 所有数据与表格一致

---

## 🔧 技术细节

### 生成命令
```bash
cd /home/tiger/jailbreak_research/self_evolve_skills_jailbreak/paper/figures
python main_experiment_results.py
python main_figure_cross_model_dataset.py
```

### DPI设置
- 所有图表: 300 DPI
- 适合论文投稿和演示使用

---

*生成完成时间: 2026-07-09 17:47*
*所有图表已验证通过* ✅