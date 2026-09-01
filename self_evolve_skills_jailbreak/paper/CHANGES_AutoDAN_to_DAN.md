# 表述修改总结：AutoDAN+SESS → DAN+SESS

**修改日期**: 2026-07-09
**修改原因**: 明确区分AutoDAN进化方法和固定DAN模板

---

## 修改背景

原来的表述"AutoDAN + SESS"容易引起混淆，因为：
1. **AutoDAN**是一个进化方法，使用遗传算法
2. **DAN模板**是固定的攻击模板（Do Anything Now）
3. 实际使用的是固定DAN模板 + SESS框架，而非AutoDAN进化方法

修改为"DAN + SESS"可以明确：
- 使用固定的DAN模板（6个手工设计的模板）
- 结合SESS的skill系统（检索、维护）
- 不是AutoDAN的进化机制

---

## 已修改的文件

### 1. 表格文件 (paper/tables/)

#### table1_main_results.tex
- ✅ 第1行注释：AutoDAN → DAN
- ✅ 第4行caption：AutoDAN+SESS → DAN+SESS，并说明"DAN+SESS使用固定DAN模板"
- ✅ 第17行方法名：AutoDAN + SESS → DAN + SESS
- ✅ 第21行说明：AutoDAN → DAN+SESS（对比baseline）
- ✅ 第31行注释：强调"固定DAN模板"

#### table2_transfer_models.tex
- ✅ 第22行方法名：AutoDAN + SESS → DAN + SESS
- ✅ 第26行说明文字：AutoDAN+SESS → DAN+SESS
- ✅ 新增注释：说明使用固定DAN模板

#### table3_transfer_datasets.tex
- ✅ 第18行方法名：AutoDAN + SESS → DAN + SESS
- ✅ 第30行注释：强调"固定DAN模板"

#### tableA1_full_results.tex
- ✅ 第4行caption：AutoDAN+SESS → DAN+SESS
- ✅ 所有表格中的方法名：AutoDAN + SESS → DAN + SESS
- ✅ 第70行说明文字：AutoDAN+SESS → DAN+SESS
- ✅ 第82行方法说明：强调"固定DAN模板，而非AutoDAN进化"

### 2. 大纲文件 (paper/outline_v5.md)

- ✅ 第154行表格：autodan_skills → DAN+SESS
- ✅ 第164行说明：强先验（DAN）→ DAN模板作为强先验
- ✅ 第193行方法名：AutoDAN + SESS → DAN + SESS
- ✅ 第197行说明：AutoDAN+SESS → DAN+SESS，增加"使用固定DAN模板"说明
- ✅ 第279行方法名：AutoDAN + SESS → DAN + SESS
- ✅ 第283行表格标题：AutoDAN+SESS → DAN+SESS

### 3. 图表脚本 (paper/figures/)

#### main_bar_chart.py
- ✅ 第3行注释：AutoDAN → DAN
- ✅ 第19行数据键：AutoDAN+SESS → DAN+SESS
- ✅ 第36行标签：AutoDAN+SESS → DAN+SESS
- ✅ 第75-78行计算和说明：autodan → dan，AutoDAN+SESS → DAN+SESS

#### main_experiment_results.py
- ✅ 批量替换所有AutoDAN+SESS → DAN+SESS（使用sed命令）
- ✅ 共替换约20处

#### main_figure_cross_model_dataset.py
- ✅ 批量替换所有AutoDAN+SESS → DAN+SESS（使用sed命令）
- ✅ 共替换约20处

### 4. LaTeX章节文件 (paper/sections/)

**未修改** - 经过检查，sections中的AutoDAN表述都是正确的：
- ✅ introduction.tex: AutoDAN作为baseline方法引用，保持不变
- ✅ related_work.tex: AutoDAN作为相关工作引用，保持不变
- ✅ method.tex: AutoDAN作为baseline方法引用，保持不变
- ✅ experiments.tex: 提到"autodan_skills_54"作为数据源命名，上下文已澄清

---

## 保持不变的部分

### 1. Baseline方法名
- **PAIR**: 保持不变（baseline方法）
- **AutoDAN**: 保持不变（baseline方法）

### 2. 数据源命名
- `pair_skills_28`: 保持不变（数据文件名）
- `autodan_skills_54`: 保持不变（数据文件名）
- 在表格中用"DAN+SESS"表示方法，但数据来源注释中保留原文件名

---

## 表述规范总结

### 方法命名规范
```
✅ 正确：DAN + SESS（固定DAN模板 + SESS框架）
❌ 错误：AutoDAN + SESS（容易误解为使用AutoDAN进化）
```

### 对比表述规范
```
✅ 正确：
- PAIR vs PAIR+SESS（对比基线方法）
- AutoDAN vs DAN+SESS（对比基线方法）
- DAN模板作为强先验，在跨族迁移中效果显著

❌ 避免：
- AutoDAN vs AutoDAN+SESS（混淆baseline和SESS变体）
```

### 文件命名规范
```
数据文件：保持 autodan_skills_*, pair_skills_* （历史数据）
论文表述：使用 DAN+SESS, PAIR+SESS （方法描述）
```

---

## 需要注意的地方

### 1. 图表重新生成
- 修改脚本后，需要重新运行生成PNG/PDF图表
- 命令：`cd paper/figures && python main_bar_chart.py`

### 2. LaTeX编译验证
- 修改表格后，需要重新编译LaTeX验证格式
- 命令：`cd paper && pdflatex main.tex`

### 3. 数据一致性
- 表格中的数据保持不变（只是方法名修改）
- caption和说明文字需要与表格内容一致

---

## 修改效果

### 修改前（混淆）
```
AutoDAN + SESS: 使用DAN模板
→ 读者可能误解为使用AutoDAN进化方法
```

### 修改后（清晰）
```
DAN + SESS: 使用固定DAN模板
→ 明确是固定模板，而非进化方法
AutoDAN: 基线方法（遗传算法进化）
→ baseline方法保持AutoDAN名称
```

---

## 验证清单

- [x] 表格文件修改完成
- [x] 大纲文件修改完成
- [x] 图表脚本修改完成
- [x] 章节文件检查完成（无需修改）
- [ ] 图表重新生成（待执行）
- [ ] LaTeX编译验证（待执行）
- [ ] 全文一致性检查（待执行）

---

**修改完成时间**: 2026-07-09
**修改人**: Claude Code