# 大语言模型越狱攻击与防御方法研究 - LaTeX 项目

## 项目结构

```
latex_project/
├── main.tex              # 主 LaTeX 文档
├── references.bib        # 参考文献
└── figures/              # 图片目录
    ├── AHR_GRPO_01_attack_prompt_selection.png
    ├── AHR_GRPO_02_judge_prompt_selection.png
    ├── AHR_GRPO_03_judge_dimension.png
    ├── AHR_GRPO_04_adaptive_weight.png
    ├── AHR_GRPO_05_reward_ablation.png
    ├── AHR_GRPO_06_summary.png
    ├── SESS_01_sess_main_results.png
    ├── SESS_02_transfer_comparison.png
    ├── SESS_03_transfer_comprehensive.png
    └── SESS_04_same_family_transfer.png
```

## 使用方法

### 本地编译

```bash
# 使用 XeLaTeX 编译（推荐，支持中文）
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex

# 或者使用 latexmk
latexmk -xelatex main.tex
```

### Overleaf 使用

1. 将整个 `latex_project` 文件夹上传到 Overleaf
2. 在 Overleaf 设置中选择 **XeLaTeX** 作为编译器
3. 点击 "Recompile" 即可

## 文档说明

- **文档类型**：article
- **语言**：中文（使用 ctex 宏包）
- **页面设置**：A4 纸，2.5cm 边距
- **图表**：10 张 PNG 格式图片
- **参考文献**：使用 BibTeX 管理

## 注意事项

1. 确保图片路径正确：`figures/` 目录与 `main.tex` 同级
2. 使用 XeLaTeX 编译以支持中文字体
3. 首次编译可能需要下载中文字体包
