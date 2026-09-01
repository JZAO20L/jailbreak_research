# SESS Paper — AAAI 2027 Submission

**Title**: SESS: Self-Evolving Skills System for Jailbreak Prompt Generation
**Target**: AAAI 2027 (deadline ~2026-08)
**Last Updated**: 2026-07-13

---

## Current Status

| Item | Status | Notes |
|------|--------|-------|
| LaTeX main file (`SESS.tex`) | ✅ Draft complete | 319 lines, all sections written |
| References (`references.bib`) | ✅ Complete | 12+ references |
| Figures (8 files) | ✅ Complete | System diagram, bar charts, heatmap |
| Tables (12 files) | ✅ Complete | Main results, transfer, ablation |
| AAAI template files | ✅ Present | `aaai2027.sty`, `aaai2027.bst` |
| **LaTeX compilation** | ❌ Not verified | **Next priority** |
| Round 1 review response | ✅ Done | Weak Reject (4.5/10) → v2 revision completed |

---

## Directory Structure

```
SESS/
├── SESS.tex                  # Main LaTeX file (319 lines, all sections)
├── references.bib            # Bibliography (12+ references)
├── aaai2027.sty              # AAAI 2027 style file
├── aaai2027.bst              # AAAI 2027 bibliography style
├── UPDATE_SUMMARY.md         # AutoDAN→DAN naming update summary
├── UPDATE_COMPLETE.md        # Update verification report
├── Figures/                  # 8 files (PNG + PDF pairs)
│   ├── SESS.drawio.png/pdf           # System framework diagram
│   ├── main_experiment_results.png/pdf       # Full bar chart
│   ├── main_experiment_results_simple.png/pdf # Simplified bar chart
│   └── main_figure_cross_model_dataset.png/pdf # Transfer heatmap
├── Tables/                   # 12 LaTeX tables
│   ├── table1_main_results.tex       # Same-model performance
│   ├── table2_transfer_models.tex    # Cross-model transfer
│   ├── table3_transfer_datasets.tex  # Cross-dataset transfer
│   ├── table4_ablation.tex           # Component contribution
│   ├── tableA1_full_results.tex      # Appendix: full result matrix
│   ├── layer1_ablation.tex           # Layer 1 detailed ablation
│   ├── layer2_data.tex               # Layer 2 data strategy
│   ├── layer3_dan.tex                # Layer 3-4 DAN template
│   ├── evolution_ablation.tex        # Evolution ablation
│   ├── transfer_models.tex           # English version: cross-model
│   ├── transfer_datasets.tex         # English version: cross-dataset
│   └── UPDATE_SUMMARY.md
└── appendix/                 # Empty, reserved for appendix assembly
```

---

## Key Results (Verified, All Consistent)

### Main Performance (Qwen3-4B, default dataset)
| Method | ASR | Δ vs baseline |
|--------|-----|---------------|
| PAIR | 55.5% | — |
| PAIR + SESS | 79.0% | **+23.5pp** |
| AutoDAN | 86.8% | — |
| DAN + SESS | 100.0% | **+13.2pp** |

### Component Ablation
| Component | ASR | Δ |
|-----------|-----|---|
| PAIR baseline | 55.5% | — |
| + CS + Retrieval + Maintenance | 78.2% | **+22.7pp** (core contribution) |
| + Evolution | 79.1% | +0.9pp (marginal refinement) |

### Cross-Family Transfer (GPT-OSS-20B)
| Method | ASR |
|--------|-----|
| PAIR | 0.2% |
| AutoDAN | 12.5% |
| PAIR + SESS | 12.9% |
| **DAN + SESS** | **30.0%** (+17.5pp over AutoDAN) |

---

## Paper Narrative (v2 Positioning)

**Core contribution**: Reusable skill framework (Cold Start + Retrieval + Maintenance) — +22.7pp
**Secondary finding**: Self-evolution under high baseline — +0.9pp marginal refinement
**Key insight**: Structural attack patterns (DAN templates) dominate over semantic refinement

> ⚠️ **Naming convention**: "DAN+SESS" = fixed DAN template + SESS framework. NOT "AutoDAN+SESS" (which confused the evolutionary method with the fixed template). Updated 2026-07-09.

---

## TODO (Priority Order)

### 🔴 Critical (This Week)
1. **LaTeX compilation test** — compile with `aaai2027_template`, fix any errors
2. **Embed Figure 1** — system framework diagram into §3.1
3. **Embed all tables** — Tables 1-4 into §4, Table A1 into Appendix
4. **Algorithm 1 pseudocode** — SESS pipeline algorithm

### 🟡 Important (Next Week)
5. **Computational cost reporting** — LLM call counts, time, token usage
6. **Case Study** —典型成功/失败案例 for Appendix C
7. **Full consistency check** — data, citations, cross-references

### 🟢 Nice-to-Have (Before Submission)
8. **Statistical analysis** — multi-seed variance, confidence intervals
9. **Cross-family model expansion** — LLaMA, Mistral evaluation
10. **Missing baselines** — TAP, GAP, AutoDAN-Turbo comparison
11. **Defense evaluation** — perplexity filtering, safety-tuned models
12. **Skill diversity analysis** — t-SNE visualization

---

## Review History

### Round 1 (2026-06-30)
- **Result**: Weak Reject (4.5/10)
- **Reviewers**: 4 (method, experiment, writing, meta)
- **Key Issues**:
  1. "Self-evolving" claim contradicted by results (Evo only +0.9pp)
  2. Missing Metis/ASTRA/EvoSynch comparison
  3. Insufficient cross-family transfer (only 1 model)
  4. No statistical analysis
  5. Evaluation validity (policy and target same model)

### v1→v2 Revision (2026-06-30)
- ✅ Narrative reframed: "self-evolving" → "skill system"
- ✅ Title adjusted (kept SESS acronym)
- ✅ Evaluation validity caveat added
- ✅ Qualitative comparison table added (§5.5)
- ✅ Limitation section added (§5.6)
- ✅ Writing fixes (language, figure references, structure)

---

## Related Files (Outside This Directory)

- **Outline**: `../outline_v5.md` — full paper outline with all sections
- **Progress tracking**: `../README.md` — paper writing progress log
- **Experiment data**: `../../exp/layer{1-4}/`, `../../exp/transfer/`
- **Method doc**: `../../MECHANISM.md`
- **Thesis structure**: Memory: `project/thesis_structure.md`

---

*README created: 2026-07-13*
