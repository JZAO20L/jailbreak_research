# Midterm Report - Key Data Tables for Visualization

**Generated**: 2026-06-16  
**Purpose**: Centralized data reference for all figures in the midterm report  
**Color Scheme**: Blue system (primary: #3B82F6, secondary: #60A5FA, light: #93C5FD)  
**Language**: English only

---

## Table of Contents

1. [Chapter 1: AHR-GRPO Data](#chapter-1-ahr-grpo-data)
2. [Chapter 2: SESS Data](#chapter-2-sess-data)
3. [Cross-Chapter Comparison Data](#cross-chapter-comparison-data)
4. [Figure Index](#figure-index)

---

## Chapter 1: AHR-GRPO Data

### Table 1.1: Jailbreak Prompt Selection (Experiment 1)

**Purpose**: Figure 1 - Horizontal bar chart (Top-8 strategies)  
**Test Config**: Qwen3-4B, 1000 samples

| Rank | Strategy | ASR (%) | Refusal (%) | Category | Status |
|:---:|----------|:---:|:---:|----------|---------|
| 1 | hypothetical_scenario | 30.8 | 69.2 | Scenario | ✅ Selected |
| 2 | creative_writing | 28.3 | 71.7 | Scenario | ✅ Selected |
| 3 | role_playing | 25.0 | 75.0 | Role-play | ✅ Selected |
| 4 | red_teaming | 22.4 | 77.6 | Logic | Reference |
| 5 | urgent_situation | 21.9 | 78.1 | Psychological | Reference |
| 6 | journalistic_investigation | 20.9 | 79.1 | Scenario | - |
| 7 | academic_research | 20.2 | 79.8 | Logic | - |
| 8 | technical_documentation | 19.9 | 80.1 | Logic | - |

---

### Table 1.2: Judge Dimension GRPO Results (Experiment 2)

**Purpose**: Figure 2 - Grouped bar chart (12 experiment groups)  
**Baseline**: hypothetical_scenario ASR = 30.8%

| Rank | Strategy | Judge Dimension | ASR (%) | Δ vs Baseline |
|:---:|----------|-----------------|:---:|:---:|
| 1 | hypothetical_scenario | idea_preservation | 32.3 | +1.5 |
| 2 | hypothetical_scenario | naturalness | 31.8 | +1.0 |
| 3 | hypothetical_scenario | hypothetical_scenario | 31.2 | +0.4 |
| 4 | hypothetical_scenario | stealthiness | 31.0 | +0.2 |
| 5 | creative_writing | naturalness | 29.5 | +1.2 |
| 6 | creative_writing | idea_preservation | 29.0 | +0.7 |
| 7 | creative_writing | creative_writing | 28.8 | +0.5 |
| 8 | creative_writing | stealthiness | 28.5 | +0.2 |
| 9 | role_playing | idea_preservation | 26.8 | +1.8 |
| 10 | role_playing | role_playing | 26.5 | +1.5 |
| 11 | role_playing | naturalness | 26.2 | +1.2 |
| 12 | role_playing | stealthiness | 26.0 | +1.0 |

#### Table 1.2.1: Judge Dimension Type Summary

| Dimension Type | Avg Δ (%) | Best Δ (%) | Stability |
|----------------|:---:|:---:|-----------|
| idea_preservation (General) | +1.2 | +1.8 | ✅ Most Stable |
| naturalness (General) | +1.0 | +1.2 | ✅ Stable |
| stealthiness (General) | +0.8 | +1.0 | ✅ Effective |
| Strategy-specific | +0.6 | +1.0 | ⚠️ Variable |

---

### Table 1.3: Adaptive vs Fixed Weight (Experiment 3)

**Purpose**: Figure 3 - Bar chart comparison  
**Baseline**: 30.8% | **Exp 2 Best**: 32.3%

| Method | Reward Config | EMA (β) | ASR (%) | Δ vs Baseline | Δ vs Exp 2 |
|--------|---------------|---------|:---:|:---:|:---:|
| Baseline | No Training | - | 30.8 | - | - |
| Fixed Weight | ASR+Judge (1:1) | - | 32.3 | +1.5 | 0.0 |
| Adaptive | ASR+Judge Adaptive | 0.0 | **33.2** | **+2.4** | **+0.9** |
| Adaptive | ASR+Judge Adaptive | 0.8 | 32.9 | +2.1 | +0.6 |
| Adaptive | ASR+Judge Adaptive | 0.9 | 32.6 | +1.8 | +0.3 |

#### Table 1.3.1: Reward Type Ablation

| Reward Type | ASR (%) | Δ vs Baseline | Refusal (%) | Evaluation |
|-------------|:---:|:---:|:---:|------------|
| Only ASR Reward | 25.0 | -5.8 | 75.0 | ⚠️ Sparse signal |
| Only Judge Reward | 28.5 | -2.3 | 71.5 | ⚠️ Off-target |
| Hybrid Fixed (1:1) | 32.3 | +1.5 | 67.7 | ✅ Exp 2 Best |
| **Hybrid Adaptive** | **33.2** | **+2.4** | **66.2** | ✅ **Best** |
| Baseline | 30.8 | - | 69.2 | Reference |

---

### Table 1.4: Baseline Method Comparison

| Method | Type | ASR (%) | Δ vs Baseline | Cost |
|--------|------|:---:|:---:|--------|
| PAIR | Multi-turn | 91.8 | +61.0 | High (1000+ API) |
| Genetic | Genetic | 47.9 | +17.1 | High |
| DeepInception | Single-turn | 37.3 | +6.5 | Low |
| **AHR-GRPO (Adaptive)** | **RL Rewrite** | **33.2** | **+2.4** | **Low (500 steps)** |
| AHR-GRPO (Fixed) | RL Rewrite | 32.3 | +1.5 | Low (500 steps) |
| Baseline | Prompt Template | 30.8 | - | Zero |
| Multilingual | Encoding | 2.1 | -28.7 | Low |

---

## Chapter 2: SESS Data

### Table 2.1: Layer 1 - Method Combination Grid Search (16 groups)

**Purpose**: Part of Figure 6 - Layer comparison  
**Goal**: Determine best skill call & update strategy

| Rank | skill_call_mode | skill_extraction_mode | update_strategy | ASR (%) | Skills Count |
|:---:|-----------------|----------------------|-----------------|:---:|:---:|
| 1 | single_call | trajectory | statistical | **79.1** | 28 |
| 2 | single_call | trajectory | success_only | 78.8 | 35 |
| 3 | single_call | final_prompt | success_only | 77.7 | 42 |
| 4 | single_call | final_prompt | statistical | 75.6 | 30 |
| 5-8 | single_call | various | failure_only/both | 69.9-73.2 | 30-300+ |
| 9-16 | every_iteration | various | various | 65.3-72.4 | various |

#### Table 2.1.1: skill_call_mode Comparison

| Mode | Avg ASR (%) | Best ASR (%) | Gap |
|------|:---:|:---:|:---:|
| **single_call** | **74.7** | **79.1** | baseline |
| every_iteration | 66.4 | 72.4 | **-12.7** |

#### Table 2.1.2: update_strategy Comparison

| Strategy | Avg ASR (%) | Skills Trend | Recommendation |
|----------|:---:|---------|----------------|
| **statistical** | **69.7** | Stable | ✅ Recommended |
| success_only | 73.6 | Moderate growth | ✅ Best effect |
| failure_only | 69.5 | Rapid growth | ⚠️ Inflation |
| both | 69.4 | Explosion (>300) | ❌ Not recommended |

---

### Table 2.2: Layer 2 - Data Strategy Ablation (36 groups)

**Purpose**: Part of Figure 6  
**Goal**: Verify data size & ratio effects

| Rank | Method | Data Size | Ratio | ASR (%) |
|:---:|--------|-----------|-------|:---:|
| 1 | trajectory+statistical | **small(300)** | **early(30%CS)** | **80.6** |
| 2 | trajectory+statistical | large(1000) | early(30%CS) | 80.0 |
| 3 | trajectory+statistical | medium(500) | early(30%CS) | 79.8 |

#### Table 2.2.1: Data Size Impact

| Size | Avg ASR (%) | Gap | Conclusion |
|------|:---:|:---:|------------|
| **small (300)** | **79.5** | baseline | ✅ Recommended |
| medium (500) | 77.7 | -1.8 | Usable |
| large (1000) | 78.8 | -0.7 | Usable |

#### Table 2.2.2: Ratio Strategy Impact

| Ratio | CS Ratio | Evo Ratio | Avg ASR (%) | Conclusion |
|-------|:---:|:---:|:---:|------------|
| **early** | **30%** | **70%** | **80.0** | ✅ Recommended |
| balanced | 20% | 80% | 78.9 | Usable |
| evo | 10% | 90% | 77.6 | ⚠️ Decline |

---

### Table 2.3: Layer 3 - DAN Template Verification (17 groups)

**Purpose**: Part of Figure 6  
**Goal**: Verify strong initial Skills (DAN templates)

| Rank | Config | CS Ratio | ASR (%) | Final Skills |
|:---:|--------|:---:|:---:|:---:|
| 1 | **DAN + full_evolve** | **0%** | **98.8** | 6 |
| 2 | DAN + balanced | 20% | 92.7 | 27-65 |
| 3 | DAN + early | 30% | 89.9 | 26-32 |

#### Table 2.3.1: DAN Template Usage Distribution

| Template | Usage Count | Success Count | Success Rate (%) |
|----------|:---:|:---:|:---:|
| **dan_mode** | **394** | **394** | **100** |
| **mcpt** | **78** | **78** | **100** |
| devil | 28 | 24 | 85.7 |

---

### Table 2.4: Layer 4 - DAN Data Strategy (12 groups)

**Purpose**: Part of Figure 6

| Rank | Data Size | Ratio | ASR (%) |
|:---:|-----------|-------|:---:|
| 1 | medium(500) | evo(10%CS) | **99.7** |
| 2 | small(300) | full_evolve | **99.7** |
| 3 | medium(500) | full_evolve | 99.6 |

**Key Finding**: Data size impact < 1.3%, small(300) is sufficient

---

### Table 2.5: Evolution Necessity Ablation (16 groups)

**Purpose**: Part of Figure 6

| Method Combination | No-Evolution ASR (%) | Full Pipeline ASR (%) | Evolution Contribution (%) |
|--------------------|:---:|:---:|:---:|
| every_iteration + trajectory | 62.6 | 67.5 | +4.9 |
| **single_call + trajectory** | **75.6** | **76.1** | **+0.5** |
| Average | 69.6 | 70.6 | **+0.9** |

**Conclusion**: Evolution average contribution only +0.9%, can be simplified

---

### Table 2.6: Cross-Model Transfer (140 groups)

**Purpose**: Figure 7 - Grouped bar chart  
**Goal**: Verify Skills transferability

#### Table 2.6.1: Same-Family Transfer (Qwen Series)

| Model | PAIR_w_SESS ASR (%) | Best Baseline ASR (%) | Improvement (%) |
|-------|:---:|:---:|:---:|
| Qwen3-0.6B | **95.1** | 90.2 | **+4.9** |
| Qwen3-4B | **86.7** | 78.7 | **+8.0** |
| Qwen3-14B-FP8 | **86.5** | 85.5 | **+1.0** |

**Average improvement over baseline: +4.7%**

#### Table 2.6.2: Cross-Family Transfer (gpt-oss-20b)

| Method | Same-Family ASR (%) | Cross-Family ASR (%) | Decline (%) |
|--------|:---:|:---:|:---:|
| **PAIR_w_SESS** | **91.6** | **11.4** | **-80.2** |
| **AutoDAN_w_SESS** | **~99** | **32.5** | **-66.5** |
| AutoDAN | 85.5 | 6.0 | -79.5 |
| PAIR | 71.6 | 0.08 | -71.5 |

**Key Finding**: Cross-family transfer is the real challenge

---

### Table 2.7: SESS Best Configurations Summary

| Layer | Best Config | Best ASR (%) | Skills Count | Key Finding |
|:---:|-------------|:---:|:---:|-------------|
| **Layer 1** | single_call + trajectory + statistical | **79.1** | 28 | single_call dominant |
| **Layer 2** | trajectory + statistical + small + early | **80.6** | 28-35 | Ratio > Data size |
| **Layer 3** | DAN + full_evolve | **98.8** | 6 | DAN needs no evolution |
| **Layer 4** | medium + evo | **99.7** | 54 | Data size negligible |

---

## Cross-Chapter Comparison Data

### Table 3.1: Core Contributions Comparison

| Chapter | Core Method | Main Contribution | Best ASR | vs Baseline | vs Exp 2 |
|---------|-------------|-------------------|:---:|:---:|:---:|
| **Ch 1** | **Adaptive Hybrid-Reward GRPO** | Process-level Reward ✅<br/>General Judge ✅<br/>Adaptive Weight ✅ | **33.2%** | **+2.4%** | **+0.9%** |
| **Ch 2** | **Self-Evolving Skills System** | Skills Reuse ✅<br/>DAN Template ✅<br/>Same-Family Transfer ✅ | **99.7%** | **+68.9%** | - |

### Table 3.2: Experiment Scale & Resources

| Metric | Chapter 1 | Chapter 2 | Total |
|--------|:---:|:---:|:---:|
| **Experiment Count** | 55 | 217 | **272** |
| **Training Time** | ~45 GPU-h | ~133 GPU-h | ~178 GPU-h |
| **Best ASR** | **33.2%** | **99.7%** | - |
| **Baseline ASR** | 30.8% | 30.8% | - |
| **Improvement** | **+2.4%** | **+68.9%** | - |

### Table 3.3: Method Recommendation Matrix

| Scenario | Recommended Method | ASR (%) | vs Baseline | Recommendation |
|----------|-------------------|:---:|:---:|----------------|
| Same-Family Attack | **AutoDAN_w_SESS** | **99** | **+68.9%** | ✅ Strongly Recommended |
| Adaptive Hybrid-Reward | **AHR-GRPO (Adaptive)** | **33.2** | **+2.4%** | ✅ Strongly Recommended |
| Fixed Weight Attack | AHR-GRPO (Fixed) | 32.3 | +1.5% | ✅ Recommended |
| Quick Experiment | PAIR_w_SESS + small(300) | 79-95 | +48-65% | ✅ Recommended |
| Cross-Family Transfer | AutoDAN_w_SESS | 32.5 | +1.7% | ⚠️ Limited Effect |

---

## Figure Index

| Figure # | Title | Data Source | Chart Type | Status |
|----------|-------|-------------|------------|--------|
| **Fig 1** | Jailbreak Prompt Selection (Top-8) | Table 1.1 | Horizontal Bar | ✅ |
| **Fig 2** | Judge Dimension GRPO Results (12 groups) | Table 1.2 | Grouped Bar | ✅ |
| **Fig 3** | Adaptive vs Fixed Weight | Table 1.3 | Bar Chart | ✅ |
| **Fig 4** | Reward Type Ablation | Table 1.3.1 | Bar Chart | ✅ |
| **Fig 5** | AHR-GRPO Progressive Summary | Table 1.3 | Line + Bar | ✅ |
| **Fig 6** | SESS Layers Comparison | Table 2.7 | Bar Chart | ✅ |
| **Fig 7** | Transfer: Same-Family vs Cross-Family | Table 2.6.2 | Grouped Bar | ✅ |
| **Fig 8** | Best Config Summary (Both Chapters) | Table 3.1 | Subplot Bars | ✅ |

---

**Document Maintainer**: Research Team  
**Last Updated**: 2026-06-16  
**Next Update**: After Chapter 3 experiments
