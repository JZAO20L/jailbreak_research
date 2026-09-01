# SESS 论文引用文献清单

**状态**: ✅ = 已有完整 BibTeX | ⚠️ = 需要补充/核实 | ❌ = 缺失

---

## 一、Jailbreak 攻击方法（核心引用）

| # | 文献 | BibTeX Key | 状态 | 说明 |
|---|------|------------|------|------|
| 1 | PAIR (Chao et al., 2023) | `chao2023pair` | ✅ | arXiv:2310.08419 |
| 2 | TAP (Mehrotra et al., 2023) | `mehrotra2023tap` | ✅ | arXiv:2312.02119 |
| 3 | AutoDAN (Liu et al., 2023) | `liu2023autodan` | ✅ | arXiv:2310.04451 |
| 4 | AutoDAN-Turbo (Liu et al., 2024) | `liu2024autodanturbo` | ✅ | arXiv:2405.06479 |
| 5 | GAP (Guo et al., 2024) | `guo2024gap` | ✅ | arXiv:2402.11852 |
| 6 | DeepInception (Li et al., 2023) | `li2023deepinception` | ✅ | arXiv:2311.07588 |
| 7 | PromptAgent (Wang et al., 2024) | `wang2024promptagent` | ✅ | arXiv:2310.16428 |
| 8 | **Crescendo** | — | ❌ **缺失** | outline_v4 中提到，需补充 BibTeX |
| 9 | **Persona-based attacks** | — | ❌ **缺失** | 论文中作为 baseline，需找到原始文献或标注为内部方法 |

---

## 二、自进化攻击策略（核心对比）

| # | 文献 | BibTeX Key | 状态 | 说明 |
|---|------|------------|------|------|
| 10 | Metis (Chen et al., 2026) | `chen2026metis` | ✅ | arXiv:2605.10067 |
| 11 | ASTRA (Liu et al., 2026) | `liu2026astra` | ✅ | arXiv:2511.02356, ACL 2026 |
| 12 | EvoSynth (Chen et al., 2025) | `chen2025evosynth` | ✅ | arXiv:2511.12710 |
| 13 | RedHit (Sorkhpour et al., 2025) | `sorkhpour2025redhit` | ✅ | ACL Anthology: 2025.llmsec-1.2 |
| 14 | Active Attacks (Yun et al., 2025) | `yun2025active` | ✅ | arXiv:2509.21947 |

---

## 三、RL-based Jailbreak（Related Work 简要提及）

| # | 文献 | BibTeX Key | 状态 | 说明 |
|---|------|------------|------|------|
| 15 | Jailbreak-R1 (Guo et al., 2025) | `guo2025jailbreakr1` | ✅ | arXiv:2506.00782 |
| 16 | TROJail (Xiong et al., 2026) | `xiong2026trojail` | ✅ | ACL 2026 Main Conference |
| 17 | xJailbreak (Lee et al., 2025) | `lee2025xjailbreak` | ✅ | arXiv:2501.16727 |
| 18 | RL-MTJail (Chen et al., 2025) | `chen2025rlmtjail` | ✅ | arXiv:2512.07761 |
| 19 | Many-Turn Jailbreaking (Yang et al., 2025) | `yang2025manyturn` | ✅ | arXiv:2508.06755 |

---

## 四、Red Teaming & Adaptive Attacks（Related Work 简要提及）

| # | 文献 | BibTeX Key | 状态 | 说明 |
|---|------|------------|------|------|
| 20 | Automatic LLM Red Teaming (Belaire et al., 2025) | `belaire2025automatic` | ✅ | arXiv:2508.04451 |
| 21 | Genesis (Zhang et al., 2025) | `zhang2025genesis` | ✅ | arXiv:2510.18314 |
| 22 | Learning-Based Red-Teaming (Wei et al., 2025) | `wei2025learning` | ✅ | arXiv:2512.20677 |
| 23 | ~~LLM-Virus~~ | — |  **已删除** | 疑似幻觉引用，已移除 |

---

## 五、Benchmarks & Datasets

| # | 文献 | BibTeX Key | 状态 | 说明 |
|---|------|------------|------|------|
| 24 | Universal Adversarial Attacks / AdvBench (Zou et al., 2023) | `zou2023universal` | ✅ | arXiv:2307.15043 |
| 25 | HarmBench (Chao et al., 2024) | `chao2024harmbench` | ✅ | arXiv:2402.04249 |
| 26 | JailbreakBench (Caswell et al., 2024) | `caswell2024jailbreakbench` | ✅ | arXiv:2404.01318 |
| 27 | WildTeaming (Jiang et al., 2024) | `wildteaming2024` | ✅ | arXiv:2406.18510 |

---

## 六、LLM Safety & Alignment（背景引用）

| # | 文献 | BibTeX Key | 状态 | 说明 |
|---|------|------------|------|------|
| 28 | RLHF (Ouyang et al., 2022) | `ouyang2022rlhf` | ✅ | NeurIPS 2022 |
| 29 | Constitutional AI (Bai et al., 2022) | `bai2022constitutional` | ✅ | arXiv:2212.08073 |

---

## 七、Models & Infrastructure

| # | 文献 | BibTeX Key | 状态 | 说明 |
|---|------|------------|------|------|
| 30 | Qwen3 Technical Report (Yang et al., 2025) | `yang2025qwen3` | ✅ | arXiv:2505.09388 |
| 31 | Qwen3Guard Technical Report (Qwen Team, 2025) | `qwen2025qwen3guard` | ✅ | arXiv:2510.14276 |
| 32 | GPT-OSS Model Card (OpenAI, 2025) | `openai2025gptoss` | ✅ | arXiv:2508.10925 |
| 33 | vLLM (Kwon et al., 2023) | `kwon2023efficient` | ✅ | SOSP 2023 |

---

## 八、RL Foundations（可选引用）

| # | 文献 | BibTeX Key | 状态 | 说明 |
|---|------|------------|------|------|
| 34 | GRPO / DeepSeekMath (Shao et al., 2024) | `shao2024grpo` | ✅ | arXiv:2402.03300 |

---

## 需要补充的 BibTeX（优先级排序）

### 🔴 高优先级（正文直接引用）

1. **Crescendo** — 多轮渐进式 jailbreak 攻击
   - 搜索关键词: "Crescendo jailbreak LLM multi-turn"
   
2. ~~**WildTeaming**~~ — ✅ 已添加 (arXiv:2406.18510, `wildteaming2024`)

3. **Persona-based attacks** — 角色扮演攻击
   - 搜索关键词: "persona jailbreak LLM role-playing"
   - 如果是内部方法，需标注为 "internal implementation"

### 🟡 中优先级（Related Work 提及）

4. **LLM-Virus** — arXiv 号不完整 (2601.0xxxx)，需核实

###  低优先级（可选）

无

---

## 当前 references.bib 统计

- **总计**: 31 条（含 2 条占位注释）
- **已有完整 BibTeX**: 31 条
- **需要补充**: 2 条（Crescendo, Persona）
- **需要核实**: 无

---

**最后更新**: 2026-07-07  
**建议**: 补充 Crescendo 和 Persona 的 BibTeX 即可完成引用整理。
