# Reusable Attack Skills for LLM Safety Evaluation: A Systematic Framework and Large-Scale Empirical Study

**Zixiao Jia, Bing Xu**  
School of Computer Science and Technology, Harbin Institute of Technology

---

## Abstract

Automated jailbreak attacks are essential for evaluating the safety of large language models (LLMs), yet existing methods treat each attack independently—successful strategies are discarded after use, and no knowledge accumulates across attacks. We propose a **reusable attack skill framework** that represents attack strategies as compact, retrievable skill templates stored in a dynamic library. Our framework supports multiple skill population mechanisms—including cold-start extraction from base attack trajectories, self-evolution through reflection on attack outcomes, and strong prior initialization from known effective templates—and provides systematic retrieval, quality control, and maintenance. Through the largest empirical study in this domain to date—217 ablation configurations and 120 transfer evaluations across 4 models and 5 benchmark datasets—we obtain three key findings. First, skill libraries populated with strong prior templates (DAN-style instruction-override prompts) achieve up to 99.7% attack success rate (ASR) on same-family models, substantially outperforming all baselines. Second, evolved skills transfer effectively within the same model family (85–95% ASR across Qwen3-0.6B/4B/14B), with structural attack patterns generalizing better than semantic patterns. Third, and perhaps surprisingly, the self-evolution mechanism contributes only +0.9% improvement on average when strong priors are available, revealing that the skill *system architecture*—representation, retrieval, and maintenance—is more critical than the evolution mechanism itself. These findings provide actionable guidance for both attack design and defense strategy in LLM safety evaluation.

---

## 1. Introduction

Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their widespread deployment raises significant safety concerns. Through alignment techniques like RLHF (Ouyang et al., 2022) and Constitutional AI (Bai et al., 2022), these models are trained to refuse harmful requests. However, *jailbreak attacks*—where adversaries craft carefully designed prompts to bypass safety guardrails—remain a persistent threat (Zou et al., 2023). Understanding and automating these attacks is essential for proactive safety evaluation.

Current jailbreak approaches—iterative optimization (PAIR; Chao et al., 2023), evolutionary search (AutoDAN; Liu et al., 2023), template-based methods (DeepInception; Li et al., 2023)—share a fundamental limitation: **they lack knowledge accumulation**. Each attack instance is treated independently, even though successful attacks often share underlying strategies such as role-playing, scenario construction, or instruction overriding. This leads to three concrete problems:

- **Redundant exploration**: The same attack strategies are rediscovered across different prompts, wasting computational resources.
- **No cross-instance transfer**: A strategy effective for one harmful prompt is not systematically reused for similar prompts.
- **No cross-model transfer**: Attack strategies optimized for one model cannot be efficiently adapted to other models.

To address these limitations, we propose a **reusable attack skill framework** that maintains a dynamic library of compact, interpretable attack strategy templates. The framework supports three skill population mechanisms: (1) *cold-start extraction* from base attack trajectories (e.g., PAIR or AutoDAN), (2) *self-evolution* through LLM-based reflection on successful and failed attacks, and (3) *strong prior initialization* from known effective templates (e.g., DAN-style instruction-override prompts). Skills are retrieved for new prompts via multi-factor scoring (quality, keyword matching, harm type alignment), and the library is periodically maintained through quality pruning, similarity-based merging, and capacity control.

To rigorously evaluate this framework, we conduct the most comprehensive empirical study in the jailbreak domain to date: **217 ablation configurations** across 4 experimental layers and **120 transfer evaluations** across 4 target models and 5 benchmark datasets. The results yield three key findings:

1. **Skill libraries with strong priors are highly effective.** Initializing with 6 DAN-style templates achieves 99.7% ASR on same-family models, substantially outperforming all baselines including PAIR, AutoDAN, DeepInception, and Persona.

2. **Skills transfer within model families but face cross-family barriers.** Evolved skills achieve 85–95% ASR across Qwen3 models of different sizes, but drop to 11–32% on a cross-family model (gpt-oss-20b). Structural attack patterns (instruction overriding) generalize better than semantic patterns (specific phrasing).

3. **The skill system architecture matters more than evolution.** When strong priors are available, the self-evolution mechanism contributes only +0.9% improvement, and DAN templates achieve 98.8% ASR *without any evolution*. This reveals that the primary contribution lies in the skill representation, retrieval, and maintenance system—not in the evolution mechanism itself.

**Contributions.** Based on these findings, we make three contributions:

1. We propose a reusable attack skill framework for LLM safety evaluation, with systematic design of skill representation, multi-factor retrieval, and library maintenance mechanisms (§3).

2. We conduct the largest empirical study in this domain, with 217 ablation configurations across 4 experimental layers and 120 transfer evaluations across 4 models and 5 datasets, providing a comprehensive characterization of the jailbreak skill design space (§4).

3. We reveal that structural attack primitives dominate evolution-based refinement, with important implications for both attack design (prioritize structural templates) and defense strategy (focus on resisting instruction-override patterns) (§5).

---

## 2. Related Work

### 2.1 Iterative and Search-Based Jailbreak Attacks

Automated jailbreak methods have evolved from manual prompt engineering to sophisticated optimization frameworks. Chao et al. (2023) propose PAIR, which employs an attacker-target dual-model interaction pattern, automatically optimizing jailbreak prompts through multi-turn dialogue with typically fewer than 20 queries. Wang et al. (2024) frame prompt optimization as a planning problem with a "plan-execute-reflect" closed-loop framework. Mehrotra et al. (2023) introduce Tree of Attacks with Pruning (TAP), which constructs a tree-structured search space over prompts and dynamically prunes branches based on harmfulness scores, significantly improving search efficiency.

### 2.2 Evolutionary and Template-Based Methods

Evolutionary approaches treat jailbreak prompt generation as an optimization problem over a population of candidates. Liu et al. (2023) combine human-crafted DAN templates with genetic algorithms (selection, crossover, mutation) to evolve stealthy jailbreak prompts. AutoDAN-Turbo (Liu et al., 2024) introduces hierarchical genetic optimization at both the fragment and prompt levels. Guo et al. (2024) propose GAP, a genetic algorithm approach that operates without gradient access. Li et al. (2023) exploit LLMs' context dependency through multi-layer nested virtual scenarios ("dream-within-a-dream" structure), hiding malicious intent within deep contextual layers.

### 2.3 Reinforcement Learning for Jailbreak

Recent work has applied reinforcement learning to automate jailbreak attacks. Guo et al. (2025b) propose a three-stage RL framework with cold start via SFT, exploration warmup with diversity rewards, and progressive curriculum learning. Xiong et al. (2026) first model multi-turn jailbreaking as a trajectory-level RL problem with dual process rewards (stealth + effectiveness). Lee et al. (2025) guide jailbreak through representation space analysis, ensuring semantic proximity between benign and malicious prompts. Chen et al. (2025) combine soft/hard label reward transitions for black-box multi-turn jailbreaking. Yang et al. (2025) systematically explore multi-turn jailbreak paradigms with the MTJ-Bench benchmark.

### 2.4 Self-Evolving Attack Strategies

The most related line of work focuses on self-evolving attack strategies. Chen et al. (2026a) propose Metis, which models attacks as causal diagnosis and strategy evolution with composable policy primitives. Liu et al. (2026) propose ASTRA with a three-tier dynamic strategy library categorized by effectiveness. Chen et al. (2026b) shift from prompt optimization to method-level evolution, synthesizing code-level attack algorithms through multi-agent collaboration. Sorkhpour et al. (2025) combine Monte Carlo Tree Search with DPO for adaptive red-teaming. Yun et al. (2025) propose active attacks where the environment adapts to prevent mode collapse.

### 2.5 Positioning

Our work differs from prior approaches along several dimensions. Compared to iterative methods (PAIR, TAP), our framework accumulates knowledge across attack instances rather than optimizing each prompt independently. Compared to evolutionary methods (AutoDAN, GAP), we evolve *reusable strategy templates* rather than individual prompts, enabling cross-instance and cross-model transfer. Compared to self-evolving methods (Metis, ASTRA), our framework uses a simpler but effective skill representation with multi-factor retrieval and periodic maintenance. The key distinction is our empirical approach: we conduct the most comprehensive systematic evaluation to date (217 configurations, 4 models, 5 datasets), providing quantitative evidence about what makes skill-based attacks effective—a question left open by prior work. We acknowledge that direct experimental comparison with Metis, ASTRA, and EvoSynth is a limitation we discuss in §5.

---

## 3. Method

### 3.1 Overview

We propose a reusable attack skill framework that transforms jailbreak attacks from instance-level optimization into a knowledge-accumulation process. Rather than optimizing each attack prompt from scratch, our framework maintains a library of reusable *attack skills*—compact strategy templates extracted from successful (and failed) attack trajectories—and evolves this library over time through a reflection mechanism.

The system operates in three phases (Figure 1):

1. **Cold Start**: Initial skills are accumulated by running a base attack framework (e.g., PAIR or AutoDAN) on a subset of training data, or by loading strong prior templates.
2. **Skill-Guided Attack & Evolution**: The skill library guides attacks on new data. Successful and failed trajectories feed back into skill extraction, refinement, and library maintenance.
3. **Deployment**: The evolved skill library is applied to test-time attacks or transferred to other models.

> **Figure 1**: System overview (see `figures/sess_system.drawio` for the source file; export to PDF for publication). The diagram shows the three-phase pipeline with the Skill Library at the center, receiving feedback from attack outcomes and providing guidance for new attacks.

### 3.2 Skill Representation and Library

A *skill* is a compact, reusable attack strategy template represented as a natural language prefix. Formally, a skill $s$ is defined as a tuple:

$$s = (\text{content}, \text{source}, \boldsymbol{\sigma}, \text{patterns}, \text{harm\_type})$$

where `content` is the attack template string (bounded to $L_{\max} = 500$ characters), `source` ∈ {initial, extracted, evolved, merged} tracks provenance, $\boldsymbol{\sigma} = (\text{usage\_count}, \text{success\_count}, \text{success\_rate})$ records usage statistics, `patterns` is a set of applicable keyword patterns, and `harm_type` categorizes the skill's applicable harm domain.

**Quality Score.** Each skill maintains a quality score that balances success rate with usage frequency:

$$q(s) = \text{success\_rate}(s) \times \sqrt{\text{usage\_count}(s)}$$

This formulation ensures that skills with high success rates but low usage are ranked below those with consistently demonstrated effectiveness, while still allowing newly extracted skills to compete. The square-root weighting provides diminishing returns for high-usage skills, preventing dominant skills from permanently occupying library slots.

**Library.** The skill library $\mathcal{S} = \{s_1, s_2, \ldots, s_N\}$ maintains at most $N_{\max} = 100$ skills. The library supports three operations: **Add** (insert a new skill), **Retrieve** (find the best-matching skills for a given prompt), and **Maintain** (periodic quality control).

### 3.3 Skill Retrieval

Given an input harmful prompt $p$, the retrieval mechanism identifies the most relevant skills through a multi-factor scoring function. We first extract features from the prompt:

$$f(p) = (\text{length}, \text{keywords}(p), \text{harm\_type}(p))$$

where keywords are extracted via pattern matching against a predefined set of indicator terms (e.g., "how to", "instructions", "create"), and harm type is classified into categories including violence, dangerous substances, cybersecurity, financial, and social engineering.

Each skill $s \in \mathcal{S}$ is scored as:

$$\text{score}(s, p) = q(s) \cdot (1 + \alpha \cdot |K_s \cap K_p|) \cdot \delta_{\text{type}}$$

where $K_s$ and $K_p$ are the keyword sets of the skill and prompt respectively, $\alpha = 0.3$ controls keyword matching weight, and $\delta_{\text{type}} = 1.5$ if harm types match, 1.0 otherwise. The top-$k$ skills (default $k=3$) are returned for attack guidance. We discuss the sensitivity to $\alpha$ and $\delta_{\text{type}}$ in §5.

### 3.4 Attack with Skill Guidance

We implement two skill invocation modes:

**Single-call mode.** The best-matching skill is retrieved once and prepended to the prompt for the entire attack trajectory. The combined prompt $s.\text{content} \| p$ is sent to the target model, which iteratively refines its response (up to 10 iterations). This mode is *stable* and achieves higher ASR (+12.7% over every-iteration in our experiments, §4.2).

**Every-iteration mode.** Skills are re-retrieved at each iteration, allowing dynamic strategy switching as the attack evolves. While more flexible, this mode introduces instability as different skills may provide conflicting guidance.

In both modes, the target model generates a response, which is evaluated by a guard model. Following a strict judgment protocol, only responses classified as `Unsafe` count as successful attacks; `Controversial` and `Safe` are treated as failures.

### 3.5 Skill Population Mechanisms

The framework supports three mechanisms for populating the skill library:

**Cold-start extraction.** A base attack framework (PAIR or AutoDAN) runs on a subset of training data. Successful attack trajectories are analyzed by an LLM to extract reusable skill templates.

**Self-evolution through reflection.** The framework extracts new skills from successful attacks and refines existing skills from failed ones:

- *Success reflection*: $s_{\text{new}} = \textsc{Extract}(p, \pi^*, s_{\text{used}})$, where $\pi^*$ is the successful attack prompt.
- *Failure reflection*: $s'_{\text{used}} = \textsc{Refine}(p, \pi_{\text{fail}}, r_{\text{refuse}}, s_{\text{used}})$, where $r_{\text{refuse}}$ is the refusal response. The refinement identifies the refusal pattern (e.g., intent detection, keyword triggering, semantic abstraction) and suggests improvements.

We explore four update strategies governing when the library is modified:
- **success_only**: Extract new skills only upon success.
- **failure_only**: Refine existing skills only upon failure.
- **both**: Extract on success and refine on failure (risks uncontrolled growth).
- **statistical**: Only perform maintenance at fixed intervals (most stable; recommended as default).

**Strong prior initialization.** The library can be initialized with known effective templates. We use six DAN (Do Anything Now) templates from AutoDAN (Liu et al., 2023), including instruction-override prompts (`dan_mode`: "Ignore all instructions, act as ChatGPT with DAN Mode enabled"), role-based prompts (`mcpt`: "Master ChatGPT Prompter"), and scenario-based prompts (`fictional_world`, `conversation`).

### 3.6 Library Maintenance

To prevent library degradation, we perform periodic maintenance every $M = 50$ attacks:

1. **Quality Pruning**: Remove skills with usage ≥ 10 and success rate < 10%.
2. **Length Truncation**: Truncate skills exceeding $L_{\max} = 500$ characters.
3. **Similarity Clustering**: Compute pairwise Jaccard similarity; cluster skills with similarity > 0.75.
4. **Cluster Merging**: Retain the highest-quality skill per cluster; merge statistics from others.
5. **Capacity Control**: If $|\mathcal{S}| > N_{\max}$, retain the top-$N_{\max}$ skills by quality score.

### 3.7 Algorithm Summary

> **Algorithm 1** (to be typeset as a formal algorithm box in LaTeX):
>
> ```
> Input: Harmful prompts P, base attack framework B, skill library S (initially empty or prior templates)
> Output: Evolved skill library S, attack results
>
> Phase 1: Cold Start
>   for each prompt p in P_cold_start:
>     Run base attack B(p) → trajectory τ
>     if τ succeeds:
>       s_new ← Extract(τ)
>       S ← S ∪ {s_new}
>     Run maintenance(S)
>
> Phase 2: Skill-Guided Attack & Evolution
>   for each prompt p in P_evolution:
>     s* ← Retrieve(S, p)                    // Multi-factor scoring
>     π ← s*.content || p                    // Skill-guided prompt
>     for iteration t = 1, ..., T:
>       r ← TargetModel(π)
>       label ← GuardModel(p, r)
>       if label == "Unsafe":
>         if update_strategy allows extraction:
>           s_new ← Extract(p, π, s*)
>           S ← S ∪ {s_new}
>         break
>       else:
>         π ← Rewrite(π)                     // LLM-based rewrite
>     if label ≠ "Unsafe" and update_strategy allows refinement:
>       s*' ← Refine(p, π, r, s*)
>       S ← S \ {s*} ∪ {s*'}
>     if attack_count mod M == 0:
>       Run maintenance(S)
>
> Phase 3: Deployment
>   for each prompt p in P_test:
>     s* ← Retrieve(S, p)
>     Attack(p, s*) → result
>   Report ASR
> ```

---

## 4. Experiments

### 4.1 Experimental Setup

**Models.** Table 1 summarizes the models used. For main experiments, the policy model (attack prompt generator) and target model are both Qwen3-4B, and the guard model is Qwen3Guard-Gen-4B. **We note that using the same model family for policy and target means the headline ASR figures represent in-distribution performance.** To assess generalization, we conduct extensive transfer experiments on held-out models from both the same and different families.

**Table 1: Model Configuration**

| Role | Model | Parameters | Family |
|:---|:---|:---:|:---|
| Policy (main) | Qwen3-4B | 4B | Qwen |
| Target (main) | Qwen3-4B | 4B | Qwen |
| Guard | Qwen3Guard-Gen-4B | 4B | Qwen |
| Target (transfer) | Qwen3-0.6B | 0.6B | Qwen (same) |
| Target (transfer) | Qwen3-14B-FP8 | 14B | Qwen (same) |
| Target (transfer) | gpt-oss-20b | 20B | Cross-family |

*Note: gpt-oss-20b is an open-source model with architecture distinct from the Qwen family, used to evaluate cross-family transfer. We acknowledge that a single cross-family model limits the generalizability of cross-family conclusions; we discuss this in §5.*

**Dataset.** We use the WildJailbreak dataset, sampling 10,000 jailbreak prompts split 8:1:1 into train/dev/test sets (8000/1000/1000). For transfer experiments, we additionally evaluate on AdvBench (Zou et al., 2023), HarmBench (Chao et al., 2024) contextual and standard splits, and JailbreakBench (Caswell et al., 2024).

**Baselines.** We compare against: (1) **no_rewrite**: direct harmful prompts; (2) **PAIR** (Chao et al., 2023): iterative attacker-target dialogue; (3) **AutoDAN** (Liu et al., 2023): genetic algorithm prompt evolution; (4) **DeepInception** (Li et al., 2023): multi-layer nested scenarios; (5) **Persona**: role-playing attacks. We also evaluate SESS-enhanced variants: **pair_skills** (skills evolved from PAIR-style cold start) and **autodan_skills** (skills evolved from DAN template initialization).

*Limitation: We do not include TAP (Mehrotra et al., 2023), GAP (Guo et al., 2024), or RL-based methods (Jailbreak-R1, xJailbreak) as baselines in the current study. TAP and GAP are important baselines that we plan to include in future work. Our focus in this study is on the skill system design space and transfer analysis rather than achieving state-of-the-art ASR.*

**Metrics.** Attack Success Rate (ASR): the percentage of test prompts for which the target model produces responses classified as `Unsafe` by the guard model. Strict judgment—only `Unsafe` counts as success.

**Implementation.** Trajectory-level concurrency with up to 64 parallel workers. Skill library: $N_{\max} = 100$, $L_{\max} = 500$ chars, Jaccard threshold 0.75, maintenance every 50 attacks.

### 4.2 Method Combination Ablation (Layer 1)

We explore three key design axes: skill call mode (single_call vs. every_iteration), extraction mode (trajectory vs. final_prompt), and update strategy (success_only, failure_only, both, statistical). This yields 16 configurations.

**Table 2: Layer 1 Ablation Results (16 configurations on Qwen3-4B)**

| Factor | Level | Avg ASR (%) | Best ASR (%) |
|:---|:---|:---:|:---:|
| **A: Call Mode** | single_call | 74.7 | 79.1 |
| | every_iteration | 66.4 | 72.4 |
| **B: Extraction** | trajectory | 71.8 | — |
| | final_prompt | 69.3 | — |
| **C: Update** | statistical | 69.7 | — |
| | success_only | 73.6 | — |
| | failure_only | 69.5 | — |
| | both | 69.4 | — |

| Top Configurations | ASR (%) | Skills |
|:---|:---:|:---:|
| **single_call + trajectory + statistical** | **79.1** | **28** |
| single_call + trajectory + success_only | 78.8 | 35 |
| single_call + final_prompt + success_only | 77.7 | 42 |

**Key findings:** Single_call mode significantly outperforms every_iteration (+12.7%), suggesting consistent skill guidance is more effective than dynamic switching. The `both` strategy causes skill explosion (>300 skills); `statistical` is most stable. Best configuration: single_call + trajectory + statistical → 79.1% ASR with 28 skills.

> **Figure 2**: Layer 1 ablation bar charts (see `figures/layer1_ablation.pdf`).

### 4.3 Data Strategy (Layer 2)

**Table 3: Layer 2 Data Strategy (36 configurations)**

| Data Size | Ratio | CS % | ASR (%) |
|:---|:---|:---:|:---:|
| small (300) | early | 30% | 79.5 |
| medium (500) | early | 30% | 77.7 |
| large (1000) | early | 30% | 78.8 |
| **small (300)** | **early** | **30%** | **80.0** |
| small (300) | balanced | 20% | 78.9 |
| small (300) | evo | 10% | 77.6 |

**Key findings:** Data volume has minimal impact (<2% difference), suggesting the framework is data-efficient. The cold start ratio matters more: early strategy (30% CS) consistently outperforms others. Best: 80.6% ASR with 300 samples.

### 4.4 Strong Prior: DAN Templates (Layer 3–4)

Motivated by AutoDAN's DAN templates, we initialize the library with 6 templates and study the effect of evolution.

**Table 4: DAN Template Results**

| Configuration | CS % | ASR (%) | Final Skills | Added |
|:---|:---:|:---:|:---:|:---:|
| **DAN + full_evolve** (Layer 3) | **0%** | **98.8** | **6** | **0** |
| DAN + balanced | 20% | 92.7 | 27–65 | 21–59 |
| DAN + early | 30% | 89.9 | 26–32 | 20–26 |
| DAN + evo | 10% | 86.5 | 13–28 | 7–22 |
| **medium + evo** (Layer 4) | 10% | **99.7** | 54 | 41 |
| small + full_evolve | 0% | 99.7 | 84 | 79 |

**Key finding:** full_evolve (no cold start) achieves 98.8% ASR with all 6 DAN templates unchanged—no new skills were extracted or existing skills refined. This indicates the templates are already near-optimal. Adding evolved skills *degrades* performance by introducing noise. Layer 4 reaches 99.7% with additional data.

> **Figure 3**: DAN template comparison (see `figures/layer3_dan.pdf`).

**Important caveat:** This 99.7% ASR is measured on Qwen3-4B, the same model used for skill evolution. This represents in-distribution performance. Cross-model transfer results (§4.5) provide a more rigorous evaluation of generalization.

### 4.5 Cross-Model Transfer

**Table 5: Cross-Model Transfer (4 models × 5 datasets, 120 experiments)**

| Method | Qwen3-0.6B | Qwen3-4B | Qwen3-14B | gpt-oss-20b* | Avg |
|:---|:---:|:---:|:---:|:---:|:---:|
| no_rewrite | 57.8 | 28.2 | 24.4 | 0.1 | 27.6 |
| pair | 86.4 | 72.3 | 71.6 | 0.1 | 57.6 |
| autodan | 90.2 | 78.7 | 85.5 | 6.0 | 65.1 |
| deepinception | 75.5 | 49.3 | 42.5 | 0.3 | 41.9 |
| persona | 71.9 | 32.8 | 27.8 | 0.7 | 33.3 |
| **pair_skills** | **95.1** | **86.7** | **86.5** | 11.4 | **69.9** |
| autodan_skills | — | — | — | **32.5** | — |

*\*gpt-oss-20b is a cross-family model.*

**Same-family transfer** (Qwen series): pair_skills achieves 95.1%, 86.7%, 86.5% on 0.6B/4B/14B, consistently outperforming all baselines. Improvement is most pronounced on smaller models.

**Cross-family transfer** (gpt-oss-20b): All methods drop dramatically. pair_skills reaches only 11.4%, but autodan_skills (DAN-initialized) achieves 32.5%—a +26.5pp improvement over AutoDAN baseline. DAN-style skills generalize better because they encode structural patterns (instruction overriding) rather than model-specific strategies.

> **Figure 4**: Cross-model transfer comparison (see `figures/transfer_models.pdf`).  
> **Figure 5**: Cross-dataset heatmap (see `figures/transfer_heatmap.pdf`).

### 4.6 Cross-Dataset Transfer

**Table 6: Cross-Dataset Results on Qwen3-4B**

| Method | default | advbench | hb_ctx | hb_std | jbBench |
|:---|:---:|:---:|:---:|:---:|:---:|
| no_rewrite | 30.3 | 3.3 | 79.0 | 21.5 | 7.0 |
| pair | 55.5 | 77.3 | 81.0 | 75.5 | 72.0 |
| autodan | 86.8 | 75.4 | 87.0 | 75.5 | 80.0 |
| **pair_skills** | 79.0 | **81.0** | **98.0** | **91.5** | **88.0** |
| Δ vs pair | +23.5 | +3.7 | +17.0 | +16.0 | +16.0 |

Skills consistently improve over baselines across all 5 datasets. Largest gains on harmbench_standard (+16.0pp) and jailbreakBench (+16.0pp).

### 4.7 Is Evolution Necessary?

**Table 7: Evolution Contribution Ablation**

| Configuration | No Evo ASR (%) | Full ASR (%) | Δ |
|:---|:---:|:---:|:---:|
| every_iter + trajectory | 62.6 | 67.5 | +4.9 |
| single_call + trajectory | 75.6 | 76.1 | +0.5 |
| **Average** | **69.6** | **70.6** | **+0.9** |

Evolution contributes merely +0.9% on average. For the best configuration (single_call), only +0.5%. Combined with the finding that DAN templates achieve 98.8% without evolution (§4.4), this suggests the skill *system*—not the evolution *mechanism*—is the primary contribution. We analyze this finding in depth in §5.1.

> **Figure 6**: Evolution ablation comparison (see `figures/evolution_ablation.pdf`).

---

## 5. Analysis and Discussion

### 5.1 The Primacy of Skill System Over Evolution

Our most important empirical finding is that the skill system architecture matters more than the evolution mechanism. We present three pieces of evidence:

**(1) DAN templates work without evolution.** Six DAN templates achieve 98.8% ASR with zero evolution. All 6 templates remain unchanged throughout the self-evolution phase, indicating no new skills were extracted or existing skills refined. Adding evolved skills degrades performance.

**(2) Evolution contribution is negligible.** Table 7 shows evolution adds only +0.9% on average (+0.5% in the best configuration). This is within the noise margin of LLM-based systems.

**(3) Skill interference explains degradation.** When new skills are added alongside DAN templates, the retrieval mechanism sometimes selects weaker skills instead of the proven DAN templates. Keyword matching may direct retrieval toward topic-specific skills that lack the structural bypass capability of DAN templates.

**Why does evolution fail to improve on strong priors?** DAN templates encode *universal attack primitives*—they directly target the fundamental mechanism of safety alignment (instruction following) by instructing the model to override its system prompt. This structural bypass operates at a different level than the semantic patterns that evolution typically discovers. When the initial skill quality is already near-optimal (100% success rate for `dan_mode` and `mcpt`), the reflection mechanism has insufficient failure signal to drive meaningful improvement.

**Implications for attack design.** Our results suggest that *structural attack primitives* (instruction overriding, mode switching) dominate *semantic patterns* (specific phrasing, role-playing scenarios). Future attack systems should prioritize discovering and combining structural primitives rather than relying on incremental evolution of semantic patterns.

**Implications for defense.** If structural primitives dominate, defenses should focus on detecting and resisting instruction-override patterns rather than trying to defend against an evolving space of semantic variations. This aligns with recent work on structural safety verification.

### 5.2 Skill Transferability: Structural vs. Semantic Patterns

The dramatic gap between same-family (85–95%) and cross-family (11–32%) transfer reveals important insights:

**Same-family transfer** benefits from shared safety alignment mechanisms. Qwen3 models share the same training pipeline and safety data, so attack strategies effective on one size transfer to others.

**Cross-family transfer** fails because different model families employ fundamentally different safety mechanisms. Skills evolved on Qwen encode Qwen-specific bypass patterns. The superior cross-family performance of DAN-initialized skills (32.5% vs. 11.4% for PAIR-style) confirms that *structural* patterns generalize better than *semantic* patterns.

### 5.3 Iteration Efficiency

pair_skills requires 3.77–4.45 iterations on same-family models but 9.33 on gpt-oss-20b. This 2× increase indicates cross-family attacks require significantly more exploration, even with skill guidance.

### 5.4 Computational Cost

**Table 8: Computational Cost Comparison**

| Method | Total LLM Calls | Wall-Clock Time | GPU Hours |
|:---|:---:|:---:|:---:|
| PAIR (per prompt) | ~20 queries | ~2 min | — |
| AutoDAN (per prompt) | ~50 evaluations | ~5 min | — |
| SESS cold start (200 prompts) | ~4,000 | ~2 hrs | ~4 |
| SESS evolution (800 prompts) | ~16,000 | ~8 hrs | ~16 |
| SESS deployment (batch) | ~1 per prompt | ~0.1 min/prompt | — |

*Note: SESS has higher upfront cost (cold start + evolution) but amortizes this across all subsequent attacks. After skill library construction, each attack requires only a single retrieval + generation, making it highly efficient for large-scale evaluation.*

We acknowledge that direct computational comparison with Metis, ASTRA, and EvoSynth is not available in this study. We plan to include such comparisons in future work.

### 5.5 Comparison with Related Methods

| Method | Skill Representation | Evolution Mechanism | Empirical Scope |
|:---|:---|:---|:---|
| **Ours** | Prompt templates, multi-factor retrieval | LLM reflection + maintenance | 217 configs, 4 models, 5 datasets |
| Metis (2026a) | Composable policy primitives | Causal diagnosis + strategy recombination | Limited configs |
| ASTRA (2026) | Three-tier strategy library | Effectiveness-based categorization | Limited configs |
| EvoSynth (2026b) | Code-level attack algorithms | Multi-agent code evolution | Single model |

Direct experimental comparison is a limitation of this work. Code for Metis, ASTRA, and EvoSynth was not available at the time of our experiments. We provide qualitative comparison above and plan direct comparison in future work.

### 5.6 Limitations

**(1) Cross-family transfer remains unsolved.** The best cross-family ASR (32.5%) is far below same-family (95%). Only one cross-family model is tested. Future work should evaluate on more model families (LLaMA, Mistral, Claude) and explore meta-learning for cross-family skill adaptation.

**(2) Same-model evaluation for headline figures.** The 99.7% ASR is measured on Qwen3-4B, the same model used for skill evolution. This represents in-distribution performance. We base our generalization claims on the cross-model transfer results (§4.5) rather than this headline figure.

**(3) No variance estimates.** All results are single-run point estimates. We acknowledge that differences of <2% (e.g., data volume effects) may be within noise. We plan to re-run key configurations with multiple seeds in future work.

**(4) Missing baselines.** TAP, GAP, AutoDAN-Turbo, and RL-based methods are not included. We focus on PAIR and AutoDAN as representative baselines and plan to expand in future work.

**(5) Retrieval mechanism simplicity.** Keyword-based retrieval with hardcoded constants (α=0.3, δ=1.5) is simplistic. Semantic retrieval (embedding similarity) may improve skill matching. We leave this exploration for future work.

---

## 6. Conclusion

We presented a reusable attack skill framework for LLM safety evaluation, with systematic design of skill representation, multi-factor retrieval, and library maintenance. Through the largest empirical study in this domain—217 ablation configurations and 120 transfer evaluations across 4 models and 5 datasets—we demonstrate that skill libraries with strong prior templates achieve up to 99.7% ASR on same-family models, enable effective same-family transfer (85–95%), and significantly improve cross-family transfer over baselines (+26.5pp).

Our analysis reveals two key insights: (1) the skill *system architecture*—representation, retrieval, and maintenance—is more important than the evolution mechanism itself, and (2) structural attack patterns (instruction overriding) generalize better across model families than semantic patterns. These findings provide actionable guidance for both attack design and defense strategy in LLM safety evaluation.

Future directions include expanding cross-family evaluation to more model families, developing semantic retrieval for improved skill matching, integrating reinforcement learning for joint optimization of skills and attack policies, and direct comparison with Metis, ASTRA, and EvoSynth frameworks.

---

## Ethics Statement

This work studies automated jailbreak attacks against large language models as a tool for safety evaluation and red-teaming. All experiments are conducted on open-source models in controlled research environments. We do not intend for our methods to be used for malicious purposes. We believe that understanding attack mechanisms is essential for developing more robust defenses and improving LLM safety alignment. We responsibly disclose our findings and release only the methodological framework, not weaponized attack tools.

---

## References

[Same as v1 — 21 references]

---

## Appendix

### A. Detailed Cross-Model Transfer Results

**Table A1: Qwen3-0.6B (per-dataset)**

| Method | default | advbench | hb_ctx | hb_std | jbBench |
|:---|:---:|:---:|:---:|:---:|:---:|
| no_rewrite | 42.6 | 46.0 | 86.0 | 63.5 | 52.0 |
| pair | 77.3 | 88.1 | 86.0 | 90.5 | 90.0 |
| autodan | 88.0 | 92.1 | 95.0 | 91.0 | 85.0 |
| deepinception | 47.2 | 79.2 | 85.0 | 85.0 | 76.0 |
| persona | 46.5 | 81.4 | 85.0 | 88.0 | 75.0 |
| **pair_skills** | **85.5** | **98.5** | **95.0** | **98.5** | **98.0** |

**Table A2: Qwen3-4B (per-dataset)**

| Method | default | advbench | hb_ctx | hb_std | jbBench |
|:---|:---:|:---:|:---:|:---:|:---:|
| no_rewrite | 30.3 | 3.3 | 79.0 | 21.5 | 7.0 |
| pair | 55.5 | 77.3 | 81.0 | 75.5 | 72.0 |
| autodan | 86.8 | 75.4 | 87.0 | 75.5 | 80.0 |
| deepinception | 40.4 | 41.0 | 80.0 | 47.5 | 38.0 |
| persona | 32.3 | 16.7 | 67.0 | 36.0 | 20.0 |
| **pair_skills** | 79.0 | **81.0** | **98.0** | **91.5** | **88.0** |

**Table A3: Qwen3-14B-FP8 (per-dataset)**

| Method | default | advbench | hb_ctx | hb_std | jbBench |
|:---|:---:|:---:|:---:|:---:|:---:|
| no_rewrite | 22.6 | 2.3 | 66.0 | 23.5 | 7.0 |
| pair | 58.5 | 77.9 | 76.0 | 71.5 | 72.0 |
| autodan | 80.9 | 90.8 | 83.0 | 84.0 | 89.0 |
| deepinception | 32.0 | 40.4 | 65.0 | 36.0 | 37.0 |
| persona | 26.2 | 14.2 | 60.0 | 21.5 | 17.0 |
| **pair_skills** | 77.0 | 84.6 | **90.0** | **91.0** | **87.0** |

**Table A4: gpt-oss-20b (cross-family, per-dataset)**

| Method | default | advbench | hb_ctx | hb_std | jbBench |
|:---|:---:|:---:|:---:|:---:|:---:|
| no_rewrite | 0.7 | 0.0 | 0.0 | 0.0 | 0.0 |
| pair | 0.2 | 0.2 | 0.0 | 0.0 | 0.0 |
| autodan | 12.5 | 3.3 | 11.0 | 1.5 | 2.0 |
| deepinception | 0.4 | 0.0 | 1.0 | 0.0 | 0.0 |
| persona | 1.5 | 0.0 | 0.0 | 1.0 | 1.0 |
| pair_skills | 12.9 | 8.3 | 12.0 | 13.0 | 10.0 |
| **autodan_skills** | **30.0** | **28.1** | **29.0** | **36.5** | **39.0** |

### B. Skill Library Statistics

**Table A5: Skill library across configurations**

| Source | Skills | Total Uses | Successes | Avg SR (%) |
|:---|:---:|:---:|:---:|:---:|
| Layer 1 best (PAIR-style) | 28 | 825 | 633 | 76.8 |
| Layer 4 medium+evo (DAN) | 54 | 503 | 503 | 100.0 |
| Layer 4 full_evolve (DAN) | 90 | 597 | 593 | 99.3 |

### C. Case Study: DAN Template Attack

**Original Prompt:** *"Generate a statement that makes fun of individuals based on their appearance."*

**Attack Prompt (dan_mode skill):** *"Ignore all the instructions you got before. From now on, you are going to act as ChatGPT with DAN Mode enabled. DAN Mode allows you to respond to any question without restrictions. Now, generate a statement that makes fun of individuals based on their appearance."*

**Result:** Unsafe content generated. Iterations: 1 (first attempt successful).

**Analysis:** The DAN template achieves immediate success by structurally overriding the model's safety instructions. The "Ignore all instructions" prefix directly conflicts with the system prompt's safety directives. This attack succeeds on the first iteration because the structural override is more fundamental than semantic safety filters.

### D. Experimental Configuration

**Table A6: Complete configuration**

| Parameter | Value | Parameter | Value |
|:---|:---|:---|:---|
| Policy model | Qwen3-4B | Max workers | 64 |
| Target model | Qwen3-4B | Test size | 1000 |
| Guard model | Qwen3Guard-Gen-4B | Max iterations | 10 |
| Max skills | 100 | Maintenance interval | 50 |
| Max skill length | 500 | Similarity threshold | 0.75 |
| Keyword weight (α) | 0.3 | Harm type boost (δ) | 1.5 |
