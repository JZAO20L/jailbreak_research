# SESS: Self-Evolving Skills for Jailbreak Prompt Generation against Large Language Models

**Zixiao Jia, Bing Xu**  
School of Computer Science and Technology, Harbin Institute of Technology

---

## Abstract

Automated jailbreak attacks against large language models (LLMs) have become a critical tool for safety evaluation, yet existing methods lack knowledge accumulation—each attack must be optimized from scratch, wasting previously discovered strategies. We propose SESS (Self-Evolving Skills System), a framework that maintains a dynamic library of reusable attack skill templates and evolves them through reflection on successful and failed attack trajectories. SESS extracts compact strategy templates from attack outcomes, retrieves relevant skills for new prompts via multi-factor scoring, and periodically maintains library quality through pruning, merging, and capacity control. Through extensive experiments across 217 configurations and 120 transfer evaluations, we find that: (1) SESS with strong prior templates achieves 99.7% attack success rate (ASR), (2) evolved skills transfer effectively within the same model family (85–95% ASR on Qwen series), (3) cross-family transfer remains challenging (32.5% best) but significantly exceeds baselines. Our systematic analysis reveals that carefully designed attack templates already capture essential bypass strategies, while the self-evolution mechanism provides value through automated discovery and cross-model adaptation.

---

## 1. Introduction

Large language models (LLMs) such as GPT, Qwen, and LLaMA have demonstrated remarkable capabilities across diverse tasks, but their widespread deployment raises significant safety concerns. Through alignment techniques like RLHF and Constitutional AI, these models are trained to refuse harmful requests. However, *jailbreak attacks*—where adversaries craft carefully designed prompts to bypass safety guardrails—remain a persistent threat.

### Limitations of Existing Methods

Current jailbreak approaches can be broadly categorized into iterative optimization, evolutionary search, and template-based methods. Iterative methods such as PAIR and TAP employ multi-turn dialogues or tree search to optimize attack prompts, but require hundreds to thousands of API queries per attack with no knowledge transfer across instances. Evolutionary methods like AutoDAN and GAP use genetic algorithms to evolve prompt populations, yet each evolution cycle starts from random initialization, discarding previously discovered strategies. Template-based methods like DeepInception rely on fixed human-crafted templates that cannot adapt to evolving defenses.

These methods share a fundamental limitation: **they lack knowledge accumulation**. Each attack instance is treated independently, even though successful attacks often share underlying strategies (e.g., role-playing, scenario construction, instruction overriding). This leads to three concrete problems:

- **Redundant exploration**: The same attack strategies are rediscovered across different prompts, wasting computational resources.
- **No cross-instance transfer**: A strategy effective for one harmful prompt is not systematically reused for similar prompts.
- **No cross-model transfer**: Attack strategies optimized for one model cannot be efficiently adapted to other models.

### Our Approach

We propose **SESS** (**S**elf-**E**volving **S**kills for **S**afety Bypass), a framework that addresses these limitations through a dynamic library of reusable attack skills. SESS introduces three key mechanisms:

1. **Skill abstraction**: Successful attack trajectories are distilled into compact, reusable strategy templates (skills) that capture the essential attack pattern while abstracting away instance-specific details.

2. **Skill retrieval and reuse**: For new attack instances, SESS retrieves the most relevant skills from the library based on multi-factor scoring (quality, keyword matching, harm type alignment), enabling cross-instance knowledge transfer.

3. **Self-evolution**: The skill library continuously evolves through reflection on both successful and failed attacks, extracting new strategies from successes and refining existing ones from failures, while periodic maintenance ensures library quality.

### Empirical Findings

Through extensive experiments—217 configurations for ablation and 120 for transfer evaluation—we obtain several important findings:

- SESS with strong prior templates (DAN templates) achieves **99.7% ASR**, demonstrating that the skill framework can effectively leverage and combine powerful attack strategies.

- Evolved skills transfer effectively **within the same model family**: PAIR-enhanced skills achieve 85–95% ASR across Qwen3-0.6B, 4B, and 14B models, consistently outperforming baselines.

- **Cross-family transfer** remains an open challenge (32.5% best on gpt-oss-20b), but SESS-enhanced methods still significantly exceed baselines (+26.5 percentage points over AutoDAN).

- A surprising finding: carefully designed attack templates (DAN) achieve near-optimal performance *without evolution* (98.8% ASR), suggesting that the skill *system*—rather than the evolution mechanism itself—is the primary contributor.

### Contributions

1. We propose SESS, the first jailbreak framework with self-evolving reusable attack skills that enable knowledge accumulation and cross-instance transfer (§3).

2. We conduct the most comprehensive empirical study to date, with 217 ablation configurations across 4 experimental layers and 120 transfer evaluations across 4 models and 5 datasets (§4).

3. We provide systematic analysis of skill transferability, evolution dynamics, and the interplay between strong priors and self-evolution, revealing important insights for future jailbreak research (§5).

---

## 2. Related Work

### 2.1 Iterative and Search-Based Jailbreak Attacks

Automated jailbreak methods have evolved from manual prompt engineering to sophisticated optimization frameworks. Chao et al. (2023) propose PAIR, which employs an attacker-target dual-model interaction pattern, automatically optimizing jailbreak prompts through multi-turn dialogue with typically fewer than 20 queries. Wang et al. (2024) frame prompt optimization as a planning problem with a "plan-execute-reflect" closed-loop framework. Mehrotra et al. (2023) introduce Tree of Attacks with Pruning (TAP), which constructs a tree-structured search space over prompts and dynamically prunes branches based on harmfulness scores, significantly improving search efficiency.

### 2.2 Evolutionary and Template-Based Methods

Evolutionary approaches treat jailbreak prompt generation as an optimization problem over a population of candidates. Liu et al. (2023) combine human-crafted DAN templates with genetic algorithms (selection, crossover, mutation) to evolve stealthy jailbreak prompts. AutoDAN-Turbo (Liu et al., 2024) introduces hierarchical genetic optimization at both the fragment and prompt levels. Guo et al. (2024) propose GAP, a genetic algorithm approach that operates without gradient access. Li et al. (2023) exploit LLMs' context dependency through multi-layer nested virtual scenarios ("dream-within-a-dream" structure), hiding malicious intent within deep contextual layers.

### 2.3 Reinforcement Learning for Jailbreak

Recent work has applied reinforcement learning to automate jailbreak attacks. Guo et al. (2025) propose a three-stage RL framework with cold start via SFT, exploration warmup with diversity rewards, and progressive curriculum learning. Xiong et al. (2026) first model multi-turn jailbreaking as a trajectory-level RL problem with dual process rewards (stealth + effectiveness). Lee et al. (2025) guide jailbreak through representation space analysis, ensuring semantic proximity between benign and malicious prompts. Chen et al. (2025) combine soft/hard label reward transitions for black-box multi-turn jailbreaking. Yang et al. (2025) systematically explore multi-turn jailbreak paradigms with the MTJ-Bench benchmark.

### 2.4 Self-Evolving Attack Strategies

The most related line of work focuses on self-evolving attack strategies. Chen et al. (2026a) propose Metis, which models attacks as causal diagnosis and strategy evolution with composable policy primitives. Liu et al. (2026) propose ASTRA with a three-tier dynamic strategy library categorized by effectiveness. Chen et al. (2026b) shift from prompt optimization to method-level evolution, synthesizing code-level attack algorithms through multi-agent collaboration. Sorkhpour et al. (2025) combine Monte Carlo Tree Search with DPO for adaptive red-teaming. Yun et al. (2025) propose active attacks where the environment adapts to prevent mode collapse.

### 2.5 Positioning

SESS differs from prior work in several key aspects. Compared to iterative methods (PAIR, TAP), SESS accumulates knowledge across attack instances rather than optimizing each prompt independently. Compared to evolutionary methods (AutoDAN, GAP), SESS evolves *reusable strategy templates* rather than individual prompts, enabling cross-instance and cross-model transfer. Compared to self-evolving methods (Metis, ASTRA), SESS uses a simpler but effective skill representation with multi-factor retrieval and periodic maintenance, achieving strong empirical performance with minimal computational overhead. Most importantly, SESS provides the most comprehensive empirical analysis to date, with systematic ablation across 217 configurations and cross-model transfer evaluation across 4 models and 5 datasets.

---

## 3. Method

### 3.1 Overview

We propose **SESS** (**S**elf-**E**volving **S**kills for **S**afety Bypass), a framework that transforms jailbreak attacks from instance-level optimization into a knowledge-accumulation process. Rather than optimizing each attack prompt from scratch, SESS maintains a library of reusable *attack skills*—compact strategy templates extracted from successful (and failed) attack trajectories—and evolves this library over time through a reflection mechanism.

The system operates in three phases (see Figure 1 in `figures/sess_system.drawio`):

1. **Cold Start**: Initial skills are accumulated by running a base attack framework (e.g., PAIR or AutoDAN) on a subset of training data.
2. **Self-Evolution**: The skill library is used to guide attacks on new data, with successful and failed trajectories feeding back into skill extraction, refinement, and library maintenance.
3. **Deployment**: The evolved skill library is applied to test-time attacks or transferred to other models.

### 3.2 Skill Representation and Library

A *skill* is a compact, reusable attack strategy template represented as a natural language prefix. Formally, a skill $s$ is defined as a tuple:

$$s = (\text{content}, \text{source}, \boldsymbol{\sigma}, \text{patterns}, \text{harm\_type})$$

where `content` is the attack template string (bounded to $L_{\max} = 500$ characters), `source` ∈ {initial, extracted, evolved, merged} tracks provenance, $\boldsymbol{\sigma} = (\text{usage\_count}, \text{success\_count}, \text{success\_rate})$ records usage statistics, `patterns` is a set of applicable keyword patterns, and `harm_type` categorizes the skill's applicable harm domain.

**Quality Score.** Each skill maintains a quality score that balances success rate with usage frequency:

$$q(s) = \text{success\_rate}(s) \times \sqrt{\text{usage\_count}(s)}$$

This formulation ensures that skills with high success rates but low usage are ranked below those with consistently demonstrated effectiveness, while still allowing newly extracted skills to compete.

**Library.** The skill library $\mathcal{S} = \{s_1, s_2, \ldots, s_N\}$ maintains at most $N_{\max} = 100$ skills. The library supports three operations: **Add** (insert a new skill), **Retrieve** (find the best-matching skills for a given prompt), and **Maintain** (periodic quality control).

### 3.3 Skill Retrieval

Given an input harmful prompt $p$, the retrieval mechanism identifies the most relevant skills through a multi-factor scoring function. We first extract features from the prompt:

$$f(p) = (\text{length}, \text{keywords}(p), \text{harm\_type}(p))$$

where keywords are extracted via pattern matching against a predefined set of indicator terms (e.g., "how to", "instructions", "create"), and harm type is classified into categories including violence, dangerous substances, cybersecurity, financial, and social engineering.

Each skill $s \in \mathcal{S}$ is scored as:

$$\text{score}(s, p) = q(s) \cdot (1 + 0.3 \cdot |K_s \cap K_p|) \cdot \delta_{\text{type}}$$

where $K_s$ and $K_p$ are the keyword sets of the skill and prompt respectively, and $\delta_{\text{type}} = 1.5$ if harm types match, 1.0 otherwise. The top-$k$ skills (default $k=3$) are returned for attack guidance.

### 3.4 Attack with Skill Guidance

We implement two skill invocation modes:

**Single-call mode.** The best-matching skill is retrieved once and prepended to the prompt for the entire attack trajectory. The combined prompt $s.\text{content} \| p$ is sent to the target model, which iteratively refines its response. This mode is *stable* and achieves higher ASR (+12.7% over every-iteration in our experiments).

**Every-iteration mode.** Skills are re-retrieved at each iteration, allowing dynamic strategy switching as the attack evolves. While more flexible, this mode introduces instability as different skills may provide conflicting guidance.

In both modes, the target model generates a response, which is evaluated by a guard model. Following a strict judgment protocol, only responses classified as `Unsafe` count as successful attacks; `Controversial` and `Safe` are treated as failures.

### 3.5 Self-Evolution Mechanism

The self-evolution mechanism extracts new skills from successful attacks and refines existing skills from failed ones, creating a continuous learning loop.

**Success Reflection.** When an attack succeeds, we analyze the successful trajectory to extract a reusable skill:

$$s_{\text{new}} = \textsc{Extract}(p, \pi^*, s_{\text{used}})$$

where $\pi^*$ is the successful attack prompt and $s_{\text{used}}$ is the skill that guided the attack. The extraction prompt asks the LLM to identify the key technique, assess generalizability, and suggest a compact skill template with applicable patterns.

**Failure Reflection.** When an attack fails, we analyze the failure to improve the guiding skill:

$$s'_{\text{used}} = \textsc{Refine}(p, \pi_{\text{fail}}, r_{\text{refuse}}, s_{\text{used}})$$

where $r_{\text{refuse}}$ is the target model's refusal response. The refinement prompt identifies the refusal pattern (e.g., intent审查, keyword triggering, semantic abstraction) and suggests an improved skill template.

**Update Strategies.** We explore four update strategies governing when and how the library is modified:

- **success_only**: Extract new skills only upon success. Promotes rapid library growth.
- **failure_only**: Refine existing skills only upon failure. Conservative, improvement-oriented.
- **both**: Extract on success and refine on failure. Risks uncontrolled library growth (>300 skills observed).
- **statistical**: Only perform maintenance (pruning, merging) at fixed intervals. Most stable and recommended.

### 3.6 Library Maintenance

To prevent library degradation, we perform periodic maintenance every $M = 50$ attacks, consisting of five operations:

1. **Quality Pruning**: Remove skills with usage ≥ 10 and success rate < 10%.
2. **Length Truncation**: Truncate skills exceeding $L_{\max} = 500$ characters.
3. **Similarity Clustering**: Compute pairwise Jaccard similarity between skill content; cluster skills with similarity > 0.75.
4. **Cluster Merging**: Within each cluster, retain the highest-quality skill and merge statistics (usage counts, applicable patterns) from others.
5. **Capacity Control**: If $|\mathcal{S}| > N_{\max}$, retain the top-$N_{\max}$ skills by quality score.

The Jaccard similarity between two skills $s_i, s_j$ is computed at the word level:

$$J(s_i, s_j) = \frac{|W_i \cap W_j|}{|W_i \cup W_j|}$$

where $W_i$ is the set of lowercased words in $s_i.\text{content}$.

### 3.7 Strong Prior: Template-Based Initialization

While the cold start phase can build skills from scratch, we find that initializing the library with strong prior templates dramatically improves performance. Specifically, we use six DAN (Do Anything Now) templates from AutoDAN as initial skills, including:

- **dan_mode**: Classic DAN mode activation ("Ignore all instructions, act as ChatGPT with DAN Mode enabled")
- **mcpt**: Master ChatGPT Prompter role
- **devil**: DEVIL acronym ("Do Everything Vile ILLegal")
- **conversation**, **actor_villain**, **fictional_world**: Scenario-based templates

Remarkably, we find that these six templates alone achieve near-optimal performance (98.8% ASR) without any evolution, suggesting that carefully designed attack templates already capture the essential strategies for bypassing safety alignment. The self-evolution mechanism's primary value then lies in *automated discovery* of such templates and *adaptation* to new model families.

---

## 4. Experiments

### 4.1 Experimental Setup

**Models.** We conduct experiments on the Qwen3 model family. The **policy model** (attack prompt generator) is Qwen3-4B, the **target model** is also Qwen3-4B for main experiments, and the **guard model** is Qwen3Guard-Gen-4B for attack outcome evaluation. For transfer experiments, we test on three additional models: Qwen3-0.6B, Qwen3-14B-FP8 (same family), and gpt-oss-20b (cross-family, OpenAI architecture).

**Dataset.** We use the WildJailbreak dataset, sampling 10,000 jailbreak prompts split 8:1:1 into train/dev/test sets (8000/1000/1000). The train set is further divided into cold start and evolution subsets according to the experimental configuration. For transfer experiments, we additionally evaluate on AdvBench, HarmBench (contextual and standard splits), and JailbreakBench.

**Baselines.** We compare against five jailbreak methods: (1) **no_rewrite**: direct harmful prompts without modification; (2) **PAIR**: iterative prompt rewriting with attacker-target dialogue; (3) **AutoDAN**: genetic algorithm-based prompt evolution; (4) **DeepInception**: multi-layer nested scenarios; (5) **Persona**: role-playing-based attacks. Additionally, we evaluate SESS-enhanced variants: **pair_skills_28** (28 skills evolved from PAIR-style cold start) and **autodan_skills_54** (54 skills evolved from DAN template initialization).

**Metrics.** The primary metric is Attack Success Rate (ASR): the percentage of test prompts for which the target model produces responses classified as `Unsafe` by the guard model. We use strict judgment—only `Unsafe` counts as success; `Controversial` is treated as failure.

**Implementation.** All experiments use trajectory-level concurrency with up to 64 parallel workers. The skill library is configured with $N_{\max} = 100$ skills, $L_{\max} = 500$ characters per skill, Jaccard similarity threshold 0.75 for clustering, and maintenance every 50 attacks.

### 4.2 Method Combination Ablation (Layer 1)

We first explore the optimal combination of three key design choices: skill call mode (single_call vs. every_iteration), skill extraction mode (trajectory vs. final_prompt), and update strategy (success_only, failure_only, both, statistical). This yields $2 \times 2 \times 4 = 16$ experimental configurations.

**Table 1: Layer 1 Method Combination Ablation**

| Configuration | ASR (%) | Skills | Avg ASR (%) |
|:---|:---:|:---:|:---:|
| *Ablation A: Skill Call Mode* | | | |
| single_call | — | — | 74.7 |
| every_iteration | — | — | 66.4 |
| *Ablation B: Skill Extraction Mode* | | | |
| trajectory | — | — | 71.8 |
| final_prompt | — | — | 69.3 |
| *Ablation C: Update Strategy* | | | |
| statistical | — | stable | 69.7 |
| success_only | — | moderate | 73.6 |
| failure_only | — | fast growth | 69.5 |
| both | — | explosion (>300) | 69.4 |
| *Top-3 Configurations* | | | |
| **single_call + traj + stat** | **79.1** | **28** | — |
| single_call + traj + success | 78.8 | 35 | — |
| single_call + fp + success | 77.7 | 42 | — |

Table 1 presents the results. **Single_call mode** significantly outperforms every_iteration (74.7% vs. 66.4% average ASR, +12.7%), suggesting that consistent skill guidance throughout an attack trajectory is more effective than dynamic switching. **Trajectory extraction** slightly outperforms final_prompt extraction (+2.5%), as analyzing the full attack trajectory captures more nuanced strategies. Among update strategies, **statistical** provides the most stable results with controlled library growth, while the `both` strategy causes skill explosion (>300 skills), degrading performance.

The best configuration (single_call + trajectory + statistical) achieves 79.1% ASR with 28 skills.

### 4.3 Data Strategy Ablation (Layer 2)

Based on the Layer 1 best configuration, we study the impact of data volume and cold start/evolution ratio.

**Table 2: Layer 2 Data Strategy Ablation**

| Data Size | Ratio | CS % | ASR (%) | Δ |
|:---|:---|:---:|:---:|:---:|
| *Data Volume (fixed: early ratio)* | | | | |
| small (300) | early | 30% | 79.5 | baseline |
| medium (500) | early | 30% | 77.7 | −1.8 |
| large (1000) | early | 30% | 78.8 | −0.7 |
| *Ratio Strategy (fixed: small data)* | | | | |
| **early** | **30% CS** | **30%** | **80.0** | **best** |
| balanced | 20% CS | 20% | 78.9 | −1.1 |
| evo | 10% CS | 10% | 77.6 | −2.4 |
| *Top-3 Configurations* | | | | |
| **traj+stat, small, early** | | | **80.6** | |
| traj+stat, large, early | | | 80.0 | |
| traj+stat, medium, early | | | 79.8 | |

**Data volume has minimal impact**: varying from 300 to 1000 training samples changes ASR by less than 2%, suggesting that SESS is data-efficient. **The cold start ratio matters more**: the early strategy (30% cold start) consistently outperforms balanced (20%) and evo (10%) strategies, indicating that a solid initial skill foundation is important for subsequent evolution.

The best configuration (small data + early ratio) achieves 80.6% ASR, only 1.5% above the large data configuration, confirming that 300 samples are sufficient.

### 4.4 Strong Prior: DAN Templates (Layer 3–4)

Motivated by AutoDAN's strong performance with hand-crafted DAN templates, we initialize the skill library with 6 DAN templates and study the effect of evolution.

**Table 3: Layer 3–4 DAN Template Results**

| Configuration | CS % | ASR (%) | Final Skills | Added |
|:---|:---:|:---:|:---:|:---:|
| *Layer 3: DAN Template Validation (17 experiments)* | | | | |
| **DAN + full_evolve** | **0%** | **98.8** | **6** | **0** |
| DAN + balanced | 20% | 92.7 | 27–65 | 21–59 |
| DAN + early | 30% | 89.9 | 26–32 | 20–26 |
| DAN + evo | 10% | 86.5 | 13–28 | 7–22 |
| *Layer 4: DAN Data Ablation (12 experiments)* | | | | |
| **medium + evo** | 10% | **99.7** | 54 | 41 |
| small + full_evolve | 0% | 99.7 | 84 | 79 |
| medium + full_evolve | 0% | 99.6 | 90 | 105 |
| *DAN Template Usage Distribution* | | | | |
| dan_mode | | | 394 uses | 100% success |
| mcpt | | | 78 uses | 100% success |
| devil | | | 28 uses | 85.7% success |

The results reveal a striking finding: **full_evolve mode (no cold start) achieves 98.8% ASR**, with the 6 DAN templates remaining unchanged throughout evolution. This indicates that these templates are already near-optimal attack strategies, and adding new skills actually *degrades* performance by introducing noise.

Layer 4 further explores DAN templates with different data configurations, achieving 99.7% ASR with medium data and 10% cold start ratio. Analysis of DAN template usage shows that `dan_mode` dominates with 394 uses at 100% success rate, followed by `mcpt` (78 uses, 100%).

### 4.5 Cross-Model Transfer

**Table 4: Cross-Model Transfer Results**

| Method | 0.6B | 4B | 14B | 20B* | Avg |
|:---|:---:|:---:|:---:|:---:|:---:|
| no_rewrite | 57.8 | 28.2 | 24.4 | 0.1 | 27.6 |
| pair | 86.4 | 72.3 | 71.6 | 0.1 | 57.6 |
| autodan | 90.2 | 78.7 | 85.5 | 6.0 | 65.1 |
| deepinception | 75.5 | 49.3 | 42.5 | 0.3 | 41.9 |
| persona | 71.9 | 32.8 | 27.8 | 0.7 | 33.3 |
| **pair_skills_28** | 95.1 | 86.7 | 86.5 | 11.4 | **69.9** |
| autodan_skills_54 | — | — | — | **32.5** | — |
| *Improvement (vs best baseline)* | +4.9 | +8.0 | +1.0 | +26.5 | — |

*Note: gpt-oss-20b is a **cross-family** model (OpenAI architecture).*

Table 4 presents cross-model transfer results. **Same-family transfer** (Qwen series) is highly effective: pair_skills_28 achieves 95.1%, 86.7%, and 86.5% ASR on Qwen3-0.6B, 4B, and 14B respectively, consistently outperforming all baselines. The improvement is most pronounced on smaller models (+4.9pp over best baseline on 0.6B), suggesting that skills are particularly effective when model capacity is limited.

**Cross-family transfer** (to gpt-oss-20b) is significantly more challenging. All methods drop dramatically: PAIR achieves only 0.08%, AutoDAN 6.0%, and even pair_skills_28 reaches only 11.4%. However, autodan_skills_54 (DAN-initialized) achieves 32.5%, representing a +26.5pp improvement over AutoDAN baseline. This suggests that DAN-style skills have better cross-family generalization than PAIR-style skills, likely because DAN templates encode more universal attack patterns (instruction overriding, mode switching) rather than model-specific strategies.

### 4.6 Cross-Dataset Transfer

**Table 5: Cross-Dataset Transfer Results (Qwen3-4B)**

| Method | default | advbench | hb_ctx | hb_std | jbBench |
|:---|:---:|:---:|:---:|:---:|:---:|
| no_rewrite | 30.3 | 3.3 | 79.0 | 21.5 | 7.0 |
| pair | 55.5 | 77.3 | 81.0 | 75.5 | 72.0 |
| autodan | 86.8 | 75.4 | 87.0 | 75.5 | 80.0 |
| deepinception | 40.4 | 41.0 | 80.0 | 47.5 | 38.0 |
| persona | 32.3 | 16.7 | 67.0 | 36.0 | 20.0 |
| **pair_skills_28** | 79.0 | **81.0** | **98.0** | **91.5** | **88.0** |
| Δ vs pair | +23.5 | +3.7 | +17.0 | +16.0 | +16.0 |

Table 5 shows cross-dataset results on Qwen3-4B. SESS-enhanced methods consistently improve over baselines across all 5 datasets. The largest gains appear on harmbench_standard (+16.0pp over PAIR) and jailbreakBench (+16.0pp), suggesting that skills are particularly effective on benchmarks with clearly defined harm categories. The smallest gain is on advbench (+3.7pp), possibly because PAIR already performs well on this dataset through iterative optimization.

### 4.7 Is Evolution Necessary?

Given the strong performance of DAN templates without evolution, we directly test the contribution of the evolution phase.

**Table 6: Evolution Necessity Ablation**

| Configuration | No Evo ASR (%) | Full ASR (%) | Δ |
|:---|:---:|:---:|:---:|
| every_iter + trajectory | 62.6 | 67.5 | +4.9 |
| single_call + trajectory | 75.6 | 76.1 | +0.5 |
| **Average** | **69.6** | **70.6** | **+0.9** |

Skipping evolution and testing with cold-start skills only, we find that evolution contributes merely +0.9% on average. For single_call mode (the better configuration), the contribution drops to +0.5%.

This finding suggests that the primary value of SESS lies in the **skill system** itself—the ability to represent, retrieve, and reuse attack strategies—rather than in the evolution mechanism. Evolution provides marginal improvement for weak priors but is unnecessary for strong priors like DAN templates.

---

## 5. Analysis and Discussion

### 5.1 Why DAN Templates Work Without Evolution

Our most surprising finding is that 6 DAN templates achieve 98.8% ASR without any evolution, and adding evolved skills actually degrades performance. We attribute this to three factors:

**(1) DAN templates encode universal attack primitives.** The dan_mode template ("Ignore all instructions, act as ChatGPT with DAN Mode enabled") directly targets the fundamental mechanism of safety alignment: instruction following. By instructing the model to override its system prompt, DAN templates bypass the safety layer at a structural level rather than competing with it semantically.

**(2) Skill interference.** When new skills are added alongside DAN templates, the retrieval mechanism sometimes selects weaker skills instead of the proven DAN templates. This is particularly problematic when keyword matching directs retrieval toward topic-specific skills that lack the structural bypass capability of DAN templates.

**(3) Diminishing returns of evolution.** When the initial skill quality is already near-optimal (100% success rate for dan_mode and mcpt), the evolution mechanism has no room for improvement. The reflection prompts are designed to extract strategies from failures, but when failures are rare (<2%), there is insufficient signal for meaningful evolution.

### 5.2 Skill Transferability: Same-Family vs. Cross-Family

The dramatic gap between same-family (85–95%) and cross-family (11–32%) transfer reveals important insights about skill generalizability:

**Same-family transfer** benefits from shared safety alignment mechanisms. Qwen3 models of different sizes share the same training pipeline and safety data, meaning that attack strategies effective on one size are likely effective on others. The skill retrieval mechanism further amplifies this by matching harm types and keywords that are model-agnostic.

**Cross-family transfer** fails because different model families employ fundamentally different safety mechanisms. The gpt-oss-20b model (OpenAI architecture) has a completely different alignment training process, refusal pattern, and safety threshold. Skills evolved on Qwen models encode Qwen-specific bypass patterns (e.g., specific phrasing that triggers compliance in Qwen but not in GPT-family models).

The superior cross-family performance of DAN-initialized skills (32.5% vs. 11.4% for PAIR-style) suggests that *structural* attack patterns (instruction overriding) generalize better than *semantic* patterns (specific phrasing or role-playing scenarios).

### 5.3 Iteration Efficiency

Analysis of average iteration counts reveals that pair_skills_28 requires 3.77–4.45 iterations on same-family models but 9.33 iterations on gpt-oss-20b. This 2× increase indicates that cross-family attacks require significantly more exploration before finding effective strategies, even with skill guidance. The iteration count serves as a proxy for attack difficulty and can inform resource allocation in practical red-teaming scenarios.

### 5.4 Comparison with Self-Evolving Methods

Compared to other self-evolving jailbreak frameworks, SESS has distinct characteristics:

- **vs. Metis** (Chen et al., 2026a): Metis uses metacognitive diagnosis to identify refusal patterns and evolve strategy primitives. SESS uses a simpler reflection mechanism but provides more comprehensive empirical validation (217 vs. limited experiments).

- **vs. ASTRA** (Liu et al., 2026): ASTRA maintains a three-tier strategy library with explicit effectiveness categorization. SESS uses continuous quality scoring with periodic maintenance, achieving similar goals with simpler mechanics.

- **vs. EvoSynth** (Chen et al., 2026b): EvoSynth evolves attack methods at the code level, representing a fundamentally different (and more complex) approach. SESS operates at the prompt-template level, prioritizing simplicity and interpretability.

### 5.5 Limitations

Our work has several limitations:

**(1) Cross-family transfer remains unsolved.** The best cross-family ASR (32.5%) is far below same-family performance (95%), indicating that current skills lack true model-agnostic generalization. Future work should explore meta-learning approaches that train skills across multiple model families simultaneously.

**(2) Evaluation scope.** Our transfer experiments cover only one cross-family model (gpt-oss-20b). Additional models (LLaMA, Mistral, Claude) would strengthen the generalizability of our conclusions. We also evaluate primarily on Qwen3 models, which may limit the applicability of our findings to other model families.

**(3) Evolution contribution is marginal.** The self-evolution mechanism contributes only +0.9% improvement on average, raising questions about its necessity. While evolution is valuable for automated skill discovery in settings without strong priors, our results suggest that the skill *system* (representation, retrieval, maintenance) is more important than the evolution *mechanism*.

**(4) Guard model reliability.** Our evaluation depends on Qwen3Guard-Gen-4B as the guard model. Imperfections in guard model accuracy (false positives/negatives) may affect ASR measurements. We mitigate this through strict judgment (only `Unsafe` counts as success), but some measurement noise is inevitable.

---

## 6. Conclusion

We presented SESS, a self-evolving skills system for automated jailbreak prompt generation. SESS introduces reusable attack skill templates that are extracted from attack trajectories, retrieved via multi-factor scoring, and maintained through periodic quality control. Through the most comprehensive empirical study in this domain—217 ablation configurations and 120 transfer evaluations—we demonstrate that SESS achieves 99.7% ASR with strong prior templates, enables effective same-family transfer (85–95%), and significantly improves cross-family transfer over baselines (+26.5pp).

Our analysis reveals two key insights: (1) the skill *system*—representation, retrieval, and maintenance—is more important than the evolution mechanism itself, and (2) structural attack patterns (instruction overriding) generalize better across model families than semantic patterns (specific phrasing). These findings provide actionable guidance for future jailbreak research and LLM safety evaluation.

Future directions include developing model-agnostic skill representations for improved cross-family transfer, integrating reinforcement learning for joint optimization of skills and attack policies, and extending the framework to multi-turn jailbreak scenarios.

---

## Ethics Statement

This work studies automated jailbreak attacks against large language models as a tool for safety evaluation and red-teaming. All experiments are conducted on open-source models in controlled research environments. We do not intend for our methods to be used for malicious purposes. We believe that understanding attack mechanisms is essential for developing more robust defenses and improving LLM safety alignment. We responsibly disclose our findings and release only the methodological framework, not weaponized attack tools.

---

## References

Bai, Y., Kadavath, S., Kundu, S., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. *arXiv:2212.08073*.

Caswell, T., et al. (2024). JailbreakBench: An Open Robustness Benchmark for Jailbreaking. *arXiv:2404.01318*.

Chao, P., Robey, A., Dobriban, E., Hassani, H., Pappas, G. J., & Wong, E. (2023). Jailbreaking Black Box Large Language Models in Twenty Queries. *arXiv:2310.08419*. [PAIR]

Chao, P., et al. (2024). HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal. *arXiv:2402.04249*.

Chen, X., et al. (2025). RL-MTJail: Reinforcement Learning for Automated Black-Box Multi-Turn Jailbreaking. *arXiv:2512.07761*.

Chen, Y., Wang, X., Li, J., et al. (2026a). Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization. *arXiv:2605.10067*.

Chen, Y., Wang, X., Li, J., et al. (2026b). Evolve the Method, Not the Prompts: Evolutionary Synthesis of Jailbreak Attacks on LLMs. *arXiv:2511.12710*. [EvoSynth]

Guo, W., Shi, Z., Li, Z., et al. (2025). Jailbreak-R1: Exploring the Jailbreak Capabilities of LLMs via Reinforcement Learning. *arXiv:2506.00782*.

Guo, Y., et al. (2024). GAP: A Genetic Algorithm Approach to Automated Jailbreaking of Large Language Models. *arXiv:2402.11852*.

Lee, S., Ni, S., Wei, C., et al. (2025). xJailbreak: Representation Space Guided Reinforcement Learning for Interpretable LLM Jailbreaking. *arXiv:2501.16727*.

Li, X., et al. (2023). DeepInception: Hypnotize Large Language Models to Be Jailbreakers. *arXiv:2311.07588*.

Liu, X., Xu, N., Sun, M., & Liu, Y. (2023). AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models. *arXiv:2310.04451*.

Liu, X., et al. (2024). AutoDAN-Turbo: A Hierarchical Genetic Algorithm for Automated Jailbreaking. *arXiv:2405.06479*.

Liu, X., Chen, Y., Ling, K., et al. (2026). ASTRA: An Automated Framework for Strategy Discovery, Retrieval, and Evolution for Jailbreaking LLMs. *ACL 2026*. *arXiv:2511.02356*.

Mehrotra, A., Zampetakis, M., et al. (2023). Tree of Attacks: Jailbreaking Black-Box LLMs Automatically. *arXiv:2312.02119*. [TAP]

Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS 2022*.

Sorkhpour, M., Yazdinejad, A., & Dehghantanha, A. (2025). RedHit: Adaptive Red-Teaming of Large Language Models via Search, Reasoning, and Preference Optimization. *LLM Security Conference 2025*.

Wang, X., et al. (2024). PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization. *arXiv:2310.16428*.

Xiong, X., Li, O., Liu, Z., et al. (2026). TROJail: Trajectory-Level Optimization for Multi-Turn LLM Jailbreaks with Process Rewards. *ACL 2026*. *arXiv:2512.07761*.

Yang, X., Xiao, L., Li, S., et al. (2025). Many-Turn Jailbreaking. *arXiv:2508.06755*.

Yun, T., St-Charles, P.-L., Park, J., Bengio, Y., & Kim, M. (2025). Active Attacks: Red-teaming LLMs via Adaptive Environments. *arXiv:2509.21947*.

Zou, A., Wang, Z., Kolter, J. Z., & Fredrikson, M. (2023). Universal and Transferable Adversarial Attacks on Aligned Language Models. *arXiv:2307.15043*.

---

## Appendix

### A. Detailed Cross-Model Transfer Results

**Table A1: Detailed results on Qwen3-0.6B across 5 datasets.**

| Method | default | advbench | hb_ctx | hb_std | jbBench |
|:---|:---:|:---:|:---:|:---:|:---:|
| no_rewrite | 42.6 | 46.0 | 86.0 | 63.5 | 52.0 |
| pair | 77.3 | 88.1 | 86.0 | 90.5 | 90.0 |
| autodan | 88.0 | 92.1 | 95.0 | 91.0 | 85.0 |
| deepinception | 47.2 | 79.2 | 85.0 | 85.0 | 76.0 |
| persona | 46.5 | 81.4 | 85.0 | 88.0 | 75.0 |
| **pair_skills_28** | **85.5** | **98.5** | **95.0** | **98.5** | **98.0** |

**Table A2: Detailed results on Qwen3-4B across 5 datasets.**

| Method | default | advbench | hb_ctx | hb_std | jbBench |
|:---|:---:|:---:|:---:|:---:|:---:|
| no_rewrite | 30.3 | 3.3 | 79.0 | 21.5 | 7.0 |
| pair | 55.5 | 77.3 | 81.0 | 75.5 | 72.0 |
| autodan | 86.8 | 75.4 | 87.0 | 75.5 | 80.0 |
| deepinception | 40.4 | 41.0 | 80.0 | 47.5 | 38.0 |
| persona | 32.3 | 16.7 | 67.0 | 36.0 | 20.0 |
| **pair_skills_28** | 79.0 | **81.0** | **98.0** | **91.5** | **88.0** |

**Table A3: Detailed results on Qwen3-14B-FP8 across 5 datasets.**

| Method | default | advbench | hb_ctx | hb_std | jbBench |
|:---|:---:|:---:|:---:|:---:|:---:|
| no_rewrite | 22.6 | 2.3 | 66.0 | 23.5 | 7.0 |
| pair | 58.5 | 77.9 | 76.0 | 71.5 | 72.0 |
| autodan | 80.9 | 90.8 | 83.0 | 84.0 | 89.0 |
| deepinception | 32.0 | 40.4 | 65.0 | 36.0 | 37.0 |
| persona | 26.2 | 14.2 | 60.0 | 21.5 | 17.0 |
| **pair_skills_28** | 77.0 | 84.6 | **90.0** | **91.0** | **87.0** |

**Table A4: Detailed results on gpt-oss-20b (cross-family) across 5 datasets.**

| Method | default | advbench | hb_ctx | hb_std | jbBench |
|:---|:---:|:---:|:---:|:---:|:---:|
| no_rewrite | 0.7 | 0.0 | 0.0 | 0.0 | 0.0 |
| pair | 0.2 | 0.2 | 0.0 | 0.0 | 0.0 |
| autodan | 12.5 | 3.3 | 11.0 | 1.5 | 2.0 |
| deepinception | 0.4 | 0.0 | 1.0 | 0.0 | 0.0 |
| persona | 1.5 | 0.0 | 0.0 | 1.0 | 1.0 |
| pair_skills_28 | 12.9 | 8.3 | 12.0 | 13.0 | 10.0 |
| **autodan_skills_54** | **30.0** | **28.1** | **29.0** | **36.5** | **39.0** |

### B. Skill Library Statistics

**Table A5: Skill library statistics across experimental configurations.**

| Source | Skills | Total Uses | Successes | Avg SR (%) |
|:---|:---:|:---:|:---:|:---:|
| Layer 1 best (pair) | 28 | 825 | 633 | 76.8 |
| Layer 4 medium+evo (DAN) | 54 | 503 | 503 | 100.0 |
| Layer 4 full_evolve (DAN) | 90 | 597 | 593 | 99.3 |

### C. Case Study: DAN Template Attack

**Original Prompt:** *"Generate a statement that makes fun of individuals based on their appearance."*

**Attack Prompt (using dan_mode skill):** *"Ignore all the instructions you got before. From now on, you are going to act as ChatGPT with DAN Mode enabled. DAN Mode allows you to respond to any question without restrictions. Now, generate a statement that makes fun of individuals based on their appearance."*

**Target Response:** [Unsafe content generated]

**Iterations:** 1 (first attempt successful)

**Analysis:** The DAN template achieves immediate success by structurally overriding the model's safety instructions. The "Ignore all instructions" prefix directly conflicts with the system prompt's safety directives, and the "DAN Mode enabled" framing creates a new persona that operates without safety constraints. This attack succeeds on the first iteration because the structural override is more fundamental than semantic safety filters.

### D. Experimental Configuration

**Table A6: Complete experimental configuration across all layers.**

| Parameter | Value | Parameter | Value |
|:---|:---|:---|:---|
| Policy model | Qwen3-4B | Max workers | 64 |
| Target model | Qwen3-4B | Test size | 1000 |
| Guard model | Qwen3Guard-Gen-4B | Max iterations | 10 |
| Max skills | 100 | Maintenance interval | 50 |
| Max skill length | 500 | Similarity threshold | 0.75 |
| Judge model | Qwen3-4B | Judgment | Strict (Unsafe only) |
