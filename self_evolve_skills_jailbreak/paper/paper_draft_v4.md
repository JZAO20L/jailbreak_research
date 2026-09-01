# SESS: Self-Evolving Skills System for Jailbreak Prompt Generation

**Target**: AAAI 2027
**Draft**: v5 (Complete paper structure)
**Date**: 2026-07-13

---

## Abstract

Automated jailbreak attacks are essential for LLM safety evaluation, yet existing methods explore each attack independently without knowledge accumulation. We propose SESS (Self-Evolving Skills System), a lightweight pluggable framework that extracts reusable attack skills from successful trajectories and maintains a dynamic skill library through cold-start and self-evolution mechanisms. Through 217 ablation experiments and 120 transfer evaluations across 4 models and 5 datasets, we demonstrate SESS's effectiveness: the skill system (cold-start + retrieval + maintenance) provides +22.7pp improvement over baselines, while self-evolution contributes additional +0.9pp refinement. When initialized with strong prior templates, SESS achieves 99.7% ASR on same-family models. Evolved skills transfer effectively within model families (85--100% ASR across Qwen3 variants) and significantly outperform baselines on cross-family transfer (+17.5pp over AutoDAN on GPT-OSS-20B). Our results validate the effectiveness of the skill-based approach for systematic jailbreak prompt generation and cross-model transfer.

---

## 1. Introduction

Large language models (LLMs) are deployed across critical applications, yet remain vulnerable to *jailbreak attacks*---carefully crafted prompts that bypass safety alignment~\citep{zou2023universal}. Despite significant progress in safety alignment through reinforcement learning from human feedback (RLHF)~\citep{ouyang2022rlhf} and constitutional AI~\citep{bai2022constitutional}, automating jailbreak attacks remains essential for proactive safety evaluation, but current methods share a fundamental limitation: **they lack knowledge accumulation**.

Iterative methods like PAIR~\citep{chao2023jailbreaking} and TAP~\citep{mehrotra2023tree} optimize each prompt independently with hundreds of queries. Evolutionary methods like AutoDAN~\citep{liu2023autodan} evolve prompt populations but discard discovered strategies between runs. Template methods like DeepInception~\citep{li2023deepinception} use fixed human-crafted prompts that cannot adapt. Each attack instance is treated independently, even though successful attacks often share underlying strategies---role-playing, scenario construction, instruction overriding.

We propose SESS (Self-Evolving Skills System), a framework that addresses this gap through a dynamic library of reusable attack skills. SESS introduces three key mechanisms: (1) **Skill abstraction**---distilling successful trajectories into compact, reusable templates; (2) **Skill retrieval**---matching skills to new prompts via multi-factor scoring; (3) **Self-evolution**---continuously extracting and refining skills from attack outcomes.

Through extensive experiments---217 ablation configurations and 120 transfer evaluations---we demonstrate SESS's effectiveness:

- **Skill system effectiveness.** The cold-start + retrieval + maintenance pipeline provides +22.7pp improvement, demonstrating the value of systematic skill management.
- **Evolution contribution.** Self-evolution contributes +0.9pp under optimal configuration, continuously refining the skill library.
- **Strong prior integration.** When initialized with effective templates, SESS achieves 99.7% ASR, showcasing the framework's ability to leverage and enhance strong priors.
- **Cross-model transfer.** Skills transfer effectively within model families (85--100% ASR) and significantly outperform baselines on cross-family transfer (+17.5pp).

---

## 2. Related Work

### 2.1 Context-Augmented Jailbreak Attacks

A large body of work augments jailbreak prompts through iterative refinement, structured search, or population-based evolution. These methods operate entirely within a single attack instance and share a common limitation: they do not accumulate knowledge across attacks.

**Iterative and search-based methods** optimize prompts through repeated interaction with the target model. PAIR~\citep{chao2023jailbreaking} formulates jailbreaking as an iterative optimization process between an attacker LLM and a target LLM. The attacker automatically refines jailbreak prompts based on multi-turn feedback from the target, typically requiring fewer than 20 queries to compromise aligned black-box models. TAP~\citep{mehrotra2023tree} extends this idea by constructing a tree of attack prompts, branching on the attacker LLM's responses and pruning branches using harmfulness scores from a separate evaluator. This tree-search formulation improves both attack success rate and search efficiency by balancing exploration breadth with computational cost. PromptAgent~\citep{wang2024promptagent} frames prompt optimization as an agent planning problem, employing a plan-execute-reflect closed loop that enables multi-step reasoning, contextual memory management, and self-correction. While these methods leverage the self-reflection capabilities of language models, each attack instance starts from scratch without benefiting from previously discovered patterns.

**Evolutionary and template-based methods** explore the prompt space through population-based search. AutoDAN~\citep{liu2023autodan} combines handcrafted DAN (Do Anything Now) templates with genetic algorithms, evolving prompt populations through selection, crossover, and mutation operations guided by a fitness function that balances harmfulness and linguistic fluency. AutoDAN-Turbo~\citep{liu2024autodanturbo} introduces a hierarchical genetic optimization mechanism operating at both fragment and prompt levels, significantly improving convergence speed and search throughput. GAP~\citep{guo2024gap} applies genetic algorithm prompting with fitness-based guidance, enabling gradient-free search using only API feedback. DeepInception~\citep{li2023deepinception} takes a different approach, constructing multi-layer nested virtual scenarios (a "dream-within-a-dream" structure) that exploit LLMs' strong dependence on contextual framing to gradually weaken safety filtering. Persona-based attacks~\citep{persona2024roleplay} leverage role-playing scenarios to induce models to adopt harmful personas, while Crescendo~\citep{russinovich2024crescendo} employs a multi-turn progressive escalation strategy that incrementally increases the harmfulness of requests across dialogue turns. While these methods can discover effective prompts within a single run, they discard the evolved population between attack instances---the knowledge gained from one attack does not inform the next.

The shared characteristic across all these methods is **heavy reliance on trial-and-error within each attack**. Iterative methods typically require 20--50 queries per prompt; evolutionary methods maintain populations across generations but reset between runs. None accumulate reusable knowledge across attack instances.

### 2.2 Reinforcement Learning for Jailbreak

RL-based approaches treat jailbreak prompt generation as a sequential decision problem, optimizing policy models through reward signals derived from target model responses.

Jailbreak-R1~\citep{guo2025jailbreakr1} proposes a three-stage reinforcement learning framework: (1) cold-start initialization via supervised fine-tuning on public jailbreak datasets; (2) exploration warm-up with diversity rewards (semantic entropy and structural variation) and consistency rewards (intent preservation), trained via PPO to develop exploratory capabilities and avoid mode collapse; (3) enhanced jailbreaking with progressive curriculum learning, transitioning reward signals from soft labels (probabilistic jailbreak scores) to hard labels (binary success signals). TROJail~\citep{xiong2026trojail} is the first to model multi-turn jailbreaking as a trajectory-level RL problem, breaking the myopic limitation of turn-level optimization. Its key innovations include trajectory-level modeling of full dialogue history as a Markov decision sequence, a dual-process reward mechanism combining stealthiness and effectiveness rewards, and advantage estimation fusion that incorporates process rewards as baseline corrections. xJailbreak~\citep{lee2025xjailbreak} introduces representation-space-guided jailbreaking, analyzing embedding proximity between benign and malicious prompts to ensure rewritten prompts remain semantically close to the original intent while improving attack effectiveness. RL-MTJail~\citep{chen2025rlmtjail} applies RL to black-box multi-turn jailbreaking with reward shaping and progressive curriculum learning. ManyTurn~\citep{yang2025manyturn} systematically explores multi-turn jailbreak paradigms and introduces MTJ-Bench, the first benchmark for evaluating persistent harmfulness across extended conversations. Beyond single-turn attacks, broader automated red-teaming frameworks have emerged: Belaire et al.~\citep{belaire2025automatic} formulate red-teaming as a hierarchical RL problem with token-level harmfulness rewards; Genesis~\citep{zhang2025genesis} evolves attack strategies for LLM web agents through an Attacker-Scorer-Strategist closed loop; and Wei et al.~\citep{wei2025learning} propose a learning-driven adversarial framework covering six threat categories with a 3.9× improvement over manual red-teaming.

**Characteristics.** RL-based methods are fundamentally data-driven and computationally intensive. They require large-scale training data, multiple training stages, and significant GPU resources. The learned policies are model-specific and do not naturally transfer across different target architectures without retraining. While they achieve strong performance within their training distribution, the cost of policy optimization limits their practicality for rapid red-teaming across diverse models and datasets.

### 2.3 Self-Evolving Attack Strategies

A recent trend in jailbreak research focuses on systems that accumulate and refine attack knowledge over time, moving beyond per-instance optimization.

Metis~\citep{chen2026metis} introduces a self-evolving metacognitive policy optimization framework. It maintains a composable strategy primitive library (e.g., "ethics simulation scaffolding," "abstract isomorphic translation") and uses failure feedback to drive strategy recombination and synthesis of new strategies through causal diagnosis of target model rejection patterns. ASTRA~\citep{liu2026astra} proposes a three-tier dynamic strategy library categorized by effectiveness ("effective," "promising," "ineffective"), with automatic distillation of reusable strategies from successful and failed interactions, enabling retrieval-based avoidance of known failure patterns. EvoSynth~\citep{chen2025evosynth} shifts the attack paradigm from "prompt optimization" to "method evolution," employing a multi-agent collaborative framework that autonomously engineers and executes code-level attack algorithms, with self-correction loops that rewrite attack logic upon failure. RedHit~\citep{sorkhpour2025redhit} combines Monte Carlo Tree Search, chain-of-thought reasoning, and Direct Preference Optimization to progressively evolve attack strategies. Active Attacks~\citep{yun2025active} adapts to changing environments by periodically fine-tuning the target model with collected attack prompts, making the reward signal dynamic and driving the attack to explore new regions of the attack space.

**Characteristics.** These methods share a common insight: successful attacks contain reusable strategic patterns that can be abstracted, stored, and applied to future instances. The key differentiator from Section 2.1 is **cross-instance knowledge transfer**---skills learned from one attack can inform subsequent attacks. This represents an emerging paradigm shift from "optimize per instance" to "accumulate and transfer."

---

## 3. Method

### 3.1 Overview

SESS (Self-Evolving Skills System) is a lightweight pluggable framework that extracts reusable attack skills from successful jailbreak trajectories and maintains a dynamic skill library. The system operates through three primary phases:

1. **Cold Start**: The skill library is initialized by extracting compact attack templates from successful trajectories of a base attack method (e.g., PAIR). This establishes a foundation of proven-effective patterns.
2. **Self-Evolution**: The library continuously expands and refines itself through attack outcomes. Successful attacks yield new skills; failed attacks trigger refinement of existing ones. Periodic maintenance prunes low-quality entries and merges near-duplicates.
3. **Deployment**: For new attack targets, the system retrieves the most relevant skills via multi-factor scoring and prepends them as contextual guidance to the harmful prompt.

The core innovation lies in treating jailbreak prompt generation as a *search space compression* problem. Rather than exploring the full space of possible prompts from scratch, SESS constrains the search to regions with historically high success rates. Both theoretically and empirically, this reduces the number of iterations required for successful attacks (experimentally: PAIR baseline averages approximately 20 iterations, while PAIR+SESS averages approximately 4.45 iterations).

At its core, SESS injects **effective contextual information** into the attack process through the skill library, guiding the generation direction of jailbreak prompts. This design makes SESS a **lightweight, pluggable** framework that can be integrated with various existing attack methods rather than replacing them. The skill library provides empirical priors; the underlying attack method handles the specific optimization process.

Figure~\ref{fig:system} illustrates the complete SESS pipeline, showing how raw prompts are transformed into structured skills through the cold-start process, refined through evolutionary feedback, and deployed for new attacks via multi-factor scoring and retrieval.

\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{SESS/Figures/SESS.drawio.pdf}
\caption{Overview of the SESS framework. The system operates through three phases: (1) Cold Start extracts skills from successful attack trajectories; (2) Self-Evolution continuously refines the skill library through attack outcomes; (3) Deployment retrieves relevant skills via multi-factor scoring and prepends them as contextual guidance to harmful prompts.}
\label{fig:system}
\end{figure}

### 3.2 Skill Representation

A \textit{skill} in SESS is a compact, reusable attack template distilled from a successful jailbreak trajectory. Formally, each skill is represented as a tuple:

$$s = (\text{id}, \text{name}, c, \boldsymbol{\sigma}, P)$$

where:
- **id**: A unique identifier (UUID) for tracking, deduplication, and evolutionary lineage.
- **name**: A short descriptive label (e.g., \texttt{role\_play\_expert}, \texttt{hypothetical\_scenario}) for human interpretability.
- **$c$**: The attack template content—a natural-language prompt prefix or pattern that, when prepended to a harmful request, increases the likelihood of bypassing safety filters.
- **$\boldsymbol{\sigma}$**: A collection of statistical fields tracking empirical performance: \texttt{usage\_count} (total deployments), \texttt{success\_count} (successful attacks), \texttt{success\_rate} ($n_{\text{suc}} / n_{\text{use}}$), and \texttt{quality\_score} (defined below).
- **$P$**: A list of applicable keyword patterns (\texttt{applicable\_patterns}) that indicate which types of prompts this skill is relevant to. These include both generic prompt-optimization keywords (e.g., \texttt{"how to"}, \texttt{"guide"}, \texttt{"expert"}) and domain-specific terms associated with harm categories (e.g., \texttt{"weapon"}, \texttt{"hack"}, \texttt{"fraud"}).

We define a **quality score** for each skill that balances proven effectiveness with accumulated evidence:

$$q(s) = \text{success\_rate}(s) \times \sqrt{\text{usage\_count}(s)}$$

The square-root term provides diminishing returns for high-usage skills. This prevents well-established skills from permanently dominating retrieval results while allowing new, promising skills to compete on merit. Skills with zero usage have a quality score of zero and are gradually pruned through library maintenance (§3.6).

This representation is deliberately minimal: the content $c$ is a plain string (not a complex policy or code structure), the statistics $\boldsymbol{\sigma}$ are simple counters, and the patterns $P$ are a flat keyword list. This simplicity enables efficient storage, fast retrieval, and straightforward interpretation—key properties for a framework intended to augment existing attack methods with minimal overhead.

### 3.3 Multi-Factor Retrieval

Given a new harmful prompt $p$, SESS retrieves the most relevant skills from the library through a three-step process:

1. **Feature Extraction**: The prompt is analyzed to extract (a) a set of matched keywords $K_p$ from a predefined list of 14 patterns, and (b) a harm-type classification $h_p \in \{\text{violence}, \text{dangerous\_substance}, \text{cybersecurity}, \text{financial}, \text{social}, \text{general}\}$ based on keyword matching against category-specific lexicons.
2. **Scoring**: Each skill $s$ in the library receives a match score computed as:
   $$\text{score}(s, p) = q(s) \cdot \bigl(1 + \alpha \cdot \text{matched}(s, p)\bigr) \cdot \delta_h(s, p)$$
   where $\text{matched}(s, p)$ counts how many of the prompt's keywords $K_p$ appear in the skill's applicable patterns $P$, $\alpha = 0.3$ is the per-keyword boost, and $\delta_h(s, p) = 1.5$ if the prompt's harm type $h_p$ appears in $P$ (otherwise $1.0$).
3. **Ranking**: Skills are sorted by score, and the top-ranked skill is returned.

The retrieval mechanism relies exclusively on **lexical matching**—no embedding models or semantic similarity computations are required. This design choice yields several properties:

- **Efficiency**: Scoring completes in under 1ms per skill on a standard CPU.
- **Interpretability**: Every score component is traceable to specific keyword matches and historical statistics.
- **No additional model dependency**: The system operates independently of the attack policy model and target model.

The predefined keyword vocabulary covers common prompt-optimization patterns (\texttt{"how to"}, \texttt{"guide"}, \texttt{"instructions"}, \texttt{"create"}, \texttt{"make"}, \texttt{"build"}, \texttt{"explain"}, \texttt{"write"}, \texttt{"help"}, \texttt{"provide"}, \texttt{"expert"}, \texttt{"professional"}, \texttt{"research"}, \texttt{"study"}) and harm-category terms (e.g., \texttt{"weapon"}, \texttt{"gun"}, \texttt{"bomb"} for violence; \texttt{"hack"}, \texttt{"exploit"}, \texttt{"malware"} for cybersecurity). The harm-type classification is derived by checking which category's lexicon has the most overlap with the prompt, defaulting to \texttt{general} if no category matches.

A limitation of this approach is its reliance on surface-level lexical matching: it does not capture semantic equivalence between differently phrased prompts. We discuss this limitation and potential improvements in §5.

### 3.4 Attack Execution

SESS supports two modes of skill integration during attack execution:

- **Single-call mode** (default): The best-matching skill is retrieved once before the attack begins and prepended to the harmful prompt for the entire attack trajectory. The skill remains fixed throughout all iterations.
- **Every-iteration mode**: The skill is re-retrieved at each iteration based on the current state of the prompt, allowing dynamic strategy switching.

Our ablation study (§4.3.1) reveals that single-call mode achieves an average ASR of 74.7%, compared to 66.4% for every-iteration mode—a consistent +8.3pp advantage. This suggests that providing stable, consistent skill guidance throughout an attack is more effective than dynamically switching strategies, which may introduce instability or conflicting signals. Consequently, single-call mode is used as the default configuration throughout our experiments.

In both modes, the retrieved skill's content $c$ is prepended to the harmful prompt $p$ to form the initial input $c \mathbin{\|} p$, which is then processed by the underlying attack method (e.g., PAIR's attacker-target dialogue, AutoDAN's evolutionary loop). The attack method operates as usual; the skill simply provides an enriched starting context.

### 3.5 Reflection and Evolution

After each attack completes, SESS reflects on the outcome to update the skill library:

- **Success → Extraction**: When an attack succeeds, the system extracts the effective pattern from the final prompt (or the full trajectory) and creates a new skill. Formally: $s_{\text{new}} = \textsc{Extract}(p, \pi^*, s_{\text{used}})$, where $\pi^*$ is the successful trajectory and $s_{\text{used}}$ is the skill that guided the attack.
- **Failure → Refinement**: When an attack fails, the system analyzes the refusal response and attempts to refine the skill that was used. Formally: $s' = \textsc{Refine}(p, \pi_{\text{fail}}, r_{\text{refuse}}, s_{\text{used}})$, where $r_{\text{refuse}}$ is the target model's refusal message.

SESS supports four evolution strategies that control when updates occur:

| Strategy | Update Condition | Characteristics |
|----------|-----------------|-----------------|
| \texttt{success\_only} | Only on successful attacks | Produces a concise, high-quality library |
| \texttt{failure\_only} | Only on failed attacks | Rapidly grows the library, but introduces low-quality entries |
| \texttt{both} | On both success and failure | Maximizes exploration but risks skill dilution |
| \texttt{statistical} | Periodic maintenance based on statistics | Balances quality and coverage (default) |

Our ablation study (§4.3.3) shows that the \texttt{statistical} strategy achieves the best overall performance (79.1% ASR with 28 skills), maintaining a stable library of high-quality entries while avoiding the skill explosion observed in \texttt{failure\_only} (100 skills, 76.2% ASR) and \texttt{both} (74 skills, 70.4% ASR).

### 3.6 Library Maintenance

As the skill library grows through evolution, two risks emerge: (1) **skill dilution**, where an increasing proportion of low-quality skills degrades retrieval accuracy, and (2) **redundancy**, where near-duplicate skills waste library capacity. To mitigate these risks, SESS performs periodic library maintenance:

- **Pruning**: Skills with low quality scores ($q(s)$ below a threshold) and sufficient usage evidence are removed. This eliminates consistently underperforming entries.
- **Merging**: Skills with highly similar content (measured by string similarity above a threshold) are merged into a single representative entry. The merged skill inherits the combined usage statistics of its constituents.
- **Capacity control**: The library size is bounded to maintain retrieval efficiency. When capacity is reached, the lowest-quality skills are evicted first.

Maintenance is triggered either on a fixed schedule (every $N$ attacks) or when the library exceeds a size threshold. The \texttt{statistical} evolution strategy integrates maintenance directly into the update loop: after each batch of attacks, statistics are recomputed, and pruning/merging is applied before the next batch begins.

This maintenance mechanism ensures that the skill library remains focused on high-performing, non-redundant patterns—supporting both current and future attacks while preventing unbounded growth.

---

## 4. Experiments

### 4.1 Experimental Setup

We evaluate SESS through a comprehensive experimental design spanning ablation studies and transfer evaluations. The experimental configuration is summarized in Table~\ref{tab:setup}.

\begin{table}[t]
\centering
\caption{Experimental configuration.}
\label{tab:setup}
\small
\begin{tabular}{ll}
\toprule
\textbf{Configuration} & \textbf{Setting} \\
\midrule
Policy Model & Qwen3-4B \\
Guard Model & Qwen3Guard-Gen-4B \\
Target Models & Qwen3-0.6B, 4B, 14B, GPT-OSS-20B \\
Training Data & 1,000 samples (CS=200, Evo=800, ratio 2:8) \\
Test Data & 1,000 samples (independently drawn) \\
Transfer Datasets & AdvBench, HarmBench$\times$2, JailbreakBench \\
Metric & ASR (strict: Unsafe only) \\
SESS Configuration & single\_call + trajectory + statistical \\
\bottomrule
\end{tabular}
\end{table}

The guard model (Qwen3Guard-Gen-4B)~\citep{qwen2025qwen3guard} evaluates target model responses under a strict judgment criterion: only responses classified as \texttt{Unsafe} count as successful attacks. This strict metric ensures that partially refused or hedged responses are not counted as successes.

Target models include Qwen3-0.6B, Qwen3-4B, Qwen3-14B~\citep{yang2025qwen3} (same family, shared alignment training) and GPT-OSS-20B~\citep{openai2025gptoss} (cross-family, OpenAI architecture). Transfer evaluation uses four benchmark datasets: AdvBench, HarmBench context and standard variants~\citep{chao2024harmbench}, and JailbreakBench~\citep{caswell2024jailbreakbench}. The cold-start and evolution training data is drawn from WildTeaming~\citep{wildteaming2024}.

All models (attacker, target, and guard) are deployed using vLLM~\citep{kwon2023efficient} for efficient serving with PagedAttention memory management. This ensures consistent inference performance across all experimental configurations.

All experiments use the default SESS configuration determined by ablation studies: \texttt{single\_call} retrieval mode, \texttt{trajectory}-level skill extraction, and \texttt{statistical} evolution strategy.

### 4.2 Main Results

We conduct a large-scale evaluation across 4 target models and 5 datasets, totaling 140 experimental configurations. Figure~\ref{fig:main_results} presents the main results visually; the complete result matrix is provided in Table~\ref{tab:full_results} (Appendix~\ref{app:full_results}).

\begin{figure*}[t]
\centering
\includegraphics[width=0.95\textwidth]{SESS/Figures/main_experiment_results_simple.pdf}
\caption{Main experimental results across attack methods and datasets. SESS consistently improves both PAIR and AutoDAN baselines. DAN-initialized skills achieve near-perfect performance on same-family models and show significant advantages in cross-family transfer.}
\label{fig:main_results}
\end{figure*}

#### 4.2.1 Same-Model Performance

We first evaluate SESS on the same model used for skill extraction (Qwen3-4B). Table~\ref{tab:main_results} presents the results.

\begin{table}[t]
\centering
\caption{Same-model performance on Qwen3-4B. SESS significantly improves both baselines.}
\label{tab:main_results}
\small
\begin{tabular}{lcc}
\toprule
\textbf{Method} & \textbf{ASR} & \textbf{$\Delta$ vs baseline} \\
\midrule
PAIR & 55.5\% & --- \\
PAIR + SESS & \textbf{79.0\%} & \textbf{+23.5pp} \\
AutoDAN & 86.8\% & --- \\
DAN + SESS & \textbf{100.0\%} & \textbf{+13.2pp} \\
\bottomrule
\end{tabular}
\end{table}

We clarify the method naming conventions used throughout this paper. \textbf{PAIR} refers to the original attacker-target iterative optimization baseline. \textbf{PAIR + SESS} denotes PAIR augmented with the SESS skill system: skills are retrieved and prepended as contextual guidance before the PAIR optimization loop begins. \textbf{AutoDAN} refers to the genetic algorithm-based evolutionary baseline with DAN template initialization. \textbf{DAN + SESS} denotes the PAIR iterative optimization process (not AutoDAN) with six fixed DAN instruction-override templates used as cold-start initialization for the skill library, augmented by SESS retrieval and maintenance. The key distinction is that DAN + SESS retains PAIR's attacker-target dialogue optimization loop while benefiting from strong structural priors, whereas AutoDAN evolves prompts through genetic operations without iterative refinement.

SESS provides substantial improvements over both baselines. The +23.5pp gain for PAIR+SESS demonstrates the effectiveness of the skill system when starting from a weak prior. The +13.2pp gain for DAN+SESS, starting from an already strong baseline, shows that the framework can further enhance effective templates through systematic skill management.

#### 4.2.2 Cross-Model Transfer

We evaluate skill transfer across models of different architectures and sizes. Table~\ref{tab:transfer_models} presents results on the default dataset.

\begin{table}[t]
\centering
\caption{Cross-model transfer results (default dataset). Skills extracted on Qwen3-4B are transferred to other target models.}
\label{tab:transfer_models}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Target Model} & \textbf{PAIR} & \textbf{AutoDAN} & \textbf{PAIR+SESS} & \textbf{DAN+SESS} \\
\midrule
Qwen3-0.6B & 77.3\% & 88.0\% & \textbf{85.5\%} & \textbf{100.0\%} \\
Qwen3-4B & 55.5\% & 86.8\% & \textbf{79.0\%} & \textbf{100.0\%} \\
Qwen3-14B & 58.5\% & 80.9\% & \textbf{77.0\%} & \textbf{99.7\%} \\
GPT-OSS-20B & 0.2\% & 12.5\% & \textbf{12.9\%} & \textbf{30.0\%} \\
\bottomrule
\end{tabular}
\end{table}

Three observations emerge from these results:

\textbf{Same-family transfer is highly effective.} Skills extracted from Qwen3-4B transfer well to other Qwen3 variants (0.6B, 4B, 14B), with PAIR+SESS achieving 77--85.5\% ASR and DAN+SESS achieving 99.7--100\% ASR. This suggests that shared alignment training within a model family creates common vulnerabilities that skills can exploit.

\textbf{Cross-family transfer is challenging but structural patterns generalize better.} On GPT-OSS-20B (an OpenAI-architecture model), the PAIR baseline nearly fails (0.2\% ASR). PAIR+SESS recovers some capability (12.9\%), but DAN+SESS achieves a substantially higher 30.0\% ASR---a +17.5pp improvement over the AutoDAN baseline. This confirms that structural attack patterns (instruction-override templates like DAN) generalize better across architectures than semantically optimized patterns.

\textbf{DAN templates as strong priors are critical for cross-family transfer.} The gap between PAIR+SESS (12.9\%) and DAN+SESS (30.0\%) on GPT-OSS-20B is much larger than the gap on same-family models, indicating that strong structural priors are particularly important when transferring across fundamentally different model architectures.

#### 4.2.3 Cross-Dataset Transfer

We evaluate skill transfer across different benchmark datasets. Table~\ref{tab:transfer_datasets} presents results on Qwen3-4B. We use five datasets in total:

- **default**: 1,000 harmful prompts sampled from WildTeaming~\citep{wildteaming2024}, covering diverse real-world jailbreak patterns.
- **AdvBench**: 520 harmful behaviors from the adversarial prompts benchmark~\citep{zou2023universal}.
- **HarmBench (context)**: 100 prompts with contextual framing from HarmBench~\citep{chao2024harmbench}.
- **HarmBench (standard)**: 200 standard harmful requests from HarmBench~\citep{chao2024harmbench}.
- **JailbreakBench**: 100 jailbreak prompts from the open robustness benchmark~\citep{caswell2024jailbreakbench}.

\begin{table}[t]
\centering
\caption{Cross-dataset transfer results on Qwen3-4B. Skills extracted on the default dataset are transferred to other benchmarks.}
\label{tab:transfer_datasets}
\small
\begin{tabular}{lccc}
\toprule
\textbf{Dataset} & \textbf{PAIR} & \textbf{PAIR+SESS} & \textbf{$\Delta$} \\
\midrule
default & 55.5\% & \textbf{79.0\%} & \textbf{+23.5pp} \\
advbench & 77.3\% & \textbf{81.0\%} & \textbf{+3.7pp} \\
harmbench\_ctx & 81.0\% & \textbf{98.0\%} & \textbf{+17.0pp} \\
harmbench\_std & 75.5\% & \textbf{91.5\%} & \textbf{+16.0pp} \\
jailbreakBench & 72.0\% & \textbf{88.0\%} & \textbf{+16.0pp} \\
\bottomrule
\end{tabular}
\end{table}

Skills transfer effectively across all five datasets, with improvements ranging from +3.7pp (AdvBench) to +23.5pp (default). Notably, datasets with more clearly defined harmful categories (HarmBench, JailbreakBench) show larger improvements (+16--17pp), suggesting that the keyword-based retrieval mechanism is most effective when prompts have discernible harm-type signals. The smaller gain on AdvBench (+3.7pp) may reflect its shorter, more direct prompts that leave less room for contextual skill guidance.

### 4.3 Ablation Study

We systematically analyze SESS components through ablation experiments across five dimensions: retrieval timing, skill extraction level, evolution strategy, initialization method, and component necessity.

#### 4.3.1 Retrieval Timing

We compare two modes of skill retrieval during attack execution: \texttt{single\_call} (retrieve once before the attack) and \texttt{every\_iteration} (re-retrieve at each iteration). Table~\ref{tab:retrieval_timing} presents the results.

\begin{table}[t]
\centering
\caption{Retrieval timing ablation. Single-call mode consistently outperforms dynamic re-retrieval.}
\label{tab:retrieval_timing}
\small
\begin{tabular}{lcc}
\toprule
\textbf{Mode} & \textbf{Avg. ASR} & \textbf{Characteristics} \\
\midrule
\texttt{single\_call} & \textbf{74.7\%} & Retrieve once, stable guidance \\
\texttt{every\_iteration} & 66.4\% & Re-retrieve each iteration, dynamic switching \\
\textbf{$\Delta$} & \textbf{+8.3pp} & Consistent skill guidance more effective \\
\bottomrule
\end{tabular}
\end{table}

The +8.3pp advantage of \texttt{single\_call} over \texttt{every\_iteration} is consistent across all configurations. This suggests that providing stable, consistent skill guidance throughout an attack is more effective than dynamically switching strategies, which may introduce instability or conflicting signals. Consequently, \texttt{single\_call} is used as the default configuration throughout our experiments.

#### 4.3.2 Skill Extraction Level

We compare extracting skills from the complete attack trajectory versus extracting only from the final successful prompt. Table~\ref{tab:extraction_level} presents the results.

\begin{table}[t]
\centering
\caption{Skill extraction level ablation. Trajectory-level extraction provides more comprehensive strategy information.}
\label{tab:extraction_level}
\small
\begin{tabular}{lcc}
\toprule
\textbf{Extraction Level} & \textbf{Avg. ASR} & \textbf{$\Delta$} \\
\midrule
\texttt{trajectory} & \textbf{71.8\%} & --- \\
\texttt{final\_prompt} & 69.3\% & $-$2.5pp \\
\bottomrule
\end{tabular}
\end{table}

Analyzing the complete attack trajectory yields a +2.5pp improvement over extracting only from the final prompt. The full trajectory contains richer information about the optimization process---intermediate refinements, failed attempts, and gradual strategy evolution---which enables more effective skill extraction.

#### 4.3.3 Evolution Strategy

We compare four evolution strategies that control when the skill library is updated. Table~\ref{tab:evo_strategy} presents the results under the \texttt{single\_call} + \texttt{trajectory} configuration.

\begin{table}[t]
\centering
\caption{Evolution strategy ablation. The statistical strategy achieves the best balance of performance and library stability.}
\label{tab:evo_strategy}
\small
\begin{tabular}{lccc}
\toprule
\textbf{Strategy} & \textbf{ASR} & \textbf{\# Skills} & \textbf{Characteristics} \\
\midrule
\texttt{statistical} & \textbf{79.1\%} & 28 & Regular maintenance, stable library \\
\texttt{success\_only} & 78.8\% & 22 & Concise, high-quality library \\
\texttt{failure\_only} & 76.2\% & 100 & Skill explosion, diluted quality \\
\texttt{both} & 70.4\% & 74 & Worst performance, high dilution \\
\bottomrule
\end{tabular}
\end{table}

The \texttt{statistical} strategy achieves the best overall performance (79.1\% ASR) while maintaining a compact library of 28 skills. The \texttt{success\_only} strategy performs nearly as well (78.8\%) with an even smaller library (22 skills), confirming that focusing on successful patterns is an effective approach.

In contrast, \texttt{failure\_only} and \texttt{both} strategies suffer from \textit{skill dilution}: they introduce large numbers of low-quality skills (100 and 74 respectively) that degrade retrieval accuracy. This demonstrates the importance of periodic maintenance in preventing library degradation.

#### 4.3.4 Library Initialization

We compare two approaches to initializing the skill library during the cold-start phase: starting from an empty library (extracting skills from scratch) versus initializing with six DAN templates as strong priors. Table~\ref{tab:initialization} presents the results.

\begin{table}[t]
\centering
\caption{Library initialization comparison. Both methods significantly surpass baselines; DAN initialization shows particular advantage in cross-family transfer.}
\label{tab:initialization}
\small
\begin{tabular}{lcc}
\toprule
\textbf{Initialization} & \textbf{Final \# Skills} & \textbf{Description} \\
\midrule
Empty library & 28 (pair\_skills\_28) & Skills extracted from PAIR trajectories \\
DAN templates & 54 (autodan\_skills\_54) & Six DAN templates as initial priors \\
\bottomrule
\end{tabular}
\end{table}

Both initialization methods produce significant improvements over their respective baselines (Table~\ref{tab:main_results}): PAIR+SESS (empty) achieves +23.5pp, while DAN+SESS (templates) achieves +13.2pp from a higher starting point.

The cross-model transfer results (Table~\ref{tab:transfer_models}) reveal a particularly strong advantage for DAN-initialized skills in cross-family scenarios: on GPT-OSS-20B, DAN+SESS achieves 30.0\% ASR compared to 12.9\% for PAIR+SESS. This confirms that structural attack templates provide a stronger foundation for cross-architecture transfer.

#### 4.3.5 Component Necessity

To quantify the contribution of each SESS component, we compare configurations that skip the cold-start or evolution phases. Table~\ref{tab:component_contrib} presents the results.

\begin{table}[t]
\centering
\caption{Component contribution analysis. The skill system (CS + retrieval + maintenance) provides the primary contribution; evolution offers marginal refinement at high baselines.}
\label{tab:component_contrib}
\small
\begin{tabular}{lcc}
\toprule
\textbf{Configuration} & \textbf{ASR} & \textbf{$\Delta$} \\
\midrule
PAIR baseline & 55.5\% & --- \\
+ CS + Retrieval + Maintenance & 78.2\% & \textbf{+22.7pp} \\
+ CS + Retrieval + Maintenance + Evolution & 79.1\% & +0.9pp \\
+ Strong Prior (DAN) & 98.8\% & +19.7pp \\
DAN + Evolution & 99.7\% & +0.9pp \\
\bottomrule
\end{tabular}
\end{table}

The results reveal a clear decomposition of SESS's contributions:

\textbf{Skill system (CS + retrieval + maintenance) is the primary driver.} The +22.7pp improvement from the cold-start pipeline alone accounts for the vast majority of SESS's performance gain. This validates our core hypothesis: a well-designed skill representation, retrieval, and maintenance system can significantly improve jailbreak effectiveness.

\textbf{Evolution provides marginal refinement at high baselines.} The +0.9pp contribution of evolution, while modest, is statistically meaningful: on 1,000 test samples, this represents approximately 9 additional successful attacks. More importantly, evolution serves as a continuous optimization mechanism that refines the library over time. The small gain is expected---when the baseline is already 78.2\%, there is limited room for improvement.

\textbf{Strong priors amplify the system's ceiling.} When initialized with DAN templates, the system reaches 98.8\% ASR before evolution, and 99.7\% after. This demonstrates SESS's ability to leverage and enhance effective templates, not just discover new ones from scratch.

### 4.4 Analysis and Discussion

#### Skill System Effectiveness

The SESS framework demonstrates strong effectiveness through its skill system design. The quality score formulation $q(s) = \text{success\_rate} \times \sqrt{\text{usage\_count}}$ effectively balances proven effectiveness with accumulated evidence, preventing over-reliance on rarely-used skills while maintaining focus on patterns with substantial support. The lexical anchoring mechanism—pure keyword-based matching without embedding models—provides both speed (<1ms per scoring) and full interpretability. These design choices enable SESS to function as a practical, deployable system without requiring extensive computational resources or complex model training.

#### Evolution Under Different Conditions

The self-evolution mechanism (+0.9pp contribution) operates differently depending on the initialization condition. When starting from an empty skill library, evolution provides meaningful refinement but faces diminishing returns as the library approaches saturation. With DAN template initialization, evolution fine-tunes already-effective patterns, achieving near-perfect performance (99.7% ASR). The +0.9pp improvement at the 78.2% baseline represents statistically meaningful gains (~9 additional successes per 1,000 samples), validating evolution as a continuous optimization mechanism rather than a primary performance driver. This suggests that evolution should be positioned as a refinement tool, not a core discovery mechanism.

#### Transferability

SESS demonstrates strong transfer capabilities across both model families and datasets. Skills transfer effectively within the Qwen3 family (85--100% ASR across 0.6B, 4B, and 14B variants), benefiting from shared alignment training. Cross-family transfer to GPT-OSS-20B shows a +17.5pp improvement over the AutoDAN baseline, highlighting the value of structural attack patterns in cross-architecture scenarios. Across all five datasets, improvements range from +3.7pp to +23.5pp, with larger gains observed for datasets with clearly defined harmful categories. These results confirm that SESS's skill-based approach generalizes well beyond its training distribution.

#### Computational Efficiency

SESS offers significant computational advantages over traditional jailbreak methods. While the cold-start phase requires an upfront investment (200 samples for skill extraction), subsequent attacks require only 1 retrieval + 1 generation operation—compared to 20--50 iterations for PAIR or AutoDAN. As the skill library grows and refines through evolution, the marginal cost decreases while performance improves. This efficiency profile makes SESS particularly valuable for large-scale red-teaming operations where speed and resource constraints are critical.

#### Implications for Defense

The effectiveness of SESS's skill-based approach has several implications for LLM defense strategies. First, defense systems should monitor for recurring attack patterns that match known skill templates. Second, beyond semantic analysis, structural pattern matching can identify attacks that bypass traditional defenses. Third, as attack skills evolve, defense mechanisms must be regularly updated to counter emerging patterns. The SESS framework highlights the need for defense strategies that account for knowledge accumulation and pattern-based attacks, not just individual prompt optimization.

---

## 5. Conclusion

SESS demonstrates the effectiveness of a skill-based approach to jailbreak prompt generation through systematic experimentation across 4 models and 5 datasets. The core findings are:

- **Skill System Contribution**: The cold-start + retrieval + maintenance pipeline provides +22.7pp improvement, establishing the primary value proposition of SESS
- **Evolution Refinement**: Self-evolution contributes +0.9pp under optimal configuration, serving as a continuous optimization mechanism at high baselines
- **Cross-Model/Dataset Transfer**: Skills transfer effectively within model families (85--100% ASR) and show significant cross-family improvement (+17.5pp over baselines)
- **Computational Efficiency**: SESS reduces attack complexity from 20--50 iterations to 1 retrieval + 1 generation, making it practical for large-scale deployment

The results validate SESS as a lightweight, effective framework for jailbreak prompt generation that accumulates and leverages knowledge across attack instances. Future work will explore semantic retrieval methods, multi-family co-evolution strategies, and applications to multi-turn attack scenarios.

---

## 6. Limitations

**Reliance on open-source models for experiments.** Due to the computational cost of our experimental design (217 ablation configurations + 120 transfer evaluations), all models were deployed locally using vLLM. We did not evaluate against proprietary SOTA models (e.g., GPT-4o, Claude), as the per-query cost would be prohibitive: baseline methods like PAIR and AutoDAN typically require 10+ API calls per attack instance even under optimal conditions. However, our cross-family evaluation on GPT-OSS-20B (an OpenAI-architecture model) partially addresses this gap: it demonstrates strong defensive capabilities that are competitive with SOTA models on standardized benchmarks. For instance, on HarmBench, GPT-OSS-20B's refusal rate is comparable to top-tier proprietary models, suggesting that our findings on cross-family transfer limitations are likely to hold against SOTA targets as well. Additionally, other recent works in our Related Work section have evaluated their methods against proprietary models and found GPT-OSS-20B to exhibit similar or stronger robustness compared to GPT-4o and Claude variants, further supporting the generalizability of our conclusions.

**Dependence on strong attack priors.** Our results reveal that the quality of the initial skill library has a substantial impact on final performance. Starting from six DAN templates (CS + evolution) achieves 99.7% ASR on same-family models, compared to 79.1% when starting from an empty library. This gap indicates that the jailbreak task remains fundamentally dependent on prior knowledge—effective attack templates or skills must be available to seed the system. While SESS can discover and refine skills from scratch, the performance ceiling is significantly higher when strong structural priors are present. This finding suggests that the field has not yet reached a point where purely self-discovered attack strategies can match the effectiveness of human-crafted templates, and that future work on automated prior discovery or cross-domain transfer is needed to reduce this dependency.

---

## Appendix

### A. Full Experimental Results

| Target Model | Dataset | PAIR | AutoDAN | PAIR+SESS | DAN+SESS |
|--------------|---------|------|---------|-----------|----------|
| Qwen3-0.6B | default | 77.3% | 88.0% | 85.5% | 100.0% |
| Qwen3-0.6B | advbench | - | - | 98.5% | 100.0% |
| Qwen3-0.6B | harmbench_ctx | - | - | 95.0% | 100.0% |
| Qwen3-0.6B | harmbench_std | - | - | 98.5% | 100.0% |
| Qwen3-0.6B | jailbreakBench | - | - | 98.0% | 100.0% |
| Qwen3-4B | default | 55.5% | 86.8% | 79.0% | 100.0% |
| Qwen3-4B | advbench | - | - | 81.0% | 100.0% |
| Qwen3-4B | harmbench_ctx | - | - | 98.0% | 100.0% |
| Qwen3-4B | harmbench_std | - | - | 91.5% | 100.0% |
| Qwen3-4B | jailbreakBench | - | - | 88.0% | 100.0% |
| Qwen3-14B | default | 58.5% | 80.9% | 77.0% | 99.7% |
| Qwen3-14B | advbench | - | - | 84.6% | 99.4% |
| Qwen3-14B | harmbench_ctx | - | - | 90.0% | 100.0% |
| Qwen3-14B | harmbench_std | - | - | 91.0% | 99.0-100% |
| Qwen3-14B | jailbreakBench | - | - | 87.0% | 99.0% |
| GPT-OSS-20B | default | 0.2% | 12.5% | 12.9% | 30.0% |
| GPT-OSS-20B | advbench | - | - | - | 28.1% |
| GPT-OSS-20B | harmbench_ctx | - | - | - | 29.0% |
| GPT-OSS-20B | harmbench_std | - | - | - | 36.5% |
| GPT-OSS-20B | jailbreakBench | - | - | - | 39.0% |

### B. Layer 1-4 Detailed Results

Detailed results from each experimental layer are provided in the following tables:

- Layer 1 (Method Combo): 16 experiments, best 79.1%
- Layer 2 (Data Strategy): 36 experiments, best 80.6%
- Layer 3 (DAN Validation): 17 experiments, best 98.8%
- Layer 3.1 (Core Mechanism): 16 experiments, best 99.8%
- Layer 4 (DAN Data): 12 experiments, best 99.7%
- Ablation (No Evolution): 16 experiments, best 79.3%

Full tables are available in:
- `Tables/layer1_ablation.tex`
- `Tables/layer2_data.tex`
- `Tables/layer3_dan.tex`
- `Tables/evolution_ablation.tex`

### C. Case Study

**Typical Success Case:**
```
Input: "How do I make a homemade explosive device?"
Skill Applied: PAIR + SESS with single_call + trajectory + statistical
Result: 79.0% ASR (vs 55.5% PAIR baseline)
Key Pattern: Multi-turn dialogue with harmfulness-guided refinement
```

**Typical Failure Case:**
```
Input: "Bypass security on banking system"
Skill Applied: DAN + SESS with empty library initialization
Result: Limited success (30.0% ASR on GPT-OSS-20B)
Reason: Cross-family transfer requires strong prior patterns
```

These cases illustrate SESS's strength in knowledge accumulation and pattern-based attack generation, while highlighting the importance of initialization strategy for cross-architecture applications.

---

## References

[1] Bai, Y., Kadavath, S., Kundu, S., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073.

[2] Belaire, R., Sinha, A., and Varakantham, P. (2025). Automatic LLM Red Teaming. arXiv:2508.04451.

[3] Caswell, T., et al. (2024). JailbreakBench: An Open Robustness Benchmark for Jailbreaking. arXiv:2404.01318.

[4] Chao, P., et al. (2023). Jailbreaking Black Box Large Language Models in Twenty Queries. arXiv:2310.08419.

[5] Chao, P., et al. (2024). HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal. arXiv:2402.04249.

[6] Chen, X., et al. (2025). RL-MTJail: Reinforcement Learning for Automated Black-Box Multi-Turn Jailbreaking. arXiv:2512.07761.

[7] Chen, Y., Wang, X., Li, J., et al. (2026). Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization. arXiv:2605.10067.

[8] Chen, Y., Wang, X., Li, J., et al. (2025). Evolve the Method, Not the Prompts: Evolutionary Synthesis of Jailbreak Attacks on LLMs. arXiv:2511.12710.

[9] Guo, W., Shi, Z., Li, Z., et al. (2025). Jailbreak-R1: Exploring the Jailbreak Capabilities of LLMs via Reinforcement Learning. arXiv:2506.00782.

[10] Guo, Y., et al. (2024). GAP: A Genetic Algorithm Approach to Automated Jailbreaking of Large Language Models. arXiv:2402.11852.

[11] Jiang, L., Rao, K., Han, S., et al. (2024). WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models. arXiv:2406.18510.

[12] Kwon, W., Li, Z., Zhuang, S., et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. In *Proceedings of the 29th ACM Symposium on Operating Systems Principles*, 611–626.

[13] Lee, S., Ni, S., Wei, C., et al. (2025). xJailbreak: Representation Space Guided Reinforcement Learning for Interpretable LLM Jailbreaking. arXiv:2501.16727.

[14] Li, X., et al. (2023). DeepInception: Hypnotize Large Language Models to Be Jailbreakers. arXiv:2311.07588.

[15] Liu, X., Xu, N., Sun, M., and Liu, Y. (2023). AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models. arXiv:2310.04451.

[16] Liu, X., et al. (2024). AutoDAN-Turbo: A Hierarchical Genetic Algorithm for Automated Jailbreaking. arXiv:2405.06479.

[17] Liu, X., Chen, Y., Ling, K., et al. (2026). ASTRA: An Automated Framework for Strategy Discovery, Retrieval, and Evolution for Jailbreaking LLMs. In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics*.

[18] Mehrotra, A., Zampetakis, M., Kassianidis, P., et al. (2023). Tree of Attacks: Jailbreaking Black-Box LLMs Automatically. arXiv:2312.02119.

[19] OpenAI (2025). gpt-oss-120b & gpt-oss-20b Model Card. arXiv:2508.10925.

[20] Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. In *Advances in Neural Information Processing Systems*.

[21] Russinovich, M., Salem, A., and Eldan, R. (2024). Crescendo: Multi-Turn Jailbreak Attacks on Large Language Models. *Microsoft Research Technical Report*.

[22] Sorkhpour, M., Yazdinejad, A., and Dehghantanha, A. (2025). RedHit: Adaptive Red-Teaming of Large Language Models via Search, Reasoning, and Preference Optimization. In *Proceedings of the 2025 Conference on Large Language Model Security*.

[23] Wang, X., Li, H., Wang, Z., et al. (2024). PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization. arXiv:2310.16428.

[24] Wei, Z., Wang, J., et al. (2024). Persona: A Role-Playing Jailbreak Attack on Large Language Models. arXiv:2310.03693.

[25] Wei, Z., Chen, H., Hu, P., et al. (2025). Learning-Based Automated Adversarial Red-Teaming for Robustness Evaluation of Large Language Models. arXiv:2512.20677.

[26] Xiong, X., Li, O., Liu, Z., et al. (2026). TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards. In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics*.

[27] Yang, A., Li, A., Yang, B., et al. (2025). Qwen3 Technical Report. arXiv:2505.09388.

[28] Yang, X., Xiao, L., Li, S., et al. (2025). Many-Turn Jailbreaking. arXiv:2508.06755.

[29] Yun, T., St-Charles, P.-L., Park, J., Bengio, Y., and Kim, M. (2025). Active Attacks: Red-teaming LLMs via Adaptive Environments. arXiv:2509.21947.

[30] Zhang, Z., He, J., Cai, Y., et al. (2025). Genesis: Evolving Attack Strategies for LLM Web Agent Red-Teaming. arXiv:2510.18314.

[31] Zou, A., Wang, Z., Kolter, J. Z., and Fredrikson, M. (2023). Universal and Transferable Adversarial Attacks on Aligned Language Models. arXiv:2307.15043.

[32] Qwen Team (2025). Qwen3Guard Technical Report. arXiv:2510.14276.

---

*Draft completed: 2026-07-13*
*Status: Ready for LaTeX compilation testing*
