# Skill形式化定义改进建议

**核心理念**: Skill作为"Shortcut"引导收敛

---

## 🎯 当前问题

### 现有定义（过于技术化）
```latex
s = (content, source, σ, patterns, harm_type)
q(s) = success_rate(s) × √(usage_count(s))
```

**不足之处**：
- ❌ 只是一个数据结构定义
- ❌ 缺少概念层面的动机
- ❌ 没有体现"引导收敛"的作用
- ❌ 没有说明为什么skill能work

---

## 💡 核心洞察

### Skill的本质：Shortcut in Search Space

**用户的关键观点**：
> 在自动化攻击过程中需要多轮循环重写——而合适的skills应当充当这些攻击成功路径中的"shortcut"，引导收束于攻击成功

**深入理解**：

1. **搜索空间视角**
   ```
   攻击过程 = 在prompt空间中的迭代搜索
   
   无skill引导:
   - 搜索空间巨大
   - 需要~20-50轮迭代
   - 大量无效探索
   
   有skill引导:
   - Skill提供"shortcut"
   - 压缩搜索空间
   - 加速收敛
   ```

2. **Skill作为Shortcut**
   ```
   Attack Trajectory: p₀ → p₁ → p₂ → ... → p* (success)
   
   无skill: 需要完整的trajectory，每步随机探索
   有skill: skill ~ (p₀, strategy) → p* (shortcut)
   ```

3. **收敛引导**
   ```
   Skill作用机制：
   1. 提供初始方向（起始点）
   2. 约束搜索空间（边界）
   3. 加速收敛过程（梯度）
   ```

---

## 📝 改进方案

### 方案A：在Method Overview中增加概念阐述

**建议新增段落**（在Section 3.1 Overview之后）：

```latex
\paragraph{Skill as Shortcut.}
The core insight behind SESS is that successful attack trajectories 
contain reusable patterns that can act as \emph{shortcuts} in the 
attack search space. In iterative jailbreak methods like PAIR 
(~20 iterations) and AutoDAN (~50 iterations), each attack 
typically explores a large prompt space through costly trial-and-error. 
A skill serves as a compressed representation of a successful 
trajectory, providing:
\begin{itemize}
    \item \textbf{Search space reduction}: Skills constrain the 
    exploration to promising regions, avoiding ineffective regions.
    \item \textbf{Convergence acceleration}: Skills provide a 
    strong starting point, reducing the number of iterations needed 
    to reach success.
    \item \textbf{Cross-instance transfer}: Skills capture abstract 
    patterns that generalize across different prompts and models.
\end{itemize}

This shortcut perspective explains why skill-guided attacks 
achieve higher ASR with fewer iterations (Section~\ref{sec:experiments}).
```

**位置建议**: 在Figure 1之后，Section 3.2之前

---

### 方案B：改进Skill Representation的形式化定义

**建议修改**（Section 3.2）：

```latex
\subsection{Skill Representation and Library}
\label{sec:skill_representation}

\paragraph{Conceptual Definition.}
A \emph{skill} is a reusable strategy template that acts as a 
\textbf{shortcut} in the attack trajectory space. Formally, given 
an attack process as a sequence of prompt refinements 
$\pi = (p_0, p_1, \ldots, p_T)$ where $p_T$ succeeds, a skill 
$s$ captures the essential transformation pattern:
\begin{equation}
    s \approx \text{shortcut}(p_0 \rightarrow p_T) = f_{\theta}(p_0)
\end{equation}
where $f_{\theta}$ is a parameterized transformation that maps 
an initial prompt $p_0$ to a region in the prompt space with high 
success probability, effectively bypassing intermediate exploration steps.

\paragraph{Concrete Representation.}
In practice, we represent a skill $s$ as a natural language prefix 
(template string) with associated metadata:
\begin{equation}
    s = (\text{content}, \text{source}, \boldsymbol{\sigma}, 
         \text{patterns}, \text{harm\_type})
\end{equation}
where:
\begin{itemize}
    \item $\text{content}$: The template string that prepends to prompts
    \item $\text{source} \in \{\text{initial, extracted, evolved, merged}\}$: Provenance
    \item $\boldsymbol{\sigma} = (\text{usage}, \text{success}, \text{rate})$: Statistics
    \item $\text{patterns}$: Keyword patterns for retrieval
    \item $\text{harm\_type}$: Applicable harm domain
\end{itemize}

\paragraph{Quality Score.}
The quality of a skill measures its effectiveness as a shortcut:
\begin{equation}
    q(s) = \underbrace{\text{success\_rate}(s)}_{\text{Reliability}} 
           \times \underbrace{\sqrt{\text{usage\_count}(s)}}_{\text{Evidence}}
\end{equation}
This balances reliability (how often it leads to success) with 
evidence (how many times it has been validated).
```

---

### 方案C：在Introduction中阐述动机

**建议新增段落**（Section 1.2 现有方法的局限）：

```latex
\paragraph{The Search Space Problem.}
Current jailbreak methods face a fundamental challenge: each attack 
must explore a large prompt space through iterative refinement. 
PAIR typically requires ~20 iterations, AutoDAN ~50 iterations, 
with each iteration querying the target model. This high cost arises 
from treating each attack as an independent optimization problem, 
discarding the knowledge gained from previous successful attacks.

\paragraph{The Shortcut Opportunity.}
We observe that successful attack trajectories share common patterns:
certain prompt structures, phrasings, or framing strategies 
consistently lead to jailbreak success across different instances. 
If we could extract and reuse these patterns, we could transform 
jailbreak from instance-level optimization into a knowledge-guided 
search, where skills act as shortcuts that compress the exploration 
space and accelerate convergence.
```

---

## 🔬 理论支撑

### 1. Search Space Compression

```
无skill的搜索空间:
S_full = {all possible prompts} → |S| = ∞

有skill的搜索空间:
S_skill = {prompts matching skill patterns} → |S_skill| << |S_full|

压缩比 = |S_skill| / |S_full|
```

**实验证据**：
- PAIR需要~20次查询（无skill）
- PAIR+SESS需要~4.45次查询（有skill）
- 压缩效果：查询次数减少75%

### 2. Convergence Acceleration

```
收敛速度 = f(skill quality, prompt complexity)

高质量skill:
- 提供强初始方向
- 减少探索步骤
- 加速收敛

低质量skill:
- 方向不明确
- 需要更多迭代
- 可能导致失败
```

### 3. Transfer as Shortcut Generalization

```
Skill的泛化能力:
- 同族迁移: skill shortcut依然有效
- 跨族迁移: shortcut需要适配或调整
- DAN模板: 强先验shortcut，泛化能力强
```

---

## 📊 实验证据支持

### 关键数据点

**收敛速度对比**：
```
PAIR (无skill):
- 平均迭代次数: ~20
- ASR: 55.5%

PAIR+SESS (有skill shortcut):
- 平均迭代次数: 4.45
- ASR: 79.0%
- 迭代减少: 77.7%
```

**Shortcut质量**：
```
DAN+SESS (强先验shortcut):
- 平均迭代次数: 1.10
- ASR: 100.0%
- 迭代减少: 94.5%
```

**跨模型迁移**：
```
Skill shortcut在不同模型上的效果:
- 同族迁移: 85-100% ASR (shortcut有效)
- 跨族迁移: 12.9-30.0% ASR (需要适配)
```

---

## 🎯 建议实施

### 优先级排序

1. **高优先级**：方案A（Method Overview增加概念阐述）
   - 位置：Section 3.1之后，Figure 1之后
   - 理由：概念阐述在前，形式化定义在后
   - 效果：读者先理解"为什么"，再看"是什么"

2. **中优先级**：方案B（改进Skill Representation）
   - 位置：Section 3.2
   - 理由：形式化定义需要概念支撑
   - 效果：定义更完整，更易理解

3. **低优先级**：方案C（Introduction增加动机）
   - 位置：Section 1.2
   - 理由：可选，但能增强动机阐述
   - 效果：问题动机更清晰

### 实施步骤

```
步骤1: 在outline_v5.md中更新Section 3.1
步骤2: 更新method.tex的Section 3.2
步骤3: (可选) 更新introduction.tex的Section 1.2
步骤4: 添加到revision log
步骤5: 验证LaTeX编译
```

---

## ✅ 改进效果

### 修改前
- ❌ Skill定义过于技术化
- ❌ 缺少概念层面动机
- ❌ 审稿人可能质疑"为什么skill能work"

### 修改后
- ✅ Skill概念清晰（shortcut）
- ✅ 有理论支撑（搜索空间压缩）
- ✅ 有实验证据（收敛加速）
- ✅ 回答了"为什么"和"如何work"

---

## 📚 参考表述

### 类似概念的文献

**Search Space Reduction**:
- Neural Architecture Search中的"network morphism"
- AutoML中的"hyperparameter transfer"

**Shortcut in RL**:
- Option learning (Sutton et al., 1999)
- Hierarchical RL中的"macro-actions"
- Skill discovery in RL

**Knowledge Transfer**:
- Meta-learning中的"learning to learn"
- Curriculum learning中的"knowledge scaffolding"

---

## 📝 Outline更新建议

```markdown
## 3. Method

### 3.1 Overview
- 三阶段：Cold Start → Evolution → Deployment
- **NEW**: Skill as Shortcut概念阐述
  - 搜索空间视角
  - 收敛引导作用
  - 知识迁移机制

📊 **本节图表**: 
- **Figure 1**: `Figures/SESS.drawio.pdf` — 系统框架图

### 3.2 Skill Representation
- **NEW**: Conceptual Definition (shortcut视角)
- 形式化定义: $s = (c, \text{src}, \boldsymbol{\sigma}, K, h)$
- **IMPROVED**: 质量评分with reliability + evidence
- Library operations

📊 **本节图表**: 无
```

---

*创建时间: 2026-07-09*
*核心洞察: Skill = Shortcut in Attack Search Space*