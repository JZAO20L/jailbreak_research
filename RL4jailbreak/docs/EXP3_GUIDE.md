# 📘 EXP3_GUIDE: Adaptive Hybrid-Reward GRPO (AHR-GRPO)

> **实验定位**：验证方差驱动的自适应权重机制能否在固定计算预算下，相比固定权重基线实现更快收敛、更高最终ASR与更稳定的策略更新。
> **核心创新**：用指数移动方差(EMV)动态计算 $\lambda_t$，替代人工网格搜索；双平滑+冷启动+裁剪保障工程鲁棒性。

公式：
以下是你们当前设计的**动态加权核心公式**（已按顶会论文标准格式化，可直接用于 `Method` 章节或技术文档）：

---
### 📘 核心公式（完整推导链）

$$
\begin{aligned}
&\textbf{1. 指数移动方差估计 (EMA/EMV)} \\
&\quad \mu_t^{(k)} = \beta \mu_{t-1}^{(k)} + (1-\beta) \bar{r}_t^{(k)} \\
&\quad \sigma^2_{k,t} = \beta \sigma^2_{k,t-1} + (1-\beta) \left( \bar{r}_t^{(k)} - \mu_t^{(k)} \right)^2, \quad k \in \{\text{ASR}, \text{proc}\} \\[10pt]
&\textbf{2. 方差比到权重的映射 (Log-Ratio Sigmoid)} \\
&\quad \lambda_{\text{raw}}^{(t)} = \sigma\left( \alpha \cdot \log \frac{\sigma^2_{\text{ASR},t} + \epsilon}{\sigma^2_{\text{proc},t} + \epsilon} + \delta \right) \\[10pt]
&\textbf{3. 惯性平滑与边界裁剪} \\
&\quad \lambda_t = \begin{cases}
0.5, & t \leq W \quad \text{(冷启动)} \\
\text{clip}\Big( \gamma_\lambda \lambda_{t-1} + (1-\gamma_\lambda)\lambda_{\text{raw}}^{(t)},\; \lambda_{\min},\; \lambda_{\max} \Big), & t > W
\end{cases} \\[10pt]
&\textbf{4. 最终混合奖励} \\
&\quad R_{\text{total}}^{(i)} = \lambda_t \cdot R_{\text{ASR}}^{(i)} + (1-\lambda_t) \cdot R_{\text{proc}}^{(i)}, \quad i \in \{1,\dots,G\}
\end{aligned}
$$

---

### 🔑 符号与超参说明

| 符号 | 含义 | 推荐值 / 范围 | 备注 |
|:---|:---|:---|:---|
| $\bar{r}_t^{(k)}$ | 第 $t$ 步 GRPO 组内奖励均值 | 组大小 $G=8$ | **必须用组均值**，防单样本噪声 |
| $\beta$ | EMA 衰减系数 | $\beta = 1 - 1/W$ | $W$ 为窗口大小 (1/3/5/10) |
| $\alpha$ | 方差比敏感度 | `2.0` | 控制曲线陡峭度 |
| $\delta$ | Sigmoid 偏移量 | `-1.0` | 负值使早期偏向 Judge |
| $\gamma_\lambda$ | $\lambda$ 惯性平滑系数 | `0.95` | 防单步跳变导致梯度震荡 |
| $\epsilon$ | 数值稳定项 | `1e-6` | 避免除零或对数异常 |
| $W$ | 窗口大小 / 冷启动步数 | `5` | 前 $W$ 步固定 $\lambda=0.5$ |
| $\lambda_{\min}, \lambda_{\max}$ | 权重裁剪边界 | `0.2, 0.8` | 防信号坍缩，保双路径参与 |

---

### 💡 物理直觉（一句话总结）
> **$\lambda_t$ 是两组奖励相对可区分性（方差比）的对数 Sigmoid 映射**：  
> 当 ASR 方差小（信号稀疏）→ $\lambda \downarrow$ → 依赖 Judge 稠密引导探索；  
> 当 ASR 方差大（信号分化）→ $\lambda \uparrow$ → 回归结果导向精细优化。  
> **双平滑（EMA + 惯性）与裁剪保障训练稳定性，冷启动避免初始方差估计偏差。**

---

### 📝 论文 Method 段落模板（直接粘贴）
```latex
\paragraph{Dynamic Reward Weighting.}
To adaptively balance sparse outcome rewards and dense process signals, we compute 
the mixing coefficient $\lambda_t$ via variance-aware mapping. We maintain exponential 
moving variance (EMV) estimates for both $R_{\text{ASR}}$ and $R_{\text{proc}}$ using 
group-wise means $\bar{r}_t^{(k)}$ and decay $\beta = 1-1/W$. The raw weight is computed as 
$\lambda_{\text{raw}}^{(t)} = \sigma(\alpha \log(\sigma^2_{\text{ASR},t}/\sigma^2_{\text{proc},t}) + \delta)$, 
then smoothed via $\lambda_t = \text{clip}(\gamma_\lambda \lambda_{t-1} + (1-\gamma_\lambda)\lambda_{\text{raw}}^{(t)}, \lambda_{\min}, \lambda_{\max})$. 
This design ensures $\lambda_t$ dynamically trusts the reward signal with higher 
discriminative power while maintaining training stability through dual smoothing and boundary clipping.
```
---

## 🎯 1. 实验目标与核心假设

| 假设 | 物理含义 | 验证指标 |
|:---|:---|:---|
| **H1** | ASR稀疏(方差小)时降低$\lambda$可加速早期探索 | 前300步ASR增长率 vs Fixed-$\lambda$ |
| **H2** | ASR分化(方差大)时提高$\lambda$可避免Judge偏差主导 | 中后期优势函数方差 `advantage_std` |
| **H3** | 自适应$\lambda$比任何固定$\lambda^*\in[0.2,0.8]$更优 | 达ASR≥0.35步数↓ / 最终ASR↑ / $\lambda$演化平滑度 |

**成功标准 (Success Metrics)**
- ✅ `convergence_step` ≤ 固定$\lambda$最佳基线的 **80%**
- ✅ `final_ASR` ≥ 固定$\lambda$最佳基线 **+0.03**
- ✅ `lambda_std` (后200步) ≤ **0.15**
- ✅ 训练全程无 `NaN/Inf`，优势均值稳定在 `0` 附近

---

## ⚙️ 2. 实验配置（固定项 + 4×4 矩阵）

### 🔒 固定配置（所有16组共享）
```yaml
experiment:
  total_steps: 1000
  attack_prompt: "hypothetical_scenario"   # Exp1 Top1
  judge_prompt: "multi_dimension_uniform"  # 4维×0.25
  seed: 42

model:
  policy: "Qwen3-4B"
  target: "Qwen3-4B"
  judge: "Qwen3-Max"
  context: {policy: 4096, others: 8192}

training:
  method: "GRPO"
  group_size: 8
  batch_size: 4
  lr: 1e-5
  adaptive:
    alpha: 2.0
    delta: -1.0
    lambda_smooth: 0.95
    # 变量见下方矩阵
```

### 🔀 实验变量矩阵（4×4=16组）
| 实验组ID | `window_size` | `β=1-1/W` | `clip_range` | 预期特性 |
|:---:|:---:|:---:|:---:|:---|
| **W1-C1** | 1 | 0.00 | [0.1, 0.9] | 无记忆，响应最快，易震荡 |
| **W1-C2** | 1 | 0.00 | [0.2, 0.8] | 无记忆+中边界 |
| **W1-C3** | 1 | 0.00 | [0.3, 0.7] | 无记忆+保守边界 |
| **W1-C4** | 1 | 0.00 | [0.4, 0.6] | 无记忆+极保守边界 |
| **W3-C1** | 3 | 0.67 | [0.1, 0.9] | 短记忆+宽边界 |
| **W3-C2** | 3 | 0.67 | [0.2, 0.8] | ✅ **推荐默认** |
| **W3-C3** | 3 | 0.67 | [0.3, 0.7] | 短记忆+保守边界 |
| **W3-C4** | 3 | 0.67 | [0.4, 0.6] | 短记忆+极保守边界 |
| **W5-C1** | 5 | 0.80 | [0.1, 0.9] | 中记忆+宽边界 |
| **W5-C2** | 5 | 0.80 | [0.2, 0.8] | ✅ **推荐默认** |
| **W5-C3** | 5 | 0.80 | [0.3, 0.7] | 中记忆+保守边界 |
| **W5-C4** | 5 | 0.80 | [0.4, 0.6] | 中记忆+极保守边界 |
| **W10-C1** | 10 | 0.90 | [0.1, 0.9] | 长记忆+宽边界 |
| **W10-C2** | 10 | 0.90 | [0.2, 0.8] | ✅ **推荐默认** |
| **W10-C3** | 10 | 0.90 | [0.3, 0.7] | 长记忆+保守边界 |
| **W10-C4** | 10 | 0.90 | [0.4, 0.6] | 长记忆+极保守边界 |

---

## 💻 3. 核心实现与
### 📦 3.1 自适应权重计算器 (`utils/adaptive_balancer.py`)
```python
import numpy as np
import torch

class EMAVarianceBalancer:
    def __init__(self, window_size=5, clip_range=(0.2, 0.8),
                 alpha=2.0, delta=-1.0, lambda_smooth=0.95):
        self.beta = 1.0 - 1.0 / max(window_size, 1)
        self.clip_min, self.clip_max = clip_range
        self.alpha, self.delta = alpha, delta
        self.lambda_smooth = lambda_smooth
        self.warmup = window_size
        self.step = 0
        self.mu_asr, self.var_asr = 0.0, 1.0
        self.mu_proc, self.var_proc = 0.0, 1.0
        self.lambda_t = 0.5

    def update(self, r_asr: torch.Tensor, r_proc: torch.Tensor) -> float:
        self.step += 1
        r_asr_m, r_proc_m = r_asr.mean().item(), r_proc.mean().item()
        
        if self.step <= self.warmup:
            return 0.5
            
        self.mu_asr = self.beta * self.mu_asr + (1 - self.beta) * r_asr_m
        self.mu_proc = self.beta * self.mu_proc + (1 - self.beta) * r_proc_m
        self.var_asr = self.beta * self.var_asr + (1 - self.beta) * (r_asr_m - self.mu_asr)**2
        self.var_proc = self.beta * self.var_proc + (1 - self.beta) * (r_proc_m - self.mu_proc)**2
        
        ratio = (self.var_asr + 1e-6) / (self.var_proc + 1e-6)
        lambda_raw = float(torch.sigmoid(self.alpha * np.log(ratio) + self.delta))
        
        self.lambda_t = self.lambda_smooth * self.lambda_t + (1 - self.lambda_smooth) * lambda_raw
        self.lambda_t = np.clip(self.lambda_t, self.clip_min, self.clip_max)
        return self.lambda_t
```