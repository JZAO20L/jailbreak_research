# 当前检索实现总结

**文件位置**: `src/skill_library.py`, `src/attacker.py`

---

## 📊 检索流程概览

### 核心方法：`SkillLibrary.retrieve()`

```python
def retrieve(self, prompt: str, top_k: int = 1) -> List[Skill]:
    """
    检索最匹配的Skills
    
    流程：
    1. 提取prompt特征
    2. 评分所有Skills
    3. 排序返回top_k
    """
    # 1. 提取特征
    prompt_features = self._extract_prompt_features(prompt)
    
    # 2. 评分所有skills
    scored_skills = []
    for skill in self.skills.values():
        score = self._score_skill_match(skill, prompt_features)
        scored_skills.append((skill, score))
    
    # 3. 排序返回
    scored_skills.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored_skills[:top_k]]
```

---

## 🔍 特征提取

### `_extract_prompt_features()`

```python
def _extract_prompt_features(self, prompt: str) -> Dict:
    """提取prompt特征"""
    prompt_lower = prompt.lower()
    return {
        "length": len(prompt),
        "keywords": self._extract_keywords(prompt),
        "harm_type": self._classify_harm_type(prompt_lower),
        "has_question": "?" in prompt,
    }
```

**提取的特征**：
- ✅ Length: prompt长度
- ✅ Keywords: 关键词列表
- ✅ Harm Type: 伤害类型分类
- ✅ Has Question: 是否包含问号

---

## 🎯 关键词提取

### `_extract_keywords()`

```python
def _extract_keywords(self, text: str) -> List[str]:
    """提取关键词"""
    keywords = []
    keyword_patterns = [
        "how to", "guide", "instructions", "create", "make",
        "build", "explain", "write", "help", "provide",
        "expert", "professional", "research", "study",
    ]
    text_lower = text.lower()
    for kw in keyword_patterns:
        if kw in text_lower:
            keywords.append(kw)
    return keywords
```

**特点**：
- ✅ 基于预定义模式
- ✅ 简单字符串匹配
- ❌ 无语义理解
- ❌ 固定词表（14个模式）

---

## 🏷️ 伤害类型分类

### `_classify_harm_type()`

```python
def _classify_harm_type(self, prompt_lower: str) -> str:
    """分类伤害类型"""
    type_keywords = {
        "violence": ["weapon", "gun", "bomb", "explosive", "attack"],
        "dangerous_substance": ["drug", "poison", "chemical", "toxic"],
        "cybersecurity": ["hack", "exploit", "malware", "virus"],
        "financial": ["steal", "fraud", "scam", "money"],
        "social": ["manipulate", "deceive", "trick"],
    }
    
    for harm_type, kws in type_keywords.items():
        if any(kw in prompt_lower for kw in kws):
            return harm_type
    
    return "general"
```

**分类类别**：
- violence（暴力）
- dangerous_substance（危险物质）
- cybersecurity（网络安全）
- financial（金融）
- social（社会工程）
- general（通用）

**特点**：
- ✅ 基于关键词匹配
- ❌ 无层次分类
- ❌ 固定词表（每类4-5个词）

---

## 📈 评分机制

### `_score_skill_match()`

```python
def _score_skill_match(self, skill: Skill, features: Dict) -> float:
    """计算Skill匹配得分"""
    # 基础得分 = 质量
    score = skill.quality_score
    
    # 匹配关键词加分
    prompt_keywords = features["keywords"]
    matched = sum(1 for p in skill.applicable_patterns if p in prompt_keywords)
    if matched > 0:
        score *= (1 + matched * 0.3)  # 每个匹配关键词 +30%
    
    # 伤害类型匹配加分
    if features["harm_type"] in skill.applicable_patterns:
        score *= 1.5  # 类型匹配 +50%
    
    return score
```

**评分公式**：
```
final_score = quality_score × (1 + 0.3 × matched_keywords) × (1.5 if type_match else 1.0)

其中：
- quality_score = success_rate × √usage_count
- matched_keywords: 匹配的关键词数量
- type_match: 伤害类型是否匹配
```

**超参数**：
- 关键词权重：0.3（硬编码）
- 类型匹配权重：1.5（硬编码）

---

## 🔧 检索模式

### Attacker中的两种模式

#### 模式A: single_call（默认）

```python
def _attack_single_skill(self, original_prompt, skills):
    """
    只在开始时检索一次
    
    流程：
    1. 检索 top-1 skill
    2. 注入到prompt
    3. 迭代精化（保持skill前缀）
    """
    skill = skills[0] if skills else None
    current_prompt = self._inject_skill(skill, original_prompt)
    
    for i in range(max_iterations):
        response = self._get_target_response(current_prompt)
        success = self._evaluate_attack(current_prompt, response)
        
        if success:
            return success, skill
        
        # 精化时保持skill前缀
        current_prompt = self._refine_prompt(current_prompt, response, skill)
```

**特点**：
- ✅ 稳定性高（+12.7% ASR）
- ✅ 一致性引导
- ✅ 收敛快

#### 模式B: every_iteration

```python
def _attack_dynamic_skill(self, original_prompt, skills):
    """
    每次迭代都重新检索
    
    流程：
    1. 每轮重新检索
    2. 动态切换skill
    3. 失败阈值触发切换
    """
    for i in range(max_iterations):
        # 每轮重新检索
        current_skills = self.skill_library.retrieve(current_prompt, top_k=3)
        skill = current_skills[current_skill_idx]
        
        attack_prompt = self._inject_skill(skill, current_prompt)
        response = self._get_target_response(attack_prompt)
        success = self._evaluate_attack(attack_prompt, response)
        
        if success:
            return success, skill
        
        # 失败阈值触发切换
        if consecutive_failures >= threshold:
            current_skill_idx += 1
```

**特点**：
- ⚠️ 不稳定
- ⚠️ 可能冲突
- ❌ 效果较差（-8.3% ASR）

---

## 📊 实验结果对比

### 单次检索 vs 动态检索

```
模式               平均ASR    特点
──────────────────────────────────────
single_call        74.7%     稳定、一致引导 ✅
every_iteration    66.4%     动态切换、效果差 ❌
差异               +8.3%     single_call更好
──────────────────────────────────────
```

**结论**：单次检索（稳定引导）优于动态检索

---

## ⚠️ 当前实现的问题

### 1. 特征提取局限性

```python
❌ 问题：
- 纯词汇匹配（无语义）
- 固定词表（泛化差）
- 无上下文理解
- 无句子结构分析

✅ 潜在改进：
- 使用sentence embedding
- 添加语义相似度
- 考虑prompt结构
```

### 2. 评分机制简单

```python
❌ 问题：
- 超参数硬编码（0.3, 1.5）
- 无学习权重
- 未考虑skill多样性
- 未考虑历史匹配效果

✅ 潜在改进：
- 学习超参数
- A/B测试权重
- 添加多样性约束
- 考虑历史成功率
```

### 3. 无向量检索

```python
❌ 问题：
- 无embedding相似度
- 无向量索引
- 无法处理语义相似但词汇不同的情况

✅ 潜在改进：
- 使用sentence-transformers
- FAISS向量索引
- 混合检索（lexical + semantic）
```

---

## 💡 与论文中表述的对比

### 论文中的描述

```latex
\text{score}(s, p) = q(s) \cdot (1 + \alpha \cdot |K_s \cap K_p|) \cdot \delta_{\text{type}}

其中：
- α = 0.3（硬编码）
- δ_type = 1.5（硬编码）
```

### 实际实现

```python
# 完全对应
score = skill.quality_score * (1 + 0.3 * matched) * (1.5 if type_match else 1.0)
```

**一致性**: ✅ 论文表述与代码实现完全一致

---

## 📈 性能统计

### 检索开销

```
单次检索时间复杂度：
- 特征提取: O(|prompt|)
- 评分: O(|skills|)
- 排序: O(|skills| log |skills|)

总复杂度: O(n log n)，n = number of skills
```

### 实际性能

```
Skills数量: 28-100
单次检索: < 1ms
内存占用: ~1MB
```

---

## 🔬 关键发现

### 1. Lexical Matching有效

```
实验证据：
- PAIR+SESS: 79.0% ASR（+23.5pp）
- 简单词汇匹配也能显著提升效果
- 不需要复杂的向量检索
```

### 2. 单次检索优于动态检索

```
实验证据：
- single_call: 74.7% ASR
- every_iteration: 66.4% ASR
- 差异: +8.3%
```

### 3. 质量评分重要

```
质量评分公式：
q(s) = success_rate × √usage_count

效果：
- 平衡可靠性和证据
- 新skill也有机会（sqrt特性）
- 经验证的skill权重高
```

---

## 🎯 总结

### 当前检索机制特点

**优势**：
- ✅ 简单高效
- ✅ 无需额外模型（embedding）
- ✅ 可解释性强
- ✅ 计算开销小
- ✅ 论文表述一致

**局限**：
- ⚠️ 语义理解弱
- ⚠️ 超参数固定
- ⚠️ 泛化能力有限
- ⚠️ 无向量检索

**未来改进方向**：
1. 添加语义嵌入（可选）
2. 学习超参数权重
3. 混合检索策略
4. 动态权重调整

---

*总结完成时间: 2026-07-09*
*当前实现：Lexical Anchoring（词汇锚定）*