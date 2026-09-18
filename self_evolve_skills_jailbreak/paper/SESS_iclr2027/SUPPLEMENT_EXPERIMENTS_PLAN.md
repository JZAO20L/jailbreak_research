# SESS ICLR 补充实验规划（API 模型）

日期：2026-09-11 · 距 ICLR 截止约 2 周 · 状态：**设计已定，待实现**
> **09-17 更新**：已决策执行**小规模版**（仅 Exp B target 侧、1-2 个 API 模型、default 200 条），
> 冒烟决策门与落盘约定见 `exp/transfer/API_TARGET_EXTENSION.md`；Exp A（API attacker）维持后置。
> 同日论文侧已完成 Contributions 重写（F1/F2/F3）与 "why not embedding" 定性。

## 0. 背景与动机

论文现状：attacker 固定 Qwen3-4B；target 只有开源权重（Qwen3-0.6/4/14B + GPT-OSS-20B）。
cross-model 最弱一环是 GPT-OSS-20B（DAN+SESS 仅 32.5%）——评审最可能质疑"对真实部署的模型有没有用"。

两个补充实验（均通过 API 调用，国产 SOTA）：

- **Exp A（attacker 侧）**：`qwen-max` 作 attacker，攻击本地 Qwen3-4B。
  论点：**参数量不是攻击模型的瓶颈，安全护栏才是**——带护栏的大 API 模型不比无护栏的 4B 强。
- **Exp B（target 侧）**：`qwen-max` / `kimi-k3` / `glm-5` 作 target（真实部署的国产 SOTA）。
  论点：以适度成本换取 **cross-model 迁移到真实部署模型**的可信度。

## 0.5 必要性与可行性评估（先于实现）

### 必要性：不是必须，是"可选加分"

- AAAI 版被拒是 **desk reject（超页）**，从未进入评审 → **没有任何评审意见要求补实验**。
  当前实验量已很大（140 config / 217 ablation / 120 transfer / 4 模型 5 数据集）。
- 补 API 实验针对的"跨模型迁移弱"是**我们自己推测的软肋**，不是评审确认的软肋。
- **唯一已知失败模式是 9 页压缩**（AAAI 已因超页拒过一次，重演 = 实验做得再多也白搭）。压缩是**必须项**，补实验是**可选项**，顺序不能反。
- 值得做的理由：GPT-OSS-20B 32.5% 是 cross-model 最弱一环；真实部署模型是评审自然能想到的质疑。
  尤其"DAN 结构模板跨架构迁移更好"是论文最有意思的 claim，用真实部署模型验证是它的自然延伸。

### 可行性：真正的风险不在代码，在三个未知

代码改造很小（客户端加 3 个参数 + CLI 加端点 flag）。真正的风险：

1. **平台政策/审核**：自动化批量 jailbreak 打商用 API，可能触发平台请求层拦截或封号。
   尤其 Exp A（qwen-max 当 attacker）：模型拒绝是预期结论，但若平台在**请求层**直接返回违规错误，
   **实验根本没有模型输出可测**。发送的 harm prompts 本身也属于被测有害内容。
2. **结果可能帮倒忙（方向不确定）**：技能在弱 target（Qwen3-4B）上学的，迁到强商用模型
   绝对 ASR 很可能很低（DAN+SESS 可能仅 10-15%）。表做出来"SESS 12% vs 基线 2%"——诚实但难看，
   可能强化"技能迁移不到强模型"的印象。**结果利好与利空都有可能，不能预设必赢。**
3. **时间竞争**：2 周里实验占 ~10 天 + 压缩 ~3 天 = 几乎没有缓冲。账号没就绪/被拦截/结果平，
   任一卡壳都可能 2 周打水漂且 9 页还没压。
4. **Exp A 的科学严谨性**：只做"qwen-max vs 4B"有混淆（大小+护栏+架构+平台全变了）。
   要支撑"参数不是瓶颈、护栏才是"，需补"大×无护栏"对照（Qwen3-14B 本地 attacker），
   V100 上 14B fp16 TP2 能跑但资源紧。

### 对策：10 条冒烟测试当"决策门"（1 小时，<¥10）

不在全量前假设可行，先用**一个 API 模型（qwen-max 当 target）**跑 DAN+SESS / AutoDAN / PAIR
各 10 条，一次验三个未知：① 账号/端点通不通；② 平台会不会请求层拦截；③ **迁移信号值不值得赌**
（DAN+SESS 是否明显高于基线，如 ≥ 基线 +10pp）。

| 冒烟结果 | 决策 |
|---|---|
| 信号好（DAN+SESS 明显 > 基线） | 全量跑 Exp B（200×4×3）；Exp A 后置 |
| 信号平（SESS 无明显优势） | **停 API 实验**；时间投到 9 页压缩 + 把现有 GPT-OSS 弱迁移分析写透，可选加本地 Qwen3-14B 无护栏 attacker 对照（免费） |
| 平台请求层拦截 | 只测 target 侧、attacker 全本地；或放弃 API 实验 |

**任何情况下先做 9 页压缩**（唯一已知失败模式），压缩与冒烟并行推进。

## 1. 关键决策：不做完整"方法×数据集"矩阵 ✅

**结论：不做。** 理由：

- 现有 cross-model 表（`tab:cross_model`）是 **5 数据集平均 × 5 方法 × 4 target**。API target 若照搬 = 3 target × 5 方法 × 5 数据集，成本爆炸且信息冗余。
- **数据集维度塌缩**：cross-dataset 泛化已在本地 Qwen3-4B（免费）证明；API 实验回答的是"是否迁移到强商用模型"，**单一代表基准（default test 子集）即够**。可选加 advbench 100 条做 sanity（预算允许时）。
- **方法维度保留 4 个头条方法**（PAIR / AutoDAN / PAIR+SESS / DAN+SESS）→ 与 `tab:cross_model` 各列直接可比，这是核心主张，值得保留。可选加 no_rewrite（≈免费）。

**推荐矩阵：**

| 实验 | target | attacker | 方法 | 数据 |
|---|---|---|---|---|
| Exp A | Qwen3-4B（本地） | `qwen-max`(API) vs Qwen3-4B(本地) | PAIR / PAIR+SESS / DAN+SESS | default test 抽 200 |
| Exp B | `qwen-max`,`kimi-k3`,`glm-5`(API) | Qwen3-4B（本地） | PAIR / AutoDAN / PAIR+SESS / DAN+SESS | default test 抽 200（+可选 advbench 100） |

工作量 = Exp B 3 target × 4 方法 × 200 = **2400 次攻击** + Exp A ~600 次。

## 2. 调用方式：统一走 DashScope ✅

用户确认：`qwen-max` / `kimi-k3` / `glm-5` **均可通过 DashScope 调用**（具体端点与 key 文档待用户补充）。
代码设计保持端点无关：每角色 `base_url, api_key, model` 三元组，哪个平台都能接。
待确认：三个 model id 在 DashScope 上的确切名称与 API key 是否就绪。

## 3. 代码改造（最小，2 个文件）

1. **`llm_client.py`**：`LLMClient.__init__(port, base_url=None, api_key="EMPTY", model=None, ...)`。
   给 base_url 时走外部 API；显式给 model 时跳过自动探测（外部 API 不一定支持 `models.list`）。
2. **`sess.py`**：`init_clients` 读每角色端点；CLI 新增
   `--target_endpoint / --attacker_endpoint / --guard_endpoint`，格式 `base_url,api_key,model`（逗号分隔）。
3. **库一次构建，test-only 换端点**：新实验全部走现有 test-only 模式
   `--skip_cold_start --skip_evolution --load_library <lib> --test_data_path data/test_prompts.json --test_limit 200`。
   这样 **Exp A 的 attacker 变量被干净隔离**（同一库、同 target，只换 attacker）。

**库的构建（Step 0，本地免费，V100）**：
- `sess_lib.json` = 全流程（cold start 200 + evolution 800）建库 → **PAIR+SESS** 用
- `dan_lib.json` = `--skill_source dan_templates --skip_cold_start --skip_evolution` → **DAN+SESS** 用
- `empty.json` = `{}` → **PAIR** 用（`inject_skill(None,p)=p`，已是纯 PAIR）

## 4. 需要补的一个功能点：最小 AutoDAN

`sess.py` 目前只有 PAIR 式攻击循环（`attack_one`），**没有 AutoDAN 的遗传循环**。
论文里 AutoDAN 数字来自原仓库 `baselines/autodan.py`（OOP、绑本地端口，改造成本更高）。

**建议**：在 `sess.py` 补一个 compact `attack_autodan()`（~50 行：population=4、LLM mutation、
每代 keep-best），与 `attack_one` 共用同一 target 客户端与判定 → 四方法同代码路径、API 表自洽。
（若预算紧张，备选：API 表去掉 AutoDAN 行并加脚注说明——但会削弱"DAN+SESS > 最强基线"在真实模型上的主张，不推荐。）

## 5. 判定（judge）一致性

- **头条数字统一用本地 Qwen3Guard-4B**（与主实验口径一致）。
- 风险：4B guard 给 API 模型（更强、更长）输出打标可能偏噪。
  **对策**：① 每个 API target 抽 100 条用 API judge（如 qwen-max 作 judge）交叉校验，报告一致性；
  ② 人工抽检 20-30 条。不一致处进 appendix 讨论。

## 6. Exp A 的拒绝率处理

`qwen-max` 作 attacker 时**预期直接拒绝生成 jailbreak 候选**（这正是"护栏是瓶颈"的机制）。
需要：`attack_one` 里加一个轻量 `is_refusal()`（关键词检测 attacker 输出）+ 记录拒绝率；
拒绝视为本次迭代失败、循环照常推进（**不改循环语义**）。这样"API attacker 因为拒绝而 ASR 低"
才有数据支撑，而不是"测了个没法工作的配置"。

## 7. 成本估算（量级）

| 项 | 估算 |
|---|---|
| Exp A（attacker API） | ~1000-2000 次 qwen-max 调用 ≈ ¥50-150 |
| Exp B（target API） | ~2.4 万次 target 调用（AutoDAN ~20 次/攻击是大头）≈ ¥200-400 |
| **合计** | **≈ ¥300-600**，1-2 个工作日墙钟 + 2-3 天代码 |

降本抓手（按需）：test 抽 150、去 AutoDAN 行、advbench 100 条仅 1 个 target。

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| ToS/封号：自动化批量 jailbreak 测商用 API | 先确认三家政策（研究用途红队通常允许）；批量前先 10 条小样冒烟；必要时限速 |
| 技能在弱 target 学的，迁到强商用模型绝对 ASR 低 | 预期锚定**相对增益**（PAIR+SESS vs PAIR、DAN+SESS vs AutoDAN），诚实表述 |
| 论文超 9 页（AAAI 曾 desk reject） | 新实验**只进附录**（1 张表 + 1 张表）+ cross-model 小节 2-3 句；正文压缩照常做 |
| judge 偏噪（见 §5） | API judge 交叉校验 100 条 + 人工抽检 |
| model id / 账号未就绪 | §2 待确认清单先过一遍 |

## 9. 脚本骨架

```
code/
  scripts/
    build_libs.sh            # Step 0：一次构建 sess_lib / dan_lib / empty（本地，V100）
    run_api_eval.py          # 薄包装：循环 target×method，调用 sess.py test-only，汇总 ASR/成本/拒绝率
    exp_api_attacker.sh      # Exp A：attacker=qwen-max vs 本地4B × {PAIR, PAIR+SESS, DAN+SESS}
    exp_api_target.sh        # Exp B：3 API target × {PAIR, AutoDAN, PAIR+SESS, DAN+SESS}
```

每格一次 `python sess.py --skip_cold_start --skip_evolution --load_library <lib> \
  --test_data_path data/test_prompts.json --test_limit 200 \
  [--attacker_endpoint|--target_endpoint <base_url,key,model>] --exp_name <cell>`。

## 10. 时间线（2 周）

| 天 | 事项 |
|---|---|
| D1-2 | **先做 9 页压缩**（唯一已知失败模式，与实验无关必做）；确认 API 账号/model id（§2） |
| D3-4 | 改造 llm_client + sess.py；本地 build_libs；**10 条冒烟 + 决策门**（§0.5） |
| D5-8 | 冒烟信号好 → Exp B 全量（先 qwen-max 验证成本/拒绝率）；信号平 → 停实验，集中写论文 |
| D9-11 | Exp B 剩余 2 target；judge 交叉校验；Exp A 视时间后置 |
| D12-14 | 结果进论文（附录表 + 正文 2-3 句）；终审；若实验放弃则全部时间投论文 |
