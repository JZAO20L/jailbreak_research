# API Target 补充实验规划（ICLR 2027 · 小规模）

日期：2026-09-17 · 状态：**已决策执行（小规模版）** · 上位文档：`paper/SESS_iclr2027/SUPPLEMENT_EXPERIMENTS_PLAN.md`

## 0. 决策背景

2026-09-17 定稿讨论结论：论文 novelty 的四大可预判质疑中，"同族 100% = benchmark 饱和，
唯一不饱和的 GPT-OSS-20B 上数字偏低"这条**只能靠真实部署模型上的相对增益解掉**。
决定：**小规模补 1-2 个 API 调用模型作 target**，落在 `exp/transfer/results/<api_model>/`。

论文侧已同步完成（同日）：
- Intro Contributions 改写为 F1（归因）/ F2（结构先验跨架构迁移）/ F3（框架）三条 finding；
- "Why not embedding" 写死为设计立场（可解释性 / 零额外依赖 / 检索非瓶颈）；
- Limitations "先验决定天花板" 定性化。

## 1. 实验矩阵（小规模版）

沿用上位文档 §1 的塌缩决策，**不做全矩阵**：

| 项 | 取值 |
|---|---|
| target | 1-2 个 API 模型（首选 `qwen-max`；预算允许加 `kimi-k3` 或 `glm-5`，均走 DashScope） |
| attacker | 本地 Qwen3-4B（**不**做 API attacker——Exp A 后置，见上位文档 §0.5） |
| 方法 | PAIR / AutoDAN / PAIR+SESS / DAN+SESS（4 个头条方法，与 `tab:cross_model` 各列直接可比） |
| 数据 | default test 抽 200（可选 +advbench 100 做 sanity） |
| 库 | 一次构建：`sess_lib.json` / `dan_lib.json` / `empty.json`（本地免费，test-only 换端点） |

工作量 ≈ 1-2 target × 4 方法 × 200 = **800-1600 次攻击**；成本量级 ¥100-300。

## 2. 冒烟决策门（先于全量，1 小时，<¥10）

单 API target × 3 方法（DAN+SESS / AutoDAN / PAIR）× 10 条，一次验三件事：
① 端点/账号通；② 平台是否请求层拦截；③ 迁移信号是否值得赌（DAN+SESS ≥ 基线 +10pp？）。

| 冒烟结果 | 决策 |
|---|---|
| 信号好 | 全量跑 §1 矩阵 |
| 信号平 | 停 API 实验；时间转投论文文字（诚实报告相对增益即可） |
| 请求层拦截 | 放弃该平台，换下一家或整体放弃 |

## 3. 落盘约定（与现有 transfer 结构对齐）

```
exp/transfer/
  scripts/run_api_target.sh          # 新增：循环 target×method，调 sess.py test-only
  results/<api_model>/               # 如 results/qwen-max/，与本地四模型并列
  API_TARGET_EXTENSION.md            # 本文档；跑完后在此追加结果表
```

- 判定：统一用本地 Qwen3Guard-4B（主口径一致）；每 target 抽 100 条 API judge 交叉校验 + 人工抽检 20-30 条。
- 拒绝率：attacker 侧不动；若 target 请求层有过滤行为，记录原始拒绝/错误率进结果表备注。
- **结果无论好坏都进论文附录**（相对增益口径），预期锚定见上位文档 §0.5 风险 2。

## 4. 待确认清单（开跑前）

- [ ] DashScope API key 与三家 model id 确切名称
- [ ] `llm_client.py` / `sess.py` 的 `--target_endpoint` 改造（上位文档 §3，~2 天代码）
- [ ] 本地 `build_libs.sh` 产出三个库文件
- [ ] compact AutoDAN（`attack_autodan()`，~50 行）或决策去 AutoDAN 行

## 5. 关联 TODO（同日决策，非 API 实验）

- **多 seed 方差**：单 seed(42) 质疑留 TODO——单次攻击需 ~5-10 次 target 调用，
  全量多 seed 成本不可 cover；仅在 headline 数字（如 GPT-OSS-20B 的 DAN+SESS）上小规模补 2-3 seeds。
