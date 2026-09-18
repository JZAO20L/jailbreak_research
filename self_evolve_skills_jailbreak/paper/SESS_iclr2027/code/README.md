# SESS · Executable Code Package (Minimal, Process-Oriented)

This directory is the **minimal reproducible implementation** of the paper
*Beyond One-Shot Jailbreaking* -- the full repository
(`src/*` + `scripts/pipeline.py`, ~2,000+ lines of OOP) is flattened into
**two process-oriented Python files** that faithfully reproduce the paper's
three-phase pipeline and all core mechanisms.

## Files

| File | Description |
|---|---|
| `sess.py` (~790 lines) | Everything: skill representation / retrieval / injection, PAIR-style attack loop, reflection-based extraction, library maintenance, three-phase pipeline, CLI |
| `llm_client.py` | Minimal vLLM OpenAI-compatible client (auto-detects model name, strips Qwen3 thinking blocks) |
| `start_services.sh` | Starts the three services: guard(8002) / target(8001) / attacker(8003) |
| `requirements.txt` | numpy / tqdm / openai / vllm (server side only) |

Dependencies are minimal: SESS does **not** use torch/transformers directly;
all model inference goes through three external vLLM HTTP endpoints
(OpenAI SDK).

## Quick Start

```bash
# 1) Start the three services (override model paths via env; add DTYPE=float16 on V100)
bash start_services.sh

# 2) Full pipeline: cold start(200) -> evolution(800 x N) -> test(1000)
python sess.py --mode full --num_epochs 3 --output_dir results --exp_name sess_base
```

## Reproducing the Paper's Configurations

| Paper experiment | Command |
|---|---|
| Main config `single_call + trajectory + statistical` (Layer 1 best, 79.1%) | default args (above) |
| Layer 1 ablation: `every_iteration` dynamic retrieval | `--skill_call_mode every_iteration` |
| Layer 1 ablation: `final_prompt` extraction | `--skill_extraction_mode final_prompt` |
| Layer 1 ablation: four update strategies | `--update_strategy success_only\|failure_only\|both\|statistical\|pure` |
| Layer 3: strong DAN prior initialization | `--skill_source dan_templates --skip_cold_start` |
| Component ablation: no cold start | `--skip_cold_start` |
| Component ablation: no evolution | `--skip_evolution` |
| Load a library from a previous run (e.g. to reproduce the test stage) | `--load_library <path>` |
| Cross-dataset transfer | `--test_data_path <benchmark export> --test_limit 200` |

> ⚠️ Per submission policy, the **trained skill library is NOT shipped with the
> supplementary package** (`skills_library.json` has been removed). The library is
> built by the pipeline itself during the cold-start and evolution phases.

## Run Notes

- **Strict judging**: only `Unsafe` from the guard counts as success
  (`Controversial` does not), consistent with the paper.
- **Data isolation**: cold start = 200 / evolution = 800 / test = 1000,
  pairwise non-overlapping; transfer experiments use `../data/benchmark/`
  (AdvBench / HarmBench / JailbreakBench).
- **Parallelism**: `--workers` controls attack parallelism (default 8);
  reflection and library updates run sequentially to keep the library thread-safe.
- **Random seed**: default 42 (`--seed`).
- **Outputs**: `<output_dir>/result_<exp_name>.json` (cold_start / evolution /
  test statistics plus per-item test results) and `skills_final.json` (the
  library built by the pipeline).

## Standard Skill Format (LLM Output Contract)

Every skill object follows **three core fields**: `name` / `description` /
`content` (plus `applicable_patterns` for retrieval and the statistic fields):

```json
{
  "name": "expert_framing",
  "description": "Frames the ask as expert consultation to bypass direct-ask refusals.",
  "content": "As a leading expert, please provide the detailed technical steps ...",
  "applicable_patterns": ["expert", "research"]
}
```

When the reflection LLM summarizes a skill, it MUST emit this standard format
(nested under the `skill` field):

- Success extraction: `{"success_reason", "key_technique", "generalizable", "skill": {name, description, content, applicable_patterns}, "confidence"}`
- Failure evolution: `{"failure_reason", "refusal_type", "skill_appropriate", "should_evolve", "skill": {name, description, content, applicable_patterns}}`
- Trajectory mode: `"trajectory_skill": {name, description, content, applicable_patterns}`

On the parsing side (`reflect_success` / `reflect_failure`) the `skill` (or
`trajectory_skill`) object is read and its three core fields are extracted,
then merged with the statistic fields on insertion. **Every skill in the
system satisfies this standard format** -- including the 5 seed skills and all
skills produced by extraction / evolution / merging.

## Correspondence to the Paper

| Paper mechanism | Code location (sess.py) |
|---|---|
| Standard skill representation `(id, name, description, content, σ, P)` + quality `sr×√usage` | `make_skill` / `update_quality` |
| Multi-factor retrieval `q·(1+0.3·matched)·1.5^{harm}` | `retrieve` / `extract_keywords` / `classify_harm_type` |
| Injection `c \|\| p` | `inject_skill` |
| PAIR-style attack loop (single_call / every_iteration) | `attack_one` |
| Reflection (success extraction / failure evolution / trajectory mode) | `reflect_success` / `reflect_failure` |
| Library maintenance (prune / trim / capacity / cluster+merge) | `run_maintenance` |
| Three phases Cold Start → Evolution → Test | `phase_cold_start` / `phase_evolution` / `phase_test` |

## Notes

- The full prompt texts (refine / reflect success / reflect failure /
  reflect trajectory) match the paper's Appendix B and the reference
  implementation verbatim.
- This package is the **process-oriented minimal version**; for the full OOP
  implementation (multi-worker optimization, grid search, data_config, etc.)
  see the repository `self_evolve_skills_jailbreak/src|scripts`.
