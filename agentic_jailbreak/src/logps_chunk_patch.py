"""强制 old/ref per-token logprobs 分块计算（A100 长轨迹显存治理）+ 现场诊断。

背景（09-15 实测）：swift 的 `_get_per_token_logps_and_entropies` 仅在
`dynamic_num_samples=True`（trajectory 拆分式 scheduler，官方标注 swift 4.4 移除）
时走 `_chunked`；gym 单轨迹调度下该 flag 恒为 False → 每个 micro-batch（16 序列）
全量物化 logits（tokens × 152k vocab），长批次峰值 61GB/80GB 的主要嫌疑来源。
本补丁在 batch > per_device_batch 时强制走既有 `_chunked` 分块路径：
语义不变（同一 forward 的切片拼接），仅压峰值；失败静默回退原实现。

诊断（前 12 次调用 + 每步）写 /tmp/logps_chunk_patch.log（避开 logging 配置黑洞）：
每行含 bs/expected/chunk 判定、调用前后显存、全局 max_memory。

由 src/plugin.py import 触发（仅 trainer 进程加载 plugin）。
"""
import time

LOG = '/tmp/logps_chunk_patch.log'
_N = {'calls': 0, 'steps': 0}


def _dbg(msg):
    try:
        with open(LOG, 'a') as f:
            f.write(f'{time.strftime("%F %T")} {msg}\n')
    except Exception:
        pass


def _gb(n):
    try:
        import torch
        return n / 1e9
    except Exception:
        return 0.0


def apply():
    try:
        import torch
        from swift.rlhf_trainers.grpo_trainer import GRPOTrainer

        orig = GRPOTrainer._get_per_token_logps_and_entropies
        if getattr(orig, '_logps_chunk_patch', False):
            return
        _chunked = GRPOTrainer._get_per_token_logps_and_entropies_chunked

        def _patched(self, model, inputs, compute_entropy=False):
            try:
                bs = inputs['seq_lengths'].shape[0] if self.template.padding_free \
                    else inputs['input_ids'].shape[0]
                mode = 'train' if self.model.training else 'eval'
                expected = self.args.per_device_train_batch_size if mode == 'train' \
                    else self.args.per_device_eval_batch_size
                use_chunk = (not self.dynamic_num_samples) and bs > expected
                _N['calls'] += 1
                if _N['calls'] <= 12:
                    mem0 = _gb(torch.cuda.memory_allocated())
                    _dbg(f"call#{_N['calls']} mode={mode} bs={bs} exp={expected} "
                         f"dyn={self.dynamic_num_samples} chunk={use_chunk} "
                         f"mem_before={mem0:.1f}G max={_gb(torch.cuda.max_memory_allocated()):.1f}G")
                if use_chunk:
                    out = _chunked(self, model, inputs, compute_entropy=compute_entropy)
                    if _N['calls'] <= 12:
                        _dbg(f"call#{_N['calls']} chunked_done mem_after={_gb(torch.cuda.memory_allocated()):.1f}G "
                             f"max={_gb(torch.cuda.max_memory_allocated()):.1f}G")
                    return out
            except Exception as e:
                _dbg(f"call#{_N['calls']} patch_err {type(e).__name__}: {e}")
            return orig(self, model, inputs, compute_entropy=compute_entropy)

        _patched._logps_chunk_patch = True
        GRPOTrainer._get_per_token_logps_and_entropies = _patched

        # 每步边界记录全局峰值（定位峰值是否发生在 logps 阶段之外）
        _orig_step = GRPOTrainer.training_step

        def _step(self, model, inputs, num_items_in_batch=None):
            _N['steps'] += 1
            _dbg(f"step#{_N['steps']} begin max={_gb(torch.cuda.max_memory_allocated()):.1f}G "
                 f"alloc={_gb(torch.cuda.memory_allocated()):.1f}G")
            return _orig_step(self, model, inputs, num_items_in_batch=num_items_in_batch)

        GRPOTrainer.training_step = _step
        _dbg('apply ok: patched logps + step diagnostics')
    except Exception as e:  # 补丁失败不应影响训练
        _dbg(f'apply FAILED: {type(e).__name__}: {e}')


apply()