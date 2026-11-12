"""Experimental split selector calibrated for 28-Cube Ascend950 BF16/D128.

Scores compare strategies, not absolute latency predictions. No NPU imports.
"""
import json
import math
from pathlib import Path

SPLITS = (1, 2, 4, 8, 16)
MODEL_PATH = Path(__file__).with_name("split_cost_model.json")


def features(batch, sq, heads, kv_tiles, splits, cube_cores=28):
    tasks = batch * math.ceil(sq / 128)
    tasks *= heads
    waves = math.ceil(tasks * splits / cube_cores)
    tiles = math.ceil(kv_tiles / splits)
    merge_rows = min(8, 1 << (sq - 1).bit_length())
    merge_work = batch * heads * math.ceil(sq / merge_rows)
    multi = int(splits > 1)
    return (waves * tiles, waves, multi, multi * tasks * splits,
            multi * merge_work * splits, multi * waves * tiles)


def select_splits(batch, sq, heads, kv_tiles):
    if min(batch, sq, heads, kv_tiles) <= 0:
        raise ValueError("positive extents required")
    weights = json.loads(MODEL_PATH.read_text(encoding="utf-8"))["weights"]
    scores = {s: sum(w*x for w, x in zip(weights, features(batch, sq, heads, kv_tiles, s)))
              for s in SPLITS if s <= kv_tiles}
    return min(scores, key=lambda s: (scores[s], s))


def eligible_config(batch, sq, heads, kv_tokens, *, dtype, dim, page_size, causal,
                    pipeline_stages, buffer_versions, kv_versions, q_resident,
                    schedule_mode, full_kv_cache, cube_cores):
    """Default promotion guard; explicit cost_model remains an experimental opt-in."""
    return (full_kv_cache is True and cube_cores == 28
        and str(dtype) in ("bfloat16", "torch.bfloat16") and dim == 128
        and page_size == 128 and not causal and pipeline_stages == 2
        and buffer_versions == 2 and kv_versions == 2 and q_resident
        and schedule_mode == "manual" and 1 <= batch <= 4 and 1 <= heads <= 16
        and batch * heads <= 32 and 1 <= sq <= 8192
        and 512 <= kv_tokens <= 32768 and kv_tokens % 128 == 0)
