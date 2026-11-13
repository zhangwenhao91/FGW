"""Frozen experimental general-Cube split cost model; independent of v7 D128.

Pure Python prediction, no NPU imports. Model uses static tensor extents and
host-known cache lengths only to choose a split; device lengths remain dynamic.
"""
from functools import lru_cache
import json
import math
from pathlib import Path

from operators.mha.general_split_policy import SPLITS

MODEL_PATH = Path(__file__).with_name("general_split_cost_model.json")
FEATURE_NAMES = (
    "constant", "wave_tiles_times_padded_d", "wave_tiles", "waves_times_padded_d",
    "waves", "has_merge", "merge_rows_times_splits_times_padded_d",
    "partial_tasks_times_splits_times_padded_d",
)
FEATURE_NAMES_V2 = FEATURE_NAMES + ("longest_partition_excess_times_padded_d", "longest_partition_excess")


def model_path(version=1):
    if type(version) is not int or version not in (1, 2):
        raise ValueError("General model version must be1 or2")
    return MODEL_PATH if version == 1 else MODEL_PATH.with_name("general_split_cost_model_v2.json")


def feature_names(version=1):
    model_path(version)
    return FEATURE_NAMES if version == 1 else FEATURE_NAMES_V2


@lru_cache(maxsize=2)
def load_model(version=1):
    model = json.loads(model_path(version).read_text(encoding="utf-8"))
    names = feature_names(version)
    if model["version"] != version or model["feature_names"] != list(names) or len(model["weights"]) != len(names):
        raise ValueError("General split model feature mismatch")
    if any(not math.isfinite(v) or v < 0 for v in model["weights"]):
        raise ValueError("General split model requires finite nonnegative weights")
    return model


def features(batch, sq, heads, dim, lengths, splits, version=1):
    feature_names(version)
    if min(batch, sq, heads) <= 0 or not 16 <= dim <= 256 or dim % 16:
        raise ValueError("Positive B/Q/H and D16..256 step16 required")
    if len(lengths) != batch or any(type(n) is not int or n < 0 for n in lengths):
        raise ValueError("One nonnegative integer cache length per request required")
    if type(splits) is not int or splits not in SPLITS:
        raise ValueError("Unsupported split count")
    padded_d = max(32, 1 << (dim-1).bit_length()) / 64
    tasks = batch*heads*((sq+31)//32)
    waves = (tasks*splits+27)//28
    mean_tiles = sum((n+31)//32 for n in lengths)/batch
    merge_rows = min(8, 1 << (sq-1).bit_length())
    merge_work = batch*heads*((sq+merge_rows-1)//merge_rows)
    multi = int(splits > 1)
    values = [1.0, waves*mean_tiles/splits*padded_d, waves*mean_tiles/splits,
            waves*padded_d, float(waves), float(multi),
            multi*merge_work*splits*padded_d/56,
            multi*tasks*splits*padded_d/28]
    if version == 2:
        longest = (max((n+31)//32 for n in lengths)+splits-1)//splits
        excess = max(0.0, longest-mean_tiles/splits)
        values += [excess*padded_d, excess]
    return values


def predict_us(batch, sq, heads, dim, lengths, splits, model=None, version=1):
    model = load_model(version) if model is None else model
    raw = sum(a*b for a, b in zip(features(batch, sq, heads, dim, lengths, splits, model["version"]), model["weights"]))
    floor = model["latency_floor_us"]["multi" if splits > 1 else "single"]
    return max(raw, floor)


def select_model_splits(batch, sq, heads, dim, page_size, width, lengths, model=None, version=1):
    if page_size <= 0 or page_size % 16 or width <= 0:
        raise ValueError("Positive page16-multiple and table width required")
    if lengths is None:
        raise ValueError("Experimental cost model requires host-known cache_lengths")
    if len(lengths) != batch or any(n > page_size*width for n in lengths):
        raise ValueError("cache_lengths must match batch and page-table capacity")
    model = load_model(version) if model is None else model
    capacity_tiles = (page_size*width+31)//32
    costs = {s: predict_us(batch, sq, heads, dim, lengths, s, model) for s in SPLITS if s <= capacity_tiles}
    best = min(costs.values())
    # Prefer less scratch/merge work when predicted latency differs by <=2%.
    budget = max(best*model["selection_tolerance"], model.get("absolute_selection_margin_us", 0.0))
    limit = best*(1+model["selection_tolerance"]) if model["version"] == 1 else best+budget
    return min(s for s, value in costs.items() if value <= limit)
