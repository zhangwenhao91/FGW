"""Experimental merge row granularity; default v7 and its local kernel stay unchanged."""
from functools import lru_cache
import tilelang
import tilelang.ascend.language as T


@lru_cache(maxsize=128)
def build_merge(batch, sq, heads, splits, rows=16):
    if rows not in (8, 16, 32) or splits not in (2, 4, 8, 16):
        raise ValueError("rows=8/16/32 and split=2/4/8/16 required")
    if min(batch, sq, heads) <= 0:
        raise ValueError("positive tensor dimensions required")
    BR, D = 128, 128
    dtype, accum_dtype = "bfloat16", "float32"
    q_tiles = (sq + BR - 1) // BR
    tasks = batch * q_tiles * heads
    merge_rows = min(rows, tilelang.next_power_of_2(sq))
    merge_groups = (sq + merge_rows - 1) // merge_rows
    merge_work = batch * heads * merge_groups
    merge_grid = min(merge_work, 32768)

    @T.prim_func
    def merge(
        Parts: T.Tensor((tasks, splits, BR, D), accum_dtype),
        LSE: T.Tensor((tasks * splits * BR,), accum_dtype),
        O: T.Tensor((batch, sq, heads, D), dtype),
    ):
        with T.Kernel(merge_grid) as core:
            for work in T.serial(T.ceildiv(merge_work, merge_grid)):
                program = work * merge_grid + core
                if program < merge_work:
                    b = program // (heads * merge_groups)
                    h = program // merge_groups % heads
                    qi = program % merge_groups * merge_rows
                    group_task = (b * q_tiles + qi // BR) * heads + h
                    row = qi % BR
                    part = T.alloc_shared((merge_rows, D), accum_dtype)
                    score = T.alloc_shared((merge_rows,), accum_dtype)
                    maximum = T.alloc_shared((merge_rows,), accum_dtype)
                    total = T.alloc_shared((merge_rows,), accum_dtype)
                    peak_rows = T.alloc_shared((merge_rows,), accum_dtype)
                    old_weights = T.alloc_shared((merge_rows,), accum_dtype)
                    new_weights = T.alloc_shared((merge_rows,), accum_dtype)
                    acc = T.alloc_shared((merge_rows, D), accum_dtype)
                    out = T.alloc_shared((merge_rows, D), dtype)
                    with T.SimtVF(threads=128):
                        for r, d in T.Parallel(merge_rows, D):
                            acc[r, d] = 0
                        for r in T.Parallel(merge_rows):
                            maximum[r] = -1e30
                            total[r] = 0
                    for split in T.serial(splits):
                        T.copy(Parts[group_task, split, row : row + merge_rows, :], part)
                        offset = (group_task * splits + split) * BR + row
                        T.copy(LSE[offset : offset + merge_rows], score)
                        with T.SimtVF(threads=128):
                            for r in T.Parallel(merge_rows):
                                peak_rows[r] = T.max(maximum[r], score[r])
                                old_weights[r] = T.exp(maximum[r] - peak_rows[r])
                                new_weights[r] = T.if_then_else(
                                    score[r] > -1e29, T.exp(score[r] - peak_rows[r]), 0
                                )
                            for r, d in T.Parallel(merge_rows, D):
                                acc[r, d] = acc[r, d] * old_weights[r] + part[r, d] * new_weights[r]
                            for r in T.Parallel(merge_rows):
                                total[r] = total[r] * old_weights[r] + new_weights[r]
                                maximum[r] = peak_rows[r]
                    with T.SimtVF(threads=128):
                        for r, d in T.Parallel(merge_rows, D):
                            out[r, d] = T.cast(acc[r, d] / T.max(total[r], 1e-30), dtype)
                    T.copy(out, O[b, qi : qi + merge_rows, h, :])

    return tilelang.compile(merge, out_idx=-1, target="ascend", execution_backend="auto", pass_configs={})


def replace_merge(program, kernel, rows):
    """Wrap the exact same partial kernel and Q preparation with another merge."""
    if program.num_splits == 1:
        raise ValueError("Single-split direct output has no merge to replace")

    def launch(q, k, v, table, lengths):
        prepared = program.prepare_q(q)
        parts, lse = program.partial_kernel(prepared, k, v, table, lengths)
        return kernel(parts, lse)

    launch.__dict__.update(program.__dict__)
    launch.merge_kernel = kernel
    launch.merge_rows = rows
    assert launch.partial_kernel is program.partial_kernel
    return launch
