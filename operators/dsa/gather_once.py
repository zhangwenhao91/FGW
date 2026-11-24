"""Gather one contiguous KV workspace per query for both Q-head groups."""

from functools import lru_cache

import tilelang
import tilelang.ascend.language as T


@lru_cache(maxsize=4)
def build_gather(sq, sk, topk):
    rows, width, cores = 16, 576, 48
    if not (0 < sq <= sk and topk > 0 and topk % rows == 0):
        raise ValueError("expected right-aligned queries and whole gather tiles")
    per_query = topk // rows
    tasks = sq * per_query
    grid = min(cores, tasks)

    @T.prim_func
    def gather(
        KV: T.Tensor((sk, width), "bfloat16"),
        Indices: T.Tensor((sq, topk), "int32"),
        Packed: T.Tensor((sq * topk, width), "bfloat16"),
    ):
        with T.Kernel(grid) as bx:
            tile = T.alloc_shared((rows, width), "bfloat16")
            for step in T.serial(T.ceildiv(tasks, grid)):
                task = step * grid + bx
                if task < tasks:
                    query = task // per_query
                    slot = task % per_query * rows
                    with T.SimtVF(threads=128):
                        for row, col in T.Parallel(rows, width):
                            tile[row, col] = 0
                    for pair in T.serial(rows // 2):
                        pos = pair * 2
                        first = Indices[query, slot + pos]
                        second = Indices[query, slot + pos + 1]
                        limit = sk - sq + query
                        if first >= 0 and second == first + 1 and second <= limit:
                            T.copy(KV[first : first + 2, :], tile[pos : pos + 2, :])
                        else:
                            if first >= 0 and first <= limit:
                                T.copy(KV[first, :], tile[pos, :])
                            if second >= 0 and second <= limit:
                                T.copy(KV[second, :], tile[pos + 1, :])
                    T.copy(tile, Packed[task * rows : (task + 1) * rows, :])

    return tilelang.compile(gather, target="ascend", out_idx=[])
