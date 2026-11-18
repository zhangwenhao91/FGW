"""Original online R8 merge with two-version GM input buffers and a 2-stage pipeline."""
import tilelang.ascend.language as T


def make_kernel(sq, slots, merge_count, max_parts, rows):
    BR, D = 128, 128
    merge_rows = rows
    dtype, accum_dtype = "bfloat16", "float32"
    groups_per_tile = BR // merge_rows
    merge_grid = max(1, merge_count * groups_per_tile)

    @T.prim_func
    def merge_prefetch(
        Parts: T.Tensor((slots, BR, D), accum_dtype),
        LSE: T.Tensor((slots * BR,), accum_dtype),
        Merges: T.Tensor((max(1, merge_count), 3), "int32"),
        O: T.Tensor((1, sq, D), dtype),
    ):
        with T.Kernel(merge_grid) as core:
            group = core // groups_per_tile
            row = core % groups_per_tile * merge_rows
            tile = Merges[group, 0]
            first = Merges[group, 1]
            pieces = Merges[group, 2]
            qi = tile * BR + row
            if qi < sq:
                part = T.alloc_shared((merge_rows, D), accum_dtype)
                score = T.alloc_shared((merge_rows,), accum_dtype)
                maximum = T.alloc_shared((merge_rows,), accum_dtype)
                total = T.alloc_shared((merge_rows,), accum_dtype)
                peak_rows = T.alloc_shared((merge_rows,), accum_dtype)
                old_weights = T.alloc_shared((merge_rows,), accum_dtype)
                new_weights = T.alloc_shared((merge_rows,), accum_dtype)
                acc = T.alloc_shared((merge_rows, D), accum_dtype)
                out = T.alloc_shared((merge_rows, D), dtype)
                T.annotate_buffer_versions({part: 2, score: 2})
                with T.SimtVF(threads=128):
                    for r, d in T.Parallel(merge_rows, D):
                        acc[r, d] = 0
                    for r in T.Parallel(merge_rows):
                        maximum[r] = -1e30
                        total[r] = 0
                for piece in T.Pipelined(max_parts, num_stages=2, annotations={"enable_offset": True}):
                    if piece < pieces:
                        slot = first + piece
                        with T.Stage(0):
                            T.copy(Parts[slot, row : row + merge_rows, :], part)
                            offset = slot * BR + row
                            T.copy(LSE[offset : offset + merge_rows], score)
                        with T.Stage(1):
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
                T.copy(out, O[0, qi : qi + merge_rows, :])


    return merge_prefetch
