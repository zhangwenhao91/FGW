"""Fixed-R8 DMA merge with all row weights computed before Partial reduction."""
import tilelang.ascend.language as T


def make_kernel(sq, slots, merge_count, max_parts, rows):
    groups_per_tile = 128 // rows
    grid = max(1, merge_count * groups_per_tile)
    weight_stride = max(8, rows)  # 32-byte-aligned UB row for small-Q variants.

    @T.prim_func
    def merge_preweighted(
        Parts: T.Tensor((slots, 128, 128), "float32"),
        LSE: T.Tensor((slots * 128,), "float32"),
        Merges: T.Tensor((max(1, merge_count), 3), "int32"),
        O: T.Tensor((1, sq, 128), "bfloat16"),
    ):
        with T.Kernel(grid) as core:
            group = core // groups_per_tile
            row = core % groups_per_tile * rows
            tile = Merges[group, 0]
            first = Merges[group, 1]
            pieces = Merges[group, 2]
            qi = tile * 128 + row
            if qi < sq:
                weights = T.alloc_shared((max_parts, weight_stride), "float32")
                maximum = T.alloc_shared((rows,), "float32")
                total = T.alloc_shared((rows,), "float32")
                part = T.alloc_shared((rows, 128), "float32")
                acc = T.alloc_shared((rows, 128), "float32")
                out = T.alloc_shared((rows, 128), "bfloat16")
                # Retain the original contiguous LSE/Partial DMA access shapes.
                for piece in T.serial(max_parts):
                    if piece < pieces:
                        offset = (first + piece) * 128 + row
                        T.copy(LSE[offset : offset + rows], weights[piece, 0:rows])
                with T.SimtVF(threads=128):
                    for r in T.Parallel(rows):
                        row_max = T.alloc_local((1,), "float32")
                        row_max[0] = -1e30
                        for piece in T.serial(max_parts):
                            if piece < pieces:
                                row_max[0] = T.max(row_max[0], weights[piece, r])
                        maximum[r] = row_max[0]
                with T.SimtVF(threads=128):
                    for piece, r in T.Parallel(max_parts, rows):
                        if piece < pieces:
                            weights[piece, r] = T.if_then_else(
                                weights[piece, r] > -1e29,
                                T.exp(weights[piece, r] - maximum[r]), 0)
                with T.SimtVF(threads=128):
                    for r in T.Parallel(rows):
                        row_sum = T.alloc_local((1,), "float32")
                        row_sum[0] = 0
                        for piece in T.serial(max_parts):
                            if piece < pieces:
                                row_sum[0] += weights[piece, r]
                        total[r] = row_sum[0]
                with T.SimtVF(threads=128):
                    for piece, r in T.Parallel(max_parts, rows):
                        if piece < pieces:
                            weights[piece, r] /= T.max(total[r], 1e-30)
                    for r, d in T.Parallel(rows, 128):
                        acc[r, d] = 0
                for piece in T.serial(max_parts):
                    if piece < pieces:
                        T.copy(Parts[first + piece, row : row + rows, :], part)
                        with T.SimtVF(threads=128):
                            for r, d in T.Parallel(rows, 128):
                                acc[r, d] += part[r, d] * weights[piece, r]
                with T.SimtVF(threads=128):
                    for r, d in T.Parallel(rows, 128):
                        out[r, d] = T.cast(acc[r, d], "bfloat16")
                T.copy(out, O[0, qi : qi + rows, :])

    return merge_preweighted
