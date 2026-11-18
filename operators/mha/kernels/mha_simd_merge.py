"""Explicit SIMD execution of the original online R8 merge algorithm."""
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S


def make_kernel(sq, slots, merge_count, max_parts, rows, raw=False):
    groups_per_tile = 128 // rows
    grid = max(1, merge_count * groups_per_tile)
    stats_plane = slots * 128
    stats_size = stats_plane * (2 if raw else 1)
    stat_pattern = f"PAT_VL{rows}"

    @T.prim_func
    def merge_simd(
        Parts: T.Tensor((slots, 128, 128), "float32"),
        LSE: T.Tensor((stats_size,), "float32"),
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
                part = T.alloc_shared((rows, 128), "float32")
                # Padded stats keep full-width vector loads inside UB buffers.
                score = T.alloc_shared((64,), "float32")
                local_sum = T.alloc_shared((64,), "float32")
                maximum = T.alloc_shared((64,), "float32")
                total = T.alloc_shared((64,), "float32")
                old_weights = T.alloc_shared((64,), "float32")
                new_weights = T.alloc_shared((64,), "float32")
                acc = T.alloc_shared((rows, 128), "float32")
                out = T.alloc_shared((rows, 128), "bfloat16")
                with T.SimdVF(latency=200):
                    full = S.pset(32, "PAT_ALL")
                    zero = S.vdup(T.float32(0), "float32", full)
                    initial = S.vdup(T.float32(-1e30), "float32", full)
                    S.vsts(maximum[0], initial, full, "NORM_B32")
                    S.vsts(total[0], zero, full, "NORM_B32")
                    for r in range(rows):
                        for c in range(0, 128, 64):
                            S.vsts(acc[r, c], zero, full, "NORM_B32")
                for piece in T.serial(max_parts):
                    if piece < pieces:
                        slot = first + piece
                        T.copy(Parts[slot, row : row + rows, :], part)
                        offset = slot * 128 + row
                        T.copy(LSE[offset : offset + rows], score[0:rows])
                        if raw:
                            T.copy(LSE[stats_plane + offset : stats_plane + offset + rows], local_sum[0:rows])
                        with T.SimdVF(latency=400):
                            full = S.pset(32, "PAT_ALL")
                            stats = S.pset(32, stat_pattern)
                            zero = S.vdup(T.float32(0), "float32", full)
                            old_max = S.vld(maximum[0], "NORM")
                            current_score = S.vld(score[0], "NORM")
                            if raw:
                                valid_sum = S.vcmps(S.vld(local_sum[0], "NORM"), T.float32(0), stats, "gt")
                                current_score = S.vsel(current_score, S.vdup(T.float32(-1e30), "float32", full), valid_sum)
                            peak = S.vmax(old_max, current_score, stats)
                            old_weight = S.vexpdif(old_max, peak, stats)
                            valid = S.vcmps(current_score, T.float32(-1e29), stats, "gt")
                            new_weight = S.vsel(S.vexpdif(current_score, peak, stats), zero, valid)
                            if raw:
                                contribution = S.vmul(new_weight, S.vld(local_sum[0], "NORM"), stats)
                                denominator = S.vadd(S.vmul(S.vld(total[0], "NORM"), old_weight, stats), contribution, stats)
                            else:
                                denominator = S.vadd(S.vmul(S.vld(total[0], "NORM"), old_weight, stats), new_weight, stats)
                            S.vsts(maximum[0], peak, stats, "NORM_B32")
                            S.vsts(total[0], denominator, stats, "NORM_B32")
                            S.vsts(old_weights[0], old_weight, stats, "NORM_B32")
                            S.vsts(new_weights[0], new_weight, stats, "NORM_B32")
                            S.mem_bar("VST_VLD")
                            for r in range(rows):
                                alpha = S.vld(old_weights[r], "BRC_B32")
                                beta = S.vld(new_weights[r], "BRC_B32")
                                for c in range(0, 128, 64):
                                    previous = S.vld(acc[r, c], "NORM")
                                    current = S.vld(part[r, c], "NORM")
                                    updated = S.vadd(S.vmul(previous, alpha, full), S.vmul(current, beta, full), full)
                                    S.vsts(acc[r, c], updated, full, "NORM_B32")
                with T.SimdVF(latency=200):
                    full = S.pset(32, "PAT_ALL")
                    minimum = S.vdup(T.float32(1e-30), "float32", full)
                    for r in range(rows):
                        denominator = S.vmax(S.vld(total[r], "BRC_B32"), minimum, full)
                        for c in range(0, 128, 64):
                            value = S.vdiv(S.vld(acc[r, c], "NORM"), denominator, full)
                            converted = S.vcvt(value, "bfloat16", full)
                            S.vsts(out[r, c], converted, full, "PK_B32")
                T.copy(out, O[0, qi : qi + rows, :])

    return merge_simd
