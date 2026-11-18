"""Row/part weighted merge task entry for the static B1H1 BF16 prototype."""
import tilelang
import tilelang.ascend.language as T


def make_kernel(sq, slots, tasks, max_parts):
    grid = len(tasks)
    rows = tilelang.next_power_of_2(max(task[4] for task in tasks))

    @T.prim_func
    def weighted_merge(
        Parts: T.Tensor((slots, 128, 128), "float32"),
        LSE: T.Tensor((slots * 128,), "float32"),
        Tasks: T.Tensor((grid, 5), "int32"),
        O: T.Tensor((1, sq, 128), "bfloat16"),
    ):
        with T.Kernel(grid) as core:
            tile = Tasks[core, 0]
            first = Tasks[core, 1]
            pieces = Tasks[core, 2]
            begin = Tasks[core, 3]
            valid = Tasks[core, 4]
            maximum = T.alloc_shared((rows,), "float32")
            total = T.alloc_shared((rows,), "float32")
            old_weight = T.alloc_shared((rows,), "float32")
            new_weight = T.alloc_shared((rows,), "float32")
            acc = T.alloc_shared((rows, 128), "float32")
            with T.SimtVF(threads=128):
                for r in T.Parallel(rows):
                    maximum[r] = -1e30
                    total[r] = 0
                for r, d in T.Parallel(rows, 128):
                    acc[r, d] = 0
            for piece in T.serial(max_parts):
                if piece < pieces:
                    slot = first + piece
                    with T.SimtVF(threads=128):
                        for r in T.Parallel(rows):
                            if r < valid:
                                score = LSE[slot * 128 + begin + r]
                                peak = T.max(maximum[r], score)
                                old_weight[r] = T.exp(maximum[r] - peak)
                                new_weight[r] = T.if_then_else(score > -1e29, T.exp(score - peak), 0)
                                total[r] = total[r] * old_weight[r] + new_weight[r]
                                maximum[r] = peak
                        for r, d in T.Parallel(rows, 128):
                            if r < valid:
                                acc[r, d] = acc[r, d] * old_weight[r] + Parts[slot, begin + r, d] * new_weight[r]
            with T.SimtVF(threads=128):
                for r, d in T.Parallel(rows, 128):
                    if r < valid:
                        O[0, tile * 128 + begin + r, d] = T.cast(acc[r, d] / T.max(total[r], 1e-30), "bfloat16")

    return weighted_merge
