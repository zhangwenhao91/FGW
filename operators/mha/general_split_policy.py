"""Initial task-filling split policy for the general Cube candidate.

Independent of the frozen D128 cost model. Inputs are static shape/capacity;
runtime lengths are partitioned inside the device kernel.
"""
SPLITS = (1, 2, 4, 8, 16, 32)


def select_splits(tasks, capacity_tiles, num_splits=None, target_programs=28):
    if min(tasks, capacity_tiles, target_programs) <= 0:
        raise ValueError("Positive tasks, capacity tiles and target programs required")
    if num_splits is not None:
        if type(num_splits) is not int or num_splits not in SPLITS:
            raise ValueError("General Cube num_splits supports None or 1/2/4/8/16/32")
        if num_splits > capacity_tiles:
            raise ValueError("num_splits must not exceed KV tiles in page-table capacity")
        return num_splits
    splits = 1
    while splits < SPLITS[-1] and tasks * splits < target_programs and splits * 2 <= capacity_tiles:
        splits *= 2
    return splits
