"""Partition an existing static task table without changing ownership or slots."""


def split_tables(plan):
    if not plan.merges:
        raise ValueError('Overlap candidate requires split and direct tasks')
    width = plan.max_segments
    phases = []
    for split in (True, False):
        selected = [tuple(task for task in worker if (task[3] >= 0) == split)
                    for worker in plan.programs]
        if not any(selected):
            raise ValueError('Both split and direct work are required')
        counts = tuple(map(len, selected))
        tasks = tuple(worker + ((-1,-1,-1,-1),)*(width-len(worker)) for worker in selected)
        phases.append((counts,tasks))
    return tuple(phases)
