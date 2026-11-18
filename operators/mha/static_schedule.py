"""Pure-host contiguous KV-block scheduling for the B1H1 Cube prototype."""

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Plan:
    q: int
    kv: int
    programs: tuple
    # Each segment is (Q tile, first KV block, number of KV blocks, partial slot).
    # slot=-1 means complete Q tile, written directly to final output.
    merges: tuple  # (Q tile, first partial slot, number of partial slots)
    partial_count: int
    strategy: str = "balanced"

    @property
    def max_segments(self):
        return max(map(len, self.programs))

    @property
    def max_parts(self):
        return max((n for _, _, n in self.merges), default=0)

    def describe(self):
        loads = [sum(s[2] for s in program) for program in self.programs]
        return dict(
            strategy=self.strategy,
            Q=self.q,
            KV=self.kv,
            programs=len(loads),
            q_tiles=(self.q + 127) // 128,
            min_kv_blocks=min(loads),
            max_kv_blocks=max(loads),
            segments=sum(map(len, self.programs)),
            max_segments=self.max_segments,
            partial_count=self.partial_count,
            merge_q_tiles=len(self.merges),
            direct_q_tiles=(self.q + 127) // 128 - len(self.merges),
            max_parts=self.max_parts,
            partial_bytes=self.partial_count * 128 * 128 * 4,
            lse_bytes=self.partial_count * 128 * 4,
        )


@lru_cache(maxsize=128)
def make_plan(q, kv, cores=28):
    if type(q) is not int or type(kv) is not int or q <= 0 or kv <= 0 or kv % 128:
        raise ValueError("positive integer Q and positive 128-aligned KV required")
    if type(cores) is not int or not 1 <= cores <= 28:
        raise ValueError("cores must be 1..28")
    tiles, blocks = (q + 127) // 128, kv // 128
    total = tiles * blocks
    if total > 2**31 - 1:
        raise ValueError("int32 scheduling range exceeded")
    cores = min(cores, total)
    programs = [[] for _ in range(cores)]
    owners = [[] for _ in range(tiles)]
    for pid in range(cores):
        first = pid * (total // cores) + min(pid, total % cores)
        stop = first + total // cores + int(pid < total % cores)
        while first < stop:
            tile, begin = divmod(first, blocks)
            end = min(stop, (tile + 1) * blocks)
            idx = len(programs[pid])
            programs[pid].append([tile, begin, end - first, -1])
            owners[tile].append((pid, idx))
            first = end
    merges = []
    slot = 0
    for tile, segments in enumerate(owners):
        if len(segments) > 1:
            merges.append((tile, slot, len(segments)))
            for pid, idx in segments:
                programs[pid][idx][3] = slot
                slot += 1
    plan = Plan(q, kv, tuple(tuple(tuple(s) for s in p) for p in programs), tuple(merges), slot)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    blocks = plan.kv // 128
    tiles = (plan.q + 127) // 128
    slots, direct, groups, intervals, loads = [], set(), {}, {}, []
    for program in plan.programs:
        assert program
        loads.append(sum(s[2] for s in program))
        for tile, first, count, slot in program:
            assert 0 <= tile < tiles and 0 <= first < blocks and 0 < count <= blocks - first
            intervals.setdefault(tile, []).append((first, first + count))
            if slot < 0:
                assert first == 0 and count == blocks and tile not in direct
                direct.add(tile)
            else:
                slots.append(slot)
                groups.setdefault(tile, []).append(slot)
    assert set(intervals) == set(range(tiles))
    for spans in intervals.values():
        cursor = 0
        for first, stop in sorted(spans):
            assert first == cursor
            cursor = stop
        assert cursor == blocks
    assert max(loads) - min(loads) <= (1 if plan.strategy == "balanced" else blocks)
    assert sorted(slots) == list(range(plan.partial_count))
    assert set(groups).isdisjoint(direct)
    assert len(groups) + len(direct) == tiles
    assert len(plan.merges) <= len(plan.programs) - 1
    assert len({tile for tile, _, _ in plan.merges}) == len(plan.merges)
    assert {tile for tile, _, _ in plan.merges} == set(groups)
    for tile, first, count in plan.merges:
        assert count >= 2 and sorted(groups[tile]) == list(range(first, first + count))
    return True


@lru_cache(maxsize=128)
def make_tail_plan(q, kv, cores=28, pieces=None):
    """Whole Q tiles in full waves; split only Q tiles in the final wave."""
    make_plan(q, kv, cores)
    if pieces is not None and (type(pieces) is not int or pieces < 2):
        raise ValueError("pieces must be None or an integer >=2")
    tiles, blocks = (q + 127) // 128, kv // 128
    full, tail = divmod(tiles, cores)
    if not full or not tail or blocks < 2:
        return make_q_plan(q, kv, cores)
    programs = [[(tile, 0, blocks, -1) for tile in range(pid * full, (pid + 1) * full)]
                for pid in range(cores)]
    merges, slot, receiver = [], 0, 0
    for extra in range(tail):
        tile = full * cores + extra
        count = min(blocks, pieces or (cores // tail + int(extra < cores % tail)))
        # Explicit larger counts may fill multiple waves of tail fragments.
        if count == 1:
            programs[receiver % cores].append((tile, 0, blocks, -1))
            receiver += 1
            continue
        merges.append((tile, slot, count))
        for part in range(count):
            first = part * (blocks // count) + min(part, blocks % count)
            size = blocks // count + int(part < blocks % count)
            programs[receiver % cores].append((tile, first, size, slot))
            receiver += 1
            slot += 1
    if not merges:
        return make_q_plan(q, kv, cores)
    plan = Plan(q, kv, tuple(tuple(p) for p in programs), tuple(merges), slot, "tail")
    validate_plan(plan)
    return plan


@lru_cache(maxsize=128)
def make_q_plan(q, kv, cores=28):
    """Assign whole consecutive Q tiles; never create partial KV states."""
    # Reuse argument/range validation from the balanced planner.
    make_plan(q, kv, cores)
    tiles, blocks = (q + 127) // 128, kv // 128
    cores = min(cores, tiles)
    programs = []
    for pid in range(cores):
        first = pid * (tiles // cores) + min(pid, tiles % cores)
        count = tiles // cores + int(pid < tiles % cores)
        programs.append(tuple((tile, 0, blocks, -1) for tile in range(first, first + count)))
    plan = Plan(q, kv, tuple(programs), (), 0, "q_only")
    validate_plan(plan)
    return plan


# Initial stage-cost estimates in microseconds, not a fitted latency predictor.
# Fixed for the entire grid: no shape lookup or runtime timing/search on device.
COST_MODEL = dict(block_us=0.72, segment_us=2.0, merge_launch_us=12.0,
                  partial_us=0.20, merge_tile_us=0.15)


def estimate_cost(plan):
    """Critical program work plus selective merge; common call costs cancel."""
    c = COST_MODEL
    local = max(sum(s[2] for s in program) * c['block_us']
                + len(program) * c['segment_us'] for program in plan.programs)
    merge = (c['merge_launch_us'] + plan.partial_count * c['partial_us']
             + len(plan.merges) * c['merge_tile_us']) if plan.merges else 0.0
    return dict(local_us=local, merge_us=merge, total_us=local + merge)


@lru_cache(maxsize=128)
def choose_plan(q, kv):
    """Evaluate whole-Q and balanced-KV plans using 1..28 programs on host."""
    candidates = [make_q_plan(q, kv)] + [make_plan(q, kv, cores) for cores in range(1, 29)]
    return min(candidates, key=lambda p: (estimate_cost(p)['total_us'],
                                         p.partial_count, -len(p.programs)))


@lru_cache(maxsize=128)
def make_stream_plan(q, kv, cores=28):
    """Whole-Q-first host assignment with a remaining-work target per core.

    Independent simplified implementation for fixed 128x128 dense tiles;
    follows the assignment order used by FIA split_core_v2, not its ABI.
    """
    make_plan(q, kv, cores)  # Validate scalar/range constraints.
    tiles, blocks = (q + 127) // 128, kv // 128
    cores = min(cores, tiles * blocks)
    programs, cursor = [], 0
    total = tiles * blocks
    for pid in range(cores):
        if cursor == total:
            break
        # Half-block tolerance, as in the source's whole-row/block assignment.
        target = (total - cursor) / (cores - pid)
        assigned, tasks = 0, []
        while cursor < total:
            tile, first = divmod(cursor, blocks)
            remaining = blocks - first
            if assigned + remaining <= target + 0.5 or pid == cores - 1:
                take = remaining  # Prefer finishing the current Q tile.
            else:
                take = min(remaining, max(0, int(target + 0.5 - assigned)))
                if not tasks:
                    take = max(1, take)
                if take == 0:
                    break
            tasks.append([tile, first, take, -1])
            cursor += take
            assigned += take
            if assigned >= target - 0.5:
                break
        programs.append(tasks)
    assert cursor == total
    owners = {}
    for pid, tasks in enumerate(programs):
        for idx, (tile, _, _, _) in enumerate(tasks):
            owners.setdefault(tile, []).append((pid, idx))
    merges, slot = [], 0
    for tile, entries in sorted(owners.items()):
        if len(entries) > 1:
            merges.append((tile, slot, len(entries)))
            for pid, idx in entries:
                programs[pid][idx][3] = slot
                slot += 1
    if not merges:
        # The direct ABI derives ranges from pid; retain its canonical layout.
        return make_q_plan(q, kv, len(programs))
    plan = Plan(q, kv, tuple(tuple(tuple(s) for s in tasks) for tasks in programs),
                tuple(merges), slot, "stream")
    validate_plan(plan)
    return plan


@lru_cache(maxsize=128)
def choose_stream_plan(q, kv):
    """Compare split/no-split with an explicit gain threshold; no shape table."""
    direct = make_q_plan(q, kv)
    candidates = [make_stream_plan(q, kv, cores) for cores in range(1, 29)]
    # Also allow the independent final-wave assignment with adaptive pieces.
    # This is our extension, not a claim of reproducing AscendC's final plan.
    candidates.extend(make_tail_plan(q, kv, cores) for cores in range(1, 29))
    split = min(candidates, key=lambda p: (estimate_cost(p)['total_us'], p.partial_count))
    # Existing score already charges merge; require one additional block's gain.
    if estimate_cost(direct)['total_us'] - estimate_cost(split)['total_us'] > COST_MODEL['block_us']:
        return split
    return direct


@lru_cache(maxsize=128)
def make_merge_tasks(plan, vectors=56):
    """Allocate vector programs by valid Q rows times per-group KV pieces.

    Each record is (Q tile, first partial slot, pieces, first row, row count).
    This is host metadata only; no device queue or online timing.
    """
    if type(vectors) is not int or vectors < len(plan.merges) or vectors <= 0:
        raise ValueError("positive vector budget, at least one per merge group required")
    if not plan.merges:
        return ()
    groups = [(tile, first, pieces, min(128, plan.q - tile * 128))
              for tile, first, pieces in plan.merges]
    budget = min(vectors, sum(rows for _, _, _, rows in groups))
    total_work = sum(rows * pieces for _, _, pieces, rows in groups)
    average = (total_work + budget - 1) // budget
    spare = budget - len(groups)
    tasks = []
    for tile, first, pieces, rows in groups:
        count = min(rows, spare + 1, max(1, rows * pieces // average))
        stride = (rows + count - 1) // count
        count = (rows + stride - 1) // stride
        spare -= count - 1
        for begin in range(0, rows, stride):
            tasks.append((tile, first, pieces, begin, min(stride, rows - begin)))
    assert len(tasks) <= vectors
    return tuple(tasks)


@lru_cache(maxsize=128)
def make_block_merge_tasks(plan, vectors=56, rows=8):
    """Assign original fixed-size DMA chunks to weighted static workers."""
    if type(vectors) is not int or vectors <= 0 or rows != 8:
        raise ValueError("positive vector budget and rows=8 required")
    chunks = [(tile, first, pieces, begin, min(rows, plan.q - tile * 128 - begin))
              for tile, first, pieces in plan.merges
              for begin in range(0, min(128, plan.q - tile * 128), rows)]
    if not chunks:
        return ()
    workers = [[] for _ in range(min(vectors, len(chunks)))]
    loads = [0] * len(workers)
    for task in sorted(chunks, key=lambda t: (-t[2] * t[4], t[0], t[3])):
        pid = min(range(len(workers)), key=lambda i: (loads[i], len(workers[i]), i))
        workers[pid].append(task)
        loads[pid] += task[2] * task[4]
    return tuple(tuple(tasks) for tasks in workers)
