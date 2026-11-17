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

    @property
    def max_segments(self):
        return max(map(len, self.programs))

    @property
    def max_parts(self):
        return max((n for _, _, n in self.merges), default=0)

    def describe(self):
        loads = [sum(s[2] for s in program) for program in self.programs]
        return dict(
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
    cursor = 0
    slots = []
    direct = set()
    groups = {}
    loads = []
    for program in plan.programs:
        loads.append(sum(s[2] for s in program))
        for tile, first, count, slot in program:
            assert tile * blocks + first == cursor and 0 < count <= blocks - first
            cursor += count
            if slot < 0:
                assert first == 0 and count == blocks and tile not in direct
                direct.add(tile)
            else:
                slots.append(slot)
                groups.setdefault(tile, []).append(slot)
    assert cursor == ((plan.q + 127) // 128) * blocks
    assert max(loads) - min(loads) <= 1
    assert sorted(slots) == list(range(plan.partial_count))
    assert set(groups).isdisjoint(direct)
    assert len(groups) + len(direct) == (plan.q + 127) // 128
    assert len(plan.merges) <= len(plan.programs) - 1
    for tile, first, count in plan.merges:
        assert count >= 2 and groups[tile] == list(range(first, first + count))
    return True
