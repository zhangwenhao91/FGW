"""Parse retained FFTS PipeUtilization records using the inspected TileLang layout.

Layout reference: tilelang.profiler.msprof._parse_ffts_profile. Preserve raw
records/source alongside this interpretation; reject malformed or empty data.
No physical-core identifier or exclusive stall-time interpretation is assumed.
"""
import statistics
import struct
from pathlib import Path

PIPES = dict(vec=4, mad=5, scalar=6, mte1=7, mte2=8, mte3=9, fixpipe=12)


def parse_profile(directory, kernel_substring='local_split'):
    directory = Path(directory)
    names, task_names = {}, {}
    for path in directory.rglob('*hash_dic.slice_*'):
        if path.name.endswith('.done'):
            continue
        for line in path.read_bytes().decode('utf8', errors='replace').splitlines():
            key, sep, value = line.partition(':')
            if sep:
                try:
                    hashed = int(key)
                    if hashed >= 2**63:
                        hashed -= 2**64
                    names[hashed] = value.strip()
                except ValueError:
                    pass
    for path in directory.rglob('*task_track.slice_*'):
        if path.name.endswith('.done'):
            continue
        data = path.read_bytes()
        for offset in range(0, len(data) - 63, 64):
            record = struct.unpack_from('<8q', data, offset)
            if record[5]:
                task_names[(record[3] >> 32) & 0xFFFF] = names.get(record[5], '<unknown>')
    groups = {}
    observed = set()
    files = []
    for path in directory.rglob('ffts_profile*'):
        if not path.is_file() or path.name.endswith('.done') or not path.stat().st_size:
            continue
        data = path.read_bytes()
        if len(data) % 128:
            raise ValueError(f'Unsupported FFTS record length: {path} ({len(data)} bytes)')
        files.append(str(path.relative_to(directory)))
        for offset in range(0, len(data), 128):
            record = struct.unpack_from('<16q', data, offset)
            if record[1] <= 0:
                continue
            seq = (record[0] >> 32) & 0xFFFF
            name = task_names.get(seq, '<unknown>')
            observed.add(name)
            if kernel_substring not in name:
                continue
            if record[15] < record[14]:
                raise ValueError('Unsupported FFTS timestamps')
            pipes = {pipe: record[index] / record[1] for pipe, index in PIPES.items()}
            if any(value < 0 or value > 1.01 for value in pipes.values()):
                raise ValueError('FFTS pipe ratio outside expected range; inspect saved profiler source/raw files')
            kind = 'aiv' if record[2] >= 1 << 31 else 'aic'
            groups.setdefault((seq, name), {'aic': [], 'aiv': []})[kind].append(
                dict(cycles=record[1], duration_ns=record[15]-record[14], pipes=pipes,
                     start_ns=record[14], end_ns=record[15]))
    if not groups:
        raise ValueError(f'No {kernel_substring!r} FFTS records; observed names={sorted(observed)}; raw files retained')

    def summarize(records, kind):
        if not records:
            return None
        relevant = ('mad', 'scalar', 'mte1', 'mte2', 'mte3', 'fixpipe') if kind == 'aic' else ('vec', 'scalar', 'mte2', 'mte3')
        cycles = sum(r['cycles'] for r in records)
        durations = [r['duration_ns']/1000 for r in records]
        return dict(records=len(records), cycles_sum=cycles,
                    cycles_min=min(r['cycles'] for r in records),
                    cycles_max=max(r['cycles'] for r in records),
                    record_duration_us=dict(min=min(durations), median=statistics.median(durations), max=max(durations)),
                    pipes={pipe: dict(cycle_weighted=sum(r['pipes'][pipe]*r['cycles'] for r in records)/cycles,
                                      min=min(r['pipes'][pipe] for r in records), max=max(r['pipes'][pipe] for r in records))
                           for pipe in relevant})

    combined = {kind: [r for group in groups.values() for r in group[kind]] for kind in ('aic', 'aiv')}
    if not all(combined.values()):
        raise ValueError('Expected both AIC and AIV records for mixed local kernel; raw files retained')
    return dict(kernel_filter=kernel_substring, files=files, task_groups=len(groups),
                aggregate={kind: summarize(records, kind) for kind, records in combined.items()},
                tasks=[dict(sequence=seq, name=name,
                            **{kind: summarize(records, kind) for kind, records in group.items()})
                       for (seq, name), group in sorted(groups.items())],
                notes=['Ratios have separate AIC/AIV cycle denominators, overlap and are not additive.',
                       'Record durations and cycles are not Event latency or exclusive stall time.',
                       'Record counts are not physical-core counts; task sequences may separate AIC/AIV.'])
