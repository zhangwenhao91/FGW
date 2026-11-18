"""Strict 80-shape original raw SIMD / native FIA report, without tuning."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from benchmarks.mha.summarize_clean_grid import QS, KVS

POLICY = dict(rounds=1, timing_rounds=3, pairs=4, initial_warmup=100, warmup=10,
              plan_policy='stream', merge_policy='rawsimd', merge_rows=8,
              only_static_ascendc=True, raw_baseline_only=True,
              row_softmax=False, raw_breakdown=False)


def extract(report, q, kv, standalone=False):
    if bool(report.get("settings", {}).get("standalone_fused")) != standalone:
        raise ValueError("wrong kernel variant")
    repeats_expected, pairs = (6, 4) if standalone else (3, 4)
    policy = dict(POLICY, timing_rounds=repeats_expected)
    if report.get('status') != 'passed' or report.get('correctness') != 'passed':
        raise ValueError(f"status={report.get('status')} phase={report.get('phase')} {report.get('error', '')}")
    if any(report.get('settings', {}).get(k) != v for k, v in policy.items()):
        raise ValueError('measurement settings differ from frozen raw grid policy')
    if len(report['summary']) != 1 or report.get('timer_primed') is not True:
        raise ValueError('missing summary or Event priming')
    summary = report['summary'][0]
    if (summary['plan']['Q'], summary['plan']['KV']) != (q, kv):
        raise ValueError('shape mismatch')
    if summary['static_builds'] != 1 or summary.get('reference_builds', 0) != 0:
        raise ValueError('unexpected extra builds')
    accuracy = report['cases'][0]['accuracy']
    if {(r['seed'], r['label']) for r in accuracy} != {(s, l) for s in (7, 19) for l in ('static', 'ascendc')}:
        raise ValueError('missing correctness control/seed')
    if standalone:
        if not report.get('source_sha256', {}).get('standalone_fused'):
            raise ValueError('missing standalone kernel hash')
        checks = report['cases'][0].get('repeat_accuracy', [])
        if {(r['repeat'], r['label']) for r in checks} != {
            (i, label) for i in range(1, 7) for label in ('static', 'ascendc')
        }:
            raise ValueError('missing post-repeat accuracy checks')
    repeats = report['measurement_rounds']
    if len(repeats) != repeats_expected or {r['measurement_round'] for r in repeats} != set(range(1, repeats_expected + 1)):
        raise ValueError('incomplete measurement repeats')
    for row in repeats:
        samples = row['samples_us']
        if row['compile_round'] != 1 or set(samples) != {'static', 'ascendc'}:
            raise ValueError('unexpected controls/builds')
        if any(len(v) != pairs or any(not math.isfinite(x) or x <= 0 for x in v) for v in samples.values()):
            raise ValueError('missing or invalid samples')
    med = {label: statistics.median([v for r in repeats for v in r['samples_us'][label]])
           for label in ('static', 'ascendc')}
    ratios = [statistics.median(r['samples_us']['ascendc']) / statistics.median(r['samples_us']['static']) for r in repeats]
    return dict(Q=q, KV=kv, TL_us=med['static'], AscendC_us=med['ascendc'],
                speedup=med['ascendc']/med['static'], repeat_median=statistics.median(ratios),
                repeat_min=min(ratios), repeat_max=max(ratios), wins=sum(x > 1 for x in ratios),
                plan=summary['plan'], samples_us=[r['samples_us'] for r in repeats])


def summarize(directory, standalone=False):
    rows, failures, fingerprint = [], [], None
    for kv in KVS:
        for q in QS:
            key = f'q{q}_kv{kv}'
            path = directory / f'{key}.json'
            if not path.exists():
                continue
            try:
                report = json.loads(path.read_text())
                row = extract(report, q, kv, standalone)
                identity = {k: report[k] for k in ('versions', 'device', 'source_sha256')}
                if fingerprint is None:
                    fingerprint = identity
                if identity != fingerprint:
                    raise ValueError('source/device/version mismatch across shapes')
                rows.append(row)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                failures.append(f'{key}: {exc}')
    complete = len(rows) == 80 and not failures
    lines = [f"status={'passed' if complete else 'incomplete_or_error'} completed={len(rows)}/80",
             'TL=stream host plan + original softmax + raw SIMD division merge (recent retained baseline, not public v7 API).',
             'B1H1 BF16 D128 page128 noncausal; 2 accuracy seeds; 1 build/shape; 3 repeats x 4 alternating pairs.',
             '100 warmups/control/repeat + Event priming; 10 immediate warmups/sample; full-call warm-cache Event.',
             'speedup=AscendC/TL; aggregate=median of all 12 samples/control; no samples discarded.']
    if standalone:
        lines[1:5] = [
            'TL=standalone fused output + stream plan + raw SIMD merge; AscendC=native FIA.',
            'B1H1 BF16 D128 page128 noncausal; 2 accuracy seeds; 1 build/shape; 6 repeats x 4 alternating pairs.',
            '100 warmups/control/repeat + Event priming; 10 immediate warmups/sample.',
            'speedup=AscendC/TL; median of all 24 samples/control.',
        ]
    if rows:
        lines.append(f"geomean_speedup={math.exp(statistics.mean(math.log(r['speedup']) for r in rows)):.6f} "
                     f"faster={sum(r['speedup']>1 for r in rows)}/{len(rows)} min={min(r['speedup'] for r in rows):.6f} max={max(r['speedup'] for r in rows):.6f}")
    lookup = {(r['Q'], r['KV']): r for r in rows}
    lines.append('Q/KV ' + ' '.join(map(str, KVS)))
    for q in QS:
        lines.append(str(q) + ' ' + ' '.join(f"{lookup[q,kv]['speedup']:.4f}" if (q,kv) in lookup else '-' for kv in KVS))
    lines.append('Q KV TL_us AscendC_us speedup repeat_median repeat_min repeat_max wins partials merge_tiles')
    for r in rows:
        lines.append(f"{r['Q']} {r['KV']} {r['TL_us']:.3f} {r['AscendC_us']:.3f} {r['speedup']:.6f} "
                     f"{r['repeat_median']:.6f} {r['repeat_min']:.6f} {r['repeat_max']:.6f} {r['wins']}/{6 if standalone else 3} "
                     f"{r['plan']['partial_count']} {r['plan']['merge_q_tiles']}")
    if standalone:
        lines.append('PER_REPEAT for cases with fewer than six wins: Q KV round TL_us AscendC_us speedup')
        for row in rows:
            if row['wins'] == 6:
                continue
            for index, samples in enumerate(row['samples_us'], 1):
                tl = statistics.median(samples['static'])
                ac = statistics.median(samples['ascendc'])
                lines.append(f"{row['Q']} {row['KV']} {index} {tl:.3f} {ac:.3f} {ac/tl:.6f}")
    lines.extend(failures)
    output = directory.parent / ('mha_standalone_fused_grid_paste.txt' if standalone else 'mha_raw_default_grid_paste.txt')
    output.write_text('\n'.join(lines) + '\n', encoding='utf8')
    (directory/'summary.json').write_text(json.dumps(dict(status='passed' if complete else 'incomplete_or_error',
                                                        completed=len(rows), failures=failures, rows=rows), indent=2)+'\n')
    with (directory/'summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Q', 'KV', 'TL_us', 'AscendC_us', 'speedup', 'repeat_median', 'repeat_min', 'repeat_max', 'wins'])
        for r in rows:
            writer.writerow([r[k] for k in ('Q', 'KV', 'TL_us', 'AscendC_us', 'speedup', 'repeat_median', 'repeat_min', 'repeat_max', 'wins')])
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument("--standalone-fused", action="store_true")
    args = parser.parse_args()
    summarize(args.directory, args.standalone_fused)
