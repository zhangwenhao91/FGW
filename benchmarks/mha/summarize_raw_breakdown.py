"""Raw-SIMD warmed component timing; do not infer a pure gap by subtraction."""
import argparse
import json
from pathlib import Path
import statistics

LABELS = ('static', 'static/local', 'static/merge', 'static/chain', 'ascendc')


def summarize(directory):
    lines = ['Q32768 B1H1 BF16 D128 raw+SIMD division; one local/merge build; 6 repeats x 10 rotated five-control groups.',
             '100 initial warmups/control/repeat + Event priming; 10 immediate warmups/sample; no sample filtering.',
             'full: normal preparation/allocations/local/merge; local: prepared Q + normal JIT output/workspace allocations;',
             'merge: fixed valid raw fixture, updates only split rows; chain: prepared Q + allocating local + merge.',
             'Separate Event measurements are not additive. full-chain and chain-local are not pure layout/launch-gap timings.']
    complete_count = 0
    for kv in (4096, 8192):
        path = directory / f'kv{kv}.json'
        if not path.exists():
            lines.append(f'KV{kv}: missing')
            continue
        try:
            report = json.loads(path.read_text())
            settings = report.get('settings', {})
            expected = dict(rounds=1, timing_rounds=6, pairs=10, initial_warmup=100, warmup=10,
                            merge_policy='rawsimd', raw_breakdown=True, only_static_ascendc=True)
            repeats = report.get('measurement_rounds', [])
            complete = (report['status'] == 'passed' and report.get('correctness') == 'passed'
                        and all(settings.get(k) == v for k, v in expected.items())
                        and report.get('timer_primed') is True and len(repeats) == 6
                        and len(report.get('summary', [])) == 1
                        and report['summary'][0]['static_builds'] == 1
                        and report['summary'][0].get('reference_builds') == 0
                        and {r['measurement_round'] for r in repeats} == set(range(1, 7))
                        and all(r['compile_round'] == 1 and set(r['samples_us']) == set(LABELS)
                                and all(len(r['samples_us'][label]) == 10 for label in LABELS) for r in repeats))
            lines.append(f"KV{kv}: status={report['status']} phase={report.get('phase')} repeats={len(repeats)}/6 complete={complete}")
            if complete:
                samples = {label: [x for r in repeats for x in r['samples_us'][label]] for label in LABELS}
                med = {label: statistics.median(values) for label, values in samples.items()}
                lines.append('AGGREGATE_us ' + json.dumps(med, sort_keys=True))
                lines.append('Q_preparation_aliases_input=' + str(report['cases'][0].get('q_preparation_aliases_input')))
                ratios = [r['median_us']['ascendc'] / r['median_us']['static'] for r in repeats]
                lines.append(f"AscendC/full overall={med['ascendc']/med['static']:.6f} "
                             f"repeat_median={statistics.median(ratios):.6f} "
                             f"range=[{min(ratios):.6f},{max(ratios):.6f}] wins={sum(x>1 for x in ratios)}/6")
                complete_count += 1
            for r in repeats:
                lines.append(f"REPEAT {r['measurement_round']} us=" + json.dumps(r['median_us'], sort_keys=True))
                lines.append('SAMPLES_us ' + json.dumps({k: [round(x, 3) for x in v]
                                                       for k, v in r['samples_us'].items()}, sort_keys=True))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            lines.append(f'KV{kv}: invalid report {exc}')
    lines.insert(0, f"status={'passed' if complete_count == 2 else 'incomplete_or_error'} completed={complete_count}/2")
    output = directory.parent / 'mha_raw_breakdown_paste.txt'
    output.write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)
