"""All samples of original, serial/parallel two-phase and native FIA."""
import argparse
import json
from pathlib import Path
import statistics

LABELS = ('static','two_phase_serial','two_phase_parallel','ascendc')


def validate(report):
    settings = report.get('settings',{})
    expected = dict(rounds=1,timing_rounds=6,pairs=8,initial_warmup=100,warmup=10,
                    split_overlap=True,split_overlap_trace=False,raw_baseline_only=True,
                    only_static_ascendc=True,merge_policy='rawsimd',plan_policy='stream',row_softmax=False)
    if report.get('status')!='passed' or report.get('correctness')!='passed' or any(settings.get(k)!=v for k,v in expected.items()):
        raise ValueError('incomplete or incorrect measurement policy')
    if report.get('timer_primed') is not True or len(report['summary'])!=1:
        raise ValueError('missing summary/Event priming')
    repeats = report['measurement_rounds']
    if len(repeats)!=6 or {r['measurement_round'] for r in repeats}!=set(range(1,7)):
        raise ValueError('incomplete repeats')
    for r in repeats:
        if r['compile_round']!=1 or set(r['samples_us'])!=set(LABELS) or any(len(v)!=8 for v in r['samples_us'].values()):
            raise ValueError('unexpected controls or samples')
    hashes = report['cases'][0]['builds'][0]['source_sha256']
    if hashes['static_local']!=hashes['two_phase_inplace_local']:
        raise ValueError('device source mismatch')
    checks = report['cases'][0]['accuracy']
    if {(c['seed'],c['label']) for c in checks}!={(s,l) for s in (7,19) for l in LABELS}:
        raise ValueError('missing correctness control/seed')
    repeated = report['cases'][0].get('repeat_accuracy',[])
    if len(repeated)!=24 or {(r['repeat'],r['label']) for r in repeated}!={(i,l) for i in range(1,7) for l in LABELS}:
        raise ValueError('missing post-repeat correctness checks')
    return repeats


def summarize(directory):
    lines = ['Q32768 B1H1 BF16 D128 noncausal page128; original raw SIMD and same-source two-phase local.',
             'One build/control, reused for 6 repeats x 8 balanced rotated four-control groups; all samples retained.',
             '100 warmups/control/repeat + Event priming; 10 immediate warmups/sample; full-call fork/join included.']
    completed = 0
    for kv in (4096,8192):
        path = directory/f'kv{kv}.json'
        if not path.exists():
            lines.append(f'KV{kv}: missing')
            continue
        report = {}
        try:
            report = json.loads(path.read_text())
            repeats = validate(report)
            med = {label:statistics.median([v for r in repeats for v in r['samples_us'][label]]) for label in LABELS}
            lines.append(f'KV{kv}: passed; local device source identical')
            lines.append('AGGREGATE_us '+json.dumps(med,sort_keys=True))
            for numerator,denominator in (('static','two_phase_parallel'),('two_phase_serial','two_phase_parallel'),('ascendc','two_phase_parallel'),('static','two_phase_serial')):
                ratios = [r['median_us'][numerator]/r['median_us'][denominator] for r in repeats]
                lines.append(f'{numerator}/{denominator} overall={med[numerator]/med[denominator]:.6f} repeat_median={statistics.median(ratios):.6f} range=[{min(ratios):.6f},{max(ratios):.6f}] wins={sum(x>1 for x in ratios)}/6')
            for r in repeats:
                lines.append(f"REPEAT {r['measurement_round']} medians_us="+json.dumps(r['median_us'],sort_keys=True))
                lines.append('SAMPLES_us '+json.dumps(r['samples_us'],sort_keys=True))
            completed += 1
        except (ValueError,KeyError,TypeError,IndexError) as exc:
            lines.append(f'KV{kv}: {exc}')
            if report.get('error'):
                lines.append(report['error'])
        trace = directory/f'trace_kv{kv}.json'
        if trace.exists():
            r = json.loads(trace.read_text())
            lines.append(f"TRACE KV{kv} status={r['status']} phase={r.get('phase')}")
            for case in r.get('cases',[]):
                if case.get('trace'):
                    lines.append(json.dumps(case['trace'],sort_keys=True))
            if r.get('error'):
                lines.append(r['error'])
        for prefix in ('guard_kv','guard_trace_kv'):
            guard = directory/f'{prefix}{kv}.jsonl'
            if guard.exists() and guard.read_text().splitlines():
                lines.append(prefix+str(kv)+' last: '+guard.read_text().splitlines()[-1])
    lines.insert(0,f"timing_status={'passed' if completed==2 else 'incomplete_or_error'} completed={completed}/2")
    output = directory.parent/'mha_split_overlap_paste.txt'
    output.write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(output,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    summarize(parser.parse_args().directory)
