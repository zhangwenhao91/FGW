"""Five-shape direct output normalization/cast experiment, no sample trimming."""
import argparse
import json
import math
from pathlib import Path
import statistics

KVS=(512,1024,2048,4096,8192)
LABELS=('static','raw_ref','ascendc')
POLICY=dict(rounds=1,timing_rounds=6,pairs=9,initial_warmup=100,warmup=10,
            plan_policy='stream',merge_policy='rawsimd',fused_output=True,
            row_softmax=False,raw_baseline_only=False,raw_breakdown=False,
            split_overlap=False,only_static_ascendc=True)


def validate(report,kv,q=32768):
    if report.get('status')!='passed' or report.get('correctness')!='passed' or report.get('output_variant')!='simd_div_cast_v1':
        raise ValueError('incomplete report or wrong variant')
    if any(report.get('settings',{}).get(k)!=v for k,v in POLICY.items()):
        raise ValueError('wrong measurement policy')
    if report.get('timer_primed') is not True or len(report['summary'])!=1:
        raise ValueError('missing summary/Event priming')
    summary=report['summary'][0]
    if (summary['plan']['Q'],summary['plan']['KV'])!=(q,kv) or summary['static_builds']!=1 or summary.get('reference_builds')!=1:
        raise ValueError('wrong shape or build counts')
    info=report['cases'][0]
    if {(r['seed'],r['label']) for r in info['accuracy']}!={(s,l) for s in (7,19) for l in LABELS}:
        raise ValueError('missing correctness control/seed')
    if {(r['repeat'],r['label']) for r in info.get('repeat_accuracy',[])}!={(i,l) for i in range(1,7) for l in LABELS}:
        raise ValueError('missing repeated accuracy checks')
    hashes=info['builds'][0]['source_sha256']
    if hashes.get('static_merge')!=hashes.get('raw_simd_ref_merge'):
        raise ValueError('merge source differs')
    repeats=report['measurement_rounds']
    if len(repeats)!=6 or {r['measurement_round'] for r in repeats}!=set(range(1,7)):
        raise ValueError('incomplete repeats')
    for r in repeats:
        if r['compile_round']!=1 or set(r['samples_us'])!=set(LABELS):
            raise ValueError('wrong controls/builds')
        if any(len(v)!=9 or any(not math.isfinite(x) or x<=0 for x in v) for v in r['samples_us'].values()):
            raise ValueError('missing/invalid samples')
    return repeats


def summarize(directory):
    lines=['Q32768 B1H1 BF16 D128/page128 noncausal; static=fused SIMD div+cast; raw_ref=original raw SIMD; ascendc=native FIA.',
           'Same task plan/softmax/QK/PV/merge; one build per TL variant; six repeats x nine rotated triples.',
           '100 initial warmups/control/repeat + Event priming; 10 immediate warmups/sample; full-call warm-cache Event.',
           'All 54 samples/control retained in case JSON; summary uses pooled medians and per-repeat ratios; no outlier removal.']
    completed=0
    for kv in KVS:
        path=directory/f'kv{kv}.json'
        if not path.exists():
            lines.append(f'KV{kv}: missing'); continue
        report={}
        try:
            report=json.loads(path.read_text())
            repeats=validate(report,kv)
            med={label:statistics.median([v for r in repeats for v in r['samples_us'][label]]) for label in LABELS}
            lines.append(f'KV{kv}: passed')
            lines.append('AGGREGATE_us '+json.dumps(med,sort_keys=True))
            for control in ('raw_ref','ascendc'):
                ratios=[statistics.median(r['samples_us'][control])/statistics.median(r['samples_us']['static']) for r in repeats]
                lines.append(f'{control}/fused overall={med[control]/med["static"]:.6f} repeat_median={statistics.median(ratios):.6f} range=[{min(ratios):.6f},{max(ratios):.6f}] wins={sum(x>1 for x in ratios)}/6')
            for r in repeats:
                lines.append(f"REPEAT {r['measurement_round']} medians_us="+json.dumps({k:round(statistics.median(v),3) for k,v in r['samples_us'].items()},sort_keys=True))
            completed+=1
        except (ValueError,KeyError,TypeError,IndexError) as exc:
            lines.append(f'KV{kv}: {exc}')
            if report.get('error'): lines.append(report['error'])
        guard=directory/f'guard_kv{kv}.jsonl'
        if guard.exists() and guard.read_text().splitlines():
            lines.append('guard_last: '+guard.read_text().splitlines()[-1])
    lines.insert(0,f"status={'passed' if completed==5 else 'incomplete_or_error'} completed={completed}/5")
    output=directory.parent/'mha_fused_output_paste.txt'
    output.write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    summarize(parser.parse_args().directory)
