"""80-shape fused-output triple: strict completeness and resumable reports."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from benchmarks.mha.summarize_clean_grid import QS,KVS
from benchmarks.mha.summarize_fused_output import LABELS,validate

ROOT=Path(__file__).resolve().parents[2]


def current_sources():
    paths=dict(candidate='operators/mha/kernels/mha_static_host_candidate.py',
               default='operators/mha/kernels/mha_paged_decode_ascend_v7.py',
               plan='operators/mha/static_schedule.py',
               weighted_merge='operators/mha/kernels/mha_weighted_merge.py',
               preweighted_merge='operators/mha/kernels/mha_preweighted_merge.py',
               prefetch_merge='operators/mha/kernels/mha_prefetch_merge.py',
               simd_merge='operators/mha/kernels/mha_simd_merge.py')
    return {k:hashlib.sha256((ROOT/v).read_bytes()).hexdigest() for k,v in paths.items()}


def load_case(path,q,kv):
    report=json.loads(path.read_text())
    repeats=validate(report,kv,q)
    if report.get('benchmark_sha256')!=hashlib.sha256((ROOT/'benchmarks/mha/bench_static_host.py').read_bytes()).hexdigest():
        raise ValueError('benchmark source changed')
    if report.get('source_sha256')!=current_sources():
        raise ValueError('operator/source files changed')
    if report.get('model_sha256')!=hashlib.sha256((ROOT/'operators/mha/split_cost_model.json').read_bytes()).hexdigest():
        raise ValueError('model file changed')
    med={label:statistics.median([v for r in repeats for v in r['samples_us'][label]]) for label in LABELS}
    ratios={label:[statistics.median(r['samples_us'][label])/statistics.median(r['samples_us']['static']) for r in repeats]
            for label in ('raw_ref','ascendc')}
    result=dict(Q=q,KV=kv,fused_us=med['static'],original_us=med['raw_ref'],AscendC_us=med['ascendc'],
                original_over_fused=med['raw_ref']/med['static'],AscendC_over_fused=med['ascendc']/med['static'],
                AscendC_over_original=med['ascendc']/med['raw_ref'],plan=report['summary'][0]['plan'],
                ratios={k:dict(median=statistics.median(v),min=min(v),max=max(v),wins=sum(x>1 for x in v)) for k,v in ratios.items()},
                repeat_medians_us=[{label:statistics.median(r['samples_us'][label]) for label in LABELS} for r in repeats])
    return result,dict(device=report['device'],versions=report['versions'])


def summarize(directory):
    rows,errors,identity=[],[],None
    for kv in KVS:
        for q in QS:
            key=f'q{q}_kv{kv}'
            path=directory/(key+'.json')
            if not path.exists(): continue
            try:
                row,device=load_case(path,q,kv)
                if identity is None: identity=device
                if device!=identity: raise ValueError('mixed device/software versions')
                rows.append(row)
            except (ValueError,KeyError,TypeError,IndexError) as exc:
                errors.append(f'{key}: {exc}')
                guard=directory/('guard_'+key+'.jsonl')
                if guard.exists() and guard.read_text().splitlines(): errors.append('guard_last: '+guard.read_text().splitlines()[-1])
    passed=len(rows)==80 and not errors
    lines=[f"status={'passed' if passed else 'incomplete_or_error'} completed={len(rows)}/80",
           'B1H1 BF16 D128/page128 noncausal; same-field stream/rawsimd plan; fused SIMD div+cast vs original raw SIMD vs native FIA.',
           '1 build/TL variant/shape; 6 repeats x 9 balanced rotated triples; seeds7/19 and post-repeat accuracy.',
           '100 initial warmups/control/repeat + Event priming; 10 immediate warmups/sample; full-call warm-cache Event.',
           'Ratios use all 54 samples/control pooled medians; all raw samples remain in case JSON; no exclusions.']
    if rows:
        for metric in ('original_over_fused','AscendC_over_fused','AscendC_over_original'):
            lines.append(f"{metric}: geomean={math.exp(statistics.mean(math.log(r[metric]) for r in rows)):.6f} faster={sum(r[metric]>1 for r in rows)}/{len(rows)} min={min(r[metric] for r in rows):.6f} max={max(r[metric] for r in rows):.6f}")
        lines.append(f"all_6_repeats_faster_than_original={sum(r['ratios']['raw_ref']['min']>1 for r in rows)} all_6_slower_than_original={sum(r['ratios']['raw_ref']['max']<1 for r in rows)}")
    lookup={(r['Q'],r['KV']):r for r in rows}
    for metric in ('original_over_fused','AscendC_over_fused'):
        lines.extend(['MATRIX '+metric,'Q/KV '+' '.join(map(str,KVS))])
        for q in QS:
            lines.append(str(q)+' '+' '.join(f"{lookup[q,kv][metric]:.4f}" if (q,kv) in lookup else '-' for kv in KVS))
    lines.append('Q KV fused_us original_us AscendC_us original/fused AC/fused original_repeat_min original_repeat_max original_wins/6 AC_wins/6')
    for r in rows:
        ref=r['ratios']['raw_ref']; ac=r['ratios']['ascendc']
        lines.append(f"{r['Q']} {r['KV']} {r['fused_us']:.3f} {r['original_us']:.3f} {r['AscendC_us']:.3f} {r['original_over_fused']:.6f} {r['AscendC_over_fused']:.6f} {ref['min']:.6f} {ref['max']:.6f} {ref['wins']}/6 {ac['wins']}/6")
    regressions=[r for r in rows if r['fused_us']>r['original_us']*1.02]
    lines.append('FUSED_SLOWER_THAN_ORIGINAL_GT2PCT '+json.dumps([dict(Q=r['Q'],KV=r['KV'],slowdown_pct=100*(r['fused_us']/r['original_us']-1),repeat_ratio=r['ratios']['raw_ref']) for r in regressions],sort_keys=True))
    lines.extend(errors)
    output=directory.parent/'mha_fused_output_grid_paste.txt'
    output.write_text('\n'.join(lines)+'\n',encoding='utf8')
    (directory/'summary.json').write_text(json.dumps(dict(status='passed' if passed else 'incomplete_or_error',completed=len(rows),identity=identity,rows=rows,errors=errors),indent=2)+'\n')
    keys=('Q','KV','fused_us','original_us','AscendC_us','original_over_fused','AscendC_over_fused','AscendC_over_original')
    with (directory/'summary.csv').open('w',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(keys)
        writer.writerows([r[k] for k in keys] for r in rows)
    print(output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--check-case',nargs=2,type=int,metavar=('Q','KV'))
    args=parser.parse_args()
    if args.check_case:
        q,kv=args.check_case
        try:
            load_case(args.directory/f'q{q}_kv{kv}.json',q,kv)
        except (OSError,ValueError,KeyError,TypeError,IndexError):
            raise SystemExit(1)
    else:
        summarize(args.directory)
