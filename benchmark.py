"""Only the frozen 50fca6d cost plan versus installed native AscendC FIA."""
import argparse
import csv
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import traceback

ROOT = Path(__file__).resolve().parent
QS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
KVS = (512, 1024, 2048, 4096, 8192, 16384, 24576, 32768)
POLICY = dict(plan_policy='cost', builds=1, repeats=3, pairs=4, initial_warmup=100,
              immediate_warmup=10, seeds=[7,19], rtol=.01, atol=.004)
LABELS = ('TL', 'AscendC')


def verify_source():
    manifest = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    for name, item in manifest['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Frozen source changed: {name}')
    return dict(source_commit=manifest['source_commit'], files=manifest['files'],
                benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def order(pair, repeat):
    return LABELS if (pair+repeat) % 2 == 0 else LABELS[::-1]


def make_inputs(torch, qlen, kv, seed):
    # Match the prior 80-case fixture, including a repeated physical page.
    generator = torch.Generator().manual_seed(seed)
    pages = kv//128
    pool = pages+3
    table = torch.randperm(pool-1, generator=generator)[:pages].reshape(1,pages).int()
    if pages > 1:
        table[0,-1] = table[0,0]
    q = torch.randn(1,qlen,1,128,generator=generator).bfloat16()
    k = torch.randn(pool*128,1,128,generator=generator).bfloat16()
    v = torch.randn(pool*128,1,128,generator=generator).bfloat16()
    return q,k,v,table,torch.tensor([kv],dtype=torch.int32)


def reference(torch, q, k, v, table, lengths):
    rows = table[0].long()[:,None]*128 + torch.arange(128)
    kk, vv = k[rows.flatten(),0].float(), v[rows.flatten(),0].float()
    result = torch.empty_like(q)
    for start in range(0,q.shape[1],128):
        stop = min(start+128,q.shape[1])
        scores = (q[0,start:stop,0].float() @ kk.T) * 128**-.5
        result[0,start:stop,0] = (scores.softmax(-1) @ vv).bfloat16()
    return result


def run_case(args):
    from resource_guard import cpu_environment, progress, runtime_policy
    path = args.output.resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    report = dict(status='running', phase='setup', Q=args.q, KV=args.kv,
                  policy=POLICY, accuracy=[], repeats=[], builds=0)

    def save(phase):
        report['phase'] = phase
        progress(path,phase)
        path.write_text(json.dumps(report,indent=2)+'\n')

    save('setup')
    try:
        report['source'] = verify_source()
        scratch = path.parent/'compiler_tmp'
        scratch.mkdir(exist_ok=True)
        for name in ('TMPDIR','TMP','TEMP'):
            os.environ[name] = str(scratch)
        tempfile.tempdir = str(scratch)
        os.environ['TILELANG_DISABLE_CACHE'] = '1'
        threads = cpu_environment()
        save('import_torch')
        import torch
        torch.set_num_threads(threads)
        import torch_npu
        import tilelang
        from tilelang.env import env
        from operators.mha.kernels.mha_paged_decode import build_kernel
        env.disable_cache()
        torch.npu.set_device(0)
        props = torch.npu.get_device_properties(0)
        if getattr(props,'cube_core_num',None) != 28:
            raise ValueError('Frozen plan requires the validated 28-Cube device')
        report.update(device=str(props), versions=dict(torch=torch.__version__,
                      torch_npu=torch_npu.__version__,tilelang=tilelang.__version__),
                      resources=runtime_policy())
        program = None
        for seed in POLICY['seeds']:
            save(f'CPU_reference_seed{seed}')
            host = make_inputs(torch,args.q,args.kv,seed)
            expected = reference(torch,*host)
            inputs = [x.to('npu') for x in host]
            if program is None:
                save('compile_once')
                program = build_kernel(args.q,args.kv,host[1].shape[0],plan_policy='cost')
                report.update(builds=1,plan=program.plan.describe(),kernel_sha256={})
                for name,kernel in (('local',program.local_kernel),('merge',program.merge_kernel)):
                    if kernel is not None:
                        code = kernel.get_kernel_source()
                        path.with_name(path.stem+'_'+name+'.cpp').write_text(code)
                        report['kernel_sha256'][name] = hashlib.sha256(code.encode()).hexdigest()
            program.validate_inputs(*inputs)
            native = functools.partial(torch_npu.npu_fused_infer_attention_score_v2,
                inputs[0],inputs[1].view(-1,128,128),inputs[2].view(-1,128,128),
                block_table=inputs[3],num_query_heads=1,num_key_value_heads=1,
                input_layout='BSND',block_size=128,softmax_scale=128**-.5,
                actual_seq_qlen=[args.q],actual_seq_kvlen=[args.kv],
                sparse_mode=0,inner_precise=0,return_softmax_lse=False)
            calls = dict(TL=functools.partial(program,*inputs),AscendC=lambda native=native: native()[0])
            for label,call in calls.items():
                save(f'accuracy_seed{seed}_{label}')
                actual = call().cpu()
                torch.testing.assert_close(actual,expected,rtol=POLICY['rtol'],atol=POLICY['atol'])
                report['accuracy'].append(dict(seed=seed,label=label,
                    max_abs_error=(actual.float()-expected.float()).abs().max().item()))
            if seed == POLICY['seeds'][0]:
                timed = calls
        start,end = torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
        for repeat in range(POLICY['repeats']):
            save(f'warmup_repeat{repeat+1}')
            for sweep in range(POLICY['initial_warmup']):
                for label in order(sweep,repeat):
                    timed[label]()
            torch.npu.synchronize()
            start.record(); end.record(); end.synchronize()
            report['timer_primed'] = True
            save(f'timing_repeat{repeat+1}')
            samples = {label:[] for label in LABELS}
            for pair in range(POLICY['pairs']):
                for label in order(pair,repeat):
                    for _ in range(POLICY['immediate_warmup']):
                        timed[label]()
                    torch.npu.synchronize()
                    start.record()
                    output = timed[label]()
                    end.record(); end.synchronize()
                    us = start.elapsed_time(end)*1000
                    if not math.isfinite(us) or us <= 0:
                        raise ValueError('Invalid Event measurement')
                    samples[label].append(us)
                    print(f'[50fca6d] Q={args.q} KV={args.kv} repeat={repeat+1}/3 pair={pair+1}/4 {label}={us:.3f}us',flush=True)
            report['repeats'].append(dict(repeat=repeat+1,samples_us=samples))
            save(f'completed_repeat{repeat+1}')
        report.update(status='passed',correctness='passed')
        save('complete')
    except Exception:
        report.update(status='error',error=traceback.format_exc())
        traceback.print_exc()
        save('error')
    return 0 if report['status']=='passed' else 1


def extract(report,q,kv):
    if (report.get('status'),report.get('correctness'),report.get('Q'),report.get('KV')) != ('passed','passed',q,kv):
        raise ValueError(f"Incomplete/wrong shape: {report.get('phase')} {report.get('error','')}")
    if report.get('policy') != POLICY or report.get('builds') != 1 or report.get('timer_primed') is not True:
        raise ValueError('Wrong measurement policy, builds or Event state')
    if (report['plan']['Q'],report['plan']['KV']) != (q,kv):
        raise ValueError('Wrong plan shape')
    if {(r['seed'],r['label']) for r in report['accuracy']} != {(s,l) for s in (7,19) for l in LABELS}:
        raise ValueError('Missing accuracy control/seed')
    repeats = report['repeats']
    if len(repeats)!=3 or {r['repeat'] for r in repeats}!={1,2,3}:
        raise ValueError('Incomplete repeats')
    for row in repeats:
        samples = row['samples_us']
        if set(samples)!=set(LABELS) or any(len(v)!=4 or any(not math.isfinite(x) or x<=0 for x in v) for v in samples.values()):
            raise ValueError('Missing or invalid samples')
    med = {label:statistics.median([v for row in repeats for v in row['samples_us'][label]]) for label in LABELS}
    ratios = [statistics.median(row['samples_us']['AscendC'])/statistics.median(row['samples_us']['TL']) for row in repeats]
    return dict(Q=q,KV=kv,TL_us=med['TL'],AscendC_us=med['AscendC'],speedup=med['AscendC']/med['TL'],
                repeat_median=statistics.median(ratios),repeat_min=min(ratios),repeat_max=max(ratios),
                wins=sum(x>1 for x in ratios),plan=report['plan'],samples_us=repeats)


def summarize(directory):
    rows,errors,identity = [],[],None
    expected = verify_source()
    for kv in KVS:
        for q in QS:
            key = f'q{q}_kv{kv}'
            path = directory/(key+'.json')
            if not path.exists():
                continue
            try:
                report = json.loads(path.read_text())
                row = extract(report,q,kv)
                if report['source'] != expected:
                    raise ValueError('Frozen source or benchmark changed')
                current = [report['device'],report['versions']]
                if identity is None:
                    identity = current
                if current != identity:
                    raise ValueError('Mixed devices or versions')
                rows.append(row)
            except (ValueError,KeyError,TypeError,IndexError) as exc:
                errors.append(f'{key}: {exc}')
                guard = directory/('guard_'+key+'.jsonl')
                if guard.exists() and guard.read_text().splitlines():
                    errors.append('guard_last_record: '+guard.read_text().splitlines()[-1])
    passed = len(rows)==80 and not errors
    lines = [f"status={'passed' if passed else 'incomplete_or_error'} completed={len(rows)}/80",
             'TL=unchanged 50fca6d kernel + original cost planner; AscendC=native FIA v2.',
             'B1H1 BF16 D128 page128 noncausal; seeds7/19; 1 build; 3 repeats x 4 alternating pairs.',
             '100 initial warmups/control/repeat + Event priming; 10 immediate warmups/sample.',
             'Full-call warm-cache Event; speedup=AscendC/TL; median of all 12 samples/control; no sample excluded.']
    if rows:
        lines.append(f"geomean_speedup={math.exp(statistics.mean(math.log(r['speedup']) for r in rows)):.6f} faster={sum(r['speedup']>1 for r in rows)}/{len(rows)}")
    lookup = {(r['Q'],r['KV']):r for r in rows}
    lines.append('Q/KV '+' '.join(map(str,KVS)))
    for q in QS:
        lines.append(str(q)+' '+' '.join(f"{lookup[q,kv]['speedup']:.4f}" if (q,kv) in lookup else '-' for kv in KVS))
    lines.append('Q KV TL_us AscendC_us speedup repeat_median repeat_min repeat_max wins/3 partials merge_tiles')
    for r in rows:
        lines.append(f"{r['Q']} {r['KV']} {r['TL_us']:.3f} {r['AscendC_us']:.3f} {r['speedup']:.6f} {r['repeat_median']:.6f} "
                     f"{r['repeat_min']:.6f} {r['repeat_max']:.6f} {r['wins']}/3 {r['plan']['partial_count']} {r['plan']['merge_q_tiles']}")
    lines.extend(errors)
    output = directory.parent/'mha_50fca6d_grid_paste.txt'
    output.write_text('\n'.join(lines)+'\n')
    (directory/'summary.json').write_text(json.dumps(dict(status='passed' if passed else 'incomplete_or_error',rows=rows,errors=errors),indent=2)+'\n')
    keys = ('Q','KV','TL_us','AscendC_us','speedup','repeat_median','repeat_min','repeat_max','wins')
    with (directory/'summary.csv').open('w',newline='') as stream:
        writer = csv.writer(stream); writer.writerow(keys)
        writer.writerows([r[k] for k in keys] for r in rows)
    print(output,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    case = commands.add_parser('case')
    case.add_argument('--q',type=int,choices=QS,required=True)
    case.add_argument('--kv',type=int,choices=KVS,required=True)
    case.add_argument('--output',type=Path,required=True)
    summary = commands.add_parser('summarize')
    summary.add_argument('directory',type=Path)
    commands.add_parser('verify')
    args = parser.parse_args()
    if args.command=='case':
        return run_case(args)
    if args.command=='summarize':
        summarize(args.directory.resolve())
    else:
        verify_source()
        from operators.mha.static_schedule import choose_plan,validate_plan
        for q in QS:
            for kv in KVS:
                validate_plan(choose_plan(q,kv))
        print('PASS frozen source hashes and all 80 cost plans; no NPU import',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
