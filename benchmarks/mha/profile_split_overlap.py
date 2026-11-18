"""Retained trace only, separate from Event timing and PipeUtilization."""
from pathlib import Path
import json
import statistics


def analyze_traces(paths, calls=5):
    observed = set()
    for path in paths:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload,dict):
            continue
        events = [e for e in payload.get('traceEvents',[]) if e.get('ph')=='X'
                  and e.get('cat')=='kernel' and isinstance(e.get('dur'),(int,float))
                  and isinstance(e.get('ts'),(int,float))]
        observed.update(str(e.get('name')) for e in events)
        local = sorted([e for e in events if 'local_split' in str(e.get('name',''))],key=lambda e:e['ts'])
        merges = sorted([e for e in events if 'merge_simd' in str(e.get('name',''))],key=lambda e:e['ts'])
        # Reject alternate trace layouts instead of mistaking missing events
        # for zero overlap. Expected one aggregated interval per kernel call.
        if len(local)!=2*calls or len(merges)!=calls:
            continue
        partial,direct = local[::2],local[1::2]
        if any(p['ts']+p['dur']>m['ts']+.01 for p,m in zip(partial,merges)):
            continue
        if any(d.get('tid')==m.get('tid') for d,m in zip(direct,merges)):
            continue
        overlap = [max(0,min(d['ts']+d['dur'],m['ts']+m['dur'])-max(d['ts'],m['ts'])) for d,m in zip(direct,merges)]
        duration = [m['dur'] for m in merges]
        if min(duration)<=0:
            continue
        return dict(status='recognized',trace=str(path),calls=calls,
                    overlap_us=overlap,merge_duration_us=duration,
                    median_overlap_us=statistics.median(overlap),
                    fraction_of_merge_interval_overlapped=sum(overlap)/sum(duration),
                    calls_with_overlap=sum(v>0 for v in overlap),
                    note='Chrome kernel intervals in microseconds; interval overlap is not a speedup measurement.')
    return dict(status='unrecognized_layout',observed_kernel_names=sorted(observed)[:16],
                note='Inspect retained trace manually; do not interpret missing/unrecognized events as zero overlap.')


def capture(call, directory):
    import torch
    import torch_npu.profiler
    directory = Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    for _ in range(100):
        call()
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
        record_shapes=False,with_stack=False,profile_memory=False,
        schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=1,repeat=1,skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory)),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False,data_simplification=False)) as prof:
        for i in range(5):
            with torch.profiler.record_function(f'FGW_SPLIT_OVERLAP_{i}'):
                call()
        torch.npu.synchronize()
        prof.step()
    traces = [str(p) for p in directory.rglob('*.json')]
    if not traces:
        raise ValueError('No exported trace JSON; inspect raw profiler directory')
    return dict(directory=str(directory),raw_directory=str(prof.prof_if.prof_path),trace_files=traces,
                overlap=analyze_traces(traces),
                note='Two streams do not prove overlap. Inspect local/merge kernel intervals in exported trace.')
