"""Retained warm-cache NPU PipeUtilization and trace for the current raw local."""
import argparse
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import traceback

from benchmarks.mha.bench_split_breakdown import compiler_temp
from benchmarks.mha.resource_guard import cpu_environment, progress, runtime_policy
from tools.mha.parse_local_profile import parse_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kv', type=int, choices=(4096, 8192), required=True)
    parser.add_argument('--sessions', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()
    if min(args.sessions, args.repeat, args.warmup) < 1:
        parser.error('positive sessions/repeat/warmup required')
    if args.list_only:
        print(json.dumps(dict(Q=32768, KV=args.kv, target='rawsimd local only', **vars(args)), default=str))
        return 0
    path = args.output.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', phase='setup', settings=vars(args), captures=[], accuracy=[],
                  contract='Only prepared-Q raw local is called during profiling; no cache flush, merge or native FIA.',
                  limits='Pipe busy ratios overlap and do not directly measure synchronization waits; retain raw trace.')

    def save():
        progress(path, report['phase'])
        path.write_text(json.dumps(report, indent=2, default=str) + '\n', encoding='utf8')
        lines = [f"status={report['status']} phase={report['phase']} KV={args.kv} captures={len(report['captures'])}/{args.sessions}",
                 report['contract'], report['limits']]
        for row in report['captures']:
            lines.append(f"SESSION {row['session']} status={row['status']} raw={row['raw_directory']} trace={row['trace_directory']}")
            if row.get('profile'):
                lines.append('COUNTERS ' + json.dumps(row['profile']['aggregate'], sort_keys=True))
        if report.get('error'):
            lines.append(report['error'])
        path.with_name(path.stem + '_paste.txt').write_text('\n'.join(lines) + '\n', encoding='utf8')

    save()
    try:
        report['compiler_tmp'] = compiler_temp(path)
        os.environ['TILELANG_DISABLE_CACHE'] = '1'
        threads = cpu_environment()
        report['resource_policy'] = runtime_policy()
        report['phase'] = 'import_torch'; save()
        import torch
        torch.set_num_threads(threads)
        report['phase'] = 'import_torch_npu'; save()
        import torch_npu
        import torch_npu.profiler
        report['phase'] = 'import_tilelang'; save()
        import tilelang
        import tilelang.profiler.msprof as msprof_source
        from tilelang.env import env
        from operators.mha.kernels import mha_static_host_candidate as candidate
        from operators.mha import static_schedule
        from benchmarks.mha.bench_mha_mfu import chunked_cpu_reference
        from tests.mha.test_mha_paged_accuracy import make_inputs
        env.disable_cache()
        report['phase'] = 'set_device'; save()
        torch.npu.set_device(args.device)
        device = torch.device('npu', args.device)
        props = torch.npu.get_device_properties(args.device)
        if getattr(props, 'cube_core_num', None) != 28:
            raise ValueError('Current plan is validated for 28 Cube cores')
        report.update(device=str(props), versions=dict(torch=torch.__version__, torch_npu=torch_npu.__version__, tilelang=tilelang.__version__))
        source = inspect.getsource(msprof_source)
        path.with_name(path.stem + '_installed_msprof.py').write_text(source, encoding='utf8')
        report['installed_msprof_sha256'] = hashlib.sha256(source.encode()).hexdigest()
        report['source_sha256'] = {name: hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
                                   for name, mod in [('candidate', candidate), ('plan', static_schedule)]}
        program = None
        for seed in (7, 19):
            report['phase'] = f'reference_seed{seed}'; save()
            q, k, v, table = make_inputs(('profile', 1, 32768, 1, 128, 128, args.kv//128), seed, torch.bfloat16, torch)
            lengths = torch.tensor([args.kv], dtype=torch.int32)
            expected = chunked_cpu_reference(q, k, v, table, lengths, 128, False, 128)
            inputs = [x.to(device) for x in (q, k, v, table, lengths)]
            if program is None:
                report['phase'] = 'compile_rawsimd_once'; save()
                program = candidate.build_kernel(32768, args.kv, k.shape[0], 'stream', None, 8, None, 'rawsimd')
                report['plan'] = program.plan.describe()
                for name, kernel in [('local', program.local_kernel), ('merge', program.merge_kernel)]:
                    code = kernel.get_kernel_source()
                    path.with_name(path.stem + '_' + name + '.cpp').write_text(code, encoding='utf8')
                    report['source_sha256'][name] = hashlib.sha256(code.encode()).hexdigest()
                    if name == 'local':
                        if not re.search(r'void\s+local_split_kernel\s*\(', code):
                            raise ValueError('Unexpected local symbol; update profiler kernel filter before capture')
            report['phase'] = f'accuracy_seed{seed}'; save()
            program.validate_inputs(*inputs)
            actual = program(*inputs).cpu()
            torch.testing.assert_close(actual, expected, rtol=.01, atol=.004)
            report['accuracy'].append(dict(seed=seed, max_abs_error=(actual.float()-expected.float()).abs().max().item()))
            if seed == 7:
                timed_inputs = inputs
        counts, tasks, _ = program.prepare_plan(device)
        prepared = timed_inputs[0].permute(0, 2, 1, 3).contiguous()
        call = functools.partial(program.run_local, prepared, timed_inputs[1], timed_inputs[2], timed_inputs[3], counts, tasks)
        report['q_preparation_aliases_input'] = prepared.data_ptr() == timed_inputs[0].data_ptr()
        for session in range(args.sessions):
            report['phase'] = f'warmup_session{session+1}'; save()
            for index in range(args.warmup):
                call()
                if (index+1) % 25 == 0:
                    print(f'[FGW profile] warmup {index+1}/{args.warmup}', flush=True)
            torch.npu.synchronize()
            trace_dir = path.parent / f'{path.stem}_trace{session+1}'
            trace_dir.mkdir(exist_ok=True)
            row = dict(session=session+1, trace_directory=str(trace_dir), raw_directory=None, status='running')
            report['captures'].append(row)
            report['phase'] = f'capture_session{session+1}'; save()
            old_work = os.environ.get('ASCEND_WORK_PATH')
            os.environ['ASCEND_WORK_PATH'] = str(trace_dir)
            try:
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=False, with_stack=False, profile_memory=False,
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(trace_dir)),
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                        l2_cache=False, data_simplification=False)) as prof:
                    with torch.profiler.record_function(f'FGW_RAW_LOCAL_KV{args.kv}_S{session+1}'):
                        for index in range(args.repeat):
                            call()
                            print(f'[FGW profile] session={session+1}/{args.sessions} local={index+1}/{args.repeat}', flush=True)
                        torch.npu.synchronize()
                    prof.step()
                raw = Path(prof.prof_if.prof_path).resolve()
                row['raw_directory'] = str(raw)
                if not raw.is_relative_to(trace_dir):
                    saved = trace_dir / 'retained_raw'
                    shutil.copytree(raw, saved)
                    raw = saved
                row['raw_directory'] = str(raw)
                report['phase'] = f'parse_session{session+1}'; save()
                row['profile'] = parse_profile(raw)
                row['trace_files'] = [str(p) for p in trace_dir.rglob('*.json')]
                if not row['trace_files']:
                    raise ValueError('No exported trace JSON found; raw profiling data retained')
                row['status'] = 'passed'
                save()
            finally:
                if old_work is None:
                    os.environ.pop('ASCEND_WORK_PATH', None)
                else:
                    os.environ['ASCEND_WORK_PATH'] = old_work
        report.update(status='passed', phase='complete', correctness='passed')
    except Exception:
        report.update(status='error', error=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
