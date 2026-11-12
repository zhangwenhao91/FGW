"""Same-input Event and upstream msprof_detail comparison of v7 and installed FIA v2."""
import argparse
import dataclasses
import datetime
import functools
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import traceback

LABELS = ("v7_full", "v7_direct", "ascendc_full")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=30, help="msprof invocations per session")
    parser.add_argument("--event-repeat", type=int, default=51)
    parser.add_argument("--cache-mb", type=int, default=256)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="mha_v7_ascendc_msprof.json")
    args = parser.parse_args()
    if min(args.rounds, args.warmup, args.repeat, args.event_repeat, args.cache_mb) < 1:
        parser.error("rounds/warmup/repeat/event-repeat/cache-mb must be positive")
    b, sq, h, d, length, page = 2, 4096, 8, 128, 2048, 128
    flops = 4*b*sq*h*d*length
    report = {"status": "running", "settings": vars(args),
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "shape": {"B": b, "Q": sq, "H": h, "D": d, "KV": length, "page_size": page},
        "dtype": "bfloat16", "causal": False, "seed": 7, "effective_flops": flops,
        "event_results": [], "msprof_results": [],
        "notes": ["Event: warm-cache elapsed median, interleaved one invocation per sample.",
            "msprof_detail: same upstream do_bench for all candidates, cache flush before invocation; dur_ns is profiler record aggregation per invocation, not Event elapsed median or CPU wall time.",
            "Pipe ratios use their own AIC/AIV cycle denominators and may overlap; do not add or interpret as an exclusive time breakdown.",
            "Upstream sums kernel profiles for multi-kernel calls: counters are aggregate, not per-kernel attribution. Parser source is preserved.",
            "v7_full includes Q conversion and direct kernel; v7_direct consumes preconverted Q; ascendc_full is native FIA v2 with same paged storage, BSND Q/O, no KV gather.",
            "Separate full/direct profiling sessions are not additive; their difference is not an exact Q-conversion cost.",
            "FIA inner_precise=0, sparse_mode=0, no mask/quantization/LSE; model and kernel unchanged."]}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(json.dumps(report, indent=2, default=str)+"\n", encoding="utf-8")
        compact = {key: value for key, value in report.items() if key != "event_results"}
        compact["event_results"] = [{k:v for k,v in row.items() if k != "samples_us"} for row in report["event_results"]]
        path.with_name(path.stem+"_summary.json").write_text(json.dumps(compact, indent=2, default=str)+"\n", encoding="utf-8")

    save()
    try:
        import torch
        import torch_npu
        import tilelang
        import tilelang.profiler.msprof as msprof_module
        from tilelang.profiler import do_bench
        from operators.mha.kernels import mha_paged_decode_ascend_v7 as v7
        from operators.mha.split_cost_model import MODEL_PATH
        from benchmarks.mha.bench_mha_mfu import chunked_cpu_reference
        torch.npu.set_device(args.device)
        device = torch.device("npu", args.device)
        props = torch.npu.get_device_properties(args.device)
        if getattr(props, "cube_core_num", None) != 28:
            raise ValueError("This comparison uses the validated 28-Cube configuration")
        report["device"] = str(props)
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "tilelang": tilelang.__version__}
        report["v7_sha256"] = hashlib.sha256(Path(v7.__file__).read_bytes()).hexdigest()
        report["cost_model_sha256"] = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        report["cost_model_weights"] = json.loads(MODEL_PATH.read_text(encoding="utf-8"))["weights"]
        source = inspect.getsource(msprof_module)
        source_path = path.with_name(path.stem+"_msprof_source.py")
        source_path.write_text(source, encoding="utf-8")
        report["msprof_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        report["msprof_source_file"] = source_path.name
        report["do_bench_signature"] = str(inspect.signature(do_bench))
        report["fia_schema"] = str(torch.ops.npu.npu_fused_infer_attention_score_v2._schemas)
        if not callable(getattr(torch_npu, "npu_fused_infer_attention_score_v2", None)):
            raise RuntimeError("Previously validated FIA v2 API is not exposed")
        width = length//page
        gen = torch.Generator().manual_seed(7)
        table = torch.randperm(b*width, generator=gen).reshape(b,width).int()
        lengths = torch.full((b,), length, dtype=torch.int32)
        q = torch.randn(b,sq,h,d,generator=gen).bfloat16()
        k = torch.randn(b*length,h,d,generator=gen).bfloat16()
        v = torch.randn(b*length,h,d,generator=gen).bfloat16()
        expected = chunked_cpu_reference(q,k,v,table,lengths,page,False,128)
        inputs = [x.to(device) for x in (q,k,v,table,lengths)]
        program = v7.build_kernel(b,sq,h,d,k.shape[0],width,page,False,"bfloat16",args.device,
            schedule_mode="manual", full_kv_cache=True)
        report["v7_configuration"] = {"split_policy": program.split_policy, "num_splits": program.num_splits,
            "query_tile_tasks": program.query_tile_tasks, "execution_mode": program.execution_mode}
        if program.execution_mode != "direct_single_split":
            raise ValueError("Expected split=1 direct path for B2/H8/Q4096/KV2048")
        prepared = program.prepare_q(inputs[0])
        paged_k = inputs[1].view(b*width,page,h*d)
        paged_v = inputs[2].view(b*width,page,h*d)
        fia = functools.partial(torch_npu.npu_fused_infer_attention_score_v2,
            inputs[0],paged_k,paged_v,block_table=inputs[3],num_query_heads=h,num_key_value_heads=h,
            input_layout="BSND",block_size=page,softmax_scale=d**-0.5,
            actual_seq_qlen=[sq]*b,actual_seq_kvlen=[length]*b,
            sparse_mode=0,inner_precise=0,return_softmax_lse=False)
        def direct():
            return program.direct_kernel(prepared,*inputs[1:]).view(b,sq,h,d)
        functions = {"v7_full": functools.partial(program,*inputs), "v7_direct": direct,
                     "ascendc_full": lambda: fia()[0]}
        report["accuracy"] = {}
        for label in LABELS:
            got = functions[label]().cpu()
            if got.dtype != torch.bfloat16 or got.shape != expected.shape:
                raise ValueError(f"Unexpected {label} output dtype/shape")
            torch.testing.assert_close(got,expected,rtol=1e-2,atol=4e-3)
            report["accuracy"][label] = {"status": "passed", "rtol": 1e-2, "atol": 4e-3,
                "max_abs_error": (got.float()-expected.float()).abs().max().item()}
        torch.npu.synchronize()
        save()
        # Warm-cache Event checkpoint; all compilation/correctness above is excluded.
        for rid in range(args.rounds):
            order = list(LABELS) if rid%2==0 else list(reversed(LABELS))
            for _ in range(args.warmup):
                for label in order:
                    functions[label]()
            torch.npu.synchronize()
            samples = {label: [] for label in LABELS}
            start,end = torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
            for sample in range(args.event_repeat):
                offset = sample%len(order)
                rotated = order[offset:]+order[:offset]
                if (sample//len(order))%2:
                    rotated.reverse()
                for label in rotated:
                    start.record()
                    output = functions[label]()
                    end.record()
                    end.synchronize()
                    us = start.elapsed_time(end)*1000
                    if not math.isfinite(us) or us<=0:
                        raise ValueError(f"Invalid Event time {us}")
                    samples[label].append(us)
            for label in LABELS:
                median = statistics.median(samples[label])
                report["event_results"].append({"label":label,"round":rid+1,"status":"passed",
                    "median_us":median,"samples_us":samples[label],
                    "mfu_percent":100*flops/(median*1e-6)/378e12})
                print(f"EVENT round={rid+1} {label}: {median:.3f} us",flush=True)
            save()
        # Rotate profiling session order; do not modify upstream parser or kernel.
        for rid in range(args.rounds):
            order = list(LABELS[rid%len(LABELS):]+LABELS[:rid%len(LABELS)])
            if (rid//len(LABELS))%2:
                order.reverse()
            for label in order:
                row = {"label":label,"round":rid+1,"status":"running"}
                report["msprof_results"].append(row)
                save()
                print(f"MSPROF round={rid+1} {label} starting",flush=True)
                profile = do_bench(functions[label],backend="msprof_detail",device=device,
                    _n_warmup=args.warmup,_n_repeat=args.repeat,cache_size=args.cache_mb)
                if not dataclasses.is_dataclass(profile):
                    raise TypeError(f"Expected KernelProfile dataclass, got {type(profile)}")
                if not (getattr(profile, "aic_total_cycles", 0) > 0 or getattr(profile, "aiv_total_cycles", 0) > 0):
                    raise RuntimeError("Profiler returned no usable AIC/AIV cycle counters")
                duration = float(profile.dur_ns)/1000
                if not math.isfinite(duration) or duration<=0:
                    raise ValueError(f"Invalid profiler duration {duration}")
                row.update(status="passed",profile=dataclasses.asdict(profile),
                    profiler_duration_us=duration)
                print(f"MSPROF round={rid+1} {label}: {profile}",flush=True)
                save()
        report["summary"] = {}
        for label in LABELS:
            events = [r["median_us"] for r in report["event_results"] if r["label"]==label]
            profs = [r["profiler_duration_us"] for r in report["msprof_results"] if r["label"]==label]
            report["summary"][label] = {"median_round_event_us":statistics.median(events),
                "round_event_us":events,"median_round_profiler_us":statistics.median(profs),
                "round_profiler_us":profs}
        report["speedups"] = {"event_ascendc_over_v7_full":report["summary"]["ascendc_full"]["median_round_event_us"]/report["summary"]["v7_full"]["median_round_event_us"],
            "msprof_ascendc_over_v7_full":report["summary"]["ascendc_full"]["median_round_profiler_us"]/report["summary"]["v7_full"]["median_round_profiler_us"]}
        report["status"] = "passed"
    except Exception:
        report.update(status="error",error=traceback.format_exc())
        if report["msprof_results"] and report["msprof_results"][-1]["status"]=="running":
            report["msprof_results"][-1]["status"]="error"
        traceback.print_exc()
    finally:
        save()
    print(f"{report['status']}: {path}",flush=True)
    return 0 if report["status"]=="passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
