"""Repeat short-Q control timings for current versus persistent28 scheduling."""
import argparse
import datetime
import hashlib
import json
import math
import pathlib
import statistics
import traceback


MODES = ("normal", "persistent28")
CASES = [(f"b2h8_q{q}_kv2048", 2, q, 8, 2048, False, True) for q in (1,16,128)]
CASES += [(f"b1h1_q{q}_kv{kv}", 1, q, 1, kv, False, True)
          for kv in (2048,8192) for q in (1,16,128,129,512,1024)]
CASES += [("b1h1_q1_kv512",1,1,1,512,False,True),
          ("b1h1_q1_kv512_causal",1,1,1,512,True,True),
          ("anchor_b2h8_q4096",2,4096,8,2048,False,False),
          ("anchor_b1h1_q8192",1,8192,1,2048,False,False)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--pairs", type=int, default=61)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="mha_persistent28_short_paired.json")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    if args.list_only:
        print(json.dumps(CASES,indent=2))
        return 0
    if args.rounds < 2 or args.pairs < 3 or args.warmup < 1:
        parser.error("rounds>=2, pairs>=3, warmup>=1 required")
    report = {"status": "running", "settings": vars(args), "results": [], "cases": [],
              "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "policy": "Same process/input/current stream; both variants compiled and warmed before timing; balanced AB/BA order per sample pair; warm cache; no automatic default change.",
              "timing": "Full-call NPU Event microseconds including stream-visible launch gaps and allocation effects; no baseline overhead subtraction."}
    path = pathlib.Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        compact = {key:value for key,value in report.items() if key!="results"}
        compact["results"] = [{key:value for key,value in row.items() if key!="samples_us"} for row in report["results"]]
        path.with_name(path.stem+"_summary.json").write_text(json.dumps(compact,indent=2)+"\n",encoding="utf-8")

    save()
    try:
        import torch
        import torch_npu
        import tilelang
        import operators.mha.kernels.mha_paged_decode_ascend_v7 as implementation
        from benchmarks.mha.bench_mha_mfu import chunked_cpu_reference
        from operators.mha.split_cost_model import MODEL_PATH
        torch.npu.set_device(args.device)
        device = torch.device("npu", args.device)
        report["device"] = str(torch.npu.get_device_properties(args.device))
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "tilelang": tilelang.__version__}
        report["source_sha256"] = hashlib.sha256(pathlib.Path(implementation.__file__).read_bytes()).hexdigest()
        report["cost_model_sha256"] = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        bundles = []
        # Finish all compilation and correctness work before any measured round.
        for name, b, sq, h, length, causal, control in CASES:
            report["active_case"] = name
            print(f"Preparing {name}: B={b} Q={sq} H={h} KV={length} causal={causal}", flush=True)
            save()
            gen = torch.Generator().manual_seed(7)
            q = torch.randn(b, sq, h, 128, generator=gen).bfloat16()
            k = torch.randn(b * length, h, 128, generator=gen).bfloat16()
            v = torch.randn(b * length, h, 128, generator=gen).bfloat16()
            width = length // 128
            table = torch.randperm(b * width, generator=gen).reshape(b, width).int()
            lengths = torch.full((b,), length, dtype=torch.int32)
            inputs = [x.to(device) for x in (q, k, v, table, lengths)]
            programs = {mode: implementation.build_kernel(b, sq, h, 128, k.shape[0], width,
                128, causal, "bfloat16", args.device, pipeline_stages=2, buffer_versions=2,
                q_resident=True, kv_versions=None, schedule_mode="manual",full_kv_cache=True,
                persistent_programs=28 if mode=="persistent28" else None)
                for mode in MODES}
            geometry = {mode: {"num_splits": program.num_splits,
                "compute_grid": program.compute_grid, "compute_work": program.compute_work,
                "split_policy": program.split_policy, "execution_mode": program.execution_mode}
                for mode, program in programs.items()}
            if programs["normal"].num_splits != programs["persistent28"].num_splits:
                raise ValueError(f"Split changed for {name}")
            if control and geometry["normal"]["compute_grid"] != geometry["persistent28"]["compute_grid"]:
                raise ValueError(f"Expected same-grid control for {name}: {geometry}")
            hashes = {}
            for mode, program in programs.items():
                kernels = ({"direct": program.direct_kernel} if program.execution_mode=="direct_single_split"
                           else {"partial": program.partial_kernel,"merge":program.merge_kernel})
                hashes[mode] = {label: hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest()
                                for label,kernel in kernels.items()}
            report["cases"].append({"case":name,"B":b,"Q":sq,"H":h,"D":128,"KV":length,
                "causal":causal,"dtype":"bfloat16","page_size":128,"seed":7,
                "is_control":control,"geometry":geometry,"source_sha256":hashes,
                "generated_sources_identical":hashes["normal"]==hashes["persistent28"]})
            expected = chunked_cpu_reference(q,k,v,table,lengths,128,causal,128)
            for mode, program in programs.items():
                report["active_variant"] = mode
                save()
                torch.testing.assert_close(program(*inputs).cpu(), expected, rtol=1e-2, atol=4e-3)
            bundles.append((name, inputs, programs))
        report.pop("active_variant", None)
        report["correctness"] = "passed"
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)

        def measure(program, inputs):
            start.record()
            output = program(*inputs)
            end.record()
            end.synchronize()
            us = start.elapsed_time(end)*1000
            if not math.isfinite(us) or us <= 0:
                raise ValueError(f"Invalid Event time {us}")
            return us

        for round_id in range(args.rounds):
            order_cases = bundles if round_id % 2 == 0 else list(reversed(bundles))
            for name, inputs, programs in order_cases:
                report["active_case"] = name
                for _ in range(args.warmup):
                    for mode in MODES:
                        programs[mode](*inputs)
                torch.npu.synchronize()
                samples = {mode: [] for mode in MODES}
                row = {"case": name, "round": round_id+1, "status": "running"}
                report["results"].append(row)
                save()
                for pair in range(args.pairs):
                    order = MODES if (pair+round_id)%2==0 else tuple(reversed(MODES))
                    for label in order:
                        samples[label].append(measure(programs[label], inputs))
                medians = {mode: statistics.median(values) for mode, values in samples.items()}
                speedups = {mode: medians["normal"] / medians[mode] for mode in MODES}
                row.update(status="passed", samples_us=samples, median_us=medians,
                           median_pair_speedup=statistics.median(a/b for a,b in zip(samples["normal"],samples["persistent28"])),
                           min_us={mode:min(values) for mode,values in samples.items()},
                           p90_us={mode:sorted(values)[int(0.9*(len(values)-1))] for mode,values in samples.items()},
                           speedup_vs_normal=speedups)
                save()
                print(f"round={round_id+1} {name}: medians_us={medians} speedup_vs_normal={speedups}", flush=True)
        report["summary"] = []
        for name, *_ in CASES:
            rows = [r for r in report["results"] if r["case"] == name]
            for mode in MODES:
                report["summary"].append({"case": name, "variant": mode,
                    "median_round_us": statistics.median(r["median_us"][mode] for r in rows),
                    "median_round_speedup_vs_normal": statistics.median(r["speedup_vs_normal"][mode] for r in rows),
                    "min_round_speedup_vs_normal": min(r["speedup_vs_normal"][mode] for r in rows),
                    "max_round_speedup_vs_normal": max(r["speedup_vs_normal"][mode] for r in rows),
                    "faster_than_normal_rounds": sum(r["speedup_vs_normal"][mode] > 1 for r in rows),
                    "rounds": len(rows)})
        report.pop("active_case", None)
        report["status"] = "passed"
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()
    print(f"{report['status'].upper()}: {path}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
