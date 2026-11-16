"""NSA-style pipeline candidate versus frozen manual v7 and native AscendC FIA."""
import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback

CASES = [(q, kv) for q, kv in ((16, 512), (129, 512), (16, 8192), (129, 8192),
                              (512, 8192), (2048, 8192), (4096, 8192), (8192, 8192))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--pairs", type=int, default=51)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", default="mha_nsa_pipeline.json")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.rounds < 2 or args.pairs < 3 or args.warmup < 1:
        parser.error("rounds>=2, pairs>=3, warmup>=1 required")
    path = Path(args.output)
    report = {"status": "running", "phase": "compile_accuracy", "settings": vars(args),
              "cases": [], "results": [], "summary": [],
              "policy": "B1H1 BF16 D128 page128 noncausal; manual default chooses split, candidate uses identical explicit split; same compute body, tile sizes, inputs and persistent cap28; no default promotion",
              "timing": "Interleaved rotating/reversing warm-cache Event full calls, including Q layout/allocation/all kernels. Compute components use preprepared Q, measured in a separate session; component times are not additive or subtracted from full calls.",
              "reference": "https://github.com/yyttt6/TileOPs/blob/2ffc69b29d1e456f28ade01d50fc3a6f715d1392/src/tileops/_kernels/ascend/attention/nsa_compressed_varlen.py"}

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        lines = [f"status={report['status']} phase={report['phase']} correctness={report.get('correctness')} active={report.get('active')}"]
        if report.get("error"):
            lines.append(report["error"])
        for info in report["cases"]:
            lines.append("CASE " + json.dumps(info))
        for item in report["summary"]:
            lines.append("RESULT " + json.dumps(item))
        path.with_name(path.stem + "_paste.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    save()
    try:
        import torch
        import torch_npu
        import tilelang
        from operators.mha.kernels import mha_paged_decode_ascend_v7 as impl
        from benchmarks.mha.bench_mha_mfu import chunked_cpu_reference
        from tests.mha.test_mha_paged_accuracy import make_inputs
        torch.npu.set_device(args.device)
        report["device"] = str(torch.npu.get_device_properties(args.device))
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "tilelang": tilelang.__version__}
        report["source_sha256"] = hashlib.sha256(Path(impl.__file__).read_bytes()).hexdigest()
        cases = [(129, 512), (4096, 8192)] if args.smoke else CASES
        bundles = []
        for sq, kv in cases:
            key = f"q{sq}_kv{kv}"
            info = {"key": key, "Q": sq, "KV": kv, "correctness": {}}
            report["cases"].append(info)
            seed7_inputs = make_inputs((key, 1, sq, 1, 128, 128, kv//128), 7, torch.bfloat16, torch)
            pool_rows = seed7_inputs[1].shape[0]
            report["active"] = key + "/manual_compile"
            save()
            baseline = impl.build_kernel(1, sq, 1, 128, pool_rows, kv//128, 128, False,
                                         "bfloat16", args.device, full_kv_cache=True)
            info.update(split=baseline.num_splits, execution_mode=baseline.execution_mode,
                        split_policy=baseline.split_policy, compute_grid=baseline.compute_grid)
            report["active"] = key + "/nsa_compile"
            save()
            candidate = impl.build_kernel(1, sq, 1, 128, pool_rows, kv//128, 128, False,
                                          "bfloat16", args.device, schedule_mode="nsa",
                                          num_splits=baseline.num_splits, split_policy="legacy", full_kv_cache=True)
            assert candidate.num_splits == baseline.num_splits
            assert candidate.compute_grid == baseline.compute_grid
            for label, program in (("manual", baseline), ("nsa", candidate)):
                kernel = program.direct_kernel if program.num_splits == 1 else program.partial_kernel
                source = kernel.get_kernel_source()
                path.with_name(path.stem + f"_{key}_{label}.cpp").write_text(source, encoding="utf-8")
                info[label + "_generated_sha256"] = hashlib.sha256(source.encode()).hexdigest()
                info[label + "_mmad_calls"] = source.count("asc_mmad(")
                assert info[label + "_mmad_calls"] >= 2
            for seed in (7, 19):
                report["active"] = f"{key}/accuracy_seed{seed}"
                save()
                q, k, v, table = seed7_inputs if seed == 7 else make_inputs((key, 1, sq, 1, 128, 128, kv//128), seed, torch.bfloat16, torch)
                assert k.shape[0] == pool_rows
                lengths = torch.tensor([kv], dtype=torch.int32)
                expected = chunked_cpu_reference(q, k, v, table, lengths, 128, False, 128)
                inputs = [t.to("npu") for t in (q, k, v, table, lengths)]
                native = functools.partial(torch_npu.npu_fused_infer_attention_score_v2,
                    inputs[0], inputs[1].view(-1, 128, 128), inputs[2].view(-1, 128, 128),
                    block_table=inputs[3], num_query_heads=1, num_key_value_heads=1,
                    input_layout="BSND", block_size=128, softmax_scale=128**-0.5,
                    actual_seq_qlen=[sq], actual_seq_kvlen=[kv], sparse_mode=0,
                    inner_precise=0, return_softmax_lse=False)
                calls = {"manual": functools.partial(baseline, *inputs),
                         "nsa": functools.partial(candidate, *inputs), "ascendc": lambda native=native: native()[0]}
                for label, call in calls.items():
                    got = call().cpu()
                    assert got.shape == q.shape and got.dtype == q.dtype and torch.isfinite(got).all()
                    torch.testing.assert_close(got, expected, rtol=1e-2, atol=4e-3)
                    info["correctness"][f"{label}/seed{seed}"] = {"status": "passed", "max_abs_error": (got.float()-expected.float()).abs().max().item()}
                if seed == 7:
                    timed_calls = calls
                    components = {}
                    for label, program in (("manual", baseline), ("nsa", candidate)):
                        prepared = program.prepare_q(inputs[0])
                        kernel = program.direct_kernel if program.num_splits == 1 else program.partial_kernel
                        components[label] = functools.partial(kernel, prepared, *inputs[1:])
            bundles.append((key, timed_calls, components))
        report["correctness"] = "passed"
        report["phase"] = "timing"
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)

        def measure(call):
            start.record()
            output = call()
            end.record()
            end.synchronize()
            us = start.elapsed_time(end)*1000
            assert math.isfinite(us) and us > 0
            return us

        for scope in ("full", "compute"):
            for rd in range(args.rounds):
                for key, full, compute in (bundles if rd % 2 == 0 else list(reversed(bundles))):
                    calls = full if scope == "full" else compute
                    report["active"] = f"{key}/{scope}/round{rd+1}"
                    save()
                    labels = tuple(calls)
                    for _ in range(args.warmup):
                        for call in calls.values():
                            call()
                    torch.npu.synchronize()
                    samples = {label: [] for label in labels}
                    for pair in range(args.pairs):
                        offset = (pair + rd) % len(labels)
                        order = labels[offset:] + labels[:offset]
                        if (pair + rd) % 2:
                            order = tuple(reversed(order))
                        for label in order:
                            samples[label].append(measure(calls[label]))
                    medians = {label: statistics.median(v) for label, v in samples.items()}
                    report["results"].append({"key": key, "scope": scope, "round": rd+1,
                                              "median_us": medians, "samples_us": samples})
                    save()
                    print(report["active"], medians, flush=True)
            for key, *_ in bundles:
                rows = [r["median_us"] for r in report["results"] if r["key"] == key and r["scope"] == scope]
                ratio = [r["manual"]/r["nsa"] for r in rows]
                item = {"key": key, "scope": scope, "rounds": len(rows),
                        "median_us": {label: statistics.median(r[label] for r in rows) for label in rows[0]},
                        "speedup_nsa_over_manual": statistics.median(ratio),
                        "min_round_speedup": min(ratio), "max_round_speedup": max(ratio),
                        "nsa_faster_rounds": sum(v > 1 for v in ratio)}
                if scope == "full":
                    item["speedup_nsa_over_ascendc"] = statistics.median(r["ascendc"]/r["nsa"] for r in rows)
                    item["speedup_manual_over_ascendc"] = statistics.median(r["ascendc"]/r["manual"] for r in rows)
                report["summary"].append(item)
        report.update(status="passed", phase="complete", active=None)
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
