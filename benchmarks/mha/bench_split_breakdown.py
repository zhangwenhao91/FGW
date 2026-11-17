"""Clean v7 forced splits: full calls, local computation, merge and prepared-Q chain."""
import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback
import os

CASES = ((8192, 32768), (4096, 8192), (2048, 24576))
SPLITS = (1, 2, 4, 8)


def geometry(q, kv, split):
    tasks = math.ceil(q/128)
    work = tasks*split
    grid = min(work, 28)
    blocks = math.ceil((kv//128)/split)
    return dict(q_tiles=tasks, compute_work=work, programs=grid,
                min_tasks_per_program=work//grid, max_tasks_per_program=math.ceil(work/grid),
                kv_blocks_per_task=blocks, critical_kv_blocks=math.ceil(work/grid)*blocks,
                partial_bytes=0 if split == 1 else tasks*split*128*128*4,
                lse_bytes=0 if split == 1 else tasks*split*128*4)


def component_calls(program, inputs):
    q, k, v, table, lengths = inputs
    prepared = program.prepare_q(q)
    full = functools.partial(program, *inputs)
    if program.num_splits == 1:
        local = functools.partial(program.direct_kernel, prepared, k, v, table, lengths)
        return {"full": full, "local": local}
    local = functools.partial(program.partial_kernel, prepared, k, v, table, lengths)
    parts, lse = local()
    merge = functools.partial(program.merge_kernel, parts, lse)

    def chain():
        partial, score = local()
        return program.merge_kernel(partial, score)

    # merge is intentionally measured on a fixed valid partial result. chain
    # measures the actual local->merge dependency with fresh partial allocation.
    return {"full": full, "local": local, "merge": merge, "chain": chain}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="mha_split_breakdown.json")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    if args.pairs < 3 or args.warmup < 1:
        parser.error("pairs>=3 and warmup>=1 required")
    if args.list_only:
        print(json.dumps({"cases": CASES, "splits": SPLITS, "builds_per_split": 2,
                          "pairs_per_round": args.pairs,
                          "geometry": {f"q{q}_kv{kv}": {s: geometry(q, kv, s) for s in SPLITS} for q, kv in CASES}}))
        return 0
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", phase="setup", settings=vars(args), active=None,
                  cases=[], rounds=[], summary=[], policy=(
                      "Clean default v7 only, frozen selector unchanged. B1H1 BF16 D128/page128/noncausal. "
                      "Two independent builds per split per shape, TileLang cache disabled. "
                      "Full includes Q preparation, allocations, all kernels. local uses preprepared Q; "
                      "merge uses fixed valid Partial/LSE; chain runs local+merge with preprepared Q. "
                      "All component JIT calls retain their normal output allocations. "
                      "Warm each callable immediately and equally before each Event sample. "
                      "Isolated times are diagnostic and need not sum to full latency. "
                      "AscendC is the installed precompiled FIA v2."))

    def save():
        path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        lines = [f"status={report['status']} phase={report['phase']} active={report['active']} "
                 f"completed={len(report['summary'])}/3"]
        for case in report["cases"]:
            lines.append("CASE " + json.dumps(case))
        lines.extend("RESULT " + json.dumps(row) for row in report["summary"])
        if report.get("error"):
            lines.append(report["error"])
        path.with_name(path.stem+"_paste.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")

    save()
    try:
        os.environ["TILELANG_DISABLE_CACHE"] = "1"
        import torch
        import torch_npu
        import tilelang
        from tilelang.env import env
        from operators.mha.kernels import mha_paged_decode_ascend_v7 as impl
        from operators.mha.split_cost_model import MODEL_PATH
        from tests.mha.test_mha_paged_accuracy import make_inputs
        from benchmarks.mha.bench_mha_mfu import chunked_cpu_reference
        env.disable_cache()
        assert not env.is_cache_enabled()
        torch.npu.set_device(args.device)
        props = torch.npu.get_device_properties(args.device)
        assert getattr(props, "cube_core_num", None) == 28, "Requires 28 Cube cores"
        report.update(device=str(props), versions=dict(torch=torch.__version__, torch_npu=torch_npu.__version__, tilelang=tilelang.__version__),
                      source_sha256=hashlib.sha256(Path(impl.__file__).read_bytes()).hexdigest(),
                      model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest())
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)

        def measure(call):
            for _ in range(args.warmup):
                call()
            torch.npu.synchronize()
            begin.record()
            output = call()
            end.record()
            end.synchronize()
            us = begin.elapsed_time(end)*1000
            assert math.isfinite(us) and us > 0
            return us

        for qlen, kv in CASES:
            key = f"q{qlen}_kv{kv}"
            case = dict(key=key, geometry={str(s): geometry(qlen, kv, s) for s in SPLITS}, accuracy=[], builds=[])
            report["cases"].append(case)
            fixtures = {}
            for seed in (7, 19):
                report.update(phase="reference", active=f"{key}/seed{seed}"); save()
                q, k, v, table = make_inputs((key, 1, qlen, 1, 128, 128, kv//128), seed, torch.bfloat16, torch)
                lengths = torch.tensor([kv], dtype=torch.int32)
                expected = chunked_cpu_reference(q, k, v, table, lengths, 128, False, 128)
                inputs = [t.to("npu") for t in (q, k, v, table, lengths)]
                fixtures[seed] = (inputs, expected)
            pool_rows = fixtures[7][0][1].shape[0]
            rows = []
            for rd in range(2):
                programs = {}
                report.update(phase="compile_accuracy", active=f"{key}/round{rd+1}/auto"); save()
                impl.build_kernel.cache_clear()
                auto = impl.build_kernel(1, qlen, 1, 128, pool_rows, kv//128, 128, False,
                    "bfloat16", args.device, full_kv_cache=True)
                auto_split = auto.num_splits
                assert auto_split in SPLITS
                case["auto_split"] = auto_split
                case["auto_policy"] = auto.split_policy
                calls = {}
                for split in SPLITS:
                    report["active"] = f"{key}/round{rd+1}/S{split}"; save()
                    impl.build_kernel.cache_clear()
                    program = auto if split == auto_split else impl.build_kernel(
                        1, qlen, 1, 128, pool_rows, kv//128, 128, False, "bfloat16", args.device,
                        full_kv_cache=True, num_splits=split)
                    programs[split] = program
                    assert program.num_splits == split and program.compute_grid == case["geometry"][str(split)]["programs"]
                    kernels = {"direct": program.direct_kernel} if split == 1 else {"partial": program.partial_kernel, "merge": program.merge_kernel}
                    hashes = {}
                    for name, kernel in kernels.items():
                        source = kernel.get_kernel_source()
                        hashes[name] = hashlib.sha256(source.encode()).hexdigest()
                        path.with_name(f"{path.stem}_{key}_r{rd+1}_S{split}_{name}.cpp").write_text(source, encoding="utf-8")
                    case["builds"].append(dict(round=rd+1, split=split, hashes=hashes))
                    for seed, (inputs, expected) in fixtures.items():
                        components = component_calls(program, inputs)
                        for name in ("full", "merge", "chain") if split > 1 else ("full", "local"):
                            got = components[name]().cpu().reshape(expected.shape)
                            assert got.dtype == expected.dtype and torch.isfinite(got).all()
                            torch.testing.assert_close(got, expected, rtol=.01, atol=.004)
                            case["accuracy"].append(dict(round=rd+1, split=split, seed=seed, component=name,
                                max_abs_error=(got.float()-expected.float()).abs().max().item()))
                        if seed == 7:
                            calls.update({f"S{split}/{name}": call for name, call in components.items()})
                for seed, (inputs, expected) in fixtures.items():
                    native = functools.partial(torch_npu.npu_fused_infer_attention_score_v2,
                        inputs[0], inputs[1].view(-1,128,128), inputs[2].view(-1,128,128),
                        block_table=inputs[3], num_query_heads=1, num_key_value_heads=1, input_layout="BSND",
                        block_size=128, softmax_scale=128**-.5, actual_seq_qlen=[qlen], actual_seq_kvlen=[kv],
                        sparse_mode=0, inner_precise=0, return_softmax_lse=False)
                    got = native()[0].cpu()
                    assert got.shape == expected.shape and got.dtype == expected.dtype and torch.isfinite(got).all()
                    torch.testing.assert_close(got, expected, rtol=.01, atol=.004)
                    if seed == 7:
                        calls["ascendc"] = lambda native=native: native()[0]
                report.update(phase="timing", active=f"{key}/round{rd+1}"); save()
                labels = tuple(calls); samples = {label: [] for label in labels}
                for pair in range(args.pairs):
                    offset = (pair+rd)%len(labels)
                    order = labels[offset:]+labels[:offset]
                    if (pair+rd)%2:
                        order = tuple(reversed(order))
                    for label in order:
                        samples[label].append(measure(calls[label]))
                medians = {k: statistics.median(v) for k,v in samples.items()}
                row = dict(key=key, round=rd+1, median_us=medians, samples_us=samples)
                report["rounds"].append(row); rows.append(medians); save()
                print(key, "round", rd+1, medians, flush=True)
            results = []
            for split in SPLITS:
                prefix = f"S{split}/"
                times = {label[len(prefix):]: statistics.median(r[label] for r in rows) for label in rows[0] if label.startswith(prefix)}
                ratios = [r[f"S{auto_split}/full"]/r[prefix+"full"] for r in rows]
                native_ratios = [r["ascendc"]/r[prefix+"full"] for r in rows]
                results.append(dict(split=split, median_us=times, speedup_vs_auto=statistics.median(ratios),
                    speedup_vs_ascendc=statistics.median(native_ratios), round_speedup_vs_auto=ratios,
                    round_speedup_vs_ascendc=native_ratios,
                    isolated_local_plus_merge_us=times["local"]+times.get("merge",0),
                    build_count=sum(b["split"]==split for b in case["builds"])))
            report["summary"].append(dict(key=key, auto_split=auto_split,
                ascendc_us=statistics.median(r["ascendc"] for r in rows), results=results,
                round_median_us=rows,
                best_split=min(results,key=lambda r:r["median_us"]["full"])["split"]))
            save()
        report.update(status="passed", phase="complete", active=None, correctness="passed")
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
