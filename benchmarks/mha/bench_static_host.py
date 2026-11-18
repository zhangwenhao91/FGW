"""Static host plan vs default v7, previously best uniform split, and installed FIA."""

import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback
import os
from benchmarks.mha.bench_split_breakdown import compiler_temp
from operators.mha.static_schedule import make_plan, choose_plan, estimate_cost, COST_MODEL, make_tail_plan, choose_stream_plan
from benchmarks.mha.resource_guard import cpu_environment, progress, runtime_policy

CASES = ((8192, 32768, 2), (4096, 8192, 1), (2048, 24576, 4), (3584, 32768, 1), (7168, 32768, 1))
SMOKE = ((1, 512, 1), (17, 512, 1), (129, 512, 1), (3712, 512, 1), (3584, 512, 1))


def paired_order(labels, pair, compile_round=0, measurement_round=0):
    offset = (pair + compile_round + measurement_round) % len(labels)
    return labels[offset:] + labels[:offset]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="Accuracy only, including tails/mixed/direct/no-merge"
    )
    parser.add_argument("--raw-breakdown", action="store_true", help="Only raw SIMD full/local/merge/prepared-chain and native FIA; omit variant builds")
    parser.add_argument("--row-softmax", action="store_true", help="Compare two interleaved register-resident softmax rows with original raw SIMD and native FIA")
    parser.add_argument("--only-static-ascendc", action="store_true", help="Compile/time only static candidate and AscendC; omit v7 and diagnostic timings")
    parser.add_argument("--grid", action="store_true", help="Original clean-v7 80-shape grid")
    parser.add_argument("--shape", nargs=2, type=int, metavar=("Q", "KV"))
    parser.add_argument("--plan-policy", choices=("balanced", "cost", "tail", "stream"), default="balanced")
    parser.add_argument("--tail-pieces", type=int, choices=(2, 4, 7))
    parser.add_argument("--programs", type=int, choices=range(1, 29))
    parser.add_argument("--merge-policy", choices=("fixed", "weighted", "grouped", "preweighted", "prefetch", "simd", "rawsimd", "rawrecip"), default="fixed")
    parser.add_argument("--merge-programs", type=int, choices=(28, 56), default=56)
    parser.add_argument("--merge-rows", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument("--rounds", type=int, default=2, help="Independent compilation rounds per shape")
    parser.add_argument("--timing-rounds", type=int, default=1, help="Measurement repeats reusing each compiled build")
    parser.add_argument("--initial-warmup", type=int, default=0, help="Warmups per control before each measurement repeat")
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", default="mha_static_host.json")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.pairs < 3 or args.warmup < 1:
        parser.error("rounds>=1, pairs>=3, warmup>=1 required")
    if args.timing_rounds < 1 or args.initial_warmup < 0:
        parser.error("timing-rounds>=1 and initial-warmup>=0 required")
    if args.raw_breakdown and (args.merge_policy != "rawsimd" or not args.only_static_ascendc):
        parser.error("--raw-breakdown requires --merge-policy rawsimd --only-static-ascendc")
    if args.row_softmax and (args.merge_policy != "rawsimd" or not args.only_static_ascendc or args.raw_breakdown):
        parser.error("--row-softmax requires rawsimd, --only-static-ascendc and no --raw-breakdown")
    if args.smoke and (args.grid or args.shape):
        parser.error("--smoke cannot be combined with --grid/--shape")
    from benchmarks.mha.summarize_clean_grid import QS, KVS
    cases = SMOKE if args.smoke else CASES
    if args.grid:
        cases = tuple((q, kv, None) for kv in KVS for q in QS)
    if args.shape:
        q, kv = args.shape
        if q not in QS or kv not in KVS:
            parser.error("--shape must belong to the original 80-shape grid")
        cases = ((q, kv, None),)
    if args.programs is not None and args.plan_policy in ("cost", "stream"):
        parser.error("--programs requires --plan-policy balanced")
    planner = choose_plan if args.plan_policy == "cost" else functools.partial(make_plan, cores=args.programs or 28)
    if args.tail_pieces is not None and args.plan_policy != "tail":
        parser.error("--tail-pieces requires --plan-policy tail")
    if args.plan_policy == "tail":
        planner = functools.partial(make_tail_plan, cores=args.programs or 28, pieces=args.tail_pieces)
    if args.plan_policy == "stream":
        planner = choose_stream_plan
    if args.list_only:
        print(json.dumps([dict(**planner(q, kv).describe(), uniform_split=s, estimate=estimate_cost(planner(q, kv))) for q, kv, s in cases]))
        return 0
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        status="running",
        phase="setup",
        active=None,
        settings=vars(args),
        plan_cost_model=COST_MODEL,
        cases=[],
        rounds=[],
        measurement_rounds=[],
        summary=[],
        scope="B1H1 BF16 D128/page128 full live KV, noncausal, static contiguous host plan, no device queue. Default code/model unchanged.",
        timing="Full calls include Q preparation, workspace/output allocations, all kernels. Host plan construction/H2D upload and compilation excluded. Immediate equal warmup per Event sample.",
        local_contract="static/local is the local JIT on prepared Q with normal allocations; static/merge updates only boundary rows in a fixed valid final-output fixture in place; static/chain runs local then merge on prepared Q. No-merge plans omit merge/chain timing.",
        prototype_cost="Split local JIT allocates Parts/LSE/final O. No-fragment local JIT allocates final O only; no task metadata upload or merge.",
    )

    last_stage = [None]
    if args.row_softmax:
        report['softmax_variant'] = 'register_rows2_v2'
    def save():
        stage = (report['phase'], report['active'])
        if stage != last_stage[0]:
            progress(path, f"{stage[0]} active={stage[1]}")
            last_stage[0] = stage
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
        lines = [
            f"status={report['status']} phase={report['phase']} active={report['active']} completed={len(report['summary'])}/{len(cases)}"
        ]
        for c in report["cases"]:
            lines.append(
                "CASE " + json.dumps({k: v for k, v in c.items() if k not in ("accuracy", "builds")})
            )
        lines.extend("RESULT " + json.dumps(r) for r in report["summary"])
        if report.get("error"):
            lines.append(report["error"])
        path.with_name(path.stem + "_paste.txt").write_text("\n".join(lines) + "\n", encoding="utf8")

    save()
    try:
        report["compiler_tmp"] = compiler_temp(path)
        os.environ["TILELANG_DISABLE_CACHE"] = "1"
        threads = cpu_environment()
        report['resource_policy'] = runtime_policy()
        report.update(phase='import_torch',active=None);save()
        import torch
        torch.set_num_threads(threads)
        report['cpu_threads'] = torch.get_num_threads()
        report.update(phase='import_torch_npu');save()
        import torch_npu
        report.update(phase='import_tilelang');save()
        import tilelang
        from tilelang.env import env
        from operators.mha.kernels import mha_static_host_candidate as candidate
        from operators.mha.kernels import mha_paged_decode_ascend_v7 as default
        from operators.mha.kernels import mha_weighted_merge as weighted_merge_source
        from operators.mha.kernels import mha_preweighted_merge as preweighted_merge_source
        from operators.mha.kernels import mha_prefetch_merge as prefetch_merge_source
        from operators.mha.kernels import mha_simd_merge as simd_merge_source
        from operators.mha import static_schedule
        from operators.mha.split_cost_model import MODEL_PATH
        from benchmarks.mha.bench_mha_mfu import chunked_cpu_reference
        from tests.mha.test_mha_paged_accuracy import make_inputs

        env.disable_cache()
        assert not env.is_cache_enabled()
        report.update(phase='npu_set_device');save()
        torch.npu.set_device(args.device)
        props = torch.npu.get_device_properties(args.device)
        assert getattr(props, "cube_core_num", None) == 28
        report.update(
            device=str(props),
            versions=dict(
                torch=torch.__version__, torch_npu=torch_npu.__version__, tilelang=tilelang.__version__
            ),
            source_sha256={
                name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                for name, m in [("candidate", candidate), ("default", default), ("plan", static_schedule), ("weighted_merge", weighted_merge_source), ("preweighted_merge", preweighted_merge_source), ("prefetch_merge", prefetch_merge_source), ("simd_merge", simd_merge_source)]
            },
            model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        )
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)

        def measure(call):
            for _ in range(args.warmup):
                call()
            torch.npu.synchronize()
            start.record()
            output = call()
            end.record()
            end.synchronize()
            us = start.elapsed_time(end) * 1000
            assert math.isfinite(us) and us > 0
            return us

        for qlen, kv, best in cases:
            key = f"q{qlen}_kv{kv}"
            plan = planner(qlen, kv)
            info = dict(key=key, plan=plan.describe(), estimate=estimate_cost(plan),
                        previous_balanced_plan=make_plan(qlen, kv).describe(),
                        uniform_split=best, accuracy=[], builds=[])
            report["cases"].append(info)
            path.with_name(f"{path.stem}_{key}_plan.json").write_text(
                json.dumps(
                    dict(programs=plan.programs, merges=plan.merges, summary=plan.describe()), indent=2
                )
            )
            fixtures = {}
            for seed in (7, 19):
                report.update(phase="reference", active=f"{key}/seed{seed}")
                save()
                q, k, v, table = make_inputs(
                    (key, 1, qlen, 1, 128, 128, kv // 128), seed, torch.bfloat16, torch
                )
                lengths = torch.tensor([kv], dtype=torch.int32)
                expected = chunked_cpu_reference(q, k, v, table, lengths, 128, False, 128)
                fixtures[seed] = ([x.to("npu") for x in (q, k, v, table, lengths)], expected)
            rows = []
            for rd in range(1 if args.smoke else args.rounds):
                report.update(phase="compile_accuracy", active=f"{key}/round{rd+1}")
                save()
                compiler_temp(path)
                candidate.build_kernel.cache_clear()
                default.build_kernel.cache_clear()
                pool = fixtures[7][0][1].shape[0]
                print(f"[FGW compile] {key} round={rd+1}/{args.rounds} static", flush=True)
                static = candidate.build_kernel(qlen, kv, pool, args.plan_policy, args.programs, args.merge_rows, args.tail_pieces, args.merge_policy, args.merge_programs, row_softmax=args.row_softmax)
                assert static.plan == plan
                simd_reference = None
                if args.merge_policy == "rawsimd" and not args.raw_breakdown and not args.row_softmax:
                    print(f"[FGW compile] {key} normalized SIMD reference", flush=True)
                    simd_reference = candidate.build_kernel(qlen, kv, pool, args.plan_policy,
                        args.programs, args.merge_rows, args.tail_pieces, "simd", args.merge_programs, False)
                    assert simd_reference.plan == static.plan
                    info["reference_builds"] = info.get("reference_builds", 0) + 1
                    info["partial_contract"] = "Raw FP32 A + max/sum planes; reference normalized FP32 Partial + LSE. Direct tiles normalized normally."
                if args.row_softmax:
                    from operators.mha.kernels.mha_simd_merge import replace_merge
                    print(f"[FGW compile] {key} original raw SIMD softmax reference", flush=True)
                    simd_reference = candidate.build_kernel(qlen, kv, pool, args.plan_policy,
                        args.programs, args.merge_rows, args.tail_pieces, "rawsimd", args.merge_programs)
                    assert simd_reference.plan == static.plan
                    # Both full calls use the very same compiled merge object.
                    if static.merge_kernel is not None:
                        static = replace_merge(static, simd_reference.merge_kernel)
                    info["reference_builds"] = info.get("reference_builds", 0) + 1
                    info["partial_contract"] = "Only local softmax differs: two interleaved register-resident rows vs original UB roundtrip. Same plan, raw A/max/sum and shared SIMD merge."
                if args.merge_policy == "rawrecip":
                    from operators.mha.kernels.mha_simd_merge import make_kernel, replace_merge
                    print(f"[FGW compile] {key} raw SIMD division reference (shared local)", flush=True)
                    if static.merge_kernel is None:
                        simd_reference = static
                        info["reference_builds"] = 0
                    else:
                        baseline_merge = tilelang.compile(make_kernel(qlen, max(1, plan.partial_count),
                            len(plan.merges), plan.max_parts, min(args.merge_rows, tilelang.next_power_of_2(qlen)),
                            raw=True, reciprocal=False), out_idx=[], target="ascend", execution_backend="auto", pass_configs={})
                        simd_reference = replace_merge(static, baseline_merge)
                        static = replace_merge(static, static.merge_kernel)
                        assert simd_reference.local_kernel is static.local_kernel
                        info["reference_builds"] = info.get("reference_builds", 0) + 1
                    info["partial_contract"] = "Both variants share raw A/max/sum local and workspace; reference divides each element, candidate broadcasts row reciprocal."
                if args.raw_breakdown:
                    info["partial_contract"] = "Raw A/max/sum SIMD division only; local/merge/prepared chain/full breakdown; no comparison variant build."
                info["merge_assignment"] = static.merge_assignment
                base = uniform = None
                if not args.only_static_ascendc:
                    print(f"[FGW compile] {key} round={rd+1}/{args.rounds} default v7", flush=True)
                    base = default.build_kernel(
                        1, qlen, 1, 128, pool, kv // 128, 128, False, "bfloat16", args.device, full_kv_cache=True
                    )
                    default.build_kernel.cache_clear()
                    compiler_temp(path)
                    uniform = (
                        base
                        if best is None or base.num_splits == best
                        else default.build_kernel(
                            1,
                            qlen,
                            1,
                            128,
                            pool,
                            kv // 128,
                            128,
                            False,
                            "bfloat16",
                            args.device,
                            full_kv_cache=True,
                            num_splits=best,
                        )
                    )
                    info["default_split"] = base.num_splits
                kernels = {"static_local": static.local_kernel}
                if static.merge_kernel is not None:
                    kernels["static_merge"] = static.merge_kernel
                for label, program in [("default", base), ("uniform", uniform)]:
                    if program is None:
                        continue
                    if program.num_splits == 1:
                        kernels[label + "_direct"] = program.direct_kernel
                    else:
                        kernels.update(
                            {
                                label + "_partial": program.partial_kernel,
                                label + "_merge": program.merge_kernel,
                            }
                        )
                if static.fixed_merge_kernel is not None:
                    kernels["fixed_merge"] = static.fixed_merge_kernel
                if simd_reference is not None:
                    reference_prefix = "raw_simd_ref" if args.merge_policy == "rawrecip" or args.row_softmax else "normalized_simd"
                    kernels[reference_prefix + "_local"] = simd_reference.local_kernel
                    if simd_reference.merge_kernel is not None:
                        kernels[reference_prefix + "_merge"] = simd_reference.merge_kernel
                hashes = {}
                for label, kernel in kernels.items():
                    source = kernel.get_kernel_source()
                    hashes[label] = hashlib.sha256(source.encode()).hexdigest()
                    path.with_name(f"{path.stem}_{key}_r{rd+1}_{label}.cpp").write_text(
                        source, encoding="utf8"
                    )
                info["builds"].append(dict(round=rd + 1, source_sha256=hashes))
                for seed, (inputs, expected) in fixtures.items():
                    report["active"] = f"{key}/round{rd+1}/seed{seed}"
                    save()
                    static.validate_inputs(*inputs)
                    counts, tasks, merges = static.prepare_plan(inputs[0].device)
                    prepared = inputs[0].permute(0, 2, 1, 3).contiguous()
                    if args.raw_breakdown:
                        info["q_preparation_aliases_input"] = prepared.data_ptr() == inputs[0].data_ptr()
                    local = functools.partial(
                        static.run_local, prepared, inputs[1], inputs[2], inputs[3], counts, tasks
                    )
                    parts, lse, output = local()
                    calls = {
                        "static": functools.partial(static, *inputs),
                    }
                    if simd_reference is not None:
                        simd_reference.validate_inputs(*inputs)
                        calls["raw_ref" if args.merge_policy == "rawrecip" or args.row_softmax else "simd_ref"] = functools.partial(simd_reference, *inputs)
                    if static.fixed_merge_kernel is not None:
                        calls["fixed_r8"] = functools.partial(static.fixed_launch, *inputs)
                    if base is not None:
                        calls["default"] = functools.partial(base, *inputs)
                    if uniform is not base:
                        calls["uniform_best"] = functools.partial(uniform, *inputs)
                    if static.merge_kernel is not None and (not args.only_static_ascendc or args.raw_breakdown):
                        calls["static/merge"] = functools.partial(static.finish, parts, lse, output)

                        def chain(local=local, static=static):
                            return static.finish(*local())

                        calls["static/chain"] = chain
                    native = functools.partial(
                        torch_npu.npu_fused_infer_attention_score_v2,
                        inputs[0],
                        inputs[1].view(-1, 128, 128),
                        inputs[2].view(-1, 128, 128),
                        block_table=inputs[3],
                        num_query_heads=1,
                        num_key_value_heads=1,
                        input_layout="BSND",
                        block_size=128,
                        softmax_scale=128**-0.5,
                        actual_seq_qlen=[qlen],
                        actual_seq_kvlen=[kv],
                        sparse_mode=0,
                        inner_precise=0,
                        return_softmax_lse=False,
                    )
                    calls["ascendc"] = lambda native=native: native()[0]
                    for label, call in calls.items():
                        print(f"[FGW accuracy] {key} round={rd+1} seed={seed} {label}", flush=True)
                        got = call().cpu()
                        assert (
                            got.shape == expected.shape
                            and got.dtype == expected.dtype
                            and torch.isfinite(got).all()
                        )
                        torch.testing.assert_close(got, expected, rtol=0.01, atol=0.004)
                        info["accuracy"].append(
                            dict(
                                round=rd + 1,
                                seed=seed,
                                label=label,
                                max_abs_error=(got.float() - expected.float()).abs().max().item(),
                            )
                        )
                    if seed == 7:
                        if not args.only_static_ascendc or args.raw_breakdown:
                            calls["static/local"] = local
                        timed_calls = calls
                if args.smoke:
                    continue
                report.update(phase="timing", active=f"{key}/round{rd+1}")
                save()
                labels = tuple(timed_calls)
                samples = {label: [] for label in labels}
                for repeat in range(args.timing_rounds):
                    if args.initial_warmup:
                        report.update(phase="initial_warmup", active=f"{key}/build{rd+1}/measure{repeat+1}")
                        save()
                        for sweep in range(args.initial_warmup):
                            for label in paired_order(labels, sweep, rd, repeat):
                                timed_calls[label]()
                            if (sweep + 1) % 25 == 0 or sweep + 1 == args.initial_warmup:
                                print(f"[FGW warmup] {key} measure={repeat+1}/{args.timing_rounds} {sweep+1}/{args.initial_warmup} per control", flush=True)
                        torch.npu.synchronize()
                        # Create/record both reusable timing events outside measurement.
                        start.record()
                        end.record()
                        end.synchronize()
                        report['timer_primed'] = True
                    report.update(phase="timing", active=f"{key}/build{rd+1}/measure{repeat+1}")
                    save()
                    batch = {label: [] for label in labels}
                    for pair in range(args.pairs):
                        offset = (pair + rd + repeat) % len(labels)
                        order = labels[offset:] + labels[:offset]
                        if (pair + rd + repeat) % 2:
                            order = tuple(reversed(order))
                        if args.only_static_ascendc:
                            order = paired_order(labels, pair, rd, repeat)
                        for label in order:
                            print(f"[FGW timing] {key} build={rd+1}/{args.rounds} measure={repeat+1}/{args.timing_rounds} pair={pair+1}/{args.pairs} {label} starting", flush=True)
                            us = measure(timed_calls[label])
                            samples[label].append(us)
                            batch[label].append(us)
                            print(f"[FGW timing] {key} measure={repeat+1} pair={pair+1}/{args.pairs} {label}={us:.3f} us", flush=True)
                    batch_medians = {label: statistics.median(values) for label, values in batch.items()}
                    report['measurement_rounds'].append(dict(key=key, compile_round=rd+1,
                        measurement_round=repeat+1, median_us=batch_medians, samples_us=batch))
                    save()
                    print(f"[FGW repeat] {key} {repeat+1}/{args.timing_rounds} {batch_medians}", flush=True)
                medians = {k: statistics.median(v) for k, v in samples.items()}
                rows.append(medians)
                report["rounds"].append(dict(key=key, round=rd + 1, median_us=medians, samples_us=samples))
                save()
                print(key, "round", rd + 1, medians, flush=True)
            info["accuracy_checks"] = len(info["accuracy"])
            info["max_abs_error"] = max(a["max_abs_error"] for a in info["accuracy"])
            if args.smoke:
                report["summary"].append(dict(key=key, status="accuracy_passed", plan=plan.describe()))
            else:
                result = dict(
                    key=key,
                    median_us={k: statistics.median(r[k] for r in rows) for k in rows[0]},
                    round_us=rows,
                    static_builds=len(info["builds"]),
                    reference_builds=info.get("reference_builds", 0),
                    plan=plan.describe(),
                    merge_assignment={k: v for k, v in info["merge_assignment"].items() if k != "tasks"},
                )
                for control in ("default", "uniform_best", "ascendc", "fixed_r8", "simd_ref", "raw_ref"):
                    if control not in rows[0]:
                        continue
                    ratios = [r[control] / r["static"] for r in rows]
                    result["static_vs_" + control] = dict(
                        speedup=statistics.median(ratios),
                        min_round=min(ratios),
                        max_round=max(ratios),
                        faster_rounds=sum(x > 1 for x in ratios),
                    )
                report["summary"].append(result)
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
