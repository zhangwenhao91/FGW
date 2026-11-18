"""Compare frozen cost, whole-Q, tail-split, and native FIA v2."""
import argparse
import functools
import json
import statistics
from pathlib import Path
from unittest.mock import patch

from benchmark import KVS, POLICY, QS, make_inputs, reference, verify_source
from operators.mha.static_schedule import Plan, make_plan, make_q_plan


def make_tail_plan(q, kv, cores=28):
    whole = make_q_plan(q, kv, cores)
    tiles = (q + 127) // 128
    waves, remainder = divmod(tiles, cores)
    if not waves or not remainder:
        return whole

    blocks = kv // 128
    prefix_tiles = waves * cores
    tail = make_plan(remainder * 128, kv, cores)
    if not tail.merges:
        return whole

    programs = []
    for pid in range(cores):
        first_tile = pid * waves
        prefix = tuple(
            (tile, 0, blocks, -1)
            for tile in range(first_tile, first_tile + waves)
        )
        extra = tail.programs[pid] if pid < len(tail.programs) else ()
        programs.append(prefix + tuple(
            (prefix_tiles + tile, begin, count, slot)
            for tile, begin, count, slot in extra
        ))
    merges = tuple((prefix_tiles + tile, slot, count)
                   for tile, slot, count in tail.merges)
    return Plan(q, kv, tuple(programs), merges, tail.partial_count, "tail_split")


def build_split_first(mha, q, kv, pool_rows, tail_plan):
    remainder = ((q + 127) // 128) % 28
    prefix_rows = (((q + 127) // 128) - remainder) * 128
    if not tail_plan.merges:
        raise ValueError("--split-first requires a shape with tail_split merges")

    compile_kernel = mha.tilelang.compile

    def compile_with_explicit_outputs(func, **kwargs):
        return compile_kernel(func, **dict(kwargs, out_idx=[]))

    with patch.object(mha.tilelang, "compile", compile_with_explicit_outputs):
        with patch.object(mha, "make_plan", make_q_plan):
            direct = mha.build_kernel(prefix_rows, kv, pool_rows, plan_policy="balanced")
        split = mha.build_kernel(q - prefix_rows, kv, pool_rows, plan_policy="balanced")
    mha.build_kernel.cache_clear()
    return prefix_rows, direct, split


def bind_split_first(torch, programs, q, k, v, table):
    prefix_rows, direct, split = programs
    counts, tasks, merges = split.prepare_plan(q.device)
    side = torch.npu.Stream(device=q.device)
    ready, done = torch.npu.Event(), torch.npu.Event()
    slots = split.plan.partial_count

    def run(parallel):
        prepared = q.view(1, 1, q.shape[1], 128)
        output = torch.empty((1, q.shape[1], 128), dtype=q.dtype, device=q.device)
        partial = torch.empty((slots, 128, 128), dtype=torch.float32, device=q.device)
        lse = torch.empty((slots * 128,), dtype=torch.float32, device=q.device)
        prefix = output[:, :prefix_rows, :]
        tail = output[:, prefix_rows:, :]
        split.local_kernel(prepared[:, :, prefix_rows:, :], k, v, table,
                           counts, tasks, partial, lse, tail)
        if parallel:
            current = torch.npu.current_stream(q.device)
            ready.record(current)
            side.wait_event(ready)
            with torch.npu.stream(side):
                split.merge_kernel(partial, lse, merges, tail)
                done.record(side)
        direct.local_kernel(prepared[:, :, :prefix_rows, :], k, v, table, prefix)
        if parallel:
            # All workspace was allocated on current. Join side before
            # returning it to the allocator or exposing the final output.
            current.wait_event(done)
        else:
            split.merge_kernel(partial, lse, merges, tail)
        return output.view_as(q)

    return {"split_first_serial": functools.partial(run, False),
            "split_first_parallel": functools.partial(run, True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=int, choices=QS, default=32768)
    parser.add_argument("--kv", type=int, choices=KVS, default=8192)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split-first", action="store_true", help="Compare serial and dual-stream split-first launches")
    args = parser.parse_args()
    source = verify_source()

    from resource_guard import cpu_environment
    threads = cpu_environment()
    import torch
    import torch_npu
    from operators.mha.kernels import mha_paged_decode as mha

    torch.set_num_threads(threads)
    torch.npu.set_device(0)
    device = torch.npu.get_device_properties(0)
    if getattr(device, "cube_core_num", None) != 28:
        raise RuntimeError("This benchmark requires the 28-Cube target")

    pool_rows = args.kv + 3 * 128
    cost = mha.build_kernel(args.q, args.kv, pool_rows, plan_policy="cost")
    # The balanced entry selects make_plan at build time. Substitute the
    # existing whole-Q planner for this build only.
    with patch.object(mha, "make_plan", make_q_plan):
        whole = mha.build_kernel(args.q, args.kv, pool_rows, plan_policy="balanced")
    assert whole.plan.strategy == "q_only" and whole.merge_kernel is None
    # Both experimental builds use the balanced entry. Clear its Python
    # cache so the second build consumes the next planner.
    mha.build_kernel.cache_clear()
    with patch.object(mha, "make_plan", make_tail_plan):
        tail = mha.build_kernel(args.q, args.kv, pool_rows, plan_policy="balanced")
    mha.build_kernel.cache_clear()

    plans = {label: program.plan.describe()
             for label, program in (("cost", cost), ("q_only", whole), ("tail_split", tail))}
    if args.split_first:
        programs = build_split_first(mha, args.q, args.kv, pool_rows, tail.plan)
        plans.update(split_first_direct=programs[1].plan.describe(),
                     split_first_partial=programs[2].plan.describe())
    print(json.dumps(plans), flush=True)

    accuracy = {}
    for seed in POLICY["seeds"]:
        host = make_inputs(torch, args.q, args.kv, seed)
        expected = reference(torch, *host)
        inputs = [tensor.to("npu") for tensor in host]
        cost.validate_inputs(*inputs)
        whole.validate_inputs(*inputs)
        tail.validate_inputs(*inputs)
        q, k, v, table, _ = inputs
        native = functools.partial(
            torch_npu.npu_fused_infer_attention_score_v2,
            q, k.view(-1,128,128), v.view(-1,128,128), block_table=table,
            num_query_heads=1, num_key_value_heads=1, input_layout="BSND",
            block_size=128, softmax_scale=128 ** -0.5, actual_seq_qlen=[args.q],
            actual_seq_kvlen=[args.kv], sparse_mode=0, inner_precise=0,
            return_softmax_lse=False,
        )
        calls = {"cost": functools.partial(cost, *inputs),
                 "q_only": functools.partial(whole, *inputs),
                 "tail_split": functools.partial(tail, *inputs),
                 "AscendC": lambda native=native: native()[0]}
        if args.split_first:
            calls.update(bind_split_first(torch, programs, q, k, v, table))
        accuracy[seed] = {}
        for label, call in calls.items():
            actual = call().cpu()
            torch.testing.assert_close(actual, expected, rtol=POLICY["rtol"], atol=POLICY["atol"])
            accuracy[seed][label] = (actual.float() - expected.float()).abs().max().item()
        if seed == POLICY["seeds"][0]:
            timed = calls

    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    labels = tuple(timed)
    samples = {label: [] for label in labels}
    for repeat in range(POLICY["repeats"]):
        for _ in range(POLICY["initial_warmup"]):
            for call in timed.values():
                call()
        torch.npu.synchronize()
        start.record()
        end.record()
        end.synchronize()
        for pair in range(POLICY["pairs"]):
            offset = (repeat * POLICY["pairs"] + pair) % len(labels)
            for label in labels[offset:] + labels[:offset]:
                call = timed[label]
                for _ in range(POLICY["immediate_warmup"]):
                    call()
                torch.npu.synchronize()
                start.record()
                output = call()
                end.record()
                end.synchronize()
                samples[label].append(start.elapsed_time(end) * 1000)
                del output

    medians = {label: statistics.median(values) for label, values in samples.items()}
    pairs = POLICY["pairs"]
    round_medians = [
        {label: statistics.median(values[start:start + pairs])
         for label, values in samples.items()}
        for start in range(0, POLICY["repeats"] * pairs, pairs)
    ]
    result = dict(Q=args.q, KV=args.kv, device=str(device), source=source,
                  plans=plans, max_abs_error=accuracy, samples_us=samples,
                  median_us=medians, round_median_us=round_medians,
                  q_only_speedup=medians["cost"] / medians["q_only"],
                  tail_split_speedup=medians["cost"] / medians["tail_split"])
    if args.split_first:
        result.update(
            parallel_vs_serial=medians["split_first_serial"] / medians["split_first_parallel"],
            parallel_vs_tail_split=medians["tail_split"] / medians["split_first_parallel"],
        )
    report = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
