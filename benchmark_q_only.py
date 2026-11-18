"""Compare frozen cost scheduling, whole-Q scheduling, and native FIA v2."""
import argparse
import functools
import json
import statistics
from pathlib import Path
from unittest.mock import patch

from benchmark import KVS, POLICY, QS, make_inputs, reference, verify_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q", type=int, choices=QS, default=32768)
    parser.add_argument("--kv", type=int, choices=KVS, default=8192)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = verify_source()

    from resource_guard import cpu_environment
    threads = cpu_environment()
    import torch
    import torch_npu
    from operators.mha.kernels import mha_paged_decode as mha
    from operators.mha.static_schedule import make_q_plan

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
    plans = {"cost": cost.plan.describe(), "q_only": whole.plan.describe()}
    print(json.dumps(plans), flush=True)

    accuracy = {}
    for seed in POLICY["seeds"]:
        host = make_inputs(torch, args.q, args.kv, seed)
        expected = reference(torch, *host)
        inputs = [tensor.to("npu") for tensor in host]
        cost.validate_inputs(*inputs)
        whole.validate_inputs(*inputs)
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
                 "AscendC": lambda native=native: native()[0]}
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
            offset = (repeat + pair) % len(labels)
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
    result = dict(Q=args.q, KV=args.kv, device=str(device), source=source,
                  plans=plans, max_abs_error=accuracy, samples_us=samples,
                  median_us=medians, q_only_speedup=medians["cost"] / medians["q_only"])
    report = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
