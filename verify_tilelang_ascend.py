#!/usr/bin/env python3
"""Compile and verify a small FP32 Ascend vector-add kernel. No CPU fallback.

Uses the Ascend dialect/API from tile-ai/tilelang commit a7cb753d:
https://github.com/tile-ai/tilelang/blob/a7cb753d/examples/ascend/example_vecadd.py
"""
import argparse
import datetime
import json
import pathlib
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Logical NPU index after visibility filtering")
    parser.add_argument("--output", default="tilelang_ascend_smoke.json")
    args = parser.parse_args()
    report = {"status": "failed", "stage": "import", "target": "ascend",
              "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "python": sys.executable, "logical_device": args.device}
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        import torch_npu
        import tilelang
        import tilelang.ascend.language as T
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                              "tilelang": tilelang.__version__}
        report["stage"] = "device_init"
        if not torch.npu.is_available():
            raise RuntimeError("torch.npu.is_available() is False")
        torch.npu.set_device(args.device)
        report["device_name"] = torch.npu.get_device_name(args.device)
        # One pipeline iteration, aligned sizes and no tail handling. Retains the
        # upstream example's known buffer/thread geometry; total I/O is only 6 MiB.
        n, blocks, tile, threads = 524288, 64, 8192, 2048

        @T.prim_func
        def add(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"),
                C: T.Tensor((n,), "float32")):
            with T.Kernel(blocks) as block:
                lhs = T.alloc_shared((tile,), "float32")
                rhs = T.alloc_shared((tile,), "float32")
                dst = T.alloc_shared((tile,), "float32")
                T.annotate_buffer_versions({lhs: 2, rhs: 2, dst: 2})
                for step in T.Pipelined(1, num_stages=2):
                    offset = (step * blocks + block) * tile
                    T.copy(A[offset:offset + tile], lhs)
                    T.copy(B[offset:offset + tile], rhs)
                    with T.SimtVF(threads=threads):
                        for i in T.Parallel(tile):
                            dst[i] = lhs[i] + rhs[i]
                    T.copy(dst, C[offset:offset + tile])

        report["stage"] = "compile"
        print("Compiling FP32 vector add for target=ascend ...", flush=True)
        start = time.perf_counter()
        kernel = tilelang.compile(add, out_idx=-1, target="ascend", execution_backend="auto")
        report["compile_seconds"] = time.perf_counter() - start
        report["stage"] = "execute_and_check"
        cases = []
        generator = torch.Generator().manual_seed(20261110)
        for name in ("random", "changed_inputs", "zeros"):
            if name == "zeros":
                a, b = torch.zeros(n), torch.zeros(n)
            else:
                a = torch.randn(n, generator=generator)
                b = torch.randn(n, generator=generator)
            expected = a + b  # CPU reference, independent of torch_npu addition.
            actual = kernel(a.to(f"npu:{args.device}"), b.to(f"npu:{args.device}"))
            torch.npu.synchronize()
            if actual.device.type != "npu":
                raise AssertionError(f"Unexpected output device: {actual.device}")
            actual_cpu = actual.cpu()
            torch.testing.assert_close(actual_cpu, expected, rtol=1e-5, atol=1e-6)
            cases.append({"case": name, "max_abs_error": (actual_cpu - expected).abs().max().item(),
                          "status": "passed"})
            report["cases"] = cases
        report.update(status="passed", stage="complete", shape=[n], dtype="float32")
        print("PASS: compiled and executed TileLang Ascend kernel; all 3 CPU-reference checks passed.", flush=True)
    except Exception:
        report["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
