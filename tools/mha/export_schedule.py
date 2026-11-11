"""Export actual Ascend generated code and pass IR before manual scheduling."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import traceback
import zipfile

CASES = {
    "q4_split": (2, 4, 8, 2048),
    "q129_tail": (2, 129, 64, 256),
    "q4096": (2, 4096, 8, 2048),
    "q32768": (2, 32768, 8, 2048),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--schedule-mode", choices=["auto", "auto-offset", "manual"], default="auto")
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    directory = Path(args.output_dir or f"mha_{args.schedule_mode}_schedule_{stamp}").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "cases": [], "output_dir": str(directory),
              "mode": "Compilation/export only; no kernel execution or performance claim",
              "config": {"q_resident": True, "pipeline_stages": 2, "schedule_mode": args.schedule_mode,
                         "buffer_policy": "scores3/current1/alpha3/KV3; others2" if args.schedule_mode == "manual" else "baseline2"}}

    def save():
        (directory / "manifest.json").write_text(json.dumps(report, indent=2, default=str)+"\n", encoding="utf-8")

    save()
    try:
        import torch
        import torch_npu
        import tilelang
        from operators.mha.kernels import mha_paged_decode_ascend_v7 as implementation
        torch.npu.set_device(args.device)
        report["versions"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "tilelang": tilelang.__version__}
        report["device"] = str(torch.npu.get_device_properties(args.device))
        report["kernel_source_sha256"] = hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest()
        for name in args.cases:
            b, sq, h, length = CASES[name]
            dest = directory / name
            dest.mkdir(parents=True, exist_ok=True)
            row = {"case": name, "shape": {"B": b, "Sq": sq, "H": h, "D": 128, "KV": length}, "status": "running", "kernels": []}
            report["cases"].append(row)
            save()
            print(f"Compiling/exporting {name} ...", flush=True)
            program = implementation.build_kernel(b, sq, h, 128, b * length, length // 128,
                128, False, "bfloat16", args.device, q_resident=True,
                pipeline_stages=2, buffer_versions=2, kv_versions=2, dump_ir_dir=str(dest / "passes"), schedule_mode=args.schedule_mode)
            direct = getattr(program, "execution_mode", "") == "direct_single_split"
            kernels = {"direct": program.direct_kernel} if direct else {"partial": program.partial_kernel, "merge": program.merge_kernel}
            row["kernel_mode"] = getattr(program, "execution_mode", "unknown")
            for label, kernel in kernels.items():
                item = {"name": label, "optional_export_errors": {}}
                row["kernels"].append(item)
                source = kernel.get_kernel_source()
                if not isinstance(source, str) or not source.strip():
                    raise RuntimeError(f"Missing generated device source: {name}/{label}")
                (dest / f"{label}_device.cpp").write_text(source, encoding="utf-8")
                for field, getter in {
                    "host": kernel.get_host_source,
                    "input_ir": lambda k=kernel: k.prim_func.script(),
                    "lowered_device_ir": lambda k=kernel: k.artifact.device_mod.script(),
                }.items():
                    try:
                        (dest / f"{label}_{field}.txt").write_text(str(getter()), encoding="utf-8")
                    except Exception as exc:
                        item["optional_export_errors"][field] = repr(exc)
                item["device_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
            passes = [p for p in (dest / "passes").rglob("*") if p.is_file()]
            row["pass_dump_count"] = len(passes)
            row["schedule_related_pass_files"] = [str(p.relative_to(dest)) for p in passes
                if any(key in p.name.lower() for key in ("autoschedule", "sync", "multibuffer", "lowerscheduled", "estimat"))]
            if not passes:
                raise RuntimeError("Generated source exported, but no pass IR dumps found; cannot yet inspect schedule decisions")
            row["status"] = "passed"
            save()
        report["status"] = "passed"
    except Exception:
        report.update(status="error", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        save()
        archive = directory.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for p in directory.rglob("*"):
                if p.is_file():
                    bundle.write(p, p.relative_to(directory.parent))
        print(f"{report['status'].upper()}; upload: {archive}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
