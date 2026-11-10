#!/usr/bin/env python3
"""Collect Ascend/TileOPs development facts using only the Python standard library.

Every import/device probe runs in a child process so a broken runtime cannot abort
the rest of the report. No packages are installed and no network requests are made.
"""
import argparse
import datetime
import glob
import importlib.metadata
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys


def run(argv, timeout=60):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=timeout)
        return {"command": argv, "returncode": p.returncode,
                "stdout": p.stdout[-100000:], "stderr": p.stderr[-20000:]}
    except Exception as exc:
        return {"command": argv, "error": str(exc)}


def probe(code, timeout=90):
    return run([sys.executable, "-c", code], timeout)


TORCH = r'''
import json, torch, torch_npu
def attempt(fn):
    try: return fn()
    except Exception as e: return {"error": repr(e)}
npu = torch.npu
data = {"torch": torch.__version__, "torch_path": torch.__file__,
        "torch_npu": torch_npu.__version__, "torch_npu_path": torch_npu.__file__,
        "torch_git": torch.version.git_version, "torch_cuda": torch.version.cuda,
        "available": attempt(npu.is_available), "count": attempt(npu.device_count),
        "torch_config": torch.__config__.show()}
data["devices"] = []
if data["available"] is True:
    for i in range(npu.device_count()):
        data["devices"].append({"index": i,
            "name": attempt(lambda: npu.get_device_name(i)),
            "properties": attempt(lambda: str(npu.get_device_properties(i))),
            "mem_get_info": attempt(lambda: npu.mem_get_info(i))})
for name in ("_C", "version"):
    obj = getattr(torch_npu, name, None)
    data[name + "_version_symbols"] = {
        key: str(getattr(obj, key)) for key in dir(obj)
        if any(s in key.lower() for s in ("version", "cann", "soc"))}
print(json.dumps(data, indent=2, default=str))
'''

TILELANG = r'''
import json, pathlib, tilelang
root = pathlib.Path(tilelang.__file__).resolve().parent
data = {"version": getattr(tilelang, "__version__", None), "path": str(root),
        "symbols": [x for x in dir(tilelang) if not x.startswith("_")]}
try:
    import tilelang.language as T
    data["language_symbols"] = [x for x in dir(T) if not x.startswith("_")]
except Exception as e: data["language_error"] = repr(e)
try:
    from tilelang import tvm
    data["tvm_version"] = tvm.__version__
    data["tvm_build"] = tvm.support.libinfo()
    data["target_kinds"] = list(tvm.target.Target.list_kinds())
    data["ascend_global_functions"] = [x for x in tvm.get_global_func("runtime.ListGlobalNames")()
        if any(s in x.lower() for s in ("ascend", "npu", "cce"))]
except Exception as e: data["tvm_error"] = repr(e)
matches = []
for p in root.rglob("*"):
    if any(s in p.name.lower() for s in ("ascend", "cann", "npu", "pto")):
        matches.append(str(p.relative_to(root)))
        if len(matches) >= 300: break
data["ascend_related_files"] = matches
data["note"] = "Symbols/targets/files are evidence only; no TileLang NPU kernel compilation was attempted."
print(json.dumps(data, indent=2, default=str))
'''

SMOKE = r'''
import json, torch, torch_npu
results = {}
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    try:
        x = torch.linspace(-1, 1, 1024).reshape(4, 256).to(dtype)
        a = x.to("npu:0")
        y = a * 2 + 1
        norm = a.float() * torch.rsqrt(a.float().square().mean(-1, keepdim=True) + 1e-6)
        mm = a @ a.T
        torch.npu.synchronize()
        ref = x.float() @ x.float().T
        results[str(dtype)] = {"elementwise_max_error": (y.cpu().float() - (x * 2 + 1).float()).abs().max().item(),
            "rmsnorm_finite": bool(torch.isfinite(norm).all().cpu().item()),
            "matmul_max_error": (mm.cpu().float() - ref).abs().max().item()}
    except Exception as e: results[str(dtype)] = {"error": repr(e)}
print(json.dumps(results, indent=2))
'''


def collect(args):
    result = {"schema_version": 1, "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "expected": {"device": "Ascend950PR_9579", "torch": "2.13.0+cpu",
                     "torch_npu": "2.13.0", "tilelang": "0.1.15+cpu.gita7cb753d"},
        "system": {"os": platform.system(), "release": platform.release(),
                   "machine": platform.machine(), "python": sys.version,
                   "executable": sys.executable, "cpu_count": os.cpu_count()},
        "environment": {k: os.environ[k] for k in (
            "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH",
            "ASCEND_CUSTOM_OPP_PATH", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES",
            "NPU_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX",
            "TILELANG_TARGET", "TVM_HOME", "CC", "CXX") if k in os.environ}}
    packages = []
    for d in importlib.metadata.distributions():
        name = d.metadata.get("Name", "")
        item = {"name": name, "version": d.version}
        if any(s in name.lower() for s in ("torch", "tile", "tvm", "ascend", "triton", "numpy", "pytest")):
            item["location"] = str(d.locate_file(""))
        packages.append(item)
    result["packages"] = sorted(packages, key=lambda x: x["name"].lower())
    result["commands"] = {}
    for argv in (["uname", "-a"], ["lscpu"], ["free", "-h"], ["df", "-h", "."],
                 ["npu-smi", "info"], ["npu-smi", "info", "-l"],
                 ["gcc", "--version"], ["g++", "--version"], ["cmake", "--version"],
                 ["ninja", "--version"], ["bisheng", "--version"], ["ccec", "--version"],
                 ["git", "--version"], ["ldd", "--version"]):
        result["commands"][" ".join(argv)] = run(argv) if shutil.which(argv[0]) else {"status": "not_found"}
    result["version_files"] = {}
    roots = {"/usr/local/Ascend", os.path.expanduser("~/Ascend")}
    roots.update(os.environ[k] for k in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME") if os.environ.get(k))
    patterns = ("*version*.info", "version.cfg", "version.info", "*/version.cfg", "*/version.info",
                "*/version*.info", "*/*/version*.info", "*/*/*/version*.info", "*/ascend_install.info")
    for root in sorted(roots):
        for pattern in patterns:
            for filename in glob.glob(str(pathlib.Path(root) / pattern)):
                p = pathlib.Path(filename)
                if p.is_file() and str(p) not in result["version_files"]:
                    try: result["version_files"][str(p)] = p.read_text(errors="replace")[:20000]
                    except OSError as exc: result["version_files"][str(p)] = str(exc)
    for filename in ("/etc/ascend_install.info", "/usr/local/Ascend/driver/version.info"):
        p = pathlib.Path(filename)
        if p.is_file():
            try: result["version_files"][filename] = p.read_text(errors="replace")[:20000]
            except OSError as exc: result["version_files"][filename] = str(exc)
    result["torch_probe"] = probe(TORCH, args.timeout)
    result["tilelang_probe"] = probe(TILELANG, args.timeout)
    result["npu_smoke"] = probe(SMOKE, args.timeout) if not args.skip_smoke else {"status": "skipped"}
    checks = {}
    try:
        observed = json.loads(result["torch_probe"]["stdout"])
        for name in ("torch", "torch_npu"):
            checks[name] = {"expected": result["expected"][name], "actual": observed[name],
                            "match": observed[name] == result["expected"][name]}
        names = [d["name"] for d in observed["devices"]]
        checks["device"] = {"expected": result["expected"]["device"], "actual": names,
                             "match": result["expected"]["device"] in names}
    except Exception as exc: checks["torch_device_error"] = str(exc)
    try:
        actual = json.loads(result["tilelang_probe"]["stdout"])["version"]
        checks["tilelang"] = {"expected": result["expected"]["tilelang"], "actual": actual,
                              "match": actual == result["expected"]["tilelang"]}
    except Exception as exc: checks["tilelang_error"] = str(exc)
    result["verification"] = checks
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="ascend_env_report", help="Output filename prefix")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds per Python probe")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip small NPU tensor computations")
    args = parser.parse_args()
    result = collect(args)
    prefix = pathlib.Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    prefix.with_suffix(".json").write_text(payload + "\n", encoding="utf-8")
    lines = ["Ascend / TileOPs environment report", "", "Version verification:",
             json.dumps(result["verification"], ensure_ascii=False, indent=2), "", "Full report:", payload]
    prefix.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))
    print("Reports:", prefix.with_suffix(".json"), prefix.with_suffix(".txt"))
    print("Probe errors are recorded in the reports; report creation does not certify NPU readiness.")


if __name__ == "__main__":
    main()
