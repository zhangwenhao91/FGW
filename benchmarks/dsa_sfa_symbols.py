"""Inspect installed SFA device binaries for source/PC attribution support."""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


def discover(roots):
    found = set()
    for root in roots:
        if not root.is_dir():
            continue
        for directory, _, files in os.walk(root):
            folder = Path(directory)
            for name in files:
                normalized = re.sub(r"[^a-z0-9]", "", (folder.name + name).lower())
                if "sparseflashattention" in normalized and "grad" not in normalized:
                    path = folder / name
                    if path.suffix in {".o", ".so", ".bin", ".json"}:
                        found.add(path)
    return sorted(found)


def elf_report(path, readelf):
    result = {"path": str(path)}
    try:
        with path.open("rb") as stream:
            result["elf"] = stream.read(4) == b"\x7fELF"
    except OSError as error:
        result["error"] = str(error)
        return result
    if not result["elf"] or not readelf:
        return result
    process = subprocess.run(
        [readelf, "-W", "-S", "-s", str(path)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=20,
        check=False,
    )
    if process.returncode:
        result["error"] = process.stderr.strip()[:500]
        return result
    sections = sorted(
        set(re.findall(r"\.(?:z?debug_[a-z_]+|symtab|dynsym)\b", process.stdout))
    )
    result["sections"] = sections
    result["has_line_table"] = any(
        x in sections for x in (".debug_line", ".zdebug_line")
    )
    result["stage_symbols"] = [
        line.strip()[-280:]
        for line in process.stdout.splitlines()
        if re.search(r"ProcessVec[012]|FlashUpdate|SoftmaxFlash|simd_vf_", line, re.I)
    ][:12]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, default=[])
    parser.add_argument("--elf", action="append", type=Path, default=[])
    parser.add_argument("--limit", type=int, default=64)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    roots = list(args.root)
    for name in ("ASCEND_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH"):
        roots.extend(Path(p) for p in os.environ.get(name, "").split(os.pathsep) if p)
    for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        if value := os.environ.get(name):
            roots.extend(Path(value) / sub for sub in ("opp", "opp_kernel"))
    roots = sorted({p.resolve() for p in roots})
    candidates = discover(roots)
    binaries = [p for p in candidates if p.suffix != ".json"]
    reader = shutil.which("llvm-readelf") or shutil.which("readelf")
    results = [
        elf_report(p, reader)
        for p in dict.fromkeys([*args.elf, *binaries[: args.limit]])
    ]
    print(
        json.dumps(
            {
                "roots": [{"path": str(p), "exists": p.is_dir()} for p in roots],
                "readelf": reader,
                "binary_candidates": len(binaries),
                "inspected_device_files": len(results),
                "truncated": len(binaries) > args.limit,
                "metadata_examples": [
                    str(p) for p in candidates if p.suffix == ".json"
                ][:8],
                "with_line_table": [r for r in results if r.get("has_line_table")],
                "with_stage_symbols": [
                    r
                    for r in results
                    if r.get("stage_symbols") and not r.get("has_line_table")
                ],
                "other_examples": [
                    r
                    for r in results
                    if not r.get("has_line_table") and not r.get("stage_symbols")
                ][:6],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
