"""Compare executable ELF sections without loading or replacing either kernel."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

SECTION = re.compile(
    r"^\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+"
    r"([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+"
    r"[0-9a-fA-F]+\s+([A-Za-z]+)\s"
)
BF16 = "SparseFlashAttention_9ec0bacf213f2b2dce3ba70149a4903b.o"


def executable_sections(path, reader):
    output = subprocess.check_output([reader, "-WS", str(path)], text=True)
    sections = {}
    with path.open("rb") as stream:
        for line in output.splitlines():
            match = SECTION.match(line)
            if not match or "X" not in match[5]:
                continue
            name, address, offset, size = match.group(1, 2, 3, 4)
            size = int(size, 16)
            stream.seek(int(offset, 16))
            content = stream.read(size)
            if len(content) != size or name in sections:
                raise ValueError(f"invalid or duplicate section {name} in {path}")
            sections[name] = {
                "address": int(address, 16),
                "bytes": size,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
    if not sections:
        raise ValueError(f"no executable ELF sections in {path}")
    return sections


def find_object(root):
    if root.is_file():
        return root
    matches = sorted(root.rglob(BF16))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {BF16} under {root}; found {len(matches)}"
        )
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installed", type=Path)
    parser.add_argument("rebuilt", type=Path)
    args = parser.parse_args()
    reader = shutil.which("llvm-readelf") or shutil.which("readelf")
    if not reader:
        parser.error("requires readelf or llvm-readelf")
    installed, rebuilt = map(find_object, (args.installed, args.rebuilt))
    old, new = [executable_sections(path, reader) for path in (installed, rebuilt)]
    shared = sorted(old.keys() & new.keys())
    changed = [name for name in shared if old[name] != new[name]]
    print(
        json.dumps(
            {
                "installed": str(installed),
                "rebuilt": str(rebuilt),
                "installed_exec_sections": len(old),
                "rebuilt_exec_sections": len(new),
                "identical_exec_sections": len(shared) - len(changed),
                "all_exec_sections_match": old == new,
                "only_installed": sorted(old.keys() - new.keys())[:20],
                "only_rebuilt": sorted(new.keys() - old.keys())[:20],
                "changed_count": len(changed),
                "changed_examples": [
                    {"section": name, "installed": old[name], "rebuilt": new[name]}
                    for name in changed[:8]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
