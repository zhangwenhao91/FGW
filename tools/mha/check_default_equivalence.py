"""Compare clean/default and archived/default PrimFuncs without an NPU runtime.

Requires a TileLang build supporting Ascend and CPU PyTorch. Device bodies are
captured before JIT adapter construction; no torch_npu module is synthesized.
"""

import ast
import hashlib
import json
from pathlib import Path
import sys
import types

import tilelang
from tilelang import tvm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load(name):
    path = ROOT / "operators/mha/kernels" / (name + ".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        n
        for n in tree.body
        if not (
            isinstance(n, ast.Import)
            and any(a.name == "torch_npu" for a in n.names)
            or isinstance(n, ast.ImportFrom)
            and n.module == "operators.mha.kernels.mha_paged_decode_ascend_v2"
        )
    ]

    class Capture(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tilelang"
                and node.func.attr == "compile"
            ):
                node.func = ast.Name(id="capture", ctx=ast.Load())
            return node

    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.capture = lambda function, **kwargs: function
    module.build_v2 = lambda *args: ("fallback", args)
    sys.modules[name] = module
    exec(compile(ast.fix_missing_locations(Capture().visit(tree)), str(path), "exec"), module.__dict__)
    return module


def main():
    clean = load("mha_paged_decode_ascend_v7")
    old = load("mha_paged_decode_ascend_v7_experimental")
    cases = [
        (1, 128, 1, 512, 1, False),
        (1, 129, 1, 2048, 4, False),
        (1, 4096, 1, 8192, 1, False),
        (1, 8192, 1, 8192, 1, False),
        (2, 33, 3, 2048, 2, False),
        (1, 1, 1, 2048, 2, True),
    ]
    for b, q, h, kv, split, causal in cases:
        args = (b, q, h, 128, kv, kv // 128, 128, causal, "bfloat16", 0)
        a = clean.build_kernel(*args, num_splits=split, split_policy="legacy")
        z = old.build_kernel(*args, num_splits=split, split_policy="legacy")
        for attr in ("num_splits", "compute_grid", "task_assignment", "execution_mode"):
            assert getattr(a, attr) == getattr(z, attr), attr
        components = ("direct_kernel",) if split == 1 else ("partial_kernel", "merge_kernel")
        hashes = {}
        for component in components:
            x, y = getattr(a, component), getattr(z, component)
            with tvm.target.Target("ascend"):
                sx = tilelang.lower(x, target="ascend").kernel_source
                sy = tilelang.lower(y, target="ascend").kernel_source
            assert sx == sy, (args, component, "generated C++ mismatch")
            hashes[component] = hashlib.sha256(sx.encode()).hexdigest()
        print(
            json.dumps(
                {
                    "shape": [b, q, h, kv],
                    "split": split,
                    "causal": causal,
                    "status": "generated_C++_identical",
                    "sha256": hashes,
                }
            ),
            flush=True,
        )
    for dtype, d, q, causal in (
        ("float16", 128, 1, False),
        ("bfloat16", 64, 1, False),
        ("bfloat16", 128, 2, True),
    ):
        args = (1, q, 1, d, 512, 4, 128, causal, dtype, 0)
        assert clean.build_kernel(*args) == old.build_kernel(*args)
    print("Default fallback dispatch identical; all checks passed", flush=True)


if __name__ == "__main__":
    main()
