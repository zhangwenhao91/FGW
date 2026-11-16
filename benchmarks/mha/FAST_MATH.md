# Explicit Fast Math comparison

Run `python fgw.py mha run run_mha_fast_math` on the configured Ascend machine.
Return `reports/mha/latest/mha_fast_math_paste.txt` (also written on failure).
Each invocation writes a fresh run directory and its own generated C++ sources.

The benchmark compares:

- `default`: current manual v7, no Fast Math pass override.
- `off`: `TL_ENABLE_FAST_MATH=False`.
- `on`: `TL_ENABLE_FAST_MATH=True`.
- `ascendc`: installed native FIA v2 on the same original paged cache.

Default is deliberately not assumed to equal off. The default split model is
evaluated first; off/on use the identical explicit split. Pipeline stages2,
manual schedule, buffer arrangement, tile128, persistent cap28 and formulas stay
the same. The flag is applied to the direct kernel, or both partial and merge.
The ordinary public MHA defaults remain unchanged; Fast Math is an opt-in builder
parameter for this restricted v7 Cube experiment, not the general/Vector paths.

Smoke covers Q129/KV512 and Q4096/KV8192. Full coverage is the B1H1 56-case
Cartesian grid with Q1..8192 and KV512/2048/4096/8192 from the previous rerun.
All inputs/outputs are BF16 D128, page128, noncausal. Every shape checks seeds7/19
against the FP32 chunked CPU reference before measuring that shape. The input
generator includes shuffled/repeated physical pages and extra unused pool pages.
Six rounds of 51 rotating/reversing four-way Event samples measure full calls.
Compilation, host/device transfers, references and FIA metadata are outside timing.
Q preparation, output allocation and all device kernels remain inside.

Reports preserve per-variant maximum error, split/grid, generated source hashes,
raw samples, per-round medians and speedup ranges. Text-source equality alone is
not proof of identical final binaries. Ratios off/on, default/on and AscendC/on
above1 mean enabling Fast Math is faster. No default promotion is automatic.

Reference: the explicit compile option in [PR3896's pinned example](https://github.com/tile-ai/tilelang/blob/49a51226eb4068d5c0bf205cee886d2a7dcf0b05/examples/ascend/flash_attention/example_mha.py).
