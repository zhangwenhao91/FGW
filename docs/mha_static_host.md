# Static host scheduling prototype

```bash
python fgw.py mha run run_mha_static_host
cat reports/mha/latest/mha_static_host_paste.txt
```

Independent candidate only; default v7, frozen selector and merge-row experiments
are not changed. Supports B1H1 BF16/D128, page128, noncausal, positive full live
KV divisible by128, static Q/KV lengths. Q tails are padded using v7's existing
copy rules. The factory requires a whole-page physical pool at least as large as
KV capacity; the benchmark uses the same shuffled/repeated-page generator as
prior tests. `validate_inputs` checks the full-cache contract outside timing.
The low-level callable assumes that contract continues to hold on later calls.

## Host plan

Flatten (Q tile, KV block) in Q-major order, then divide contiguous intervals
among up to28 programs with per-program KV-block counts differing by at most1.
Each program receives a small segment list `(Q tile, KV begin, KV count, slot)`.
No atomics, counters, device work queue or runtime work stealing are introduced.

A complete Q tile has slot=-1 and writes BF16 final output directly. Cross-program
Q tiles receive consecutive FP32 normalized Partial/LSE slots and a separate
`(Q tile, first slot, slot count)` merge list. Only that list is reduced. Merge
uses the unchanged row8 softmax-state formula and writes the existing final O
in place, preserving directly written rows. QK/PV order, resident Q, manual
stages, K/V buffer versions and SIMD macros remain those of default v7. Online
softmax state is reset per segment, including segments starting inside KV.

| Q/KV | KV blocks per program | Direct Q tiles | Merged Q tiles | Partial slots |
|---|---:|---:|---:|---:|
|8192/32768|585..586|37|27|54|
|4096/8192|73..74|5|27|54|
|2048/24576|109..110|0|16|43|
|3584/32768|256|28|0|0|
|7168/32768|512|56|0|0|

This first prototype always performs the block-balanced partition. It does not
yet decide whether splitting is worth its cost, reproduce the AscendC host cost
model, or guarantee balanced runtime from balanced block counts. Pipeline
startup/drain, per-segment normalization and metadata accesses still have costs.

## Allocation and timing contract

Task tensors are built/uploaded once per compiled callable/device, before timing.
The plan and metadata are immutable for its Q/KV configuration. Full calls include
Q preparation, local output/workspace allocation and selective merge. The local
JIT returns Parts/LSE/O. A no-fragment plan still has one unused dummy Partial/LSE
slot allocated by the prototype's common signature, but launches no merge kernel.
This overhead is intentionally included and exposed in balanced control tests.

`static/local` uses preprepared Q and normal JIT output allocation. Isolated
`static/merge` updates boundary rows of a fixed valid fixture in place; it has no
output allocation. `static/chain` runs actual local then merge with preprepared
Q. Component times are diagnostic and need not sum to full. There is no claim
that isolated merge allocation contracts match the older allocating merge.

## NPU checks

First run host schedule tests and accuracy-only smoke shapes Q1/17/129/3712/3584,
KV512, covering many-way fragments, Q tails, mixed complete+fragment work and
no-merge plans. Smoke compiles once per shape and checks seeds7/19. On failure,
stop before larger tests and preserve active phase/report/traceback.

Performance cases are the three prior regressions plus balanced controls:
Q8192/KV32768, Q4096/KV8192, Q2048/KV24576, Q3584/KV32768, Q7168/KV32768.
Compare unchanged automatic v7, previously best uniform S2/S1/S4/S1/S1,
static candidate and installed FIA v2 on exactly the same inputs. If automatic
and uniform choices coincide, reuse the callable rather than timing duplicates.
Two independent compile rounds per shape, cache disabled, private compiler
scratch, seeds7/19 correctness per round, seven rotated paired samples with five
immediate warmups per callable before EACH Event sample. No kernel/default
promotion occurs automatically. Source hashes, generated C++, complete host
plans, accuracy records, raw samples and round speedup ranges are saved.

## Local validation

`python -m unittest tests.mha.test_static_schedule -v` checks exact independent
block coverage, unique ownership, balanced counts, slot mapping, no-merge cases,
and an independent scalar dense-attention/partial-state merge oracle including
full+fragment work and a Q tail. Additional384 schedule combinations passed.

TileLang e84ce3de lowering passed Q129/KV512 (all fragments), Q3712/KV512
(mixed segments), Q3584/KV512 (no merge). Official Ascend950PR_9579 CAModel
checks passed mixed program0 with KV counts4 then1; all8 programs at Q129/KV512;
and that Q129 local output followed by the selective merge in a separate replay
(FP32 oracle max absolute error .00113067, rtol .01/atol .004).
These are compiler/simulation checks, not the NPU end-to-end adapter or a speedup
claim. Local and server TileLang builds differ; the NPU smoke gate is required.
An additional official merge-only replay at Q3712/KV512 verified that only its
three boundary Q tiles were overwritten and all26 direct Q tiles remained
bit-identical to the initialized BF16 output fixture.
