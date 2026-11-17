# Merge rows8/16/32 candidates

```bash
python fgw.py mha run run_mha_merge_granularity
cat reports/mha/latest/mha_merge_granularity_paste.txt
```

The default v7 file and frozen split model are unchanged. Candidate code is in
`operators/mha/kernels/mha_merge_rows_candidate.py`. Only the row cap changes:
8 (original),16,32. Row softmax-state weights, accumulation, normalization,
BF16 cast, SIMT threads128, Q128 partial layout and merge grid cap32768 are retained.
Rows are capped at next_power_of_2(Q), so tiny Q remains valid. Every row cap
divides128 and chunks never cross partial Q tiles.

| Q | Merge programs at rows8 | rows16 | rows32 |
|---:|---:|---:|---:|
|8192|1024|512|256|
|4096|512|256|128|
|2048|256|128|64|

Fewer programs/larger copies trade against more per-program work and potentially
less parallelism; larger row caps are not assumed to be faster.

The same three Q/KV cases and splits1/2/4/8 from the breakdown test are retained.
S1 is a direct-output control with no merge (R8 is only its label placeholder).
For S2/4/8, each round compiles the original program once. R8 uses its original
merge; R16/32 compile only replacement merge kernels and reuse the IDENTICAL
partial callable and Q preparation. There are two independent rounds, with
cache disabled and private compiler scratch. Candidate merges are built once
per row-cap/split/shape/round; the shared partial is also rebuilt each round.

Both seeds7/19 check full, fixed-fixture merge and actual local->merge chain
against the same chunked CPU reference before timing. All candidates receive
five immediate warmups before each of seven rotating/reversing Event samples.
Measurements retain full/merge/chain and one shared local timing per split.
Normal allocations remain in JIT calls; compile, references and H2D are excluded.
Installed AscendC FIA and S1 are measured again in the same session.

Summary includes paired full and merge speedups vs original R8 at the SAME split,
full speedup vs unchanged auto selection/FIA, grids, source hashes, shared-partial
checks, two merge build counts and best (split,row-cap) full median. Isolated
local+merge remains diagnostic rather than a substitute for full measurement.
Compact paste omits lengthy individual accuracy entries; raw JSON keeps them.

Local checks: R8 generated C++ exactly matches the original merge for
B1Q129H1/S4, B2Q17H3/S2, B1Q1H1/S2 and B1Q8192H1/S2. R16/R32 also lower
successfully for these cases. Official Ascend950PR_9579 CAModel execution of
merge-only Q129/S4, grids9 and5, passed CPU Partial/LSE fixture checks for both
R16 and R32 at rtol .01/atol .004, including the final Q tail. Local compiler
e84ce3de differs from the NPU compiler a7cb753d; this is not a hardware speedup
claim. End-to-end NPU precision/performance still needs this runner.

Resume uses the same optional `--resume old/full.json` bash-runner argument.
It checks candidate/default/model hashes, device, versions and settings before
retaining completed rounds. Old non-granularity reports cannot be resumed here.
