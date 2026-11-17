# Clean default v7 vs installed AscendC FIA: 80 shapes

Run `python fgw.py mha run run_mha_v7_clean_ascendc_grid` after pulling updates.
Q is 64/128/256/512/1024/2048/4096/8192/16384/32768; KV is
512/1024/2048/4096/8192/16384/24576/32768. All combinations use B1H1, BF16,
D128, page128, noncausal attention, full live cache, and shuffled/repeated physical
pages including unused pool pages. Q is sequence length, not the Q128 tile size.

The test imports cleaned v7, not the experimental archive. Its default split
selector and 28-program cap are unchanged, including outside the cost model's
verified range where the existing legacy selector applies. It compares the
installed `torch_npu.npu_fused_infer_attention_score_v2` with identical inputs.
Both pass seeds7/19 against FP32 CPU attention computed in Q chunks of 128,
rtol .01/atol .004, before timing each shape. Larger Q/KV can increase runtime;
the reference avoids allocating a full Q-by-KV score matrix.

Exactly two independent TL builds per shape (160 builds total), with builder
LRU cleared and TileLang caching disabled: initial accuracy build reused in
round1, another fresh build for round2. Each round has seven alternating paired
Event samples, reusing that round's binary. EVERY TL and FIA sample receives
five immediate warmup calls and synchronization before Event recording. Compile,
accuracy, CPU reference and H2D are excluded. Full operator execution includes
Q preparation, allocations and all kernels. Speedup is FIA us / TL us, with
per-shape median paired-round ratios and a geometric mean across all 80 shapes.

The compact report is `reports/mha/latest/mha_v7_clean_ascendc_grid_paste.txt`.
It includes the Q-by-KV speedup matrix, per-shape timings/splits/round ranges and
completed count. The fresh run folder retains full JSON, generated device C++,
raw samples, compile counts, versions, source hashes, log and matrix CSV. Errors
stop execution and preserve phase, active shape and partial results; missing
matrix cells are marked incomplete, not unsupported.
