# Retained raw SIMD baseline versus AscendC: 80 shapes

```bash
git pull --ff-only
python fgw.py mha run run_mha_raw_default_grid
cat reports/mha/latest/mha_raw_default_grid_paste.txt
```

Here the retained baseline means `mha_static_host_candidate.build_kernel`
with `plan_policy="stream"`, `merge_policy="rawsimd"`, R8, original softmax
(`row_softmax=False`). This is the baseline retained after the failed row
fusion experiments. It does not change the separate public v7 API defaults.
Only this baseline and installed native
`torch_npu.npu_fused_infer_attention_score_v2` are timed. No normalized merge,
reciprocal, row fusion or older v7 comparison build is included.

All 80 cases use B1H1, BF16, D128, noncausal, page128, full live KV:

* Q: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768.
* KV: 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768.

Each shape runs in a separately supervised process and compiles the TL
local/merge pair once (merge omitted when unnecessary). Both implementations
pass seeds 7 and 19 against the CPU reference before timing. Three measurement
repeats reuse the compilation. Each repeat warms both controls 100 times,
primes Events, then measures four alternating pairs with ten immediate
warmups per sample. Each control occupies each order position twice per
repeat. Full-call warm-cache Event timing includes preparation, allocations
and merge; compilation and initial host task preparation/upload are excluded.
AscendC native internal caching/dispatch remains its installed behavior.

The terminal prints shape progress out of 80, compilation/accuracy stages,
and every measurement. Completed shape files and generated sources are kept
in the printed private run directory even if a later shape fails. Existing
host resource limits remain in force. No profiling is performed in this run.

The summary checks settings, shapes, two correctness seeds, exactly one TL
build, no extra reference build, Event priming, all twelve samples per
control, and consistent source hashes/device/versions across shapes. Speedup
is AscendC latency divided by TL latency, using each control's median of all
twelve samples. It also reports the three per-repeat ratios, win counts,
geometric mean, an 80-cell matrix and details. No slow sample is discarded.
Raw samples remain in the case JSON and aggregate `summary.json`; the run
directory also contains `summary.csv` for plotting.
