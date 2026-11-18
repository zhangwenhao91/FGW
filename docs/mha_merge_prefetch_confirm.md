# Confirm sub-percent prefetch merge gains with warmed repeated timing

```bash
git pull --ff-only
python fgw.py mha run run_mha_merge_prefetch_confirm
cat reports/mha/latest/mha_merge_prefetch_confirm_paste.txt
```

Only Q32768/KV4096 and Q32768/KV8192. No kernel, workspace, plan or production
policy change. One compilation round per shape: one shared local kernel and
the prefetch/fixed-R8 merge entries. Timing repeats reuse exactly this build;
no recompilation between repeats or samples.

Six measurement repeats per shape. Before every repeat, execute 100 warmup
calls for each of prefetch/fixed R8/native AscendC, rotating their order; then
synchronize and prime both reusable Event objects outside measurement. Nine
rotated timing triples follow. Immediately before each sample execute ten
warmups for that same call and synchronize. Across every nine triples each
control occupies every order position three times; shift the starting order
between measurement repeats. Fifty-four measured samples/control/shape,
324 total measured samples across both shapes. Initial/sample warmups are not
included in latency. Existing two-seed accuracy checks run before timing.

Full-call Event latency includes ordinary output/workspace allocations and all
kernels; excludes compilation and one-time host metadata upload. This is a
warm-cache benchmark, not a cold-cache or end-to-end host latency measurement.
No slow sample exclusion, best-round selection, or runtime algorithm tuning.
Report aggregate medians over all 54 samples, every repeat's medians/ratios,
win counts and ratio ranges, and every raw sample (compact display rounded to
three decimals; JSON retains full precision). Repeats share a build and device
state and should not be treated as independent statistical experiments.

Every repeat is saved immediately, with warmup/measurement progress in the
terminal. Failures retain completed repeats and are marked incomplete. Linux
resource supervisor remains active. Local tests check balanced timing order,
report completeness/no filtering and shell syntax; final NPU behavior requires
running the command above. Initial warming/event priming may reduce the first
sample effect but cannot guarantee absence of shared-device or clock variation.
