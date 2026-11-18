# Split-first local and merge/direct overlap experiment

On main, run:

```bash
git pull --ff-only
python fgw.py mha run run_mha_split_overlap
cat reports/mha/latest/mha_split_overlap_paste.txt
```

Only Q32768/KV4096 and Q32768/KV8192, B1H1 BF16 D128 noncausal page128.
The retained original raw+SIMD baseline is unchanged. No register-row softmax,
normalized reference, reciprocal merge or old v7 control is compiled.

Each existing per-program task list is partitioned into split fragments first
and complete direct Q tiles second. Ownership, KV intervals, partial slots,
merge table and original task-buffer shapes remain unchanged. Target plans
have 28 programs, each with one fragment followed by nine direct tiles:
28 fragments for four split Q tiles and 252 complete Q tiles.

The experiment compiles a second host adapter from the baseline local PrimFunc
with `out_idx=[]`. It uses caller-allocated Parts, max/sum stats and final O;
the generated device source must match the allocating baseline byte for byte.
The same in-place local kernel is launched twice with the two task tables.
All controls use the original raw SIMD division merge. There is no device
flag polling, global barrier, atomic work queue or new compute implementation.

Four controls are measured:

* `static`: original allocating local, then merge.
* `two_phase_serial`: split local, direct local, then merge on caller stream.
* `two_phase_parallel`: split local on caller stream, record ready event;
  side stream waits, performs merge, records done; caller stream performs
  direct local and waits for done before returning the final output view.
* `ascendc`: installed native FIA v2.

Split/merge output rows and direct output rows are disjoint. Parts, stats and
O record their side-stream use to protect allocator lifetime. Metadata and
streams/events are prepared once and retained. This prototype supports a
single host thread/caller stream, not concurrent calls on the same callable.
Full-call Event timing includes output/workspace allocations and all fork/join
operations. Compilation, task-table uploads and initial stream/event setup
are excluded; the native FIA API keeps its installed internal behavior.

Two seeds must pass all four controls against the CPU reference before timing.
One build per control is reused for six measurement repeats, each with 100
warmups/control, Event priming and eight rotated four-control groups (each
position twice), plus ten immediate warmups before every sample. Outputs are
checked again after every repeat outside Event samples. All 48 samples/control
are retained, with aggregate and per-repeat ratios. Serial versus parallel
measures net overlap benefit; original versus serial measures net phase-split
cost, including its different host allocation path, not pure device launch cost.

After both Event tests finish and are saved, separate processes collect five
parallel calls per shape with Level1 CPU/NPU trace. No profiler is active in
Event measurements. Trace export has an 8 GiB default process-tree RSS budget;
Event workers retain the usual 4 GiB budget. Raw profiler data and exported
JSON/CSV are retained in the private run directory. Trace errors do not erase
completed timing reports. Open exported trace JSON in Perfetto if needed.

The parser accepts only one aggregate `cat=kernel`, `ph=X` Chrome interval per
kernel call: ten local intervals and five merge intervals on distinct lanes.
It checks partial-to-merge ordering and reports direct/merge interval
intersection. Unknown layouts are marked `unrecognized_layout`, not zero
overlap. Observed timeline overlap does not establish a performance benefit;
Vector contention and extra launches/events may cancel the gain.

Local host tests verify partition ownership/slot coverage, balanced timing
order, reporting and trace intersection/rejection. Official Ascend950PR_9579
CAModel verifies sequential split-first/direct/merge using the unchanged local
source at Q257/KV512, including direct tail output and raw m/l/A states
(max output error 0.001788). It does not validate NPU multi-stream behavior.
The local TileLang dev compiler differs from the server a7cb753d; real NPU
precision, event reuse, in-place adapter source matching, concurrency and
speedup remain to be verified by this run. No default promotion is made.
