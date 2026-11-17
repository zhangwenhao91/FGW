# Targeted split and merge cost measurement

```bash
python fgw.py mha run run_mha_split_breakdown
cat reports/mha/latest/mha_split_breakdown_paste.txt
```

Cases: Q8192/KV32768, Q4096/KV8192 and Q2048/KV24576, crossed with split1/2/4/8.
B1H1 BF16 D128/page128, noncausal, full cache, shuffled/repeated physical pages.
Uses the cleaned v7 module. No operator, split model or default policy is changed.
The default selector is run per shape/round, then its callable is reused for that
split; all other splits are explicit overrides. Thus there are four builds per
round and two rounds, 24 independent TL builds total, with builder LRU cleared
and TileLang cache disabled. Each freshly built variant is checked at seeds7/19
against chunked FP32 CPU attention, rtol .01/atol .004.

Every callable receives five immediate warmups and synchronization before one
Event sample; seven samples per round, rotating/reversing order across variants
and components. Installed AscendC FIA v2 is checked and measured in the same way.
No compile, H2D or CPU reference time is inside Event timing.

- `full`: normal public compiled callable, including Q preparation, normal
  allocations and all required kernels.
- `local`: direct kernel at S1; partial kernel at S>1, both on preprepared Q.
  S1 includes final normalization/cast/store, while S>1 writes normalized FP32
  Partial and LSE. They have different output contracts and costs.
- `merge`: S>1 only, reading a fixed correct Partial/LSE fixture; normal output
  allocation is retained. This is an isolated warm-cache measurement.
- `chain`: S>1 only, actually invokes local then merge on preprepared Q, with
  fresh partial allocations and real data dependencies.

The isolated sum `local+merge` is diagnostic, not an estimate replacing measured
`full` or `chain`. Allocation/caching/launch effects can make them differ.
Accuracy checks cover full, fixed-fixture merge and the real chain for S>1;
full and direct local for S1; AscendC is also checked at both seeds each round.

The report retains actual grid/work counts, min/max tasks per program, KV blocks
per task, critical-program KV block upper bounds, Partial/LSE byte counts, source
hashes per fresh build, accuracy records, raw timings, round paired speedups vs
the automatic default and AscendC, and the best full-call median split per case.
Source C++ and full JSON remain in a fresh run directory. The compact paste
preserves partial progress and traceback on failure. Two rounds are a targeted
check; close performance differences need more repeats before changing defaults.

## Compiler scratch and continuation

Compiler temporary files now use each fresh run's `compiler_tmp` directory,
with TMPDIR/TMP/TEMP and Python's cached tempfile directory set before importing
the compiler. A create/remove probe verifies the directory, and it is checked
again before each build. This avoids dependency on global `/tmp`, whose absence
caused the reported ENOENT in tempfile.mkdtemp(). This identifies the failing
filesystem operation, not what removed the original `/tmp` path.

For a future continuation with matching guarded resource policy, from FGW root:

```bash
bash scripts/mha/run_mha_split_breakdown.sh --resume /absolute/path/to/previous-guarded-run/full.json
```

The runner resolves the previous path before changing directories, writes a new
run folder and preserves the original report. Resume validates kernel/model
hashes, versions, device, warmup and sample settings; completed cases/rounds are
kept, while unfinished-round build and accuracy records are discarded and that
round is rerun. Reports from before resource supervision do not have a matching
resource policy and are now rejected before Torch is imported. After reported
server outages, collect crash evidence first; do not resume merely to reproduce.
