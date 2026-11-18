# Row reciprocal broadcast vs per-element SIMD division

```bash
git pull --ff-only
python fgw.py mha run run_mha_merge_recip
cat reports/mha/latest/mha_merge_recip_paste.txt
```

Only Q32768/KV4096 and Q32768/KV8192, B1H1 BF16 D128 noncausal page128.
Three controls: raw+SIMD reciprocal candidate, original raw+SIMD division,
installed native AscendC FIA. One shared compiled raw local, same raw A/max/sum
workspace format, plan, R8 merge grid, DMA and online merge updates. Reference
merge is built separately once; both controls use the same host wrapper and
compute function. An identity assertion verifies the local kernel is shared.
No normalization/location, precision mode, prefetch or task-distribution change.

Final VF computes inv[r]=1/max(total[r],1e-30) once per row, stores into the
no-longer-needed old-weight UB scratch, inserts VST_VLD barrier, broadcasts
inv[r] and multiplies both 64-element chunks of D128 before the original BF16
packing/output DMA. This uses ordinary S.vdiv(1,denominator), not an explicitly
selected approximate reciprocal instruction. R8 final division changes from
16 full vector divisions (two per row) to one row-stat vector division; it
adds broadcast multiplies. This operation-count change is not a latency claim.
Separate reciprocal rounding can change BF16 output and must be validated.
All-empty rows retain zero numerator, clamped denominator and zero output.

Each merge compiles once, then six warmed measurement repeats x nine rotated
triples reuse that build. Before each repeat: 100 warmups/control and Event
priming; before each sample: ten immediate warmups. Two correctness seeds
precede timing. Full-call warm-cache Event latency includes ordinary
allocations and all kernels, excludes compilation/one-time metadata upload.
All 54 samples/control are retained, with per-repeat ratios/ranges/win counts.
Progress and Linux process supervision remain active.

CPU schedule/attention/measurement tests pass. Local Ascend lowering covers
mixed full/partial work, target shapes and direct tails. Equivalence checks
show raw local and original raw SIMD division generated C++ are unchanged.
Official Ascend950PR_9579 CAModel fixtures cover full Q257 local+merge with
mixed direct/partial work and direct tail, plus merge-only mixed two/seven
pieces, sentinel/zero-sum states, finite extreme maxima and untouched direct
rows. Local TileLang differs from the server, so NPU accuracy/performance
remain necessary. No default/production policy change.
