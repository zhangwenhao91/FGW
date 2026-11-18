# Register-resident score rows

Run `python fgw.py mha run run_mha_row_softmax` and return
`reports/mha/latest/mha_row_softmax_paste.txt`.

The experiment compares full calls for the row-softmax candidate, the current
raw+SIMD baseline, and installed AscendC FIA at B1H1 BF16 D128 page128,
noncausal, Q32768 and KV4096/8192. The candidate is opt-in with
`--row-softmax`; existing defaults are unchanged.

The current candidate is `register_rows2_v2`: it replaces the single-row
experiment in the same opt-in path. The report records this identifier and
the summarizer rejects older reports. The previous single-row candidate
increased full-call latency by 8.1%/9.7% for KV4096/8192; it was not promoted.

Only the local softmax macro changes. It loads two score rows with DINTLV,
interleaves their independent scale, max, broadcast and exp operations, then
calculates the row sums, BF16 conversion and NZ packing while all four scaled
even/odd score vectors remain in registers. Both reduced maxima are available
before either row's exp calculation. The original all-64-rows alpha and
denominator update remains. The extra live vectors can increase register
pressure; real-NPU measurements must determine the tradeoff.
This removes the full scaled-score UB store/reload and reduces the explicit
internal VST_VLD barriers from three to one. Maximum reduction pairs change
from the two contiguous halves to even/odd lanes; the finite-input maximum
is unchanged. The exp/sum lane grouping and PV accumulation order remain.

The host plan, task tables, Cube QK/PV operations, raw partial contract and
merge algorithm are unchanged. Both timed TL variants use the same compiled
merge object. There are two local builds per shape, one per variant; compiled
objects are reused for six measurement repeats. Each repeat warms every
control 100 times, primes Events, and takes nine balanced rotated triples
with ten immediate warmups before each Event sample. All samples are kept.
Full calls include normal preparation, allocations and merge. Compilation
and initial task-table upload are excluded. Both seeds 7 and 19 must pass
the CPU-reference accuracy check for all three implementations before timing.

Local lowering and official Ascend950PR_9579 CAModel checks use the local
TileLang development compiler, which differs from the server's a7cb753d
installation. These checks do not establish real-NPU performance or replace
the server accuracy checks. Performance promotion requires the paired results.

Validation before publishing: twelve host scheduling/report tests passed;
local lowering passed for Q129/KV512, Q257/KV512 and both target shapes;
candidate/reference merge sources matched. Official CAModel full local plus
selective merge at Q257/KV512 with two programs passed against the FP32
reference (maximum absolute output error 0.001788), including raw partials,
both max/sum planes, direct output and a Q tail. Python and Bash syntax checks
passed. The simulator result is a correctness check, not a speedup measurement.
