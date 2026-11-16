# V transpose location comparison

Opt-in `build_kernel(..., v_transpose_stage="l1_l0")` moves only V's transpose:

- `gm_l1` (existing default): GM-to-L1 transpose, ordinary L1-to-L0B copy.
- `l1_l0` (candidate): original-orientation GM-to-L1 copy, transpose L1-to-L0B.

L1's logical V orientation changes from [D,BC] to [BC,D]. D=BC=128, so its
allocation size and buffering remain unchanged. L0B still receives [D,BC]; the
PV GEMM, operand transpose flag and numerical formula are unchanged. Manual
stage indices, load timing, buffer versions, Q assignment, split, normalization
(default SIMT), BF16 cast and Fast Math settings are held fixed. The same option
is used in direct/partial paths; merge is untouched. General/Vector paths stay
unchanged. This is not a rerun of the NSA two-stage schedule experiment.

Run `python fgw.py mha run run_mha_v_transpose`, then return
`reports/mha/latest/mha_v_transpose_paste.txt`. Smoke covers Q129/KV2048
(split4, tail) and Q4096/KV8192 (direct). Full coverage repeats the previous56
B1H1 BF16 D128 page128 noncausal cases, seeds7/19 and six paired rounds of51
rotating/reversing full-call Event samples. The candidate uses the exact split
selected by the current default. Native AscendC FIA v2 uses identical original
paged inputs. No promotion/refit occurs automatically.

The local official CAModel run B1H1/Q128/KV512/split1/grid1 completed with random
seed7 and shuffled pages. Local compiler versions differ from the NPU machine,
so simulator times do not replace hardware results. Original/normalized traces,
accuracy reports, generated code and host-launch scripts stay in workspace output
directories rather than Git.
