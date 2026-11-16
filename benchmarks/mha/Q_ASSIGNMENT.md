# Balanced Q-tile assignment comparison

Run `python fgw.py mha run run_mha_q_assignment`, then return
`reports/mha/latest/mha_q_assignment_paste.txt`. Reports and generated sources
remain under a fresh mha_q_assignment.* directory.

Reference: [PR3896 pinned core](https://github.com/tile-ai/tilelang/blob/49a51226eb4068d5c0bf205cee886d2a7dcf0b05/examples/ascend/flash_attention/core.py).
Its explicit-block path assigns consecutive Q tiles per block and requires
divisible tile counts. Here quotient/remainder ranges extend that idea to
nondivisible counts without reducing the number of active cores.

Let W be total query-tile/split tasks and G=min(W,28). For iteration i and block b:

- Strided (existing): task=b+i*G.
- Contiguous: task=b*(W//G)+min(b,W%G)+i; valid iff i<W//G+(b<W%G).

Both assign exactly W distinct tasks; every block has the same number of tasks
in either mode. The flattened task-to-request/head/query/split mapping is retained;
the B1H1 split1 target is directly a Q-tile distribution comparison. Partial
tasks use the identical flattened partition convention. No arithmetic, data layout,
tile dimensions, manual stage arrangement, Fast Math setting, split selection,
merge or persistent core cap is changed. After the validated comparison, the user
chose contiguous as the restricted Cube default despite near-equal performance.
The benchmark still explicitly requests both modes. The benchmark
first selects frozen-model splits using default v7, then fixes that same number
for contiguous. It does not repeat the declined NSA schedule experiment.

Smoke uses Q129/KV512 and Q4096/KV8192. The full suite repeats the previous
56-case B1H1/BF16/D128/page128/noncausal Q/KV grid up to8192. Boundary coverage
adds Q3456/3457/3584/3585/4095/4097/7168/7169/8191 at KV8192, exercising
Q tails and the 28/29/56/57-task region. Each shape validates seeds7/19 against
the chunked CPU reference before six rounds of rotating/reversing three-way
Event timing (strided/contiguous/native FIA v2). Entire device callables are
timed, including Q preparation, allocations and any split/merge. Ratios
strided_us/contiguous_us and AscendC_us/contiguous_us above1 favor contiguous.

When W<=G the mapping is identical; these are control cases, identified by
`mapping_changes=false`. Generated-source equality alone does not establish
identical final binaries. Local checks verify mapping coverage/load balance and
structural/syntax properties, not NPU compilation, precision or performance.
