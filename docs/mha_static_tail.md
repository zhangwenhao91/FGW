# Selective final-wave split-KV experiment

Run `python fgw.py mha run run_mha_static_tail` after pulling main.
Paste `reports/mha/latest/mha_static_tail_paste.txt`.

Only Q32768/KV4096 and Q32768/KV8192, B1H1 BF16 D128 page128 noncausal.
Each shape has five configurations: current cost-policy R8, final-wave pieces
2/R32, 4/R32, 7/R8 and 7/R32. One build and three rotated paired samples per
configuration, five immediate warmups including AscendC FIA, two accuracy
seeds checked before timing. Ten configurations total, progress printed.

With 256 Q tiles and 28 programs, assign nine whole Q tiles to each program
(252 direct tiles). Split only tiles 252..255. Seven pieces per tail tile gives
28 fragments, four merge tiles, and ten segments per program. Full flattened
balancing uses 54 fragments/27 merge tiles/283 segments at KV4096, while this
plan has 28 fragments/4 merge tiles/280 segments. Tail fragments remain
128-KV-block aligned and differ by at most one KV block in size. Two/four-piece
plans intentionally leave more tail imbalance in exchange for fewer states.
The kernel computation and selective merge are shared with existing plans;
only host metadata changes. Current cost model and default v7 stay unchanged.

The new assignment is not globally contiguous in flattened Q/KV order. The
validator now checks sorted intervals separately for every Q tile, exact
coverage, no overlap, unique partial slots, direct/partial exclusivity, and
matching merge groups. CPU tests cover tails/degenerate/balanced shapes and an
independent dense-attention oracle for both old and new split assignments.
Local Ascend lowering checks cover no-split Q129, mixed Q3712 and both target
shapes with tail fragments. Local TileLang differs from the server; hardware
accuracy and speed still require this run. A lower merge-tile count is not a
speed guarantee: each merged tile now has more parts. Small wins must be
confirmed in a later repeated measurement.
