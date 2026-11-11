# PR3896 upstream snapshot

Files are copied byte-for-byte from tile-ai/tilelang PR3896 head
49a51226eb4068d5c0bf205cee886d2a7dcf0b05. See SOURCE.json for source URLs and SHA256.
MIT license: ../licenses/TILELANG_LICENSE.

core.py is imported directly by the comparison benchmark. No kernel edits.
example_mha.py is preserved as provenance for the original defaults:
dynamic Q/KV when SEQ_LEN=None, FP32 output, fast-math enabled.
The BF16 comparison uses core.flash_attention_fwd's existing out_dtype option.
