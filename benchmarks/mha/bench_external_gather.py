"""Upstream Gather reuse: exact cache check, MHA correctness, paired full timings."""
import sys
from benchmarks.mha import bench_cube_ascendc as benchmark


def main():
    # Exercise both upstream fast-path contracts in addition to the existing
    # arbitrary-page, dtype, Q/KV-tail and causal test matrix.
    benchmark.STANDARD = benchmark.STANDARD + [
        ("gather_fast_page16", 2, 17, 4, 256, 16, 16, (255, 129), False),
        ("gather_fast_page64", 2, 16, 3, 192, 64, 4, (256, 192), False),
        ("gather_zero_mixed", 2, 35, 2, 64, 128, 4, (0, 259), True),
    ]
    if "--external-gather" not in sys.argv:
        sys.argv.append("--external-gather")
    return benchmark.main()


if __name__ == "__main__":
    raise SystemExit(main())
