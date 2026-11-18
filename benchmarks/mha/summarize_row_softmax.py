"""Summarize the isolated row-softmax versus raw SIMD versus FIA experiment."""
import argparse
from pathlib import Path
from benchmarks.mha.summarize_merge_recip import summarize

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory, row_softmax=True)
