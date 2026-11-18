"""CPU-only checks for confirmation measurement order and complete reporting."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.mha.bench_static_host import paired_order
from benchmarks.mha.summarize_merge_prefetch_confirm import summarize


class PrefetchConfirmationTests(unittest.TestCase):
    def test_balanced_order_and_rotated_repeat_start(self):
        labels = ('static', 'fixed_r8', 'ascendc')
        for repeat in range(6):
            orders = [paired_order(labels, pair, 0, repeat) for pair in range(9)]
            for position in range(3):
                self.assertEqual(Counter(order[position] for order in orders), Counter(dict.fromkeys(labels, 3)))
            self.assertEqual(orders[0][0], labels[repeat % 3])
        labels5 = ('full', 'merge', 'chain', 'ascendc', 'local')
        for repeat in range(6):
            orders = [paired_order(labels5, pair, 0, repeat) for pair in range(10)]
            for position in range(5):
                self.assertEqual(Counter(order[position] for order in orders), Counter(dict.fromkeys(labels5, 2)))
        self.assertEqual(paired_order(('TL', 'AC'), 0), ('TL', 'AC'))
        self.assertEqual(paired_order(('TL', 'AC'), 1), ('AC', 'TL'))

    def test_report_keeps_slow_sample_and_rejects_incomplete_repeat(self):
        settings = dict(rounds=1, timing_rounds=6, pairs=9, initial_warmup=100, warmup=10)
        repeats = []
        for i in range(6):
            samples = dict(static=[20.] * 9, fixed_r8=[21.] * 9, ascendc=[22.] * 9)
            if i == 0:
                samples['static'][0] = 1000.
            repeats.append(dict(compile_round=1, measurement_round=i+1, samples_us=samples,
                                median_us=dict(static=20., fixed_r8=21., ascendc=22.)))
        report = dict(status='passed', correctness='passed', settings=settings, timer_primed=True,
                      summary=[dict(static_builds=1)], measurement_rounds=repeats)
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td) / 'run'
            directory.mkdir()
            for kv in (4096, 8192):
                (directory / f'kv{kv}.json').write_text(json.dumps(report))
            summarize(directory)
            output = directory.parent / 'mha_merge_prefetch_confirm_paste.txt'
            text = output.read_text()
            self.assertIn('status=passed completed=2/2', text)
            self.assertIn('1000.0', text)
            self.assertIn('overall=1.050000', text)
            report['measurement_rounds'] = repeats[:-1]
            (directory / 'kv8192.json').write_text(json.dumps(report))
            summarize(directory)
            self.assertIn('incomplete_or_error completed=1/2', output.read_text())


if __name__ == '__main__':
    unittest.main()
