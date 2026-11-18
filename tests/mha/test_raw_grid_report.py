import copy
import json
from pathlib import Path
import tempfile
import unittest
from benchmarks.mha.summarize_raw_grid import POLICY, extract, summarize


def fixture():
    return dict(status='passed', correctness='passed', settings=dict(POLICY), timer_primed=True,
                versions={'tilelang': 'test'}, device='test', source_sha256={'candidate': 'same'},
                summary=[dict(plan=dict(Q=64, KV=512, partial_count=2, merge_q_tiles=1),
                              static_builds=1, reference_builds=0)],
                cases=[dict(accuracy=[dict(seed=s, label=l) for s in (7,19) for l in ('static','ascendc')])],
                measurement_rounds=[dict(compile_round=1, measurement_round=i+1,
                                         samples_us=dict(static=[20.]*4, ascendc=[22.]*4)) for i in range(3)])


class RawGridReportTests(unittest.TestCase):
    def test_complete_measurements_and_preserve_slow_samples(self):
        report = fixture()
        report['measurement_rounds'][0]['samples_us']['static'][0] = 1000.
        row = extract(report, 64, 512)
        self.assertAlmostEqual(row['speedup'], 1.1)
        self.assertEqual(row['wins'], 3)
        self.assertEqual(row['samples_us'][0]['static'][0], 1000.)

    def test_reject_wrong_variant_missing_seed_or_samples(self):
        for change in ('variant', 'samples', 'seed', 'shape', 'build'):
            report = fixture()
            if change == 'variant':
                report['settings']['row_softmax'] = True
            elif change == 'samples':
                report['measurement_rounds'][0]['samples_us']['static'].pop()
            elif change == 'seed':
                report['cases'][0]['accuracy'].pop()
            elif change == 'shape':
                report['summary'][0]['plan']['Q'] = 128
            else:
                report['summary'][0]['reference_builds'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                extract(report, 64, 512)

    def test_summary_keeps_missing_and_source_mismatch_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)/'run'
            directory.mkdir()
            report = fixture()
            (directory/'q64_kv512.json').write_text(json.dumps(report))
            other = copy.deepcopy(report)
            other['summary'][0]['plan']['Q'] = 128
            other['source_sha256']['candidate'] = 'changed'
            (directory/'q128_kv512.json').write_text(json.dumps(other))
            summarize(directory)
            text = (directory.parent/'mha_raw_default_grid_paste.txt').read_text()
            self.assertIn('incomplete_or_error completed=1/80', text)
            self.assertIn('source/device/version mismatch', text)
            self.assertIn('64 1.1000 -', text)


if __name__ == '__main__':
    unittest.main()
