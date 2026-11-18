"""The experimental summary must not accept another variant's report."""
import json
from pathlib import Path
import tempfile
import unittest
from benchmarks.mha.summarize_merge_recip import summarize


class RowSoftmaxReportTests(unittest.TestCase):
    def test_variant_contract_and_complete_samples(self):
        repeats = [dict(compile_round=1, measurement_round=i+1,
                        samples_us=dict(static=[20.] * 9, raw_ref=[21.] * 9, ascendc=[22.] * 9),
                        median_us=dict(static=20., raw_ref=21., ascendc=22.)) for i in range(6)]
        repeats[0]['samples_us']['static'][0] = 1000.
        report = dict(status='passed', correctness='passed', timer_primed=True,
                      softmax_variant='register_rows2_v2',
                      settings=dict(rounds=1, timing_rounds=6, pairs=9, initial_warmup=100,
                                    warmup=10, row_softmax=True, merge_policy='rawsimd'),
                      summary=[dict(static_builds=1, reference_builds=1)], measurement_rounds=repeats)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'run'
            directory.mkdir()
            for kv in (4096, 8192):
                (directory / f'kv{kv}.json').write_text(json.dumps(report))
            output = directory.parent / 'mha_row_softmax_paste.txt'
            summarize(directory, row_softmax=True)
            text = output.read_text()
            self.assertIn('status=passed completed=2/2', text)
            self.assertIn('1000.0', text)
            self.assertIn('raw_ref/row_softmax overall=1.050000', text)
            report['softmax_variant'] = 'register_rows1_v1'
            (directory / 'kv8192.json').write_text(json.dumps(report))
            summarize(directory, row_softmax=True)
            self.assertIn('incomplete_or_error completed=1/2', output.read_text())


if __name__ == '__main__':
    unittest.main()
