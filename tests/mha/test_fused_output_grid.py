"""Full grid reporting and source-checked resume; no NPU imports."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from benchmarks.mha.summarize_fused_output_grid import ROOT,QS,KVS,current_sources,load_case,summarize
from tests.mha.test_fused_output_report import fixture


def grid_fixture(q,kv):
    report=fixture(kv)
    report['summary'][0]['plan']['Q']=q
    report.update(benchmark_sha256=hashlib.sha256((ROOT/'benchmarks/mha/bench_static_host.py').read_bytes()).hexdigest(),
                  source_sha256=current_sources(),
                  model_sha256=hashlib.sha256((ROOT/'operators/mha/split_cost_model.json').read_bytes()).hexdigest(),
                  device='same test device',versions={'torch':'same test version'})
    return report


class FusedOutputGridTests(unittest.TestCase):
    def test_full_grid_and_regression_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'run'; directory.mkdir()
            for q in QS:
                for kv in KVS:
                    report=grid_fixture(q,kv)
                    if q==64 and kv==512:
                        for repeat in report['measurement_rounds']:
                            repeat['samples_us']['static']=[23.]*9
                    (directory/f'q{q}_kv{kv}.json').write_text(json.dumps(report))
            summarize(directory)
            text=(directory.parent/'mha_fused_output_grid_paste.txt').read_text()
            self.assertIn('status=passed completed=80/80',text)
            self.assertIn('original_over_fused: geomean=',text)
            self.assertIn('MATRIX AscendC_over_fused',text)
            self.assertIn('all_6_slower_than_original=1',text)
            summary=json.loads((directory/'summary.json').read_text())
            self.assertEqual(len(summary['rows']),80)
            self.assertEqual(summary['rows'][0]['ratios']['raw_ref']['wins'],0)

    def test_resume_rejects_changed_source_and_incomplete_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'case.json'
            report=grid_fixture(64,512)
            path.write_text(json.dumps(report))
            self.assertEqual(load_case(path,64,512)[0]['Q'],64)
            report['benchmark_sha256']='changed'
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'benchmark source changed'):
                load_case(path,64,512)
            report=grid_fixture(64,512)
            report['measurement_rounds'][0]['samples_us']['static'].pop()
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'samples'):
                load_case(path,64,512)

    def test_mixed_environment_is_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'run'; directory.mkdir()
            for q in (64,128):
                report=grid_fixture(q,512)
                if q==128: report['versions']['torch']='different'
                (directory/f'q{q}_kv512.json').write_text(json.dumps(report))
            summarize(directory)
            text=(directory.parent/'mha_fused_output_grid_paste.txt').read_text()
            self.assertIn('completed=1/80',text)
            self.assertIn('mixed device/software versions',text)


if __name__=='__main__':
    unittest.main()
