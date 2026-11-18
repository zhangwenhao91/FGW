"""Only the fused epilogue triple with unchanged merge is accepted."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from benchmarks.mha.summarize_fused_output import POLICY,KVS,LABELS,validate,summarize


def fixture(kv):
    return dict(status='passed',correctness='passed',output_variant='simd_div_cast_v1',
                settings=dict(POLICY),timer_primed=True,
                summary=[dict(plan=dict(Q=32768,KV=kv),static_builds=1,reference_builds=1)],
                cases=[dict(accuracy=[dict(seed=s,label=l) for s in (7,19) for l in LABELS],
                            repeat_accuracy=[dict(repeat=i,label=l) for i in range(1,7) for l in LABELS],
                            builds=[dict(source_sha256=dict(static_merge='same',raw_simd_ref_merge='same'))])],
                measurement_rounds=[dict(compile_round=1,measurement_round=i+1,
                                         samples_us=dict(static=[20.]*9,raw_ref=[21.]*9,ascendc=[22.]*9)) for i in range(6)])


class FusedOutputReportTests(unittest.TestCase):
    def test_complete_summary_and_untrimmed_samples(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'run'
            directory.mkdir()
            for kv in KVS:
                report=fixture(kv)
                report['measurement_rounds'][0]['samples_us']['static'][0]=1000.
                (directory/f'kv{kv}.json').write_text(json.dumps(report))
                self.assertEqual(validate(report,kv)[0]['samples_us']['static'][0],1000.)
            summarize(directory)
            text=(directory.parent/'mha_fused_output_paste.txt').read_text()
            self.assertIn('status=passed completed=5/5',text)
            self.assertIn('raw_ref/fused overall=1.050000',text)
            self.assertIn('ascendc/fused overall=1.100000',text)

    def test_reject_mixed_experiment_merge_or_missing_checks(self):
        for mutation in ('row_softmax','variant','merge','repeat_accuracy','samples'):
            report=copy.deepcopy(fixture(512))
            if mutation=='row_softmax': report['settings']['row_softmax']=True
            elif mutation=='variant': report['output_variant']='other'
            elif mutation=='merge': report['cases'][0]['builds'][0]['source_sha256']['static_merge']='different'
            elif mutation=='repeat_accuracy': report['cases'][0]['repeat_accuracy'].pop()
            else: report['measurement_rounds'][0]['samples_us']['static'].pop()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                validate(report,512)


if __name__=='__main__':
    unittest.main()
