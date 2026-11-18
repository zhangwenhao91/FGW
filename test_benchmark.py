"""Host-only checks for this single frozen-version benchmark."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter
from benchmark import POLICY, LABELS, QS, KVS, verify_source, order, extract, summarize


def fixture(q=64,kv=512):
    return dict(status='passed',correctness='passed',Q=q,KV=kv,
                policy=copy.deepcopy(POLICY),builds=1,timer_primed=True,
                plan=dict(Q=q,KV=kv,partial_count=0,merge_q_tiles=0),
                accuracy=[dict(seed=s,label=l) for s in (7,19) for l in LABELS],
                repeats=[dict(repeat=i+1,samples_us=dict(TL=[20.]*4,AscendC=[22.]*4)) for i in range(3)],
                source=verify_source(),device='test',versions={'torch':'test'})


class BenchmarkTests(unittest.TestCase):
    def test_frozen_source_and_balanced_order(self):
        self.assertTrue(verify_source()['source_commit'].startswith('50fca6d'))
        for repeat in range(3):
            for position in range(2):
                self.assertEqual(Counter(order(pair,repeat)[position] for pair in range(4)),
                                 Counter(dict.fromkeys(LABELS,2)))

    def test_no_sample_discard_and_reject_incomplete_controls(self):
        report = fixture()
        report['repeats'][0]['samples_us']['TL'][0] = 1000.
        result = extract(report,64,512)
        self.assertAlmostEqual(result['speedup'],1.1)
        self.assertEqual(result['samples_us'][0]['samples_us']['TL'][0],1000.)
        for mutation in ('seed','samples','policy','build','shape'):
            broken = copy.deepcopy(report)
            if mutation=='seed':
                broken['accuracy'].pop()
            elif mutation=='samples':
                broken['repeats'][0]['samples_us']['TL'].pop()
            elif mutation=='policy':
                broken['policy']['plan_policy']='stream'
            elif mutation=='build':
                broken['builds']=2
            else:
                broken['plan']['Q']=128
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                extract(broken,64,512)

    def test_full_grid_summary_and_mixed_versions(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)/'run'
            directory.mkdir()
            source=verify_source()
            for q in QS:
                for kv in KVS:
                    report=fixture(q,kv)
                    report['source']=source
                    (directory/f'q{q}_kv{kv}.json').write_text(json.dumps(report))
            summarize(directory)
            output=directory.parent/'mha_50fca6d_grid_paste.txt'
            self.assertIn('status=passed completed=80/80',output.read_text())
            self.assertIn('geomean_speedup=1.100000',output.read_text())
            report=fixture(64,512)
            report['versions']={'torch':'different'}
            (directory/'q64_kv512.json').write_text(json.dumps(report))
            summarize(directory)
            self.assertIn('incomplete_or_error completed=1/80',output.read_text())
            self.assertIn('Mixed devices or versions',output.read_text())


if __name__=='__main__':
    unittest.main()
