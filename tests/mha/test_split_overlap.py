"""Partition ownership and benchmark contract without importing torch/NPU."""
import copy
import unittest
import json
import tempfile
from pathlib import Path
from collections import Counter
from benchmarks.mha.bench_static_host import paired_order
from operators.mha.static_schedule import choose_stream_plan,make_plan
from operators.mha.phase_schedule import split_tables
from benchmarks.mha.summarize_split_overlap import validate,LABELS
from benchmarks.mha.profile_split_overlap import analyze_traces


class SplitOverlapTests(unittest.TestCase):
    def test_four_control_order_is_balanced(self):
        for repeat in range(6):
            for position in range(4):
                self.assertEqual(Counter(paired_order(LABELS,pair,0,repeat)[position] for pair in range(8)),
                                 Counter(dict.fromkeys(LABELS,2)))

    def test_partition_preserves_every_task_owner_and_slot(self):
        for plan in (choose_stream_plan(32768,4096),choose_stream_plan(32768,8192),make_plan(257,512,2)):
            split,direct = split_tables(plan)
            self.assertEqual(sum(split[0])+sum(direct[0]),sum(map(len,plan.programs)))
            for pid,original in enumerate(plan.programs):
                parts = split[1][pid][:split[0][pid]]
                whole = direct[1][pid][:direct[0][pid]]
                self.assertEqual(parts,tuple(t for t in original if t[3]>=0))
                self.assertEqual(whole,tuple(t for t in original if t[3]<0))
                self.assertEqual(len(split[1][pid]),plan.max_segments)
                self.assertEqual(len(direct[1][pid]),plan.max_segments)
                self.assertTrue({t[0] for t in parts}.isdisjoint({t[0] for t in whole}))
            if plan.q==32768:
                self.assertEqual(split[0],(1,)*28)
                self.assertEqual(direct[0],(9,)*28)

    def test_no_split_rejected(self):
        with self.assertRaises(ValueError):
            split_tables(choose_stream_plan(32768,512))

    def test_trace_intersection_and_unknown_layout(self):
        events=[]
        for index in range(5):
            for name,ts,dur,tid in (('local_split_kernel',0,2,1),('local_split_kernel',3,10,1),('merge_simd_kernel',2,4,2)):
                events.append(dict(name=name,ts=ts+20*index,dur=dur,tid=tid,cat='kernel',ph='X'))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'trace.json'
            path.write_text(json.dumps(dict(traceEvents=events)))
            result=analyze_traces([path])
            self.assertEqual(result['status'],'recognized')
            self.assertEqual(result['median_overlap_us'],3)
            self.assertEqual(result['fraction_of_merge_interval_overlapped'],.75)
            self.assertEqual(result['calls_with_overlap'],5)
            path.write_text(json.dumps(dict(traceEvents=events[:-1])))
            self.assertEqual(analyze_traces([path])['status'],'unrecognized_layout')

    def test_cann_hardware_pid_and_decimal_timestamp(self):
        events=[dict(name='process_name',ph='M',pid=99,args={'name':'Ascend Hardware'})]
        for name,ts,dur,tid,kind in (('local_split_kernel','1795028939172582.000',.5,61,'MIX_AIC'),
                                   ('local_split_kernel','1795028939172582.757',228.361,61,'MIX_AIC'),
                                   ('merge_simd_kernel','1795028939172583.477',231.436,59,'AI_VECTOR_CORE')):
            events.append(dict(name=name,ph='X',pid=99,tid=tid,ts=ts,dur=dur,args={'Task Type':kind}))
        # A same-name host event must never enter the hardware intervals.
        events.append(dict(events[1],pid=100))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'trace.json'
            path.write_text(json.dumps(events))
            result=analyze_traces([path],calls=1)
            self.assertEqual(result['status'],'recognized')
            self.assertAlmostEqual(result['overlap_us'][0],227.641,places=6)
            self.assertAlmostEqual(result['merge_end_minus_direct_end_us'][0],3.795,places=6)

    def test_complete_four_control_contract_and_source_equality(self):
        settings=dict(rounds=1,timing_rounds=6,pairs=8,initial_warmup=100,warmup=10,
                      split_overlap=True,split_overlap_trace=False,raw_baseline_only=True,
                      only_static_ascendc=True,merge_policy='rawsimd',plan_policy='stream',row_softmax=False)
        report=dict(status='passed',correctness='passed',settings=settings,timer_primed=True,summary=[{}],
                    cases=[dict(builds=[dict(source_sha256=dict(static_local='same',two_phase_inplace_local='same'))],
                                repeat_accuracy=[dict(repeat=i,label=l) for i in range(1,7) for l in LABELS],
                                accuracy=[dict(seed=s,label=l) for s in (7,19) for l in LABELS])],
                    measurement_rounds=[dict(compile_round=1,measurement_round=i+1,
                                             samples_us={label:[10.]*8 for label in LABELS}) for i in range(6)])
        self.assertEqual(len(validate(report)),6)
        broken=copy.deepcopy(report)
        broken['cases'][0]['builds'][0]['source_sha256']['two_phase_inplace_local']='different'
        with self.assertRaises(ValueError):
            validate(broken)
        broken=copy.deepcopy(report)
        broken['measurement_rounds'][0]['samples_us']['two_phase_parallel'].pop()
        with self.assertRaises(ValueError):
            validate(broken)


if __name__=='__main__':
    unittest.main()
