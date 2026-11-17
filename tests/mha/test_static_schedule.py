"""CPU-only schedule coverage and independent attention/state-merge oracle."""

import math
import random
import unittest
from collections import Counter
from operators.mha.static_schedule import make_plan, make_q_plan, choose_plan, validate_plan


class StaticScheduleTests(unittest.TestCase):
    def test_whole_q_assignment_and_cost_selection(self):
        for q in (1, 17, 129, 3712, 32768):
            plan = make_q_plan(q, 512)
            self.assertFalse(plan.merges)
            self.assertEqual(plan.partial_count, 0)
            tiles = [tile for program in plan.programs for tile, _, _, _ in program]
            self.assertEqual(tiles, list(range((q + 127) // 128)))
            self.assertTrue(all(first == 0 and count == 4 and slot == -1
                                for program in plan.programs for _, first, count, slot in program))
        # A short workload must not pay a merge launch merely to occupy cores.
        self.assertFalse(choose_plan(128, 512).merges)
        # Long unbalanced work should retain useful split-KV parallelism.
        self.assertTrue(choose_plan(8192, 32768).merges)
        self.assertEqual(len(choose_plan(8192, 32768).programs), 28)

    def test_selected_grid_coverage(self):
        from benchmarks.mha.summarize_clean_grid import QS, KVS
        for q in (*QS, 17, 129, 3712):
            for kv in KVS:
                plan = choose_plan(q, kv)
                validate_plan(plan)
                self.assertLessEqual(len(plan.programs), 28)
                # Explicitly check exact interval coverage for each Q tile.
                intervals = {}
                for program in plan.programs:
                    for tile, first, count, _ in program:
                        intervals.setdefault(tile, []).append((first, first + count))
                self.assertEqual(set(intervals), set(range((q + 127) // 128)))
                for spans in intervals.values():
                    cursor = 0
                    for first, end in sorted(spans):
                        self.assertEqual(first, cursor)
                        cursor = end
                    self.assertEqual(cursor, kv // 128)

    def test_coverage_and_unique_ownership(self):
        for q in (1, 17, 128, 129, 257):
            for kv in (128, 256, 512, 1024):
                for cores in (1, 2, 7, 28):
                    plan = make_plan(q, kv, cores)
                    seen = Counter(
                        (tile, k)
                        for program in plan.programs
                        for tile, first, count, _ in program
                        for k in range(first, first + count)
                    )
                    self.assertEqual(
                        set(seen), {(t, k) for t in range((q + 127) // 128) for k in range(kv // 128)}
                    )
                    self.assertTrue(all(n == 1 for n in seen.values()))
                    validate_plan(plan)

    def test_target_shapes_and_balanced_no_merge(self):
        plan = make_plan(8192, 32768)
        self.assertEqual((plan.partial_count, len(plan.merges)), (54, 27))
        self.assertEqual((plan.describe()["min_kv_blocks"], plan.describe()["max_kv_blocks"]), (585, 586))
        for q in (3584, 7168, 17920, 28672):
            plan = make_plan(q, 32768)
            self.assertEqual((plan.partial_count, len(plan.merges)), (0, 0))
            self.assertEqual(plan.describe()["min_kv_blocks"], plan.describe()["max_kv_blocks"])

    def test_split_state_merge_matches_dense_attention(self):
        # Three Q tiles on two cores: full+fragment in each, only middle Q tile merges.
        qlen, kv = 257, 512
        plan = make_plan(qlen, kv, 2)
        self.assertEqual(len(plan.merges), 1)
        rng = random.Random(7)
        q = [[rng.uniform(-1, 1) for _ in range(2)] for _ in range(qlen)]
        k = [[rng.uniform(-1, 1) for _ in range(2)] for _ in range(kv)]
        v = [[rng.uniform(-1, 1) for _ in range(2)] for _ in range(kv)]

        def attention(query, first, count):
            scores = [
                sum(a * b for a, b in zip(query, key)) / math.sqrt(2) for key in k[first : first + count]
            ]
            maximum = max(scores)
            weights = [math.exp(x - maximum) for x in scores]
            total = sum(weights)
            return [
                sum(w * value[d] for w, value in zip(weights, v[first : first + count])) / total
                for d in range(2)
            ], maximum + math.log(total)

        outputs, partials = {}, {}
        for program in plan.programs:
            for tile, first, count, slot in program:
                values = [
                    attention(qi, first * 128, count * 128)
                    for qi in q[tile * 128 : min((tile + 1) * 128, qlen)]
                ]
                if slot < 0:
                    for row, (value, _) in enumerate(values):
                        outputs[tile * 128 + row] = value
                else:
                    partials[slot] = values
        for tile, first, count in plan.merges:
            for row in range(min(128, qlen - tile * 128)):
                fragments = [partials[s][row] for s in range(first, first + count)]
                maximum = max(lse for _, lse in fragments)
                weights = [math.exp(lse - maximum) for _, lse in fragments]
                outputs[tile * 128 + row] = [
                    sum(w * value[d] for w, (value, _) in zip(weights, fragments)) / sum(weights)
                    for d in range(2)
                ]
        for row in range(qlen):
            expected, _ = attention(q[row], 0, kv)
            for actual, reference in zip(outputs[row], expected):
                self.assertAlmostEqual(actual, reference, places=12)


if __name__ == "__main__":
    unittest.main()
