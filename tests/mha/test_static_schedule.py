"""CPU-only schedule coverage and independent attention/state-merge oracle."""

import math
import random
import unittest
from collections import Counter
from operators.mha.static_schedule import make_plan, make_q_plan, choose_plan, validate_plan, make_tail_plan, make_stream_plan, choose_stream_plan, make_merge_tasks, make_block_merge_tasks


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

    def test_weighted_merge_exact_row_ownership(self):
        from benchmarks.mha.summarize_clean_grid import QS, KVS
        for q in (*QS, 1, 129, 3712):
            for kv in KVS:
                plan = choose_stream_plan(q, kv)
                tasks = make_merge_tasks(plan)
                self.assertLessEqual(len(tasks), 56)
                expected = {(tile, r) for tile, _, _ in plan.merges
                            for r in range(min(128, q - tile * 128))}
                seen = Counter((tile, r) for tile, _, _, begin, rows in tasks
                               for r in range(begin, begin + rows))
                self.assertEqual(set(seen), expected)
                self.assertTrue(all(n == 1 for n in seen.values()))
                groups = {tile: (first, count) for tile, first, count in plan.merges}
                for tile, first, count, begin, rows in tasks:
                    self.assertEqual(groups[tile], (first, count))
                    self.assertGreater(rows, 0)
                    self.assertLessEqual(begin + rows, min(128, q - tile * 128))
        plan = make_tail_plan(32768, 4096, pieces=7)
        tasks = make_merge_tasks(plan)
        self.assertEqual(len(tasks), 52)
        self.assertEqual(max(task[4] for task in tasks), 10)

    def test_grouped_dma_chunk_ownership(self):
        from benchmarks.mha.summarize_clean_grid import QS, KVS
        for q in (*QS, 129, 3712):
            for kv in KVS:
                plan = choose_stream_plan(q, kv)
                expected = {(t, r) for t, _, _ in plan.merges for r in range(min(128, q - t * 128))}
                for budget in (28, 56):
                    workers = make_block_merge_tasks(plan, budget)
                    seen = Counter((t, r) for worker in workers for t, _, _, begin, rows in worker
                                   for r in range(begin, begin + rows))
                    self.assertEqual(set(seen), expected)
                    self.assertTrue(all(n == 1 for n in seen.values()))
                    self.assertLessEqual(len(workers), budget)
                    self.assertTrue(all(worker for worker in workers))
                    self.assertTrue(all(begin % 8 == 0 and 0 < rows <= 8
                                        for worker in workers for _, _, _, begin, rows in worker))

    def test_stream_policy_coverage_and_parallelism(self):
        from benchmarks.mha.summarize_clean_grid import QS, KVS
        for q in (*QS, 1, 17, 129, 3712):
            for kv in KVS:
                validate_plan(choose_stream_plan(q, kv))
                for cores in (1, 2, 7, 27, 28):
                    validate_plan(make_stream_plan(q, kv, cores))
        self.assertGreater(len(choose_stream_plan(64, 32768).programs), 1)
        self.assertFalse(choose_stream_plan(16384, 1024).merges)
        self.assertFalse(choose_stream_plan(128, 512).merges)

    def test_tail_wave_coverage_and_fragment_count(self):
        for kv in (4096, 8192):
            plan = make_tail_plan(32768, kv)
            validate_plan(plan)
            self.assertEqual((len(plan.merges), plan.partial_count), (4, 28))
            self.assertEqual({tile for tile, _, _ in plan.merges}, set(range(252, 256)))
            self.assertEqual(plan.describe()['direct_q_tiles'], 252)
            self.assertLessEqual(plan.describe()['max_kv_blocks'] - plan.describe()['min_kv_blocks'], 1)
        for q in (1, 129, 3584, 3712, 4096, 8192, 32768):
            for kv in (128, 256, 512, 4096, 8192):
                for pieces in (None, 2, 4, 7):
                    validate_plan(make_tail_plan(q, kv, pieces=pieces))

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
        for plan in (make_plan(257, 512, 2), make_tail_plan(257, 512, 2), make_stream_plan(257, 512, 2)):
            self.assertEqual(len(plan.merges), 1)
            self._check_attention(plan)
            self._check_attention(plan, weighted=True)
            self._check_attention(plan, weighted="grouped")

    def _check_attention(self, plan, weighted=False):
        qlen, kv = plan.q, plan.kv
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
        merge_tasks = make_merge_tasks(plan) if weighted else tuple(
            (tile, first, count, 0, min(128, qlen - tile * 128)) for tile, first, count in plan.merges)
        if weighted == "grouped":
            merge_tasks = tuple(task for worker in make_block_merge_tasks(plan) for task in worker)
        for tile, first, count, begin, rows in merge_tasks:
            for row in range(begin, begin + rows):
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
