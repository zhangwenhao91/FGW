"""Independent dense attention vs raw (A, max, sum) partition state oracle."""
import math
import unittest


class RawPartialStateTests(unittest.TestCase):
    def test_raw_merge_matches_dense_with_unequal_and_empty_partitions(self):
        keys = [(1., 2.), (-3., 4.), (5., -6.), (2., -1.), (-4., -2.), (3., 5.)]
        values = [(2., -1.), (4., 3.), (-2., 5.), (7., -3.), (1., 6.), (-5., 8.)]
        for query in ((1., -2.), (-3., 4.), (100., -100.)):
            scores = [sum(a*b for a, b in zip(query, key)) / math.sqrt(2) for key in keys]
            peak = max(scores)
            weights = [math.exp(x - peak) for x in scores]
            dense = [sum(w*v[d] for w, v in zip(weights, values))/sum(weights) for d in range(2)]
            states = []
            for begin, end in ((0, 2), (2, 5), (5, 6)):
                m = max(scores[begin:end])
                w = [math.exp(x-m) for x in scores[begin:end]]
                states.append(([sum(a*v[d] for a, v in zip(w, values[begin:end])) for d in range(2)], m, sum(w)))
            states.append(([0., 0.], -1e30, 0.))
            maximum = max(m for _, m, total in states if total > 0)
            factors = [math.exp(m-maximum) if total > 0 else 0. for _, m, total in states]
            denominator = sum(f*total for f, (_, _, total) in zip(factors, states))
            raw = [sum(f*a[d] for f, (a, _, _) in zip(factors, states))/denominator for d in range(2)]
            for actual, expected in zip(raw, dense):
                self.assertAlmostEqual(actual, expected, places=12)


if __name__ == '__main__':
    unittest.main()
