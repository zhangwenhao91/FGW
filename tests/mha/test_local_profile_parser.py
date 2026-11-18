import struct
import tempfile
import unittest
from pathlib import Path

from tools.mha.parse_local_profile import parse_profile


class LocalProfileParserTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'hash_dic.slice_0').write_text('42:local_split_kernel\n43:other_kernel\n')
        tracks = []
        for seq, hashed in ((7, 42), (8, 43)):
            row = [0] * 8
            row[3], row[5] = seq << 32, hashed
            tracks.append(struct.pack('<8q', *row))
        (self.root / 'task_track.slice_0').write_bytes(b''.join(tracks))

    def record(self, cycles=100, mad=60, aiv=False, seq=7):
        row = [0] * 16
        row[0], row[1], row[2] = seq << 32, cycles, (1 << 31 if aiv else 0)
        row[4], row[5], row[8] = 70, mad, 30
        row[14], row[15] = 1000, 1100
        return struct.pack('<16q', *row)

    def write_records(self, records):
        (self.root / 'ffts_profile.slice_0').write_bytes(b''.join(records))

    def test_weighted_counters_and_kernel_filter(self):
        self.write_records([self.record(), self.record(cycles=200), self.record(aiv=True),
                            self.record(seq=8)])
        (self.root / 'ffts_profile.slice_1.done').write_bytes(b'incomplete marker')
        result = parse_profile(self.root)
        self.assertEqual(result['task_groups'], 1)
        aic, aiv = result['aggregate']['aic'], result['aggregate']['aiv']
        self.assertEqual(aic['records'], 2)
        self.assertAlmostEqual(aic['pipes']['mad']['cycle_weighted'], .4)
        self.assertAlmostEqual(aic['record_duration_us']['median'], .1)
        self.assertEqual(aiv['records'], 1)
        self.assertAlmostEqual(aiv['pipes']['vec']['cycle_weighted'], .7)

    def test_missing_aiv_rejected(self):
        self.write_records([self.record()])
        with self.assertRaisesRegex(ValueError, 'both AIC and AIV'):
            parse_profile(self.root)

    def test_unmatched_kernel_rejected(self):
        self.write_records([self.record(seq=8)])
        with self.assertRaisesRegex(ValueError, 'other_kernel'):
            parse_profile(self.root)

    def test_unsupported_layout_rejected(self):
        self.write_records([self.record(mad=200), self.record(aiv=True)])
        with self.assertRaisesRegex(ValueError, 'ratio outside'):
            parse_profile(self.root)
        (self.root / 'ffts_profile.slice_0').write_bytes(b'incomplete')
        with self.assertRaisesRegex(ValueError, 'record length'):
            parse_profile(self.root)


if __name__ == '__main__':
    unittest.main()
