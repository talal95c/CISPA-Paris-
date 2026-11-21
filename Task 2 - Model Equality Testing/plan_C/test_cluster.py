import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cluster


COMPACT = '2,3,5,7,11,13,17,19,23,29'
SPACED = COMPACT.replace(',', ', ')


class ClusterTests(unittest.TestCase):
    def test_f4_signature(self):
        self.assertEqual(cluster.f4_signature({'F4': [COMPACT]}), 'compact')
        self.assertEqual(cluster.f4_signature({'F4': [SPACED]}), 'spaced')
        for responses in [[], ['Summary: ' + SPACED], ['2,3,4'],
                          [COMPACT.replace(',', ', ', 1)], [COMPACT, SPACED]]:
            with self.subTest(responses=responses):
                self.assertIsNone(cluster.f4_signature({'F4': responses}))

    def test_split_threshold_missing_and_disabled(self):
        data = {str(i): {'F4': [COMPACT if i < 3 else SPACED]} for i in range(6)}
        data.update({'6': {}, '7': {}})
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cluster.split_f4('qwen', list(data), data), [['0', '1', '2', '6'], ['3', '4', '5', '7']])
            self.assertIsNone(cluster.split_f4('qwen', list(data), data, enabled=False))
            self.assertIsNone(cluster.split_f4('qwen', ['0', '1', '3', '4', '5'], data))

    def test_f4_skips_profile_split_and_keeps_model_merges_local(self):
        data = {f'ep-{i + 1:02}': {'F1': ['I am Qwen, a mentor.'],
                                  'F4': [COMPACT if i < 3 else SPACED],
                                  'L01': ['identical answer ' * 20]} for i in range(6)}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'submission.csv'
            args = ['cluster.py', 'unused.jsonl', '-o', str(output), '--endpoints', '6']
            with patch('sys.argv', args), patch.object(cluster, 'load', return_value=data), patch.object(cluster, 'split_two', side_effect=AssertionError('F4 already split')), contextlib.redirect_stderr(io.StringIO()):
                cluster.main()
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r['family'] for r in rows], ['qwenA'] * 3 + ['qwenB'] * 3)
            self.assertEqual([r['model'] for r in rows], ['qwenA-S'] * 3 + ['qwenB-S'] * 3)
            args.append('--no-f4-split')
            with patch('sys.argv', args), patch.object(cluster, 'load', return_value=data), patch.object(cluster, 'split_two', return_value=([list(data)], None)) as split, contextlib.redirect_stderr(io.StringIO()):
                cluster.main()
            split.assert_called_once()
            with output.open() as handle:
                self.assertEqual({r['family'] for r in csv.DictReader(handle)}, {'qwen'})


if __name__ == '__main__':
    unittest.main()
