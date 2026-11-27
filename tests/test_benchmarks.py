# SPDX-License-Identifier: Apache-2.0
import io
import json
from pathlib import Path
import statistics
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmarks import bench


RESULTS = Path(__file__).resolve().parents[1] / 'benchmarks' / 'results'


def receipt(name):
    return json.loads((RESULTS / name).read_text())


class BenchmarkTests(unittest.TestCase):
    def test_published_speed_records_recompute(self):
        profiles = {
            'baseline-v5-fast.json': (56.91, 59.40, 100.35),
            'v6-baked-fast.json': (75.44, 70.34, 104.67),
            'v6-tuning-full.json': (76.75, 70.40, 95.35),
        }
        for path, expected in profiles.items():
            data = receipt(path)
            for effort, mean in zip(('none', 'medium'), expected[:2]):
                suite = data['c1_' + effort]
                self.assertEqual(set(suite['tasks']), set(bench.PROMPTS_C1))
                for row in suite['tasks'].values():
                    self.assertAlmostEqual(row['med_decode_tps'], statistics.median(
                        run['decode_tps'] for run in row['runs']), delta=0.0051)
                self.assertEqual(round(statistics.mean(
                    row['med_decode_tps'] for row in suite['tasks'].values()), 2), mean)
                self.assertEqual(suite['mean_decode_tps'], mean)
            c4 = data['c4_decode']
            self.assertEqual(c4['peak_running'], 4)
            self.assertEqual(c4['agg_tps'], expected[2])
            self.assertAlmostEqual(c4['total_tokens'] / c4['wall_s'], expected[2], delta=0.01)
        cold = receipt('v6-baked-fast.json')['prefill']['32768']
        self.assertAlmostEqual(cold['prompt_tokens'] / cold['cold_ttft_s'], 3278.8, delta=0.1)

    def test_lan_mean_capacity_and_residency(self):
        data = receipt('v6-lan.json')
        suite = data['lan_c1_medium']
        self.assertEqual(set(suite['tasks']), set(bench.PROMPTS_C1))
        self.assertEqual(round(statistics.mean(task['decode_tps'] for task in suite['tasks'].values()), 2), 70.46)
        self.assertEqual(suite['mean_decode_tps'], 70.46)
        self.assertEqual(set(data['stages']), {'smoke', 'code', 'c4', 'mixed', 'long'})
        self.assertTrue(all(stage['status'] == 'passed' for stage in data['stages'].values()))
        long = data['stages']['long']
        self.assertEqual(long['prompt_tokens'] + long['reserved_output_tokens'], data['context_limit'])
        self.assertEqual(long['needles_verified'], 3)
        for name in ('c4', 'mixed'):
            stage = data['stages'][name]
            self.assertEqual((stage['peak_running'], stage['lanes_passed']), (4, 4))

    def test_stream_complete_and_truncated(self):
        events = [
            {'choices': [{'delta': {'reasoning': 'private'}}], 'usage': {'prompt_tokens': 7, 'completion_tokens': 1}},
            {'choices': [{'delta': {'content': 'answer'}, 'finish_reason': 'stop'}],
             'usage': {'prompt_tokens': 7, 'completion_tokens': 2}},
        ]
        stream = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events).encode()
        for trailer in (b'data: [DONE]\n\n', b''):
            class Response(io.BytesIO):
                status = 200

            conn = SimpleNamespace(request=lambda *a: None, getresponse=lambda: Response(stream + trailer), close=lambda: None)
            with patch.object(bench.http.client, 'HTTPConnection', return_value=conn):
                if trailer:
                    row = bench.stream_chat([], 16)
                    self.assertEqual((row['finish_reason'], row['completion_tokens']), ('stop', 2))
                    self.assertNotIn('text_preview', row)
                    self.assertNotIn('private', json.dumps(row))
                else:
                    with self.assertRaisesRegex(RuntimeError, 'incomplete SSE'):
                        bench.stream_chat([], 16)
