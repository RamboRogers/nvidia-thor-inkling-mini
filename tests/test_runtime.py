# SPDX-License-Identifier: Apache-2.0
import base64
import io
import json
from pathlib import Path
import struct
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

import build
import fetch_model
import serve


class RuntimeTests(unittest.TestCase):
    def test_launch_preserves_profile_and_loopback(self):
        args = SimpleNamespace(name='test-inkling', image='sha256:test', port=9234,
                               model_root=Path('/example/models'), cache_dir=Path('/example/cache'))
        cmd = serve.launch_args(args)

        def value(flag):
            return cmd[cmd.index(flag) + 1]

        self.assertEqual(value('--publish'), '127.0.0.1:9234:8000/tcp')
        self.assertIn(f'type=bind,src=/example/models/{fetch_model.REV},dst=/model,readonly', cmd)
        self.assertEqual(value('--max-model-len'), '262144')
        self.assertEqual(value('--max-num-seqs'), '4')
        self.assertEqual(value('--max-num-batched-tokens'), '2048')
        self.assertEqual(value('--kv-cache-memory-bytes'), '30064771072')
        self.assertEqual(value('--kv-cache-dtype'), 'bfloat16')
        self.assertEqual(value('--served-model-name'), 'inkling-mini')
        self.assertEqual(json.loads(value('--limit-mm-per-prompt')), {'image': 16})
        self.assertNotIn('--language-model-only', cmd)
        self.assertEqual(json.loads(value('--speculative-config')), {
            'method': 'dspark', 'num_speculative_tokens': 7, 'draft_sample_method': 'probabilistic',
        })
        self.assertIn('--restart=no', cmd)
        self.assertNotIn('--privileged', cmd)

    def test_smoke_checks_text_and_sends_a_real_png(self):
        model = {'data': [{'id': 'inkling-mini', 'max_model_len': 262144}]}
        answer = {'model': 'inkling-mini', 'choices': [{'finish_reason': 'stop', 'message': {'content': 'READY'}}]}
        vision = {'model': 'inkling-mini', 'choices': [{'finish_reason': 'stop', 'message': {'content': 'Red.'}}]}
        with patch.object(serve, 'http_json', side_effect=[model, answer, vision]) as request:
            serve.smoke(8010, 0)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(request.call_args_list[1].args[2]['reasoning_effort'], 'none')
        self.assertEqual(request.call_args.args[:2], (8010, '/v1/chat/completions'))
        payload = request.call_args.args[2]
        self.assertEqual(payload['temperature'], 0)
        self.assertEqual(payload['reasoning_effort'], 'low')
        image, prompt = payload['messages'][0]['content']
        self.assertEqual(image['type'], 'image_url')
        self.assertEqual(prompt['type'], 'text')
        prefix, encoded = image['image_url']['url'].split(',', 1)
        self.assertEqual(prefix, 'data:image/png;base64')
        png = base64.b64decode(encoded, validate=True)
        self.assertEqual(png[:8], b'\x89PNG\r\n\x1a\n')
        self.assertEqual(struct.unpack('>II', png[16:24]), (64, 64))
        size, kind = struct.unpack('>I4s', png[33:41])
        self.assertEqual(kind, b'IDAT')
        self.assertEqual(zlib.decompress(png[41:41 + size]),
                         (b'\x00' + bytes((220, 20, 20)) * 64) * 64)

    def test_smoke_rejects_wrong_image_answer_or_incomplete_finish(self):
        model = {'data': [{'id': 'inkling-mini', 'max_model_len': 262144}]}
        answer = {'model': 'inkling-mini', 'choices': [{'finish_reason': 'stop', 'message': {'content': 'READY'}}]}
        for content, finish in [('green', 'stop'), ('not red', 'stop'), ('red', 'length')]:
            with self.subTest(content=content, finish=finish):
                vision = {'model': 'inkling-mini', 'choices': [{'finish_reason': finish, 'message': {'content': content}}]}
                with patch.object(serve, 'http_json', side_effect=[model, answer, vision]):
                    with self.assertRaisesRegex(ValueError, 'vision contract'):
                        serve.smoke(8010, 0)

    def test_smoke_rejects_wrong_service_or_failed_reply(self):
        model = {'data': [{'id': 'inkling-mini', 'max_model_len': 262144}]}
        bad_answer = {'model': 'inkling-mini', 'choices': [{'finish_reason': 'length', 'message': {'content': 'READY'}}]}
        with patch.object(serve, 'http_json', side_effect=[model, bad_answer]):
            with self.assertRaisesRegex(ValueError, 'chat contract'):
                serve.smoke(8010, 0)
        with patch.object(serve, 'http_json', return_value={'data': [{'id': 'other', 'max_model_len': 4096}]}):
            with self.assertRaisesRegex(ValueError, 'unexpected model ID or context'):
                serve.smoke(8010, 0)

    def test_builder_rejects_changed_bytes_and_archive_shape(self):
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            build.check_bytes(b'other', '0' * 64, 'source.py')
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode='w') as out:
            info = tarfile.TarInfo('vllm.py')
            info.size = 4
            out.addfile(info, io.BytesIO(b'pass'))
        self.assertEqual(build.read_tar(raw.getvalue(), {'vllm.py'}), {'vllm.py': b'pass'})
        with self.assertRaisesRegex(ValueError, 'archive inventory'):
            build.read_tar(raw.getvalue(), {'unexpected.py'})
