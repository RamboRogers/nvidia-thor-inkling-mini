import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fetch_model as stage


class FetchModelTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.data = {'config.json': b'{"test":true}', 'weights.safetensors': b'weight', 'tiktoken/tokenizer.model': b'tokens'}
        self.files = {k: (len(b), (hashlib.sha1(f'blob {len(b)}\0'.encode() + b) if k.endswith('.json') else hashlib.sha256(b)).hexdigest()) for k, b in self.data.items()}
        f = patch.object(stage, 'FILES', self.files)
        f.start()
        self.addCleanup(f.stop)
        free = patch.object(stage.shutil, 'disk_usage', return_value=type('Usage', (), {'free': 10 * 2**30})())
        free.start()
        self.addCleanup(free.stop)
        self.final = self.root / stage.REV

    def curl(self, args, check):
        self.assertTrue(check)
        name = args[-1].split(stage.REV + '/')[1]
        Path(args[args.index('-o') + 1]).write_bytes(self.data[name])

    def stage(self):
        with patch.object(stage.subprocess, 'run', side_effect=self.curl) as fetch:
            stage.run(self.root)
            self.assertEqual(fetch.call_count, len(self.files))

    def test_roundtrip_and_verification_without_network(self):
        self.stage()
        with patch.object(stage.subprocess, 'run', side_effect=AssertionError('unexpected network')):
            stage.run(self.root, True)
            stage.run(self.root)
        self.assertEqual(json.loads((self.final / '.verified').read_text())['revision'], stage.REV)

    def test_extra_file_and_symlink_rejected(self):
        self.stage()
        extra = self.final / 'modeling.py'
        extra.write_text('unexpected')
        with self.assertRaises(ValueError):
            stage.run(self.root, True)
        extra.unlink()
        f = self.final / 'weights.safetensors'
        target = self.root / 'weights'
        f.rename(target)
        f.symlink_to(target)
        with self.assertRaises(ValueError):
            stage.run(self.root, True)

    def test_wrong_bytes_fail_closed(self):
        with patch.object(stage.subprocess, 'run', side_effect=lambda a, check: Path(a[a.index('-o') + 1]).write_bytes(b'x' * len(self.data['config.json']))):
            with self.assertRaisesRegex(ValueError, 'checksum'):
                stage.run(self.root)
        self.assertFalse(self.final.exists())

    def test_incomplete_or_false_receipt_rejected(self):
        self.stage()
        (self.final / '.verified').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'receipt'):
            stage.run(self.root, True)
        (self.final / 'config.json').unlink()
        with self.assertRaisesRegex(ValueError, 'size'):
            stage.run(self.root, True)

    def test_missing_snapshot_no_changes(self):
        with self.assertRaises(ValueError):
            stage.run(self.root, True)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
