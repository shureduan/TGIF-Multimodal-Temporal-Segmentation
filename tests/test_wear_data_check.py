import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_wear_commands import script_module


class WearDataCheckTests(unittest.TestCase):
    def test_preflight_reads_prepared_inputs_and_rejects_unusable_arrays(self):
        check = script_module("check_wear_data").check_subject
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kind in ("video", "imu", "label"):
                (root / kind).mkdir()
            video_path = root / "video/sbj_0.npy"
            imu_path = root / "imu/sbj_0.npy"
            label_path = root / "label/wear_split_18.json"

            def labels(label_id):
                label_path.write_text(json.dumps({"database": {"sbj_0": {
                    "annotations": [{"segment": [0, 4], "label_id": label_id}]
                }}}))

            np.save(video_path, np.zeros((4, 2048), dtype=np.float32))
            np.save(imu_path, np.zeros((600, 5), dtype=np.float32))
            labels(0)
            self.assertEqual(check(root, "sbj_0"), 4)
            labels(99)
            with self.assertRaisesRegex(ValueError, "label IDs"):
                check(root, "sbj_0")
            labels(0)
            np.save(imu_path, np.full((4, 600), np.nan, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "non-finite"):
                check(root, "sbj_0")
            np.save(imu_path, np.zeros((8, 600), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "length difference"):
                check(root, "sbj_0")
            np.save(video_path, np.zeros((0, 2048), dtype=np.float32))
            np.save(imu_path, np.zeros((0, 600), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "empty"):
                check(root, "sbj_0")


if __name__ == "__main__":
    unittest.main()
