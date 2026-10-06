"""Integrity and safe-install checks for the pretrained bundle downloader."""
import copy
import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "download_wear_weights", Path(__file__).resolve().parents[1] / "scripts/download_wear_weights.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class WeightDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payloads = {f"FINAL_MODEL/seed_47/split_{fold:02d}/{name}": f"{fold}-{name}".encode()
                         for fold in range(1, 19) for name in ("parent.pt", "background_probe.pt")}
        self.archive = {"method": "FINAL_MODEL", "seed": 47, "asset": "bundle.tar.gz",
                        "files": [{"path": name, "bytes": len(data),
                                   "sha256": hashlib.sha256(data).hexdigest()}
                                  for name, data in self.payloads.items()]}
        self.repack()

    def repack(self, symlink=None, extra=False):
        self.path = self.root / self.archive["asset"]
        with tarfile.open(self.path, "w:gz") as stream:
            for name, data in self.payloads.items():
                info = tarfile.TarInfo(name)
                if name == symlink:
                    info.type = tarfile.SYMTYPE
                    info.linkname = "../../outside"
                    stream.addfile(info)
                else:
                    info.size = len(data)
                    stream.addfile(info, io.BytesIO(data))
            if extra:
                stream.addfile(tarfile.TarInfo("../outside.pt"), io.BytesIO(b""))
        self.archive.update(bytes=self.path.stat().st_size, sha256=download.sha256(self.path))

    def test_offline_install_verifies_and_reuses_complete_pairs(self):
        destination = self.root / "models"
        download.ensure_bundle(self.archive, destination, self.root)
        for name, data in self.payloads.items():
            self.assertEqual((destination / name).read_bytes(), data)
        self.path.unlink()
        download.ensure_bundle(self.archive, destination, verify_only=True)
        self.assertFalse(list(destination.glob(".wear-download-*")))

    def test_tampered_archive_is_rejected_before_install(self):
        self.path.write_bytes(self.path.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "Archive size/SHA-256"):
            download.ensure_bundle(self.archive, self.root / "models", self.root)
        self.assertFalse(list((self.root / "models").rglob("*.pt")))

    def test_checkpoint_hash_failure_installs_nothing(self):
        name = next(iter(self.payloads))
        self.payloads[name] = b"x" * len(self.payloads[name])
        self.repack()
        with self.assertRaisesRegex(ValueError, "Checkpoint SHA-256"):
            download.ensure_bundle(self.archive, self.root / "models", self.root)
        self.assertFalse(list((self.root / "models").rglob("*.pt")))

    def test_unexpected_path_and_symlink_members_are_rejected(self):
        self.repack(extra=True)
        with self.assertRaisesRegex(ValueError, "unexpected members"):
            download.ensure_bundle(self.archive, self.root / "models", self.root)
        self.assertFalse((self.root / "outside.pt").exists())
        self.repack(symlink=next(iter(self.payloads)))
        with self.assertRaisesRegex(ValueError, "Invalid checkpoint member"):
            download.ensure_bundle(self.archive, self.root / "models", self.root)

    def test_local_conflict_is_preserved_and_missing_files_fail_verification(self):
        destination = self.root / "models"
        with self.assertRaisesRegex(ValueError, "missing checkpoint"):
            download.ensure_bundle(self.archive, destination, verify_only=True)
        path = destination / next(iter(self.payloads))
        path.parent.mkdir(parents=True)
        path.write_bytes(b"different experiment")
        with self.assertRaisesRegex(ValueError, "Existing checkpoint differs"):
            download.ensure_bundle(self.archive, destination, self.root)
        self.assertEqual(path.read_bytes(), b"different experiment")

    def test_incomplete_pair_manifest_and_unsafe_destination_are_rejected(self):
        broken = copy.deepcopy(self.archive)
        broken["files"].pop()
        with self.assertRaisesRegex(ValueError, "complete 18-fold"):
            download.validate_archive_record(broken)
        for name in ("../escape", "/absolute", "a/../escape", "./relative", "a\\b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                download.safe_path(self.root, name)


if __name__ == "__main__":
    unittest.main()
