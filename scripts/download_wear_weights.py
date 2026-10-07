#!/usr/bin/env python3
"""Download and verify the pretrained WEAR v2 checkpoint bundles (stdlib only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
METHODS = ("FINAL_MODEL", "VIDEO_ONLY", "EARLY_CONCAT", "FIXED_WINDOW_ATTENTION")
SEEDS = (41, 47, 53)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(root, name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts
            or str(relative) != name or "\\" in name):
        raise ValueError(f"Unsafe checkpoint path: {name}")
    path = root.joinpath(*relative.parts)
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError(f"Symlink in checkpoint destination: {part}")
    return path


def validate_archive_record(archive):
    method, seed = archive["method"], archive["seed"]
    if method not in METHODS or seed not in SEEDS:
        raise ValueError("Unknown method or seed in checkpoint manifest")
    expected = {
        f"{method}/seed_{seed}/split_{fold:02d}/{name}"
        for fold in range(1, 19)
        for name in (["parent.pt", "background_probe.pt"] if method == "FINAL_MODEL" else ["parent.pt"])
    }
    return validate_file_records(archive, expected)


def validate_file_records(archive, expected):
    """Check an exact, caller-declared set of checkpoint members."""
    names = [row["path"] for row in archive["files"]]
    if len(names) != len(expected) or set(names) != expected:
        raise ValueError("Manifest must contain the complete 18-fold checkpoint set (or declared release cohort)")
    if PurePosixPath(archive["asset"]).name != archive["asset"]:
        raise ValueError("Invalid archive filename")
    return {row["path"]: row for row in archive["files"]}


def check_file(path, record):
    return path.is_file() and path.stat().st_size == record["bytes"] and sha256(path) == record["sha256"]


def unpack_verified_archive(archive_path, archive, staging, validator=validate_archive_record):
    """Validate every member before any checkpoint is installed."""
    expected = validator(archive)
    if not check_file(archive_path, archive):
        raise ValueError(f"Archive size/SHA-256 mismatch: {archive_path.name}")
    with tarfile.open(archive_path, "r:gz") as stream:
        members = stream.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError("Archive contains missing, duplicate or unexpected members")
        for member in members:
            record = expected[member.name]
            if not member.isfile() or member.size != record["bytes"]:
                raise ValueError(f"Invalid checkpoint member: {member.name}")
            path = safe_path(staging, member.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, path.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            if not check_file(path, record):
                raise ValueError(f"Checkpoint SHA-256 mismatch: {member.name}")


def ensure_bundle(archive, root, archive_dir=None, verify_only=False, validator=validate_archive_record):
    expected = validator(archive)
    missing = []
    for name, record in expected.items():
        path = safe_path(root, name)
        if path.exists():
            if not check_file(path, record):
                raise ValueError(f"Existing checkpoint differs from the release: {path}. "
                                 "Move it aside or choose a different --model-root.")
        else:
            missing.append(name)
    label = f"{archive['method']} seed {archive['seed']}"
    if not missing:
        print(f"Verified {label}: {len(expected)} checkpoint files", flush=True)
        return
    if verify_only:
        raise ValueError(f"{label}: {len(missing)} missing checkpoint files")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".wear-download-", dir=root) as temporary:
        temporary = Path(temporary)
        if archive_dir is not None:
            archive_path = archive_dir / archive["asset"]
            print(f"Installing {label} from {archive_path}", flush=True)
        else:
            archive_path = temporary / archive["asset"]
            print(f"Downloading {label} ({archive['bytes'] / 1024**2:.1f} MiB)", flush=True)
            request = Request(archive["url"], headers={"User-Agent": "tgif-dwa-weight-downloader"})
            with urlopen(request, timeout=60) as response, archive_path.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        staging = temporary / "verified"
        unpack_verified_archive(archive_path, archive, staging, validator)
        for name in missing:
            destination = safe_path(root, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if not check_file(destination, expected[name]):
                    raise ValueError(f"Checkpoint changed during download: {destination}")
                continue
            (staging / name).replace(destination)
    print(f"Installed and verified {label}: {len(expected)} checkpoint files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO / "models/wear_v2/manifest.json")
    parser.add_argument("--model-root", type=Path, default=REPO / "models/wear_v2")
    parser.add_argument("--method", choices=METHODS, default="FINAL_MODEL")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=47)
    parser.add_argument("--all", action="store_true", help="download all four methods and all three seeds")
    parser.add_argument("--archive-dir", type=Path, help="use already downloaded release archives instead of the network")
    parser.add_argument("--verify-only", action="store_true", help="check installed files without downloading")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["protocol"] != "fixed_epoch_loso_v2" or manifest["schema_version"] != 1:
        parser.error("Expected a WEAR v2 release manifest")
    archives = manifest["archives"]
    identities = [(row["method"], row["seed"]) for row in archives]
    if len(identities) != 12 or set(identities) != {(m, s) for m in METHODS for s in SEEDS}:
        parser.error("Release manifest must contain all 12 method/seed bundles")
    selected = archives if args.all else [row for row in archives if (row["method"], row["seed"]) == (args.method, args.seed)]
    for archive in selected:
        ensure_bundle(archive, args.model_root.resolve(), args.archive_dir, args.verify_only)
    print(f"Weights ready under {args.model_root.resolve()}")


if __name__ == "__main__":
    main()
