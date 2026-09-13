#!/usr/bin/env python3
"""Verify model files against the release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("models/wear_final/manifest.json"))
    parser.add_argument("--model-root", type=Path, default=Path("models/wear_final"))
    parser.add_argument("--fold", type=int, choices=range(1, 19), help="check one fold only")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    failed = []
    expected = [
        item for item in manifest["files"]
        if args.fold is None or item["fold"] == args.fold
    ]
    for item in expected:
        path = args.model_root / item["release_path"]
        if not path.exists():
            failed.append({"path": str(path), "status": "missing"})
            continue
        observed = sha256(path)
        size = path.stat().st_size
        if observed != item["sha256"] or size != item["bytes"]:
            failed.append(
                {"path": str(path), "status": "mismatch", "sha256": observed, "bytes": size}
            )
    result = {"checked": len(expected), "fold": args.fold, "failures": failed, "ok": not failed}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()
