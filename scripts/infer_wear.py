#!/usr/bin/env python3
"""Run one fold-specific WEAR final model on prepared features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tgif_dwa.wear_data import load_sequence, orient_feature
from tgif_dwa.wear_baselines import load_wear_predictor
from tgif_dwa.wear_model import normalize_wear_inertial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", type=Path, help="WEAR root containing video/, imu/, label/")
    source.add_argument("--video", type=Path, help="standalone [T,2048] NumPy array")
    parser.add_argument("--subject", help="subject name, required with --data-root (for example sbj_0)")
    parser.add_argument("--imu", type=Path, help="standalone [T,600] NumPy array")
    parser.add_argument("--parent", type=Path, required=True, help="fold parent checkpoint")
    parser.add_argument("--probe", type=Path, help="paired probe, required for FINAL_MODEL")
    parser.add_argument("--output", type=Path, required=True, help="output .npz path")
    parser.add_argument("--device", default="cpu", help="cpu, mps, or cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, mean, std = load_wear_predictor(args.parent, args.probe, device=args.device)
    metadata = model.run_metadata
    if args.data_root is not None:
        if not args.subject:
            raise SystemExit("--subject is required with --data-root")
        if metadata.get("test_subject") and args.subject != metadata["test_subject"]:
            raise ValueError("--subject must match the checkpoint's outer test subject")
        sequence = load_sequence(args.data_root, args.subject)
        video = sequence["video"]
        inertial = sequence["inertial"]
        target = sequence["labels"]
        sequence_id = args.subject
    else:
        if args.imu is None:
            raise SystemExit("--imu is required with --video")
        video = orient_feature(np.load(args.video), 2048)
        inertial = orient_feature(np.load(args.imu), 600)
        length = min(len(video), len(inertial))
        video, inertial = video[:length], inertial[:length]
        target = None
        sequence_id = args.video.stem

    normalized = normalize_wear_inertial(inertial, mean, std)
    device = torch.device(args.device)
    with torch.inference_mode():
        output = model(
            torch.as_tensor(video, dtype=torch.float32, device=device),
            torch.as_tensor(normalized, dtype=torch.float32, device=device),
        )

    payload = {
        "id": np.asarray(sequence_id),
        "pred": output["predictions"].cpu().numpy(),
        "probabilities": output["probabilities"].cpu().numpy(),
        "feature_stride_seconds": np.asarray(0.5),
        "protocol": np.asarray(metadata["protocol"]),
    }
    if "p_background" in output:
        payload["p_background"] = output["p_background"].cpu().numpy()
    for key in ("fold", "seed", "method", "run_id"):
        if key in metadata:
            payload[key] = np.asarray(metadata[key])
    if target is not None:
        payload["true"] = target
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    summary = {
        "output": str(args.output),
        "sequence_id": sequence_id,
        "frames": int(len(video)),
        "classes": 19,
        "contains_ground_truth": target is not None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
