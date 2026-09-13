"""WEAR feature-grid loader used by the released training and inference tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

VIDEO_DIM = 2048
INERTIAL_DIM = 600
N_CLASSES = 19
BACKGROUND = 18
FEATURE_STRIDE_SECONDS = 0.5


def orient_feature(array: np.ndarray, feature_dim: int) -> np.ndarray:
    """Return a two-dimensional feature array in ``[T,D]`` orientation."""

    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array, got {array.shape}")
    if array.shape[1] == feature_dim:
        return array.astype(np.float32)
    if array.shape[0] == feature_dim:
        return array.T.astype(np.float32)
    raise ValueError(f"neither axis matches feature_dim={feature_dim}: {array.shape}")


def official_feature_timestamps(length: int) -> np.ndarray:
    """WEAR feature centers used by the released 18-fold checkpoints."""

    return (np.arange(length, dtype=np.float64) + 1.0) * FEATURE_STRIDE_SECONDS


def _annotation_entry(label_dir: Path, subject: str) -> Dict:
    paths = [label_dir / "wear_split_18.json"]
    paths.extend(label_dir / f"wear_test_split_{index}.json" for index in range(1, 7))
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if subject in payload.get("database", {}):
            return payload["database"][subject]
    raise KeyError(f"{subject} was not found in the WEAR annotation JSON files")


def build_labels(label_dir: str | Path, subject: str, length: int) -> np.ndarray:
    """Map continuous WEAR annotations to the 2 Hz feature grid."""

    entry = _annotation_entry(Path(label_dir), subject)
    labels = np.full(length, BACKGROUND, dtype=np.int64)
    centers = official_feature_timestamps(length)
    for annotation in sorted(entry["annotations"], key=lambda item: item["segment"][0]):
        start, end = annotation["segment"]
        label_id = int(annotation["label_id"])
        labels[(centers >= start) & (centers < end)] = label_id
    return labels


def load_sequence(data_root: str | Path, subject: str) -> Dict:
    """Load one official-feature sequence and its 19-class frame labels."""

    root = Path(data_root)
    video = orient_feature(np.load(root / "video" / f"{subject}.npy"), VIDEO_DIM)
    inertial = orient_feature(np.load(root / "imu" / f"{subject}.npy"), INERTIAL_DIM)
    if abs(len(video) - len(inertial)) > 2:
        raise ValueError(
            f"{subject}: video/IMU length difference exceeds two frames: "
            f"{len(video)} vs {len(inertial)}"
        )
    length = min(len(video), len(inertial))
    return {
        "sequence_id": subject,
        "video": video[:length],
        "inertial": inertial[:length],
        "labels": build_labels(root / "label", subject, length),
        "feature_stride_seconds": FEATURE_STRIDE_SECONDS,
    }


def loso_subjects(fold: int) -> Tuple[List[str], str]:
    """Return the official first-18-subject LOSO train/held-out identities."""

    if fold not in range(1, 19):
        raise ValueError("fold must be in 1..18")
    held_out = f"sbj_{fold - 1}"
    train = [f"sbj_{index}" for index in range(18) if index != fold - 1]
    return train, held_out


def fit_inertial_normalizer(sequences: Iterable[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Fit 12 raw-channel statistics using training subjects only."""

    total = np.zeros(12, dtype=np.float64)
    squared = np.zeros(12, dtype=np.float64)
    count = 0
    for sequence in sequences:
        x = np.nan_to_num(sequence["inertial"]).reshape(-1, 12, 50).astype(np.float64)
        total += x.sum(axis=(0, 2))
        squared += (x * x).sum(axis=(0, 2))
        count += x.shape[0] * 50
    if count == 0:
        raise ValueError("cannot fit normalization on an empty sequence set")
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - mean * mean, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)
