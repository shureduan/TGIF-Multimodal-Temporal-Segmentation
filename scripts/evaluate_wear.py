#!/usr/bin/env python3
"""Evaluate saved WEAR predictions on the released 2 Hz feature grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tgif_dwa.wear_metrics import (
    official_macro_19,
    official_wear_concat_metrics,
    tal_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", nargs="+", type=Path, help="prediction .npz files")
    parser.add_argument("--output", type=Path, help="optional JSON output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    truths, predictions = [], []
    identities = set()
    run_groups = set()
    for path in args.predictions:
        with np.load(path, allow_pickle=False) as payload:
            required = {"true", "pred", "probabilities"}
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(f"{path}: missing arrays {sorted(missing)}")
            truth = np.asarray(payload["true"])
            prediction = np.asarray(payload["pred"])
            probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
            if (truth.ndim != 1 or truth.shape != prediction.shape or
                    probabilities.shape != (len(truth), 19)):
                raise ValueError(f"{path}: inconsistent prediction shapes")
            if not (np.issubdtype(truth.dtype, np.integer) and
                    np.issubdtype(prediction.dtype, np.integer)):
                raise ValueError(f"{path}: labels and predictions must be integers")
            record_id = str(payload["id"].item()) if "id" in payload.files else path.stem
            metadata = {
                key: payload[key].item() for key in ("protocol", "method", "fold", "seed", "run_id")
                if key in payload.files
            }
            if metadata.get("protocol") == "fixed_epoch_loso_v2":
                if not {"fold", "seed", "method", "run_id"}.issubset(metadata):
                    raise ValueError(f"{path}: incomplete v2 prediction metadata")
                if record_id != f"sbj_{metadata['fold'] - 1}":
                    raise ValueError(f"{path}: fold/subject identity mismatch")
            if not np.isfinite(probabilities).all() or (probabilities < 0).any():
                raise ValueError(f"{path}: invalid probabilities")
            if not np.allclose(probabilities.sum(-1), 1.0, atol=1e-5):
                raise ValueError(f"{path}: probabilities must sum to one")
            if len(truth) == 0 or not np.isin(truth, np.arange(19)).all():
                raise ValueError(f"{path}: labels must be a nonempty 19-class sequence")
            if not np.array_equal(prediction, probabilities.argmax(-1)):
                raise ValueError(f"{path}: predictions must match probability argmax")
        if record_id in identities:
            raise ValueError(f"duplicate subject: {record_id}")
        identities.add(record_id)
        run_groups.add(tuple(metadata.get(key) for key in ("protocol", "method", "seed")))
        if len(run_groups) > 1:
            raise ValueError("evaluate one protocol/method/seed at a time")
        truths.append(truth)
        predictions.append(prediction)
        records.append(
            {"id": record_id, "true": truth, "pred": prediction,
             "probabilities": probabilities, "metadata": metadata}
        )

    frame = official_wear_concat_metrics(truths, predictions)
    per_fold = []
    for record in records:
        values = official_macro_19(record["true"], record["pred"])
        values["accuracy"] = float((record["true"] == record["pred"]).mean())
        values.update(tal_map([record]))
        per_fold.append({"subject": record["id"], **record["metadata"], **values})
    fold_keys = [
        "macro_f1_19", "accuracy", "map_at_0.3", "map_at_0.4",
        "map_at_0.5", "map_at_0.6", "map_at_0.7", "avg_map",
    ]
    mean_fold = {
        key: {
            "mean": float(np.mean([row[key] for row in per_fold])),
            "std": float(np.std([row[key] for row in per_fold])),
        }
        for key in fold_keys
    }
    concatenated_accuracy = float(
        (np.concatenate(truths) == np.concatenate(predictions)).mean()
    )
    result = {
        "protocol": "wear_feature_grid_2hz",
        "n_sequences": len(records),
        "n_frames": int(sum(map(len, truths))),
        "mean_fold": mean_fold,
        "per_subject": per_fold,
        "concatenated": {
            "accuracy": concatenated_accuracy,
            **{
                key: frame[key]
                for key in ("macro_precision", "macro_recall", "macro_f1")
            },
        },
        "scope_note": (
            "Mean-fold metrics average per-subject fold scores; concatenated record metrics "
            "use the fixed 19-class mapping on the 2 Hz feature grid. TAL metrics are derived "
            "from contiguous MS-TCN predictions, not official ActionFormer/TriDet outputs."
        ),
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
