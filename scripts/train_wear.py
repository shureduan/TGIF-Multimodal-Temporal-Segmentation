#!/usr/bin/env python3
"""Train one WEAR LOSO parent and its task-presence probe.

This command intentionally accepts one fold only.  It preserves the audited
seed, loss, training-only inertial normalization, and held-out-subject checkpoint
selection used by the existing experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from tgif_dwa.iterative_dwa import IterativeRawDWA, WindowConfig
from tgif_dwa.wear_data import fit_inertial_normalizer, load_sequence, loso_subjects
from tgif_dwa.wear_model import TaskPresenceProbe, normalize_wear_inertial

SEED = 47
TMSE_WEIGHT = 0.15


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stage_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    total = logits.new_tensor(0.0)
    for stage in logits:
        scores = stage.squeeze(0).T
        classification = F.cross_entropy(scores, target)
        log_probability = F.log_softmax(stage.squeeze(0), dim=0)
        difference = torch.clamp(
            (log_probability[:, 1:] - log_probability[:, :-1]).abs(), max=4.0
        ).square().mean()
        total = total + classification + TMSE_WEIGHT * difference
    return total


def evaluate_parent(model, sequence, mean, std, device):
    model.eval()
    video = torch.as_tensor(sequence["video"], dtype=torch.float32, device=device)
    inertial = torch.as_tensor(
        normalize_wear_inertial(sequence["inertial"], mean, std),
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        prediction = model(video, inertial)["round_predictions"][1].cpu().numpy()
    return float(
        f1_score(
            sequence["labels"], prediction, labels=np.arange(19),
            average="macro", zero_division=0,
        )
    )


def train_parent(train, held_out, mean, std, epochs, device):
    set_seed(SEED)
    model = IterativeRawDWA(
        window=WindowConfig(max_escape_seconds=0.0), n_rounds=2
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_score, best_epoch, best_state = -1.0, -1, None
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for index in np.random.permutation(len(train)):
            sequence = train[int(index)]
            video = torch.as_tensor(sequence["video"], dtype=torch.float32, device=device)
            inertial = torch.as_tensor(
                normalize_wear_inertial(sequence["inertial"], mean, std),
                dtype=torch.float32,
                device=device,
            )
            target = torch.as_tensor(sequence["labels"], dtype=torch.long, device=device)
            output = model(video, inertial)
            loss = 0.5 * stage_loss(output["round_logits"][0], target)
            loss = loss + stage_loss(output["round_logits"][1], target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = evaluate_parent(model, held_out, mean, std, device)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "held_out_macro_f1": score})
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f"parent fold epoch={epoch:02d} macro_f1={score:.6f}", flush=True)
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    return best_state, best_epoch, best_score, history


def train_probe(train, mean, std, fold, epochs, device):
    set_seed(SEED + fold * 100 + 2)
    probe = TaskPresenceProbe(2648).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(epochs):
        probe.train()
        losses = []
        for index in np.random.permutation(len(train)):
            sequence = train[int(index)]
            video = torch.as_tensor(sequence["video"], dtype=torch.float32, device=device)
            inertial = torch.as_tensor(
                normalize_wear_inertial(sequence["inertial"], mean, std),
                dtype=torch.float32,
                device=device,
            )
            target = torch.as_tensor(
                sequence["labels"] == 18, dtype=torch.float32, device=device
            )
            loss = criterion(probe(torch.cat([video, inertial], dim=-1)), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(f"probe fold epoch={epoch:02d} loss={history[-1]['loss']:.6f}", flush=True)
    return copy.deepcopy(probe.state_dict()), history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 19), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-epochs", type=int, default=30)
    parser.add_argument("--probe-epochs", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train_ids, held_out_id = loso_subjects(args.fold)
    train = [load_sequence(args.data_root, subject) for subject in train_ids]
    held_out = load_sequence(args.data_root, held_out_id)
    mean, std = fit_inertial_normalizer(train)
    device = torch.device(args.device)
    parent_state, best_epoch, best_score, parent_history = train_parent(
        train, held_out, mean, std, args.parent_epochs, device
    )
    probe_state, probe_history = train_probe(
        train, mean, std, args.fold, args.probe_epochs, device
    )

    fold_root = args.output / f"split_{args.fold:02d}"
    fold_root.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": parent_state,
            "normalization_mean": mean,
            "normalization_std": std,
            "best_epoch": best_epoch,
            "best": best_score,
            "seed": SEED,
        },
        fold_root / "parent.pt",
    )
    torch.save({"model": probe_state, "seed": SEED}, fold_root / "background_probe.pt")
    metadata = {
        "fold": args.fold,
        "train_subjects": train_ids,
        "held_out_subject": held_out_id,
        "seed": SEED,
        "selection": "best 19-class Macro-F1 on the held-out subject",
        "parent_history": parent_history,
        "probe_history": probe_history,
    }
    (fold_root / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
