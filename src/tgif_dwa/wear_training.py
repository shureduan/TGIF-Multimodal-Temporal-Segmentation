"""Fixed-epoch LOSO training. This module never loads the outer test subject."""

from __future__ import annotations

import json
import platform
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .wear_baselines import METHODS, WearBaseline
from .wear_data import fit_inertial_normalizer, load_sequence, loso_subjects
from .wear_model import (
    DEFAULT_MODEL_CONFIG, TRAINING_PROTOCOL, TaskPresenceProbe,
    build_wear_parent, normalize_wear_inertial,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def stage_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    total = logits.new_tensor(0.0)
    for stage in logits:
        classification = F.cross_entropy(stage.squeeze(0).T, target)
        log_probability = F.log_softmax(stage.squeeze(0), dim=0)
        smoothing = logits.new_tensor(0.0)
        if log_probability.shape[1] > 1:
            smoothing = torch.clamp(
                (log_probability[:, 1:] - log_probability[:, :-1]).abs(), max=4.0
            ).square().mean()
        total = total + classification + 0.15 * smoothing
    return total


def training_tensors(sequence, mean, std, device):
    return (
        torch.as_tensor(sequence["video"], dtype=torch.float32, device=device),
        torch.as_tensor(normalize_wear_inertial(sequence["inertial"], mean, std),
                        dtype=torch.float32, device=device),
        torch.as_tensor(sequence["labels"], dtype=torch.long, device=device),
    )


def train_parent(train, mean, std, epochs, device, seed, method="FINAL_MODEL", log_path=None):
    if epochs < 1 or not train:
        raise ValueError("parent training requires positive epochs and training sequences")
    set_seed(seed)
    if method == "FINAL_MODEL":
        model = build_wear_parent(
            uncertainty_seconds=DEFAULT_MODEL_CONFIG["uncertainty_seconds"],
            margin_prior=DEFAULT_MODEL_CONFIG["margin_prior"],
        )
    else:
        model = WearBaseline(method)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        started = time.monotonic()
        model.train()
        losses = []
        for index in np.random.permutation(len(train)):
            video, inertial, target = training_tensors(train[int(index)], mean, std, device)
            output = (model(video, inertial, return_diagnostics=False)
                      if method == "FINAL_MODEL" else model(video, inertial))
            rounds = output["round_logits"]
            loss = stage_loss(rounds[-1], target)
            if len(rounds) == 2:
                loss = loss + 0.5 * stage_loss(rounds[0], target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "seconds": time.monotonic() - started}
        history.append(row)
        if log_path is not None:
            with Path(log_path).open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        print(f"{method} epoch={epoch:02d} train_loss={row['train_loss']:.6f} "
              f"seconds={row['seconds']:.1f}", flush=True)
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, history


def train_probe(train, mean, std, epochs, device, seed, log_path=None):
    if epochs < 1 or not train:
        raise ValueError("probe training requires positive epochs and training sequences")
    set_seed(seed)
    probe = TaskPresenceProbe(2648).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        started = time.monotonic()
        probe.train()
        losses = []
        for index in np.random.permutation(len(train)):
            video, inertial, labels = training_tensors(train[int(index)], mean, std, device)
            loss = F.binary_cross_entropy_with_logits(
                probe(torch.cat([video, inertial], dim=-1)), (labels == 18).float()
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite probe loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "seconds": time.monotonic() - started}
        history.append(row)
        if log_path is not None:
            with Path(log_path).open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        print(f"probe epoch={epoch:02d} train_loss={row['train_loss']:.6f}", flush=True)
    return {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}, history


def run_training(*, data_root, fold, output, parent_epochs=30, probe_epochs=15,
                 seed=47, device="cpu", method="FINAL_MODEL") -> Path:
    if parent_epochs < 1 or probe_epochs < 1:
        raise ValueError("epoch budgets must be positive")
    if seed < 0 or seed >= 2**32:
        raise ValueError("seed must be in 0..2**32-1")
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    train_ids, test_id = loso_subjects(fold)
    probe_seed = (seed + fold * 100 + 2) % (2**32)
    fold_root = Path(output) / method / f"seed_{seed}" / f"split_{fold:02d}"
    fold_root.mkdir(parents=True, exist_ok=False)
    run_metadata = {
        "protocol": TRAINING_PROTOCOL, "run_id": uuid.uuid4().hex,
        "method": method, "fold": fold, "train_subjects": train_ids,
        "test_subject": test_id, "seed": seed, "probe_seed": probe_seed,
        "parent_epochs": parent_epochs,
        "probe_epochs": probe_epochs if method == "FINAL_MODEL" else 0,
        "checkpoint_selection": "fixed_epoch_last",
        "model_config": dict(DEFAULT_MODEL_CONFIG),
    }
    # Freeze the run's settings before any training; never consult test scores.
    (fold_root / "protocol.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    train = [load_sequence(data_root, subject) for subject in train_ids]
    mean, std = fit_inertial_normalizer(train)
    device = torch.device(device)
    parent_state, parent_history = train_parent(
        train, mean, std, parent_epochs, device, seed, method, fold_root / "parent_log.jsonl"
    )
    torch.save(
        {"model": parent_state, "normalization_mean": mean, "normalization_std": std,
         "epoch": parent_epochs - 1, "seed": seed, "run_metadata": run_metadata},
        fold_root / "parent.pt",
    )
    probe_history = []
    if method == "FINAL_MODEL":
        probe_state, probe_history = train_probe(
            train, mean, std, probe_epochs, device, probe_seed, fold_root / "probe_log.jsonl"
        )
        torch.save(
            {"model": probe_state, "epoch": probe_epochs - 1, "seed": probe_seed,
             "run_metadata": run_metadata}, fold_root / "background_probe.pt",
        )
    metadata = {
        **run_metadata,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "device": str(device)},
        "parent_history": parent_history, "probe_history": probe_history,
    }
    (fold_root / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return fold_root
