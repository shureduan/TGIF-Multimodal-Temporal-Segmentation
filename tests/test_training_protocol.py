import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tgif_dwa import wear_training as training
from tgif_dwa.wear_baselines import BASELINE_METHODS, WearBaseline, load_wear_predictor
from tgif_dwa.wear_model import WearFinalModel, build_wear_parent, load_wear_final


def synthetic_sequence(subject="sbj_1", length=6):
    rng = np.random.default_rng(3)
    return {"sequence_id": subject, "video": rng.normal(size=(length, 2048)).astype(np.float32),
            "inertial": rng.normal(size=(length, 600)).astype(np.float32),
            "labels": np.resize(np.array([0, 1, 18]), length)}


class TrainingProtocolTests(unittest.TestCase):
    def test_outer_subject_is_never_loaded_and_both_components_use_fixed_epochs(self):
        loaded = []

        def load(_root, subject):
            self.assertNotEqual(subject, "sbj_0", "outer test subject leaked into training")
            loaded.append(subject)
            return synthetic_sequence(subject)

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(training, "load_sequence", side_effect=load), \
                patch.object(training, "train_parent", return_value=({}, [])) as parent, \
                patch.object(training, "train_probe", return_value=({}, [])) as probe:
            output = training.run_training(data_root="unused", fold=1, output=directory,
                                           seed=41, parent_epochs=3, probe_epochs=2)
            self.assertEqual(set(loaded), {f"sbj_{i}" for i in range(1, 18)})
            self.assertEqual(parent.call_args.args[3], 3)
            self.assertEqual(probe.call_args.args[3], 2)
            self.assertEqual(probe.call_args.args[5], 143)
            a = torch.load(output / "parent.pt", weights_only=False)
            b = torch.load(output / "background_probe.pt", weights_only=False)
            self.assertEqual(a["run_metadata"], b["run_metadata"])
            self.assertEqual((a["epoch"], b["epoch"]), (2, 1))
            self.assertEqual(b["seed"], 143)
            self.assertNotIn("best", a)
            self.assertNotIn("best_epoch", a)
            metadata = json.loads((output / "training.json").read_text())
            self.assertEqual(metadata["checkpoint_selection"], "fixed_epoch_last")
            self.assertEqual(metadata["test_subject"], "sbj_0")
            with self.assertRaises(FileExistsError):
                training.run_training(data_root="unused", fold=1, output=directory, seed=41)

    def test_training_and_inference_have_identical_round1_support_and_scores(self):
        torch.manual_seed(5)
        # Compare factories in eval mode: MS-TCN has ordinary training dropout.
        parent = build_wear_parent().eval()
        final = WearFinalModel().eval()
        final.parent.load_state_dict(parent.state_dict(), strict=True)
        video, sensor = torch.randn(8, 2048), torch.randn(8, 600)
        with torch.no_grad():
            a = parent(video, sensor)
            b = final(video, sensor)["parent"]
        self.assertTrue(torch.equal(a["logits"], b["logits"]))
        for key in ("selected_window_start", "selected_window_end", "core_attention_mass",
                    "margin_attention_mass"):
            self.assertTrue(torch.equal(a["round_diagnostics"][1][key],
                                        b["round_diagnostics"][1][key]), key)
        self.assertEqual(parent.uncertainty_seconds, 1.0)
        self.assertEqual(parent.margin_prior, 0.5)

    def test_real_optimization_updates_parent_and_baseline_weights(self):
        sequence = synthetic_sequence()
        mean, std = np.zeros(12, np.float32), np.ones(12, np.float32)
        for method in ("FINAL_MODEL", *BASELINE_METHODS):
            with self.subTest(method=method):
                training.set_seed(9)
                initial = build_wear_parent() if method == "FINAL_MODEL" else WearBaseline(method)
                before = copy.deepcopy(initial.state_dict())
                state, history = training.train_parent([sequence], mean, std, 1, "cpu", 9, method)
                self.assertEqual(len(history), 1)
                self.assertTrue(np.isfinite(history[0]["train_loss"]))
                self.assertTrue(any(not torch.equal(state[k], before[k]) for k in state))
                initial.load_state_dict(state, strict=True)

    def test_checkpoint_pair_rejects_mixed_runs_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(training, "load_sequence", side_effect=lambda _, s: synthetic_sequence(s)):
            output = training.run_training(data_root="unused", fold=1, output=directory,
                                           seed=47, parent_epochs=1, probe_epochs=1)
            model, mean, std = load_wear_final(output / "parent.pt", output / "background_probe.pt")
            self.assertEqual(model.run_metadata["test_subject"], "sbj_0")
            self.assertEqual(mean.shape, (12,))
            self.assertTrue((std > 0).all())
            model.eval()
            with torch.no_grad():
                probabilities = model(torch.randn(6, 2048), torch.randn(6, 600))["probabilities"]
            self.assertTrue(torch.isfinite(probabilities).all())
            self.assertTrue(torch.allclose(probabilities.sum(-1), torch.ones(6)))
            probe = torch.load(output / "background_probe.pt", weights_only=False)
            probe["run_metadata"]["seed"] = 99
            torch.save(probe, output / "wrong_probe.pt")
            with self.assertRaisesRegex(ValueError, "metadata differ"):
                load_wear_final(output / "parent.pt", output / "wrong_probe.pt")
            parent = torch.load(output / "parent.pt", weights_only=False)
            parent["epoch"] = 99
            torch.save(parent, output / "wrong_parent.pt")
            with self.assertRaisesRegex(ValueError, "final epoch/seed"):
                load_wear_final(output / "wrong_parent.pt", output / "background_probe.pt")
            parent["epoch"] = 0
            parent["run_metadata"]["train_subjects"].append("sbj_0")
            probe["run_metadata"] = parent["run_metadata"]
            torch.save(parent, output / "wrong_parent.pt")
            torch.save(probe, output / "wrong_probe.pt")
            with self.assertRaisesRegex(ValueError, "train/test split"):
                load_wear_final(output / "wrong_parent.pt", output / "wrong_probe.pt")

    def test_baseline_checkpoint_roundtrips_without_a_probe(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(training, "load_sequence", side_effect=lambda _, s: synthetic_sequence(s)):
            for method in BASELINE_METHODS:
                with self.subTest(method=method):
                    output = training.run_training(data_root="unused", fold=1, output=directory,
                                                   seed=47, parent_epochs=1, method=method)
                    model, _, _ = load_wear_predictor(output / "parent.pt")
                    self.assertEqual(model.run_metadata["method"], method)
                    self.assertEqual(model.run_metadata["probe_epochs"], 0)
                    self.assertFalse((output / "background_probe.pt").exists())
                    with self.assertRaisesRegex(ValueError, "does not use"):
                        load_wear_predictor(output / "parent.pt", probe="unused")

    def test_single_frame_loss_is_finite(self):
        loss = training.stage_loss(torch.randn(4, 1, 19, 1), torch.tensor([18]))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
