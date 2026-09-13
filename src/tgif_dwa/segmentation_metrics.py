"""Frame and segment metrics used by the TGIF experiments."""

from typing import List, Tuple

import numpy as np
from sklearn.metrics import f1_score


def frames_to_segments(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    segments = []
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and labels[end] == labels[start]:
            end += 1
        segments.append((int(labels[start]), start, end))
        start = end
    return segments


def _iou(left, right):
    intersection = max(0, min(left[2], right[2]) - max(left[1], right[1]))
    union = (left[2] - left[1]) + (right[2] - right[1]) - intersection
    return intersection / union if union else 0.0


def f1_at(predicted, target, threshold):
    true_positives = 0
    used = [False] * len(target)
    for prediction in predicted:
        best_iou, best_index = 0.0, -1
        for index, truth in enumerate(target):
            if used[index] or truth[0] != prediction[0]:
                continue
            overlap = _iou(prediction, truth)
            if overlap > best_iou:
                best_iou, best_index = overlap, index
        if best_iou >= threshold and best_index >= 0:
            true_positives += 1
            used[best_index] = True
    false_positives = len(predicted) - true_positives
    false_negatives = len(target) - true_positives
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def _levenshtein(left, right):
    previous = list(range(len(right) + 1))
    for i, left_value in enumerate(left, start=1):
        current = [i] + [0] * len(right)
        for j, right_value in enumerate(right, start=1):
            current[j] = previous[j - 1] if left_value == right_value else 1 + min(
                previous[j], current[j - 1], previous[j - 1]
            )
        previous = current
    return previous[-1]


def core_metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    predicted = np.asarray(predicted, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if predicted.shape != target.shape or predicted.ndim != 1:
        raise ValueError("predicted and target must be aligned one-dimensional arrays")
    predicted_segments = frames_to_segments(predicted)
    target_segments = frames_to_segments(target)
    pred_labels = [segment[0] for segment in predicted_segments]
    target_labels = [segment[0] for segment in target_segments]
    denominator = max(len(pred_labels), len(target_labels))
    edit = 100.0 if denominator == 0 else (
        1.0 - _levenshtein(pred_labels, target_labels) / denominator
    ) * 100.0
    return {
        "accuracy": float((predicted == target).mean() * 100.0),
        "macro_f1": float(
            f1_score(target, predicted, labels=[0, 1, 2], average="macro", zero_division=0)
            * 100.0
        ),
        "edit": float(edit),
        "f1_10": float(f1_at(predicted_segments, target_segments, 0.10) * 100.0),
        "f1_25": float(f1_at(predicted_segments, target_segments, 0.25) * 100.0),
        "f1_50": float(f1_at(predicted_segments, target_segments, 0.50) * 100.0),
    }
