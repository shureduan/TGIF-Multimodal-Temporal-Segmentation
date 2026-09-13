"""WEAR-compatible metrics used by the released 2 Hz evaluation path.

Frame metrics use the fixed official 19-class space (18 actions + null). TAL
metrics use contiguous non-null predictions, their mean frame probability as the
segment confidence, one-to-one matching, and interpolated AP.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

N_CLASSES = 19
BACKGROUND_ID = 18
TAL_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def official_macro_19(y_true, y_pred):
    """Fixed-label-space macro P/R/F1; absent classes contribute zero."""
    p, r, f, _ = precision_recall_fscore_support(
        np.asarray(y_true), np.asarray(y_pred), labels=np.arange(N_CLASSES),
        average="macro", zero_division=0,
    )
    return {"macro_precision_19": float(p), "macro_recall_19": float(r),
            "macro_f1_19": float(f)}


def official_wear_per_class_metrics(y_true, y_pred):
    """Reproduce official WEAR main.py per-class record metrics.

    The official code uses all 19 labels and ``zero_division=1``.  Arrays may
    use any consistent semantic ID permutation; see the audit remapping test.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(N_CLASSES), average=None,
        zero_division=1,
    )
    return {
        "per_class_precision": p.astype(float).tolist(),
        "per_class_recall": r.astype(float).tolist(),
        "per_class_f1": f.astype(float).tolist(),
        "per_class_support": support.astype(int).tolist(),
        "macro_precision": float(np.nanmean(p)),
        "macro_recall": float(np.nanmean(r)),
        "macro_f1": float(np.nanmean(f)),
    }


def official_wear_concat_metrics(list_of_fold_gt, list_of_fold_pred):
    """Official WEAR final aggregation: concatenate fold pairs, then score."""
    if len(list_of_fold_gt) != len(list_of_fold_pred) or not list_of_fold_gt:
        raise ValueError("non-empty paired fold lists are required")
    for gt, pred in zip(list_of_fold_gt, list_of_fold_pred):
        if len(gt) != len(pred):
            raise ValueError("ground truth/prediction length mismatch")
    return official_wear_per_class_metrics(
        np.concatenate([np.asarray(x) for x in list_of_fold_gt]),
        np.concatenate([np.asarray(x) for x in list_of_fold_pred]),
    )


def _segments(labels, probabilities=None):
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return []
    cuts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    out = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        cls = int(labels[start])
        if cls == BACKGROUND_ID:
            continue
        score = 1.0 if probabilities is None else float(
            np.asarray(probabilities)[start:end, cls].mean())
        out.append((cls, int(start), int(end), score))
    return out


def _iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[1], b[1]))
    union = max(a[2], b[2]) - min(a[1], b[1])
    return inter / union if union else 0.0


def _ap(rec, prec):
    mrec = np.r_[0.0, rec, 1.0]
    mpre = np.r_[0.0, prec, 0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1]) + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))


def tal_map(records, thresholds=TAL_THRESHOLDS):
    """TAL mAP for records with keys id, true, pred, probabilities.

    Background is excluded. AP is averaged over the 18 action classes; an action
    absent from the validation ground truth receives AP=0, matching the fixed
    official label-space convention used by frame metrics.
    """
    gt_by_class = {c: {} for c in range(BACKGROUND_ID)}
    pred_by_class = {c: [] for c in range(BACKGROUND_ID)}
    for rec in records:
        rid = str(rec["id"])
        for seg in _segments(rec["true"]):
            gt_by_class[seg[0]].setdefault(rid, []).append(seg)
        for seg in _segments(rec["pred"], rec.get("probabilities")):
            pred_by_class[seg[0]].append((rid, *seg[1:]))
    result = {}
    for threshold in thresholds:
        aps = []
        for cls in range(BACKGROUND_ID):
            preds = sorted(pred_by_class[cls], key=lambda x: x[3], reverse=True)
            total_gt = sum(len(v) for v in gt_by_class[cls].values())
            used = {rid: np.zeros(len(v), bool) for rid, v in gt_by_class[cls].items()}
            tp, fp = [], []
            for rid, start, end, _score in preds:
                candidates = gt_by_class[cls].get(rid, [])
                overlaps = np.asarray([_iou((cls, start, end, 0), g) for g in candidates])
                j = int(overlaps.argmax()) if overlaps.size else -1
                ok = j >= 0 and overlaps[j] >= threshold and not used[rid][j]
                tp.append(float(ok)); fp.append(float(not ok))
                if ok:
                    used[rid][j] = True
            if total_gt == 0:
                aps.append(0.0)
            elif not preds:
                aps.append(0.0)
            else:
                tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
                aps.append(_ap(tp_c / total_gt, tp_c / np.maximum(tp_c + fp_c, 1e-12)))
        result[f"map_at_{threshold:.1f}"] = float(np.mean(aps))
    result["avg_map"] = float(np.mean(list(result.values())))
    return result


def tal_map_present_gt_classes(records, thresholds=TAL_THRESHOLDS):
    """ANET-style class averaging over action classes present in GT.

    This retains our frame-derived segments and scores, so it is an additional
    audit view rather than an ActionFormer-output reproduction.
    """
    gt_by_class = {c: {} for c in range(BACKGROUND_ID)}
    pred_by_class = {c: [] for c in range(BACKGROUND_ID)}
    for rec in records:
        rid = str(rec["id"])
        for seg in _segments(rec["true"]):
            gt_by_class[seg[0]].setdefault(rid, []).append(seg)
        for seg in _segments(rec["pred"], rec.get("probabilities")):
            pred_by_class[seg[0]].append((rid, *seg[1:]))
    present = [c for c in range(BACKGROUND_ID)
               if sum(map(len, gt_by_class[c].values())) > 0]
    result = {}
    for threshold in thresholds:
        aps = []
        for cls in present:
            preds = sorted(pred_by_class[cls], key=lambda x: x[3], reverse=True)
            total_gt = sum(len(v) for v in gt_by_class[cls].values())
            used = {rid: np.zeros(len(v), bool) for rid, v in gt_by_class[cls].items()}
            tp, fp = [], []
            for rid, start, end, _score in preds:
                candidates = gt_by_class[cls].get(rid, [])
                overlaps = np.asarray([_iou((cls, start, end, 0), g) for g in candidates])
                j = int(overlaps.argmax()) if overlaps.size else -1
                ok = j >= 0 and overlaps[j] >= threshold and not used[rid][j]
                tp.append(float(ok)); fp.append(float(not ok))
                if ok:
                    used[rid][j] = True
            if not preds:
                aps.append(0.0)
            else:
                tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
                aps.append(_ap(tp_c / total_gt, tp_c / np.maximum(tp_c + fp_c, 1e-12)))
        result[f"map_at_{threshold:.1f}"] = float(np.mean(aps)) if aps else float("nan")
    result["avg_map"] = float(np.nanmean(list(result.values())))
    result["present_gt_classes"] = present
    return result
