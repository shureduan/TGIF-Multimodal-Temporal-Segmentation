"""Dynamic window manager (README §4-§6, §22).

- fixed windows: 6 s window / 3 s stride
- predicted-segment windows: per-frame preds → 1 s majority-vote smoothing → merge runs
  → merge fragments < 2 s → split runs > 30 s → (start,end) frame windows
- curriculum probability p_seg(e) = clip(p_max·(e-E_start)/E_transition, 0, p_max)
- segment cache refresh (no_grad, from the video pre-classifier)

Windows are (start_frame, end_frame) tuples on the video frame grid; the
predicted CLASS never enters the vibration tokens (README §4.2 禁止).
"""
import numpy as np


def build_fixed_windows(T, fps, window_seconds=6.0, stride_seconds=3.0):
    w = max(1, int(round(window_seconds * fps)))
    s = max(1, int(round(stride_seconds * fps)))
    out = []
    a = 0
    while a < T:
        out.append((a, min(a + w, T)))
        if a + w >= T:
            break
        a += s
    return out


def _median_smooth(pred, k):
    """Historical helper name; this is a categorical majority-vote filter."""
    if k <= 1:
        return pred.copy()
    k = k + 1 if k % 2 == 0 else k
    r = k // 2
    p = np.pad(pred, (r, r), mode="edge")
    return np.array([np.bincount(p[i:i + k], minlength=3).argmax()
                     for i in range(len(pred))])


def predict_segments(pre_pred, fps, smoothing_seconds=1.0,
                     min_segment_seconds=2.0, max_segment_seconds=30.0):
    """pre_pred: [T] int per-frame prediction → list of (a,b) frame windows.
    Only boundaries are returned (class is intentionally dropped)."""
    T = len(pre_pred)
    if T == 0:
        return []
    sm = _median_smooth(np.asarray(pre_pred, int), int(round(smoothing_seconds * fps)))
    # contiguous runs
    runs, i = [], 0
    while i < T:
        j = i
        while j + 1 < T and sm[j + 1] == sm[i]:
            j += 1
        runs.append([i, j + 1]); i = j + 1
    # merge fragments shorter than min into the longer neighbour
    min_f = int(round(min_segment_seconds * fps))
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for k, (a, b) in enumerate(runs):
            if b - a >= min_f:
                continue
            if k == 0:
                runs[1][0] = a; runs.pop(0)
            elif k == len(runs) - 1:
                runs[-2][1] = b; runs.pop(-1)
            else:
                left, right = runs[k - 1], runs[k + 1]
                if (left[1] - left[0]) >= (right[1] - right[0]):
                    left[1] = b
                else:
                    right[0] = a
                runs.pop(k)
            changed = True
            break
    # split runs longer than max into equal chunks
    max_f = max(min_f, int(round(max_segment_seconds * fps)))
    out = []
    for a, b in runs:
        L = b - a
        if L <= max_f:
            out.append((a, b))
        else:
            n = int(np.ceil(L / max_f))
            step = int(np.ceil(L / n))
            for s in range(a, b, step):
                out.append((s, min(s + step, b)))
    return out


# piecewise-constant schedule: (from_epoch, p_seg); applies from that epoch on
SCHEDULE = [(0, 0.10), (5, 0.50), (15, 0.80), (30, 0.95), (60, 1.00)]


def segment_probability(epoch, schedule=None):
    """Schedule: 0-5→0.10, 5-15→0.50, 15-30→0.80, 30-60→0.95, 60+→1.00."""
    p = 0.0
    for e0, v in (schedule or SCHEDULE):
        if epoch >= e0:
            p = v
    return float(p)


# ── multi-scale tokens + candidate mask (anti-collapse README §7-§9) ──────────

def build_multiscale_tokens(segments, T, fps, win_s, stride_s):
    """Each segment [a,b] → 1 GLOBAL token (whole segment) + LOCAL tokens
    (win_s/stride_s inside the segment). Returns:
      windows : list of (start,end) frame windows, one per token
      mask    : [T, n_tokens] bool candidate set — for frame t in segment k:
                {global_k, local covering t, prev local, next local}; if a
                segment yields <2 candidates, neighbour segments' globals are
                added (never duplicate a token to pad — README §9).
      kinds   : list[str] 'global'/'local' per token (for logging)
    `segments` may be predicted segments OR, in fixed mode, a single [0,T]
    'segment' so the whole clip still gets a global + local tokens."""
    windows, kinds = [], []
    seg_global, seg_locals, frame_seg = [], [], np.full(T, -1, int)
    for k, (a, b) in enumerate(segments):
        gi = len(windows); windows.append((a, b)); kinds.append("global")
        seg_global.append(gi)
        frame_seg[a:b] = k
        locals_k = []
        for la, lb in build_fixed_windows(b - a, fps, win_s, stride_s):
            li = len(windows); windows.append((a + la, a + lb)); kinds.append("local")
            locals_k.append(li)
        seg_locals.append(locals_k)
    frame_seg[frame_seg < 0] = 0                       # any gap → first segment

    n = len(windows)
    centers = np.array([(a + b) / 2 for a, b in windows])
    mask = np.zeros((T, n), bool)
    for t in range(T):
        k = frame_seg[t]
        cand = [seg_global[k]]
        loc = seg_locals[k]
        if loc:
            cover = [i for i, li in enumerate(loc) if windows[li][0] <= t < windows[li][1]]
            r = cover[len(cover) // 2] if cover else int(
                np.abs(np.array([centers[li] for li in loc]) - t).argmin())
            for rr in (r - 1, r, r + 1):
                if 0 <= rr < len(loc):
                    cand.append(loc[rr])
        if len(cand) < 2:                              # neighbour-segment globals, no dup
            for kk in (k - 1, k + 1):
                if 0 <= kk < len(seg_global):
                    cand.append(seg_global[kk])
        mask[t, list(dict.fromkeys(cand))] = True
    return windows, mask, kinds


def build_segment_tokens(segments, T, fps, min_seconds=2.0, neighbors=1):
    """Adaptive: ONE token per segment, its length = the (video-predicted) segment
    length. Segments shorter than `min_seconds` are widened symmetrically for a
    stable vibration read, but the frame→token adjacency uses the ORIGINAL segment
    extents. Returns (windows [clamped, for features], mask [T,n], kinds)."""
    import numpy as np
    min_f = max(1, int(round(min_seconds * fps)))
    wins = []
    for a, b in segments:
        if b - a < min_f:
            c = (a + b) // 2
            a2 = max(0, min(c - min_f // 2, max(T - min_f, 0)))
            wins.append((int(a2), int(min(a2 + min_f, T))))
        else:
            wins.append((int(a), int(b)))
    n = len(wins)
    frame_seg = np.full(T, -1, int)
    for k, (a, b) in enumerate(segments):
        frame_seg[max(0, a):min(T, b)] = k
    frame_seg[frame_seg < 0] = 0
    mask = np.zeros((T, n), bool)
    for t in range(T):
        k = frame_seg[t]
        mask[t, max(0, k - neighbors):min(n, k + neighbors + 1)] = True
    return wins, mask, ["seg"] * n
