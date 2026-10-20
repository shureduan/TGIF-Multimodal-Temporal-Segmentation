# Envelope-shape DWA weights

The manifest identifies all 18 LOSO parent/probe pairs for seed 47 and checks
archive and individual checkpoint SHA-256 values. Protocol:
`envelope_shape_loso_v4_pilot`. Parent/probe budgets: 30/15 fixed last epochs.

```bash
python scripts/download_shape_weights.py
python scripts/download_shape_weights.py --verify-only
```

Weights are stored as release assets rather than Git objects. The downloader
never overwrites a differing existing checkpoint. See
[the v4 reproduction guide](../../docs/envelope_shape_pilot.md).
