# WEAR final-v1 checkpoint layout

Weights are intentionally not committed. After obtaining the trusted model
archive, extract it so that each fold contains `parent.pt` and
`background_probe.pt`, then run:

```bash
python scripts/verify_model_files.py
```

All 36 expected files, byte sizes, and SHA-256 values are recorded in
`manifest.json`. No public download URL was available when this release
candidate was assembled.
