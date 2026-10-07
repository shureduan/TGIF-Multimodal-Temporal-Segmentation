# Numerical result sets

- `wear_signal_v3/`: current sensor-driven DWA, four reference models, matched
  retrained controls, frozen outputs, paired statistics and figure inputs.
- `wear_18fold_metrics.csv`, `wear_summary.csv`, `wear_aggregate.json`, and
  `wear_figure_data/`: unchanged fixed-epoch v2 result record; its `FINAL_MODEL`
  is the SWA reference, not sensor-driven DWA.
- TGIF numerical files and assets remain unchanged.

See [metric definitions and cohorts](../docs/results.md) and
[reproduction commands](../docs/reproduction.md). Figures are generated from
numerical inputs; pretrained weights live in versioned GitHub Releases.
