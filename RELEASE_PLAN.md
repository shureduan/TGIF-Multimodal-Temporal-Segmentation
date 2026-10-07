# Distribution roadmap

The current WEAR release provides sensor-driven DWA, reference baselines,
versioned checkpoints and reproducible numerical figures. Follow-up work:

1. Preserve the sensor-driven DWA release and its checkpoint manifest under
   `models/wear_signal_v3/`. WEAR v2 checkpoints remain in the
   [pretrained release](https://github.com/shureduan/TGIF-Multimodal-Temporal-Segmentation/releases/tag/wear-v2-pretrained-20261006): 54 Final parent/probe pairs and 162 baseline parents. Keep its assets
   immutable; use a new version and manifest for future experiments.
2. Document how users obtain or prepare the exact WEAR feature-grid inputs, then
   run one licensed sample through the fresh-clone inference/evaluation chain.
3. Decide which TGIF assets, if any, may be distributed. The confirmed figures
   and numbers remain unchanged whether or not private data are released.
4. If public TGIF inference is desired, add a small dataset adapter and a
   complete model bundle—checkpoint, configuration, normalization, labels, and
   input metadata—on top of the maintained DWA interfaces.
5. Choose the project code license and add stable project citation metadata.
6. Keep the current 2 Hz WEAR result label. Add official 50 Hz record scoring or
   official detector-format TAL only as separately named, fixture-tested modes.
7. After model and sample-data publication, perform the public clean-clone
   check: install, verify weights, infer, evaluate,
   compare with the frozen expected output, and inspect README image rendering.
