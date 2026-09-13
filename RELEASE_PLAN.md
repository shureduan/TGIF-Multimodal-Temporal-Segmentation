# Remaining release work after the presentation update

The repository now treats the maintained final DWA model as the code baseline
and the confirmed TGIF figures as the result record. Before public release:

1. Publish the 18 WEAR parent/probe pairs as one immutable archive. Add a real
   download URL only after all files pass `models/wear_final/manifest.json`.
2. Document how users obtain or prepare the exact WEAR feature-grid inputs, then
   run one licensed sample through the fresh-clone inference/evaluation chain.
3. Decide which TGIF assets, if any, may be distributed. The confirmed figures
   and numbers remain unchanged whether or not private data are released.
4. If public TGIF inference is desired, add a small dataset adapter and a
   complete model bundle—checkpoint, configuration, normalization, labels, and
   input metadata—on top of the maintained DWA interfaces.
5. Add a three-track WEAR GT/video-only/multimodal timeline when an approved
   figure is available. Until then, keep the existing GT-versus-final trace
   explicitly labeled as a two-track qualitative example.
6. Choose the project code license and add stable project citation metadata.
7. Keep the current 2 Hz WEAR result label. Add official 50 Hz record scoring or
   official detector-format TAL only as separately named, fixture-tested modes.
8. After model and sample-data publication, perform the public clean-clone
   check: install, verify weights, infer, evaluate,
   compare with the frozen expected output, and inspect README image rendering.
