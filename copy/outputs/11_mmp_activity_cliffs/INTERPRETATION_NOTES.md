# Stage 11 interpretation notes

- Analysis used 5,134 training/development molecules only. Locked-test pIC50 labels were not read.
- Morgan fingerprints were regenerated with radius 2, 2048 bits, chirality=True; every bit vector matched the Stage 03 raw ECFP4 table exactly.
- Primary activity cliffs required chiral Morgan Tanimoto ≥0.90 and |ΔpIC50| ≥2.0. Threshold sensitivity is in `activity_cliff_threshold_sensitivity.csv`.
- MMP analysis splits the dot-disconnected fragments returned by one-cut RDKit MMPA records, tests both core/sidechain orientations, and saves per-molecule diagnostics. If no exact pairs are present, cliff/stereo analyses continue. MMP directions use a deterministic lexical fragment order. A positive mean ΔpIC50 means fragment_from → fragment_to was associated with increased potency. Supported transformations require at least 3 pairs and ≥60% directional consistency; depleting rules require consistent negative deltas.
- The OOF pair table reports same-fold and cross-fold comparisons separately. Same-fold pairs were held out together by the same CV model; cross-fold pairs were not jointly held out by one model and are weaker evidence.
- Pairs share compounds and chemical series, so pair counts are not independent observations. MMP/cliff associations are descriptive and can reflect assay heterogeneity, activity rounding/censoring, or curation effects.
- Stereochemical pairs represent different recorded isomeric structures with the same stereo-removed canonical connectivity. Database records alone do not establish experimental confirmation or assay comparability of the stereoisomers.
- No models were refitted, no features were selected, and no candidate molecules were generated. Candidate prioritization is Stage 12.
