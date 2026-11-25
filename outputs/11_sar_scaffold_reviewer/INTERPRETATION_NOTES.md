# Stage 11 interpretation notes (combined SAR / reviewer evidence)

- Parts 0–C used 5,134 training molecules only; locked-test labels, predictions and SHAP values were first read in Part D, which selects nothing.
- Fingerprints were rebuilt and matched Stage 03 exactly; the fixed Stage 05 list of 250 features and the saved Stage 08 models were used for prediction/explanation only. No final model was refitted and no feature selection was repeated.
- Scaffold, bit, MMP and cliff results are descriptive associations. Pairs overlap in molecules and series; assay heterogeneity, rounding and curation can create apparent cliffs. A Morgan bit may mark a chemotype, a substituent, or a hashed collision of several environments (see Part B environments).
- MMP gains are descriptive associations, not measurements or mechanistic claims.
- Test uncertainty is a scaffold-group cluster bootstrap (2000 resamples; groups: saved column 'group' in Stage 08 predictions); all methods share the same resamples so paired differences are valid. Ridge/1-NN/5-NN/mean baselines are fitted on training molecules only.
- SHAP values describe the fitted tree models on the test set and are not causal evidence of binding.
