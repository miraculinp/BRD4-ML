# Stage 12 interpretation notes

- Candidate generation used Stage 11 supported enriching single-cut MMP rules and exact training core/substituent contexts. Low-activity parents and primary activity-cliff members were prioritized for generation.
- 71 novel analogue structures were generated; the fixed Stage 05 feature set of 250 columns was reconstructed. Any candidate with a missing or non-finite selected descriptor was excluded without imputation.
- The saved Stage 08 ExtraTrees, CatBoost, and SVR models were used only for prediction. No model was refitted and no feature selection was repeated.
- Recommendations require nearest training Morgan Tanimoto >=0.65 and pass all three training-defined AD methods (standardized 5-NN, bounding box, leverage). Greedy diversity avoids selecting candidates with Morgan Tanimoto >=0.90 to an already selected recommendation. If no candidates meet these criteria, no recommendation is made.
- Predicted pIC50 values and MMP median gains are computational hypotheses, not observed measurements. SHAP values describe fitted model behavior and are not causal evidence. Candidates require synthesis/availability review and experimental testing. No mechanistic or binding-pocket claims are made.
- Locked-test activities, predictions, and test SHAP files were not used in candidate generation or ranking.
