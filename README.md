# BRD4-QSAR

Interpretable ligand-based regression models for BRD4 inhibitor potency,
with scaffold-grouped validation and structural interpretation.
Code and results accompany the manuscript submitted to *Discover Chemistry*.

## What this repository contains
- Curated ChEMBL BRD4 dataset (6,417 structures; median pIC50 per structure)
- Scaffold-group assignments (5,134 training / 1,283 test compounds)
- Fixed 250-feature set (103 Morgan bits, 146 Mordred 2D descriptors, MolLogP)
- Trained ExtraTrees, CatBoost and SVR models
- Analysis code: cross-validation, Y-randomisation, applicability domain,
  TreeSHAP, atom-level mapping of Morgan bits, matched molecular pairs,
  activity cliffs, scaffold analysis
- Result tables and figures used in the paper

## Main results (locked test set)
| Quantity | Value |
|---|---|
| R² | 0.798–0.811 |
| RMSE (pIC50) | 0.562–0.581 |
| Spearman ρ | 0.89–0.90 |
| Enrichment of pIC50 ≥ 8 in top 10% | 8- to 9-fold |

A similarity-weighted nearest-neighbour baseline performed comparably, so
the scores apply to compounds in or near the represented chemical series.

## Repository layout
<adjust to your folders>
data/        curated dataset, scaffold groups, feature list
notebooks/   analysis notebooks (run in numbered order)
models/      saved models
results/     tables and figures
src/         helper functions

## Setup
Python 3.10. Install dependencies from `requirements.txt`
(package versions are pinned). Random seed: 42.

## Reproducing the analysis
1. Run the notebooks in order.
2. The 250-feature list is fixed and reused unchanged in every fold,
   permutation and model.
3. The locked test set is used only for final evaluation.

## Web application
Predictions (pIC50 from the three models, plus applicability-domain flags):
https://brd-4-frontend.vercel.app
SHAP explanations are not returned by the application.

## Limitations
Labels are drawn from mixed assays; some repeated values (e.g. pIC50 = 4.2967)
appear to be assigned potency levels. SHAP values describe model behaviour,
not causal substituent effects. No prospective testing was performed.

## Citation
<add the DOI after publication>

## License
<choose, e.g. MIT for code>
