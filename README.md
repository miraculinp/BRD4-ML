<<<<<<< HEAD
# BRD4 ligand-based QSAR pipeline (through LazyPredict)

This project implements Stages 00–06 from setup through training-only LazyPredict screening. It does not contain or infer the source assay data. Place the supplied `data.csv` at the project root with exactly these columns: `Molecule ID`, `Smiles`, and `IC50 (nM)`.

## Important data boundary

Stage 04 creates a Bemis–Murcko scaffold-group holdout split. The IDs are recorded in `outputs/04_train_test_split/tables/test_ids.csv`. Stages 05 and 06 load only training IDs and training feature rows. Do not use the locked test set for preprocessing choices, feature selection, LazyPredict, or hyperparameter tuning.

## Run

Use the requested Windows environment in PowerShell:

```powershell
cd C:\Users\USER\Documents\E\Code
& C:\Users\USER\Documents\E\BRD4\brd4_env\python.exe -m pip install -r requirements.txt
& C:\Users\USER\Documents\E\BRD4\brd4_env\python.exe -m src.load_data
& C:\Users\USER\Documents\E\BRD4\brd4_env\python.exe -m src.clean_data
& C:\Users\USER\Documents\E\BRD4\brd4_env\python.exe -m src.features
& C:\Users\USER\Documents\E\BRD4\brd4_env\python.exe -m src.split
& C:\Users\USER\Documents\E\BRD4\brd4_env\python.exe -m src.feature_selection
& C:\Users\USER\Documents\E\BRD4\brd4_env\python.exe -m src.lazy_predict_screen
```

Stage 00 notebook initializes directories and prints dependency versions. Each stage also has a notebook in `notebooks/`.

## Notes

- pIC50 is calculated as `9 - log10(IC50 in nM)` only for positive numeric IC50 values.
- Duplicate canonical isomeric SMILES are consolidated using median pIC50. All original ChEMBL IDs are retained as a semicolon-separated `chembl_id` value and in `chembl_ids`.
- ECFP4 fingerprints use radius 2, 2048 bits, and chirality. Mordred is best effort; if unavailable or a descriptor calculation fails, the script reports it and continues with available features.
- 3D conformers are generated for descriptor calculation only. This is a ligand-based QSAR workflow; it contains no docking or protein modeling.
- LazyPredict is run independently within each GroupKFold fold; fold metrics are aggregated for ranking. Small datasets may not support all estimators or the requested number of folds.
- Generated figure files are saved as 1200-dpi PNG and PDF.
=======
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
>>>>>>> e8a1d3653ba1deb6edb1f437a7dabb20d1105eda
