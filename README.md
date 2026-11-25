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
