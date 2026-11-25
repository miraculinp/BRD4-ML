# BRD4 QSAR prediction API (Render, Python 3.10)

This is a standalone FastAPI API for the saved ExtraTrees, CatBoost, and SVR models. It uses the fixed Stage 05 feature list and training-only applicability-domain reference. It does not fit models, tune parameters, impute missing descriptors, or use 3D descriptors.

## Add the model assets before deployment

From the completed research project root, use the Python 3.10 environment that can load the saved Stage 08 models. Point the preparation script at this backend repository's `app_assets` directory:

```powershell
python prepare_stage13_assets.py --assets-dir "<path-to-this-backend-repo>\app_assets"
```

The script needs Stage 05 selected features and training features, Stage 08 model bundles, and `config.json`. It writes `app_config.json`, `selected_features.csv`, `training_reference.joblib`, and the three model bundles. Add those generated files to this backend repository before pushing it. Test IDs, test activities, and test predictions are not copied.

Run a model-load and prediction preflight with the same Python 3.10 environment:

```powershell
python -m pip install -r requirements.txt
python validate_assets.py
```

Do not deploy until this preflight passes.

## Deploy to Render

1. Push this backend folder's contents to its own Git repository.
2. In Render, create a Blueprint instance from that backend repository. Render reads `render.yaml`; its Dockerfile uses `python:3.10-slim` and installs the pinned dependencies.
3. Set `CORS_ORIGINS` to the exact Vercel frontend URL, such as `https://brd4-qsar.vercel.app`.
4. Confirm `/api/health` returns status `ok` and the three loaded model names.

## Local run

```bash
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

The API accepts pasted SMILES and SDF/MOL2 uploads up to 20 MB and 100 molecules per request. It returns pIC50 predictions, standardized 5-NN distance, training-range/bounding-box status, leverage status, combined AD status, and not-scored reasons. The fixed model set is 103 chirality-aware Morgan bits, 146 Mordred Community 2D descriptors, and `phys_MolLogP`.
