"""Stage 10B: fixed-250 family ablation, family SHAP summaries, and Morgan maps.

Stage 05's exact 250 selected features are reused throughout. No Stage 03
3291-feature ranking is repeated. No test activities are loaded.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from src.feature_selection import family
from src.optuna_tuning import THREADS, make_estimator
from src.utils import OUTPUTS, load_config, stage_dir

MODELS = ("ExtraTreesRegressor", "CatBoostRegressor")
FAMILIES = ("Morgan", "Mordred 2D", "Mordred 3D", "Physicochemical")


def _fixed250_training_data():
    feature_path = OUTPUTS / "05_feature_selection/tables/train_selected_features.parquet"
    target_path = OUTPUTS / "05_feature_selection/tables/train_target.parquet"
    groups_path = OUTPUTS / "04_train_test_split/tables/train_ids.csv"
    selected_path = OUTPUTS / "05_feature_selection/tables/selected_features.csv"
    for path in (feature_path, target_path, groups_path, selected_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required fixed-250 Stage 10B input missing: {path}")
    selected = pd.read_csv(selected_path)["feature"].astype(str).tolist()
    if len(selected) != 250 or len(set(selected)) != 250:
        raise ValueError(f"Expected exactly 250 unique Stage 05 features; found {len(selected)}.")
    X = pd.read_parquet(feature_path)
    y = pd.read_parquet(target_path)["pIC50"]
    ids = pd.read_csv(groups_path, dtype={"chembl_id": str})
    X.index = X.index.astype(str)
    y.index = y.index.astype(str)
    ids["chembl_id"] = ids.chembl_id.astype(str)
    X = X.reindex(ids.chembl_id).loc[:, selected].apply(pd.to_numeric, errors="coerce")
    y = y.reindex(ids.chembl_id)
    groups = ids["group"].to_numpy()
    if X.shape != (len(ids), 250) or X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("The fixed Stage 05 training matrix must be complete and finite; no imputation is used.")
    if y.isna().any() or not np.isfinite(y.to_numpy(dtype=float)).all() or pd.isna(groups).any():
        raise ValueError("Training targets or scaffold groups are missing/non-finite.")
    return X, y, groups, selected


def _load_params(model_name: str) -> dict:
    path = OUTPUTS / "07_optuna_tuning/tables/optuna_best_params.json"
    if not path.is_file():
        raise FileNotFoundError(f"Optuna parameter file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return dict(payload["params_by_model"][model_name]["params"])
    except KeyError as exc:
        raise ValueError(f"Optuna best parameters missing for {model_name}.") from exc


def run_fixed250_family_ablation(model_name: str) -> pd.DataFrame:
    """Compare fixed-250 full vs leave-one-family-out models on scaffold CV.

    The selected feature list is intentionally held constant from Stage 05.
    Since Stage 05 selected it using the full training partition, these CV
    metrics are a fixed-feature sensitivity analysis, not a fresh independent
    performance estimate and must not replace Stage 08's reported CV.
    """
    if model_name not in MODELS:
        raise ValueError(f"Choose one of {MODELS}.")
    cfg = load_config()
    X, y, groups, selected = _fixed250_training_data()
    families = pd.Series([family(col) for col in selected], index=selected)
    variants = {"all_features": None}
    for family_name in FAMILIES:
        variants[f"without_{family_name.lower().replace(' ', '_')}"] = family_name
    params = _load_params(model_name)
    cv = GroupKFold(n_splits=5, shuffle=True, random_state=int(cfg.get("seed", 42)) + 2026)
    stage = stage_dir("10b_interpretation")
    rows = []

    for variant, excluded_family in variants.items():
        candidate_cols = selected if excluded_family is None else families.index[families != excluded_family].tolist()
        if not candidate_cols:
            raise ValueError(f"No fixed-250 features remain for {variant}.")
        print(f"{model_name} | {variant} | candidate features={len(candidate_cols)}", flush=True)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups), start=1):
            X_tr, X_va = X.iloc[train_idx].loc[:, candidate_cols], X.iloc[valid_idx].loc[:, candidate_cols]
            y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]
            estimator = make_estimator(model_name, params, int(cfg.get("seed", 42)) + fold)
            with threadpool_limits(limits=THREADS):
                estimator.fit(X_tr, y_tr)
                pred = np.asarray(estimator.predict(X_va), dtype=float)
            obs = y_va.to_numpy(dtype=float)
            if not np.isfinite(pred).all():
                raise ValueError(f"Non-finite predictions for {model_name}/{variant}/fold {fold}.")
            rows.extend(
                {"variant": variant, "model": model_name, "fold": fold,
                 "chembl_id": str(molecule_id), "group": str(group),
                 "observed_pIC50": float(actual), "predicted_pIC50": float(predicted),
                 "residual": float(actual - predicted), "n_features": len(candidate_cols)}
                for molecule_id, group, actual, predicted in zip(
                    X.index[valid_idx], np.asarray(groups)[valid_idx], obs, pred
                )
            )
            print(f"  fold {fold}/5: RMSE={np.sqrt(mean_squared_error(obs, pred)):.4f}", flush=True)

    oof = pd.DataFrame(rows)
    metric_rows = []
    for variant, sub in oof.groupby("variant", sort=False):
        fold_rows = []
        for fold_number, fold_frame in sub.groupby("fold", sort=True):
            fold_rows.append({
                "fold": fold_number,
                "rmse": float(np.sqrt(mean_squared_error(fold_frame.observed_pIC50, fold_frame.predicted_pIC50))),
                "mae": float(mean_absolute_error(fold_frame.observed_pIC50, fold_frame.predicted_pIC50)),
                "r2": float(r2_score(fold_frame.observed_pIC50, fold_frame.predicted_pIC50)),
            })
        fold_metrics = pd.DataFrame(fold_rows)
        metric_rows.append({
            "model": model_name, "variant": variant,
            "mean_fold_rmse": float(fold_metrics.rmse.mean()),
            "sd_fold_rmse": float(fold_metrics.rmse.std(ddof=1)),
            "pooled_oof_rmse": float(np.sqrt(mean_squared_error(sub.observed_pIC50, sub.predicted_pIC50))),
            "pooled_oof_mae": float(mean_absolute_error(sub.observed_pIC50, sub.predicted_pIC50)),
            "pooled_oof_r2": float(r2_score(sub.observed_pIC50, sub.predicted_pIC50)),
            "n_features": int(sub.n_features.iloc[0]),
            "delta_rmse_vs_all_features": np.nan,
        })
    summary = pd.DataFrame(metric_rows)
    baseline = float(summary.loc[summary.variant == "all_features", "pooled_oof_rmse"].iloc[0])
    summary["delta_rmse_vs_all_features"] = summary.pooled_oof_rmse - baseline
    summary.to_csv(stage / "tables" / f"fixed250_family_ablation_{model_name}.csv", index=False)
    oof.to_csv(stage / "tables" / f"fixed250_family_ablation_oof_{model_name}.csv", index=False)
    return summary.sort_values("pooled_oof_rmse", kind="stable").reset_index(drop=True)


def _load_fixed250_shap(model_name: str):
    if model_name not in MODELS:
        raise ValueError(f"Choose one of {MODELS}.")
    selected_path = OUTPUTS / "05_feature_selection/tables/selected_features.csv"
    shap_path = OUTPUTS / f"10_shap_interpretation/tables/test_shap_values_{model_name}.parquet"
    ids_path = OUTPUTS / "04_train_test_split/tables/test_ids.csv"
    pred_path = OUTPUTS / "08_final_model_evaluation/tables/locked_test_predictions_all_models.csv"
    for path in (selected_path, shap_path, ids_path, pred_path):
        if not path.is_file():
            raise FileNotFoundError(f"Run Stage 10 first; missing: {path}")
    selected = pd.read_csv(selected_path)["feature"].astype(str).tolist()
    if len(selected) != 250:
        raise ValueError(f"Expected 250 Stage 05 features; found {len(selected)}.")
    shap_frame = pd.read_parquet(shap_path)
    if "chembl_id" not in shap_frame.columns or set(selected) - set(shap_frame.columns):
        raise ValueError(f"Stage 10 SHAP matrix is missing selected features for {model_name}.")
    shap_frame["chembl_id"] = shap_frame.chembl_id.astype(str)
    if shap_frame.chembl_id.duplicated().any():
        raise ValueError(f"Duplicate test IDs in Stage 10 SHAP values for {model_name}.")
    test_ids = pd.read_csv(ids_path, dtype={"chembl_id": str})
    test_ids["chembl_id"] = test_ids.chembl_id.astype(str)
    if set(test_ids.chembl_id) != set(shap_frame.chembl_id):
        raise ValueError(f"Stage 10 SHAP molecule IDs do not match locked test IDs for {model_name}.")
    shap_frame = shap_frame.set_index("chembl_id").reindex(test_ids.chembl_id)
    if not np.isfinite(shap_frame.loc[:, selected].to_numpy(dtype=float)).all():
        raise ValueError("Stage 10 SHAP values contain missing/non-finite entries.")
    predictions = pd.read_csv(pred_path, dtype={"chembl_id": str})
    predictions["chembl_id"] = predictions.chembl_id.astype(str)
    predictions = predictions.loc[predictions.model == model_name].copy()
    predictions = predictions.set_index("chembl_id").reindex(test_ids.chembl_id)
    if predictions.predicted_pIC50.isna().any():
        raise ValueError(f"Stage 08 predictions do not align for {model_name}.")
    return selected, shap_frame.loc[:, selected], predictions


def summarize_fixed250_shap_families() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize family-level SHAP contributions for the fixed Stage 08 models."""
    stage = stage_dir("10b_interpretation")
    summary_rows, molecule_rows = [], []
    for model_name in MODELS:
        selected, shap_frame, _ = _load_fixed250_shap(model_name)
        family_by_feature = {feature_name: family(feature_name) for feature_name in selected}
        values = shap_frame.to_numpy(dtype=float)
        for family_name in FAMILIES:
            names = [f for f in selected if family_by_feature[f] == family_name]
            if not names:
                summary_rows.append({"model": model_name, "feature_family": family_name,
                                     "n_features": 0, "mean_abs_family_contribution": 0.0,
                                     "mean_signed_family_contribution": 0.0,
                                     "share_of_total_abs_contribution_pct": 0.0})
                continue
            indices = [selected.index(name) for name in names]
            block = values[:, indices]
            row_abs = np.abs(block).sum(axis=1)
            row_signed = block.sum(axis=1)
            for chembl_id, abs_value, signed_value in zip(shap_frame.index.astype(str), row_abs, row_signed):
                molecule_rows.append({"model": model_name, "chembl_id": chembl_id,
                                       "feature_family": family_name,
                                       "sum_abs_shap": float(abs_value),
                                       "sum_signed_shap": float(signed_value)})
            summary_rows.append({"model": model_name, "feature_family": family_name,
                                 "n_features": len(names),
                                 "mean_abs_family_contribution": float(row_abs.mean()),
                                 "mean_signed_family_contribution": float(row_signed.mean()),
                                 "share_of_total_abs_contribution_pct": np.nan})
    summary = pd.DataFrame(summary_rows)
    for model_name, indices in summary.groupby("model").groups.items():
        total = float(summary.loc[indices, "mean_abs_family_contribution"].sum())
        if total:
            summary.loc[indices, "share_of_total_abs_contribution_pct"] = (
                100 * summary.loc[indices, "mean_abs_family_contribution"] / total
            )
    molecule_detail = pd.DataFrame(molecule_rows)
    summary.to_csv(stage / "tables/fixed250_shap_family_contributions.csv", index=False)
    molecule_detail.to_csv(stage / "tables/fixed250_shap_family_by_test_molecule.csv", index=False)
    return summary, molecule_detail


def draw_fixed250_morgan_highlights(model_name: str, cases: tuple[str, ...] = ("low", "median", "high")):
    """Map Morgan-bit SHAP values from Stage 08 models to locked-test structures."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw, rdFingerprintGenerator
    except ImportError as exc:
        raise ImportError("Morgan atom highlights require RDKit.") from exc
    cfg = load_config()
    selected, shap_frame, predictions = _load_fixed250_shap(model_name)
    morgan_features = [name for name in selected if name.startswith("ecfp4_")]
    stage = stage_dir("10b_interpretation")
    if not morgan_features:
        print(f"{model_name}: none of the fixed 250 features are Morgan bits, so no Morgan-bit structure map can be drawn for this model.")
        return pd.DataFrame(), pd.DataFrame(), []

    ids_path = OUTPUTS / "04_train_test_split/tables/test_ids.csv"
    smiles_path = OUTPUTS / "02_clean_data/tables/cleaned_activity.csv"
    test_ids = pd.read_csv(ids_path, dtype={"chembl_id": str})
    test_ids["chembl_id"] = test_ids.chembl_id.astype(str)
    smiles = pd.read_csv(smiles_path, usecols=["chembl_id", "canonical_smiles"], dtype={"chembl_id": str})
    smiles["chembl_id"] = smiles.chembl_id.astype(str)
    smiles_map = smiles.set_index("chembl_id").canonical_smiles
    absent_smiles = set(test_ids.chembl_id) - set(smiles_map.index)
    if absent_smiles:
        raise ValueError(f"Missing locked-test structures for {len(absent_smiles)} molecules.")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(cfg.get("fingerprint_radius", 2)),
        fpSize=int(cfg.get("fingerprint_bits", 2048)),
        includeChirality=bool(cfg.get("use_chirality", True)),
    )
    positions = {"low": 0, "median": len(predictions) // 2, "high": len(predictions) - 1}
    ordered_ids = predictions.predicted_pIC50.sort_values(kind="stable").index.tolist()
    case_rows, environment_rows, images = [], [], []
    positive = np.array([0.95, 0.43, 0.10])
    negative = np.array([0.10, 0.45, 0.80])
    neutral = np.array([0.96, 0.96, 0.96])

    for label in cases:
        if label not in positions:
            raise ValueError(f"Unknown case {label}; expected {tuple(positions)}.")
        chembl_id = str(ordered_ids[positions[label]])
        smiles_value = smiles_map.loc[chembl_id]
        mol = Chem.MolFromSmiles(smiles_value)
        if mol is None:
            raise ValueError(f"Invalid locked-test SMILES for {chembl_id}.")
        additional = rdFingerprintGenerator.AdditionalOutput()
        additional.AllocateBitInfoMap()
        generator.GetFingerprint(mol, additionalOutput=additional)
        bit_info = additional.GetBitInfoMap()
        row_values = shap_frame.loc[chembl_id]
        bit_contributions = {}
        for feature_name in morgan_features:
            bit = int(feature_name.rsplit("_", 1)[1])
            if bit in bit_info:
                bit_contributions[bit] = float(row_values[feature_name])
        atom_contributions = np.zeros(mol.GetNumAtoms(), dtype=float)
        bond_environment = set()
        for bit, contribution in bit_contributions.items():
            occurrences = bit_info[bit]
            for center, radius in occurrences:
                bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, int(radius), int(center)))
                atoms = {int(center)}
                for bond_id in bonds:
                    bond = mol.GetBondWithIdx(int(bond_id))
                    atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                bond_environment.update(int(bond_id) for bond_id in bonds)
                share = contribution / max(1, len(occurrences)) / max(1, len(atoms))
                for atom_id in atoms:
                    atom_contributions[atom_id] += share
                environment_rows.append({
                    "model": model_name, "case": label, "chembl_id": chembl_id,
                    "morgan_bit": bit, "bit_shap_pIC50": contribution,
                    "center_atom_index": int(center), "radius": int(radius),
                    "atom_indices": ",".join(map(str, sorted(atoms))),
                    "hashed_bit_occurrences_in_molecule": len(occurrences),
                })

        max_abs = float(np.max(np.abs(atom_contributions))) if len(atom_contributions) else 0.0
        atom_colors = {}
        for atom_id, contribution in enumerate(atom_contributions):
            intensity = 0.0 if max_abs == 0 else min(1.0, abs(float(contribution)) / max_abs)
            target_color = positive if contribution > 0 else negative
            atom_colors[atom_id] = tuple(float(x) for x in (neutral * (1 - intensity) + target_color * intensity))
        highlighted_atoms = [i for i, value in enumerate(atom_contributions) if abs(value) > 0]
        bond_colors = {bond_id: atom_colors[mol.GetBondWithIdx(bond_id).GetBeginAtomIdx()] for bond_id in bond_environment}
        predicted = float(predictions.loc[chembl_id, "predicted_pIC50"])
        image = Draw.MolToImage(
            mol, size=(900, 600),
            legend=f"{model_name} fixed-250 SHAP | {label} prediction={predicted:.3f} pIC50",
            highlightAtoms=highlighted_atoms, highlightAtomColors=atom_colors,
            highlightBonds=sorted(bond_environment), highlightBondColors=bond_colors,
        )
        safe_model = model_name.replace("Regressor", "")
        image_path = stage / "figures" / f"fixed250_morgan_SHAP_{safe_model}_{label}_{chembl_id}.png"
        image.save(image_path)
        images.append(image)
        case_rows.append({"model": model_name, "case": label, "chembl_id": chembl_id,
                          "predicted_pIC50": predicted, "n_morgan_features_in_fixed250": len(morgan_features),
                          "n_present_morgan_bits_explained": len(bit_contributions), "figure": str(image_path)})

    case_table = pd.DataFrame(case_rows)
    environment_table = pd.DataFrame(environment_rows)
    safe_model = model_name.replace("Regressor", "")
    case_table.to_csv(stage / "tables" / f"fixed250_morgan_SHAP_cases_{safe_model}.csv", index=False)
    environment_table.to_csv(stage / "tables" / f"fixed250_morgan_SHAP_environments_{safe_model}.csv", index=False)
    return case_table, environment_table, images
