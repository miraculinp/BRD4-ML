"""Stage 09: applicability domain and ExtraTrees Y-randomization checks.

Uses existing feature tables and Stage 08 predictions. Y-randomization is run
only for the training-CV-selected ExtraTrees model; it never fits SVR.
"""

import json
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.optuna_tuning import (
    _evaluate_best_params,
    _load_training_data,
    _prepare_folds,
)
from src.utils import OUTPUTS, load_config, stage_dir, write_manifest

MODEL = "ExtraTreesRegressor"


def _load_ad_data(selected_features):
    tables03 = OUTPUTS / "03_feature_generation/tables"
    tables04 = OUTPUTS / "04_train_test_split/tables"
    paths = [
        tables03 / "all_features.parquet",
        tables03 / "target.parquet",
        tables04 / "train_ids.csv",
        tables04 / "test_ids.csv",
    ]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Stage 03/04 inputs: " + ", ".join(missing))

    X_all = pd.read_parquet(paths[0])
    y_all = pd.read_parquet(paths[1])["pIC50"]
    X_all.index = X_all.index.astype(str)
    y_all.index = y_all.index.astype(str)
    train_ids = pd.read_csv(paths[2], dtype={"chembl_id": str})
    test_ids = pd.read_csv(paths[3], dtype={"chembl_id": str})
    train_ids["chembl_id"] = train_ids.chembl_id.astype(str)
    test_ids["chembl_id"] = test_ids.chembl_id.astype(str)

    X_train = X_all.reindex(train_ids.chembl_id).loc[:, selected_features]
    y_train = y_all.reindex(train_ids.chembl_id)
    X_test = X_all.reindex(test_ids.chembl_id).loc[:, selected_features]
    y_test = y_all.reindex(test_ids.chembl_id)
    for label, X, y in (("training", X_train, y_train), ("test", X_test, y_test)):
        X = X.apply(pd.to_numeric, errors="coerce")
        if X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError(f"{label.title()} AD features contain missing/infinite values.")
        if y.isna().any() or not np.isfinite(y.to_numpy(dtype=float)).all():
            raise ValueError(f"{label.title()} pIC50 contains missing/infinite values.")
        if label == "training":
            X_train = X
        else:
            X_test = X
    if set(train_ids.group) & set(test_ids.group):
        raise ValueError("Train/test scaffold overlap found during AD analysis.")
    return X_train, y_train, train_ids, X_test, y_test, test_ids, paths


def _run_ad(selected_features, out_dir):
    X_train, y_train, train_ids, X_test, y_test, test_ids, input_paths = _load_ad_data(
        selected_features
    )

    # Scale with training statistics only so unlike raw Euclidean distance,
    # large-unit descriptors do not dominate the distance calculation.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)
    k = min(5, len(X_train) - 1)
    if k < 1:
        raise ValueError("At least two training molecules are required for kNN AD.")

    train_neighbors = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    train_neighbors.fit(train_scaled)
    train_distances = train_neighbors.kneighbors(train_scaled, return_distance=True)[0][:, 1:]
    mean_train_distance = train_distances.mean(axis=1)

    test_neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    test_neighbors.fit(train_scaled)
    mean_test_distance = test_neighbors.kneighbors(test_scaled, return_distance=True)[0].mean(axis=1)
    threshold = float(mean_train_distance.mean() + 2 * mean_train_distance.std(ddof=0))
    inside = mean_test_distance <= threshold

    pred_path = OUTPUTS / "08_final_model_evaluation/tables/locked_test_predictions_all_models.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Run Stage 08 first; missing {pred_path}")
    predictions = pd.read_csv(pred_path)
    predictions = predictions.loc[predictions.model == MODEL].copy()
    predictions["chembl_id"] = predictions.chembl_id.astype(str)
    if predictions.chembl_id.duplicated().any():
        raise ValueError("Duplicate ExtraTrees test predictions found.")
    predictions = predictions.set_index("chembl_id").reindex(test_ids.chembl_id)
    if predictions.predicted_pIC50.isna().any():
        raise ValueError("ExtraTrees test predictions do not align with test IDs.")
    abs_error = np.abs(
        y_test.to_numpy(dtype=float) - predictions.predicted_pIC50.to_numpy(dtype=float)
    )

    ad = pd.DataFrame(
        {
            "chembl_id": test_ids.chembl_id.to_numpy(),
            "scaffold_group": test_ids.group.to_numpy(),
            "observed_pIC50": y_test.to_numpy(dtype=float),
            "predicted_pIC50": predictions.predicted_pIC50.to_numpy(dtype=float),
            "absolute_error": abs_error,
            "mean_5nn_distance": mean_test_distance,
            "inside_applicability_domain": inside,
            "knn_threshold": threshold,
        }
    )
    ad_path = out_dir / "tables/applicability_domain_test.csv"
    ad.to_csv(ad_path, index=False)

    coverage = pd.DataFrame(
        [
            {
                "method": "standardized descriptor-space 5-nearest-neighbor distance",
                "threshold_rule": "training mean + 2 population SD",
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_inside_ad": int(inside.sum()),
                "fraction_inside_ad": float(inside.mean()),
                "threshold": threshold,
                "inside_test_rmse": float(
                    np.sqrt(mean_squared_error(y_test.to_numpy()[inside], predictions.predicted_pIC50.to_numpy()[inside]))
                ) if inside.sum() else np.nan,
                "outside_test_rmse": float(
                    np.sqrt(mean_squared_error(y_test.to_numpy()[~inside], predictions.predicted_pIC50.to_numpy()[~inside]))
                ) if (~inside).sum() else np.nan,
            }
        ]
    )
    coverage.to_csv(out_dir / "tables/applicability_domain_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(mean_train_distance, bins=40, alpha=0.65, label="Training molecules")
    ax.hist(mean_test_distance, bins=40, alpha=0.65, label="Test molecules")
    ax.axvline(threshold, color="red", linestyle="--", label=f"AD threshold = {threshold:.3g}")
    ax.set(xlabel="Mean standardized distance to 5 nearest training molecules", ylabel="Molecule count",
           title="ExtraTrees applicability-domain coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures/applicability_domain_knn.png", dpi=300)
    plt.close(fig)
    return coverage, input_paths + [pred_path]


def _run_y_randomization(n_randomizations, cfg, best_params, out_dir):
    X, y, groups, feature_path, target_path, train_ids_path = _load_training_data()
    cv = GroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=int(cfg.get("seed", 42)) + 2026,
    )
    # The observed score uses the exact same folds and fold-local feature-selection
    # procedure as each randomized run.
    with tempfile.TemporaryDirectory(prefix="brd4_yreal_") as tmp:
        real_folds = _prepare_folds(X, y, groups, cfg, __import__("pathlib").Path(tmp), cv=cv)
        real_summary, _, _ = _evaluate_best_params(MODEL, best_params, real_folds, cfg)
    real_rmse = float(real_summary["cv_pooled_oof_rmse"])

    rng = np.random.default_rng(int(cfg.get("seed", 42)) + 9001)
    rows = []
    for iteration in range(1, n_randomizations + 1):
        y_shuffled = pd.Series(rng.permutation(y.to_numpy()), index=y.index, name=y.name)
        with tempfile.TemporaryDirectory(prefix=f"brd4_y{iteration:02d}_") as tmp:
            shuffled_folds = _prepare_folds(
                X, y_shuffled, groups, cfg, __import__("pathlib").Path(tmp), cv=cv
            )
            summary, _, _ = _evaluate_best_params(MODEL, best_params, shuffled_folds, cfg)
        rows.append(
            {
                "iteration": iteration,
                "model": MODEL,
                "pooled_oof_rmse": float(summary["cv_pooled_oof_rmse"]),
                "pooled_oof_r2": float(summary["cv_pooled_oof_r2"]),
                "mean_fold_rmse": float(summary["cv_mean_fold_rmse"]),
            }
        )
        print(
            f"Y-randomization {iteration}/{n_randomizations}: "
            f"pooled OOF RMSE={rows[-1]['pooled_oof_rmse']:.4f}",
            flush=True,
        )

    null = pd.DataFrame(rows)
    null.to_csv(out_dir / "tables/y_randomization_extratrees_iterations.csv", index=False)
    # Smaller RMSE is better; add-one correction gives a valid finite-sample p estimate.
    p_value = float((1 + (null.pooled_oof_rmse <= real_rmse).sum()) / (n_randomizations + 1))
    result = pd.DataFrame(
        [
            {
                "model": MODEL,
                "n_randomizations": n_randomizations,
                "real_pooled_oof_rmse": real_rmse,
                "randomized_mean_rmse": float(null.pooled_oof_rmse.mean()),
                "randomized_sd_rmse": float(null.pooled_oof_rmse.std(ddof=1)) if n_randomizations > 1 else np.nan,
                "randomized_min_rmse": float(null.pooled_oof_rmse.min()),
                "empirical_p_value_lower_rmse": p_value,
                "best_params_fixed_from_optuna": json.dumps(best_params, sort_keys=True),
                "randomized_target_feature_selection": "re-fit inside each CV fold",
            }
        ]
    )
    result.to_csv(out_dir / "tables/y_randomization_extratrees_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null.pooled_oof_rmse, bins=min(12, n_randomizations), color="#4C78A8", alpha=0.8)
    ax.axvline(real_rmse, color="#D62728", linestyle="--", linewidth=2,
               label=f"Real-label CV RMSE = {real_rmse:.3f}")
    ax.set(xlabel="Pooled out-of-fold RMSE", ylabel="Randomization count",
           title=f"Y-randomization — {MODEL} ({n_randomizations} shuffles)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures/y_randomization_extratrees.png", dpi=300)
    plt.close(fig)
    return result, [feature_path, target_path, train_ids_path,
                   OUTPUTS / "07_optuna_tuning/tables/optuna_best_params.json"]


def run(n_randomizations=20):
    if int(n_randomizations) < 1:
        raise ValueError("n_randomizations must be at least 1.")
    cfg = load_config()
    tuning_path = OUTPUTS / "07_optuna_tuning/tables/optuna_best_params.json"
    selected_path = OUTPUTS / "05_feature_selection/tables/selected_features.csv"
    for path in (tuning_path, selected_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}. Run Stages 05 and 07 first.")

    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    try:
        best_params = dict(tuning["params_by_model"][MODEL]["params"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Saved Optuna parameters for {MODEL} are missing.") from exc

    out_dir = stage_dir("09_validation")
    selected = pd.read_csv(selected_path)["feature"].astype(str).tolist()
    if not selected:
        raise ValueError("Stage 05 selected feature list is empty.")

    ad_summary, ad_inputs = _run_ad(selected, out_dir)
    yrand_summary, yrand_inputs = _run_y_randomization(
        int(n_randomizations), cfg, best_params, out_dir
    )
    summary = ad_summary.merge(yrand_summary, how="cross")
    summary.to_csv(out_dir / "tables/validation_summary.csv", index=False)
    print("Applicability-domain summary:")
    print(ad_summary.to_string(index=False))
    print("\nExtraTrees Y-randomization summary:")
    print(yrand_summary.to_string(index=False))
    print("SVR was not fitted in Stage 09.")

    write_manifest(
        "09_validation",
        {
            "applicability_domain": "standardized 5-nearest-neighbor distance; threshold training mean + 2 SD",
            "y_randomization_model": MODEL,
            "n_randomizations": int(n_randomizations),
            "cv": "five-fold shuffled GroupKFold by scaffold, matching Stage 08 split seed",
            "fold_feature_selection": "cleaning and ExtraTrees feature ranking refit within every fold and permutation",
            "optuna_best_parameters_refit": False,
            "svr_fitted": False,
            "test_used_for_tuning": False,
        },
        ad_inputs + yrand_inputs,
        [
            out_dir / "tables/applicability_domain_summary.csv",
            out_dir / "tables/applicability_domain_test.csv",
            out_dir / "tables/y_randomization_extratrees_summary.csv",
            out_dir / "tables/y_randomization_extratrees_iterations.csv",
            out_dir / "tables/validation_summary.csv",
        ],
    )
    return summary


if __name__ == "__main__":
    run()
