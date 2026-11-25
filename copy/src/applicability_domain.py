"""Stage 09b: compare three applicability-domain methods on the locked test set.

This stage reuses Stage 08 predictions. It does not fit or tune prediction
models, and it does not run Y-randomization.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.utils import (
    OUTPUTS,
    load_config,
    save_figure,
    setup_plotting,
    stage_dir,
    write_manifest,
)


STAGE = "09b_applicability_domain"
MODELS = ("ExtraTreesRegressor", "CatBoostRegressor", "SVR")
PREDICTION_FILE = OUTPUTS / "08_final_model_evaluation/tables/locked_test_predictions_all_models.csv"


def _score_subset(y_true, y_pred):
    """Return metrics for a subset, using NaN R2 if fewer than two rows exist."""
    n = len(y_true)
    if n == 0:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan}
    return {
        "r2": float(r2_score(y_true, y_pred)) if n >= 2 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def run():
    cfg = load_config()
    tables03 = OUTPUTS / "03_feature_generation/tables"
    tables04 = OUTPUTS / "04_train_test_split/tables"
    tables05 = OUTPUTS / "05_feature_selection/tables"
    feature_file = tables03 / "all_features.parquet"
    target_file = tables03 / "target.parquet"
    train_ids_file = tables04 / "train_ids.csv"
    test_ids_file = tables04 / "test_ids.csv"
    selected_file = tables05 / "selected_features.csv"
    oof_file = OUTPUTS / "08_final_model_evaluation/tables/fresh_cv_oof_predictions_ExtraTreesRegressor.csv"

    required_paths = [
        feature_file,
        target_file,
        train_ids_file,
        test_ids_file,
        selected_file,
        PREDICTION_FILE,
        oof_file,
    ]
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Run Stages 03, 04, 05, and 08 first. Missing: " + ", ".join(missing_paths)
        )

    selected_features = pd.read_csv(selected_file)["feature"].astype(str).tolist()
    if not selected_features:
        raise ValueError("Stage 05 selected feature list is empty.")
    if len(selected_features) != 250:
        raise ValueError(
            f"Expected the fixed Stage 05 set of 250 features; found {len(selected_features)}."
        )

    X_all = pd.read_parquet(feature_file)
    y_all = pd.read_parquet(target_file)["pIC50"]
    X_all.index = X_all.index.astype(str)
    y_all.index = y_all.index.astype(str)
    train_ids = pd.read_csv(train_ids_file, dtype={"chembl_id": str})
    test_ids = pd.read_csv(test_ids_file, dtype={"chembl_id": str})
    for ids in (train_ids, test_ids):
        ids["chembl_id"] = ids["chembl_id"].astype(str)
    if train_ids.chembl_id.duplicated().any() or test_ids.chembl_id.duplicated().any():
        raise ValueError("Train or test ID table contains duplicate chembl_id values.")
    if set(train_ids.group) & set(test_ids.group):
        raise ValueError("Train/test scaffold overlap found; AD analysis stopped.")

    missing_features = sorted(set(selected_features) - set(X_all.columns))
    if missing_features:
        raise ValueError(f"Selected features absent from feature table: {missing_features[:10]}")
    missing_train = sorted(set(train_ids.chembl_id) - set(X_all.index))
    missing_test = sorted(set(test_ids.chembl_id) - set(X_all.index))
    if missing_train or missing_test:
        raise ValueError(
            f"Feature table lacks {len(missing_train)} training and {len(missing_test)} test IDs."
        )

    X_train = X_all.reindex(train_ids.chembl_id).loc[:, selected_features]
    X_test = X_all.reindex(test_ids.chembl_id).loc[:, selected_features]
    y_test = y_all.reindex(test_ids.chembl_id)
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise ValueError("Training features contain missing or infinite values; no imputation is used.")
    if not np.isfinite(X_test.to_numpy(dtype=float)).all():
        raise ValueError("Test features contain missing or infinite values; no imputation is used.")
    if y_test.isna().any() or not np.isfinite(y_test.to_numpy(dtype=float)).all():
        raise ValueError("Test pIC50 contains missing or infinite values.")

    predictions = pd.read_csv(PREDICTION_FILE, dtype={"chembl_id": str})
    required_prediction_columns = {
        "chembl_id", "model", "observed_pIC50", "predicted_pIC50", "residual"
    }
    missing_prediction_columns = required_prediction_columns - set(predictions.columns)
    if missing_prediction_columns:
        raise ValueError(
            "Stage 08 prediction file is missing columns: "
            + ", ".join(sorted(missing_prediction_columns))
        )
    if set(MODELS) - set(predictions.model.astype(str).unique()):
        absent = sorted(set(MODELS) - set(predictions.model.astype(str).unique()))
        raise ValueError("Stage 08 predictions are missing models: " + ", ".join(absent))
    for model_name in MODELS:
        part = predictions.loc[predictions.model.astype(str) == model_name]
        if part.chembl_id.duplicated().any():
            raise ValueError(f"Duplicate test prediction IDs for {model_name}.")
        if set(part.chembl_id.astype(str)) != set(test_ids.chembl_id):
            raise ValueError(f"Test prediction IDs for {model_name} do not match the locked test set.")

    # 1) kNN distance: scale using training values only; threshold is the
    # training mean 5-NN distance plus two population standard deviations.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)
    k = min(5, len(X_train) - 1)
    if k < 1:
        raise ValueError("At least two training molecules are needed for kNN AD.")
    neighbor_model = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    neighbor_model.fit(train_scaled)
    train_knn_distances = neighbor_model.kneighbors(return_distance=True)[0].mean(axis=1)
    test_knn_distances = neighbor_model.kneighbors(test_scaled, return_distance=True)[0].mean(axis=1)
    knn_threshold = float(train_knn_distances.mean() + 2 * train_knn_distances.std(ddof=0))
    inside_knn = test_knn_distances <= knn_threshold

    # 2) Bounding box: a test molecule is inside only when every selected
    # feature falls between that feature's training minimum and maximum.
    train_min = X_train.min(axis=0).to_numpy(dtype=float)
    train_max = X_train.max(axis=0).to_numpy(dtype=float)
    test_values = X_test.to_numpy(dtype=float)
    outside_box_by_feature = (test_values < train_min) | (test_values > train_max)
    out_of_bounds_count = outside_box_by_feature.sum(axis=1)
    inside_bbox = out_of_bounds_count == 0

    # 3) Leverage: standardized training descriptors plus an intercept. The
    # warning threshold follows the requested h*=3p/n rule, p=selected features.
    design_train = np.column_stack([np.ones(len(X_train)), train_scaled])
    design_test = np.column_stack([np.ones(len(X_test)), test_scaled])
    gram_inverse = np.linalg.pinv(design_train.T @ design_train, hermitian=True)
    train_leverage = np.einsum("ij,jk,ik->i", design_train, gram_inverse, design_train)
    test_leverage = np.einsum("ij,jk,ik->i", design_test, gram_inverse, design_test)
    leverage_threshold = float(3 * len(selected_features) / len(X_train))
    inside_leverage = test_leverage <= leverage_threshold

    method_flags = {
        "knn": inside_knn,
        "bounding_box": inside_bbox,
        "leverage": inside_leverage,
    }
    out_dir = stage_dir(STAGE)
    test_ad = pd.DataFrame(
        {
            "chembl_id": test_ids.chembl_id.to_numpy(),
            "scaffold_group": test_ids.group.to_numpy(),
            "observed_pIC50": y_test.to_numpy(dtype=float),
            "mean_5nn_distance": test_knn_distances,
            "knn_threshold": knn_threshold,
            "inside_knn_ad": inside_knn,
            "out_of_bounds_feature_count": out_of_bounds_count,
            "fraction_features_out_of_bounds": out_of_bounds_count / len(selected_features),
            "inside_bounding_box_ad": inside_bbox,
            "leverage": test_leverage,
            "leverage_threshold": leverage_threshold,
            "inside_leverage_ad": inside_leverage,
        }
    )
    for model_name in MODELS:
        part = predictions.loc[predictions.model.astype(str) == model_name].set_index("chembl_id")
        part = part.reindex(test_ids.chembl_id)
        test_ad[f"predicted_{model_name}"] = part.predicted_pIC50.to_numpy(dtype=float)
        test_ad[f"residual_{model_name}"] = (
            y_test.to_numpy(dtype=float) - part.predicted_pIC50.to_numpy(dtype=float)
        )

    test_ad.to_csv(out_dir / "tables/test_ad_flags_all_methods.csv", index=False)

    rows = []
    thresholds = {
        "knn": ("standardized 5-nearest-neighbor distance", "training mean + 2 population SD", knn_threshold),
        "bounding_box": ("training-feature bounding box", "all 250 features within training min/max", np.nan),
        "leverage": ("leverage (hat value)", "h*=3p/n with p selected features", leverage_threshold),
    }
    y_true_all = y_test.to_numpy(dtype=float)
    for method, inside_mask in method_flags.items():
        method_name, threshold_rule, threshold_value = thresholds[method]
        for model_name in MODELS:
            y_pred = test_ad[f"predicted_{model_name}"].to_numpy(dtype=float)
            for label, mask in (("inside", inside_mask), ("outside", ~inside_mask)):
                metrics = _score_subset(y_true_all[mask], y_pred[mask])
                rows.append(
                    {
                        "method": method_name,
                        "method_key": method,
                        "threshold_rule": threshold_rule,
                        "threshold": threshold_value,
                        "model": model_name,
                        "domain_status": label,
                        "n_molecules": int(mask.sum()),
                        "fraction_of_test": float(mask.mean()),
                        "r2": metrics["r2"],
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                    }
                )
    performance = pd.DataFrame(rows)
    performance.to_csv(out_dir / "tables/ad_performance_inside_outside_all_methods.csv", index=False)

    # Compact manuscript-ready table using the kNN-defined domain, matching
    # the earlier Table 4 layout while reflecting the current three models.
    knn_table_rows = []
    for model_name in MODELS:
        y_pred = test_ad[f"predicted_{model_name}"].to_numpy(dtype=float)
        inside_metrics = _score_subset(y_true_all[inside_knn], y_pred[inside_knn])
        outside_metrics = _score_subset(y_true_all[~inside_knn], y_pred[~inside_knn])
        all_metrics = _score_subset(y_true_all, y_pred)
        knn_table_rows.append(
            {
                "Model": model_name,
                "R2_All": all_metrics["r2"],
                "R2_Inside_AD": inside_metrics["r2"],
                "R2_Outside_AD": outside_metrics["r2"],
                "N_Inside": int(inside_knn.sum()),
                "N_Outside": int((~inside_knn).sum()),
                "RMSE_All": all_metrics["rmse"],
                "RMSE_Inside_AD": inside_metrics["rmse"],
                "RMSE_Outside_AD": outside_metrics["rmse"],
                "MAE_All": all_metrics["mae"],
                "MAE_Inside_AD": inside_metrics["mae"],
                "MAE_Outside_AD": outside_metrics["mae"],
            }
        )
    knn_performance = pd.DataFrame(knn_table_rows)
    knn_performance.to_csv(out_dir / "tables/knn_domain_model_performance.csv", index=False)

    coverage = (
        pd.DataFrame(
            [
                {
                    "method": thresholds[method][0],
                    "method_key": method,
                    "threshold_rule": thresholds[method][1],
                    "threshold": thresholds[method][2],
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "n_inside_ad": int(mask.sum()),
                    "n_outside_ad": int((~mask).sum()),
                    "fraction_inside_ad": float(mask.mean()),
                }
                for method, mask in method_flags.items()
            ]
        )
    )
    coverage.to_csv(out_dir / "tables/ad_method_coverage.csv", index=False)

    joblib.dump(
        {
            "feature_names": selected_features,
            "n_train": len(X_train),
            "scaler": scaler,
            "knn_model": neighbor_model,
            "knn_k": k,
            "knn_train_mean_distance": float(train_knn_distances.mean()),
            "knn_train_sd_distance": float(train_knn_distances.std(ddof=0)),
            "knn_threshold": knn_threshold,
            "bounding_box_min": pd.Series(train_min, index=selected_features),
            "bounding_box_max": pd.Series(train_max, index=selected_features),
            "leverage_gram_pseudoinverse": gram_inverse,
            "leverage_threshold": leverage_threshold,
            "leverage_p": len(selected_features),
        },
        out_dir / "models/ad_statistics_all_methods.joblib",
    )

    setup_plotting()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(train_knn_distances, bins=40, alpha=0.65, label="Training", color="#0072B2")
    ax.hist(test_knn_distances, bins=40, alpha=0.55, label="Locked test", color="#E69F00")
    ax.axvline(knn_threshold, color="#D55E00", linestyle="--", label=f"Threshold = {knn_threshold:.3f}")
    ax.set(xlabel=f"Mean distance to {k} nearest training molecules", ylabel="Molecule count",
           title="Applicability domain: standardized kNN distance")
    ax.legend()
    save_figure(fig, out_dir / "figures/ad_knn_distance")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(out_of_bounds_count, bins=np.arange(-0.5, len(selected_features) + 1.5, 1),
            color="#009E73", alpha=0.85)
    ax.set(xlabel="Number of features outside training min/max", ylabel="Test molecule count",
           title="Applicability domain: bounding-box violations")
    save_figure(fig, out_dir / "figures/ad_bounding_box_violations")
    plt.close(fig)

    et_residual = test_ad["residual_ExtraTreesRegressor"].to_numpy(dtype=float)
    oof = pd.read_csv(oof_file)
    oof_scale = float(np.sqrt(mean_squared_error(oof.observed_pIC50, oof.predicted_pIC50)))
    if not np.isfinite(oof_scale) or oof_scale <= 0:
        raise ValueError("Could not compute a positive residual scale for the leverage diagnostic.")
    scaled_test_residual = et_residual / oof_scale
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(test_leverage[inside_leverage], scaled_test_residual[inside_leverage],
               alpha=0.55, s=22, color="#0072B2", label="Within leverage threshold")
    ax.scatter(test_leverage[~inside_leverage], scaled_test_residual[~inside_leverage],
               alpha=0.7, s=28, color="#D55E00", label="Above leverage threshold")
    ax.axvline(leverage_threshold, color="#CC79A7", linestyle="--", label=f"h* = {leverage_threshold:.3f}")
    ax.axhline(3, color="#777777", linestyle=":")
    ax.axhline(-3, color="#777777", linestyle=":")
    ax.set(xlabel="Test-set leverage", ylabel="ExtraTrees residual / training OOF RMSE",
           title="Applicability domain: leverage diagnostic")
    ax.legend()
    save_figure(fig, out_dir / "figures/ad_leverage_diagnostic")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#0072B2", "#E69F00", "#009E73"]
    ax.bar(["5-NN", "Bounding box", "Leverage"],
           [method_flags[key].mean() * 100 for key in ("knn", "bounding_box", "leverage")],
           color=colors)
    ax.set(ylabel="Test compounds inside domain (%)", ylim=(0, 105),
           title="Applicability-domain coverage by method")
    for i, key in enumerate(("knn", "bounding_box", "leverage")):
        ax.text(i, method_flags[key].mean() * 100 + 1.5,
                f"{method_flags[key].sum():,}/{len(X_test):,}", ha="center", fontsize=11)
    save_figure(fig, out_dir / "figures/ad_method_coverage")
    plt.close(fig)

    print("Applicability-domain coverage (no Y-randomization; no model refitting):")
    print(coverage.to_string(index=False))
    print("Inside/outside test metrics for all Stage 08 models:")
    print(performance.to_string(index=False))
    print("kNN-defined AD performance (manuscript Table 4 format):")
    print(knn_performance.to_string(index=False))

    input_paths = required_paths
    output_paths = [
        out_dir / "tables/test_ad_flags_all_methods.csv",
        out_dir / "tables/ad_performance_inside_outside_all_methods.csv",
        out_dir / "tables/knn_domain_model_performance.csv",
        out_dir / "tables/ad_method_coverage.csv",
        out_dir / "models/ad_statistics_all_methods.joblib",
        *[
            out_dir / f"figures/{stem}.{extension}"
            for stem in (
                "ad_knn_distance",
                "ad_bounding_box_violations",
                "ad_leverage_diagnostic",
                "ad_method_coverage",
            )
            for extension in ("png", "pdf")
        ],
    ]
    write_manifest(
        STAGE,
        {
            "methods": ["standardized 5-NN distance", "training-feature bounding box", "leverage h*=3p/n"],
            "features": len(selected_features),
            "knn_k": k,
            "knn_threshold_rule": "training mean + 2 population SD",
            "leverage_threshold_rule": "3p/n, p is number of selected features",
            "models_evaluated_from_saved_predictions": list(MODELS),
            "models_refit": False,
            "y_randomization_run": False,
            "test_used_only_for_domain_coverage_and_performance": True,
        },
        input_paths,
        output_paths,
    )
    return coverage, performance, knn_performance, test_ad


if __name__ == "__main__":
    run()
