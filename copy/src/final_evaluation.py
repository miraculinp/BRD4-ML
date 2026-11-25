"""Stage 08: fresh training-only scaffold CV, then refit and score locked test."""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from src.optuna_tuning import (
    _evaluate_best_params,
    _load_training_data,
    _prepare_folds,
    make_estimator,
)
from src.utils import OUTPUTS, load_config, save_figure, setup_plotting, stage_dir, write_manifest

MODELS = ("ExtraTreesRegressor", "CatBoostRegressor", "SVR")


def run():
    cfg = load_config()
    tables05 = OUTPUTS / "05_feature_selection/tables"
    tables03 = OUTPUTS / "03_feature_generation/tables"
    tables04 = OUTPUTS / "04_train_test_split/tables"
    tuning_json = OUTPUTS / "07_optuna_tuning/tables/optuna_best_params.json"

    paths = [
        tables05 / "train_selected_features.parquet",
        tables05 / "train_target.parquet",
        tables05 / "train_groups.csv",
        tables05 / "selected_features.csv",
        tables03 / "all_features.parquet",
        tables03 / "target.parquet",
        tables04 / "test_ids.csv",
        tuning_json,
    ]
    absent = [str(path) for path in paths if not path.exists()]
    if absent:
        raise FileNotFoundError(
            "Run Stages 03, 04, 05, and 07 first. Missing: " + ", ".join(absent)
        )

    X_train = pd.read_parquet(paths[0])
    y_train = pd.read_parquet(paths[1])["pIC50"]
    train_groups = pd.read_csv(paths[2])
    selected_features = pd.read_csv(paths[3])["feature"].astype(str).tolist()
    all_features = pd.read_parquet(paths[4])
    all_target = pd.read_parquet(paths[5])["pIC50"]
    test_ids = pd.read_csv(paths[6], dtype={"chembl_id": str})
    tuning = json.loads(tuning_json.read_text(encoding="utf-8"))

    if len(selected_features) != 250:
        raise ValueError(
            f"Expected the fixed Stage 05 set of 250 features; found {len(selected_features)}."
        )
    params_by_model = tuning.get("params_by_model", {})
    for model_name in MODELS[:2]:
        if model_name not in params_by_model or "params" not in params_by_model[model_name]:
            raise ValueError(f"Optuna parameters for {model_name} are missing; rerun Stage 07.")
    if len(X_train) != len(y_train) or len(train_groups) != len(X_train):
        raise ValueError("Training features, target, and scaffold groups are misaligned.")
    if test_ids["chembl_id"].duplicated().any():
        raise ValueError("The locked test ID file contains duplicate molecule IDs.")
    if train_groups["group"].isna().any() or test_ids["group"].isna().any():
        raise ValueError("Missing scaffold group labels were found.")

    overlap = set(train_groups["group"]) & set(test_ids["group"])
    if overlap:
        raise ValueError(f"Scaffold leakage detected: {len(overlap)} train/test groups overlap.")

    X_train = X_train.reindex(columns=selected_features).apply(pd.to_numeric, errors="coerce")
    if X_train.isna().any().any() or not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise ValueError("Training features contain missing or infinite values; no imputation is used.")
    if y_train.isna().any() or not np.isfinite(y_train.to_numpy(dtype=float)).all():
        raise ValueError("Training pIC50 values contain missing or infinite values.")

    all_features.index = all_features.index.astype(str)
    all_target.index = all_target.index.astype(str)
    test_ids["chembl_id"] = test_ids["chembl_id"].astype(str)
    if not set(selected_features).issubset(all_features.columns):
        absent_features = sorted(set(selected_features) - set(all_features.columns))
        raise ValueError(f"Selected features missing from feature table: {absent_features[:10]}")
    missing_ids = sorted(set(test_ids.chembl_id) - set(all_features.index))
    if missing_ids:
        raise ValueError(f"{len(missing_ids)} test IDs are absent from the feature table.")

    X_test = all_features.reindex(test_ids.chembl_id).loc[:, selected_features]
    X_test = X_test.apply(pd.to_numeric, errors="coerce")
    y_test = all_target.reindex(test_ids.chembl_id)
    invalid_test = ~np.isfinite(X_test.to_numpy(dtype=float))
    if invalid_test.any():
        bad_features = X_test.columns[invalid_test.any(axis=0)].tolist()
        bad_molecules = X_test.index[invalid_test.any(axis=1)].tolist()
        raise ValueError(
            "Selected test features contain missing or infinite values and this workflow "
            "does not impute. Affected molecules: "
            f"{len(bad_molecules)}; affected features: {len(bad_features)}. "
            f"Feature examples: {bad_features[:10]}; molecule examples: {bad_molecules[:10]}."
        )
    if y_test.isna().any() or not np.isfinite(y_test.to_numpy(dtype=float)).all():
        raise ValueError("Locked test target contains missing or infinite pIC50 values.")

    d = stage_dir("08_final_model_evaluation")
    cv_dir = d / "tables/fresh_cv_feature_audits"
    cv_dir.mkdir(parents=True, exist_ok=True)

    # A new, shuffled scaffold-grouped five-fold split is used after tuning.
    # Filtering and supervised feature ranking are refit within every fold.
    X_cv, y_cv, groups_cv, feature_path, target_path, train_ids_path = _load_training_data()
    fresh_cv = GroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=int(cfg.get("seed", 42)) + 2026,
    )
    cv_folds = _prepare_folds(X_cv, y_cv, groups_cv, cfg, cv_dir, cv=fresh_cv)

    params_for_final = {
        "ExtraTreesRegressor": dict(params_by_model["ExtraTreesRegressor"]["params"]),
        "CatBoostRegressor": dict(params_by_model["CatBoostRegressor"]["params"]),
        # No Optuna search for SVR: use sklearn defaults (RBF, C=1.0,
        # epsilon=0.1, gamma='scale') inside the StandardScaler pipeline.
        "SVR": {},
    }
    parameter_source = {
        "ExtraTreesRegressor": "Optuna best parameters",
        "CatBoostRegressor": "Optuna best parameters",
        "SVR": "scikit-learn defaults",
    }

    comparison = []
    prediction_rows = []
    fitted_models = {}
    cv_fold_tables = []
    cv_oof_tables = []
    for model_name in MODELS:
        params = params_for_final[model_name]
        cv_summary, fold_metrics, oof = _evaluate_best_params(
            model_name, params, cv_folds, cfg
        )
        cv_summary["parameter_source"] = parameter_source[model_name]
        cv_fold_tables.append(fold_metrics)
        cv_oof_tables.append(oof)
        cv_fold_metrics = fold_metrics.rename(
            columns={"r2": "cv_fold_r2", "rmse": "cv_fold_rmse", "mae": "cv_fold_mae"}
        )
        cv_fold_metrics.to_csv(d / f"tables/fresh_cv_fold_metrics_{model_name}.csv", index=False)
        oof.to_csv(d / f"tables/fresh_cv_oof_predictions_{model_name}.csv", index=False)

        seed = int(cfg.get("seed", 42))
        model = make_estimator(model_name, params, seed)
        with threadpool_limits(limits=2):
            model.fit(X_train, y_train)
            predicted = np.asarray(model.predict(X_test), dtype=float)
        if not np.isfinite(predicted).all():
            raise ValueError(f"{model_name} returned non-finite test predictions.")

        observed = y_test.to_numpy(dtype=float)
        comparison.append(
            {
                "model": model_name,
                "parameter_source": parameter_source[model_name],
                "parameters": json.dumps(params, sort_keys=True),
                "cv_mean_fold_rmse": cv_summary["cv_mean_fold_rmse"],
                "cv_sd_fold_rmse": cv_summary["cv_sd_fold_rmse"],
                "cv_pooled_oof_rmse": cv_summary["cv_pooled_oof_rmse"],
                "cv_pooled_oof_mae": cv_summary["cv_pooled_oof_mae"],
                "cv_pooled_oof_r2": cv_summary["cv_pooled_oof_r2"],
                "test_r2": float(r2_score(observed, predicted)),
                "test_rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
                "test_mae": float(mean_absolute_error(observed, predicted)),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "n_features_final_fit": int(len(selected_features)),
                "n_cv_folds": 5,
            }
        )
        frame = pd.DataFrame(
            {
                "chembl_id": test_ids.chembl_id.to_numpy(),
                "group": test_ids.group.to_numpy(),
                "model": model_name,
                "observed_pIC50": observed,
                "predicted_pIC50": predicted,
                "residual": observed - predicted,
            }
        )
        prediction_rows.append(frame)
        fitted_models[model_name] = {
            "model": model,
            "feature_names": selected_features,
            "parameters": params,
            "parameter_source": parameter_source[model_name],
            "seed": seed,
        }
        joblib.dump(fitted_models[model_name], d / f"models/final_{model_name}.joblib")

    results = pd.DataFrame(comparison).sort_values("cv_pooled_oof_rmse").reset_index(drop=True)
    primary_model = str(results.iloc[0].model)
    primary_predictions = next(x for x in prediction_rows if x.model.iloc[0] == primary_model)
    results.to_csv(d / "tables/final_model_comparison.csv", index=False)
    pd.concat(cv_fold_tables, ignore_index=True).to_csv(
        d / "tables/fresh_cv_fold_metrics_all_models.csv", index=False
    )
    pd.concat(cv_oof_tables, ignore_index=True).to_csv(
        d / "tables/fresh_cv_oof_predictions_all_models.csv", index=False
    )
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        d / "tables/locked_test_predictions_all_models.csv", index=False
    )
    joblib.dump(
        {
            "primary_model_by_fresh_training_cv": primary_model,
            "models": fitted_models,
            "feature_names": selected_features,
            "cv_feature_selection": "fold-local cleaning and ExtraTrees top-250 ranking",
        },
        d / "models/final_model_bundle.joblib",
    )

    setup_plotting()
    observed = primary_predictions.observed_pIC50.to_numpy()
    predicted = primary_predictions.predicted_pIC50.to_numpy()
    low = float(min(observed.min(), predicted.min()))
    high = float(max(observed.max(), predicted.max()))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(observed, predicted, alpha=0.45, s=22, color="#0072B2", edgecolors="none")
    ax.plot([low, high], [low, high], "--", color="#D55E00", linewidth=2)
    ax.set(
        xlabel="Observed pIC50",
        ylabel="Predicted pIC50",
        title=f"Locked scaffold test: {primary_model}",
        xlim=(low, high),
        ylim=(low, high),
    )
    save_figure(fig, d / "figures/locked_test_parity_primary_model")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(predicted, primary_predictions.residual, alpha=0.45, s=22, color="#009E73", edgecolors="none")
    ax.axhline(0, linestyle="--", color="#D55E00", linewidth=2)
    ax.set(
        xlabel="Predicted pIC50",
        ylabel="Residual (observed − predicted)",
        title=f"Locked scaffold test residuals: {primary_model}",
    )
    save_figure(fig, d / "figures/locked_test_residuals_primary_model")
    plt.close(fig)

    print("Fresh five-fold scaffold CV on training data, then final full-training fit and locked-test evaluation:")
    print(results.to_string(index=False))
    print("CV feature cleaning and ranking were fitted within each fold; no imputation was used.")
    print("The final test-set models use the fixed 250-feature Stage 05 list.")
    print("Do not use locked-test results to retune parameters or change the selected model.")

    write_manifest(
        "08_final_model_evaluation",
        {
            "models_fitted": list(MODELS),
            "parameters": parameter_source,
            "primary_model_by_fresh_training_cv": primary_model,
            "feature_budget_final_fit": len(selected_features),
            "cv_feature_budget": 250,
            "cv_folds": 5,
            "cv_split": "shuffled GroupKFold by scaffold, separate seed from Stage 07",
            "cv_feature_selection": "fold-local cleaning and ExtraTrees top-250 ranking",
            "test_used_for_tuning": False,
            "test_metrics_created": True,
            "no_imputation": True,
        },
        paths + [feature_path, target_path, train_ids_path],
        [
            d / "tables/final_model_comparison.csv",
            d / "tables/fresh_cv_fold_metrics_all_models.csv",
            d / "tables/fresh_cv_oof_predictions_all_models.csv",
            d / "tables/locked_test_predictions_all_models.csv",
            d / "models/final_model_bundle.joblib",
        ],
    )
    return results


if __name__ == "__main__":
    run()
