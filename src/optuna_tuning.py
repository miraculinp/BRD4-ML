"""Stage 07: Optuna tuning with five-fold scaffold-grouped CV on training data."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

from src.utils import OUTPUTS, load_config, sha256, stage_dir, write_manifest


TUNED_MODELS = ("ExtraTreesRegressor", "CatBoostRegressor", "SVR")
MODELS = TUNED_MODELS
THREADS = 2
FEATURE_BUDGET = 250


def make_estimator(model_name, params, seed):
    """Create a model with trial parameters and fixed reproducibility settings."""
    params = dict(params)
    if model_name == "ExtraTreesRegressor":
        # Optuna's categorical distributions cannot contain None. A value of
        # zero is the explicit sentinel for an unlimited tree depth.
        if params.get("max_depth") == 0:
            params["max_depth"] = None
        return ExtraTreesRegressor(
            **params,
            random_state=int(seed),
            n_jobs=THREADS,
        )
    if model_name == "CatBoostRegressor":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("Install catboost to tune CatBoostRegressor.") from exc
        return CatBoostRegressor(
            **params,
            loss_function="RMSE",
            random_seed=int(seed),
            thread_count=THREADS,
            verbose=False,
            allow_writing_files=False,
        )
    if model_name == "SVR":
        return make_pipeline(StandardScaler(), SVR(**params))
    raise ValueError(f"Unsupported model: {model_name}")


def suggest_params(trial, model_name):
    if model_name == "ExtraTreesRegressor":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=100),
            "max_features": trial.suggest_float("max_features", 0.4, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_depth": trial.suggest_categorical(
                "max_depth", [0, 10, 20, 30, 40]
            ),
            "bootstrap": trial.suggest_categorical("bootstrap", [False, True]),
        }
    if model_name == "CatBoostRegressor":
        return {
            "iterations": trial.suggest_int("iterations", 300, 1000, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        }
    if model_name == "SVR":
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
        params = {
            "kernel": kernel,
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-3, 0.5, log=True),
        }
        if kernel == "rbf":
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        return params
    raise ValueError(f"Unsupported model: {model_name}")


def _load_training_data():
    feature_path = OUTPUTS / "05_feature_selection/tables/train_selected_features.parquet"
    target_path = OUTPUTS / "05_feature_selection/tables/train_target.parquet"
    selected_path = OUTPUTS / "05_feature_selection/tables/selected_features.csv"
    train_ids_path = OUTPUTS / "04_train_test_split/tables/train_ids.csv"
    for path in (feature_path, target_path, selected_path, train_ids_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input is missing: {path}. Run earlier stages first.")

    selected = pd.read_csv(selected_path)["feature"].astype(str).tolist()
    if len(selected) != FEATURE_BUDGET or len(set(selected)) != FEATURE_BUDGET:
        raise ValueError(
            f"Expected exactly {FEATURE_BUDGET} unique Stage 05 features; found {len(selected)}."
        )

    X_all = pd.read_parquet(feature_path)
    y_all = pd.read_parquet(target_path)["pIC50"]
    train_ids = pd.read_csv(train_ids_path, dtype={"chembl_id": str})
    if train_ids.chembl_id.duplicated().any():
        raise ValueError("The training ID file contains duplicate molecule IDs.")

    X_all.index = X_all.index.astype(str)
    y_all.index = y_all.index.astype(str)
    train_ids["chembl_id"] = train_ids.chembl_id.astype(str)
    missing_features = [name for name in selected if name not in X_all.columns]
    if missing_features:
        raise ValueError(f"Stage 05 matrix is missing selected features: {missing_features[:10]}")
    X = X_all.reindex(columns=selected).reindex(train_ids.chembl_id.to_list())
    y = y_all.reindex(train_ids.chembl_id.to_list())
    groups = train_ids["group"].to_numpy()

    target_ok = y.notna().to_numpy() & np.isfinite(y.to_numpy(dtype=float))
    X, y, groups = X.loc[target_ok].copy(), y.loc[target_ok], groups[target_ok]
    if X.empty or pd.isna(groups).any():
        raise ValueError("No usable training rows remain or some training groups are missing.")

    X = X.apply(pd.to_numeric, errors="coerce")
    values = X.to_numpy(dtype=np.float64)
    if X.isna().any().any() or not np.isfinite(values).all():
        raise ValueError(
            "The fixed Stage 05 training matrix contains missing or non-finite values; "
            "this pipeline does not impute."
        )
    if X.shape[1] != FEATURE_BUDGET:
        raise RuntimeError(f"Expected {FEATURE_BUDGET} fixed features; got {X.shape[1]}.")

    return X, y, groups, feature_path, target_path, selected_path, train_ids_path


def _prepare_folds(X, y, groups, cfg, out_dir, cv=None):
    n_splits = min(int(cfg.get("cv_splits", 5)), len(np.unique(groups)))
    if n_splits != 5:
        raise ValueError(
            f"Optuna tuning requires five distinct scaffold groups at minimum; got {n_splits}."
        )

    cv = cv or GroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=int(cfg.get("seed", 42)),
    )
    folds = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_fold_train = X.iloc[train_idx].loc[:, X.columns]
        X_fold_valid = X.iloc[valid_idx].loc[:, X.columns]
        y_fold_train = y.iloc[train_idx]
        y_fold_valid = y.iloc[valid_idx]
        folds.append(
            {
                "fold": fold,
                "X_train": X_fold_train,
                "X_valid": X_fold_valid,
                "y_train": y_fold_train,
                "y_valid": y_fold_valid,
                "valid_ids": X.index[valid_idx].astype(str).tolist(),
                "valid_groups": np.asarray(groups)[valid_idx].tolist(),
            }
        )

    # Store a concise, auditable mapping instead of repeating it for every row
    # and every feature in every fold.
    assignment_rows = [
        {
            "chembl_id": chembl_id,
            "group": str(group),
            "fold": fold_data["fold"],
        }
        for fold_data in folds
        for chembl_id, group in zip(fold_data["valid_ids"], fold_data["valid_groups"])
    ]
    assignment_frame = pd.DataFrame(assignment_rows)
    if assignment_frame.chembl_id.duplicated().any() or len(assignment_frame) != len(X):
        raise RuntimeError("Each training molecule must be assigned to exactly one validation fold.")
    assignment_frame.to_csv(
        out_dir / "cv_fold_assignments.csv", index=False
    )
    return folds


def _run_study(model_name, folds, cfg, storage, study_prefix, n_trials):
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Install Optuna with: python -m pip install optuna") from exc

    study_name = f"{study_prefix}_{model_name}"
    sampler = optuna.samplers.TPESampler(seed=int(cfg.get("seed", 42)))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        # Every completed trial is scored on all five folds so partial-fold
        # pruning cannot make the objective incomparable across trials.
        pruner=optuna.pruners.NopPruner(),
    )
    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, int(n_trials) - completed)

    def objective(trial):
        params = suggest_params(trial, model_name)
        fold_metrics = []
        oof_observed = []
        oof_predicted = []
        for fold_data in folds:
            model = make_estimator(
                model_name,
                params,
                int(cfg.get("seed", 42)) + fold_data["fold"],
            )
            with threadpool_limits(limits=THREADS):
                model.fit(fold_data["X_train"], fold_data["y_train"])
                prediction = np.asarray(model.predict(fold_data["X_valid"]), dtype=float)
            observed = fold_data["y_valid"].to_numpy(dtype=float)
            if not np.isfinite(prediction).all():
                raise ValueError(f"{model_name} returned non-finite predictions in CV.")
            mse = float(mean_squared_error(observed, prediction))
            fold_metrics.append(
                {
                    "fold": int(fold_data["fold"]),
                    "n_validation": int(len(observed)),
                    "rmse": float(np.sqrt(mse)),
                    "mse": mse,
                    "mae": float(mean_absolute_error(observed, prediction)),
                    "r2": float(r2_score(observed, prediction)),
                }
            )
            oof_observed.extend(observed.tolist())
            oof_predicted.extend(prediction.tolist())
        mean_fold_rmse = float(np.mean([row["rmse"] for row in fold_metrics]))
        pooled_mse = float(mean_squared_error(oof_observed, oof_predicted))
        trial.set_user_attr("fold_metrics", fold_metrics)
        trial.set_user_attr(
            "pooled_oof_metrics",
            {
                "rmse": float(np.sqrt(pooled_mse)),
                "mse": pooled_mse,
                "mae": float(mean_absolute_error(oof_observed, oof_predicted)),
                "r2": float(r2_score(oof_observed, oof_predicted)),
            },
        )
        return mean_fold_rmse

    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    try:
        best_trial = study.best_trial
    except ValueError as exc:
        raise RuntimeError(f"Optuna produced no completed trial for {model_name}.")
    return study, best_trial


def run(n_trials=None):
    cfg = load_config()
    n_trials = int(n_trials or cfg.get("optuna_trials", 30))
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")

    X, y, groups, feature_path, target_path, selected_path, train_ids_path = _load_training_data()
    d = stage_dir("07_optuna_tuning")
    fold_dir = d / "models/fold_feature_cache"
    fold_dir.mkdir(parents=True, exist_ok=True)

    run_material = {
        "features": sha256(feature_path),
        "selected_feature_list": sha256(selected_path),
        "target": sha256(target_path),
        "train_ids": sha256(train_ids_path),
        "tuner_script": sha256(Path(__file__)),
        "feature_budget": FEATURE_BUDGET,
        "feature_selection": "fixed ordered Stage 05 selected 250-feature list; no fold-specific cleaning or reranking",
        "folds": 5,
        "models": MODELS,
        "tuned_models": TUNED_MODELS,
        "seed": int(cfg.get("seed", 42)),
    }
    run_id = hashlib.sha256(
        json.dumps(run_material, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    # Fixed Stage 05 features are used unchanged in every fold and every trial.
    cache_path = fold_dir / f"prepared_folds_{run_id}.joblib"
    if cache_path.exists():
        import joblib
        folds = joblib.load(cache_path)
    else:
        folds = _prepare_folds(X, y, groups, cfg, fold_dir)
        import joblib
        joblib.dump(folds, cache_path)

    database_path = d / "models/optuna_studies.sqlite3"
    storage = f"sqlite:///{database_path.resolve().as_posix()}"
    studies = {}
    for model_name in TUNED_MODELS:
        print(f"Optuna: {model_name}, fixed 250 features, 5-fold GroupKFold, up to {n_trials} trials", flush=True)
        studies[model_name] = _run_study(
            model_name,
            folds,
            cfg,
            storage,
            f"brd4_{run_id}",
            n_trials,
        )

    best_records = []
    best_params = {}
    best_cv_folds = []
    for model_name, (study, trial) in studies.items():
        fold_frame = pd.DataFrame(trial.user_attrs.get("fold_metrics", []))
        if len(fold_frame) != 5:
            raise RuntimeError(
                f"Best {model_name} trial does not contain metrics for all five folds. "
                "Start a new tuning study with the updated script."
            )
        fold_frame.insert(0, "model", model_name)
        pooled = trial.user_attrs.get("pooled_oof_metrics", {})
        cv_summary = {
            "model": model_name,
            **{
                f"cv_mean_fold_{metric}": float(fold_frame[metric].mean())
                for metric in ("rmse", "mse", "mae", "r2")
            },
            **{
                f"cv_sd_fold_{metric}": float(fold_frame[metric].std(ddof=1))
                for metric in ("rmse", "mse", "mae", "r2")
            },
            "cv_pooled_oof_rmse": float(pooled["rmse"]),
            "cv_pooled_oof_mse": float(pooled["mse"]),
            "cv_pooled_oof_mae": float(pooled["mae"]),
            "cv_pooled_oof_r2": float(pooled["r2"]),
        }
        best_params[model_name] = {
            "params": trial.params,
            "parameter_source": "Optuna best parameters",
            "best_cv_rmse": float(trial.value),
            **cv_summary,
            "fold_metrics": trial.user_attrs["fold_metrics"],
            "trial_number": int(trial.number),
            "study_name": study.study_name,
        }
        best_records.append(
            {
                "model": model_name,
                "best_cv_rmse": float(trial.value),
                "parameter_source": "Optuna best parameters",
                "trial_number": int(trial.number),
                "params": json.dumps(trial.params, sort_keys=True),
                **cv_summary,
            }
        )
        best_cv_folds.append(fold_frame)
        study.trials_dataframe().to_csv(
            d / f"tables/optuna_trials_{model_name}.csv", index=False
        )

    summary = pd.DataFrame(best_records).sort_values("best_cv_rmse")
    summary.to_csv(d / "tables/optuna_model_comparison.csv", index=False)
    pd.concat(best_cv_folds, ignore_index=True).to_csv(
        d / "tables/best_params_cv_fold_metrics.csv", index=False
    )
    pd.read_csv(selected_path).to_csv(
        d / "tables/fixed_250_features_used_for_tuning.csv", index=False
    )
    (d / "tables/optuna_best_params.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "cv": "5-fold shuffled GroupKFold by training scaffold/connectivity group",
                "objective": "mean fold RMSE",
                "feature_budget": FEATURE_BUDGET,
                "feature_selection": "fixed ordered Stage 05 selected 250-feature list used unchanged for tuning and every fold",
                "fold_metrics": "RMSE, MSE, MAE, and R2 per fold; reported as mean and sample SD across the five folds",
                "best_trial_cv_metrics_reused": True,
                "extra_best_parameter_cv_refit": False,
                "cv_interpretation": "All three metrics are selected-trial tuning-CV estimates and may be optimistic after parameter selection. Use the locked test set for final external performance.",
                "no_imputation": True,
                "params_by_model": best_params,
                "recommended_model_by_tuning_cv": str(summary.iloc[0].model),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nOptuna tuning summary (training-only grouped CV; fixed Stage 05 250 features):")
    print(summary.to_string(index=False))
    print("Fold metrics above come from the best trial's existing CV evaluations; no second CV refit was run.")
    print("All three tuning-CV metrics may be optimistic after parameter selection.")
    print("Use the locked test set for final external performance.")

    write_manifest(
        "07_optuna_tuning",
        {
            **run_material,
            "models": list(MODELS),
            "best_model_by_cv": str(summary.iloc[0].model),
            "test_used": False,
        },
        [feature_path, target_path, selected_path, train_ids_path],
        [
            d / "tables/optuna_model_comparison.csv",
            d / "tables/optuna_best_params.json",
            d / "tables/best_params_cv_fold_metrics.csv",
            d / "tables/fixed_250_features_used_for_tuning.csv",
        ],
    )
    return summary


if __name__ == "__main__":
    run()
