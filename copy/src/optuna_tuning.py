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

from src.feature_selection import fit_fold_cleaner, make_ranker
from src.utils import OUTPUTS, load_config, sha256, stage_dir, write_manifest


MODELS = ("ExtraTreesRegressor", "CatBoostRegressor")
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
    feature_path = OUTPUTS / "03_feature_generation/tables/all_features.parquet"
    target_path = OUTPUTS / "03_feature_generation/tables/target.parquet"
    train_ids_path = OUTPUTS / "04_train_test_split/tables/train_ids.csv"
    for path in (feature_path, target_path, train_ids_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input is missing: {path}. Run earlier stages first.")

    X_all = pd.read_parquet(feature_path)
    y_all = pd.read_parquet(target_path)["pIC50"]
    train_ids = pd.read_csv(train_ids_path, dtype={"chembl_id": str})
    if train_ids.chembl_id.duplicated().any():
        raise ValueError("The training ID file contains duplicate molecule IDs.")

    X_all.index = X_all.index.astype(str)
    y_all.index = y_all.index.astype(str)
    train_ids["chembl_id"] = train_ids.chembl_id.astype(str)
    X = X_all.reindex(train_ids.chembl_id.to_list())
    y = y_all.reindex(train_ids.chembl_id.to_list())
    groups = train_ids["group"].to_numpy()

    target_ok = y.notna().to_numpy() & np.isfinite(y.to_numpy(dtype=float))
    X, y, groups = X.loc[target_ok].copy(), y.loc[target_ok], groups[target_ok]
    if X.empty or pd.isna(groups).any():
        raise ValueError("No usable training rows remain or some training groups are missing.")

    # No imputation: remove features incomplete anywhere in the training
    # partition. This uses predictors only; fold-wise supervised ranking below
    # is learned separately within each CV training split.
    X = X.apply(pd.to_numeric, errors="coerce")
    values = X.to_numpy(dtype=np.float64)
    complete_finite = X.notna().all(axis=0).to_numpy() & np.isfinite(values).all(axis=0)
    X = X.loc[:, complete_finite]
    if X.shape[1] == 0:
        raise ValueError("All training features contain missing or non-finite values.")

    return X, y, groups, feature_path, target_path, train_ids_path


def _prepare_folds(X, y, groups, cfg, out_dir, cv=None):
    n_splits = min(int(cfg.get("cv_splits", 5)), len(np.unique(groups)))
    if n_splits != 5:
        raise ValueError(
            f"Optuna tuning requires five distinct scaffold groups at minimum; got {n_splits}."
        )

    cv = cv or GroupKFold(n_splits=5)
    folds = []
    threshold = float(cfg.get("correlation_threshold", 0.90))
    selection_rows = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_fold_train = X.iloc[train_idx]
        X_fold_valid = X.iloc[valid_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_valid = y.iloc[valid_idx]

        retained, audit = fit_fold_cleaner(X_fold_train, threshold)
        X_clean_train = X_fold_train.loc[:, retained]
        X_clean_valid = X_fold_valid.loc[:, retained]
        with threadpool_limits(limits=THREADS):
            ranker = make_ranker(int(cfg.get("seed", 42)) + fold).fit(
                X_clean_train, y_fold_train
            )
        ranking = pd.DataFrame(
            {"feature": retained, "importance": ranker.feature_importances_}
        ).sort_values("importance", ascending=False, kind="stable")
        chosen = ranking.feature.iloc[: min(FEATURE_BUDGET, len(ranking))].tolist()
        if not chosen:
            raise ValueError(f"Fold {fold} has no features after cleaning and ranking.")
        audit.to_csv(out_dir / f"fold_{fold:02d}_cleaning_audit.csv", index=False)
        pd.DataFrame({"feature": chosen}).to_csv(
            out_dir / f"fold_{fold:02d}_selected_features.csv", index=False
        )
        selection_rows.extend({"fold": fold, "feature": name} for name in chosen)
        folds.append(
            {
                "fold": fold,
                "X_train": X_clean_train.loc[:, chosen],
                "X_valid": X_clean_valid.loc[:, chosen],
                "y_train": y_fold_train,
                "y_valid": y_fold_valid,
                "selected_features": chosen,
                "valid_ids": X.index[valid_idx].astype(str).tolist(),
                "valid_groups": np.asarray(groups)[valid_idx].tolist(),
            }
        )

    pd.DataFrame(selection_rows).to_csv(
        out_dir / "fold_specific_selected_features.csv", index=False
    )
    return folds


def _evaluate_best_params(model_name, params, folds, cfg):
    """Refit the selected settings on each training fold and save OOF records."""
    predictions = []
    fold_metrics = []
    for fold_data in folds:
        model = make_estimator(
            model_name,
            params,
            int(cfg.get("seed", 42)) + fold_data["fold"],
        )
        with threadpool_limits(limits=THREADS):
            model.fit(fold_data["X_train"], fold_data["y_train"])
            predicted = np.asarray(model.predict(fold_data["X_valid"]), dtype=float)
        observed = fold_data["y_valid"].to_numpy(dtype=float)
        if not np.isfinite(predicted).all():
            raise ValueError(f"{model_name} returned non-finite CV predictions.")
        predictions.append(
            pd.DataFrame(
                {
                    "chembl_id": fold_data["valid_ids"],
                    "group": fold_data["valid_groups"],
                    "fold": fold_data["fold"],
                    "model": model_name,
                    "observed_pIC50": observed,
                    "predicted_pIC50": predicted,
                    "residual": observed - predicted,
                }
            )
        )
        fold_metrics.append(
            {
                "model": model_name,
                "fold": fold_data["fold"],
                "r2": float(r2_score(observed, predicted)),
                "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
                "mae": float(mean_absolute_error(observed, predicted)),
            }
        )
    oof = pd.concat(predictions, ignore_index=True)
    if oof.chembl_id.duplicated().any() or len(oof) != sum(len(f["y_valid"]) for f in folds):
        raise RuntimeError(f"OOF prediction alignment failed for {model_name}.")
    fold_frame = pd.DataFrame(fold_metrics)
    summary = {
        "model": model_name,
        "cv_mean_fold_rmse": float(fold_frame.rmse.mean()),
        "cv_sd_fold_rmse": float(fold_frame.rmse.std(ddof=1)),
        "cv_pooled_oof_rmse": float(
            np.sqrt(mean_squared_error(oof.observed_pIC50, oof.predicted_pIC50))
        ),
        "cv_pooled_oof_mae": float(
            mean_absolute_error(oof.observed_pIC50, oof.predicted_pIC50)
        ),
        "cv_pooled_oof_r2": float(
            r2_score(oof.observed_pIC50, oof.predicted_pIC50)
        ),
    }
    return summary, fold_frame, oof


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
        fold_rmse = []
        for fold_data in folds:
            model = make_estimator(
                model_name,
                params,
                int(cfg.get("seed", 42)) + fold_data["fold"],
            )
            with threadpool_limits(limits=THREADS):
                model.fit(fold_data["X_train"], fold_data["y_train"])
                prediction = model.predict(fold_data["X_valid"])
            rmse = float(
                np.sqrt(mean_squared_error(fold_data["y_valid"], prediction))
            )
            fold_rmse.append(rmse)
        trial.set_user_attr("fold_rmse", fold_rmse)
        return float(np.mean(fold_rmse))

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

    X, y, groups, feature_path, target_path, train_ids_path = _load_training_data()
    d = stage_dir("07_optuna_tuning")
    fold_dir = d / "models/fold_feature_cache"
    fold_dir.mkdir(parents=True, exist_ok=True)

    run_material = {
        "features": sha256(feature_path),
        "target": sha256(target_path),
        "train_ids": sha256(train_ids_path),
        "tuner_script": sha256(Path(__file__)),
        "cleaner_script": sha256(Path(__file__).with_name("feature_selection.py")),
        "feature_budget": FEATURE_BUDGET,
        "correlation_threshold": float(cfg.get("correlation_threshold", 0.90)),
        "feature_selection": "fold-specific cleaning, ExtraTrees ranking, and top-250 selection",
        "folds": 5,
        "models": MODELS,
        "seed": int(cfg.get("seed", 42)),
    }
    run_id = hashlib.sha256(
        json.dumps(run_material, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    # Scaffold folds are computed once and reused across Optuna trials.
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
    for model_name in MODELS:
        print(f"Optuna: {model_name}, 5-fold GroupKFold, up to {n_trials} trials", flush=True)
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
    best_cv_oof = []
    for model_name, (study, trial) in studies.items():
        cv_summary, cv_folds, cv_oof = _evaluate_best_params(
            model_name, trial.params, folds, cfg
        )
        best_params[model_name] = {
            "params": trial.params,
            "best_cv_rmse": float(trial.value),
            **cv_summary,
            "fold_rmse": trial.user_attrs.get("fold_rmse", []),
            "trial_number": int(trial.number),
            "study_name": study.study_name,
        }
        best_records.append(
            {
                "model": model_name,
                "best_cv_rmse": float(trial.value),
                "trial_number": int(trial.number),
                "fold_rmse": json.dumps(trial.user_attrs.get("fold_rmse", [])),
                "params": json.dumps(trial.params, sort_keys=True),
                **cv_summary,
            }
        )
        best_cv_folds.append(cv_folds)
        best_cv_oof.append(cv_oof)
        cv_oof.to_csv(
            d / f"tables/best_params_oof_predictions_{model_name}.csv", index=False
        )
        study.trials_dataframe().to_csv(
            d / f"tables/optuna_trials_{model_name}.csv", index=False
        )

    summary = pd.DataFrame(best_records).sort_values("best_cv_rmse")
    summary.to_csv(d / "tables/optuna_model_comparison.csv", index=False)
    pd.concat(best_cv_folds, ignore_index=True).to_csv(
        d / "tables/best_params_cv_fold_metrics.csv", index=False
    )
    pd.concat(best_cv_oof, ignore_index=True).to_csv(
        d / "tables/best_params_cv_oof_predictions_all_models.csv", index=False
    )
    (d / "tables/optuna_best_params.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "cv": "5-fold GroupKFold by Bemis-Murcko scaffold/connectivity group",
                "objective": "mean fold RMSE",
                "feature_budget": FEATURE_BUDGET,
                "feature_selection": "filters and ExtraTrees top-250 ranking fitted within each training fold",
                "no_imputation": True,
                "params_by_model": best_params,
                "recommended_model_by_tuning_cv": str(summary.iloc[0].model),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nOptuna tuning summary (training-only grouped CV):")
    print(summary[["model", "best_cv_rmse", "trial_number"]].to_string(index=False))
    print("These tuned CV scores are for model selection; final generalization is assessed on the locked test set in Stage 08.")

    write_manifest(
        "07_optuna_tuning",
        {
            **run_material,
            "models": list(MODELS),
            "best_model_by_cv": str(summary.iloc[0].model),
            "test_used": False,
        },
        [feature_path, target_path, train_ids_path],
        [
            d / "tables/optuna_model_comparison.csv",
            d / "tables/optuna_best_params.json",
            d / "tables/best_params_cv_fold_metrics.csv",
            d / "tables/best_params_cv_oof_predictions_all_models.csv",
        ],
    )
    return summary


if __name__ == "__main__":
    run()
