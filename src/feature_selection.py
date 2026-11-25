"""Stage 05: fold-wise cleaning, Extra Trees ranking, and grouped model comparison.

Cleaning is learned using training data only. Per the user's pipeline rule,
features with any missing/non-finite value are dropped; no imputation is used.
"""
import hashlib
import json
import warnings
from pathlib import Path

import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from src.utils import (
    OUTPUTS,
    load_config,
    save_figure,
    save_joblib,
    sha256,
    setup_plotting,
    stage_dir,
    write_manifest,
)


SETTINGS = {
    "folds": 5,
    "budgets": [100, 250, 500, "all"],
    "models": ["ExtraTrees", "HistGradientBoosting", "LightGBM"],
    "seed": 42,
    "threads": 2,
    "selector_trees": 200,
    "extra_trees": 300,
}


def family(feature):
    name = str(feature)
    if name.startswith("ecfp4_"):
        return "Morgan"
    if name.startswith("mordred2d_"):
        return "Mordred 2D"
    if name.startswith("mordred3d_"):
        return "Mordred 3D"
    return "Physicochemical"


def fit_fold_cleaner(X_train, threshold):
    """Fit no-imputation filters on one fold's training rows only."""
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    values = X_train.to_numpy(dtype=np.float64)
    complete = X_train.notna().all(axis=0).to_numpy()
    finite = np.isfinite(values).all(axis=0)
    valid_mask = complete & finite
    valid_features = X_train.columns[valid_mask].tolist()
    dropped = []

    for i, feature in enumerate(X_train.columns):
        reasons = []
        if not complete[i]:
            reasons.append("missing_value_in_fold_training")
        if not finite[i]:
            reasons.append("nonfinite_value_in_fold_training")
        if reasons:
            dropped.append({"feature": feature, "reason": ";".join(reasons)})

    if not valid_features:
        raise ValueError("No complete, finite features remain in this fold's training data.")
    complete_train = X_train.loc[:, valid_features]

    variable_mask = (complete_train.nunique(dropna=False) > 1).to_numpy()
    variable_features = complete_train.columns[variable_mask].tolist()
    for feature in complete_train.columns[~variable_mask]:
        dropped.append({"feature": feature, "reason": "constant_in_fold_training"})
    if not variable_features:
        raise ValueError("No nonconstant features remain in this fold.")
    variable_train = complete_train.loc[:, variable_features]

    # Pearson correlation, absolute threshold, deterministic greedy retention:
    # earlier columns are prioritized; later highly correlated columns are removed.
    correlation = variable_train.corr(method="pearson").abs().fillna(0.0)
    retained = []
    for feature in variable_features:
        correlated_with = next(
            (
                kept
                for kept in retained
                if correlation.at[feature, kept] >= threshold
            ),
            None,
        )
        if correlated_with is None:
            retained.append(feature)
        else:
            dropped.append(
                {
                    "feature": feature,
                    "reason": f"abs_pearson_correlation_ge_{threshold:.2f}",
                    "correlated_with": correlated_with,
                    "absolute_correlation": float(correlation.at[feature, correlated_with]),
                }
            )
    if not retained:
        raise ValueError("No features remain after the correlation filter.")

    audit = pd.DataFrame(dropped)
    return retained, audit


def make_ranker(seed):
    return ExtraTreesRegressor(
        n_estimators=SETTINGS["selector_trees"],
        min_samples_leaf=2,
        max_features=1.0,
        random_state=seed,
        n_jobs=SETTINGS["threads"],
    )


def make_model(name, seed):
    if name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=SETTINGS["extra_trees"],
            random_state=seed,
            n_jobs=SETTINGS["threads"],
        )
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            random_state=seed,
            early_stopping=False,
        )
    if name == "LightGBM":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("Install lightgbm to run the supplied Stage 05 model comparison.") from exc
        return LGBMRegressor(
            random_state=seed,
            n_jobs=SETTINGS["threads"],
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    raise ValueError(f"Unsupported model: {name}")


def _scores(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def run():
    cfg = load_config()
    feature_path = OUTPUTS / "03_feature_generation/tables/all_features.parquet"
    target_path = OUTPUTS / "03_feature_generation/tables/target.parquet"
    train_ids_path = OUTPUTS / "04_train_test_split/tables/train_ids.csv"
    if not feature_path.exists() or not target_path.exists() or not train_ids_path.exists():
        raise FileNotFoundError("Complete Stages 03 and 04 before Stage 05.")

    X_all = pd.read_parquet(feature_path)
    y_all = pd.read_parquet(target_path)["pIC50"]
    ids = pd.read_csv(train_ids_path, dtype={"chembl_id": str})
    ids["chembl_id"] = ids["chembl_id"].astype(str)
    if ids.chembl_id.duplicated().any():
        raise ValueError("Training ID table contains duplicate molecule IDs.")

    X_all.index = X_all.index.astype(str)
    y_all.index = y_all.index.astype(str)
    train_ids = ids.set_index("chembl_id")
    X = X_all.loc[X_all.index.isin(train_ids.index)].copy()
    y = y_all.reindex(X.index)
    groups = train_ids.reindex(X.index)["group"].to_numpy()

    target_ok = y.notna().to_numpy() & np.isfinite(y.to_numpy(dtype=float))
    X = X.loc[target_ok].copy()
    y = y.loc[target_ok]
    groups = groups[target_ok]
    if X.empty:
        raise ValueError("No valid training rows remain after aligning IDs and pIC50.")
    if pd.isna(groups).any():
        raise ValueError("Some training molecules have no scaffold/connectivity group.")

    # Match the user's strict no-imputation policy. This completeness screen uses
    # only the training partition; the locked test rows are never loaded here.
    X = X.apply(pd.to_numeric, errors="coerce")
    full_values = X.to_numpy(dtype=np.float64)
    complete = X.notna().all(axis=0).to_numpy()
    finite = np.isfinite(full_values).all(axis=0)
    globally_valid = complete & finite
    global_drop_records = []
    for i, feature in enumerate(X.columns):
        reasons = []
        if not complete[i]:
            reasons.append("missing_value_in_training_partition")
        if not finite[i]:
            reasons.append("nonfinite_value_in_training_partition")
        if reasons:
            global_drop_records.append(
                {"feature": feature, "reason": ";".join(reasons)}
            )
    X = X.loc[:, globally_valid]
    if X.shape[1] == 0:
        raise ValueError("All features contain missing or nonfinite training values.")

    n_splits = min(SETTINGS["folds"], len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("At least two training scaffold groups are required for GroupKFold.")
    cv = GroupKFold(n_splits=n_splits)
    folds = list(cv.split(X, y, groups))
    for train_idx, valid_idx in folds:
        if set(groups[train_idx]) & set(groups[valid_idx]):
            raise RuntimeError("Scaffold/connectivity leakage detected between CV folds.")

    d = stage_dir("05_feature_selection")
    run_identity = {
        "features_sha256": sha256(feature_path),
        "target_sha256": sha256(target_path),
        "train_ids_sha256": sha256(train_ids_path),
        "stage_script_sha256": sha256(Path(__file__)),
        "threshold": float(cfg.get("correlation_threshold", 0.90)),
        "settings": SETTINGS,
        "seed": cfg["seed"],
    }
    run_id = hashlib.sha256(
        json.dumps(run_identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    checkpoint_root = d / "models/cv_checkpoints" / run_id
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (d / "models/cv_checkpoints" / "active_run.json").write_text(
        json.dumps({"run_id": run_id, **run_identity}, indent=2), encoding="utf-8"
    )
    (d / "tables/dropped_features.csv")
    pd.DataFrame(global_drop_records).to_csv(
        d / "tables/dropped_features.csv", index=False
    )
    assignments = pd.DataFrame(
        {"chembl_id": X.index.astype(str), "group": groups, "cv_fold": -1}
    )
    for fold_no, (_, valid_idx) in enumerate(folds, start=1):
        assignments.loc[valid_idx, "cv_fold"] = fold_no
    assignments.to_csv(d / "tables/train_cv_fold_assignments.csv", index=False)

    fold_records = []
    selection_records = []
    for fold_no, (train_idx, valid_idx) in enumerate(folds, start=1):
        fold_dir = checkpoint_root / f"fold_{fold_no:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        X_train_raw = X.iloc[train_idx]
        X_valid_raw = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        # Learn constant/correlation filters and feature importance inside this fold.
        retained, audit = fit_fold_cleaner(
            X_train_raw,
            threshold=float(cfg.get("correlation_threshold", 0.90)),
        )
        audit.to_csv(fold_dir / "cleaning_decisions.csv", index=False)
        X_train = X_train_raw.loc[:, retained]
        X_valid = X_valid_raw.loc[:, retained]

        rank_path = fold_dir / "training_importance.csv"
        ranker_path = fold_dir / "ranker.joblib"
        if rank_path.exists() and ranker_path.exists():
            ranking = pd.read_csv(rank_path)
            ranker = joblib.load(ranker_path)
            # Reuse only if the cached ranking covers exactly this fold's retained features.
            if len(ranking) != len(retained) or set(ranking.feature) != set(retained):
                with threadpool_limits(limits=SETTINGS["threads"]):
                    ranker = make_ranker(cfg["seed"] + fold_no).fit(X_train, y_train)
                ranking = pd.DataFrame(
                    {"feature": retained, "family": [family(f) for f in retained],
                     "importance": ranker.feature_importances_}
                ).sort_values("importance", ascending=False).reset_index(drop=True)
                ranking.to_csv(rank_path, index=False)
                save_joblib(ranker, ranker_path)
        else:
            with threadpool_limits(limits=SETTINGS["threads"]):
                ranker = make_ranker(cfg["seed"] + fold_no).fit(X_train, y_train)
            ranking = pd.DataFrame(
                {"feature": retained, "family": [family(f) for f in retained],
                 "importance": ranker.feature_importances_}
            ).sort_values("importance", ascending=False).reset_index(drop=True)
            ranking.to_csv(rank_path, index=False)
            save_joblib(ranker, ranker_path)

        ordered_features = ranking.feature.tolist()
        for budget in SETTINGS["budgets"]:
            k = len(ordered_features) if budget == "all" else min(int(budget), len(ordered_features))
            selected = ordered_features[:k]
            for model_name in SETTINGS["models"]:
                tag = f"{model_name}_k{budget}"
                metrics_file = fold_dir / f"{tag}.json"
                predictions_file = fold_dir / f"{tag}_predictions.csv"
                if metrics_file.exists() and predictions_file.exists():
                    result = json.loads(metrics_file.read_text(encoding="utf-8"))
                    fold_records.append(result)
                    selection_records.extend(
                        {"fold": fold_no, "model": model_name, "budget": str(budget),
                         "feature": feature, "family": family(feature)}
                        for feature in selected
                    )
                    continue

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with threadpool_limits(limits=SETTINGS["threads"]):
                        model = make_model(model_name, cfg["seed"] + fold_no)
                        model.fit(X_train[selected], y_train)
                        prediction = model.predict(X_valid[selected])
                if not np.isfinite(prediction).all():
                    raise ValueError(f"Nonfinite predictions in fold {fold_no}, {tag}.")
                result = {
                    "fold": fold_no,
                    "model": model_name,
                    "budget": str(budget),
                    "n_features": len(selected),
                    "n_train": len(train_idx),
                    "n_validation": len(valid_idx),
                    **_scores(y_valid, prediction),
                    "warnings": " | ".join(sorted({str(w.message) for w in caught})),
                }
                metrics_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
                pd.DataFrame(
                    {
                        "chembl_id": X.index[valid_idx],
                        "observed_pIC50": y_valid.to_numpy(),
                        "predicted_pIC50": prediction,
                    }
                ).to_csv(predictions_file, index=False)
                fold_records.append(result)
                selection_records.extend(
                    {"fold": fold_no, "model": model_name, "budget": str(budget),
                     "feature": feature, "family": family(feature)}
                    for feature in selected
                )
                print(
                    f"Fold {fold_no}/{n_splits} | {model_name} | {budget} features | "
                    f"RMSE={result['RMSE']:.4f}",
                    flush=True,
                )

    results = pd.DataFrame(fold_records).drop_duplicates(
        subset=["fold", "model", "budget"], keep="last"
    )
    if results.empty or not results.groupby(["model", "budget"]).fold.nunique().eq(n_splits).all():
        raise RuntimeError("CV results are incomplete; inspect fold checkpoint files.")
    results.to_csv(d / "tables/feature_selection_cv_fold_results.csv", index=False)

    summary = (
        results.groupby(["model", "budget"], sort=False)
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_sd=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            R2_mean=("R2", "mean"),
            n_features_min=("n_features", "min"),
            n_features_max=("n_features", "max"),
            n_folds=("fold", "nunique"),
        )
        .reset_index()
    )
    summary["RMSE_se_descriptive"] = summary.RMSE_sd / np.sqrt(summary.n_folds)

    # Pool out-of-fold predictions separately from mean-fold metrics.
    pooled = []
    for (model_name, budget), group in results.groupby(["model", "budget"]):
        frames = []
        for fold_no in range(1, n_splits + 1):
            file = checkpoint_root / f"fold_{fold_no:02d}" / f"{model_name}_k{budget}_predictions.csv"
            frames.append(pd.read_csv(file))
        oof = pd.concat(frames, ignore_index=True)
        if len(oof) != len(X) or oof.chembl_id.duplicated().any():
            raise RuntimeError(f"OOF prediction alignment failed for {model_name}, {budget}.")
        pooled.append(
            {
                "model": model_name,
                "budget": budget,
                "OOF_R2": r2_score(oof.observed_pIC50, oof.predicted_pIC50),
                "OOF_RMSE": float(np.sqrt(mean_squared_error(oof.observed_pIC50, oof.predicted_pIC50))),
                "OOF_MAE": mean_absolute_error(oof.observed_pIC50, oof.predicted_pIC50),
            }
        )
        oof["model"], oof["budget"] = model_name, budget
        oof.to_csv(
            d / f"tables/oof_predictions_{model_name}_k{budget}.csv", index=False
        )
    summary = summary.merge(pd.DataFrame(pooled), on=["model", "budget"])
    summary.to_csv(d / "tables/feature_selection_cv_summary.csv", index=False)

    # One-standard-error parsimony rule from the supplied Stage 05 notebook.
    best = summary.loc[summary.RMSE_mean.idxmin()]
    cutoff = float(best.RMSE_mean + best.RMSE_se_descriptive)
    candidates = summary.loc[summary.RMSE_mean <= cutoff].copy()
    candidates["budget_order"] = candidates.budget.map(
        lambda value: float("inf") if value == "all" else int(value)
    )
    recommendation = candidates.sort_values(
        ["budget_order", "RMSE_mean", "MAE_mean"]
    ).iloc[0]
    summary["within_one_SE_heuristic"] = summary.RMSE_mean <= cutoff
    summary.to_csv(d / "tables/feature_selection_cv_summary.csv", index=False)
    pd.DataFrame([recommendation.drop(labels="budget_order")]).to_csv(
        d / "tables/recommended_model_and_budget.csv", index=False
    )

    selection = pd.DataFrame(selection_records).drop_duplicates(
        subset=["fold", "model", "budget", "feature"]
    )
    selection.to_csv(d / "tables/fold_selected_features.csv", index=False)
    frequency = (
        selection.groupby(["model", "budget", "feature", "family"]).fold.nunique()
        .rename("n_folds_selected").reset_index()
    )
    frequency["selection_fraction"] = frequency.n_folds_selected / n_splits
    frequency.to_csv(d / "tables/feature_selection_frequency.csv", index=False)

    # Match the source notebook's stability and family-coverage diagnostics.
    jaccard_records = []
    family_records = []
    for (model_name, budget), subset in selection.groupby(["model", "budget"]):
        by_fold = {
            fold_no: set(part.feature)
            for fold_no, part in subset.groupby("fold")
        }
        for left in range(1, n_splits + 1):
            for right in range(left + 1, n_splits + 1):
                left_features = by_fold.get(left, set())
                right_features = by_fold.get(right, set())
                union = left_features | right_features
                jaccard_records.append(
                    {
                        "model": model_name,
                        "budget": budget,
                        "fold_a": left,
                        "fold_b": right,
                        "jaccard": (
                            len(left_features & right_features) / len(union)
                            if union
                            else 1.0
                        ),
                    }
                )
        counts = (
            subset.groupby(["fold", "family"]).feature.nunique()
            .rename("n_features").reset_index()
        )
        for family_name in sorted(set(selection.family)):
            family_fold_counts = counts.loc[counts.family.eq(family_name), "n_features"]
            family_records.append(
                {
                    "model": model_name,
                    "budget": budget,
                    "family": family_name,
                    "mean_features_across_folds": (
                        float(family_fold_counts.sum()) / n_splits
                    ),
                }
            )
    pd.DataFrame(jaccard_records).to_csv(
        d / "tables/feature_selection_stability_jaccard.csv", index=False
    )
    pd.DataFrame(family_records).to_csv(
        d / "tables/selected_feature_family_counts.csv", index=False
    )

    setup_plotting()
    fig, ax = plt.subplots(figsize=(10, 6))
    budget_order = [str(value) for value in SETTINGS["budgets"]]
    colors = {"ExtraTrees": "#0072B2", "HistGradientBoosting": "#D55E00", "LightGBM": "#009E73"}
    x_positions = np.arange(len(budget_order))
    offsets = np.linspace(-0.18, 0.18, len(SETTINGS["models"]))
    for offset, model_name in zip(offsets, SETTINGS["models"]):
        part = summary.loc[summary.model.eq(model_name)].set_index("budget").reindex(budget_order)
        ax.errorbar(
            x_positions + offset,
            part.RMSE_mean,
            yerr=part.RMSE_sd,
            marker="o",
            capsize=4,
            linewidth=2.8,
            label=model_name,
            color=colors[model_name],
        )
    ax.set_xticks(x_positions, [f"Top {b}" if b != "all" else "All retained" for b in budget_order])
    ax.set(xlabel="Feature budget", ylabel="GroupKFold RMSE (pIC50)", title="Model performance by feature budget")
    ax.legend(frameon=False)
    save_figure(fig, d / "figures/feature_budget_model_comparison")
    plt.close(fig)

    # Save a single final feature ranking from all training rows for downstream use.
    retained, final_audit = fit_fold_cleaner(
        X, threshold=float(cfg.get("correlation_threshold", 0.90))
    )
    final_audit.to_csv(d / "tables/final_training_cleaning_decisions.csv", index=False)
    X_final = X.loc[:, retained]
    final_ranker = make_ranker(cfg["seed"]).fit(X_final, y)
    final_ranking = pd.DataFrame(
        {"feature": retained, "family": [family(f) for f in retained],
         "importance": final_ranker.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    final_ranking.to_csv(d / "tables/final_training_feature_ranking.csv", index=False)
    chosen_budget = recommendation.budget
    actual_k = len(final_ranking) if chosen_budget == "all" else min(int(chosen_budget), len(final_ranking))
    selected_features = final_ranking.feature.iloc[:actual_k].tolist()
    selected_features_table = final_ranking.head(actual_k).copy()
    selected_features_table.to_csv(d / "tables/selected_features.csv", index=False)
    if actual_k == 250:
        selected_features_table.to_csv(d / "tables/selected_features_250.csv", index=False)
    X_selected = X_final[selected_features].copy()
    X_selected.to_parquet(d / "tables/train_selected_features.parquet")
    y.rename("pIC50").to_frame().to_parquet(d / "tables/train_target.parquet")
    pd.DataFrame({"chembl_id": X.index, "group": groups}).to_csv(
        d / "tables/train_groups.csv", index=False
    )
    save_joblib(
        {
            "complete_training_features": list(X_all.columns),
            "training_complete_features": list(X.columns),
            "retained_after_constant_and_correlation": retained,
            "selected_features": selected_features,
            "correlation_threshold": cfg.get("correlation_threshold", 0.90),
            "missing_policy": "drop any feature with missing values; no imputation",
            "nonfinite_policy": "drop any feature with nonfinite values",
            "constant_policy": "fit per training fold and final training partition",
            "final_ranker": final_ranker,
            "recommendation": recommendation.to_dict(),
        },
        d / "models/cleaner_and_selector.joblib",
    )

    print(
        f"Training-only GroupKFold: {len(X):,} molecules, {X.shape[1]:,} complete finite features; "
        f"selected {actual_k} features. Test set was not loaded."
    )
    print(f"One-SE recommendation: {recommendation.model} with {recommendation.budget} features.")
    write_manifest(
        "05_feature_selection",
        {
            **SETTINGS,
            "cv": "GroupKFold using scaffold/connectivity groups",
            "n_splits": n_splits,
            "missing_policy": "drop columns with any missing value; no imputation",
            "nonfinite_policy": "drop columns with any nonfinite value",
            "constant_policy": "drop constants fitted on each fold training partition",
            "correlation_threshold": float(cfg.get("correlation_threshold", 0.90)),
            "selector": "ExtraTreesRegressor MDI ranking per fold",
            "models_compared": SETTINGS["models"],
            "budgets": SETTINGS["budgets"],
            "test_used": False,
            "recommendation_rule": "smallest budget within best mean RMSE + descriptive fold SE",
        },
        [feature_path, target_path, train_ids_path],
        [
            d / "tables/feature_selection_cv_summary.csv",
            d / "tables/selected_features.csv",
            d / "models/cleaner_and_selector.joblib",
        ],
    )
    return X_selected, y


if __name__ == "__main__":
    run()
