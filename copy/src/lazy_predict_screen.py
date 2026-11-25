"""Stage 06: scaffold-grouped LazyPredict screening on training data only."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from src.utils import (
    OUTPUTS,
    load_config,
    save_figure,
    setup_plotting,
    stage_dir,
    write_manifest,
)


def _finite_scores(y_true, y_pred):
    """Calculate metrics using only finite observed/predicted values."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2:
        return None

    observed = y_true[valid]
    predicted = y_pred[valid]

    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "n_scored": int(valid.sum()),
    }


def _normalise_name(value):
    """Normalize metric labels for version-tolerant matching."""
    return "".join(char for char in str(value).lower() if char.isalnum())


def _metrics_from_lazy_scores(scores, model_name):
    """Read metrics from LazyPredict's score table if predictions are absent."""
    if not isinstance(scores, pd.DataFrame) or scores.empty:
        return None

    if model_name in scores.index:
        row = scores.loc[model_name]
    else:
        matching = scores.index.astype(str) == str(model_name)
        if not matching.any():
            return None
        row = scores.loc[matching].iloc[0]

    metrics = {_normalise_name(key): value for key, value in row.items()}

    def find_value(*names):
        for name in names:
            value = metrics.get(_normalise_name(name))
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
        return None

    r2 = find_value("R-Squared", "R2", "R Squared")
    rmse = find_value("RMSE", "Root Mean Squared Error")
    mae = find_value("mean_absolute_error", "MAE", "Mean Absolute Error")

    if r2 is None or rmse is None or mae is None:
        return None

    return {"r2": r2, "rmse": rmse, "mae": mae, "n_scored": np.nan}


def run():
    cfg = load_config()
    feature_dir = OUTPUTS / "05_feature_selection/tables"

    feature_path = feature_dir / "train_selected_features.parquet"
    target_path = feature_dir / "train_target.parquet"
    groups_path = feature_dir / "train_groups.csv"

    for path in (feature_path, target_path, groups_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required Stage 05 file is missing: {path}. "
                "Run Stage 05 before Stage 06."
            )

    X = pd.read_parquet(feature_path)
    y = pd.read_parquet(target_path)["pIC50"]
    groups = pd.read_csv(groups_path)["group"].to_numpy()

    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError(
            "Feature, target, and group row counts differ: "
            f"X={len(X)}, y={len(y)}, groups={len(groups)}."
        )
    if X.empty or X.shape[1] == 0:
        raise ValueError("The selected training feature table is empty.")
    if y.isna().any() or not np.isfinite(y.to_numpy(dtype=float)).all():
        raise ValueError("The training target contains missing or infinite values.")
    if pd.isna(groups).any():
        raise ValueError("The training group table contains missing scaffold groups.")

    X = X.apply(pd.to_numeric, errors="coerce")
    values = X.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(
            "Selected training features contain missing or infinite values. "
            "Rerun Stage 05 to apply the no-imputation feature filter."
        )

    try:
        from lazypredict.Supervised import LazyRegressor
    except ImportError as exc:
        raise ImportError(
            "LazyPredict is not available in this notebook environment. "
            "Install it with: python -m pip install lazypredict"
        ) from exc

    n_splits = min(int(cfg.get("cv_splits", 5)), len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError(
            "At least two distinct training scaffold groups are required "
            "for GroupKFold."
        )

    cv = GroupKFold(n_splits=n_splits)
    records = []
    fold_errors = []

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):
        # Each model sees only this fold's training rows and validation rows.
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # LazyPredict returns a prediction DataFrame only when
                # predictions=True. Some versions return scores alone when
                # predictions=False.
                regressor = LazyRegressor(
                    verbose=0,
                    ignore_warnings=False,
                    custom_metric=mean_absolute_error,
                    predictions=True,
                )

                with threadpool_limits(limits=2):
                    fit_result = regressor.fit(
                        X_train,
                        X_valid,
                        y_train,
                        y_valid,
                    )

        except Exception as exc:
            message = f"Fold {fold}: LazyPredict fit failed: {type(exc).__name__}: {exc}"
            print(message, flush=True)
            fold_errors.append(message)
            continue

        scores = None
        predictions = None

        if isinstance(fit_result, tuple):
            if len(fit_result) >= 1:
                scores = fit_result[0]
            if len(fit_result) >= 2:
                predictions = fit_result[1]
        else:
            scores = fit_result

        if not isinstance(scores, pd.DataFrame):
            scores = pd.DataFrame(scores) if scores is not None else pd.DataFrame()

        if predictions is not None and not isinstance(predictions, pd.DataFrame):
            predictions = pd.DataFrame(predictions)

        fold_record_count = 0

        # Prefer recomputing all metrics from returned fold predictions.
        if isinstance(predictions, pd.DataFrame) and not predictions.empty:
            for model_name in predictions.columns:
                metrics = _finite_scores(
                    y_valid.to_numpy(),
                    pd.to_numeric(
                        predictions[model_name], errors="coerce"
                    ).to_numpy(),
                )
                if metrics is None:
                    continue

                records.append(
                    {
                        "fold": fold,
                        "model": str(model_name),
                        "r2": metrics["r2"],
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "n_scored": metrics["n_scored"],
                        "n_validation": len(valid_idx),
                        "metric_source": "validation_predictions",
                    }
                )
                fold_record_count += 1

        # Compatibility fallback: use LazyPredict's score table when that
        # installed version does not return prediction columns.
        if fold_record_count == 0 and not scores.empty:
            score_table = scores.copy()
            if "Model" in score_table.columns:
                score_table = score_table.set_index("Model")

            for model_name in score_table.index:
                metrics = _metrics_from_lazy_scores(score_table, model_name)
                if metrics is None:
                    continue

                records.append(
                    {
                        "fold": fold,
                        "model": str(model_name),
                        "r2": metrics["r2"],
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "n_scored": metrics["n_scored"],
                        "n_validation": len(valid_idx),
                        "metric_source": "lazypredict_score_table",
                    }
                )
                fold_record_count += 1

        if fold_record_count == 0:
            available_columns = list(scores.columns)
            message = (
                f"Fold {fold}: no usable model scores or prediction columns. "
                f"Score-table columns: {available_columns}; "
                f"prediction columns: "
                f"{list(predictions.columns) if isinstance(predictions, pd.DataFrame) else 'none'}."
            )
            print(message, flush=True)
            fold_errors.append(message)
        else:
            print(
                f"Fold {fold}/{n_splits}: collected "
                f"{fold_record_count} model results.",
                flush=True,
            )

    if not records:
        details = "\n".join(fold_errors) if fold_errors else "No fold results were returned."
        raise RuntimeError(
            "LazyPredict produced no successful fold-level results.\n"
            f"{details}\n"
            "Check that this notebook imports the updated src/lazy_predict_screen.py, "
            "then inspect the per-model errors printed by LazyPredict."
        )

    raw = pd.DataFrame(records)
    output_dir = stage_dir("06_lazypredict")

    raw.to_csv(
        output_dir / "tables/lazypredict_fold_metrics.csv",
        index=False,
    )

    summary = (
        raw.groupby("model")
        .agg(
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            folds=("fold", "nunique"),
            min_validation_rows_scored=("n_scored", "min"),
        )
        .reset_index()
        .sort_values(
            ["mean_r2", "mean_rmse"],
            ascending=[False, True],
        )
    )
    summary["expected_folds"] = n_splits
    summary["complete_across_folds"] = summary["folds"] == n_splits

    summary.to_csv(
        output_dir / "tables/lazypredict_results.csv",
        index=False,
    )
    summary.head(3).to_csv(
        output_dir / "tables/recommended_top3.csv",
        index=False,
    )

    setup_plotting()
    top = summary.head(20).sort_values("mean_r2")
    fig, ax = plt.subplots(
        figsize=(10, max(6, len(top) * 0.34))
    )
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(top)))
    ax.barh(top["model"], top["mean_r2"], color=colors)
    ax.set(
        xlabel="Mean GroupKFold R²",
        title="LazyPredict screening on training data",
    )
    save_figure(fig, output_dir / "figures/lazypredict_top20_r2")
    plt.close(fig)

    print("\nTop LazyPredict models:")
    print(summary.head(10).to_string(index=False))
    if not summary["complete_across_folds"].all():
        print(
            "\nSome models failed in one or more folds. "
            "Check the 'folds' and 'complete_across_folds' columns before comparing them."
        )
    print("\nThe locked test set was not loaded or evaluated.")

    write_manifest(
        "06_lazypredict",
        {
            "screening": "fold-wise LazyRegressor with scaffold GroupKFold",
            "n_splits": n_splits,
            "features": int(X.shape[1]),
            "test_used": False,
            "top3": summary["model"].head(3).tolist(),
            "prediction_output_enabled": True,
        },
        [feature_path, target_path, groups_path],
        [
            output_dir / "tables/lazypredict_results.csv",
            output_dir / "tables/recommended_top3.csv",
        ],
    )

    return summary


if __name__ == "__main__":
    run()