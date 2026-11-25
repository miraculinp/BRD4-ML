"""
Training-only Y-randomization for the three BRD4 QSAR regression models,
anchored to the Stage 07 Optuna results.

Models:
    ExtraTreesRegressor
    CatBoostRegressor
    SVR

What this runner reuses from Stage 07
-------------------------------------
1. Observed-label metrics
   The "observed" side of the comparison is NOT recomputed. It is read from
   outputs/07_optuna_tuning/tables/optuna_best_params.json, i.e. the pooled
   out-of-fold (OOF) RMSE / R2 / MAE and the mean-fold RMSE that Stage 07
   stored for each model's best Optuna trial. Because of this, the observed
   number here is identical to the one in the Stage 07 tables.

2. Exact cross-validation folds
   Every shuffled-target run is evaluated on the very same five scaffold-grouped
   folds that Stage 07 used (prepared_folds_<run_id>.joblib in
   07_optuna_tuning/models/fold_feature_cache). No new GroupKFold seed is
   drawn. If that cache is missing, the fold map written next to it
   (cv_fold_assignments.csv) is used instead.

3. Estimators, parameters and seeds
   make_estimator() from Stage 07, the Stage 07 best parameters, the same
   fixed ordered 250-feature matrix, and the same per-fold estimator seed
   rule (seed + fold number).

The only thing that changes between the observed run and a permutation run is
the target vector: the training pIC50 values are shuffled across molecules
(one permutation per run, seed = seed + 9001 + permutation number) and then
split by the unchanged fold map. The locked test set is never loaded.

Run one model at a time from the notebook:
    summary, permutations = run_model("ExtraTreesRegressor", 100)
    summary, permutations = run_model("CatBoostRegressor", 100)
    summary, permutations = run_model("SVR", 100)

Completed permutation runs are checkpointed and safely reused on rerun. A
fingerprint of the training data, parameters, Stage 07 observed metrics and
fold assignment prevents stale checkpoints from being mixed with a newly
tuned model.

Interpretation note: the Stage 07 observed metrics belong to the best of
~30 Optuna trials, so they can be slightly optimistic after parameter
selection. The permutation runs use that fixed best parameter set without any
selection step. This makes the empirical p-value mildly favourable to the real
model; the locked test set remains the unbiased performance estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from src.optuna_tuning import THREADS, _load_training_data, make_estimator
from src.utils import OUTPUTS, load_config


MODEL_NAMES = (
    "ExtraTreesRegressor",
    "CatBoostRegressor",
    "SVR",
)

DEFAULT_N_RANDOMIZATIONS = 100
N_FOLDS = 5
PERMUTATION_SEED_OFFSET = 9001
VERIFY_TOLERANCE_RMSE = 1e-3

STAGE07_DIR = OUTPUTS / "07_optuna_tuning"
STAGE07_PARAMS_PATH = STAGE07_DIR / "tables" / "optuna_best_params.json"
STAGE07_FOLD_DIR = STAGE07_DIR / "models" / "fold_feature_cache"
STAGE07_ASSIGNMENT_PATH = STAGE07_FOLD_DIR / "cv_fold_assignments.csv"

# A new output root is used on purpose: results from the earlier version of this
# runner (which drew its own CV seed) must not be mixed with Stage 07-anchored runs.
OUTPUT_ROOT = OUTPUTS / "16_y_randomization_stage07_folds"

# Observed-label metrics that must be present in the Stage 07 JSON.
REQUIRED_OBSERVED_KEYS = (
    "cv_pooled_oof_rmse",
    "cv_pooled_oof_mse",
    "cv_pooled_oof_mae",
    "cv_pooled_oof_r2",
    "cv_mean_fold_rmse",
    "cv_sd_fold_rmse",
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _json_default(value: Any) -> Any:
    """Make NumPy/scalar-like values JSON serializable."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _atomic_json(path: Path, payload: dict) -> None:
    """Write JSON through a temporary file so interrupted runs leave no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
    )
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Stage 07 observed-label metrics (read, never recomputed)
# --------------------------------------------------------------------------- #
def _load_stage07(model_name: str) -> tuple[dict, dict, dict, Path]:
    """
    Return (best_params, observed_metrics, stage07_info, json_path) for a model.

    The observed metrics are the best-trial CV metrics that Stage 07 saved.
    """
    if not STAGE07_PARAMS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing Stage 07 results: {STAGE07_PARAMS_PATH}. "
            "Run the Stage 07 tuning workflow first."
        )

    tuning = json.loads(STAGE07_PARAMS_PATH.read_text(encoding="utf-8"))
    by_model = tuning.get("params_by_model", {})
    if model_name not in by_model:
        raise ValueError(
            f"No Stage 07 record for {model_name}. "
            f"Models present in params_by_model: {list(by_model)}"
        )
    record = by_model[model_name]

    params = record.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError(f"Stage 07 parameters for {model_name} are empty or invalid.")

    missing = [k for k in REQUIRED_OBSERVED_KEYS if k not in record]
    if missing:
        raise ValueError(
            f"Stage 07 record for {model_name} lacks {missing}. "
            "Re-run Stage 07 with the current tuning script so the pooled OOF "
            "metrics are saved."
        )
    observed = {k: float(record[k]) for k in REQUIRED_OBSERVED_KEYS}
    bad = [k for k, v in observed.items() if not np.isfinite(v)]
    if bad:
        raise ValueError(f"Non-finite Stage 07 metrics for {model_name}: {bad}")

    run_id = tuning.get("run_id")
    if not run_id:
        raise ValueError("Stage 07 JSON has no run_id; cannot locate the fold cache.")

    info = {
        "stage07_run_id": str(run_id),
        "stage07_trial_number": int(record.get("trial_number", -1)),
        "stage07_study_name": str(record.get("study_name", "")),
        "stage07_objective": str(tuning.get("objective", "")),
    }
    return dict(params), observed, info, STAGE07_PARAMS_PATH


def stage07_reference_table() -> pd.DataFrame:
    """One row per model with the Stage 07 observed metrics used as the baseline."""
    rows = []
    for model_name in MODEL_NAMES:
        params, observed, info, _ = _load_stage07(model_name)
        rows.append({"model": model_name, **info, **observed, "best_params": json.dumps(
            params, sort_keys=True, default=_json_default)})
    table = pd.DataFrame(rows)
    run_id = table["stage07_run_id"].iloc[0]
    cache = STAGE07_FOLD_DIR / f"prepared_folds_{run_id}.joblib"
    table.attrs["fold_cache"] = str(cache)
    table.attrs["fold_cache_exists"] = cache.is_file()
    table.attrs["fold_assignment_csv_exists"] = STAGE07_ASSIGNMENT_PATH.is_file()
    return table


# --------------------------------------------------------------------------- #
# Stage 07 folds
# --------------------------------------------------------------------------- #
def _load_stage07_folds(run_id: str, X: pd.DataFrame, y: np.ndarray,
                        groups: np.ndarray) -> tuple[list[dict], str, str]:
    """
    Rebuild the exact Stage 07 folds as integer positions into X.

    Preferred source: prepared_folds_<run_id>.joblib (the objects Stage 07 itself
    evaluated). Fallback: cv_fold_assignments.csv. Either way the partition is
    validated against the current training matrix.
    """
    cache_path = STAGE07_FOLD_DIR / f"prepared_folds_{run_id}.joblib"
    cached = None

    if cache_path.is_file():
        import joblib

        cached = sorted(joblib.load(cache_path), key=lambda f: int(f["fold"]))
        valid_ids = {int(f["fold"]): [str(i) for i in f["valid_ids"]] for f in cached}
        source = f"cache: {cache_path}"
    elif STAGE07_ASSIGNMENT_PATH.is_file():
        assignment = pd.read_csv(STAGE07_ASSIGNMENT_PATH, dtype={"chembl_id": str})
        valid_ids = {
            int(fold): frame["chembl_id"].astype(str).tolist()
            for fold, frame in assignment.groupby("fold")
        }
        source = f"assignment table: {STAGE07_ASSIGNMENT_PATH}"
        print(
            "[warning] Stage 07 fold cache not found; using cv_fold_assignments.csv. "
            "This file is not keyed by run_id, so the partition is validated below."
        )
    else:
        raise FileNotFoundError(
            f"No Stage 07 folds found. Expected {cache_path} or {STAGE07_ASSIGNMENT_PATH}. "
            "Run Stage 07 first."
        )

    if sorted(valid_ids) != list(range(1, N_FOLDS + 1)):
        raise ValueError(f"Expected folds 1..{N_FOLDS}; found {sorted(valid_ids)}.")

    index = pd.Index(X.index.astype(str))
    folds: list[dict] = []
    fold_of = np.full(len(X), -1, dtype=int)

    for fold_no in range(1, N_FOLDS + 1):
        valid_pos = index.get_indexer(valid_ids[fold_no])
        if (valid_pos < 0).any():
            raise ValueError(
                f"Fold {fold_no} contains molecules absent from the current training "
                "matrix; Stage 07 folds do not match the current data."
            )
        if (fold_of[valid_pos] != -1).any():
            raise ValueError(f"A molecule appears in more than one validation fold (fold {fold_no}).")
        fold_of[valid_pos] = fold_no
        mask = np.ones(len(X), dtype=bool)
        mask[valid_pos] = False
        folds.append({
            "fold": fold_no,
            "train_pos": np.flatnonzero(mask),
            "valid_pos": valid_pos,
        })

    if (fold_of == -1).any():
        raise ValueError("Some training molecules are not in any Stage 07 validation fold.")

    # Scaffold-grouped means no scaffold group may sit on both sides of a split.
    for fold in folds:
        overlap = set(groups[fold["train_pos"]]) & set(groups[fold["valid_pos"]])
        if overlap:
            raise ValueError(
                f"Fold {fold['fold']} shares {len(overlap)} scaffold groups between "
                "training and validation; these are not grouped Stage 07 folds."
            )

    # If the cached Stage 07 objects are available, confirm they hold the same data.
    if cached is not None:
        for fold, c in zip(folds, cached):
            ok = (
                c["X_valid"].index.astype(str).tolist() == index[fold["valid_pos"]].tolist()
                and c["X_train"].index.astype(str).tolist() == index[fold["train_pos"]].tolist()
                and np.array_equal(c["X_valid"].to_numpy(), X.iloc[fold["valid_pos"]].to_numpy())
                and np.array_equal(c["X_train"].to_numpy(), X.iloc[fold["train_pos"]].to_numpy())
                and np.array_equal(
                    c["y_valid"].to_numpy(dtype=float), y[fold["valid_pos"]]
                )
                and np.array_equal(
                    c["y_train"].to_numpy(dtype=float), y[fold["train_pos"]]
                )
            )
            if not ok:
                raise ValueError(
                    f"Cached Stage 07 fold {fold['fold']} does not match the current "
                    "training matrix/target. Re-run Stage 07 or check the inputs."
                )

    fold_hash = hashlib.sha256(fold_of.astype(np.int64).tobytes()).hexdigest()[:16]
    return folds, source, fold_hash


# --------------------------------------------------------------------------- #
# Fold evaluation
# --------------------------------------------------------------------------- #
def _evaluate_target(model_name: str, best_params: dict, target: np.ndarray,
                     X: pd.DataFrame, folds: list[dict], seed: int) -> dict:
    """Fit the tuned estimator on each Stage 07 fold and pool the OOF predictions."""
    oof_observed: list[float] = []
    oof_predicted: list[float] = []
    fold_rmses: list[float] = []

    for fold in folds:
        model = make_estimator(model_name, best_params, int(seed) + int(fold["fold"]))
        X_train = X.iloc[fold["train_pos"]]
        X_valid = X.iloc[fold["valid_pos"]]
        y_train = target[fold["train_pos"]]
        observed = target[fold["valid_pos"]]

        with threadpool_limits(limits=THREADS):
            model.fit(X_train, y_train)
            prediction = np.asarray(model.predict(X_valid), dtype=float)

        if not np.isfinite(prediction).all():
            raise ValueError(
                f"{model_name} returned non-finite predictions in fold {fold['fold']}."
            )
        fold_rmses.append(float(np.sqrt(mean_squared_error(observed, prediction))))
        oof_observed.extend(observed.tolist())
        oof_predicted.extend(prediction.tolist())

    pooled_mse = float(mean_squared_error(oof_observed, oof_predicted))
    return {
        "cv_mean_fold_rmse": float(np.mean(fold_rmses)),
        "cv_pooled_oof_rmse": float(np.sqrt(pooled_mse)),
        "cv_pooled_oof_mse": pooled_mse,
        "cv_pooled_oof_mae": float(mean_absolute_error(oof_observed, oof_predicted)),
        "cv_pooled_oof_r2": float(r2_score(oof_observed, oof_predicted)),
    }


def _data_fingerprint(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                      model_name: str, best_params: dict, observed: dict,
                      info: dict, seed: int, fold_hash: str) -> str:
    """Fingerprint the data and every setting that affects the results."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    digest.update(np.asarray(y, dtype=np.float64).tobytes())
    digest.update(pd.util.hash_pandas_object(
        pd.Series(np.asarray(groups).astype(str)), index=False
    ).values.tobytes())
    settings = {
        "model": model_name,
        "best_params": best_params,
        "observed_stage07": observed,
        "stage07": info,
        "seed": int(seed),
        "permutation_seed_offset": PERMUTATION_SEED_OFFSET,
        "fold_hash": fold_hash,
        "n_folds": N_FOLDS,
        "evaluation": "Stage 07 folds + make_estimator; observed metrics read from Stage 07",
    }
    digest.update(json.dumps(settings, sort_keys=True, default=_json_default).encode("utf-8"))
    return digest.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def _checkpoint_file(model_dir: Path, iteration: int) -> Path:
    return model_dir / "checkpoints" / f"permutation_{iteration:04d}.json"


def _read_checkpoint(path: Path, expected_model: str, expected_iteration: int) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("model") != expected_model or int(record.get("iteration", -1)) != expected_iteration:
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    for key in ("pooled_oof_rmse", "pooled_oof_r2", "mean_fold_rmse"):
        if not np.isfinite(float(record.get(key, np.nan))):
            raise ValueError(f"Checkpoint has invalid {key}: {path}")
    return record


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_model(model_name: str,
              n_randomizations: int = DEFAULT_N_RANDOMIZATIONS,
              verify_observed: bool = False):
    """
    Run Y-randomization for one model against its Stage 07 observed metrics.

    Parameters
    ----------
    model_name : one of MODEL_NAMES
    n_randomizations : number of shuffled-target runs (>= 2)
    verify_observed : if True, also refit the real labels once on the Stage 07
        folds and report the difference from the saved Stage 07 metrics. This is
        a reproducibility check only; the saved Stage 07 numbers remain the
        observed values used for the p-value.

    Returns
    -------
    summary : dict
    permutation_df : pandas.DataFrame, one row per permutation
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"model_name must be one of {MODEL_NAMES}; got {model_name!r}")
    n_randomizations = int(n_randomizations)
    if n_randomizations < 2:
        raise ValueError("Use at least 2 randomizations to estimate a null SD.")

    cfg = load_config()
    seed = int(cfg.get("seed", 42))

    best_params, observed, info, tuning_path = _load_stage07(model_name)
    X, y, groups, feature_path, target_path, selected_path, train_ids_path = _load_training_data()

    X = X.copy()
    y = np.asarray(y, dtype=float).reshape(-1)
    groups = np.asarray(groups).astype(str).reshape(-1)

    if len(X) != len(y) or len(y) != len(groups):
        raise ValueError("Training features, target, and scaffold groups are not aligned.")
    if not np.isfinite(y).all():
        raise ValueError("Training target contains missing or infinite values.")
    if not X.index.is_unique:
        raise ValueError("Training feature index contains duplicate molecule IDs.")
    if len(np.unique(groups)) < N_FOLDS:
        raise ValueError(f"At least {N_FOLDS} unique scaffold groups are required.")

    folds, fold_source, fold_hash = _load_stage07_folds(
        info["stage07_run_id"], X, y, groups
    )

    real_rmse = observed["cv_pooled_oof_rmse"]
    real_r2 = observed["cv_pooled_oof_r2"]
    real_mean_fold_rmse = observed["cv_mean_fold_rmse"]

    fingerprint = _data_fingerprint(
        X, y, groups, model_name, best_params, observed, info, seed, fold_hash
    )
    model_dir = OUTPUT_ROOT / model_name / fingerprint
    checkpoint_dir = model_dir / "checkpoints"
    figure_dir = model_dir / "figures"
    table_dir = model_dir / "tables"
    for folder in (checkpoint_dir, figure_dir, table_dir):
        folder.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model": model_name,
        "fingerprint": fingerprint,
        "n_training_molecules": int(len(y)),
        "n_candidate_features": int(X.shape[1]),
        "n_scaffold_groups": int(len(np.unique(groups))),
        "n_folds": N_FOLDS,
        "cv": "Stage 07 scaffold-grouped folds reused exactly (no new CV seed)",
        "fold_source": fold_source,
        "fold_assignment_hash": fold_hash,
        "estimator_seed_rule": "seed + fold number (same as Stage 07)",
        "seed": seed,
        "permutation_seed_rule": f"seed + {PERMUTATION_SEED_OFFSET} + permutation_number",
        "n_randomizations_requested": n_randomizations,
        "best_params": best_params,
        "observed_metrics_source": str(tuning_path),
        "observed_metrics": observed,
        **info,
        "training_feature_source": str(feature_path),
        "training_target_source": str(target_path),
        "selected_feature_list_source": str(selected_path),
        "training_id_source": str(train_ids_path),
        "test_data_loaded": False,
    }
    _atomic_json(model_dir / "run_metadata.json", metadata)

    print(
        f"[{model_name}] Observed-label baseline read from Stage 07 "
        f"(trial #{info['stage07_trial_number']}, {info['stage07_study_name']}) | "
        f"pooled OOF RMSE={real_rmse:.5f} | R²={real_r2:.5f}"
    )
    print(f"[{model_name}] Folds: {fold_source}")

    _atomic_json(model_dir / "observed_label_baseline.json", {
        **observed, **info, "source": str(tuning_path), "recomputed": False,
    })

    verification = None
    if verify_observed:
        verif_path = model_dir / "observed_verification.json"
        if verif_path.is_file():
            verification = json.loads(verif_path.read_text(encoding="utf-8"))
        else:
            print(f"[{model_name}] Verifying: refitting real labels on the Stage 07 folds...")
            refit = _evaluate_target(model_name, best_params, y, X, folds, seed)
            verification = {
                "refit": refit,
                "delta_pooled_rmse": refit["cv_pooled_oof_rmse"] - real_rmse,
                "delta_pooled_r2": refit["cv_pooled_oof_r2"] - real_r2,
                "delta_mean_fold_rmse": refit["cv_mean_fold_rmse"] - real_mean_fold_rmse,
            }
            _atomic_json(verif_path, verification)
        print(
            f"[{model_name}] Verification | ΔRMSE(pooled)={verification['delta_pooled_rmse']:+.2e} "
            f"| ΔR²={verification['delta_pooled_r2']:+.2e}"
        )
        if abs(verification["delta_pooled_rmse"]) > VERIFY_TOLERANCE_RMSE:
            print(
                f"[warning] {model_name}: refit differs from Stage 07 by more than "
                f"{VERIFY_TOLERANCE_RMSE}. Check that parameters, seed and data match Stage 07."
            )

    records = []
    for iteration in range(1, n_randomizations + 1):
        cp = _checkpoint_file(model_dir, iteration)
        if cp.is_file():
            records.append(_read_checkpoint(cp, model_name, iteration))
    existing = {int(r["iteration"]) for r in records}

    progress = tqdm(
        total=n_randomizations,
        initial=len(existing),
        desc=f"{model_name} Y-randomization",
        unit="perm",
        dynamic_ncols=True,
        smoothing=0.08,
    )
    try:
        for iteration in range(1, n_randomizations + 1):
            if iteration in existing:
                continue

            permutation_seed = seed + PERMUTATION_SEED_OFFSET + iteration
            rng = np.random.default_rng(permutation_seed)
            y_shuffled = rng.permutation(y)

            result = _evaluate_target(model_name, best_params, y_shuffled, X, folds, seed)
            row = {
                "model": model_name,
                "iteration": iteration,
                "permutation_seed": permutation_seed,
                "pooled_oof_rmse": float(result["cv_pooled_oof_rmse"]),
                "pooled_oof_r2": float(result["cv_pooled_oof_r2"]),
                "mean_fold_rmse": float(result["cv_mean_fold_rmse"]),
            }
            _atomic_json(_checkpoint_file(model_dir, iteration), row)
            records.append(row)

            current = pd.DataFrame(records)
            n_le = int((current["pooled_oof_rmse"] <= real_rmse).sum())
            progress.set_postfix({
                "RMSE": f"{row['pooled_oof_rmse']:.4f}",
                "null_mean": f"{current['pooled_oof_rmse'].mean():.4f}",
                "p_now": f"{(1 + n_le) / (len(current) + 1):.4f}",
            })
            progress.update(1)
    finally:
        progress.close()

    permutation_df = pd.DataFrame(records).sort_values("iteration").reset_index(drop=True)
    if len(permutation_df) != n_randomizations:
        raise RuntimeError(
            f"{model_name}: expected {n_randomizations} permutations, "
            f"but found {len(permutation_df)}."
        )
    permutation_df.to_csv(table_dir / f"{model_name}_permutation_metrics.csv", index=False)

    null_rmse = permutation_df["pooled_oof_rmse"].to_numpy(dtype=float)
    null_r2 = permutation_df["pooled_oof_r2"].to_numpy(dtype=float)
    null_fold_rmse = permutation_df["mean_fold_rmse"].to_numpy(dtype=float)
    n_le = int(np.sum(null_rmse <= real_rmse))
    n_le_fold = int(np.sum(null_fold_rmse <= real_mean_fold_rmse))
    empirical_p = float((1 + n_le) / (n_randomizations + 1))
    empirical_p_fold = float((1 + n_le_fold) / (n_randomizations + 1))

    summary = {
        "model": model_name,
        "fingerprint": fingerprint,
        "n_randomizations": n_randomizations,
        "n_folds": N_FOLDS,
        "n_training_molecules": int(len(y)),
        "n_scaffold_groups": int(len(np.unique(groups))),
        "real_pooled_oof_rmse": real_rmse,
        "real_pooled_oof_r2": real_r2,
        "real_pooled_oof_mae": observed["cv_pooled_oof_mae"],
        "real_mean_fold_rmse": real_mean_fold_rmse,
        "real_sd_fold_rmse": observed["cv_sd_fold_rmse"],
        "randomized_mean_rmse": float(np.mean(null_rmse)),
        "randomized_sd_rmse": float(np.std(null_rmse, ddof=1)),
        "randomized_min_rmse": float(np.min(null_rmse)),
        "randomized_max_rmse": float(np.max(null_rmse)),
        "randomized_mean_r2": float(np.mean(null_r2)),
        "randomized_sd_r2": float(np.std(null_r2, ddof=1)),
        "randomized_mean_fold_rmse": float(np.mean(null_fold_rmse)),
        "n_randomized_rmse_le_real": n_le,
        "empirical_p_value_lower_rmse": empirical_p,
        "empirical_p_value_lower_mean_fold_rmse": empirical_p_fold,
        "minimum_attainable_p_value": float(1 / (n_randomizations + 1)),
        "observed_metrics_source": str(tuning_path),
        "observed_metrics_recomputed": False,
        "observed_verification_delta_pooled_rmse": (
            None if verification is None else verification["delta_pooled_rmse"]
        ),
        "stage07_trial_number": info["stage07_trial_number"],
        "stage07_study_name": info["stage07_study_name"],
        "stage07_run_id": info["stage07_run_id"],
        "fold_source": fold_source,
        "fold_assignment_hash": fold_hash,
        "best_params": json.dumps(best_params, sort_keys=True, default=_json_default),
        "feature_selection_and_preprocessing": (
            "Fixed ordered Stage 05 selected 250-feature matrix reused unchanged "
            "in every fold and permutation; no fold-local feature selection or imputation"
        ),
        "test_data_loaded": False,
    }
    pd.DataFrame([summary]).to_csv(
        table_dir / f"{model_name}_y_randomization_summary.csv", index=False
    )
    _atomic_json(model_dir / "run_summary.json", summary)

    # Diagnostic figure: null distributions with the Stage 07 observed values.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].hist(null_rmse, bins="auto", color="#3977A8", edgecolor="white", alpha=0.9)
    axes[0].axvline(real_rmse, color="#C43C39", linestyle="--", linewidth=2,
                    label=f"Stage 07 observed RMSE = {real_rmse:.3f}")
    axes[0].set(xlabel="Pooled out-of-fold RMSE", ylabel="Permutation count",
                title="Shuffled-target RMSE distribution")
    axes[0].legend(frameon=False)

    axes[1].hist(null_r2, bins="auto", color="#4C9475", edgecolor="white", alpha=0.9)
    axes[1].axvline(real_r2, color="#C43C39", linestyle="--", linewidth=2,
                    label=f"Stage 07 observed R² = {real_r2:.3f}")
    axes[1].set(xlabel="Pooled out-of-fold R²", ylabel="Permutation count",
                title="Shuffled-target R² distribution")
    axes[1].legend(frameon=False)

    fig.suptitle(
        f"Y-randomization — {model_name} ({n_randomizations} permutations, Stage 07 folds)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(figure_dir / f"{model_name}_y_randomization.png", dpi=400, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("\n" + "=" * 76)
    print(f"Y-RANDOMIZATION COMPLETE: {model_name}")
    print(f"Observed pooled OOF RMSE (Stage 07): {real_rmse:.5f}")
    print(f"Randomized RMSE mean ± SD          : {np.mean(null_rmse):.5f} ± {np.std(null_rmse, ddof=1):.5f}")
    print(f"Observed pooled OOF R² (Stage 07)  : {real_r2:.5f}")
    print(f"Randomized R² mean ± SD            : {np.mean(null_r2):.5f} ± {np.std(null_r2, ddof=1):.5f}")
    print(f"Empirical lower-tail p (pooled RMSE): {empirical_p:.5f}")
    print(f"Minimum attainable p               : {1 / (n_randomizations + 1):.5f}")
    print(f"Results directory                  : {model_dir}")
    print("=" * 76)

    return summary, permutation_df


def combine_completed_summaries():
    """Collect the latest completed run for each of the three models."""
    rows = []
    for model_name in MODEL_NAMES:
        model_root = OUTPUT_ROOT / model_name
        candidates = sorted(
            model_root.glob("*/run_summary.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if model_root.exists() else []
        if not candidates:
            raise FileNotFoundError(
                f"No completed Y-randomization summary found for {model_name}. "
                "Run that model's notebook cell first."
            )
        rows.append(json.loads(candidates[0].read_text(encoding="utf-8")))

    combined = pd.DataFrame(rows)
    summary_dir = OUTPUT_ROOT / "combined"
    summary_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(summary_dir / "three_model_y_randomization_summary.csv", index=False)
    print("Saved combined three-model summary:",
          summary_dir / "three_model_y_randomization_summary.csv")
    return combined


def main():
    parser = argparse.ArgumentParser(description="BRD4 three-model Y-randomization (Stage 07 folds)")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--n-randomizations", type=int, default=DEFAULT_N_RANDOMIZATIONS)
    parser.add_argument("--verify-observed", action="store_true",
                        help="Refit real labels once on the Stage 07 folds and compare to Stage 07.")
    args = parser.parse_args()
    run_model(args.model, args.n_randomizations, verify_observed=args.verify_observed)


if __name__ == "__main__":
    main()
