"""Stage 08: final full-training fit and one-shot locked-test evaluation.

Design rules
------------
* No cross-validation is run here. The five-fold scaffold-grouped CV metrics of
  the best Optuna trial are read from Stage 07 (``optuna_best_params.json``).
* The training matrix is loaded with the *same* function Stage 07 used, so the
  final fit sees exactly the rows, row order and ordered 250-feature list that
  the tuner saw.
* Every Optuna best parameter is applied and then audited against the fitted
  estimator; the full resolved parameter set (tuned + fixed + library default)
  is written to disk.
* The primary model is chosen from Stage 07 CV only. The locked test set is
  never used to choose, tune, or re-rank anything.
* Nothing is imputed.
"""

from __future__ import annotations

import json
import warnings
from itertools import combinations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from threadpoolctl import threadpool_limits

from src.optuna_tuning import (
    FEATURE_BUDGET,
    THREADS,
    TUNED_MODELS,
    _load_training_data,
    make_estimator,
)
from src.utils import (
    OUTPUTS,
    load_config,
    save_figure,
    setup_plotting,
    stage_dir,
    write_manifest,
)

MODELS = TUNED_MODELS
ENSEMBLE = "Equal-weight ensemble (supplementary)"
BASELINES = (
    "Training mean",
    "Morgan Tanimoto 1-NN",
    "Morgan Tanimoto weighted 5-NN",
)
N_BOOTSTRAP = 2000
SIMILARITY_BINS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0001)
LOG10_2 = float(np.log10(2.0))  # two-fold error in log units

CV_KEYS = (
    "cv_mean_fold_rmse", "cv_sd_fold_rmse",
    "cv_mean_fold_mse", "cv_sd_fold_mse",
    "cv_mean_fold_mae", "cv_sd_fold_mae",
    "cv_mean_fold_r2", "cv_sd_fold_r2",
    "cv_pooled_oof_rmse", "cv_pooled_oof_mse",
    "cv_pooled_oof_mae", "cv_pooled_oof_r2",
)

# Keys that make_estimator() fixes for reproducibility (not tuned).
FIXED_BY_PIPELINE = {
    "random_state", "random_seed", "n_jobs", "thread_count",
    "loss_function", "verbose", "allow_writing_files",
}

COLOURS = {
    "ExtraTreesRegressor": "#0072B2",
    "CatBoostRegressor": "#009E73",
    "SVR": "#E69F00",
    ENSEMBLE: "#CC79A7",
    "Training mean": "#999999",
    "Morgan Tanimoto 1-NN": "#999999",
    "Morgan Tanimoto weighted 5-NN": "#999999",
}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _safe_corr(func, a, b):
    """Correlation that returns NaN, not a warning, for constant input."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(func(a, b)[0])


def _ccc(obs, pred):
    """Lin's concordance correlation coefficient (population moments)."""
    cov = np.mean((obs - obs.mean()) * (pred - pred.mean()))
    denom = obs.var() + pred.var() + (obs.mean() - pred.mean()) ** 2
    return float(2.0 * cov / denom) if denom > 0 else float("nan")


def _through_origin(obs, pred):
    """Golbraikh-Tropsha (2002) regression-through-origin quantities."""
    sp2, so2 = float(np.sum(pred ** 2)), float(np.sum(obs ** 2))
    if sp2 == 0 or so2 == 0:
        return {k: float("nan") for k in ("k", "k_prime", "r0_sq", "r0_prime_sq")}
    k = float(np.sum(obs * pred) / sp2)          # obs ~ k * pred
    k_prime = float(np.sum(obs * pred) / so2)    # pred ~ k' * obs
    ss_obs = float(np.sum((obs - obs.mean()) ** 2))
    ss_pred = float(np.sum((pred - pred.mean()) ** 2))
    r0 = 1.0 - float(np.sum((obs - k * pred) ** 2)) / ss_obs if ss_obs else float("nan")
    r0p = 1.0 - float(np.sum((pred - k_prime * obs) ** 2)) / ss_pred if ss_pred else float("nan")
    return {"k": k, "k_prime": k_prime, "r0_sq": r0, "r0_prime_sq": r0p}


def test_metrics(obs, pred, y_train):
    """Complete external-validation metric set for one set of predictions."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n = len(obs)
    resid = obs - pred
    sse = float(np.sum(resid ** 2))
    mse = sse / n
    train_mean = float(np.mean(y_train))
    ss_ref_train = float(np.sum((obs - train_mean) ** 2))
    ss_train_dev = float(np.sum((y_train - train_mean) ** 2))
    pearson = _safe_corr(pearsonr, obs, pred)
    r_sq = pearson ** 2 if np.isfinite(pearson) else float("nan")
    if np.ptp(pred) > 0:
        slope, intercept = np.polyfit(pred, obs, 1)
    else:
        slope, intercept = float("nan"), float("nan")
    origin = _through_origin(obs, pred)

    gt_pass = False
    if np.isfinite(r_sq):
        c1 = r_sq > 0.6
        c2 = any(
            np.isfinite(v) and (r_sq - v) / r_sq < 0.1
            for v in (origin["r0_sq"], origin["r0_prime_sq"])
        )
        c3 = any(
            np.isfinite(v) and 0.85 <= v <= 1.15
            for v in (origin["k"], origin["k_prime"])
        )
        gt_pass = bool(c1 and c2 and c3)

    return {
        "test_n": int(n),
        "test_rmse": float(np.sqrt(mse)),
        "test_mse": float(mse),
        "test_mae": float(mean_absolute_error(obs, pred)),
        "test_median_ae": float(median_absolute_error(obs, pred)),
        "test_max_abs_error": float(np.max(np.abs(resid))),
        "test_r2": float(r2_score(obs, pred)),                      # = Q2_F2
        "test_q2_f1": float(1.0 - sse / ss_ref_train),
        "test_q2_f3": float(1.0 - mse / (ss_train_dev / len(y_train))),
        "test_pearson": pearson,
        "test_r_squared": r_sq,
        "test_spearman": _safe_corr(spearmanr, obs, pred),
        "test_kendall": _safe_corr(kendalltau, obs, pred),
        "test_ccc": _ccc(obs, pred),
        "test_bias_obs_minus_pred": float(np.mean(resid)),
        "test_residual_sd": float(np.std(resid, ddof=1)),
        "test_calibration_slope": float(slope),
        "test_calibration_intercept": float(intercept),
        "test_k": origin["k"],
        "test_k_prime": origin["k_prime"],
        "test_r0_sq": origin["r0_sq"],
        "test_r0_prime_sq": origin["r0_prime_sq"],
        "test_golbraikh_tropsha_pass": gt_pass,
        "test_frac_within_2fold": float(np.mean(np.abs(resid) <= LOG10_2)),
        "test_frac_within_0p5": float(np.mean(np.abs(resid) <= 0.5)),
        "test_frac_within_10fold": float(np.mean(np.abs(resid) <= 1.0)),
    }


def _rank_corr(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _fast_bootstrap_metrics(obs, pred):
    """Cheap per-resample metrics used inside the bootstrap loop."""
    resid = obs - pred
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((obs - obs.mean()) ** 2))
    pear = (
        float(np.corrcoef(obs, pred)[0, 1])
        if obs.std() > 0 and pred.std() > 0 else np.nan
    )
    return (
        np.sqrt(sse / len(obs)),
        float(np.mean(np.abs(resid))),
        1.0 - sse / sst if sst > 0 else np.nan,
        pear,
        _rank_corr(obs, pred),
        _ccc(obs, pred),
    )


BOOT_METRICS = ("rmse", "mae", "r2", "pearson", "spearman", "ccc")


def _group_bootstrap_indices(groups, n_boot, seed):
    """Resample whole scaffold groups: molecules of one scaffold stay together."""
    codes, uniques = pd.factorize(pd.Series(groups).astype(str))
    members = [np.flatnonzero(codes == i) for i in range(len(uniques))]
    rng = np.random.default_rng(seed)
    return [
        np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        for _ in range(n_boot)
    ]


def _bootstrap(prediction_by_model, observed, boot_indices):
    """Scaffold-cluster bootstrap CIs and paired RMSE differences."""
    samples = {m: np.empty((len(boot_indices), len(BOOT_METRICS))) for m in prediction_by_model}
    for b, idx in enumerate(boot_indices):
        o = observed[idx]
        for m, p in prediction_by_model.items():
            samples[m][b] = _fast_bootstrap_metrics(o, p[idx])

    ci_rows = []
    for m, arr in samples.items():
        row = {"model": m}
        for j, name in enumerate(BOOT_METRICS):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # constant baseline -> all-NaN correlation
                lo, hi = np.nanpercentile(arr[:, j], [2.5, 97.5])
            row[f"test_{name}_ci_low"] = float(lo)
            row[f"test_{name}_ci_high"] = float(hi)
        ci_rows.append(row)

    paired = []
    for a, b in combinations(prediction_by_model, 2):
        delta = samples[a][:, 0] - samples[b][:, 0]  # RMSE_a - RMSE_b
        point = float(
            np.sqrt(np.mean((observed - prediction_by_model[a]) ** 2))
            - np.sqrt(np.mean((observed - prediction_by_model[b]) ** 2))
        )
        lo, hi = np.percentile(delta, [2.5, 97.5])
        paired.append(
            {
                "model_a": a,
                "model_b": b,
                "rmse_a_minus_rmse_b": point,
                "ci_low": float(lo),
                "ci_high": float(hi),
                "prob_a_lower_rmse": float(np.mean(delta < 0)),
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
            }
        )
    return pd.DataFrame(ci_rows), pd.DataFrame(paired)


# --------------------------------------------------------------------------
# Parameters: apply every Optuna parameter and audit the fitted estimator
# --------------------------------------------------------------------------
def _expected(model_name, key, value):
    # Stage 07 encodes "unlimited depth" as 0 because Optuna categoricals
    # cannot contain None; make_estimator converts it back.
    if model_name == "ExtraTreesRegressor" and key == "max_depth" and value == 0:
        return None
    return value


def _same(expected, actual):
    if expected is None or actual is None:
        return expected is actual
    if isinstance(expected, (bool, str)) or isinstance(actual, (bool, str)):
        return expected == actual
    try:
        return bool(np.isclose(float(expected), float(actual), rtol=1e-6, atol=0.0))
    except (TypeError, ValueError):
        return expected == actual


def _resolved_parameters(model_name, fitted):
    """Every parameter of the fitted estimator (defaults included)."""
    if model_name == "SVR":
        return dict(fitted.named_steps["svr"].get_params())
    if model_name == "CatBoostRegressor":
        merged = dict(fitted.get_params())
        merged.update(fitted.get_all_params())
        return merged
    return dict(fitted.get_params())


def _audit_parameters(model_name, tuned, fitted):
    """Raise unless every tuned parameter reached the fitted estimator."""
    resolved = _resolved_parameters(model_name, fitted)
    problems = []
    for key, value in tuned.items():
        want = _expected(model_name, key, value)
        if key not in resolved:
            problems.append(f"{key}: not found on fitted estimator")
        elif not _same(want, resolved[key]):
            problems.append(f"{key}: expected {want!r}, fitted {resolved[key]!r}")
    if problems:
        raise RuntimeError(
            f"Optuna parameters were not applied to {model_name}: " + "; ".join(problems)
        )
    rows = []
    for key in sorted(resolved):
        if key in tuned:
            origin = "Optuna best parameter"
        elif key in FIXED_BY_PIPELINE:
            origin = "Fixed by pipeline"
        else:
            origin = "Library default"
        rows.append(
            {
                "model": model_name,
                "parameter": key,
                "value": json.dumps(resolved[key], default=str),
                "origin": origin,
            }
        )
    return rows, resolved


# --------------------------------------------------------------------------
# Data: locked test set and similarity-based baselines
# --------------------------------------------------------------------------
def _load_locked_test(selected, cfg, on_invalid):
    tables03 = OUTPUTS / "03_feature_generation/tables"
    test_ids = pd.read_csv(OUTPUTS / "04_train_test_split/tables/test_ids.csv", dtype={"chembl_id": str})
    if test_ids["chembl_id"].duplicated().any():
        raise ValueError("The locked test ID file contains duplicate molecule IDs.")
    if test_ids["group"].isna().any():
        raise ValueError("Missing scaffold group labels in the locked test IDs.")

    features = pd.read_parquet(tables03 / "all_features.parquet")
    target = pd.read_parquet(tables03 / "target.parquet")["pIC50"]
    features.index = features.index.astype(str)
    target.index = target.index.astype(str)

    absent_features = sorted(set(selected) - set(features.columns))
    if absent_features:
        raise ValueError(f"Selected features missing from the feature table: {absent_features[:10]}")
    absent_ids = sorted(set(test_ids.chembl_id) - set(features.index))
    if absent_ids:
        raise ValueError(f"{len(absent_ids)} test IDs are absent from the feature table.")

    X = features.reindex(test_ids.chembl_id).loc[:, selected].apply(pd.to_numeric, errors="coerce")
    y = target.reindex(test_ids.chembl_id)
    bad_rows = ~np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    bad_target = ~np.isfinite(y.to_numpy(dtype=float))

    audit = pd.DataFrame(columns=["chembl_id", "reason"])
    if bad_target.any():
        raise ValueError("Locked test target contains missing or infinite pIC50 values.")
    if bad_rows.any():
        bad_cols = X.columns[~np.isfinite(X.to_numpy(dtype=float)).all(axis=0)].tolist()
        audit = pd.DataFrame(
            {
                "chembl_id": X.index[bad_rows],
                "reason": "missing_or_nonfinite_selected_feature",
            }
        )
        if on_invalid == "error":
            raise ValueError(
                f"{int(bad_rows.sum())} locked-test molecules have missing/infinite selected "
                f"features in {len(bad_cols)} feature(s) (e.g. {bad_cols[:5]}). This workflow "
                "does not impute. Re-run with on_invalid_test='exclude' to score the remaining "
                "molecules and record the exclusions."
            )
        keep = ~bad_rows
        test_ids, X, y = test_ids.loc[keep].reset_index(drop=True), X.loc[keep], y.loc[keep]
    return test_ids, X, y, audit


def _similarity_matrix(smiles_train, smiles_test, cfg):
    """Tanimoto similarity (test x train) with the Stage 03 fingerprint settings."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(cfg.get("fingerprint_radius", 2)),
        fpSize=int(cfg.get("fingerprint_bits", 2048)),
        includeChirality=bool(cfg.get("use_chirality", True)),
    )

    def fp(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid cleaned SMILES; cannot compute similarity: {smiles}")
        return generator.GetFingerprint(mol)

    train_fps = [fp(s) for s in smiles_train]
    return np.vstack(
        [np.asarray(DataStructs.BulkTanimotoSimilarity(fp(s), train_fps), dtype=float) for s in smiles_test]
    )


def _smiles_lookup(train_ids, test_ids):
    path = OUTPUTS / "02_clean_data/tables/cleaned_activity.csv"
    activity = pd.read_csv(path, dtype={"chembl_id": str})
    column = "canonical_smiles" if "canonical_smiles" in activity.columns else next(
        (c for c in activity.columns if "smiles" in c.lower()), None
    )
    if column is None:
        raise ValueError(f"No SMILES column in {path}.")
    activity["chembl_id"] = activity["chembl_id"].astype(str)
    if activity["chembl_id"].duplicated().any():
        raise ValueError("Cleaned activity table has duplicate molecule IDs.")
    lookup = activity.set_index("chembl_id")[column].astype(str)
    missing = [i for i in pd.Index(train_ids).union(pd.Index(test_ids)) if i not in lookup.index]
    if missing:
        raise ValueError(f"Missing cleaned SMILES for {len(missing)} train/test molecules.")
    return lookup, path


def _baseline_predictions(similarity, y_train):
    """Training mean, 1-NN and similarity-weighted 5-NN from a similarity matrix."""
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) < 5:
        raise ValueError("At least five training molecules are required for the 5-NN baseline.")
    n_test = similarity.shape[0]
    nearest = np.empty(n_test)
    weighted = np.empty(n_test)
    for row in range(n_test):
        # Default argsort keeps the tie-handling convention of the earlier baseline analysis.
        order = np.argsort(-similarity[row])
        top5 = order[:5]
        nearest[row] = y_train[order[0]]
        weighted[row] = np.average(y_train[top5], weights=np.clip(similarity[row, top5], 1e-6, None))
    return {
        BASELINES[0]: np.full(n_test, float(y_train.mean())),
        BASELINES[1]: nearest,
        BASELINES[2]: weighted,
    }


_BASELINE_PARAMS = {
    BASELINES[0]: {"statistic": "mean of training pIC50"},
    BASELINES[1]: {"neighbors": 1, "weighting": "none"},
    BASELINES[2]: {"neighbors": 5, "weighting": "Tanimoto similarity"},
}


# --------------------------------------------------------------------------
# Tuning-file validation
# --------------------------------------------------------------------------
def _validate_tuning(tuning, selected):
    if int(tuning.get("feature_budget", -1)) != FEATURE_BUDGET:
        raise ValueError(f"Stage 07 used a feature budget other than {FEATURE_BUDGET}; rerun Stage 07.")
    if tuning.get("extra_best_parameter_cv_refit", True):
        raise ValueError("Stage 07 JSON reports an extra CV refit; rerun the updated Stage 07.")
    by_model = tuning.get("params_by_model", {})
    for name in MODELS:
        entry = by_model.get(name)
        if not entry or not entry.get("params"):
            raise ValueError(f"Optuna parameters for {name} are missing; rerun Stage 07.")
        absent = [k for k in CV_KEYS if k not in entry or not np.isfinite(float(entry[k]))]
        if absent:
            raise ValueError(f"Stage 07 CV metrics missing/non-finite for {name}: {absent}.")

    used = OUTPUTS / "07_optuna_tuning/tables/fixed_250_features_used_for_tuning.csv"
    if used.exists():
        tuned_on = pd.read_csv(used)["feature"].astype(str).tolist()
        if tuned_on != list(selected):
            raise ValueError(
                "The feature list used for tuning differs from Stage 05 selected_features.csv "
                "(content or order). Rerun Stage 07 so CV and the final fit use the same list."
            )
    return by_model


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _parity(ax, obs, pred, title, colour):
    lo, hi = float(min(obs.min(), pred.min())), float(max(obs.max(), pred.max()))
    pad = 0.03 * (hi - lo)
    ax.scatter(obs, pred, s=18, alpha=0.45, color=colour, edgecolors="none")
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#D55E00", linewidth=1.8)
    ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad), title=title,
           xlabel="Observed pIC50", ylabel="Predicted pIC50")
    ax.set_aspect("equal", adjustable="box")


def _make_figures(d, observed, predictions, primary, results, ad_table, max_similarity):
    setup_plotting()

    # 1. Parity: primary model
    fig, ax = plt.subplots(figsize=(7, 7))
    _parity(ax, observed, predictions[primary], f"Locked scaffold test: {primary}", COLOURS[primary])
    save_figure(fig, d / "figures/locked_test_parity_primary_model")
    plt.close(fig)

    # 2. Residuals: primary model
    resid = observed - predictions[primary]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(predictions[primary], resid, s=18, alpha=0.45, color="#009E73", edgecolors="none")
    ax.axhline(0, linestyle="--", color="#D55E00", linewidth=2)
    ax.axhline(LOG10_2, linestyle=":", color="#999999")
    ax.axhline(-LOG10_2, linestyle=":", color="#999999")
    ax.set(xlabel="Predicted pIC50", ylabel="Residual (observed − predicted)",
           title=f"Locked scaffold test residuals: {primary}")
    save_figure(fig, d / "figures/locked_test_residuals_primary_model")
    plt.close(fig)

    # 3. Parity grid: all QSAR models + ensemble
    names = [*MODELS, ENSEMBLE]
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, name in zip(axes.ravel(), names):
        row = results.loc[results.model == name].iloc[0]
        _parity(ax, observed, predictions[name],
                f"{name.split(' (')[0]}\nRMSE {row.test_rmse:.3f} | R² {row.test_r2:.3f}", COLOURS[name])
    fig.tight_layout()
    save_figure(fig, d / "figures/locked_test_parity_all_models")
    plt.close(fig)

    # 4. RMSE with scaffold-bootstrap CI, all rows
    ordered = results.sort_values("test_rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(ordered) + 2))
    ypos = np.arange(len(ordered))
    ax.barh(ypos, ordered.test_rmse, color=[COLOURS[m] for m in ordered.model], alpha=0.85)
    ax.errorbar(
        ordered.test_rmse, ypos,
        xerr=[ordered.test_rmse - ordered.test_rmse_ci_low, ordered.test_rmse_ci_high - ordered.test_rmse],
        fmt="none", ecolor="black", capsize=3,
    )
    ax.set_yticks(ypos, [m.split(" (")[0] for m in ordered.model])
    ax.set(xlabel="Locked-test RMSE (pIC50, 95% scaffold-bootstrap CI)",
           title="Models versus baselines")
    save_figure(fig, d / "figures/locked_test_rmse_with_ci")
    plt.close(fig)

    # 5. Error versus similarity to training set (primary model)
    sub = ad_table[ad_table.model == primary]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sub.similarity_bin, sub.rmse, color=COLOURS[primary], alpha=0.85)
    for x, (rmse, n) in enumerate(zip(sub.rmse, sub.n)):
        ax.text(x, rmse, f"n={n}", ha="center", va="bottom", fontsize=9)
    ax.set(xlabel="Max Tanimoto similarity to training set", ylabel="RMSE (pIC50)",
           title=f"Error versus training-set similarity: {primary}")
    save_figure(fig, d / "figures/locked_test_error_by_similarity")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(on_invalid_test="error", n_bootstrap=N_BOOTSTRAP):
    """Fit the three tuned models once on all training data and score the locked test.

    on_invalid_test : "error" (default) stops if a test molecule has a missing or
        non-finite selected feature; "exclude" scores the rest and logs the exclusions.
    """
    if on_invalid_test not in {"error", "exclude"}:
        raise ValueError("on_invalid_test must be 'error' or 'exclude'.")
    cfg = load_config()
    seed = int(cfg.get("seed", 42))

    tuning_json = OUTPUTS / "07_optuna_tuning/tables/optuna_best_params.json"
    required = [
        tuning_json,
        OUTPUTS / "04_train_test_split/tables/test_ids.csv",
        OUTPUTS / "03_feature_generation/tables/all_features.parquet",
        OUTPUTS / "03_feature_generation/tables/target.parquet",
        OUTPUTS / "02_clean_data/tables/cleaned_activity.csv",
    ]
    absent = [str(p) for p in required if not p.exists()]
    if absent:
        raise FileNotFoundError("Run Stages 02-05 and 07 first. Missing: " + ", ".join(absent))

    # Same loader as Stage 07: identical rows, row order and ordered 250 features.
    X_train, y_train, train_groups, f_path, t_path, s_path, ids_path = _load_training_data()
    selected = X_train.columns.astype(str).tolist()
    tuning = json.loads(tuning_json.read_text(encoding="utf-8"))
    by_model = _validate_tuning(tuning, selected)

    test_ids, X_test, y_test, excluded = _load_locked_test(selected, cfg, on_invalid_test)
    overlap = set(map(str, train_groups)) & set(test_ids["group"].astype(str))
    if overlap:
        raise ValueError(f"Scaffold leakage detected: {len(overlap)} train/test groups overlap.")
    if set(X_train.index) & set(X_test.index):
        raise ValueError("Train and locked-test molecule IDs overlap.")

    d = stage_dir("08_final_model_evaluation")
    (d / "tables").mkdir(parents=True, exist_ok=True)
    (d / "models").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature_order": np.arange(1, len(selected) + 1), "feature": selected}).to_csv(
        d / "tables/fixed250_features_used_for_final_fit.csv", index=False)
    excluded.to_csv(d / "tables/locked_test_excluded_molecules.csv", index=False)

    observed = y_test.to_numpy(dtype=float)
    y_train_arr = y_train.to_numpy(dtype=float)

    # ---- fit once on the complete training set, predict the locked test ----
    predictions, fitted, parameter_rows = {}, {}, []
    for name in MODELS:
        tuned = dict(by_model[name]["params"])
        model = make_estimator(name, tuned, seed)
        with threadpool_limits(limits=THREADS):
            model.fit(X_train, y_train)
            pred = np.asarray(model.predict(X_test), dtype=float)
        if not np.isfinite(pred).all():
            raise ValueError(f"{name} returned non-finite test predictions.")
        rows, resolved = _audit_parameters(name, tuned, model)
        parameter_rows.extend(rows)
        predictions[name] = pred
        fitted[name] = {
            "model": model,
            "feature_names": selected,
            "tuned_parameters": tuned,
            "resolved_parameters": json.loads(json.dumps(resolved, default=str)),
            "parameter_source": "Optuna best parameters",
            "seed": seed,
            "stage07_trial_number": by_model[name].get("trial_number"),
        }
        joblib.dump(fitted[name], d / f"models/final_{name}.joblib")
        print(f"Fitted {name}: {len(tuned)} Optuna parameters applied and verified.", flush=True)
    predictions[ENSEMBLE] = np.mean([predictions[m] for m in MODELS], axis=0)
    pd.DataFrame(parameter_rows).to_csv(d / "tables/final_model_parameters.csv", index=False)

    # ---- similarity, baselines, applicability domain ----
    smiles, smiles_path = _smiles_lookup(X_train.index, X_test.index)
    similarity = _similarity_matrix(smiles.loc[X_train.index], smiles.loc[X_test.index], cfg)
    max_similarity = similarity.max(axis=1)
    baseline_pred = _baseline_predictions(similarity, y_train_arr)
    predictions.update(baseline_pred)
    order = [*MODELS, ENSEMBLE, *BASELINES]
    group_by_row = test_ids["group"].astype(str).to_numpy()

    # ---- point metrics ----
    rows = []
    for name in order:
        is_model = name in MODELS
        entry = by_model[name] if is_model else None
        rows.append(
            {
                "comparison_type": "QSAR model" if is_model else (
                    "Supplementary ensemble" if name == ENSEMBLE else "Baseline"),
                "model": name,
                "parameter_source": "Optuna best parameters" if is_model else (
                    "Mean of the three fitted models" if name == ENSEMBLE else "Training data only"),
                "parameters": json.dumps(
                    entry["params"] if is_model else _BASELINE_PARAMS.get(name, {}), sort_keys=True),
                **{k: (float(entry[k]) if is_model else np.nan) for k in CV_KEYS},
                **test_metrics(observed, predictions[name], y_train_arr),
                "n_train": int(len(X_train)),
                "n_features_final_fit": int(len(selected)) if is_model else np.nan,
                "stage07_trial_number": entry.get("trial_number") if is_model else np.nan,
            }
        )
    results = pd.DataFrame(rows)

    # CV -> test generalisation gap (QSAR models only; CV is a tuning-CV estimate)
    is_model = results.model.isin(MODELS)
    results["gap_rmse_test_minus_cv"] = np.where(
        is_model, results.test_rmse - results.cv_mean_fold_rmse, np.nan)
    results["gap_rmse_in_cv_sd"] = np.where(
        is_model, results.gap_rmse_test_minus_cv / results.cv_sd_fold_rmse, np.nan)
    results["gap_r2_test_minus_cv"] = np.where(
        is_model, results.test_r2 - results.cv_mean_fold_r2, np.nan)

    # Primary model: chosen from Stage 07 CV only, never from the test set.
    cv_rank = results.loc[is_model].sort_values(["cv_mean_fold_rmse", "model"])
    primary = str(cv_rank.iloc[0].model)
    results["cv_rank"] = results.model.map({m: i + 1 for i, m in enumerate(cv_rank.model)})
    results["is_primary_model"] = results.model.eq(primary)

    # ---- scaffold-cluster bootstrap ----
    n_test_groups = int(pd.Series(group_by_row).nunique())
    if n_test_groups < 30:
        print(f"WARNING: only {n_test_groups} scaffold groups in the scored test set; "
              "cluster-bootstrap intervals will be unreliable.", flush=True)
    boot_idx = _group_bootstrap_indices(group_by_row, int(n_bootstrap), seed)
    ci, paired = _bootstrap({m: predictions[m] for m in order}, observed, boot_idx)
    results = results.merge(ci, on="model", how="left", validate="one_to_one")
    results["n_bootstrap"] = int(n_bootstrap)

    # ---- applicability-domain breakdown ----
    labels = [f"{lo:.1f}-{min(hi, 1.0):.1f}" for lo, hi in zip(SIMILARITY_BINS[:-1], SIMILARITY_BINS[1:])]
    bin_of = pd.cut(max_similarity, bins=SIMILARITY_BINS, labels=labels, right=False)
    ad_rows = []
    for name in order:
        resid = observed - predictions[name]
        for label in labels:
            mask = np.asarray(bin_of == label)
            if mask.sum() == 0:
                continue
            ad_rows.append(
                {
                    "model": name,
                    "similarity_bin": label,
                    "n": int(mask.sum()),
                    "rmse": float(np.sqrt(np.mean(resid[mask] ** 2))),
                    "mae": float(np.mean(np.abs(resid[mask]))),
                    "bias_obs_minus_pred": float(np.mean(resid[mask])),
                }
            )
    ad_table = pd.DataFrame(ad_rows)
    results["spearman_max_similarity_vs_abs_error"] = [
        _safe_corr(spearmanr, max_similarity, np.abs(observed - predictions[m])) for m in results.model
    ]

    # ---- ordering: QSAR models by CV rank, then ensemble, then baselines by test RMSE ----
    block = results.comparison_type.map({"QSAR model": 0, "Supplementary ensemble": 1, "Baseline": 2})
    results = (
        results.assign(_block=block, _key=np.where(block == 0, results.cv_rank, results.test_rmse))
        .sort_values(["_block", "_key"]).drop(columns=["_block", "_key"]).reset_index(drop=True)
    )

    # ---- write tables ----
    compact = [
        "comparison_type", "model", "cv_rank", "is_primary_model",
        "cv_mean_fold_rmse", "cv_sd_fold_rmse", "cv_mean_fold_mae", "cv_mean_fold_r2",
        "test_n", "test_rmse", "test_rmse_ci_low", "test_rmse_ci_high",
        "test_mae", "test_mae_ci_low", "test_mae_ci_high",
        "test_r2", "test_r2_ci_low", "test_r2_ci_high",
        "test_pearson", "test_spearman", "test_spearman_ci_low", "test_spearman_ci_high",
        "test_ccc", "test_ccc_ci_low", "test_ccc_ci_high",
        "gap_rmse_test_minus_cv", "gap_rmse_in_cv_sd", "gap_r2_test_minus_cv",
    ]
    results[compact].to_csv(d / "tables/final_model_comparison.csv", index=False)
    results.to_csv(d / "tables/final_test_metrics_full.csv", index=False)
    paired.to_csv(d / "tables/paired_rmse_differences_bootstrap.csv", index=False)
    ad_table.to_csv(d / "tables/applicability_domain_error.csv", index=False)

    prediction_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "chembl_id": test_ids.chembl_id.to_numpy(),
                    "group": group_by_row,
                    "model": name,
                    "observed_pIC50": observed,
                    "predicted_pIC50": predictions[name],
                    "residual": observed - predictions[name],
                    "max_train_tanimoto": max_similarity,
                }
            )
            for name in order
        ],
        ignore_index=True,
    )
    prediction_frame[~prediction_frame.model.isin(BASELINES)].to_csv(
        d / "tables/locked_test_predictions_all_models.csv", index=False)
    prediction_frame[prediction_frame.model.isin(BASELINES)].to_csv(
        d / "tables/locked_test_baseline_predictions.csv", index=False)

    worst = prediction_frame[prediction_frame.model == primary].copy()
    worst["abs_error"] = worst.residual.abs()
    worst["canonical_smiles"] = worst.chembl_id.map(smiles)
    worst.sort_values("abs_error", ascending=False).head(25).to_csv(
        d / "tables/largest_errors_primary_model.csv", index=False)

    joblib.dump(
        {
            "primary_model_by_stage07_cv_mean_fold_rmse": primary,
            "models": fitted,
            "feature_names": selected,
            "cv_source": "Stage 07 fixed-250 five-fold scaffold-grouped tuning CV; no Stage 08 CV",
        },
        d / "models/final_model_bundle.joblib",
    )

    _make_figures(d, observed, predictions, primary, results, ad_table, max_similarity)

    # ---- console report ----
    show = results[[
        "model", "cv_rank", "cv_mean_fold_rmse", "test_rmse", "test_rmse_ci_low", "test_rmse_ci_high",
        "test_mae", "test_r2", "test_pearson", "test_spearman", "test_ccc",
        "test_frac_within_2fold", "gap_rmse_test_minus_cv",
    ]]
    print("\nLocked-test results (CV columns are Stage 07 values; no CV was run here):")
    print(show.round(4).to_string(index=False))
    print(f"\nPrimary model (lowest Stage 07 mean-fold CV RMSE): {primary}")
    tied = cv_rank[cv_rank.cv_mean_fold_rmse - cv_rank.cv_mean_fold_rmse.iloc[0]
                   <= cv_rank.cv_sd_fold_rmse.iloc[0]].model.tolist()
    if len(tied) > 1:
        print(f"CV RMSE of {tied} lie within one CV SD of the best; treat them as statistically tied.")
    if len(excluded):
        print(f"{len(excluded)} locked-test molecule(s) excluded for missing/non-finite features "
              "(see tables/locked_test_excluded_molecules.csv).")
    print("Stage 08 ran no CV and no imputation. The ensemble is supplementary and never primary.")
    print("CV is conditional on the pre-selected 250 features and on Optuna selection, so it may be "
          "optimistic; the locked test is the external estimate. Do not retune from these results.")

    write_manifest(
        "08_final_model_evaluation",
        {
            "models_fitted": list(MODELS),
            "parameters": "Optuna best parameters, applied and audited against the fitted estimators",
            "primary_model_by_stage07_cv_mean_fold_rmse": primary,
            "feature_budget_final_fit": len(selected),
            "cv_source": "Stage 07 five-fold scaffold-grouped tuning CV; no Stage 08 CV",
            "bootstrap": f"{int(n_bootstrap)} scaffold-cluster resamples of the locked test set",
            "on_invalid_test": on_invalid_test,
            "n_test_scored": int(len(observed)),
            "n_test_excluded": int(len(excluded)),
            "test_used_for_tuning": False,
            "no_imputation": True,
        },
        [f_path, t_path, s_path, ids_path, tuning_json, smiles_path,
         OUTPUTS / "04_train_test_split/tables/test_ids.csv",
         OUTPUTS / "03_feature_generation/tables/all_features.parquet"],
        [
            d / "tables/final_model_comparison.csv",
            d / "tables/final_test_metrics_full.csv",
            d / "tables/final_model_parameters.csv",
            d / "tables/paired_rmse_differences_bootstrap.csv",
            d / "tables/applicability_domain_error.csv",
            d / "tables/locked_test_predictions_all_models.csv",
            d / "tables/locked_test_baseline_predictions.csv",
            d / "tables/fixed250_features_used_for_final_fit.csv",
            d / "models/final_model_bundle.joblib",
        ],
    )
    return results


if __name__ == "__main__":
    run()
