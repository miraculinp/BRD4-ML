"""Stage 09b: applicability-domain (AD) analysis of the locked test set.

What this stage does
--------------------
* Reads the Stage 07 tuning JSON and the Stage 08 locked-test predictions.
  It never refits a prediction model, never tunes, and never runs
  Y-randomization.
* Loads the training matrix with the *same* loader as Stages 07 and 08
  (``_load_training_data``), so the AD is defined on exactly the rows and the
  ordered 250 features the models were tuned and fitted on.
* Defines the domain with four complementary methods, every limit estimated
  from training data only:
    1. standardized 5-NN distance in descriptor space,
    2. training-feature bounding box,
    3. leverage (hat value),
    4. maximum Morgan Tanimoto similarity to the training set.
* Reports, for every QSAR model (and the supplementary ensemble), test
  performance inside and outside each domain with scaffold-cluster bootstrap
  intervals, and tests whether the AD actually tracks prediction error.

Design decisions that differ from the earlier version
-----------------------------------------------------
* Group-aware training reference. The split is scaffold-grouped, so a test
  molecule never has a same-scaffold neighbour in training. Training
  distances/similarities used to set thresholds therefore ignore neighbours
  from the molecule's own scaffold group. Without this the thresholds are
  set too tight and good test predictions are flagged as outside.
* Leverage threshold uses the rank of the training design matrix,
  h* = 3 * rank / n (equal to the usual 3(p+1)/n when it has full rank).
* Residuals are scaled by each model's own Stage 07 pooled out-of-fold RMSE.
  The old ``fresh_cv_oof_predictions_*`` files no longer exist because
  Stage 08 runs no cross-validation.
* The set of scored test molecules is taken from the Stage 08 prediction
  file, so molecules Stage 08 excluded (``on_invalid_test='exclude'``) are
  handled instead of crashing the stage.
* The primary model is read from Stage 07 CV (never from the test set).
"""

from __future__ import annotations

import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.final_evaluation import ENSEMBLE
from src.optuna_tuning import TUNED_MODELS, _load_training_data
from src.utils import (
    OUTPUTS,
    load_config,
    save_figure,
    setup_plotting,
    stage_dir,
    write_manifest,
)


STAGE = "09b_applicability_domain"
MODELS = tuple(TUNED_MODELS)
STAGE08 = OUTPUTS / "08_final_model_evaluation/tables"
PREDICTION_FILE = STAGE08 / "locked_test_predictions_all_models.csv"
EXCLUDED_FILE = STAGE08 / "locked_test_excluded_molecules.csv"
TUNING_JSON = OUTPUTS / "07_optuna_tuning/tables/optuna_best_params.json"

K_NEIGHBORS = 5
KNN_SD_MULTIPLIER = 2.0          # threshold = mean + 2 SD of training distances
SIMILARITY_PERCENTILE = 5.0      # inside if >= this percentile of training similarity
CONSENSUS_MIN_METHODS = 3        # inside at least 3 of the 4 methods
MIN_N_FOR_R2 = 10                # no R2 for subsets smaller than this
N_BOOTSTRAP = 2000
CHUNK = 500

METHOD_ORDER = ("knn", "bounding_box", "leverage", "similarity")
METHOD_LABEL = {
    "knn": "Standardized 5-NN distance",
    "bounding_box": "Training-feature bounding box",
    "leverage": "Leverage (hat value)",
    "similarity": "Max Tanimoto similarity",
    "consensus": "Consensus (inside >= 3 of 4)",
}
INSIDE_COLORS = {
    "ExtraTreesRegressor": "#2586B8",
    "CatBoostRegressor": "#20955A",
    "SVR": "#8055A5",
}
OUTSIDE_COLOR = "#D94B45"


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def _short(name):
    return name.removesuffix("Regressor")


def _score_subset(y_true, y_pred):
    """R2 / RMSE / MAE for a subset; R2 is NaN below MIN_N_FOR_R2 rows."""
    n = len(y_true)
    if n == 0:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan}
    return {
        "r2": float(r2_score(y_true, y_pred)) if n >= MIN_N_FOR_R2 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _safe_spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(a, b)[0])


# --------------------------------------------------------------------------
# Group-aware training reference (mimics the scaffold-split test situation)
# --------------------------------------------------------------------------
def _group_codes(groups):
    return pd.factorize(pd.Series(groups).astype(str))[0]


def _group_aware_knn_distance(train_scaled, codes, k):
    """Mean distance to the k nearest training molecules of OTHER scaffold groups."""
    if len(np.unique(codes)) < 2:
        raise ValueError("Need at least two training scaffold groups for a group-aware AD.")
    out = np.empty(len(codes), dtype=float)
    for start in range(0, len(codes), CHUNK):
        sl = slice(start, min(start + CHUNK, len(codes)))
        dist = cdist(train_scaled[sl], train_scaled)
        dist[codes[sl][:, None] == codes[None, :]] = np.inf
        nearest = np.partition(dist, k - 1, axis=1)[:, :k]
        if not np.isfinite(nearest).all():
            raise ValueError(
                f"Some training molecules have fewer than {k} neighbours outside their own group."
            )
        out[sl] = nearest.mean(axis=1)
    return out


def _fingerprints(smiles, cfg):
    """Morgan fingerprints with the same settings as Stage 08."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(cfg.get("fingerprint_radius", 2)),
        fpSize=int(cfg.get("fingerprint_bits", 2048)),
        includeChirality=bool(cfg.get("use_chirality", True)),
    )
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise ValueError(f"Invalid cleaned SMILES; cannot compute similarity: {s}")
        fps.append(generator.GetFingerprint(mol))
    return fps


def _max_similarity(query_fps, ref_fps, query_codes=None, ref_codes=None):
    """Max Tanimoto of each query to the reference set; optionally skip own group."""
    from rdkit import DataStructs

    out = np.empty(len(query_fps), dtype=float)
    for i, fp in enumerate(query_fps):
        sim = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, ref_fps), dtype=float)
        if query_codes is not None:
            sim[ref_codes == query_codes[i]] = -np.inf
        out[i] = sim.max()
    if not np.isfinite(out).all():
        raise ValueError("Similarity reference has no molecules outside a query's own group.")
    return out


def _smiles_for(ids):
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
    missing = [i for i in ids if i not in lookup.index]
    if missing:
        raise ValueError(f"Missing cleaned SMILES for {len(missing)} molecules.")
    return lookup.loc[list(ids)].tolist(), path


# --------------------------------------------------------------------------
# Scaffold-cluster bootstrap on the locked test set
# --------------------------------------------------------------------------
def _bootstrap_weights(codes, n_boot, seed):
    """Row b gives how many times each test scaffold group is drawn in resample b."""
    n_groups = int(codes.max()) + 1
    draws = np.random.default_rng(seed).integers(0, n_groups, size=(n_boot, n_groups))
    weights = np.zeros((n_boot, n_groups), dtype=float)
    for b in range(n_boot):
        weights[b] = np.bincount(draws[b], minlength=n_groups)
    return weights


def _boot_rmse(sq_err, mask, codes, weights):
    """RMSE of the masked subset in every cluster resample (NaN if the subset is empty)."""
    n_groups = weights.shape[1]
    group_sse = np.bincount(codes, weights=sq_err * mask, minlength=n_groups)
    group_n = np.bincount(codes, weights=mask.astype(float), minlength=n_groups)
    num, den = weights @ group_sse, weights @ group_n
    return np.where(den > 0, np.sqrt(num / np.where(den > 0, den, 1.0)), np.nan)


def _interval(samples):
    """95% percentile interval; NaN when too many resamples had an empty subset."""
    if np.isfinite(samples).mean() < 0.9:
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# Applying a saved domain to new compounds (e.g. Stage 10 virtual screening)
# --------------------------------------------------------------------------
def assess_new_compounds(X_new, bundle_path=None):
    """Descriptor-based AD flags for new molecules using the saved training statistics.

    ``X_new`` needs the 250 selected feature columns. The similarity method needs
    SMILES and is not included here. New molecules are not in the training set,
    so the plain (not group-aware) kNN distance to the full training set is used.
    """
    bundle = joblib.load(bundle_path or (OUTPUTS / STAGE / "models/ad_statistics_all_methods.joblib"))
    names = bundle["feature_names"]
    missing = [c for c in names if c not in X_new.columns]
    if missing:
        raise ValueError(f"New data lacks {len(missing)} selected features, e.g. {missing[:5]}")
    X = X_new.loc[:, names].apply(pd.to_numeric, errors="coerce")
    values = X.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("New data contain missing or infinite values; no imputation is used.")
    scaled = bundle["scaler"].transform(values)
    knn = bundle["knn_model"].kneighbors(scaled, return_distance=True)[0].mean(axis=1)
    out_count = ((values < bundle["bounding_box_min"].to_numpy())
                 | (values > bundle["bounding_box_max"].to_numpy())).sum(axis=1)
    design = np.column_stack([np.ones(len(scaled)), scaled])
    leverage = np.einsum("ij,jk,ik->i", design, bundle["leverage_gram_pseudoinverse"], design)
    frame = pd.DataFrame(
        {
            "mean_5nn_distance": knn,
            "inside_knn_ad": knn <= bundle["knn_threshold"],
            "out_of_bounds_feature_count": out_count,
            "inside_bounding_box_ad": out_count == 0,
            "leverage": leverage,
            "inside_leverage_ad": leverage <= bundle["leverage_threshold"],
        },
        index=X.index,
    )
    frame["n_descriptor_methods_inside"] = frame[
        ["inside_knn_ad", "inside_bounding_box_ad", "inside_leverage_ad"]
    ].sum(axis=1)
    return frame


# --------------------------------------------------------------------------
# Figures (one standalone figure each, saved as PNG and PDF)
# --------------------------------------------------------------------------
def _figures(d, ctx):
    setup_plotting()
    test_ad, flags = ctx["test_ad"], ctx["flags"]
    primary, k = ctx["primary"], K_NEIGHBORS

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ctx["train_knn_group_aware"], bins=40, alpha=0.65, color="#0072B2",
            label="Training (other scaffold groups only)")
    ax.hist(test_ad["mean_5nn_distance"], bins=40, alpha=0.55, color="#E69F00", label="Locked test")
    ax.axvline(ctx["knn_threshold"], color="#D55E00", linestyle="--",
               label=f"Threshold = {ctx['knn_threshold']:.2f}")
    ax.set(xlabel=f"Mean distance to {k} nearest training molecules (standardized descriptors)",
           ylabel="Molecule count", title="Applicability domain: kNN distance")
    ax.legend()
    save_figure(fig, d / "figures/ad_knn_distance")
    plt.close(fig)

    n_feat = len(ctx["selected"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(test_ad["out_of_bounds_feature_count"], bins=np.arange(-0.5, n_feat + 1.5, 1),
            color="#009E73", alpha=0.85)
    ax.set(xlabel="Number of features outside training min/max", ylabel="Test molecule count",
           title="Applicability domain: bounding-box violations")
    save_figure(fig, d / "figures/ad_bounding_box_violations")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ctx["train_sim_group_aware"], bins=40, alpha=0.65, color="#0072B2",
            label="Training (other scaffold groups only)")
    ax.hist(test_ad["max_train_tanimoto"], bins=40, alpha=0.55, color="#E69F00", label="Locked test")
    ax.axvline(ctx["similarity_threshold"], color="#D55E00", linestyle="--",
               label=f"Threshold = {ctx['similarity_threshold']:.2f}")
    ax.set(xlabel="Maximum Tanimoto similarity to the training set", ylabel="Molecule count",
           title="Applicability domain: structural similarity")
    ax.legend()
    save_figure(fig, d / "figures/ad_similarity_distribution")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [METHOD_LABEL[m].replace("Standardized ", "").replace("Training-feature ", "")
              for m in METHOD_ORDER]
    pct = [flags[m].mean() * 100 for m in METHOD_ORDER]
    ax.bar(labels, pct, color=["#0072B2", "#E69F00", "#009E73", "#CC79A7"])
    for i, m in enumerate(METHOD_ORDER):
        ax.text(i, pct[i] + 1.5, f"{flags[m].sum():,}/{len(test_ad):,}", ha="center", fontsize=10)
    ax.set(ylabel="Test compounds inside domain (%)", ylim=(0, 108),
           title="Applicability-domain coverage by method")
    ax.tick_params(axis="x", labelrotation=15)
    save_figure(fig, d / "figures/ad_method_coverage")
    plt.close(fig)

    specs = {
        "knn": ("mean_5nn_distance", "Mean distance to 5 nearest training molecules",
                ctx["knn_threshold"], "kNN threshold", "knn"),
        "bounding_box": ("out_of_bounds_feature_count", "Number of descriptors outside training min/max",
                         0.5, "Inside: 0 violations", "bounding_box"),
        "leverage": ("leverage", "Leverage (hat value)", ctx["leverage_threshold"],
                     "Leverage threshold h*", "leverage"),
        "similarity": ("max_train_tanimoto", "Maximum Tanimoto similarity to training set",
                       ctx["similarity_threshold"], "Similarity threshold", "similarity"),
    }
    for model in MODELS:
        resid = test_ad[f"std_residual_{model}"].to_numpy(dtype=float)
        for method in METHOD_ORDER:
            column, xlabel, threshold, limit_label, stem = specs[method]
            inside = flags[method]
            x = test_ad[column].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(8, 5.5))
            ax.scatter(x[inside], resid[inside], color=INSIDE_COLORS[model], marker="o", s=24,
                       alpha=0.65, edgecolors="none", label=f"Inside AD ({_short(model)})")
            ax.scatter(x[~inside], resid[~inside], color=OUTSIDE_COLOR, marker="^", s=34,
                       alpha=0.85, edgecolors="none", label="Outside AD")
            ax.axvline(threshold, color="black", linestyle="--", label=limit_label)
            ax.axhline(0, color="gray", linewidth=1)
            ax.axhline(3, color="gray", linestyle=":", label="±3 residual limit")
            ax.axhline(-3, color="gray", linestyle=":")
            ax.set(xlabel=xlabel, ylabel="Test residual / Stage 07 pooled CV RMSE",
                   title=f"{METHOD_LABEL[method]}: {model}")
            ax.legend()
            save_figure(fig, d / f"figures/ad_residual_{stem}_{_short(model)}")
            plt.close(fig)

    perf = ctx["performance"]
    sub = perf[(perf.model == primary) & (perf.method_key.isin(METHOD_ORDER))]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.38
    for j, (status, colour) in enumerate((("inside", "#0072B2"), ("outside", OUTSIDE_COLOR))):
        part = sub[sub.domain_status == status].set_index("method_key").reindex(METHOD_ORDER)
        xs = np.arange(len(METHOD_ORDER)) + (j - 0.5) * width
        yerr = np.vstack([
            (part.rmse - part.rmse_ci_low).fillna(0).clip(lower=0),
            (part.rmse_ci_high - part.rmse).fillna(0).clip(lower=0),
        ])
        ax.bar(xs, part.rmse.fillna(0), width, color=colour, alpha=0.85, label=status.capitalize())
        ax.errorbar(xs, part.rmse.fillna(0), yerr=yerr, fmt="none", ecolor="black", capsize=3)
        for x_pos, n in zip(xs, part.n_molecules):
            ax.text(x_pos, 0.01, f"n={int(n)}", ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks(np.arange(len(METHOD_ORDER)), [_short(METHOD_LABEL[m]) for m in METHOD_ORDER], rotation=15)
    ax.set(ylabel="Locked-test RMSE (pIC50, 95% scaffold-bootstrap CI)",
           title=f"Error inside vs outside the domain: {primary}")
    ax.legend()
    save_figure(fig, d / "figures/ad_rmse_inside_outside_primary_model")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(n_bootstrap=N_BOOTSTRAP):
    cfg = load_config()
    seed = int(cfg.get("seed", 42))
    tables03 = OUTPUTS / "03_feature_generation/tables"
    test_ids_file = OUTPUTS / "04_train_test_split/tables/test_ids.csv"
    required = [tables03 / "all_features.parquet", tables03 / "target.parquet",
                test_ids_file, TUNING_JSON, PREDICTION_FILE,
                OUTPUTS / "02_clean_data/tables/cleaned_activity.csv"]
    absent = [str(p) for p in required if not p.is_file()]
    if absent:
        raise FileNotFoundError("Run Stages 02-05, 07 and 08 first. Missing: " + ", ".join(absent))

    # ---- training data: same loader, rows and 250 features as Stages 07/08 ----
    X_train_df, y_train, train_groups, f_path, t_path, s_path, ids_path = _load_training_data()
    selected = X_train_df.columns.astype(str).tolist()
    X_train = X_train_df.to_numpy(dtype=float)
    n_train, n_feat = X_train.shape
    train_codes = _group_codes(train_groups)

    # ---- Stage 07 information (scaling of residuals, primary model) ----
    tuning = json.loads(TUNING_JSON.read_text(encoding="utf-8"))
    by_model = tuning["params_by_model"]
    cv_scale = {}
    for name in MODELS:
        scale = float(by_model[name]["cv_pooled_oof_rmse"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Stage 07 pooled OOF RMSE for {name} is not positive.")
        cv_scale[name] = scale
    primary = min(MODELS, key=lambda m: (float(by_model[m]["cv_mean_fold_rmse"]), m))

    # ---- locked test molecules actually scored in Stage 08 ----
    predictions = pd.read_csv(PREDICTION_FILE, dtype={"chembl_id": str, "group": str})
    need = {"chembl_id", "group", "model", "observed_pIC50", "predicted_pIC50", "residual",
            "max_train_tanimoto"}
    if need - set(predictions.columns):
        raise ValueError("Stage 08 prediction file lacks: " + ", ".join(sorted(need - set(predictions.columns))))
    if set(MODELS) - set(predictions.model.unique()):
        raise ValueError("Stage 08 predictions lack models: "
                         + ", ".join(sorted(set(MODELS) - set(predictions.model.unique()))))
    report_models = [*MODELS] + ([ENSEMBLE] if ENSEMBLE in set(predictions.model) else [])

    test_ids_all = pd.read_csv(test_ids_file, dtype={"chembl_id": str})
    first = predictions[predictions.model == MODELS[0]].reset_index(drop=True)
    scored_ids = first.chembl_id.astype(str).tolist()
    if len(set(scored_ids)) != len(scored_ids):
        raise ValueError("Duplicate test IDs in the Stage 08 predictions.")
    if not set(scored_ids) <= set(test_ids_all.chembl_id):
        raise ValueError("Stage 08 scored molecules that are not in the locked test ID list.")
    n_excluded = len(test_ids_all) - len(scored_ids)
    if n_excluded:
        print(f"Note: {n_excluded} locked-test molecule(s) were excluded in Stage 08 and are not analysed here.")
    by_model_pred = {}
    for name in report_models:
        part = predictions[predictions.model == name].set_index("chembl_id").reindex(scored_ids)
        if part.predicted_pIC50.isna().any():
            raise ValueError(f"{name} lacks predictions for some scored test molecules.")
        by_model_pred[name] = part
    observed = first.observed_pIC50.to_numpy(dtype=float)
    test_groups = first.group.astype(str).to_numpy()
    if set(test_groups) & set(map(str, train_groups)):
        raise ValueError("Train/test scaffold overlap found; AD analysis stopped.")

    features = pd.read_parquet(tables03 / "all_features.parquet")
    target = pd.read_parquet(tables03 / "target.parquet")["pIC50"]
    features.index, target.index = features.index.astype(str), target.index.astype(str)
    X_test_df = features.reindex(scored_ids).loc[:, selected].apply(pd.to_numeric, errors="coerce")
    X_test = X_test_df.to_numpy(dtype=float)
    if not np.isfinite(X_test).all():
        raise ValueError("Scored test features contain missing or infinite values; no imputation is used.")
    if not np.allclose(target.reindex(scored_ids).to_numpy(dtype=float), observed, atol=1e-8):
        raise ValueError("Observed pIC50 in the Stage 08 file differs from the Stage 03 target table.")
    test_codes = _group_codes(test_groups)
    n_test = len(scored_ids)

    # ---- 1) kNN distance ----
    scaler = StandardScaler().fit(X_train)
    train_scaled, test_scaled = scaler.transform(X_train), scaler.transform(X_test)
    k = min(K_NEIGHBORS, n_train - 1)
    if k < 1:
        raise ValueError("At least two training molecules are needed for kNN AD.")
    neighbor_model = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1).fit(train_scaled)
    train_knn_naive = neighbor_model.kneighbors(return_distance=True)[0].mean(axis=1)   # self excluded
    train_knn = _group_aware_knn_distance(train_scaled, train_codes, k)                 # own group excluded
    test_knn = neighbor_model.kneighbors(test_scaled, return_distance=True)[0].mean(axis=1)
    knn_threshold = float(train_knn.mean() + KNN_SD_MULTIPLIER * train_knn.std(ddof=0))
    knn_threshold_naive = float(train_knn_naive.mean() + KNN_SD_MULTIPLIER * train_knn_naive.std(ddof=0))

    # ---- 2) bounding box ----
    box_min, box_max = X_train.min(axis=0), X_train.max(axis=0)
    out_of_bounds = (X_test < box_min) | (X_test > box_max)
    out_count = out_of_bounds.sum(axis=1)

    # ---- 3) leverage ----
    design_train = np.column_stack([np.ones(n_train), train_scaled])
    design_test = np.column_stack([np.ones(n_test), test_scaled])
    gram_inverse = np.linalg.pinv(design_train.T @ design_train, hermitian=True)
    test_leverage = np.einsum("ij,jk,ik->i", design_test, gram_inverse, design_test)
    train_leverage = np.einsum("ij,jk,ik->i", design_train, gram_inverse, design_train)
    design_rank = int(np.linalg.matrix_rank(design_train))
    leverage_threshold = float(3 * design_rank / n_train)
    if n_train <= n_feat + 1:
        print("WARNING: n_train <= p + 1, leverage values saturate and are not informative.", flush=True)

    # ---- 4) maximum Tanimoto similarity (group-aware training reference) ----
    smiles_all, smiles_path = _smiles_for([*X_train_df.index.astype(str), *scored_ids])
    train_fps = _fingerprints(smiles_all[:n_train], cfg)
    test_fps = _fingerprints(smiles_all[n_train:], cfg)
    train_sim = _max_similarity(train_fps, train_fps, train_codes, train_codes)
    test_sim = _max_similarity(test_fps, train_fps)
    if not np.allclose(test_sim, first.max_train_tanimoto.to_numpy(dtype=float), atol=1e-6):
        raise ValueError("Recomputed test similarity differs from Stage 08 max_train_tanimoto; "
                         "check the fingerprint settings in the config.")
    similarity_threshold = float(np.percentile(train_sim, SIMILARITY_PERCENTILE))

    flags = {
        "knn": test_knn <= knn_threshold,
        "bounding_box": out_count == 0,
        "leverage": test_leverage <= leverage_threshold,
        "similarity": test_sim >= similarity_threshold,
    }
    votes = np.sum([flags[m] for m in METHOD_ORDER], axis=0)
    flags["consensus"] = votes >= CONSENSUS_MIN_METHODS
    all_methods = (*METHOD_ORDER, "consensus")

    d = stage_dir(STAGE)
    for sub in ("tables", "figures", "models"):
        (d / sub).mkdir(parents=True, exist_ok=True)

    # ---- per-molecule table ----
    test_ad = pd.DataFrame({
        "chembl_id": scored_ids,
        "scaffold_group": test_groups,
        "observed_pIC50": observed,
        "mean_5nn_distance": test_knn,
        "knn_threshold": knn_threshold,
        "inside_knn_ad": flags["knn"],
        "out_of_bounds_feature_count": out_count,
        "fraction_features_out_of_bounds": out_count / n_feat,
        "inside_bounding_box_ad": flags["bounding_box"],
        "leverage": test_leverage,
        "leverage_threshold": leverage_threshold,
        "inside_leverage_ad": flags["leverage"],
        "max_train_tanimoto": test_sim,
        "similarity_threshold": similarity_threshold,
        "inside_similarity_ad": flags["similarity"],
        "n_methods_inside": votes,
        "inside_consensus_ad": flags["consensus"],
    })
    for name in report_models:
        pred = by_model_pred[name].predicted_pIC50.to_numpy(dtype=float)
        test_ad[f"predicted_{name}"] = pred
        test_ad[f"residual_{name}"] = observed - pred
        if name in cv_scale:
            test_ad[f"std_residual_{name}"] = (observed - pred) / cv_scale[name]
    test_ad.to_csv(d / "tables/test_ad_flags_all_methods.csv", index=False)

    # ---- inside / outside performance with scaffold-cluster bootstrap ----
    weights = _bootstrap_weights(test_codes, int(n_bootstrap), seed)
    rules = {
        "knn": f"training (other-group) mean + {KNN_SD_MULTIPLIER:g} SD of {k}-NN distance",
        "bounding_box": f"all {n_feat} features within training min/max",
        "leverage": "h* = 3 * rank(design) / n_train",
        "similarity": f"max Tanimoto >= {SIMILARITY_PERCENTILE:g}th percentile of training (other-group)",
        "consensus": f"inside at least {CONSENSUS_MIN_METHODS} of the 4 methods",
    }
    thresholds = {"knn": knn_threshold, "bounding_box": np.nan, "leverage": leverage_threshold,
                  "similarity": similarity_threshold, "consensus": np.nan}
    perf_rows, delta_rows = [], []
    for method in all_methods:
        inside_mask = flags[method]
        for name in report_models:
            pred = by_model_pred[name].predicted_pIC50.to_numpy(dtype=float)
            sq = (observed - pred) ** 2
            boots = {}
            for status, mask in (("inside", inside_mask), ("outside", ~inside_mask)):
                met = _score_subset(observed[mask], pred[mask])
                boots[status] = _boot_rmse(sq, mask, test_codes, weights)
                lo, hi = _interval(boots[status])
                perf_rows.append({
                    "method": METHOD_LABEL[method], "method_key": method,
                    "threshold_rule": rules[method], "threshold": thresholds[method],
                    "model": name, "domain_status": status,
                    "n_molecules": int(mask.sum()),
                    "n_scaffold_groups": int(np.unique(test_codes[mask]).size),
                    "fraction_of_test": float(mask.mean()),
                    "r2": met["r2"], "rmse": met["rmse"], "mae": met["mae"],
                    "rmse_ci_low": lo, "rmse_ci_high": hi,
                })
            delta = boots["outside"] - boots["inside"]
            lo, hi = _interval(delta)
            rm_in = perf_rows[-2]["rmse"]
            rm_out = perf_rows[-1]["rmse"]
            delta_rows.append({
                "method": METHOD_LABEL[method], "method_key": method, "model": name,
                "n_inside": int(inside_mask.sum()), "n_outside": int((~inside_mask).sum()),
                "rmse_inside": rm_in, "rmse_outside": rm_out,
                "rmse_outside_minus_inside": (rm_out - rm_in) if np.isfinite(rm_in) and np.isfinite(rm_out) else np.nan,
                "ci_low": lo, "ci_high": hi,
                "ci_excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
            })
    performance = pd.DataFrame(perf_rows)
    performance.to_csv(d / "tables/ad_performance_inside_outside_all_methods.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(d / "tables/ad_rmse_difference_outside_minus_inside.csv", index=False)

    # ---- manuscript table (kNN-defined domain), all models + ensemble ----
    knn_rows = []
    for name in report_models:
        pred = by_model_pred[name].predicted_pIC50.to_numpy(dtype=float)
        inside_m = _score_subset(observed[flags["knn"]], pred[flags["knn"]])
        outside_m = _score_subset(observed[~flags["knn"]], pred[~flags["knn"]])
        all_m = _score_subset(observed, pred)
        knn_rows.append({
            "Model": name,
            "R2_All": all_m["r2"], "R2_Inside_AD": inside_m["r2"], "R2_Outside_AD": outside_m["r2"],
            "N_Inside": int(flags["knn"].sum()), "N_Outside": int((~flags["knn"]).sum()),
            "RMSE_All": all_m["rmse"], "RMSE_Inside_AD": inside_m["rmse"], "RMSE_Outside_AD": outside_m["rmse"],
            "MAE_All": all_m["mae"], "MAE_Inside_AD": inside_m["mae"], "MAE_Outside_AD": outside_m["mae"],
        })
    knn_performance = pd.DataFrame(knn_rows)
    knn_performance.to_csv(d / "tables/knn_domain_model_performance.csv", index=False)

    # ---- coverage ----
    coverage = pd.DataFrame([{
        "method": METHOD_LABEL[m], "method_key": m, "threshold_rule": rules[m], "threshold": thresholds[m],
        "n_train": n_train, "n_test": n_test,
        "n_inside_ad": int(flags[m].sum()), "n_outside_ad": int((~flags[m]).sum()),
        "fraction_inside_ad": float(flags[m].mean()),
    } for m in all_methods])
    coverage.to_csv(d / "tables/ad_method_coverage.csv", index=False)

    # ---- does the AD track error? rank correlation and distance quintiles ----
    scores = {
        "mean_5nn_distance": test_knn,
        "leverage": test_leverage,
        "dissimilarity (1 - max Tanimoto)": 1.0 - test_sim,
        "fraction_features_out_of_bounds": out_count / n_feat,
    }
    rel_rows = []
    for name in report_models:
        abs_err = np.abs(observed - by_model_pred[name].predicted_pIC50.to_numpy(dtype=float))
        for score_name, score in scores.items():
            rel_rows.append({"model": name, "ad_score": score_name,
                             "spearman_with_abs_error": _safe_spearman(score, abs_err), "n": n_test})
    pd.DataFrame(rel_rows).to_csv(d / "tables/ad_score_vs_error_spearman.csv", index=False)

    quintile = pd.qcut(pd.Series(test_knn).rank(method="first"), 5,
                       labels=[f"Q{i}" for i in range(1, 6)])
    q_rows = []
    for name in report_models:
        resid = observed - by_model_pred[name].predicted_pIC50.to_numpy(dtype=float)
        for q in quintile.cat.categories:
            m = (quintile == q).to_numpy()
            q_rows.append({"model": name, "knn_distance_quintile": q, "n": int(m.sum()),
                           "distance_min": float(test_knn[m].min()), "distance_max": float(test_knn[m].max()),
                           "rmse": float(np.sqrt(np.mean(resid[m] ** 2))),
                           "mae": float(np.mean(np.abs(resid[m]))),
                           "bias_obs_minus_pred": float(np.mean(resid[m]))})
    pd.DataFrame(q_rows).to_csv(d / "tables/error_by_knn_distance_quintile.csv", index=False)

    # ---- sensitivity of the kNN rule (primary model) ----
    p_pred = by_model_pred[primary].predicted_pIC50.to_numpy(dtype=float)
    sens_rows = []
    for label, thr in (
        ("mean + 2 SD, training distances ignoring own group (used)", knn_threshold),
        ("mean + 2 SD, training distances excluding only self (earlier rule)", knn_threshold_naive),
        ("95th percentile, ignoring own group", float(np.percentile(train_knn, 95))),
        ("99th percentile, ignoring own group", float(np.percentile(train_knn, 99))),
    ):
        inside_m = test_knn <= thr
        sens_rows.append({
            "rule": label, "threshold": thr, "n_inside": int(inside_m.sum()),
            "n_outside": int((~inside_m).sum()), "fraction_inside": float(inside_m.mean()),
            f"rmse_inside_{_short(primary)}": _score_subset(observed[inside_m], p_pred[inside_m])["rmse"],
            f"rmse_outside_{_short(primary)}": _score_subset(observed[~inside_m], p_pred[~inside_m])["rmse"],
        })
    pd.DataFrame(sens_rows).to_csv(d / "tables/knn_threshold_sensitivity.csv", index=False)

    # ---- saved statistics for Stage 10 / new compounds ----
    joblib.dump({
        "feature_names": selected, "n_train": n_train,
        "scaler": scaler, "knn_model": neighbor_model, "knn_k": k,
        "knn_train_mean_distance": float(train_knn.mean()),
        "knn_train_sd_distance": float(train_knn.std(ddof=0)),
        "knn_threshold": knn_threshold, "knn_threshold_naive": knn_threshold_naive,
        "bounding_box_min": pd.Series(box_min, index=selected),
        "bounding_box_max": pd.Series(box_max, index=selected),
        "leverage_gram_pseudoinverse": gram_inverse,
        "leverage_threshold": leverage_threshold, "leverage_rank": design_rank,
        "similarity_threshold": similarity_threshold,
        "primary_model": primary, "stage07_pooled_oof_rmse": cv_scale,
    }, d / "models/ad_statistics_all_methods.joblib")

    ctx = {
        "test_ad": test_ad, "flags": flags, "primary": primary, "selected": selected,
        "knn_threshold": knn_threshold, "leverage_threshold": leverage_threshold,
        "similarity_threshold": similarity_threshold, "train_knn_group_aware": train_knn,
        "train_sim_group_aware": train_sim, "performance": performance,
    }
    _figures(d, ctx)

    print("Applicability-domain coverage (no model refitting, no Y-randomization):")
    print(coverage[["method", "threshold", "n_inside_ad", "n_outside_ad", "fraction_inside_ad"]]
          .round(4).to_string(index=False))
    print(f"\nPrimary model (lowest Stage 07 CV RMSE): {primary}")
    print(pd.DataFrame(delta_rows).query("model == @primary")
          [["method", "n_inside", "n_outside", "rmse_inside", "rmse_outside",
            "rmse_outside_minus_inside", "ci_low", "ci_high"]].round(3).to_string(index=False))
    print(f"\nLeverage design rank {design_rank} of {n_feat + 1}; n_train = {n_train}.")
    print(f"kNN threshold {knn_threshold:.3f} (group-aware) vs {knn_threshold_naive:.3f} (self-excluded only).")

    outputs = [d / f"tables/{n}.csv" for n in (
        "test_ad_flags_all_methods", "ad_performance_inside_outside_all_methods",
        "ad_rmse_difference_outside_minus_inside", "knn_domain_model_performance",
        "ad_method_coverage", "ad_score_vs_error_spearman", "error_by_knn_distance_quintile",
        "knn_threshold_sensitivity")]
    outputs.append(d / "models/ad_statistics_all_methods.joblib")
    figure_stems = ["ad_knn_distance", "ad_bounding_box_violations", "ad_similarity_distribution",
                    "ad_method_coverage", "ad_rmse_inside_outside_primary_model"]
    figure_stems += [f"ad_residual_{m}_{_short(n)}" for n in MODELS for m in METHOD_ORDER]
    outputs += [d / f"figures/{s}.{e}" for s in figure_stems for e in ("png", "pdf")]
    write_manifest(
        STAGE,
        {
            "methods": [METHOD_LABEL[m] for m in METHOD_ORDER],
            "features": n_feat, "n_train": n_train, "n_test_scored": n_test,
            "n_test_excluded_in_stage08": int(n_excluded),
            "knn_k": k, "knn_threshold_rule": rules["knn"],
            "leverage_threshold_rule": rules["leverage"],
            "similarity_threshold_rule": rules["similarity"],
            "training_reference": "group-aware (neighbours from the molecule's own scaffold group ignored)",
            "primary_model_by_stage07_cv": primary,
            "residual_scale": "Stage 07 pooled out-of-fold RMSE of each model",
            "bootstrap": f"{int(n_bootstrap)} scaffold-cluster resamples of the locked test set",
            "models_evaluated_from_saved_predictions": report_models,
            "models_refit": False, "y_randomization_run": False,
            "test_used_only_for_domain_coverage_and_performance": True,
        },
        [f_path, t_path, s_path, ids_path, test_ids_file, TUNING_JSON, PREDICTION_FILE,
         tables03 / "all_features.parquet", smiles_path],
        outputs,
    )
    return coverage, performance, knn_performance, test_ad


if __name__ == "__main__":
    run()
