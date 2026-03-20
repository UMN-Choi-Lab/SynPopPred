"""Evaluation metrics for synthetic population quality.

Two key metrics:
    - SRMSE (Standardized RMSE): Measures marginal distribution accuracy.
      Compares the frequency of each category (e.g., % Male vs % Female)
      between synthetic and real data.

    - JSD (Jensen-Shannon Divergence): Measures joint distribution accuracy.
      Compares the joint probability table across multiple attributes.
      This is the harder metric — it captures attribute interactions.

Both metrics: lower = better, 0 = perfect match.
"""

from itertools import combinations, product as iterproduct

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from config import CATEGORICAL_ATTRIBUTES, CENSUS_ATTRIBUTES


def _get_distribution(series: pd.Series, categories: list[str]) -> np.ndarray:
    """Compute normalized frequency distribution aligned to category list."""
    counts = series.value_counts().reindex(categories, fill_value=0)
    total = counts.sum()
    if total == 0:
        return np.zeros(len(counts))
    return (counts / total).values.astype(np.float64)


# ── SRMSE (marginal accuracy) ────────────────────────────────────────────

def compute_srmse(p_syn: np.ndarray, p_real: np.ndarray) -> float:
    """Standardized RMSE between two distributions.

    Normalized by 1/C (uniform probability), so values are comparable
    across attributes with different numbers of categories.
    """
    C = len(p_real)
    if C == 0:
        return 0.0
    return float(np.sqrt(np.mean((p_syn - p_real) ** 2)) / (1.0 / C))


def compute_all_srmse(
    syn_df: pd.DataFrame,
    real_df: pd.DataFrame,
    attributes: list[str] | None = None,
    category_map: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    """Compute SRMSE for each attribute's marginal distribution."""
    attributes = attributes or list(CENSUS_ATTRIBUTES)
    category_map = category_map or dict(CATEGORICAL_ATTRIBUTES)

    results = {}
    for attr in attributes:
        cats = category_map.get(attr, sorted(
            set(syn_df[attr].unique()) | set(real_df[attr].unique())))
        p_syn = _get_distribution(syn_df[attr], cats)
        p_real = _get_distribution(real_df[attr], cats)
        results[attr] = compute_srmse(p_syn, p_real)
    return results


# ── JSD (joint distribution accuracy) ────────────────────────────────────

def compute_jsd(p_syn: np.ndarray, p_real: np.ndarray) -> float:
    """Jensen-Shannon Divergence (squared) between two distributions."""
    eps = 1e-12
    p_syn = np.maximum(p_syn, eps)
    p_real = np.maximum(p_real, eps)
    p_syn /= p_syn.sum()
    p_real /= p_real.sum()
    return float(jensenshannon(p_syn, p_real, base=2.0) ** 2)


def _joint_distribution(
    df: pd.DataFrame,
    attrs: list[str],
    category_map: dict[str, list[str]],
) -> np.ndarray:
    """Compute flattened joint probability vector over k attributes."""
    cat_lists = [category_map.get(a, sorted(df[a].unique())) for a in attrs]
    counts = df.groupby(attrs, observed=False).size()

    tuples = list(iterproduct(*cat_lists))
    if len(attrs) == 1:
        full_index = pd.Index([t[0] for t in tuples], name=attrs[0])
    else:
        full_index = pd.MultiIndex.from_tuples(tuples, names=attrs)

    counts = counts.reindex(full_index, fill_value=0)
    total = counts.sum()
    if total == 0:
        return np.zeros(len(counts))
    return (counts / total).values.astype(np.float64)


def compute_pairwise_jsd(
    syn_df: pd.DataFrame,
    real_df: pd.DataFrame,
    attributes: list[str] | None = None,
    category_map: dict[str, list[str]] | None = None,
) -> dict[tuple[str, str], float]:
    """JSD for all pairs of attributes (captures 2-way interactions)."""
    attributes = attributes or list(CENSUS_ATTRIBUTES)
    category_map = category_map or dict(CATEGORICAL_ATTRIBUTES)

    results = {}
    for a1, a2 in combinations(attributes, 2):
        p_syn = _joint_distribution(syn_df, [a1, a2], category_map)
        p_real = _joint_distribution(real_df, [a1, a2], category_map)
        results[(a1, a2)] = compute_jsd(p_syn, p_real)
    return results


def compute_full_joint_jsd(
    syn_df: pd.DataFrame,
    real_df: pd.DataFrame,
    attributes: list[str] | None = None,
    category_map: dict[str, list[str]] | None = None,
) -> float:
    """JSD over the full 5-way joint distribution (8,064 cells).

    This is the single most demanding fidelity metric — it captures
    ALL interaction orders simultaneously.
    """
    attributes = attributes or list(CENSUS_ATTRIBUTES)
    category_map = category_map or dict(CATEGORICAL_ATTRIBUTES)

    p_syn = _joint_distribution(syn_df, attributes, category_map)
    p_real = _joint_distribution(real_df, attributes, category_map)
    return compute_jsd(p_syn, p_real)


# ── IPF post-processing reweighting ─────────────────────────────────────

def ipf_reweight(
    syn_df: pd.DataFrame,
    target_marginals: dict[str, pd.Series],
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 42,
) -> pd.DataFrame:
    """Reweight synthetic records to match target marginals (raking).

    Adjusts sample weights iteratively, then resamples. This can be
    applied as post-processing to CTGAN or PopLLM output to improve
    marginal consistency.
    """
    if syn_df.empty:
        return syn_df.copy()

    n = len(syn_df)
    weights = np.ones(n, dtype=np.float64)

    for _ in range(max_iter):
        max_change = 0.0
        for attr, target in target_marginals.items():
            col = syn_df[attr].values
            for cat, target_prop in target.items():
                if target_prop <= 0:
                    continue
                mask = col == cat
                current = weights[mask].sum()
                if current <= 0:
                    continue
                adj = (target_prop * weights.sum()) / current
                old = weights[mask].copy()
                weights[mask] *= adj
                max_change = max(max_change, np.abs(weights[mask] - old).max())

        if max_change < tol:
            break

    probs = weights / weights.sum()
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=n, replace=True, p=probs)
    return syn_df.iloc[indices].reset_index(drop=True)


# ── Convenience: full evaluation ─────────────────────────────────────────

def evaluate(syn_df: pd.DataFrame, real_df: pd.DataFrame) -> dict:
    """Run the full evaluation suite and return a results dict."""
    srmse = compute_all_srmse(syn_df, real_df)
    pairwise = compute_pairwise_jsd(syn_df, real_df)
    full_jsd = compute_full_joint_jsd(syn_df, real_df)

    return {
        "srmse_per_attr": srmse,
        "srmse_avg": np.mean(list(srmse.values())),
        "jsd_pairwise_avg": np.mean(list(pairwise.values())),
        "jsd_full_joint": full_jsd,
    }
