"""IPF (Iterative Proportional Fitting) baseline synthesizer.

IPF is the traditional method for synthetic population generation.
It works by:
    1. Building a seed contingency table from observed microdata
    2. Fitting the table to target marginal distributions via IPF
    3. Converting the fitted table back to individual records

This is the "gold standard" for marginal consistency — by construction,
IPF output matches the target marginals exactly.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from ipfn import ipfn

from config import CATEGORICAL_ATTRIBUTES, CENSUS_ATTRIBUTES, RANDOM_SEED


class IPFSynthesizer:
    """IPF-based synthetic population generator."""

    def __init__(
        self,
        attributes: list[str] | None = None,
        category_map: dict[str, list[str]] | None = None,
        seed: int = RANDOM_SEED,
    ):
        self.attributes = attributes or list(CENSUS_ATTRIBUTES)
        self.category_map = category_map or dict(CATEGORICAL_ATTRIBUTES)
        self.seed = seed
        self._seed_table = None
        self._dims = None

    def fit(self, train_df: pd.DataFrame) -> "IPFSynthesizer":
        """Build seed contingency table from the latest year in training data."""
        # Use the most recent year as seed
        if "year" in train_df.columns:
            latest = train_df["year"].max()
            seed_df = train_df[train_df["year"] == latest].copy()
        else:
            seed_df = train_df.copy()

        self._dims = [len(self.category_map[a]) for a in self.attributes]

        # Count occurrences in each cell of the 5D contingency table
        table = np.zeros(self._dims, dtype=np.float64)
        for _, row in seed_df.iterrows():
            indices = []
            valid = True
            for attr in self.attributes:
                cats = self.category_map[attr]
                if str(row[attr]) in cats:
                    indices.append(cats.index(str(row[attr])))
                else:
                    valid = False
                    break
            if valid:
                table[tuple(indices)] += 1

        # Laplace smoothing to avoid zero cells
        table += 0.5
        self._seed_table = table
        return self

    def generate(
        self,
        n_samples: int,
        target_marginals: dict[str, pd.Series] | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic population via IPF fitting + expansion.

        Args:
            n_samples: Number of records to generate.
            target_marginals: Dict of attribute -> proportions Series.
                If None, uses seed distribution directly.
        """
        if self._seed_table is None:
            raise RuntimeError("Must call fit() first.")

        if target_marginals is not None:
            fitted = self._run_ipf(n_samples, target_marginals)
        else:
            fitted = self._seed_table * (n_samples / self._seed_table.sum())

        int_table = self._trs_integerize(fitted, n_samples)
        return self._expand_table(int_table)

    def _run_ipf(self, n_samples, target_marginals):
        """Run IPF to fit seed table to target marginals."""
        seed_scaled = self._seed_table * (n_samples / self._seed_table.sum())

        aggregates, dimensions = [], []
        for i, attr in enumerate(self.attributes):
            if attr not in target_marginals:
                continue
            target = target_marginals[attr]
            cats = self.category_map[attr]
            counts = np.array([target.get(c, 0.0) for c in cats]) * n_samples
            aggregates.append(counts)
            dimensions.append([i])

        ipf_obj = ipfn.ipfn(seed_scaled, aggregates, dimensions,
                            convergence_rate=1e-6, max_iteration=500)
        return ipf_obj.iteration()

    def _trs_integerize(self, table, n_samples):
        """Truncate-Replicate-Sample: convert float table to integer counts."""
        rng = np.random.default_rng(self.seed)
        floored = np.floor(table).astype(np.int64)
        residuals = table - floored
        deficit = n_samples - floored.sum()

        if deficit > 0:
            flat = residuals.ravel()
            probs = flat / flat.sum() if flat.sum() > 0 else np.ones_like(flat) / flat.size
            chosen = rng.choice(flat.size, size=int(deficit), replace=True, p=probs)
            flat_int = floored.ravel()
            for idx in chosen:
                flat_int[idx] += 1
            floored = flat_int.reshape(table.shape)

        return floored

    def _expand_table(self, int_table):
        """Convert integer contingency table to individual records."""
        records = []
        for indices in product(*(range(d) for d in int_table.shape)):
            count = int(int_table[indices])
            if count <= 0:
                continue
            record = {self.attributes[i]: self.category_map[self.attributes[i]][idx]
                      for i, idx in enumerate(indices)}
            records.extend([record.copy() for _ in range(count)])

        df = pd.DataFrame(records)
        return df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

    @staticmethod
    def compute_marginals(df, attributes=None, category_map=None):
        """Extract marginal distributions from a DataFrame."""
        attributes = attributes or list(CENSUS_ATTRIBUTES)
        category_map = category_map or dict(CATEGORICAL_ATTRIBUTES)
        n = len(df)
        marginals = {}
        for attr in attributes:
            cats = category_map[attr]
            counts = df[attr].value_counts()
            marginals[attr] = pd.Series([counts.get(c, 0) / n for c in cats],
                                        index=cats, name=attr)
        return marginals
