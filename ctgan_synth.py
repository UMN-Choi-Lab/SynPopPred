"""CTGAN (Conditional Tabular GAN) baseline synthesizer.

CTGAN learns the joint distribution of census attributes using a
conditional GAN architecture. It is a popular deep generative model
for tabular data synthesis.

This wraps the SDV (Synthetic Data Vault) library's CTGANSynthesizer.
"""

from __future__ import annotations

import logging

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sdv.single_table import CTGANSynthesizer

from config import CENSUS_ATTRIBUTES, RANDOM_SEED

logger = logging.getLogger(__name__)


class CTGANWrapper:
    """CTGAN-based synthetic population generator.

    Trains on census data with 'year' as a categorical feature,
    enabling year-conditioned generation.
    """

    def __init__(
        self,
        epochs: int = 300,
        batch_size: int = 500,
        generator_dim: tuple[int, int] = (256, 256),
        discriminator_dim: tuple[int, int] = (256, 256),
        seed: int = RANDOM_SEED,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.seed = seed
        self._model = None

    def fit(self, train_df: pd.DataFrame) -> "CTGANWrapper":
        """Train CTGAN on census data (year included as a feature)."""
        cols = ["year"] + [a for a in CENSUS_ATTRIBUTES if a in train_df.columns]
        df = train_df[cols].copy()

        # SDV requires string columns for categorical data
        for col in df.columns:
            df[col] = df[col].astype(str)

        # Build metadata: all columns are categorical
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        for col in df.columns:
            metadata.update_column(col, sdtype="categorical")

        self._model = CTGANSynthesizer(
            metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            verbose=True,
        )
        self._model.fit(df)
        return self

    def generate(self, n_samples: int, target_year: int | None = None) -> pd.DataFrame:
        """Generate synthetic records, optionally conditioned on year."""
        if self._model is None:
            raise RuntimeError("Must call fit() first.")

        if target_year is not None:
            condition = Condition(num_rows=n_samples,
                                 column_values={"year": str(target_year)})
            try:
                syn_df = self._model.sample_from_conditions(
                    conditions=[condition], max_tries_per_batch=200)
            except Exception as e:
                logger.warning("Conditional sampling failed: %s. Using unconditional.", e)
                syn_df = self._model.sample(n_samples)
        else:
            syn_df = self._model.sample(n_samples)

        if "year" in syn_df.columns:
            syn_df = syn_df.drop(columns=["year"])

        out_cols = [c for c in CENSUS_ATTRIBUTES if c in syn_df.columns]
        return syn_df[out_cols].reset_index(drop=True)
