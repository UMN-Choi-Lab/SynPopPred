"""Record serializer for LLM-based population synthesis.

Converts structured census records to/from natural language text using
the GReaT (Generation of Realistic Tabular data) approach.

Example serialized record:
    "In 2020, a resident of Daejeon was recorded: Sex is Female,
     Age Group is 30-34, Education is High school, Occupation is Clerks,
     Marital Status is Married"
"""

import random

import pandas as pd

from config import RANDOM_SEED

# ── Attribute name mapping ───────────────────────────────────────────────

ATTRIBUTE_DISPLAY_NAMES = {
    "sex": "Sex",
    "age_group": "Age Group",
    "edu": "Education",
    "occupation": "Occupation",
    "marital_status": "Marital Status",
}

DISPLAY_TO_INTERNAL = {v: k for k, v in ATTRIBUTE_DISPLAY_NAMES.items()}


# ── Serialization ────────────────────────────────────────────────────────

def serialize_record(
    record: dict,
    year: int | None = None,
    region: str = "Daejeon",
    permute: bool = True,
    rng: random.Random | None = None,
) -> str:
    """Convert a single census record dict to natural language text.

    Args:
        record: Dict mapping column names to values (e.g., {"sex": "Male"}).
        year: Census year for temporal prefix. None = no prefix.
        permute: Randomly shuffle attribute order (for training augmentation).
        rng: Random generator for reproducible permutation.

    Returns:
        Serialized text string.
    """
    if rng is None:
        rng = random.Random(RANDOM_SEED)

    # Build (display_name, value) pairs, skipping metadata columns
    items = []
    for col, value in record.items():
        if col in ("year", "region"):
            continue
        display = ATTRIBUTE_DISPLAY_NAMES.get(col, col)
        items.append((display, str(value)))

    if permute:
        rng.shuffle(items)

    body = ", ".join(f"{name} is {value}" for name, value in items)

    if year is not None:
        return f"In {year}, a resident of {region} was recorded: {body}"
    return body


def serialize_dataframe(
    df: pd.DataFrame,
    year_col: str = "year",
    region: str = "Daejeon",
    permute: bool = True,
    seed: int = RANDOM_SEED,
) -> list[str]:
    """Serialize an entire DataFrame to a list of text sequences."""
    rng = random.Random(seed)
    sequences = []
    attr_cols = [c for c in df.columns if c != year_col]

    for _, row in df.iterrows():
        year = int(row[year_col]) if year_col in df.columns else None
        record = {col: row[col] for col in attr_cols if pd.notna(row[col])}
        sequences.append(serialize_record(record, year=year, region=region,
                                          permute=permute, rng=rng))
    return sequences


# ── Parsing (deserialization) ────────────────────────────────────────────

def parse_record(text: str) -> dict:
    """Parse serialized text back into a record dictionary.

    Handles prefixed format ("In 2020, a resident of Daejeon was recorded: ...")
    and plain format ("Sex is Male, Age Group is 30-34, ...").

    Returns:
        Dict mapping internal attribute names to values.
    """
    record = {}
    body = text

    # Extract year and body from prefix
    if text.startswith("In "):
        parts = text.split(",", 1)
        year_str = parts[0].replace("In ", "").strip()
        try:
            record["year"] = str(int(year_str))
        except ValueError:
            pass

        if "was recorded: " in text:
            body = text.split("was recorded: ", 1)[1]
        elif ": " in text:
            body = text.split(": ", 1)[1]

    # Truncate at second record boundary (multi-record contamination)
    import re
    second = re.search(r'(?:,\s*)?In \d{4}[,\s]', body)
    if second:
        body = body[:second.start()]

    body = body.rstrip(". \t\n")

    # Parse "Attribute is Value" pairs
    for pair in body.split(", "):
        if " is " in pair:
            name, value = pair.split(" is ", 1)
            name = name.strip()
            value = value.strip().rstrip(".")
            internal = DISPLAY_TO_INTERNAL.get(name, name.lower().replace(" ", "_"))
            if internal not in record:  # first-wins
                record[internal] = value

    return record


def build_generation_prefix(
    year: int,
    region: str = "Daejeon",
    conditions: dict | None = None,
) -> str:
    """Build a prefix for conditional LLM generation.

    Example:
        >>> build_generation_prefix(2035, conditions={"sex": "Male"})
        'In 2035, a resident of Daejeon was recorded: Sex is Male, '
    """
    prefix = f"In {year}, a resident of {region} was recorded: "
    if conditions:
        parts = [f"{ATTRIBUTE_DISPLAY_NAMES.get(k, k)} is {v}"
                 for k, v in conditions.items()]
        prefix += ", ".join(parts) + ", "
    return prefix
