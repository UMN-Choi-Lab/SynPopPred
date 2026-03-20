"""Era context narratives for PopLLM framework.

Provides per-year Korean demographic narratives that capture each era's
defining characteristics. Used as semantic context prefixes during both
training and generation to ground the LLM's understanding of temporal
demographic shifts.

Example (prepended to each training record):
    "[Context: In 2020, South Korea's TFR hit a record low of 0.84. ...]
     In 2020, a resident of Daejeon was recorded: Sex is Female, ..."

This is the key differentiator of PopLLM vs. standard tabular LLMs:
the model receives era-specific context about demographic trends (TFR,
aging, education, employment) that helps it extrapolate to future years.

Sources: KOSTAT historical statistics, KOSIS data portal
"""

# ── Per-year era narratives ──────────────────────────────────────────────

ERA_NARRATIVES: dict[int, str] = {
    # Historical census years (1990-2020)
    1990: (
        "In 1990, South Korea had a TFR of 1.57 and rapid industrialization. "
        "The elderly (65+) were only 5.1% of the population. "
        "University enrollment was expanding but most adults had high school education or less."
    ),
    1995: (
        "In 1995, South Korea's TFR dropped to 1.63 amid economic growth. "
        "The elderly share rose to 5.9%. Manufacturing employment peaked "
        "and the service sector expanded rapidly."
    ),
    2000: (
        "In 2000, South Korea recovered from the 1997 financial crisis. "
        "TFR fell to 1.47. The elderly were 7.2% of the population. "
        "University education became widespread among young adults."
    ),
    2005: (
        "In 2005, South Korea's TFR plunged to 1.08, one of the world's lowest. "
        "The elderly reached 9.1%. Dual-income households increased "
        "and delayed marriage became common among young adults."
    ),
    2010: (
        "In 2010, South Korea's TFR was 1.23 with an aging society. "
        "The elderly constituted 11.0%. Professional and service occupations "
        "grew while agriculture continued declining."
    ),
    2015: (
        "In 2015, South Korea's TFR was 1.24 and the elderly reached 13.1%. "
        "Graduate education expanded significantly. "
        "Youth unemployment rose despite high educational attainment."
    ),
    2020: (
        "In 2020, South Korea's TFR hit a record low of 0.84. "
        "The elderly reached 15.7% of the population. "
        "Never-married rates surged among 30-somethings and divorce rates stabilized."
    ),
    # Future projection years (from KOSTAT medium-variant)
    2025: (
        "By 2025, South Korea's TFR is projected at 0.75. "
        "The elderly exceed 20% of the population, entering a super-aged society. "
        "Single-person households surpass 35% and educational attainment remains very high."
    ),
    2030: (
        "By 2030, South Korea's TFR is projected at 0.80 with 25.5% elderly. "
        "The working-age population shrinks significantly. "
        "Most young adults hold university degrees and professional jobs dominate."
    ),
    2035: (
        "By 2035, South Korea's population begins declining with TFR at 0.80. "
        "The elderly reach 30.1%. Single-person households approach 40%. "
        "Labor force participation of elderly increases substantially."
    ),
    2040: (
        "By 2040, South Korea's population has declined to 47.7 million with 34.4% elderly. "
        "The child population (0-14) falls below 8%. "
        "Agricultural workers are almost entirely elderly."
    ),
}


def build_era_prefix(year: int, region: str = "Daejeon") -> str:
    """Build an era-context prefix for training or generation.

    Combines the year's era narrative with the standard record prefix.
    If the year has no narrative, returns the standard prefix only.

    Args:
        year: Census or target year.
        region: Region name.

    Returns:
        Prefix string like:
        "[Context: In 2020, South Korea's TFR hit ...] In 2020, a resident of Daejeon was recorded: "
    """
    narrative = ERA_NARRATIVES.get(year, "")
    if narrative:
        return f"[Context: {narrative}] In {year}, a resident of {region} was recorded: "
    return f"In {year}, a resident of {region} was recorded: "
