"""Configuration for synthetic population generation tutorial.

Defines the census attribute schema, category values, and structural
zero rules for the Daejeon (South Korea) census microdata.
"""

# ── Census attributes (5 demographic variables) ─────────────────────────

CENSUS_ATTRIBUTES = ["sex", "age_group", "edu", "occupation", "marital_status"]

CATEGORICAL_ATTRIBUTES = {
    "sex": ["Male", "Female"],
    "age_group": [
        "0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34",
        "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69",
        "70-74", "75-79", "80-84", "85+",
    ],
    "edu": [
        "No schooling", "Elementary school", "Middle school",
        "High school", "2-year college", "4-year university",
        "Graduate school",
    ],
    "occupation": [
        "Not economically active", "Managers", "Professionals", "Clerks",
        "Service workers", "Agriculture/Forestry/Fishery", "Blue collar",
        "Armed forces",
    ],
    "marital_status": ["Never married", "Married", "Widowed", "Divorced"],
}

# ── Structural zero rules ────────────────────────────────────────────────
# Impossible attribute combinations (e.g., a 3-year-old with a PhD).
# Used to filter invalid synthetic records.

STRUCTURAL_ZERO_RULES = [
    {
        "rule_id": "Z1",
        "description": "Age 0-4 must have Education = No schooling",
        "age_group_max_idx": 0,  # 0-based index into age_group list
        "valid_edu": ["No schooling"],
    },
    {
        "rule_id": "Z2",
        "description": "Age 5-9: Education in {No schooling, Elementary school}",
        "age_group_max_idx": 1,
        "valid_edu": ["No schooling", "Elementary school"],
    },
    {
        "rule_id": "Z3",
        "description": "Age 10-14: Education in {No schooling, Elementary, Middle}",
        "age_group_max_idx": 2,
        "valid_edu": ["No schooling", "Elementary school", "Middle school"],
    },
    {
        "rule_id": "Z4",
        "description": "Age 0-14 must be Never married",
        "age_group_max_idx": 2,
        "valid_marital_status": ["Never married"],
    },
    {
        "rule_id": "Z5",
        "description": "Age 0-14 must be Not economically active",
        "age_group_max_idx": 2,
        "valid_occupation": ["Not economically active"],
    },
]

RANDOM_SEED = 42
