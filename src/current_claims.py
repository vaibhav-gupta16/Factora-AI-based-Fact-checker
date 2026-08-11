"""
LIAR's claims (2007-2016, no date column) cannot be evidence-checked by
live news retrieval — current news search naturally favours recent
coverage and simply does not discuss most old, no-longer-newsworthy
political statements. This is a structural mismatch, not a retrieval bug
(see config.py's note on LIAR_COLUMNS).

This module provides a small alternative claim set for exercising the
LIVE retrieval + verdict + credibility-weighting pipeline end to end,
where "unverifiable" genuinely means "no evidence," not "too old to
possibly be covered."

Every claim below is drawn directly from a real, named fact-checking
source found via live web search on 2026-08-10, not invented. Sources
are cited in each entry's "source" field. This is a STARTER set (3
claims) meant to prove the pipeline works — for a real evaluation
sample, add your own current claims via a CSV (see load_from_csv below).
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# Ground-truth label uses the same 3-way scale as LIAR (false / mixed / true)
# so it plugs into the same evaluation code (src/evaluate.py) unchanged.
STARTER_CLAIMS = [
    {
        "id": "current_001",
        "statement": (
            "Earth will lose its gravity for seven seconds on August 12, 2026, "
            "according to a leaked NASA document called 'Project Anchor'."
        ),
        "label_3way": "false",
        "source": (
            "NASA officially denied this claim, quoted directly in a Snopes "
            "fact-check: 'Earth's gravity is determined by its mass... The only "
            "way for Earth to lose gravity would be for the planet itself to "
            "lose mass.' Reported by newskarnataka.com, citing Snopes, Jan 2026."
        ),
    },
    {
        "id": "current_002",
        "statement": (
            "A solar eclipse can cause a temporary loss of Earth's gravitational force."
        ),
        "label_3way": "false",
        "source": (
            "NASA clarified that while the Sun and Moon influence tides, they do "
            "not alter Earth's overall gravitational force, and gravity cannot "
            "briefly pause and restart. Same Snopes/NASA fact-check as above."
        ),
    },
    {
        "id": "current_003",
        "statement": (
            "Amazon canceled development of its Scout delivery robots."
        ),
        "label_3way": "true",
        "source": (
            "Confirmed in a Snopes fact-check archive entry (Emery Winter, "
            "Aug 6 2026): 'Amazon canceled development of its Scout delivery "
            "robots in 2023.'"
        ),
    },
]


def load_starter_claims() -> pd.DataFrame:
    """Returns the small, verified starter set as a DataFrame matching
    the shape src/pipeline.py expects (id, statement, label_3way)."""
    df = pd.DataFrame(STARTER_CLAIMS)
    df["label_6way"] = df["label_3way"]  # no finer-grained label available
    df["speaker"] = "n/a"
    df["subject"] = "n/a"
    return df


def load_from_csv(path: str) -> pd.DataFrame:
    """
    Load your own current claims from a CSV with columns:
    id, statement, label_3way (values: false / mixed / true)

    Add real, current claims you can verify yourself — from a
    fact-checking site, a news article, or a claim you already know the
    outcome of — so the live pipeline has something genuinely
    checkable against current news coverage.
    """
    df = pd.read_csv(path)
    required = {"id", "statement", "label_3way"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df["label_6way"] = df["label_3way"]
    if "speaker" not in df.columns:
        df["speaker"] = "n/a"
    if "subject" not in df.columns:
        df["subject"] = "n/a"
    return df


if __name__ == "__main__":
    df = load_starter_claims()
    print(f"Starter claims: {len(df)}")
    for _, row in df.iterrows():
        print(f"  [{row['label_3way']}] {row['statement'][:70]}...")
        print(f"    source: {row['source'][:90]}...")
