"""
Loads the raw TSVs into clean pandas DataFrames.
"""
import sys
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def normalize_domain(url_or_domain: str) -> str:
    if not url_or_domain:
        return ""
    s = url_or_domain.strip().lower()
    if "://" in s:
        s = urlparse(s).netloc
    if s.startswith("www."):
        s = s[4:]
    return s.split("/")[0]


def load_liar(split: str) -> pd.DataFrame:
    path = config.DATA_RAW / f"liar_{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.download_data")
    df = pd.read_csv(path, sep="\t", header=None, names=config.LIAR_COLUMNS)
    df["label_6way"] = df["label"].str.strip().str.lower()
    df["label_3way"] = df["label_6way"].map(config.LABEL_MAP_3WAY)
    df = df.dropna(subset=["label_3way", "statement"]).reset_index(drop=True)
    return df


def load_liar_all() -> pd.DataFrame:
    frames = []
    for split in ["train", "valid", "test"]:
        d = load_liar(split)
        d["split"] = split
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_mbfc() -> pd.DataFrame:
    path = config.DATA_RAW / "mbfc_corpus.tsv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.download_data")
    df = pd.read_csv(path, sep="\t")
    df["domain"] = df["source_url_normalized"].apply(normalize_domain)
    df["fact"] = df["fact"].str.strip().str.lower()
    df = df[df["fact"].isin(config.CREDIBILITY_SCORE_MAP.keys())]
    df = df.drop_duplicates(subset="domain").reset_index(drop=True)
    return df[["domain", "fact", "bias"]]


if __name__ == "__main__":
    liar = load_liar_all()
    mbfc = load_mbfc()
    print("LIAR claims loaded:", len(liar))
    print(liar["label_3way"].value_counts())
    print("\nMBFC rated domains loaded:", len(mbfc))
    print(mbfc["fact"].value_counts())
