"""
Downloads the two real datasets this project relies on, from verified
GitHub mirrors. Run: python -m src.download_data
"""
import sys
from pathlib import Path
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def _download(url: str, dest: Path, label: str) -> None:
    if dest.exists():
        print(f"[skip] {label} already at {dest}")
        return
    print(f"[fetch] {label} <- {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"[ok]   {label}: {resp.text.count(chr(10))} lines written to {dest}")


def download_liar() -> None:
    for split, fname in config.LIAR_FILES.items():
        _download(f"{config.LIAR_BASE_URL}/{fname}",
                   config.DATA_RAW / f"liar_{split}.tsv", f"LIAR {split} split")


def download_mbfc() -> None:
    _download(config.MBFC_URL, config.DATA_RAW / "mbfc_corpus.tsv",
               "MBFC source credibility corpus")


def main() -> None:
    print("Downloading real, publicly available datasets ...\n")
    download_liar()
    download_mbfc()
    print("\nAll datasets present in", config.DATA_RAW)


if __name__ == "__main__":
    main()
