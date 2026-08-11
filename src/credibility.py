"""
Maps a news source's domain to a credibility score using the MBFC-derived
corpus.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import load_mbfc, normalize_domain


class CredibilityScorer:
    def __init__(self):
        mbfc = load_mbfc()
        self._lookup = dict(zip(mbfc["domain"], mbfc["fact"]))

    def label(self, domain_or_url: str) -> str:
        return self._lookup.get(normalize_domain(domain_or_url), "unrated")

    def score(self, domain_or_url: str) -> float:
        label = self.label(domain_or_url)
        if label == "unrated":
            return config.UNKNOWN_SOURCE_SCORE
        return config.CREDIBILITY_SCORE_MAP[label]

    def is_rated(self, domain_or_url: str) -> bool:
        return normalize_domain(domain_or_url) in self._lookup

    def coverage(self) -> int:
        return len(self._lookup)


if __name__ == "__main__":
    cs = CredibilityScorer()
    print("Rated domains loaded:", cs.coverage())
    for d in ["cnn.com", "reuters.com", "some-random-blog.xyz"]:
        print(f"  {d:30s} -> label={cs.label(d):8s} score={cs.score(d)}")
