"""
Deterministic confidence scoring — computed from measurable evidence
signals, not invented by the LLM. See generate_verdict() in verdict.py
for how this is wired in.

Formula:
  confidence = 35% × avg_credibility
             + 25% × scaled_relevance
             + 20% × corroboration
             + 20% × source_diversity

Each component is purely deterministic:
  • credibility  — MBFC-derived score per source domain (0–1)
  • relevance    — TF-IDF cosine similarity, claim vs article (0–1)
  • corroboration — min(1, n_articles / SATURATION)
  • diversity     — min(1, n_distinct_publishers / DIVERSITY_SATURATION)
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

W_CREDIBILITY = 0.35
W_RELEVANCE = 0.25
W_CORROBORATION = 0.20
W_SOURCE_DIVERSITY = 0.20
CORROBORATION_SATURATION = 4

# Aggregator domains: articles from these have the actual publisher in
# source_name, not in domain.  Diversity scoring uses source_name for
# them so each real publisher is counted separately.
_AGGREGATOR_DOMAINS = frozenset({"news.google.com", "news.google.co.uk"})


def _corroboration_score(n_supporting: int) -> float:
    if n_supporting <= 0:
        return 0.0
    return min(1.0, n_supporting / CORROBORATION_SATURATION)


def _source_diversity_score(supporting_evidence: list[dict]) -> float:
    """How many *distinct* publishers back the verdict.

    3 articles from The Guardian + BBC + Reuters = diversity 1.0.
    3 articles all from The Guardian = diversity 0.33.

    For articles sourced via an aggregator (Google News RSS), the domain
    field is the aggregator's domain, so we fall back to source_name to
    identify the actual publisher."""
    if not supporting_evidence:
        return 0.0
    unique_sources: set[str] = set()
    for e in supporting_evidence:
        domain = (e.get("domain") or "").lower().strip()
        source_name = (e.get("source_name") or "").lower().strip()
        # For aggregator links, use the real publisher name
        if domain in _AGGREGATOR_DOMAINS or not domain:
            identifier = source_name or domain
        else:
            identifier = domain
        if identifier:
            unique_sources.add(identifier)
    return min(1.0, len(unique_sources) / config.DIVERSITY_SATURATION)


def compute_confidence(supporting_evidence: list[dict]) -> float:
    if not supporting_evidence:
        return 0.0
    n = len(supporting_evidence)
    avg_credibility = sum(e.get("credibility_score", 0.0) for e in supporting_evidence) / n
    avg_relevance = sum(e.get("relevance", 0.0) for e in supporting_evidence) / n
    scaled_relevance = min(1.0, avg_relevance / 0.4)
    corroboration = _corroboration_score(n)
    diversity = _source_diversity_score(supporting_evidence)
    confidence = (W_CREDIBILITY * avg_credibility
                  + W_RELEVANCE * scaled_relevance
                  + W_CORROBORATION * corroboration
                  + W_SOURCE_DIVERSITY * diversity)
    return round(max(0.0, min(1.0, confidence)), 2)


def confidence_breakdown(supporting_evidence: list[dict]) -> dict:
    if not supporting_evidence:
        return {"n_supporting_sources": 0, "n_distinct_publishers": 0,
                "avg_credibility": 0.0, "avg_relevance": 0.0,
                "corroboration": 0.0, "source_diversity": 0.0,
                "confidence": 0.0}
    n = len(supporting_evidence)
    avg_credibility = round(sum(e.get("credibility_score", 0.0) for e in supporting_evidence) / n, 3)
    avg_relevance = round(sum(e.get("relevance", 0.0) for e in supporting_evidence) / n, 3)
    diversity = _source_diversity_score(supporting_evidence)
    # Count distinct publishers for the breakdown display
    unique_sources: set[str] = set()
    for e in supporting_evidence:
        domain = (e.get("domain") or "").lower().strip()
        source_name = (e.get("source_name") or "").lower().strip()
        if domain in _AGGREGATOR_DOMAINS or not domain:
            identifier = source_name or domain
        else:
            identifier = domain
        if identifier:
            unique_sources.add(identifier)
    return {
        "n_supporting_sources": n,
        "n_distinct_publishers": len(unique_sources),
        "avg_credibility": avg_credibility,
        "avg_relevance": avg_relevance,
        "corroboration": round(_corroboration_score(n), 3),
        "source_diversity": round(diversity, 3),
        "confidence": compute_confidence(supporting_evidence),
    }


if __name__ == "__main__":
    strong = [{"credibility_score": 0.9, "relevance": 0.42,
               "domain": "reuters.com", "source_name": "Reuters"},
              {"credibility_score": 0.9, "relevance": 0.38,
               "domain": "bbc.com", "source_name": "BBC News"}]
    weak = [{"credibility_score": 0.2, "relevance": 0.10,
             "domain": "example.com", "source_name": "Example Blog"}]
    same_source = [{"credibility_score": 0.8, "relevance": 0.30,
                    "domain": "theguardian.com", "source_name": "Guardian"},
                   {"credibility_score": 0.8, "relevance": 0.28,
                    "domain": "theguardian.com", "source_name": "Guardian"}]
    print("strong (2 diff publishers) ->", confidence_breakdown(strong))
    print("weak   (1 weak source)     ->", confidence_breakdown(weak))
    print("same   (2 same publisher)  ->", confidence_breakdown(same_source))
    print("none                       ->", confidence_breakdown([]))
