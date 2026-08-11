"""
Deterministic confidence scoring — computed from measurable evidence
signals, not invented by the LLM. See generate_verdict() in verdict.py
for how this is wired in.
"""
W_CREDIBILITY = 0.45
W_RELEVANCE = 0.30
W_CORROBORATION = 0.25
CORROBORATION_SATURATION = 3


def _corroboration_score(n_supporting: int) -> float:
    if n_supporting <= 0:
        return 0.0
    return min(1.0, n_supporting / CORROBORATION_SATURATION)


def compute_confidence(supporting_evidence: list[dict]) -> float:
    if not supporting_evidence:
        return 0.0
    n = len(supporting_evidence)
    avg_credibility = sum(e.get("credibility_score", 0.0) for e in supporting_evidence) / n
    avg_relevance = sum(e.get("relevance", 0.0) for e in supporting_evidence) / n
    scaled_relevance = min(1.0, avg_relevance / 0.4)
    corroboration = _corroboration_score(n)
    confidence = (W_CREDIBILITY * avg_credibility
                  + W_RELEVANCE * scaled_relevance
                  + W_CORROBORATION * corroboration)
    return round(max(0.0, min(1.0, confidence)), 2)


def confidence_breakdown(supporting_evidence: list[dict]) -> dict:
    if not supporting_evidence:
        return {"n_supporting_sources": 0, "avg_credibility": 0.0,
                "avg_relevance": 0.0, "corroboration": 0.0, "confidence": 0.0}
    n = len(supporting_evidence)
    avg_credibility = round(sum(e.get("credibility_score", 0.0) for e in supporting_evidence) / n, 3)
    avg_relevance = round(sum(e.get("relevance", 0.0) for e in supporting_evidence) / n, 3)
    return {
        "n_supporting_sources": n, "avg_credibility": avg_credibility,
        "avg_relevance": avg_relevance,
        "corroboration": round(_corroboration_score(n), 3),
        "confidence": compute_confidence(supporting_evidence),
    }


if __name__ == "__main__":
    strong = [{"credibility_score": 0.9, "relevance": 0.42},
              {"credibility_score": 0.9, "relevance": 0.38}]
    weak = [{"credibility_score": 0.2, "relevance": 0.10}]
    print("strong ->", confidence_breakdown(strong))
    print("weak   ->", confidence_breakdown(weak))
    print("none   ->", confidence_breakdown([]))
