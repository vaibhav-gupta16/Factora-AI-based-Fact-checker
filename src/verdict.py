"""
Verdict generation: the LLM decides direction (true/false/mixed/
unverifiable) and marks which evidence supports it; confidence is
computed in Python from that evidence's credibility/relevance/count.
"""
import json
import sys
from pathlib import Path
from groq import Groq

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.confidence import compute_confidence, confidence_breakdown


class VerdictError(RuntimeError):
    pass


def _client() -> Groq:
    if not config.GROQ_API_KEY:
        raise VerdictError(
            f"GROQ_API_KEY is not set. Get a free key at {config.GROQ_SIGNUP}."
        )
    return Groq(api_key=config.GROQ_API_KEY)


def _format_evidence(evidence: list[dict], show_credibility: bool) -> str:
    lines = []
    for i, e in enumerate(evidence, 1):
        snippet = f"{e['title']}. {e['description']}".strip()
        if show_credibility:
            lines.append(f"[{i}] Source: {e['source_name']} "
                         f"(credibility: {e['credibility_label']}) — {snippet}")
        else:
            lines.append(f"[{i}] Source: {e['source_name']} — {snippet}")
    return "\n".join(lines) if lines else "(no evidence retrieved)"


SYSTEM_PROMPT = """You are a careful fact-checking assistant. You will be \
given a claim and a numbered list of evidence snippets retrieved from news \
sources. Decide whether the claim is TRUE, FALSE, or MIXED based only on \
the evidence given. If credibility ratings are shown, weigh higher-\
credibility sources more heavily, and say so explicitly in your reasoning.

If there is no evidence, or the evidence has no real bearing on the claim, \
use the verdict "unverifiable" instead of guessing.

Also list the numbers of the evidence items that directly SUPPORT your \
verdict. If none do, use an empty list.

Respond with ONLY a JSON object, no other text:
{"verdict": "true" | "false" | "mixed" | "unverifiable", \
"supporting_evidence_numbers": [<ints>], "reasoning": "<2-3 sentences>"}

Do NOT include a confidence score — that is computed separately."""

_VALID_VERDICTS = {"true", "false", "mixed", "unverifiable"}


_PLAUSIBILITY_PROMPT = """A claim was searched for and NO news coverage was \
found anywhere. Using only your own general knowledge (no evidence is given \
to you here), assess one thing: if this claim were true, would it almost \
certainly be covered by mainstream news? Claims about official government \
policy, a named company's confirmed announcement, a large infrastructure or \
space program, or a large specific sum of money/statistic are the kind of \
thing that gets reported immediately if real. Claims about private, local, \
very recent, or minor matters may genuinely have no coverage yet.

Respond with ONLY a JSON object:
{"would_be_newsworthy": true | false, "note": "<one sentence, plain, no hedging>"}
"""


def _assess_plausibility_without_evidence(claim: str) -> dict | None:
    """
    Called only when zero evidence was found. Doesn't confirm or deny the
    claim — that would require evidence this function deliberately doesn't
    have. It only assesses whether the claim's total absence from the news
    is itself notable, which is a genuinely useful signal for claims that
    describe something that would obviously be huge news if real (e.g. "NASA
    confirms permanent lunar settlement"). Returns None on any failure so
    the caller falls back to the plain generic message.
    """
    try:
        client = _client()
        completion = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[{"role": "system", "content": _PLAUSIBILITY_PROMPT},
                      {"role": "user", "content": f"Claim: {claim}"}],
            temperature=0.1, max_tokens=120,
        )
        raw = completion.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        parsed = json.loads(raw)
        return {"would_be_newsworthy": bool(parsed.get("would_be_newsworthy", False)),
                "note": str(parsed.get("note", "")).strip()}
    except Exception:
        return None


def generate_verdict(claim: str, evidence: list[dict], show_credibility: bool = True) -> dict:
    if not evidence:
        base_reasoning = (
            "No evidence was retrieved for this claim, so there is nothing to "
            "weigh it against. This does not mean the claim is false — only "
            "that no supporting or refuting evidence was found."
        )
        assessment = _assess_plausibility_without_evidence(claim)
        if assessment and assessment["would_be_newsworthy"]:
            reasoning = (
                f"{base_reasoning} Worth noting: {assessment['note']} A claim "
                f"of this scale having zero coverage anywhere is itself "
                f"suspicious, though this is not confirmation the claim is false."
            )
        elif assessment:
            reasoning = f"{base_reasoning} {assessment['note']}"
        else:
            reasoning = base_reasoning

        return {
            "verdict": "unverifiable", "confidence": 0.0,
            "reasoning": reasoning,
            "confidence_breakdown": confidence_breakdown([]),
            "no_coverage_is_suspicious": bool(assessment and assessment["would_be_newsworthy"]),
        }

    client = _client()
    evidence_block = _format_evidence(evidence, show_credibility)
    user_prompt = f"Claim: {claim}\n\nEvidence:\n{evidence_block}"

    completion = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_prompt}],
        temperature=0.1, max_tokens=300,
    )
    raw = completion.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise VerdictError(f"Could not parse model output as JSON: {raw!r}") from e

    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in _VALID_VERDICTS:
        raise VerdictError(f"Unexpected verdict value: {verdict!r}")

    support_nums = parsed.get("supporting_evidence_numbers", []) or []
    supporting = []
    for num in support_nums:
        try:
            idx = int(num) - 1
        except (ValueError, TypeError):
            continue
        if 0 <= idx < len(evidence):
            supporting.append(evidence[idx])

    if verdict in {"true", "false", "mixed"} and not supporting:
        supporting = evidence

    if verdict == "unverifiable":
        confidence, breakdown = 0.0, confidence_breakdown([])
    else:
        confidence = compute_confidence(supporting)
        breakdown = confidence_breakdown(supporting)

    return {"verdict": verdict, "confidence": confidence,
            "reasoning": parsed.get("reasoning", ""), "confidence_breakdown": breakdown}
