"""
Batch pipeline: sample claims, retrieve real evidence, generate verdicts,
save results.

IMPORTANT — claim source matters for live retrieval:
LIAR's claims (2007-2016, no date column) mostly predate what current news
search engines index and are no longer covered by current news. Running
this pipeline against LIAR will produce a very high 'unverifiable' rate — this
is a structural mismatch between an old claims benchmark and a live-news
retrieval system, not a bug (see config.py's note on LIAR_COLUMNS).

Use --claims_source current for a small, verified set of CURRENT claims
that live retrieval can actually find evidence for (see
src/current_claims.py), or --claims_file your_claims.csv to supply your
own. Use --claims_source liar only when you specifically want to see how
the pipeline behaves on claims that predate the evidence sources — that
result (mostly unverifiable) is itself a valid, reportable finding about
the limits of live-retrieval fact-checking.
"""
import argparse
import sys
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import load_liar
from src.current_claims import load_starter_claims, load_from_csv
from src.credibility import CredibilityScorer
from src.retrieval import fetch_evidence_articles, rank_evidence, NewsAPIError
from src.verdict import generate_verdict, VerdictError


def _load_claims(claims_source: str, claims_file: str | None,
                  n_samples: int, split: str, seed: int) -> pd.DataFrame:
    if claims_file:
        df = load_from_csv(claims_file)
        return df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)
    if claims_source == "current":
        df = load_starter_claims()
        return df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)
    # claims_source == "liar"
    liar = load_liar(split)
    return liar.sample(n=min(n_samples, len(liar)), random_state=seed).reset_index(drop=True)


def run_batch(n_samples: int, top_k: int, split: str = "test",
              sleep_sec: float = 0.5, seed: int = config.RANDOM_SEED,
              claims_source: str = "current", claims_file: str | None = None) -> pd.DataFrame:
    sample = _load_claims(claims_source, claims_file, n_samples, split, seed)
    cred_scorer = CredibilityScorer()

    rows = []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Fact-checking claims"):
        claim = row["statement"]
        record = {
            "claim_id": row["id"], "claim": claim,
            "true_label_6way": row["label_6way"], "true_label_3way": row["label_3way"],
            "speaker": row["speaker"], "subject": row["subject"],
        }
        try:
            articles, query_used = fetch_evidence_articles(claim, page_size=10)
            record["query_used"] = query_used
            time.sleep(sleep_sec)
            record["n_articles_found"] = len(articles)

            weighted_ev = rank_evidence(claim, articles, cred_scorer, top_k, use_credibility=True)
            baseline_ev = rank_evidence(claim, articles, cred_scorer, top_k, use_credibility=False)
            record["avg_credibility_weighted"] = (
                sum(e["credibility_score"] for e in weighted_ev) / len(weighted_ev)
                if weighted_ev else None
            )

            wv = generate_verdict(claim, weighted_ev, show_credibility=True)
            record["pred_label_weighted"] = wv["verdict"]
            record["confidence_weighted"] = wv["confidence"]
            record["reasoning_weighted"] = wv["reasoning"]

            bv = generate_verdict(claim, baseline_ev, show_credibility=False)
            record["pred_label_baseline"] = bv["verdict"]
            record["confidence_baseline"] = bv["confidence"]
            record["reasoning_baseline"] = bv["reasoning"]
            record["status"] = "ok"
        except (NewsAPIError, VerdictError) as e:
            record["status"] = f"error: {e}"
        rows.append(record)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Run the fact-checking pipeline.")
    parser.add_argument("--n_samples", type=int, default=config.DEFAULT_N_SAMPLES)
    parser.add_argument("--top_k", type=int, default=config.DEFAULT_TOP_K_EVIDENCE)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"],
                         help="Only used when --claims_source liar")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--claims_source", type=str, default="current",
                         choices=["current", "liar"],
                         help="'current' = small verified set live news can actually check (default). "
                              "'liar' = historical LIAR claims (mostly unverifiable — see module docstring).")
    parser.add_argument("--claims_file", type=str, default=None,
                         help="Path to your own CSV (columns: id, statement, label_3way). Overrides --claims_source.")
    args = parser.parse_args()

    df = run_batch(args.n_samples, args.top_k, args.split, args.sleep,
                    claims_source=args.claims_source, claims_file=args.claims_file)
    out_path = config.RESULTS_DIR / "pipeline_results.csv"
    df.to_csv(out_path, index=False)
    n_ok = (df["status"] == "ok").sum()
    print(f"\nSaved {len(df)} records ({n_ok} succeeded) to {out_path}")


if __name__ == "__main__":
    main()
