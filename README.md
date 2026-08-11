# Factora

A retrieval-augmented fact-checking system. Give it a claim; it searches real
news sources, scores each source's credibility, and returns a verdict (true /
false / mixed / unverifiable) with a confidence score computed from the
evidence — not guessed by a model.

Built for: **BCSE206L Foundations of Data Science**

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env               # then add your GROQ_API_KEY (required)
                                    # and GUARDIAN_API_KEY (recommended, primary)

python main.py --skip_pipeline     # offline steps only — download data, EDA, SQL
                                    # (no API keys needed for this step)

python test_connectivity.py        # confirms Guardian/NewsAPI are reachable
                                    # from your network before a real run

python main.py --n_samples 3       # full run: live retrieval + verdicts over
                                    # 3 verified CURRENT claims (default source)

streamlit run app.py               # interactive demo — check any claim live
```

## Getting API keys (all free, no card required)

| Key | Where | Required? |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | **Yes** — used for query understanding and verdicts |
| `GUARDIAN_API_KEY` | [open-platform.theguardian.com/access](https://open-platform.theguardian.com/access/) | Recommended — **primary** evidence source, key emailed instantly |
| `NEWSAPI_KEY` | [newsapi.org/register](https://newsapi.org/register) | Recommended — **secondary/fallback** evidence source |

At least one of `GUARDIAN_API_KEY` or `NEWSAPI_KEY` must be set for
retrieval to work at all.

## How it works

1. **Query understanding** — the LLM reads the claim and extracts the phrases
   a real news headline would use, rather than mechanically grabbing
   capitalised words. Falls back to a regex heuristic if the LLM call fails.
2. **Retrieval** — queries a specific-to-broad search ladder. At each stage,
   **The Guardian** (primary) is tried first; **NewsAPI** (secondary) is only
   called for that stage if The Guardian returns nothing usable, to conserve
   its daily quota.
3. **Relevance filtering** — every article is scored against the claim with
   TF-IDF cosine similarity; anything below a calibrated floor is discarded
   before it can reach the verdict step.
4. **Credibility weighting** — each surviving article's source domain is
   scored using an MBFC-derived credibility corpus (859 rated domains).
5. **Verdict** — the LLM decides true/false/mixed/unverifiable and marks
   which evidence supports it. **Confidence is computed in Python** — not by
   the model — from that evidence's credibility, relevance, and source count.
   If a claim has zero coverage but describes something that would obviously
   be major news if true, the reasoning notes that the total absence of
   coverage is itself worth flagging.

## Project layout

```
factora/
├── main.py                  <- runs the whole project, in order
├── app.py                   <- Streamlit interactive demo
├── config.py                <- paths, dataset URLs, API config, tuning constants
├── requirements.txt
├── .env.example              <- copy to .env and add your API keys
├── test_connectivity.py      <- isolated check that each API is reachable
├── LICENSE
├── data/
│   ├── raw/                  <- downloaded LIAR + MBFC datasets
│   └── processed/            <- factora.db (SQLite)
├── outputs/
│   ├── plots/                 <- EDA + evaluation charts (.png)
│   ├── results/                <- CSVs, evaluation_report.txt
│   └── sql/                     <- queries.sql + query results
└── src/
    ├── download_data.py      <- pulls real datasets from GitHub mirrors
    ├── data_loader.py        <- parses LIAR + MBFC into DataFrames
    ├── credibility.py        <- domain -> credibility score lookup
    ├── retrieval.py           <- LLM query understanding + Guardian/NewsAPI retrieval
    ├── confidence.py           <- computed (not model-generated) confidence scoring
    ├── verdict.py                <- LLM verdict generation
    ├── current_claims.py          <- small verified claim set for live-pipeline testing
    ├── pipeline.py                  <- batch: claim -> evidence -> verdict
    ├── evaluate.py                    <- metrics, confusion matrices
    ├── eda.py                          <- exploratory data analysis
    └── sql_analysis.py                  <- SQLite + joins/aggregation/window functions
```

## Datasets and APIs

| Component | Source | Notes |
|---|---|---|
| Claims | [LIAR dataset](https://github.com/tfs4/liar_dataset) (William Wang, ACL 2017) | 12,836 political statements, downloaded automatically |
| Source credibility | [MBFC-derived corpus](https://github.com/ramybaly/News-Media-Reliability) (Baly et al., ACL 2020) | 859 rated news domains, downloaded automatically |
| Live evidence (primary) | [The Guardian Open Platform](https://open-platform.theguardian.com) | Free key, emailed instantly. Returns real article snippets. |
| Live evidence (secondary) | [NewsAPI](https://newsapi.org) `/v2/everything` | Free tier: 100 requests/day. Only called when The Guardian finds nothing. |
| Query understanding + verdicts | [Groq](https://console.groq.com) (Llama 3.3 70B) | Free tier |

Both datasets download automatically on first run — no manual downloading needed.

## If retrieval seems stuck or every claim comes back "unverifiable"

Run `python test_connectivity.py` first — it makes one isolated call to each
API with a short timeout and prints exactly what happened (success, timeout,
connection error, or a non-200 response). This tells you in under a minute
whether the problem is retrieval logic or your network being unable to reach
a given API.

The main pipeline also prints this same diagnostic detail for every API call
by default (`DEBUG_RETRIEVAL=1` in `.env`). Set `DEBUG_RETRIEVAL=0` once
things are confirmed working, to quiet the output.

## Important: LIAR vs. live retrieval

LIAR's claims are historical political statements (roughly 2007–2016) with
**no date column**, and live news search naturally favours recent coverage.
Running the pipeline against LIAR will show a very high **unverifiable**
rate — this is a structural mismatch between an old claims benchmark and a
live-news retrieval system, not a bug.

- **LIAR** remains valid for offline analysis (EDA, SQL) — no live retrieval needed.
- **For the live pipeline**, use the default `--claims_source current` — a
  small, verified starter set in `src/current_claims.py`, each claim sourced
  from a real, cited fact-check.
- **To extend it**, supply your own claims with `--claims_file yours.csv`
  (columns: `id, statement, label_3way`).
- To see how the system behaves on LIAR anyway, run `--claims_source liar`.

## Known limitations

- LIAR is not suitable for live-retrieval evaluation (see above).
- A claim with no real news coverage correctly returns **unverifiable**, not
  **false** — catching a fabricated claim as "false" requires a source that
  actively contradicts it, which rarely exists for random viral claims. When
  a claim describes something that would obviously be major news if true,
  the system flags the total absence of coverage as suspicious without
  overclaiming it as confirmed false.
- NewsAPI's free tier is capped at 100 requests/day; since it's only called
  as a fallback when The Guardian finds nothing, this is rarely a bottleneck.

## License

MIT — see `LICENSE`.
