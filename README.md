# Factora

Factora is an evidence-based fact-checking system that retrieves real news coverage, evaluates source credibility, and produces a verdict with a confidence score calculated from the evidence.

It includes both an end-to-end data-science pipeline and a Streamlit web app for checking claims interactively.

## Features

- LLM-assisted claim and search-query understanding using Groq (Llama 3.3 70B)
- Live evidence retrieval from The Guardian, with NewsAPI as a fallback
- TF-IDF cosine-similarity filtering for article relevance
- MBFC-derived credibility scores for 859 news domains
- Evidence-grounded verdict generation with supporting sources
- Confidence scoring computed in Python from credibility, relevance, and corroboration
- Offline exploratory data analysis, SQLite analysis, and evaluation reports using the LIAR dataset
- Interactive Streamlit interface

## How it works

1. The claim is converted into focused search terms.
2. Factora searches The Guardian first and uses NewsAPI only when needed.
3. Retrieved articles are filtered by TF-IDF relevance.
4. Remaining sources are weighted using domain credibility scores.
5. Groq generates the verdict and reasoning from the selected evidence.
6. Python calculates the final confidence score; it is not supplied by the model.

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key
GUARDIAN_API_KEY=your_guardian_key
NEWSAPI_KEY=your_newsapi_key
DEBUG_RETRIEVAL=1
```

`GROQ_API_KEY` is required for query understanding and verdict generation. At least one of `GUARDIAN_API_KEY` or `NEWSAPI_KEY` is required for live evidence retrieval.

Free keys are available from [Groq](https://console.groq.com), [The Guardian Open Platform](https://open-platform.theguardian.com/access/), and [NewsAPI](https://newsapi.org/register).

### 3. Run the project

```bash
# Download datasets, run EDA, and perform SQLite analysis without API calls
python main.py --skip_pipeline

# Check API connectivity before a live run
python test_connectivity.py

# Run the live pipeline on three current, verified claims
python main.py --n_samples 3

# Launch the interactive web app
streamlit run app.py
```

Useful options:

```bash
python main.py --claims_source current --n_samples 10
python main.py --claims_source liar --n_samples 10
python main.py --claims_file your_claims.csv
```

Custom claim files should contain `id`, `statement`, and `label_3way` columns.

## Data sources

| Purpose | Source |
|---|---|
| Historical claims | [LIAR dataset](https://github.com/tfs4/liar_dataset), William Wang, ACL 2017 |
| Source credibility | [MBFC-derived corpus](https://github.com/ramybaly/News-Media-Reliability), Baly et al., ACL 2020 |
| Primary live evidence | [The Guardian Open Platform](https://open-platform.theguardian.com) |
| Fallback live evidence | [NewsAPI](https://newsapi.org) |
| Query understanding and verdicts | [Groq](https://groq.com) with Llama 3.3 70B |

Datasets are downloaded automatically on the first pipeline run.

## Project structure

```text
factora/
├── app.py                 # Streamlit interface
├── main.py                # End-to-end pipeline runner
├── config.py              # Paths, API settings, and tuning constants
├── test_connectivity.py   # API connectivity checks
├── requirements.txt
├── data/
│   ├── raw/               # Downloaded source datasets
│   └── processed/         # SQLite database
├── outputs/
│   ├── plots/             # EDA and evaluation charts
│   ├── results/           # Pipeline results and evaluation reports
│   └── sql/               # SQL queries and outputs
└── src/
    ├── retrieval.py       # Query understanding and evidence retrieval
    ├── credibility.py     # Domain credibility lookup
    ├── confidence.py      # Evidence-based confidence calculation
    ├── verdict.py         # LLM verdict generation
    ├── pipeline.py        # Batch fact-checking
    ├── evaluate.py        # Metrics and confusion matrices
    ├── eda.py             # Exploratory data analysis
    └── sql_analysis.py    # SQLite analysis
```

## Important limitation

The LIAR dataset contains historical political statements, mostly from 2007–2016, and has no date column. Because live news search favors recent coverage, using LIAR claims with the live retrieval pipeline can produce many `unverifiable` results. LIAR is intended for offline EDA, SQL, and evaluation; use the default `current` claims or a recent custom claims file for live fact-checking.

An absence of coverage is not automatically treated as proof that a claim is false. Factora reports `unverifiable` when reliable contradictory evidence cannot be found.

## License

This project is licensed under the [MIT License](LICENSE).
