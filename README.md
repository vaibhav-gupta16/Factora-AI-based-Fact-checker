# Factora

**Factora** is an evidence-based, hallucination-resistant fact-checking system that retrieves live news and encyclopedic coverage, assesses source credibility, and generates a verdict with a **deterministic mathematical confidence score** computed from measurable signals.

It features both an offline data-science pipeline (EDA, SQLite analysis, and LIAR benchmark evaluation) and an interactive **Streamlit web application**.

---

## Key Highlights

- **Multi-Source Evidence Retrieval**:
  - **Google News RSS**: Zero-auth aggregator searching across major global publishers (BBC, Reuters, AP, Washington Post, etc.).
  - **The Guardian Open Platform**: High-credibility journalism API.
  - **NewsAPI**: Secondary fallback for global news headlines.
  - **Wikipedia API**: Fast, batched encyclopedic fallback for scientific, historical, and well-established facts.
- **4-Tier Verdict System**: Categorizes claims into **`true`**, **`false`**, **`mixed`**, or **`unverifiable`**.
- **Deterministic Confidence Scoring (Non-Hallucinated)**:
  - The LLM **does not invent or estimate the confidence percentage**.
  - Confidence is calculated strictly in Python using four measurable evidence metrics:
    $$\text{Confidence} = 0.35 \times \text{Credibility} + 0.25 \times \text{Relevance} + 0.20 \times \text{Corroboration} + 0.20 \times \text{Source Diversity}$$
- **Source Credibility Weighting**: Domain reliability scores mapped from an MBFC-derived dataset across 859 news domains.
- **Content Relevance Filtering**: TF-IDF cosine similarity measures claim-to-article overlap, filtering out tangential snippets.
- **Offline Data Science & Evaluation**: LIAR benchmark evaluation, SQLite query analysis, and confusion matrices comparing credibility-weighted versus baseline retrieval.

---

## How It Works

```text
Claim Input
    │
    ▼
[1] Query Formulation (Groq LLM / Regex Fallback)
    │
    ▼
[2] Evidence Retrieval
    ├─► The Guardian API
    ├─► Google News RSS (Aggregating Reuters, BBC, AP, etc.)
    ├─► NewsAPI (Quota-safe fallback)
    └─► Wikipedia API (Batched encyclopaedic fallback)
    │
    ▼
[3] Relevance Filtering & Credibility Weighting
    ├─► TF-IDF Cosine Similarity (Claim vs. Article snippet)
    └─► Domain Credibility Score (MBFC corpus: 0.0 – 1.0)
    │
    ▼
[4] Verdict Direction (Groq LLM)
    └─► Assigns verdict: true | false | mixed | unverifiable
    └─► Identifies exact supporting snippet IDs & rationale
    │
    ▼
[5] Deterministic Confidence Scoring (Python Formula)
    └─► 35% Source Credibility + 25% Scaled Relevance
        + 20% Corroboration Count + 20% Publisher Diversity
```

---

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/vaibhav-gupta16/Factora.git
cd Factora

# Create and activate a virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GUARDIAN_API_KEY=your_guardian_api_key
NEWSAPI_KEY=your_newsapi_key
DEBUG_RETRIEVAL=1
```

- `GROQ_API_KEY`: Required for query extraction and verdict reasoning ([Get free key](https://console.groq.com)).
- `GUARDIAN_API_KEY`: Required for The Guardian live search ([Get free key](https://open-platform.theguardian.com/access/)).
- `NEWSAPI_KEY`: Optional fallback ([Get free key](https://newsapi.org/register)).
- *Google News RSS and Wikipedia require no keys and work out of the box.*

### 3. Verify Connectivity

```bash
python test_connectivity.py
```

### 4. Run the Web Application

```bash
streamlit run app.py
```

Or run directly through Python:
```bash
python -m streamlit run app.py
```

---

## Batch Pipeline & Offline Evaluation

```bash
# Run EDA and SQLite analysis on the LIAR dataset (no API calls needed)
python main.py --skip_pipeline

# Run the live pipeline on current verified claims
python main.py --claims_source current --n_samples 5

# Evaluate weighted vs. baseline pipeline metrics
python -m src.evaluate
```

---

## Project Structure

```text
factora/
├── app.py                 # Streamlit web application
├── main.py                # Command-line pipeline entrypoint
├── config.py              # Configuration constants, weights, and API endpoints
├── test_connectivity.py   # Diagnostics and API connection checks
├── requirements.txt       # Python dependencies
├── VIVA_GUIDE.md          # Comprehensive Viva preparation guide (What, Why, How)
├── data/
│   ├── raw/               # LIAR benchmark and MBFC credibility datasets
│   └── processed/         # SQLite database for SQL analysis
├── outputs/
│   ├── plots/             # EDA charts and confusion matrices
│   ├── results/           # Pipeline output CSVs and evaluation reports
│   └── sql/               # SQLite query scripts and logs
└── src/
    ├── retrieval.py       # Query understanding and multi-stage live retrieval
    ├── credibility.py     # Domain credibility scorer (MBFC dataset)
    ├── confidence.py      # Mathematical, deterministic confidence formula
    ├── verdict.py         # LLM verdict reasoning & direction
    ├── pipeline.py        # Batch fact-checking runner
    ├── evaluate.py        # Accuracy, Precision, Recall, and confusion plots
    ├── eda.py             # Exploratory Data Analysis
    ├── current_claims.py  # Verified modern benchmark claims
    ├── data_loader.py     # LIAR dataset parser
    ├── download_data.py   # Automatic dataset downloader
    └── sql_analysis.py    # SQL analytical queries
```

---

## License

This project is licensed under the [MIT License](LICENSE).
