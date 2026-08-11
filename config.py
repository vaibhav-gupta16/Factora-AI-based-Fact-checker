"""
Central configuration for Factora.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"
SQL_DIR = OUTPUTS_DIR / "sql"

for _p in [DATA_RAW, DATA_PROCESSED, PLOTS_DIR, RESULTS_DIR, SQL_DIR]:
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset sources (verified working, real, downloadable via HTTPS)
# ---------------------------------------------------------------------------
LIAR_BASE_URL = "https://raw.githubusercontent.com/tfs4/liar_dataset/master"
LIAR_FILES = {"train": "train.tsv", "valid": "valid.tsv", "test": "test.tsv"}
LIAR_COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title",
    "state", "party", "barely_true_ct", "false_ct", "half_true_ct",
    "mostly_true_ct", "pants_fire_ct", "context",
]
# IMPORTANT: LIAR has NO date column, and its statements span roughly
# 2007-2016 (the dataset was published in 2017). Live news retrieval
# (Guardian/NewsAPI) can only find evidence for claims with CURRENT or
# recent coverage. Using LIAR claims for the live retrieval+verdict
# pipeline will produce a very high "unverifiable" rate — this is a
# structural mismatch, not a retrieval bug. LIAR remains valid for the
# offline EDA/SQL analysis, which needs no live retrieval at all. For
# the live pipeline, use src/current_claims.py instead (see below).

MBFC_URL = (
    "https://raw.githubusercontent.com/ramybaly/News-Media-Reliability/"
    "master/data/acl2020/corpus.tsv"
)

# ---------------------------------------------------------------------------
# APIs
# ---------------------------------------------------------------------------
# The Guardian Open Platform (PRIMARY evidence source) — free "Developer"
# key, emailed instantly (no approval wait), non-commercial use. Returns
# real article text (trailText), not just a title. Confirmed via The
# Guardian's own Open Platform docs
# (open-platform.theguardian.com/documentation) and multiple independent
# working code examples.
GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY", "")
GUARDIAN_URL = "https://content.guardianapis.com/search"
GUARDIAN_SIGNUP = "https://open-platform.theguardian.com/access/"
# Note: Guardian's q parameter is confirmed as free-text search; explicit
# AND/OR/quote-boolean support is NOT confirmed in the same way NewsAPI's
# is, so queries sent to Guardian use plain space-joined terms rather than
# boolean keywords, to avoid assuming unverified syntax.

# Prints exactly what happened on every Guardian/NewsAPI call: status code,
# timeout, connection error, or article count. ON by default right now so
# retrieval failures are visible instead of silently returning empty
# results — set DEBUG_RETRIEVAL=0 in your .env once retrieval is confirmed
# working end to end.
DEBUG_RETRIEVAL = os.getenv("DEBUG_RETRIEVAL", "1") == "1"

# NewsAPI (SECONDARY / fallback evidence source) — free key, 100 req/day.
# Only called for a given search stage if The Guardian returns nothing
# usable at that stage, to conserve this daily quota.
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
NEWSAPI_URL = "https://newsapi.org/v2/everything"
NEWSAPI_SIGNUP = "https://newsapi.org/register"
# NewsAPI's q parameter DOES support explicit AND / OR / NOT keywords and
# quoted phrases (confirmed via NewsAPI's own docs, mirrored at
# newsapi.org/docs/endpoints/everything).

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_SIGNUP = "https://console.groq.com"

# ---------------------------------------------------------------------------
# Label collapsing
# ---------------------------------------------------------------------------
LABEL_MAP_3WAY = {
    "pants-fire": "false", "false": "false", "barely-true": "false",
    "half-true": "mixed",
    "mostly-true": "true", "true": "true",
}

CREDIBILITY_SCORE_MAP = {"low": 0.2, "mixed": 0.55, "high": 0.9}
UNKNOWN_SOURCE_SCORE = 0.5

# ---------------------------------------------------------------------------
# Retrieval tuning
# ---------------------------------------------------------------------------
# Relevance floors, by query specificity. Exact-phrase and AND-narrowed
# queries are already precise, so a modest floor is enough to catch stray
# noise. The OR safety-net stage is broad by design and needs a firmer
# floor to reject generic matches. Both The Guardian and NewsAPI provide
# real article snippets (not just a title), so relevance scores tend to be
# meaningfully higher for genuinely on-topic results.
RELEVANCE_FLOOR = {"phrase": 0.06, "and": 0.06, "or": 0.15}

DEFAULT_N_SAMPLES = 60
DEFAULT_TOP_K_EVIDENCE = 5
RANDOM_SEED = 42
