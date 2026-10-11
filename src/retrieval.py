import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.credibility import CredibilityScorer

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "he", "she", "they", "in", "is", "it", "its", "of", "on",
    "that", "the", "to", "was", "were", "will", "would", "should", "could",
    "with", "this", "these", "those", "if", "not", "no", "stopped",
    "claims", "claim", "claimed", "viral", "post", "posts", "social",
    "media", "says", "said", "according", "report", "reports", "reported",
}

# Wikipedia gets a fixed credibility score equivalent to a high-credibility
# news source: crowd-edited but heavily cited and fact-checked for major topics.
_WIKIPEDIA_CREDIBILITY = 0.75
_WIKIPEDIA_DOMAINS = {"en.wikipedia.org", "wikipedia.org"}


class NewsAPIError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Step 1a: LLM-based context-aware term extraction (primary)
# ---------------------------------------------------------------------------

_QUERY_SYSTEM_PROMPT = """You extract search terms from a news claim so it \
can be looked up against a news search engine. Read the claim for its \
actual meaning, not just its capitalised words. Return the terms a real \
news headline about this story would use.

Return ONLY a JSON object of this exact shape, nothing else:
{"phrases": ["<multi-word entity or concept, 2-4 words>", ...], \
"keywords": ["<single distinctive word>", ...]}

Rules:
- "phrases" = named entities (people, places, organisations) AND the core \
newsworthy action/topic as a short phrase (e.g. "rupee intervention", \
"media broadcast ban"). 2-5 phrases, most important first.
- "keywords" = extra single distinctive words not already inside a phrase \
(specific numbers, unusual terms). 0-4 keywords.
- Skip filler words: "believed", "according to", "traders", "reportedly", \
"warning comes as", generic verbs. These do not help find the story.
- Do not include the word "claim", "viral", or similar meta-commentary.
"""


def _groq_client():
    from groq import Groq
    if not config.GROQ_API_KEY:
        return None
    return Groq(api_key=config.GROQ_API_KEY)


def generate_search_terms_llm(claim: str) -> dict | None:
    """
    Uses Groq to extract context-understood phrases/keywords from a claim.
    Returns {"phrases": [...], "keywords": [...]} or None if unavailable
    (missing key, network error, bad response) so callers fall back to
    the regex heuristic.
    """
    client = _groq_client()
    if client is None:
        return None
    try:
        completion = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": _QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Claim: {claim}"},
            ],
            temperature=0.1,
            max_completion_tokens=600,
            reasoning_effort="low",
            reasoning_format="hidden",
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        parsed = json.loads(raw)
        phrases = [p.strip() for p in parsed.get("phrases", []) if p and p.strip()]
        keywords = [k.strip() for k in parsed.get("keywords", []) if k and k.strip()]
        if not phrases and not keywords:
            return None
        return {"phrases": phrases[:5], "keywords": keywords[:4]}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step 1b: regex-based fallback term extraction (no API needed)
# ---------------------------------------------------------------------------

def _extract_phrases_regex(words: list[str]) -> tuple[list[str], set[int]]:
    _SKIP = {"the", "a", "an", "in", "of", "at", "its", "is", "has", "was",
              "for", "from", "who", "not", "are", "be", "by"}
    _GENERIC = {"prime", "former", "new", "old", "north", "south", "east",
                "west", "late", "early", "minister", "president", "senator",
                "governor", "director", "chief", "general", "captain",
                "doctor", "professor", "the"}

    def is_cap(w):
        return w[0].isupper() and len(w) > 2 and w.lower() not in _SKIP

    runs, i = [], 0
    while i < len(words):
        if is_cap(words[i]):
            run = [i]
            j = i + 1
            while j < len(words) and (is_cap(words[j]) or words[j].lower() in _SKIP):
                if is_cap(words[j]):
                    run.append(j)
                j += 1
            if len(run) >= 2:
                runs.append(run)
            i = j
        else:
            i += 1

    phrases, used = [], set()
    for run in runs:
        pairs = []
        for k in range(len(run) - 1):
            a, b = run[k], run[k + 1]
            score = sum(1 for w in [words[a], words[b]] if w.lower() in _GENERIC)
            pairs.append((score, a, b))
        pairs.sort()
        assigned = set()
        for _, a, b in pairs:
            if a not in assigned and b not in assigned:
                phrases.append(f"{words[a]} {words[b]}")
                used.add(a); used.add(b)
                assigned.add(a); assigned.add(b)
    return phrases, used


def generate_search_terms_regex(claim: str) -> dict:
    """No-API fallback: mechanical extraction of phrases/keywords."""
    words = re.findall(r"[A-Za-z0-9]+", claim)
    phrases, phrase_idx = _extract_phrases_regex(words)

    seen, keywords = set(), []
    for i, w in enumerate(words):
        lw = w.lower()
        if i in phrase_idx or lw in _STOPWORDS or len(w) <= 2 or lw in seen:
            continue
        seen.add(lw)
        if w[0].isupper() or any(ch.isdigit() for ch in w):
            keywords.append(w)
    return {"phrases": phrases[:5], "keywords": keywords[:4]}


def get_search_terms(claim: str) -> dict:
    """LLM-based extraction first, regex fallback if that's unavailable."""
    llm_terms = generate_search_terms_llm(claim)
    if llm_terms:
        return llm_terms
    return generate_search_terms_regex(claim)


# ---------------------------------------------------------------------------
# Step 2: build per-API query ladders from the extracted terms
# ---------------------------------------------------------------------------

def build_guardian_queries(terms: dict) -> list[tuple[str, str]]:
    """
    Guardian syntax: free-text search, no confirmed boolean operators, so
    terms are plain space-joined rather than using AND/OR keywords.
    """
    phrases = terms["phrases"]  # no quotes -- Guardian's q is plain free text
    keywords = terms["keywords"]
    ladder = []

    stage1 = phrases[:2] + keywords[:1]
    if len(stage1) >= 2:
        ladder.append((" ".join(stage1), "and"))

    if len(phrases) >= 2:
        ladder.append((" ".join(phrases[:2]), "and"))

    if phrases:
        ladder.append((phrases[0], "phrase"))
    elif keywords:
        ladder.append((keywords[0], "phrase"))

    all_terms = phrases[:3] + keywords[:3]
    if all_terms:
        ladder.append((" ".join(all_terms), "or"))

    seen, out = set(), []
    for q, t in ladder:
        if q and q not in seen:
            seen.add(q)
            out.append((q, t))
    return out


def build_newsapi_queries(terms: dict) -> list[tuple[str, str]]:
    """NewsAPI syntax: quoted phrases, explicit AND / OR keywords."""
    phrases = [f'"{p}"' for p in terms["phrases"]]
    keywords = terms["keywords"]
    ladder = []

    stage1 = phrases[:2] + keywords[:1]
    if len(stage1) >= 2:
        ladder.append((" AND ".join(stage1), "and"))

    if len(phrases) >= 2:
        ladder.append((" AND ".join(phrases[:2]), "and"))

    if phrases:
        ladder.append((phrases[0], "phrase"))
    elif keywords:
        ladder.append((keywords[0], "phrase"))

    all_terms = phrases[:3] + keywords[:3]
    if all_terms:
        ladder.append((" OR ".join(all_terms), "or"))

    seen, out = set(), []
    for q, t in ladder:
        if q and q not in seen:
            seen.add(q)
            out.append((q, t))
    return out


def _build_wikipedia_query(terms: dict) -> str:
    """Build a plain-text Wikipedia search query from extracted terms."""
    parts = terms.get("phrases", [])[:2] + terms.get("keywords", [])[:2]
    return " ".join(parts)


def build_wikipedia_queries(terms: dict) -> list[str]:
    """Build a multi-stage query ladder for Wikipedia:
    1. Core single concept phrase (e.g. 'Moonlight', 'Earth orbit')
    2. Multiple concept phrases if present
    3. Phrases + distinctive keywords (e.g. 'Earth orbit 365.25')
    4. Keywords alone if phrases are absent
    """
    ladder = []
    phrases = terms.get("phrases", [])
    keywords = terms.get("keywords", [])

    if phrases:
        ladder.append(phrases[0])
    if len(phrases) >= 2:
        ladder.append(" ".join(phrases[:2]))
    if phrases and keywords:
        ladder.append(" ".join(phrases[:2] + keywords[:2]))
    elif keywords:
        ladder.append(" ".join(keywords[:3]))

    seen, out = set(), []
    for q in ladder:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


# ---------------------------------------------------------------------------
# Step 3: API calls
# ---------------------------------------------------------------------------

def _call_guardian(query: str, page_size: int) -> list[dict]:
    """The Guardian Open Platform (PRIMARY source) — free key, real
    article snippets (trailText). Response shape confirmed via The
    Guardian's own docs and multiple independent working examples:
    response.results[].{webTitle, webUrl, webPublicationDate, fields.trailText}."""
    if not config.GUARDIAN_API_KEY:
        if config.DEBUG_RETRIEVAL:
            print("[guardian] skipped -- GUARDIAN_API_KEY not set")
        return []
    params = {
        "q": query, "page-size": min(page_size, 50), "order-by": "relevance",
        "show-fields": "trailText", "api-key": config.GUARDIAN_API_KEY,
    }
    try:
        resp = requests.get(config.GUARDIAN_URL, params=params, timeout=10)
        if resp.status_code != 200:
            if config.DEBUG_RETRIEVAL:
                print(f"[guardian] HTTP {resp.status_code} for query: {query!r} "
                      f"-- body: {resp.text[:200]!r}")
            return []
        data = resp.json()
    except requests.exceptions.Timeout:
        if config.DEBUG_RETRIEVAL:
            print(f"[guardian] TIMEOUT after 10s for query: {query!r}")
        return []
    except Exception as e:
        if config.DEBUG_RETRIEVAL:
            print(f"[guardian] {type(e).__name__} for query: {query!r} -- {e}")
        return []

    articles = []
    for r in (data.get("response", {}) or {}).get("results", []) or []:
        url = r.get("webUrl", "") or ""
        domain = urlparse(url).netloc.replace("www.", "") or "theguardian.com"
        fields = r.get("fields", {}) or {}
        articles.append({
            "title": r.get("webTitle") or "",
            "description": fields.get("trailText") or "",
            "source_name": "theguardian.com",
            "domain": domain,
            "url": url,
            "published_at": r.get("webPublicationDate", ""),
        })
    if config.DEBUG_RETRIEVAL:
        print(f"[guardian] {len(articles)} article(s) for query: {query!r}")
    return articles


def _call_newsapi(query: str, page_size: int) -> list[dict]:
    """NewsAPI (SECONDARY / fallback source) — free key, 100 req/day."""
    if not config.NEWSAPI_KEY:
        if config.DEBUG_RETRIEVAL:
            print("[newsapi] skipped -- NEWSAPI_KEY not set")
        return []
    params = {
        "q": query, "language": "en", "sortBy": "relevancy",
        "pageSize": page_size, "apiKey": config.NEWSAPI_KEY,
    }
    try:
        resp = requests.get(config.NEWSAPI_URL, params=params, timeout=10)
        data = resp.json()
        if data.get("status") != "ok":
            if config.DEBUG_RETRIEVAL:
                print(f"[newsapi] status={data.get('status')} "
                      f"message={data.get('message')!r} for query: {query!r}")
            return []
    except requests.exceptions.Timeout:
        if config.DEBUG_RETRIEVAL:
            print(f"[newsapi] TIMEOUT after 10s for query: {query!r}")
        return []
    except Exception as e:
        if config.DEBUG_RETRIEVAL:
            print(f"[newsapi] {type(e).__name__} for query: {query!r} -- {e}")
        return []

    articles = []
    for a in data.get("articles", []):
        url = a.get("url", "") or ""
        domain = urlparse(url).netloc.replace("www.", "")
        articles.append({
            "title": a.get("title") or "",
            "description": a.get("description") or "",
            "source_name": (a.get("source") or {}).get("name", domain),
            "domain": domain,
            "url": url,
            "published_at": a.get("publishedAt", ""),
        })
    if config.DEBUG_RETRIEVAL:
        print(f"[newsapi] {len(articles)} article(s) for query: {query!r}")
    return articles


def _call_wikipedia(query: str, n_results: int = 3) -> list[dict]:
    """
    Wikipedia (ENCYCLOPAEDIC fallback) — no key, no rate limit beyond
    politeness. Used when Guardian + NewsAPI both return no usable evidence,
    typically for well-known factual claims that predate recent news cycles
    (e.g. Chandrayaan-3 landing, scientific facts, historical events).

    Two-step with batching: search API to find matching page IDs, then a
    SINGLE batched extracts API request to get all lead sections at once.
    This avoids sequential HTTP requests, latency, and HTTP 429 rate limits.
    """
    headers = {
        "User-Agent": "Factora/1.0 (https://github.com/vaibhav-gupta16/Factora; fact-checking-bot)"
    }
    try:
        search_url = (
            "https://en.wikipedia.org/w/api.php?action=query"
            "&list=search&format=json&srlimit=8&srsearch="
            + urllib.parse.quote(query)
        )
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as r:
            results = json.loads(r.read()).get("query", {}).get("search", [])
    except Exception as e:
        if config.DEBUG_RETRIEVAL:
            print(f"[wikipedia] search error: {e}")
        return []

    if config.DEBUG_RETRIEVAL:
        print(f"[wikipedia] {len(results)} search hit(s) for query: {query!r}")

    hits = results[:n_results]
    if not hits:
        return []

    pageids = [str(h["pageid"]) for h in hits if h.get("pageid")]
    if not pageids:
        return []

    try:
        page_url = (
            "https://en.wikipedia.org/w/api.php?action=query"
            "&prop=extracts&exintro=1&explaintext=1&format=json&pageids="
            + "|".join(pageids)
        )
        req2 = urllib.request.Request(page_url, headers=headers)
        with urllib.request.urlopen(req2, timeout=8) as r2:
            page_data = json.loads(r2.read())
        pages = page_data.get("query", {}).get("pages", {})
    except Exception as e:
        if config.DEBUG_RETRIEVAL:
            print(f"[wikipedia] batched extract error: {e}")
        return []

    articles = []
    # Preserve original search ranking order
    for hit in hits:
        pid = str(hit["pageid"])
        page_info = pages.get(pid, {})
        extract = page_info.get("extract", "")
        if not extract:
            continue
        snippet = extract[:1200].strip()
        wiki_title = hit.get("title") or page_info.get("title", "")
        articles.append({
            "title": wiki_title,
            "description": snippet,
            "source_name": "Wikipedia",
            "domain": "en.wikipedia.org",
            "url": (
                "https://en.wikipedia.org/wiki/"
                + urllib.parse.quote(wiki_title.replace(" ", "_"))
            ),
            "published_at": "",
        })
        if config.DEBUG_RETRIEVAL:
            print(f"[wikipedia] fetched article: {wiki_title!r}")

    return articles


def _call_google_news_rss(query: str, n_results: int = 10) -> list[dict]:
    """Google News RSS (AGGREGATOR source) — no API key, no rate limit.
    Aggregates articles from ALL major publishers (BBC, Reuters, AP, NYT,
    etc.) and supports free-text search queries.  This single source
    dramatically improves evidence diversity: instead of 2-3 articles from
    one publisher, we get articles from many independent outlets in one
    call, making contradictory/mixed evidence findable."""
    try:
        import feedparser
    except ImportError:
        if config.DEBUG_RETRIEVAL:
            print("[google_news] feedparser not installed — pip install feedparser")
        return []

    encoded = urllib.parse.quote(query)
    url = f"{config.GOOGLE_NEWS_RSS_URL}?q={encoded}&hl=en&gl=US&ceid=US:en"

    try:
        resp = requests.get(
            url, timeout=10,
            headers={"User-Agent": "Factora/1.0 (fact-checking project)"},
        )
        if resp.status_code != 200:
            if config.DEBUG_RETRIEVAL:
                print(f"[google_news] HTTP {resp.status_code} for query: {query!r}")
            return []
        feed = feedparser.parse(resp.content)
    except requests.exceptions.Timeout:
        if config.DEBUG_RETRIEVAL:
            print(f"[google_news] TIMEOUT after 10s for query: {query!r}")
        return []
    except Exception as e:
        if config.DEBUG_RETRIEVAL:
            print(f"[google_news] {type(e).__name__} for query: {query!r} -- {e}")
        return []

    articles = []
    for entry in feed.entries[:n_results]:
        raw_title = entry.get("title", "")
        link = entry.get("link", "")

        # Google News titles end with " - Source Name"
        title = raw_title
        source_from_title = ""
        if " - " in raw_title:
            parts = raw_title.rsplit(" - ", 1)
            if len(parts[1]) < 50:  # looks like a source name, not headline text
                title = parts[0].strip()
                source_from_title = parts[1].strip()

        # Extract actual publisher from the RSS <source> element
        entry_source = entry.get("source", {}) or {}
        source_name = ""
        source_domain = ""
        if entry_source:
            source_name = entry_source.get("title", "") or ""
            source_href = (entry_source.get("href", "")
                           or entry_source.get("url", "") or "")
            if source_href:
                source_domain = urlparse(source_href).netloc.replace("www.", "")

        if not source_name:
            source_name = source_from_title
        if not source_domain:
            source_domain = urlparse(link).netloc.replace("www.", "")

        # Strip HTML tags from description
        desc = entry.get("summary", "") or entry.get("description", "")
        desc = re.sub(r'<[^>]+>', '', desc).strip()

        articles.append({
            "title": title,
            "description": desc[:500] if desc else "",
            "source_name": source_name or source_domain,
            "domain": source_domain,
            "url": link,
            "published_at": entry.get("published", ""),
        })

    if config.DEBUG_RETRIEVAL:
        print(f"[google_news] {len(articles)} article(s) for query: {query!r}")
    return articles


def _dedupe(articles: list[dict]) -> list[dict]:
    seen_urls, out = set(), []
    for art in articles:
        u = art.get("url", "")
        if u and u not in seen_urls:
            seen_urls.add(u)
            out.append(art)
    return out


# ---------------------------------------------------------------------------
# Step 4: relevance scoring
# ---------------------------------------------------------------------------

def score_relevance(claim: str, articles: list[dict]) -> list[float]:
    if not articles:
        return []
    texts = [claim] + [f"{a['title']} {a['description']}" for a in articles]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vectorizer.fit_transform(texts)
        return cosine_similarity(tfidf[0:1], tfidf[1:]).flatten().tolist()
    except Exception:
        return [0.0] * len(articles)


def _matched_terms(terms: dict, article: dict) -> list[str]:
    """Which of the extracted phrases/keywords literally appear (case-
    insensitive substring) in this article's title+description."""
    text = f"{article.get('title', '')} {article.get('description', '')}".lower()
    matched = []
    for p in terms.get("phrases", []):
        if p.lower() in text:
            matched.append(p)
    for k in terms.get("keywords", []):
        if k.lower() in text:
            matched.append(k)
    return matched


def _passes_term_coverage(terms: dict, article: dict) -> bool:
    """
    Requires at least 2 distinct extracted terms (or all of them, if fewer
    than 2 were extracted) to literally appear in the article.

    TF-IDF cosine similarity alone is not enough: a claim like "Silver Lake
    in Talks to Buy Workday" can score an unrelated movie review (which
    happens to mention the LA neighbourhood "Silver Lake") or a lifestyle
    article (which happens to use the common word "workday") non-trivially,
    because those are ordinary words/names with unrelated everyday meanings.
    Requiring TWO distinct terms to co-occur is a much stronger signal: an
    article coincidentally containing "Silver Lake" as a place name is
    common; one that ALSO happens to mention "Workday" is not, unless it is
    genuinely about this story.

    Wikipedia articles are exempt from this check: the intro paragraph of a
    Wikipedia article about "Chandrayaan-3" will contain "Chandrayaan" and
    "Moon" and "India" by definition, so the coverage check would always
    pass anyway -- but the article title itself is used as the search query,
    so we trust the search-level relevance and skip the redundant substring
    check for Wikipedia sources.
    """
    if article.get("domain") in _WIKIPEDIA_DOMAINS:
        return True  # Wikipedia search already filtered by topic
    total_terms = len(terms.get("phrases", [])) + len(terms.get("keywords", []))
    if total_terms == 0:
        return True  # no terms to check against (e.g. manual query override)
    required = min(2, total_terms)
    return len(_matched_terms(terms, article)) >= required


def _filter_relevant(claim: str, articles: list[dict], floor: float,
                      terms: dict | None = None) -> list[dict]:
    """
    Scores articles for relevance ONCE and attaches the score to each
    article dict (so later steps like rank_evidence reuse this exact
    number instead of recomputing TF-IDF against a different, smaller
    corpus -- cosine similarity is corpus-relative, so recomputing it on a
    different document set silently changes the numbers for the same
    article). Also applies the term-coverage check when `terms` is given.
    """
    if not articles:
        return []
    rels = score_relevance(claim, articles)
    for a, r in zip(articles, rels):
        a["_relevance"] = float(r)
    kept = [a for a in articles if a["_relevance"] >= floor]
    if terms is not None:
        kept = [a for a in kept if _passes_term_coverage(terms, a)]
    return kept


# ---------------------------------------------------------------------------
# Step 5: main fetch function
# ---------------------------------------------------------------------------

def fetch_evidence_articles(
    claim: str, page_size: int = 10, query: str | None = None
) -> tuple[list[dict], str]:
    """
    Fetch relevant evidence for a claim. Fully automatic by default.

    Source priority:
      1. The Guardian + Google News RSS (primary, both free / no quota)
         — queried together and merged for maximum diversity.
      2. NewsAPI (tertiary fallback, 100 req/day quota)
      3. Wikipedia (encyclopaedic fallback — well-known facts that
         predate recent news cycles)

    Returns (relevant_articles, description_of_query_used).
    """
    if query:
        # Manual override: merge Guardian + Google News RSS, then
        # fall back to NewsAPI. Wikipedia not tried for manual
        # overrides since the user chose the query explicitly.
        raw_combined = _dedupe(
            _call_guardian(query, page_size)
            + _call_google_news_rss(query, page_size)
        )
        relevant = _filter_relevant(claim, raw_combined,
                                     config.RELEVANCE_FLOOR["and"])
        if relevant:
            return relevant, f"Guardian + Google News: {query}"
        newsapi_arts = _filter_relevant(claim, _call_newsapi(query, page_size),
                                         config.RELEVANCE_FLOOR["and"])
        return newsapi_arts, f"NewsAPI: {query}"

    terms = get_search_terms(claim)
    guardian_ladder = build_guardian_queries(terms)
    newsapi_ladder = build_newsapi_queries(terms)
    n_stages = max(len(guardian_ladder), len(newsapi_ladder))

    last_desc = "no usable search terms found"
    for i in range(n_stages):
        uq, utype = guardian_ladder[i] if i < len(guardian_ladder) else (None, None)
        nq, ntype = newsapi_ladder[i] if i < len(newsapi_ladder) else (None, None)
        stage_type = utype or ntype
        floor = config.RELEVANCE_FLOOR[stage_type]

        # Primary: Guardian + Google News RSS (both free, no daily quota).
        # Merged and deduped so the LLM sees evidence from many independent
        # publishers, making true/false/mixed verdicts all reachable.
        if uq:
            combined = _dedupe(
                _call_guardian(uq, page_size)
                + _call_google_news_rss(uq, page_size)
            )
            last_desc = f"Guardian + Google News: {uq}"
            if combined:
                relevant = _filter_relevant(claim, combined, floor, terms)
                if relevant:
                    return relevant, last_desc

        # Tertiary fallback: NewsAPI (daily quota — only used if both free
        # sources found nothing usable at this query stage).
        if nq:
            newsapi_arts = _dedupe(_call_newsapi(nq, page_size))
            last_desc = f"NewsAPI: {nq}"
            if newsapi_arts:
                relevant = _filter_relevant(claim, newsapi_arts, floor, terms)
                if relevant:
                    return relevant, last_desc

    # Encyclopaedic fallback: Wikipedia.
    # Reached only when all news sources failed across all query stages.
    # Typical for well-known facts no longer in current news cycles
    # (missions, scientific discoveries, historical events).
    wiki_queries = build_wikipedia_queries(terms)
    if not wiki_queries:
        single_q = _build_wikipedia_query(terms)
        if single_q:
            wiki_queries = [single_q]

    for wiki_query in wiki_queries:
        wiki_arts = _call_wikipedia(wiki_query, n_results=6)
        if wiki_arts:
            # Lower relevance floor for Wikipedia: intro paragraphs are
            # shorter than news articles, so TF-IDF scores are lower.
            wiki_floor = config.RELEVANCE_FLOOR.get("wiki", 0.05)
            relevant = _filter_relevant(claim, wiki_arts, wiki_floor, terms)
            if relevant:
                last_desc = f"Wikipedia: {wiki_query}"
                return relevant, last_desc

    return [], last_desc


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_evidence(
    claim: str, articles: list[dict], credibility_scorer: CredibilityScorer,
    top_k: int = config.DEFAULT_TOP_K_EVIDENCE, use_credibility: bool = True,
) -> list[dict]:
    if not articles:
        return []
    # Reuse the relevance score computed during fetch_evidence_articles if
    # present (see _filter_relevant) -- recomputing TF-IDF here against a
    # different, smaller article set would silently change the numbers for
    # the same claim/article pair, since cosine similarity is corpus-
    # relative. Only recompute if scores genuinely weren't attached yet
    # (e.g. this function is called directly in isolation, such as from
    # tests or get_evidence_for_claim's baseline ranking).
    if all("_relevance" in a for a in articles):
        relevances = [a["_relevance"] for a in articles]
    else:
        relevances = score_relevance(claim, articles)
    enriched = []
    for art, rel in zip(articles, relevances):
        art = {k: v for k, v in art.items() if k != "_relevance"}  # drop internal field
        domain = art["domain"]
        # Wikipedia gets a fixed credibility score; all other domains go
        # through the MBFC-derived scorer as before.
        if domain in _WIKIPEDIA_DOMAINS:
            cred_label = "high"
            cred_score = _WIKIPEDIA_CREDIBILITY
        else:
            cred_label = credibility_scorer.label(domain)
            cred_score = credibility_scorer.score(domain)
        weighted = rel * cred_score if use_credibility else rel
        enriched.append({
            **art, "relevance": round(float(rel), 4),
            "credibility_label": cred_label, "credibility_score": cred_score,
            "weighted_score": round(float(weighted), 4),
        })
    enriched.sort(key=lambda x: x["weighted_score"], reverse=True)
    return enriched[:top_k]


def get_evidence_for_claim(
    claim: str, credibility_scorer: CredibilityScorer,
    top_k: int = config.DEFAULT_TOP_K_EVIDENCE, sleep_sec: float = 0.5,
) -> dict:
    articles, query_used = fetch_evidence_articles(claim)
    time.sleep(sleep_sec)
    weighted = rank_evidence(claim, articles, credibility_scorer, top_k, True)
    baseline = rank_evidence(claim, articles, credibility_scorer, top_k, False)
    return {
        "query_used": query_used, "n_articles_found": len(articles),
        "weighted_evidence": weighted, "baseline_evidence": baseline,
    }