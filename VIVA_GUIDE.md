# Factora — Viva & Project Defense Guide

This guide is structured to help you master your project presentation and viva examination. It covers the **WHAT**, **WHY**, and **HOW** of the architecture, along with anticipated examiner questions and concise, technical answers.

---

## 1. Quick Project Summary (Elevator Pitch)

> **"Factora is an evidence-grounded, hallucination-resistant fact-checking system. Instead of asking an LLM to guess whether a claim is true or invent a confidence percentage, Factora retrieves live evidence across independent news aggregators and encyclopedias, filters them with TF-IDF cosine similarity, weights them against an MBFC credibility corpus, and computes the final confidence score using a deterministic mathematical formula in Python."**

---

## 2. THE "WHAT" (Problem & Solution)

### The Core Problem
1. **LLM Hallucination**: Standalone Large Language Models answer from static weights, frequently hallucinating citations, outdated facts, or fabricated statistics.
2. **Arbitrary Confidence Scores**: When asked *"Rate your confidence from 0 to 100%"*, LLMs exhibit poor probability calibration—often reporting 95% confidence on false information.
3. **Single-Source Bias & Paywalls**: Traditional fact-checkers rely on single news APIs that hit strict rate limits or only represent a single editorial viewpoint.

### Factora's Solution
* **4-Way Verdict Classification**: Evaluates claims into **`true`**, **`false`**, **`mixed`**, or **`unverifiable`**.
* **Live Evidence Retrieval**: Pulls from Google News RSS (wire services), The Guardian, NewsAPI, and Wikipedia.
* **Deterministic Confidence Formula**: Strips confidence scoring entirely away from the LLM, computing it via measurable mathematical signals.
* **Dual Architecture**: Includes an interactive **Streamlit Web UI** for end-users and an offline **Data Science Pipeline** (EDA, SQLite queries, and LIAR benchmark evaluation).

---

## 3. THE "WHY" (Design Decisions & Justifications)

### Q1: Why not let the LLM calculate the confidence score?
* **Answer**: LLMs are autoregressive next-token predictors; they have no internal mechanism to compute mathematical probabilities. They suffer from severe **overconfidence bias** and invent numbers based on rhetoric rather than evidence volume. By computing confidence in Python, the score is 100% reproducible, auditable, and un-hallucinated.

### Q2: Why use a 4-signal confidence formula?
$$\text{Confidence} = 0.35 \times \text{Credibility} + 0.25 \times \text{Relevance} + 0.20 \times \text{Corroboration} + 0.20 \times \text{Diversity}$$
* **35% Source Credibility**: Evidence from Reuters or BBC must mathematically count more than an unverified blog.
* **25% Topic Relevance**: The retrieved snippet must genuinely address the claim, measured via TF-IDF cosine similarity.
* **20% Corroboration**: Rewards multiple articles corroborating the story (saturates at 4 articles).
* **20% Source Diversity**: Distinct publishers (e.g. Guardian + BBC + AP) ensure the story is not an echo chamber from a single outlet (saturates at 3 publishers).

### Q3: Why Google News RSS + Guardian + Wikipedia?
* **Google News RSS**: Aggregates across all wire services (BBC, AP, Reuters, Washington Post) in a single request with **zero API keys and no rate limits**.
* **The Guardian**: Authoritative, in-depth investigative reporting.
* **Wikipedia (Batched)**: Serves as the encyclopedic fallback for scientific, historical, or astronomical facts (e.g. *Earth's orbit*, *speed of light*) that no longer appear in daily news cycles.

### Q4: Why TF-IDF cosine similarity instead of dense vector embeddings?
* **Speed & Zero Overhead**: TF-IDF computes instantly on local CPU without loading heavy multi-gigabyte neural models.
* **Exact Keyword Sensitivity**: Fact-checking often hinges on specific named entities, numbers, and dates (e.g., "$50 billion", "2024", "NASA"). Dense embeddings often blur distinctive numbers into generic embeddings, whereas TF-IDF penalizes mismatches.

---

## 4. THE "HOW" (Step-by-Step Architecture)

```text
[Step 1: Input] Claim string entered
       │
[Step 2: Term Extraction] LLM parses claim into core phrases + keywords
       │                  (Regex fallback ensures 100% uptime)
       │
[Step 3: Multi-Source Retrieval Ladder]
       ├─► Guardian API (Journalism)
       ├─► Google News RSS (Multi-publisher aggregator)
       ├─► NewsAPI (Global news fallback)
       └─► Wikipedia API (Batched encyclopaedic fallback)
       │
[Step 4: Relevance Filtering]
       └─► TF-IDF vectorizer computes cosine similarity.
       └─► Snippets below relevance threshold are discarded.
       │
[Step 5: Credibility Scoring]
       └─► Publisher domain mapped to MBFC dataset (859 domains).
       └─► Weighted score = Relevance × Domain Credibility.
       │
[Step 6: LLM Verdict Direction]
       └─► Groq (openai/gpt-oss-120b) receives claim + ranked snippets.
       └─► Returns JSON: verdict direction, supporting IDs, reasoning.
       │
[Step 7: Python Confidence Computation]
       └─► Python evaluates supporting items: credibility, relevance,
           corroboration count, and distinct publisher diversity.
```

---

## 5. Top 15 Viva Questions & Model Answers

### 1. What is the role of the LLM in your project?
> **Answer**: The LLM is restricted to two cognitive tasks: (1) Understanding search intent to generate query terms from the claim, and (2) Reading the retrieved evidence to determine verdict direction (`true`, `false`, `mixed`, `unverifiable`) and citing supporting snippet numbers. It is strictly prohibited from inventing facts or generating confidence scores.

### 2. How do you prevent hallucinations?
> **Answer**: Through strict Retrieval-Augmented Generation (RAG) constraints. The LLM is given numbered evidence snippets in its system prompt and instructed that if evidence is absent or tangential, it must output `unverifiable`. The final confidence score is calculated mathematically outside the model.

### 3. What dataset did you use for credibility scoring?
> **Answer**: An established academic corpus derived from Media Bias/Fact Check (MBFC), originally published by Baly et al. (ACL 2020). It maps 859 global news domains to verified factual reliability ratings (low: 0.25, mixed: 0.50, high: 1.0, and Wikipedia: 0.75).

### 4. What is the LIAR dataset and how is it used here?
> **Answer**: LIAR is a benchmark of 12,836 political statements collected by William Yang Wang (ACL 2017). In Factora, LIAR is used for offline Exploratory Data Analysis (EDA), SQLite database queries, and testing the limits of live retrieval on historical claims.

### 5. Why do historical claims from LIAR produce "unverifiable" in live search?
> **Answer**: LIAR contains political statements from 2007–2016 without publication timestamps. Live news APIs (The Guardian, Google News) index recent coverage. Searching live news for a local 2009 statement returns no modern news articles, which Factora correctly flags as `unverifiable` rather than hallucinating an answer.

### 6. How do you evaluate the fact-checking performance?
> **Answer**: We compare two pipelines in `evaluate.py`:
> 1. **Baseline**: Equal weighting (pure TF-IDF relevance without source credibility).
> 2. **Credibility-Weighted**: TF-IDF relevance scaled by the domain's credibility score.
> We calculate Accuracy, Macro Precision, Recall, and F1-score, and plot confusion matrices.

### 7. What happens if an API is down or the network is offline?
> **Answer**: The system has cascading resilience:
> - If Guardian fails, it checks Google News RSS.
> - If Google News RSS fails, it checks NewsAPI.
> - If news returns no results, it falls back to Wikipedia.
> - If Groq LLM extraction fails, it falls back to a regex-based noun-phrase extractor.

### 8. How did you optimize Wikipedia retrieval to avoid 429 rate limits?
> **Answer**: We implemented a **batched query system**. Instead of sending sequential HTTP requests for every search hit, we gather the top `pageids` and request all introductory sections in a single batched HTTP call (`prop=extracts&pageids=id1|id2|...`). This reduced network requests by 80% and eliminated HTTP 429 errors.

### 9. What is "Source Diversity" and why does it matter?
> **Answer**: If three supporting articles all come from `theguardian.com`, that represents a single editorial perspective. If the articles come from `bbc.com`, `reuters.com`, and `apnews.com`, the claim has independent verification across diverse organizations. The diversity score rewards this by counting distinct publishers.

### 10. Why is confidence 0% when the verdict is "unverifiable"?
> **Answer**: By definition, `unverifiable` means the system could not find evidence proving or disproving the statement. Assigning any confidence percentage greater than zero to an unverifiable claim would be deceptive.

### 11. Can Factora detect false claims?
> **Answer**: Yes. When fact-checking reports or news articles explicitly refute a rumor (e.g., *"COVID-19 vaccines contain microchips"*), the evidence contains counter-arguments. The LLM marks the claim as `false` citing those snippets, and Python computes a confidence score based on the credibility and relevance of the refuting evidence.

### 12. What does "Mixed" verdict signify?
> **Answer**: A `mixed` verdict is returned when a claim contains half-truths, misleading statistics, or accurate facts presented in a deceptive context (e.g. *"Electric cars produce zero lifecycle emissions"* — true for driving emissions, false for battery manufacturing).

### 13. What is the SQLite database in the project used for?
> **Answer**: In `src/sql_analysis.py`, we load the raw LIAR dataset into an indexed SQLite database (`data/processed/factora.db`). We execute analytical SQL queries to inspect claim distribution by political party, subject frequency, speaker credibility history, and truth ratios.

### 14. What tech stack did you use?
> **Answer**:
> - **Language**: Python 3.10+
> - **Inference Engine**: Groq Cloud API (`openai/gpt-oss-120b` or `qwen/qwen3.6-27b`)
> - **NLP & Retrieval**: scikit-learn (TF-IDF), feedparser (RSS parsing), requests
> - **Database & Analysis**: SQLite3, Pandas, Matplotlib
> - **Web Interface**: Streamlit with custom CSS design tokens

### 15. What are the limitations and future work?
> **Answer**:
> - **Current Limitations**: Text-only; does not parse audio/video or deepfakes. Paywalled news sites (e.g. WSJ, FT) cannot be indexed without subscription APIs.
> - **Future Work**: Integration with ClaimReview schema (schema.org), multi-modal image verification, and multilingual fact-checking for non-English claims.

---

## 6. Checklist Before Your Viva

- [ ] Run `python test_connectivity.py` to ensure all APIs are active.
- [ ] Have Streamlit running (`streamlit run app.py`).
- [ ] Prepare 3 demo claims to test live:
  1. **True Fact**: *"NASA successfully launched the James Webb Space Telescope"*
  2. **False Myth**: *"COVID-19 vaccines contain 5G microchips"*
  3. **Unverifiable Statement**: *"Aliens landed in a private backyard last night"*
- [ ] Be ready to point to [`src/confidence.py`](file:///d:/S/Fact%20Checker/factora/src/confidence.py) to show the exact mathematical formula.
