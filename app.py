"""
Factora — Streamlit app.
Run: streamlit run app.py
Requires GROQ_API_KEY in .env. NEWSAPI_KEY and GUARDIAN_API_KEY are optional
but at least one of them should be set — The Guardian is tried first
(primary), NewsAPI is the fallback.
"""
import sys
from pathlib import Path
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
import config
from src.credibility import CredibilityScorer
from src.retrieval import fetch_evidence_articles, rank_evidence, NewsAPIError
from src.verdict import generate_verdict, VerdictError

st.set_page_config(page_title="Factora", page_icon="🔎", layout="centered",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 1.5rem 4rem; max-width: 760px; }

/* ---- Header / wordmark ---- */
.fc-header { margin-bottom: 2.4rem; }
.fc-wordmark {
  display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem;
}
.fc-logo {
  width: 34px; height: 34px; border-radius: 9px;
  background: linear-gradient(135deg, #3b82f6, #6366f1);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem;
}
.fc-header h1 {
  font-size: 1.55rem; font-weight: 800; letter-spacing: -0.03em;
  color: #f8fafc; margin: 0;
}
.fc-header p { font-size: 0.87rem; color: #64748b; margin: 0.4rem 0 0 0; line-height: 1.55; max-width: 46ch; }

/* ---- Input ---- */
.stTextArea textarea {
  font-family: 'Inter', sans-serif !important; font-size: 0.95rem !important;
  background: #0f172a !important; border: 1px solid #1e293b !important; border-radius: 10px !important;
  color: #e2e8f0 !important; padding: 14px !important; line-height: 1.5 !important;
}
.stTextArea textarea:focus { border-color: #3b82f6 !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important; }
.stButton > button {
  background: #3b82f6 !important; color: white !important; border: none !important;
  border-radius: 8px !important; font-size: 0.9rem !important; font-weight: 600 !important;
  padding: 0.65rem 1.7rem !important; transition: background 0.15s, transform 0.1s !important;
}
.stButton > button:hover { background: #2563eb !important; }
.stButton > button:active { transform: scale(0.98); }

/* ---- Evidence cards ---- */
.section-label {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
  color: #475569; margin: 2rem 0 0.85rem 0;
}
.ev-card {
  background: #0f172a; border: 1px solid #1e293b; border-radius: 12px;
  padding: 1.1rem 1.2rem; margin-bottom: 0.7rem; transition: border-color 0.15s;
}
.ev-card:hover { border-color: #2d3d54; }
.source-tag {
  display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
  font-weight: 600; padding: 3px 9px; border-radius: 5px; margin-right: 6px; letter-spacing: 0.02em;
}
.cred-high { background: #052e16; color: #4ade80; }
.cred-mixed { background: #1c1917; color: #fb923c; }
.cred-low { background: #2d0a0a; color: #f87171; }
.cred-unrated { background: #1e293b; color: #94a3b8; }
.score-row { display: flex; gap: 6px; margin-bottom: 0.55rem; flex-wrap: wrap; align-items: center; }
.score-chip {
  font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; background: #1e293b;
  color: #94a3b8; padding: 3px 8px; border-radius: 5px;
}
.rel-strong { background: #064e3b; color: #4ade80; border: 1px solid #166534; }
.rel-mod { background: #1e293b; color: #cbd5e1; border: 1px solid #334155; }
.rel-weak { background: #18181b; color: #71717a; border: 1px solid #27272a; }
.ev-title { font-size: 0.92rem; font-weight: 600; color: #e2e8f0; margin-bottom: 0.4rem; line-height: 1.42; }
.ev-desc { font-size: 0.83rem; color: #94a3b8; line-height: 1.55; margin-bottom: 0.55rem; }
.ev-meta { font-size: 0.76rem; color: #475569; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.ev-link { color: #3b82f6 !important; text-decoration: none !important; font-weight: 500; }
.ev-link:hover { color: #60a5fa !important; }

/* ---- Verdict block ---- */
.fc-divider { height: 1px; background: #1e293b; margin: 2rem 0 1.6rem 0; }
.verdict-block { border-radius: 14px; padding: 1.6rem 1.7rem; }
.verdict-true { background: #052e16; border: 1px solid #166534; }
.verdict-false { background: #2d0a0a; border: 1px solid #7f1d1d; }
.verdict-mixed { background: #1c1400; border: 1px solid #78350f; }
.verdict-unverifiable { background: #0f172a; border: 1px solid #1e293b; }
.verdict-label {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.11em; text-transform: uppercase;
  margin-bottom: 0.35rem;
}
.verdict-label-true { color: #4ade80; } .verdict-label-false { color: #f87171; }
.verdict-label-mixed { color: #fb923c; } .verdict-label-unverifiable { color: #64748b; }
.verdict-conf {
  font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1; margin-bottom: 0.85rem;
}
.verdict-conf-true { color: #4ade80; } .verdict-conf-false { color: #f87171; }
.verdict-conf-mixed { color: #fb923c; } .verdict-conf-unverifiable { color: #475569; }
.verdict-reasoning { font-size: 0.89rem; color: #cbd5e1; line-height: 1.62; }

/* ---- Confidence breakdown ---- */
.conf-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 1.1rem; }
.conf-cell {
  background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
  border-radius: 9px; padding: 0.75rem; text-align: center;
}
.conf-cell-val { font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 700; color: #e2e8f0; }
.conf-cell-label { font-size: 0.7rem; color: #64748b; margin-top: 3px; letter-spacing: 0.02em; }
.conf-formula { font-size: 0.71rem; color: #475569; margin-top: 0.7rem; }

/* ---- Footer ---- */
.fc-footer { margin-top: 3rem; font-size: 0.75rem; color: #334155; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="fc-header">
  <div class="fc-wordmark">
    <div class="fc-logo">🔎</div>
    <h1>Factora</h1>
  </div>
  <p>Retrieves real news evidence, weighs each source by credibility, and returns a verdict with a confidence score computed from that evidence.</p>
</div>
""", unsafe_allow_html=True)

if not config.GROQ_API_KEY:
    st.warning(f"GROQ_API_KEY missing — add it to your .env file. Free at {config.GROQ_SIGNUP}")

if not config.GUARDIAN_API_KEY and not config.NEWSAPI_KEY:
    st.warning(
        f"No evidence source configured — add at least one of GUARDIAN_API_KEY "
        f"(free at {config.GUARDIAN_SIGNUP}, key emailed instantly) or "
        f"NEWSAPI_KEY (free at {config.NEWSAPI_SIGNUP}) to your .env file."
    )

claim = st.text_area("Claim to check", height=100,
                      placeholder="Paste or type any factual claim here…",
                      label_visibility="collapsed")

check = st.button("Check claim", type="primary")

with st.expander("Advanced: override search query"):
    manual_query = st.text_input("Custom search query (leave blank for automatic)",
                                  value="", label_visibility="collapsed",
                                  placeholder='e.g. "Sheikh Hasina" Bangladesh')

if check and claim.strip():
    cred_scorer = CredibilityScorer()
    with st.spinner("Understanding the claim and searching for evidence…"):
        try:
            articles, query_used = fetch_evidence_articles(
                claim.strip(), page_size=10, query=manual_query.strip() or None)
        except NewsAPIError as e:
            st.error(str(e)); st.stop()

    weighted_ev = rank_evidence(claim.strip(), articles, cred_scorer, top_k=5, use_credibility=True)

    if weighted_ev:
        st.markdown('<div class="section-label">Evidence</div>', unsafe_allow_html=True)
        for e in weighted_ev:
            cl = e["credibility_label"]
            title = e["title"] or "(no title)"
            desc = e["description"] or ""
            url = e["url"] or "#"
            source = e["source_name"] or e["domain"]
            rel_val = e.get("relevance", 0.0)
            if rel_val >= 0.25:
                rel_cls, rel_txt = "rel-strong", f"topic match: {rel_val:.2f} (strong)"
            elif rel_val >= 0.12:
                rel_cls, rel_txt = "rel-mod", f"topic match: {rel_val:.2f} (moderate)"
            else:
                rel_cls, rel_txt = "rel-weak", f"topic match: {rel_val:.2f} (weak/tangential)"

            st.markdown(f"""
<div class="ev-card">
  <div class="score-row">
    <span class="source-tag cred-{cl}">source: {cl}</span>
    <span class="score-chip {rel_cls}">{rel_txt}</span>
    <span class="score-chip">cred {e['credibility_score']:.2f}</span>
    <span class="score-chip">score {e['weighted_score']:.3f}</span>
  </div>
  <div class="ev-title">{title}</div>
  {"<div class='ev-desc'>" + desc[:220] + ("…" if len(desc) > 220 else "") + "</div>" if desc else ""}
  <div class="ev-meta"><span>{source}</span><a href="{url}" target="_blank" class="ev-link">Read article ↗</a></div>
</div>""", unsafe_allow_html=True)

    with st.spinner("Generating verdict…"):
        try:
            result = generate_verdict(claim.strip(), weighted_ev, show_credibility=True)
        except VerdictError as e:
            st.error(str(e)); st.stop()

    v, conf, reasoning = result["verdict"], result["confidence"], result["reasoning"]
    bd = result.get("confidence_breakdown", {})
    suspicious = result.get("no_coverage_is_suspicious", False)
    label_map = {"true": "Likely true", "false": "Likely false",
                 "mixed": "Mixed evidence", "unverifiable": "Cannot verify"}
    display_label = "Cannot verify — but suspicious" if (v == "unverifiable" and suspicious) else label_map.get(v, v)

    st.markdown(f"""
<div class="fc-divider"></div>
<div class="verdict-block verdict-{v}">
  <div class="verdict-label verdict-label-{v}">{display_label}</div>
  <div class="verdict-conf verdict-conf-{v}">{int(conf * 100)}% confidence</div>
  <div class="verdict-reasoning">{reasoning}</div>
""", unsafe_allow_html=True)

    if bd and bd.get("n_supporting_sources", 0) > 0:
        n_pub = bd.get('n_distinct_publishers', bd['n_supporting_sources'])
        st.markdown(f"""
  <div class="conf-grid">
    <div class="conf-cell"><div class="conf-cell-val">{bd['n_supporting_sources']}</div><div class="conf-cell-label">sources</div></div>
    <div class="conf-cell"><div class="conf-cell-val">{n_pub}</div><div class="conf-cell-label">publishers</div></div>
    <div class="conf-cell"><div class="conf-cell-val">{bd['avg_credibility']:.2f}</div><div class="conf-cell-label">avg credibility</div></div>
    <div class="conf-cell"><div class="conf-cell-val">{bd['avg_relevance']:.2f}</div><div class="conf-cell-label">avg relevance</div></div>
  </div>
  <div class="conf-formula">Confidence = 35% credibility + 25% relevance + 20% corroboration + 20% source diversity</div>
""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="fc-footer">Factora — built on The Guardian, Google News, NewsAPI, Wikipedia, and Groq</div>', unsafe_allow_html=True)
