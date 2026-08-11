"""
SQLite export + SQL analysis (joins, aggregation, window functions).
"""
import sqlite3
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import load_liar_all, load_mbfc

DB_PATH = config.DATA_PROCESSED / "factcheck.db"

QUERIES = {
    "1_claims_per_party": """
        SELECT party, COUNT(*) AS n_claims,
               SUM(CASE WHEN label_3way = 'false' THEN 1 ELSE 0 END) AS n_false,
               ROUND(100.0 * SUM(CASE WHEN label_3way = 'false' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_false
        FROM claims WHERE party IS NOT NULL AND party != ''
        GROUP BY party HAVING COUNT(*) >= 20 ORDER BY n_claims DESC LIMIT 10;
    """,
    "2_credibility_band_counts": """
        SELECT fact AS credibility_band, COUNT(*) AS n_domains
        FROM mbfc_sources GROUP BY fact ORDER BY n_domains DESC;
    """,
    "3_speaker_truthfulness_ranking": """
        SELECT party, speaker, n_claims,
               RANK() OVER (PARTITION BY party ORDER BY n_claims DESC) AS rank_in_party
        FROM (SELECT party, speaker, COUNT(*) AS n_claims FROM claims
              WHERE party IS NOT NULL AND party != '' GROUP BY party, speaker)
        WHERE n_claims >= 5 ORDER BY party, rank_in_party LIMIT 30;
    """,
    "4_pipeline_join_example": """
        SELECT r.claim_id, c.speaker, c.party, r.true_label_3way,
               r.pred_label_weighted, r.pred_label_baseline, r.avg_credibility_weighted
        FROM pipeline_results r JOIN claims c ON c.id = r.claim_id LIMIT 20;
    """,
}


def build_database() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    load_liar_all().to_sql("claims", conn, if_exists="replace", index=False)
    load_mbfc().to_sql("mbfc_sources", conn, if_exists="replace", index=False)
    results_path = config.RESULTS_DIR / "pipeline_results.csv"
    if results_path.exists():
        # Same dtype fix as evaluate.py — see comment there.
        pd.read_csv(results_path, dtype={
            "true_label_3way": str, "pred_label_weighted": str, "pred_label_baseline": str,
        }).to_sql("pipeline_results", conn, if_exists="replace", index=False)
    return conn


def run_queries(conn: sqlite3.Connection) -> None:
    out_lines = []
    for name, sql in QUERIES.items():
        try:
            df = pd.read_sql_query(sql, conn)
            out_lines.append(f"=== {name} ===\n{sql.strip()}\n\n{df.to_string(index=False)}\n")
            print(f"[ok] {name}: {len(df)} rows")
        except Exception as e:
            out_lines.append(f"=== {name} ===\n{sql.strip()}\n\n(skipped: {e})\n")
            print(f"[skip] {name}: {e}")
    (config.SQL_DIR / "sql_query_results.txt").write_text("\n\n".join(out_lines))
    (config.SQL_DIR / "queries.sql").write_text(
        "\n\n".join(f"-- {name}\n{sql.strip()}" for name, sql in QUERIES.items()))


if __name__ == "__main__":
    conn = build_database()
    print(f"[ok] SQLite database built at {DB_PATH}")
    run_queries(conn)
    conn.close()
