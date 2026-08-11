"""
Runs the full project pipeline end to end.
Usage:
    python main.py --skip_pipeline     # offline steps only, no API keys needed
    python main.py --n_samples 30      # full run including live retrieval + verdicts
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config
from src.download_data import main as download_main
from src.eda import run_eda
from src.sql_analysis import build_database, run_queries
from src.pipeline import run_batch
from src.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Run the full Factora pipeline.")
    parser.add_argument("--n_samples", type=int, default=config.DEFAULT_N_SAMPLES)
    parser.add_argument("--top_k", type=int, default=config.DEFAULT_TOP_K_EVIDENCE)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--skip_pipeline", action="store_true")
    parser.add_argument("--claims_source", type=str, default="current", choices=["current", "liar"],
                         help="'current' = verified claims live news can check (default). 'liar' = historical claims, mostly unverifiable.")
    parser.add_argument("--claims_file", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70 + "\nSTEP 1/5: Downloading real datasets\n" + "=" * 70)
    download_main()

    print("\n" + "=" * 70 + "\nSTEP 2/5: Exploratory Data Analysis\n" + "=" * 70)
    run_eda()

    print("\n" + "=" * 70 + "\nSTEP 3/5: SQL analysis\n" + "=" * 70)
    conn = build_database()
    run_queries(conn)
    conn.close()

    if args.skip_pipeline:
        print("\n--skip_pipeline set: stopping here. No API calls made.")
        return

    print("\n" + "=" * 70 + f"\nSTEP 4/5: Fact-checking pipeline ({args.n_samples} claims)\n" + "=" * 70)
    df = run_batch(args.n_samples, args.top_k, args.split,
                    claims_source=args.claims_source, claims_file=args.claims_file)
    df.to_csv(config.RESULTS_DIR / "pipeline_results.csv", index=False)
    n_ok = (df["status"] == "ok").sum()
    print(f"Saved {len(df)} records ({n_ok} succeeded)")

    print("\n" + "=" * 70 + "\nSTEP 5/5: Evaluation\n" + "=" * 70)
    evaluate()
    print("\nAll done. Check outputs/plots/, outputs/results/, and outputs/sql/.")


if __name__ == "__main__":
    main()
