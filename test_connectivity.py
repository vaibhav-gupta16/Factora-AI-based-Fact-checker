"""
Standalone connectivity test — isolates the network question from
everything else. Run this FIRST if retrieval seems stuck or slow:

    python test_connectivity.py

It makes one direct call to The Guardian (primary evidence source) and one
to NewsAPI (secondary/fallback), with a short timeout and full diagnostic
output, so you can see exactly what's happening without waiting through a
full ladder of search stages per claim.
"""
import sys
import time
import requests

sys.path.append(".")
import config


def test_guardian():
    print("=" * 60)
    print("Testing The Guardian Open Platform (primary source)...")
    print("=" * 60)
    if not config.GUARDIAN_API_KEY:
        print("[SKIP] GUARDIAN_API_KEY not set in .env.")
        print(f"       Get a free key (emailed instantly): {config.GUARDIAN_SIGNUP}")
        return
    params = {"q": "Bangladesh", "page-size": 3, "api-key": config.GUARDIAN_API_KEY}
    t0 = time.time()
    try:
        resp = requests.get(config.GUARDIAN_URL, params=params, timeout=10)
        elapsed = time.time() - t0
        print(f"\n[RESULT] Responded in {elapsed:.1f}s, HTTP {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            n = len(data.get("response", {}).get("results", []))
            print(f"[RESULT] {n} article(s) returned")
            print("[RESULT] The Guardian IS WORKING.")
        else:
            print(f"[RESULT] Non-200 response: {resp.text[:300]}")
    except requests.exceptions.Timeout:
        print(f"\n[RESULT] TIMED OUT after {time.time()-t0:.1f}s.")
        print("[RESULT] The Guardian is NOT reachable from this network within 10s.")
        print("         Try opening this URL directly in your browser to compare:")
        print(f"         {config.GUARDIAN_URL}?{requests.compat.urlencode(params)}")
    except Exception as e:
        print(f"\n[RESULT] UNEXPECTED ERROR: {type(e).__name__}: {e}")


def test_newsapi():
    print("\n" + "=" * 60)
    print("Testing NewsAPI (secondary / fallback source)...")
    print("=" * 60)
    if not config.NEWSAPI_KEY:
        print("[SKIP] NEWSAPI_KEY not set in .env -- optional, skipping.")
        print(f"       Get a free key: {config.NEWSAPI_SIGNUP}")
        return
    params = {"q": "Bangladesh", "language": "en", "pageSize": 3,
              "apiKey": config.NEWSAPI_KEY}
    t0 = time.time()
    try:
        resp = requests.get(config.NEWSAPI_URL, params=params, timeout=10)
        elapsed = time.time() - t0
        print(f"\n[RESULT] Responded in {elapsed:.1f}s, HTTP {resp.status_code}")
        data = resp.json()
        if data.get("status") == "ok":
            print(f"[RESULT] {len(data.get('articles', []))} article(s) returned")
            print("[RESULT] NewsAPI IS WORKING.")
        else:
            print(f"[RESULT] NewsAPI returned an error: {data.get('message')}")
    except requests.exceptions.Timeout:
        print(f"\n[RESULT] TIMED OUT after {time.time()-t0:.1f}s.")
    except Exception as e:
        print(f"\n[RESULT] UNEXPECTED ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    test_guardian()
    test_newsapi()
    print("\n" + "=" * 60)
    if not config.GUARDIAN_API_KEY and not config.NEWSAPI_KEY:
        print("Neither key is set — Factora has no evidence source configured.")
        print("Set at least one of GUARDIAN_API_KEY or NEWSAPI_KEY in .env.")
    print("Done. Paste this whole output back if either test failed unexpectedly.")
    print("=" * 60)
