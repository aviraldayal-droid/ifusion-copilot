"""
Two-part rate-limit smoke test.

PART 1 — Your FastAPI /db/chat rate limit
    Bucket: 30 requests / 60s, keyed by authenticated user (see
    app/middleware/rate_limit.py:33). Logs in once, then fires 35
    chat requests concurrently. The first ~30 should return 200,
    the rest should return 429 instantly.

    Cost note: the 30 that pass DO call Ollama Cloud (real LLM round
    trip). The 5 over the limit are instant (rate-limit dep runs
    before the route handler). To keep cost down we send a tiny
    prompt and let the agent answer however it wants.

PART 2 — Ollama Cloud's upstream rate limit
    Hits OLLAMA_BASE_URL directly with OLLAMA_API_KEY. Ollama Cloud
    publishes per-key limits that aren't documented in this repo, so
    this part is exploratory — we fire bursts of 20 / 50 / 100 with
    minimal prompts (num_predict=1) and report any non-200 codes.

Run:
    # Make sure the FastAPI server is up first:
    #   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    python test_rate_limit_ollama.py
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import requests

# Pull config from the app so we use the same key/model the server uses.
sys.path.insert(0, ".")
from app.config.settings import settings  # noqa: E402

# ---------------------------------------------------------------------------
# Config — edit if needed
# ---------------------------------------------------------------------------
BASE_URL = "http://localhost:8000"
LOGIN_URL = f"{BASE_URL}/api/v1/auth/login"
CHAT_URL = f"{BASE_URL}/api/v1/db/chat"

LOGIN_EMAIL = "aviral.dayal@ksolves.com"
LOGIN_PASSWORD = "Aviral@250302"

CHAT_BUCKET_LIMIT = 30          # matches RATE_LIMITS["chat"][0]
CHAT_ATTEMPTS = 35              # > limit so we see 429s
CHAT_CONCURRENCY = 35           # fire them ~simultaneously so the limit hits fast

OLLAMA_BURSTS = [20, 50, 100]   # escalate until we see a 429 (or exhaust the list)
OLLAMA_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Part 1: /db/chat (your middleware)
# ---------------------------------------------------------------------------
def login() -> str | None:
    """Returns access_token, or None on failure."""
    try:
        r = requests.post(
            LOGIN_URL,
            json={"email": LOGIN_EMAIL, "password": LOGIN_PASSWORD},
            timeout=10,
        )
    except requests.RequestException as e:
        print(f"[login] connection error: {e}")
        return None

    if r.status_code == 429:
        retry = r.headers.get("Retry-After", "?")
        print(f"[login] BLOCKED by login bucket (5/min/IP). Wait {retry}s and re-run.")
        return None
    if r.status_code != 200:
        print(f"[login] HTTP {r.status_code}: {r.text[:200]}")
        return None
    token = r.json().get("access_token")
    print(f"[login] OK — got token ({len(token)} chars)")
    return token


def fire_chat(idx: int, token: str) -> tuple[int, int, str]:
    """Returns (idx, status_code, short_body)."""
    try:
        r = requests.post(
            CHAT_URL,
            json={"message": "ping", "language": "en"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=120,
        )
    except requests.RequestException as e:
        return idx, -1, f"ERR {e}"
    return idx, r.status_code, r.text[:80].replace("\n", " ")


def part1_chat_endpoint(token: str) -> None:
    print("\n" + "=" * 70)
    print(f"PART 1 — POST {CHAT_URL}")
    print(f"Bucket: {CHAT_BUCKET_LIMIT}/60s per user. Firing {CHAT_ATTEMPTS} "
          f"concurrent requests.")
    print("=" * 70)
    print("(The 200s call the real LLM — this may take a while. 429s return instantly.)")

    t0 = time.monotonic()
    results: list[tuple[int, int, str]] = []
    with ThreadPoolExecutor(max_workers=CHAT_CONCURRENCY) as pool:
        futures = [pool.submit(fire_chat, i, token) for i in range(1, CHAT_ATTEMPTS + 1)]
        for f in as_completed(futures):
            results.append(f.result())

    results.sort(key=lambda x: x[0])
    counts = Counter(s for _, s, _ in results)

    for idx, status, body in results:
        tag = (
            "OK   " if status == 200
            else "LIMIT" if status == 429
            else "ERR  " if status == -1
            else "????"
        )
        print(f"  [{idx:>2}] {tag} HTTP {status}  {body}")

    elapsed = time.monotonic() - t0
    print(f"\n  Summary: {dict(counts)}   elapsed={elapsed:.1f}s")
    if counts.get(200, 0) <= CHAT_BUCKET_LIMIT and counts.get(429, 0) > 0:
        print("  PASS — middleware blocked excess requests with 429.")
    else:
        print("  UNEXPECTED — check server logs.")


# ---------------------------------------------------------------------------
# Part 2: Ollama Cloud direct
# ---------------------------------------------------------------------------
def fire_ollama(idx: int) -> tuple[int, int, str]:
    """Tiny direct call to Ollama Cloud's /api/generate. Returns (idx, status, body[:120])."""
    try:
        r = requests.post(
            f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            headers={"Authorization": f"Bearer {settings.OLLAMA_API_KEY}"},
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": "hi",
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=OLLAMA_TIMEOUT,
        )
    except requests.RequestException as e:
        return idx, -1, f"ERR {e}"
    return idx, r.status_code, r.text[:120].replace("\n", " ")


def part2_ollama_direct() -> None:
    print("\n" + "=" * 70)
    print(f"PART 2 — POST {settings.OLLAMA_BASE_URL}/api/generate (direct)")
    print(f"Model: {settings.OLLAMA_MODEL}   prompt='hi'   num_predict=1")
    print("=" * 70)

    if not settings.is_ollama_cloud:
        print("OLLAMA_API_KEY is empty — you're on local Ollama, no upstream limit to test.")
        return

    for burst in OLLAMA_BURSTS:
        print(f"\n  Burst of {burst} concurrent requests...")
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=burst) as pool:
            futures = [pool.submit(fire_ollama, i) for i in range(1, burst + 1)]
            results = [f.result() for f in as_completed(futures)]
        elapsed = time.monotonic() - t0
        counts = Counter(s for _, s, _ in results)
        print(f"  Statuses: {dict(counts)}   elapsed={elapsed:.1f}s")

        non_200 = [(i, s, b) for i, s, b in results if s != 200]
        if non_200:
            print(f"  → Saw {len(non_200)} non-200 responses. Samples:")
            for i, s, b in non_200[:5]:
                print(f"      [{i}] HTTP {s}  {b}")
            print("  → Ollama Cloud is throttling. Stop here.")
            return
        else:
            print("  → All 200. No upstream throttle hit at this burst size.")

    print("\n  Reached max burst without triggering an upstream limit. Either your "
          "key's limit is higher than the tested bursts, or the limits are token-"
          "based rather than request-based.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"OLLAMA_BASE_URL = {settings.OLLAMA_BASE_URL}")
    print(f"OLLAMA_MODEL    = {settings.OLLAMA_MODEL}")
    print(f"On cloud?       = {settings.is_ollama_cloud}")

    token = login()
    if not token:
        print("\nAbort — can't proceed to Part 1 without a token.")
        return 1

    part1_chat_endpoint(token)
    part2_ollama_direct()
    return 0


if __name__ == "__main__":
    sys.exit(main())
