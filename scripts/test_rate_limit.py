"""
Smoke test for the rate-limit middleware in app/middleware/rate_limit.py.

Hits POST /api/v1/auth/login with bad credentials in a tight loop. The
'login' bucket is configured at 5 requests / 60s per IP, so the first 5
calls should return 401 and the rest should return 429 with a Retry-After
header.

Run the server first:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Then in another terminal:
    python test_rate_limit.py
"""
import sys
import time

import requests

BASE_URL = "http://localhost:8000"
LOGIN_URL = f"{BASE_URL}/api/v1/auth/login"
BUCKET_LIMIT = 5          # must match RATE_LIMITS["login"][0]
WINDOW_SECONDS = 60       # must match RATE_LIMITS["login"][1]
ATTEMPTS = 8              # > limit so we see the cutover

# Bogus creds — we only care about the status code, not the auth result.
PAYLOAD = {"email": "aviral.dayal@ksolves.com", "password": "Aviral@250302"}


def fire(i: int) -> int:
    try:
        r = requests.post(LOGIN_URL, json=PAYLOAD, timeout=5)
    except requests.RequestException as e:
        print(f"[{i:>2}] ERROR connecting: {e}")
        return -1

    retry_after = r.headers.get("Retry-After", "-")
    tag = "OK   " if r.status_code in (200, 401) else "LIMIT" if r.status_code == 429 else "????"
    body = r.text[:120].replace("\n", " ")
    print(f"[{i:>2}] {tag} HTTP {r.status_code}  Retry-After={retry_after}  body={body}")
    return r.status_code


def main() -> int:
    print(f"Probing {LOGIN_URL}")
    print(f"Expecting: first {BUCKET_LIMIT} -> 401, then 429 with Retry-After.\n")

    statuses = []
    for i in range(1, ATTEMPTS + 1):
        statuses.append(fire(i))

    pre_limit = statuses[:BUCKET_LIMIT]
    post_limit = statuses[BUCKET_LIMIT:]

    ok = (
        all(s in (401, 200) for s in pre_limit if s != -1)
        and all(s == 429 for s in post_limit if s != -1)
    )

    print()
    if ok:
        print(f"PASS: rate limiting kicked in after {BUCKET_LIMIT} requests.")
    else:
        print("FAIL: cutover did not match expectation.")
        print(f"  first {BUCKET_LIMIT} statuses: {pre_limit}")
        print(f"  remaining statuses:           {post_limit}")
        print("  (Common causes: server not running, RATE_LIMITS edited,")
        print("   bucket already filled from a previous run — wait 60s and retry.)")

    print(
        f"\nNote: the bucket holds state in-memory for {WINDOW_SECONDS}s. "
        "Restart uvicorn or wait the window out before re-running."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
