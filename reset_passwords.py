"""
Reset passwords for existing users.
Usage:  python3 reset_passwords.py
"""
from app.auth.jwt_utils import hash_password
from app.db.auth_store import get_pool, ensure_schema

USERS = [
    ("aviral.dayal@ksolves.com",        "Aviral@250302"),
    ("youseef.naifi@digiwise.com",       "Youseef@123"),
    ("fatoukine.diengsarr@digiwise.io",  "Fatou@123"),
]

ensure_schema()
pool = get_pool()
conn = pool.getconn()
try:
    with conn.cursor() as cur:
        for email, pw in USERS:
            cur.execute(
                "UPDATE public.copilot_users SET password_hash = %s WHERE email = %s",
                (hash_password(pw), email.lower().strip()),
            )
            print(f"RESET {email} — {cur.rowcount} row updated")
    conn.commit()
finally:
    pool.putconn(conn)
