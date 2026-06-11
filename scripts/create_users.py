"""
Run once to seed initial users into copilot_users table.
Usage:  python3 create_users.py
"""
from app.auth.jwt_utils import hash_password
from app.db.auth_store import create_user, get_user_by_email

USERS = [
    ("aviral.dayal@ksolves.com",         "Aviral",               "Aviral@250302"),
    ("youseef.naifi@digiwise.com",        "Youseef Naifi",        "Youseef@123"),
    ("fatoukine.diengsarr@digiwise.io",   "Fatoukine Dieng Sarr", "Fatou@123"),
]

for email, name, password in USERS:
    if get_user_by_email(email):
        print(f"SKIP  {email} — already exists")
    else:
        user = create_user(email, name, hash_password(password))
        print(f"OK    {email} — created (id={user['id']})")
