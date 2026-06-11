"""
Set roles for existing users.
Usage:  python3 set_user_roles.py

Valid roles: admin, executive, manager, viewer
"""
from app.db.auth_store import get_user_by_email, update_user_role

ASSIGNMENTS = [
    ("aviral.dayal@ksolves.com",        "admin"),
    ("youssef.naifi@digiwise.com",       "executive"),
    ("fatoukine.diengsarr@digiwise.io",  "manager"),
]

for email, role in ASSIGNMENTS:
    user = get_user_by_email(email)
    if not user:
        print(f"SKIP   {email} — not found")
        continue
    update_user_role(user["id"], role)
    print(f"OK     {email} → role: {role}")
