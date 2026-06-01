"""
Create test users for RBAC role validation.
Each user has a predictable role + password so you can log in to different
browsers/sessions to compare what each role can see.

Usage:  python3 create_test_users.py
"""
from app.auth.jwt_utils import hash_password
from app.db.auth_store import (
    create_user,
    get_user_by_email,
    update_user_role,
    update_session_token,
)

TEST_USERS = [
    # (email,                                   name,            password,         role)
    ("test.executive@digiwise.test",           "Test Executive", "Exec@test123",   "executive"),
    ("test.manager@digiwise.test",             "Test Manager",   "Mgr@test123",    "manager"),
    ("test.viewer@digiwise.test",              "Test Viewer",    "View@test123",   "viewer"),
]

print("=" * 64)
print("Creating / updating test users for RBAC validation")
print("=" * 64)

for email, name, password, role in TEST_USERS:
    existing = get_user_by_email(email)
    if existing:
        update_user_role(existing["id"], role)
        # Clear session so the user is forced to re-login and pick up new role
        update_session_token(existing["id"], "")
        print(f"UPDATED  {email:<40}  role={role}")
    else:
        user = create_user(email, name, hash_password(password))
        update_user_role(user["id"], role)
        print(f"CREATED  {email:<40}  role={role}")

print()
print("=" * 64)
print("Login credentials (use a different browser per role):")
print("=" * 64)
for email, _name, password, role in TEST_USERS:
    print(f"  {role:<10}  {email:<40}  {password}")
print()
print("To test, sign in as each user in a separate browser (or incognito)")
print("and run the same question to see how the response differs by role.")
