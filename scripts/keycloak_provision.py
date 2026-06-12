#!/usr/bin/env python3
"""
Idempotent Keycloak provisioning script for the iFusion Copilot dashboard.

Creates (or re-uses if they already exist):
    - Realm:   tbg-copilot
    - Client:  tbg-copilot-dashboard   (OIDC, confidential, PKCE, auth-code flow)
    - Roles:   tbg-admin, tbg-manager, tbg-viewer
    - User:    a test user with the role you pick, email_verified=true

Re-running is safe — every step checks "does this already exist?" first.

Run from the repo root:

    # one-shot, prompts for password
    python scripts/keycloak_provision.py

    # non-interactive (CI-style)
    KEYCLOAK_ADMIN_USER=admin \\
    KEYCLOAK_ADMIN_PASSWORD=... \\
    python scripts/keycloak_provision.py

Override defaults via flags:

    python scripts/keycloak_provision.py \\
        --base-url https://197.230.47.51:8082 \\
        --realm tbg-copilot \\
        --client-id tbg-copilot-dashboard \\
        --redirect-uri http://localhost:8000/api/v1/auth/keycloak/callback \\
        --redirect-uri http://localhost:8009/api/v1/auth/keycloak/callback \\
        --test-user-email aviral.dayal@ksolves.com \\
        --test-user-password 'choose-a-real-one' \\
        --test-user-role tbg-admin

The script prints the client secret at the end — paste it into .env as
KEYCLOAK_CLIENT_SECRET.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from typing import Any, Optional

import httpx


# ──────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_BASE_URL     = "https://197.230.47.51:8082"
DEFAULT_REALM        = "tbg-copilot"
DEFAULT_CLIENT_ID    = "tbg-copilot-dashboard"
DEFAULT_REDIRECTS    = [
    "http://localhost:8000/api/v1/auth/keycloak/callback",
    "http://localhost:8009/api/v1/auth/keycloak/callback",
]
DEFAULT_WEB_ORIGINS  = ["http://localhost:8000", "http://localhost:8009"]
DEFAULT_ROLES        = ["tbg-admin", "tbg-manager", "tbg-viewer"]


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def log(msg: str, kind: str = "info") -> None:
    glyph = {"info": "·", "ok": "✓", "warn": "!", "err": "✗"}.get(kind, "·")
    print(f"  {glyph}  {msg}")


def fatal(msg: str) -> None:
    print(f"\n✗ {msg}\n", file=sys.stderr)
    sys.exit(1)


def get_admin_token(client: httpx.Client, base_url: str, user: str, password: str) -> str:
    """Authenticate against the master realm with the admin-cli client."""
    url = f"{base_url}/realms/master/protocol/openid-connect/token"
    r = client.post(url, data={
        "grant_type": "password",
        "client_id":  "admin-cli",
        "username":   user,
        "password":   password,
    })
    if r.status_code != 200:
        fatal(f"admin login failed (HTTP {r.status_code}): {r.text[:400]}")
    return r.json()["access_token"]


def ensure_realm(client: httpx.Client, base_url: str, token: str, realm: str) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    r = client.get(f"{base_url}/admin/realms/{realm}", headers=headers)
    if r.status_code == 200:
        log(f"realm '{realm}' already exists", "ok")
        return
    if r.status_code != 404:
        fatal(f"unexpected status checking realm: HTTP {r.status_code} — {r.text[:200]}")
    log(f"creating realm '{realm}'…")
    r = client.post(
        f"{base_url}/admin/realms",
        headers={**headers, "Content-Type": "application/json"},
        json={"realm": realm, "enabled": True, "displayName": "iFusion Copilot"},
    )
    if r.status_code not in (201, 204):
        fatal(f"create realm failed (HTTP {r.status_code}): {r.text[:400]}")
    log(f"realm '{realm}' created", "ok")


def ensure_client(
    client: httpx.Client, base_url: str, token: str, realm: str, *,
    client_id: str, redirect_uris: list[str], web_origins: list[str],
) -> tuple[str, str]:
    """Returns (kc_uuid_for_client, client_secret)."""
    headers = {"Authorization": f"Bearer {token}"}
    r = client.get(
        f"{base_url}/admin/realms/{realm}/clients",
        headers=headers, params={"clientId": client_id},
    )
    r.raise_for_status()
    found = r.json()

    payload = {
        "clientId":                    client_id,
        "name":                        "iFusion Copilot Dashboard",
        "protocol":                    "openid-connect",
        "publicClient":                False,        # confidential client
        "clientAuthenticatorType":     "client-secret",
        "standardFlowEnabled":         True,         # auth-code flow
        "directAccessGrantsEnabled":   False,        # no password grant
        "implicitFlowEnabled":         False,
        "serviceAccountsEnabled":      False,
        "redirectUris":                redirect_uris,
        "webOrigins":                  web_origins,
        "attributes": {
            # Force PKCE on the client — matches what our backend sends.
            "pkce.code.challenge.method": "S256",
        },
    }

    if found:
        kc_uuid = found[0]["id"]
        log(f"client '{client_id}' already exists (id={kc_uuid[:8]}…) — updating settings", "ok")
        r = client.put(
            f"{base_url}/admin/realms/{realm}/clients/{kc_uuid}",
            headers={**headers, "Content-Type": "application/json"},
            json={**payload, "id": kc_uuid},
        )
        if r.status_code not in (200, 204):
            fatal(f"client update failed (HTTP {r.status_code}): {r.text[:400]}")
    else:
        log(f"creating client '{client_id}'…")
        r = client.post(
            f"{base_url}/admin/realms/{realm}/clients",
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
        )
        if r.status_code not in (201, 204):
            fatal(f"create client failed (HTTP {r.status_code}): {r.text[:400]}")
        # Fetch the UUID
        r = client.get(
            f"{base_url}/admin/realms/{realm}/clients",
            headers=headers, params={"clientId": client_id},
        )
        r.raise_for_status()
        kc_uuid = r.json()[0]["id"]
        log(f"client created (id={kc_uuid[:8]}…)", "ok")

    # Fetch the secret
    r = client.get(
        f"{base_url}/admin/realms/{realm}/clients/{kc_uuid}/client-secret",
        headers=headers,
    )
    r.raise_for_status()
    secret = r.json().get("value")
    if not secret:
        # Generate one if missing
        r = client.post(
            f"{base_url}/admin/realms/{realm}/clients/{kc_uuid}/client-secret",
            headers=headers,
        )
        r.raise_for_status()
        secret = r.json()["value"]
        log("generated a new client secret", "ok")
    return kc_uuid, secret


def ensure_roles(client: httpx.Client, base_url: str, token: str, realm: str, roles: list[str]) -> None:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    for role in roles:
        r = client.get(f"{base_url}/admin/realms/{realm}/roles/{role}",
                       headers={"Authorization": f"Bearer {token}"})
        if r.status_code == 200:
            log(f"role '{role}' already exists", "ok")
            continue
        r = client.post(
            f"{base_url}/admin/realms/{realm}/roles",
            headers=headers, json={"name": role},
        )
        if r.status_code not in (201, 204):
            fatal(f"create role '{role}' failed (HTTP {r.status_code}): {r.text[:400]}")
        log(f"role '{role}' created", "ok")


def ensure_test_user(
    client: httpx.Client, base_url: str, token: str, realm: str, *,
    email: str, password: str, role: str,
) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    json_headers = {**headers, "Content-Type": "application/json"}

    # Already exists?
    r = client.get(
        f"{base_url}/admin/realms/{realm}/users",
        headers=headers, params={"email": email, "exact": "true"},
    )
    r.raise_for_status()
    found = r.json()
    if found:
        user_id = found[0]["id"]
        log(f"user '{email}' already exists (id={user_id[:8]}…)", "ok")
    else:
        log(f"creating user '{email}'…")
        r = client.post(
            f"{base_url}/admin/realms/{realm}/users",
            headers=json_headers,
            json={
                "username":      email,
                "email":         email,
                "emailVerified": True,
                "enabled":       True,
                "credentials":   [{"type": "password", "value": password, "temporary": False}],
            },
        )
        if r.status_code not in (201, 204):
            fatal(f"create user failed (HTTP {r.status_code}): {r.text[:400]}")
        # Re-fetch to get id
        r = client.get(
            f"{base_url}/admin/realms/{realm}/users",
            headers=headers, params={"email": email, "exact": "true"},
        )
        r.raise_for_status()
        user_id = r.json()[0]["id"]
        log(f"user created (id={user_id[:8]}…)", "ok")

    # Make sure password is set (in case the user existed without one)
    r = client.put(
        f"{base_url}/admin/realms/{realm}/users/{user_id}/reset-password",
        headers=json_headers,
        json={"type": "password", "value": password, "temporary": False},
    )
    if r.status_code not in (200, 204):
        log(f"could not reset password (HTTP {r.status_code}): {r.text[:200]}", "warn")

    # Assign the role
    r = client.get(f"{base_url}/admin/realms/{realm}/roles/{role}", headers=headers)
    if r.status_code != 200:
        fatal(f"role '{role}' not found in realm")
    role_obj = r.json()
    r = client.post(
        f"{base_url}/admin/realms/{realm}/users/{user_id}/role-mappings/realm",
        headers=json_headers,
        json=[{"id": role_obj["id"], "name": role_obj["name"]}],
    )
    if r.status_code not in (204, 200):
        log(f"role assign returned HTTP {r.status_code} — likely already assigned", "warn")
    else:
        log(f"role '{role}' assigned to '{email}'", "ok")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description="Provision a Keycloak realm + client for iFusion Copilot.")
    p.add_argument("--base-url",   default=os.getenv("KEYCLOAK_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--realm",      default=os.getenv("KEYCLOAK_REALM", DEFAULT_REALM))
    p.add_argument("--client-id",  default=os.getenv("KEYCLOAK_CLIENT_ID", DEFAULT_CLIENT_ID))
    p.add_argument("--redirect-uri", action="append", default=None,
                   help="Valid redirect URI for the client (repeatable). Defaults to localhost:8000 + 8009.")
    p.add_argument("--web-origin", action="append", default=None,
                   help="Allowed web origin (repeatable). Defaults to localhost:8000 + 8009.")
    p.add_argument("--test-user-email",    default=os.getenv("TEST_USER_EMAIL", "test@example.com"))
    p.add_argument("--test-user-password", default=os.getenv("TEST_USER_PASSWORD"))
    p.add_argument("--test-user-role",     default=os.getenv("TEST_USER_ROLE", "tbg-admin"),
                   choices=DEFAULT_ROLES)
    p.add_argument("--no-tls-verify", action="store_true",
                   help="Skip TLS verification (use for self-signed certs).")
    args = p.parse_args()

    base_url      = args.base_url.rstrip("/")
    redirect_uris = args.redirect_uri or DEFAULT_REDIRECTS
    web_origins   = args.web_origin   or DEFAULT_WEB_ORIGINS

    # Admin credentials
    admin_user = os.getenv("KEYCLOAK_ADMIN_USER")
    admin_pass = os.getenv("KEYCLOAK_ADMIN_PASSWORD")
    if not admin_user:
        admin_user = input("Keycloak admin username: ").strip()
    if not admin_pass:
        admin_pass = getpass.getpass("Keycloak admin password: ")
    if not admin_user or not admin_pass:
        fatal("admin username and password are required")

    # Test-user password
    if not args.test_user_password:
        args.test_user_password = getpass.getpass(
            f"Password for test user {args.test_user_email}: "
        )
    if not args.test_user_password:
        fatal("test user password is required")

    # TLS verify default: False for self-signed; honored via --no-tls-verify or env.
    tls_verify = not args.no_tls_verify and os.getenv("KEYCLOAK_TLS_VERIFY", "false").lower() in ("1", "true", "yes")

    print()
    print(f"  Keycloak:    {base_url}")
    print(f"  Realm:       {args.realm}")
    print(f"  Client:      {args.client_id}")
    print(f"  Redirects:   {', '.join(redirect_uris)}")
    print(f"  Web origins: {', '.join(web_origins)}")
    print(f"  Test user:   {args.test_user_email} (role: {args.test_user_role})")
    print(f"  TLS verify:  {tls_verify}")
    print()

    with httpx.Client(verify=tls_verify, timeout=30.0) as client:
        log("logging in as admin…")
        token = get_admin_token(client, base_url, admin_user, admin_pass)
        log("admin token acquired", "ok")

        ensure_realm(client, base_url, token, args.realm)
        kc_uuid, secret = ensure_client(
            client, base_url, token, args.realm,
            client_id=args.client_id,
            redirect_uris=redirect_uris,
            web_origins=web_origins,
        )
        ensure_roles(client, base_url, token, args.realm, DEFAULT_ROLES)
        ensure_test_user(
            client, base_url, token, args.realm,
            email=args.test_user_email,
            password=args.test_user_password,
            role=args.test_user_role,
        )

    # Final summary
    print()
    print("──────────────────────────────────────────────────────────────────")
    print("  DONE.  Paste this into your .env (replacing the placeholder):")
    print("──────────────────────────────────────────────────────────────────")
    print()
    print(f"KEYCLOAK_ENABLED=true")
    print(f"KEYCLOAK_BASE_URL={base_url}")
    print(f"KEYCLOAK_REALM={args.realm}")
    print(f"KEYCLOAK_CLIENT_ID={args.client_id}")
    print(f"KEYCLOAK_CLIENT_SECRET={secret}")
    print(f"KEYCLOAK_REDIRECT_URI={redirect_uris[0]}")
    print(f"KEYCLOAK_POST_LOGIN_REDIRECT=http://localhost:8000/")
    print(f"KEYCLOAK_TLS_VERIFY={'true' if tls_verify else 'false'}")
    print(f'KEYCLOAK_ROLE_MAP={{"tbg-admin":"admin","tbg-manager":"manager","tbg-viewer":"viewer"}}')
    print(f"KEYCLOAK_DEFAULT_ROLE=viewer")
    print()
    print(f"  Test login:  {args.test_user_email}")
    print(f"  Then:        restart uvicorn → hard-reload the dashboard → click 'Sign in with Keycloak'")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
