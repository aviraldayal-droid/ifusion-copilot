# iFusion Copilot — Roles & Access Control

## Overview

Every user is assigned one of four roles. The role controls which sheets, sections, and topics the AI can answer questions about. Blocked queries return a polite refusal and are logged for audit.

---

## Roles at a Glance

| Role | Sheets | Restricted Areas | Manage Users |
|---|---|---|---|
| **Admin** | All | None | Yes |
| **Executive** | All | None | No |
| **Manager** | All | Personnel costs, Management fees, Treasury | No |
| **Viewer** | 7 sheets (KPIs only) | Personnel, Mgmt fees, Treasury, Frais généraux, Taxes | No |

---

## Role Details

### Admin
Full access to all data and system features, including creating users and assigning roles.

### Executive *(CEO / CFO / COO)*
Full access to all financial data. Cannot create or manage user accounts.

### Manager *(Finance / Department Head)*
Access to all sheets — P&L, revenue, OPEX, CAPEX, subscribers, traffic.

**Restricted from:**
- Personnel section (individual salary lines)
- Management fees (executive compensation, SHL fees)
- Provisions clients (legal provisions / write-offs)
- Any question containing: `individual salary`, `executive compensation`, `severance`, `treasury position`, `cash position`, `management fee`, `SHL`, `shareholder loan`

### Viewer *(Read-only Analyst)*
High-level KPIs only. Default role assigned to new users.

**Accessible sheets:**
`pnl_conso` · `ca_mobile` · `parc_mobile` · `data_mobile` · `mobile_money` · `trafic_mobile` · `marge_mobile`

**Restricted from:**
- `cash_conso` (entire treasury sheet)
- Sections: Personnel, Management fees, Provisions clients, Frais généraux, Impôts & taxes
- Any question containing: `salary`, `compensation`, `treasury`, `cash position`, `cash flow`, `provision`, `severance`, `personnel`, `line item`, `detail by employee`

---

## Sample Questions by Role

### Executive / Admin — everything allowed

| Question | Result |
|---|---|
| What is the total salary cost for personnel? | Allowed |
| What is our treasury position? | Allowed |
| Show me management fees paid this quarter | Allowed |
| What is the EBITDA margin this month? | Allowed |
| What are the provisions clients for Q3? | Allowed |

---

### Manager

| Question | Result | Reason |
|---|---|---|
| What is the EBITDA for this month? | Allowed | — |
| How did revenue compare to budget? | Allowed | — |
| What is the ARPU for mobile prepaid? | Allowed | — |
| Show me CAPEX this quarter | Allowed | — |
| How many subscribers did we add? | Allowed | — |
| What is the total salary cost for personnel? | **Blocked** | Section: Personnel |
| Show me management fees paid to executives | **Blocked** | Section: Management fees |
| What are the provisions clients for Q3? | **Blocked** | Section: Provisions clients |
| What is our treasury position? | **Blocked** | Keyword: treasury position |
| What is the cash position this week? | **Blocked** | Keyword: cash position |
| What are the shareholder loan details? | **Blocked** | Keyword: shareholder loan |
| What was the severance payout last quarter? | **Blocked** | Keyword: severance |

---

### Viewer

| Question | Result | Reason |
|---|---|---|
| What is the total mobile revenue? | Allowed | — |
| How many mobile subscribers do we have? | Allowed | — |
| What is the EBITDA margin? | Allowed | — |
| Show me data usage trend this month | Allowed | — |
| What is the mobile money performance? | Allowed | — |
| Compare prepaid vs postpaid subscribers | Allowed | — |
| Show me the cash flow statement | **Blocked** | Sheet: cash_conso |
| What is our cash position? | **Blocked** | Keyword: cash position |
| Show me frais généraux breakdown | **Blocked** | Section: Frais généraux |
| What are the taxes and levies this quarter? | **Blocked** | Section: Impôts, taxes |
| Show me individual salary details | **Blocked** | Keyword: salary |
| Break down OPEX line items for me | **Blocked** | Keyword: line item |

---

## Login Credentials

### Production Users

| Name | Email | Password | Role |
|---|---|---|---|
| Aviral | aviral.dayal@ksolves.com | `Aviral@250302` | admin |
| Youssef Naifi | youssef.naifi@digiwise.com | `Youssef@123` | — |
| Fatoukine Dieng Sarr | fatoukine.diengsarr@digiwise.io | `Fatou@123` | — |

### Test Users *(RBAC validation)*

| Email | Password | Role |
|---|---|---|
| test.executive@digiwise.test | `Exec@test123` | executive |
| test.manager@digiwise.test | `Mgr@test123` | manager |
| test.viewer@digiwise.test | `View@test123` | viewer |

> Seed test users by running: `python3 create_test_users.py`
> Use a separate browser (or incognito window) per role to compare behaviour side by side.

---

## Changing a User's Role

**Via API** (requires admin key):
```bash
curl -X POST http://localhost:8000/api/v1/auth/admin/set-role \
  -H "X-Admin-Key: <admin_key>" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 2, "role": "manager"}'
```

**Via script:**
```bash
python3 set_user_roles.py
```

---

## Audit Log

Every blocked query is recorded in `public.copilot_policy_audit` with:
- User ID and role at time of block
- The question asked (truncated to 2000 chars)
- Which keyword or section triggered the block

This log is append-only and never shown to end users.
