"""
workspace_tools.py — Google Workspace access for the agent, from scratch.

The agent loop doesn't change: Workspace actions are just more typed tools. What
is new lives *underneath* the tools — OAuth 2.0. A file tool works because the OS
already trusts this process; Google trusts no one by default, so before any tool
can read your mail we run a one-time consent flow that mints a token, then cache
it for next time.

Auth model — installed-app ("Desktop app") OAuth client:

  credentials.json : the APP's identity, downloaded once from the Google Cloud
                     Console. It says "this program is allowed to ask for
                     consent." It is not, by itself, a key to your data.
  token.json       : minted after YOU consent in the browser. THIS is the
                     sensitive file — it is a key to your account, limited to
                     SCOPES. Gitignored. Delete it to force re-consent.
  SCOPES           : the capability boundary (least privilege). This is the
                     Google-side analogue of Stage 3's permission gating.
                     Changing this list invalidates token.json — delete it and
                     re-consent so the new scopes take effect.

Run stage5_auth_check.py once to perform the consent flow and verify it works.
"""

import base64
import os.path
from datetime import datetime, timezone
from email.message import EmailMessage

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Capability boundary. Each entry becomes a permission you must approve on
# Google's consent screen. Least privilege, but batched: changing this list
# invalidates token.json, so we request everything the tools below need at once.
#   gmail.readonly  : search + read existing mail
#   gmail.compose   : create drafts AND send (compose implies send)
#   calendar.events : read AND create/edit events
#   drive.readonly  : search + read/export files (no writes)
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/drive.readonly",
]

CREDENTIALS_FILE = "credentials.json"   # the app's identity, from Cloud Console
TOKEN_FILE = "token.json"               # your minted token, cached after consent


def get_credentials() -> Credentials:
    """Load cached creds, refreshing or running the consent flow as needed.

    First run : opens a browser, you consent, token.json is written.
    Later runs: loads token.json; silently refreshes it when it expires.
    """
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())            # silent: use the refresh token
        else:
            if not os.path.exists(CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"{CREDENTIALS_FILE} not found. In the Google Cloud Console, "
                    f"create an OAuth 'Desktop app' client, download the JSON, and "
                    f"save it here as {CREDENTIALS_FILE}."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)   # opens browser for consent
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            f.write(creds.to_json())            # cache for next time
    return creds


# Cache built service clients so we don't rebuild one per tool call.
_services = {}


def get_service(api: str, version: str):
    """Return an authenticated Google API client, e.g. get_service('gmail','v1')."""
    key = (api, version)
    if key not in _services:
        _services[key] = build(api, version, credentials=get_credentials())
    return _services[key]


# --- Gmail tools -----------------------------------------------------------
#
# Each of these is an ordinary function the agent loop will call exactly like a
# local file tool. The only difference is they talk to Google over the network,
# authenticated by the token minted above. Read tools (search/read) run freely;
# gmail_send is gated by the harness (see stage5_agent.py).

def _extract_text(payload: dict) -> str:
    """Walk a Gmail message payload and return the first text/plain body."""
    if payload.get("mimeType") == "text/plain":
        data = payload.get("body", {}).get("data")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", "replace")
    for part in payload.get("parts", []) or []:
        text = _extract_text(part)
        if text:
            return text
    return ""


def _build_raw(to: str, subject: str, body: str) -> str:
    """Build a MIME message and base64url-encode it the way the Gmail API wants."""
    msg = EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    return base64.urlsafe_b64encode(msg.as_bytes()).decode()


def gmail_search(query: str, max_results: int = 10) -> str:
    """Search mail with Gmail's query syntax (e.g. 'from:foo is:unread newer_than:7d')."""
    svc = get_service("gmail", "v1")
    resp = svc.users().messages().list(
        userId="me", q=query, maxResults=max_results).execute()
    ids = resp.get("messages", [])
    if not ids:
        return "No messages matched."
    lines = []
    for m in ids:
        full = svc.users().messages().get(
            userId="me", id=m["id"], format="metadata",
            metadataHeaders=["From", "Subject", "Date"]).execute()
        h = {x["name"]: x["value"] for x in full["payload"].get("headers", [])}
        lines.append(
            f'[{m["id"]}] {h.get("Date", "?")} | {h.get("From", "?")}\n'
            f'    {h.get("Subject", "(no subject)")}\n'
            f'    {full.get("snippet", "")}')
    return "\n".join(lines)


def gmail_read(message_id: str) -> str:
    """Read one full message (headers + plain-text body) by its id."""
    svc = get_service("gmail", "v1")
    full = svc.users().messages().get(
        userId="me", id=message_id, format="full").execute()
    h = {x["name"]: x["value"] for x in full["payload"].get("headers", [])}
    body = _extract_text(full["payload"])
    meta = "\n".join(f"{k}: {h.get(k, '')}" for k in ("From", "To", "Date", "Subject"))
    return f"{meta}\n\n{body or '(no plain-text body)'}"


def gmail_create_draft(to: str, subject: str, body: str) -> str:
    """Create a draft (NOT sent). Safe: you finish/send it yourself in Gmail."""
    svc = get_service("gmail", "v1")
    draft = svc.users().drafts().create(
        userId="me", body={"message": {"raw": _build_raw(to, subject, body)}}).execute()
    return f"Draft created (id {draft['id']}). It is NOT sent — review it in Gmail."


def gmail_send(to: str, subject: str, body: str) -> str:
    """Actually send a message. Gated behind user confirmation by the harness."""
    svc = get_service("gmail", "v1")
    sent = svc.users().messages().send(
        userId="me", body={"raw": _build_raw(to, subject, body)}).execute()
    return f"Sent (message id {sent['id']}) to {to}."


# --- exports the agent composes into its toolset ---------------------------

GMAIL_DISPATCH = {
    "gmail_search": gmail_search,
    "gmail_read": gmail_read,
    "gmail_create_draft": gmail_create_draft,
    "gmail_send": gmail_send,
}

# Mutating tool(s) the harness must gate behind confirmation.
GMAIL_GATED = {"gmail_send"}

GMAIL_TOOLS = [
    {
        "name": "gmail_search",
        "description": (
            "Search the user's Gmail using Gmail query syntax (e.g. "
            "'from:alice@x.com is:unread newer_than:7d'). Returns a list of "
            "matching messages with id, date, sender, subject, and snippet."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Gmail search query."},
                "max_results": {"type": "integer",
                                "description": "Max messages to return. Default 10."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "gmail_read",
        "description": "Read one full email (headers + plain-text body) by its message id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message_id": {"type": "string",
                               "description": "The Gmail message id (from gmail_search)."},
            },
            "required": ["message_id"],
        },
    },
    {
        "name": "gmail_create_draft",
        "description": (
            "Create a Gmail draft. The draft is NOT sent — the user reviews and "
            "sends it themselves. Prefer this over gmail_send unless told to send."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string", "description": "Subject line."},
                "body": {"type": "string", "description": "Plain-text body."},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "gmail_send",
        "description": (
            "Actually send an email. This is irreversible — only use it when the "
            "user explicitly asks to send. The user must confirm before it runs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string", "description": "Subject line."},
                "body": {"type": "string", "description": "Plain-text body."},
            },
            "required": ["to", "subject", "body"],
        },
    },
]


# --- Calendar tools --------------------------------------------------------

def calendar_list(time_min: str = "", time_max: str = "", max_results: int = 10) -> str:
    """List upcoming events on the primary calendar. Times are RFC3339 strings."""
    svc = get_service("calendar", "v3")
    params = {
        "calendarId": "primary",
        "maxResults": max_results,
        "singleEvents": True,        # expand recurring events into instances
        "orderBy": "startTime",
        "timeMin": time_min or datetime.now(timezone.utc).isoformat(),
    }
    if time_max:
        params["timeMax"] = time_max
    items = svc.events().list(**params).execute().get("items", [])
    if not items:
        return "No events in that range."
    lines = []
    for e in items:
        start = e["start"].get("dateTime", e["start"].get("date", "?"))
        lines.append(f'[{e["id"]}] {start} | {e.get("summary", "(no title)")}')
    return "\n".join(lines)


def calendar_create_event(summary: str, start: str, end: str,
                          description: str = "", location: str = "",
                          time_zone: str = "UTC") -> str:
    """Create a timed event on the primary calendar. Gated by the harness."""
    svc = get_service("calendar", "v3")
    body = {
        "summary": summary,
        "description": description,
        "location": location,
        "start": {"dateTime": start, "timeZone": time_zone},
        "end": {"dateTime": end, "timeZone": time_zone},
    }
    ev = svc.events().insert(calendarId="primary", body=body).execute()
    return f"Event created: {ev.get('htmlLink', ev['id'])}"


# --- Drive tools (read-only) -----------------------------------------------

# Google-native formats can't be downloaded directly; they must be exported to a
# concrete type. Everything else (txt, pdf, csv…) downloads as-is.
_DRIVE_EXPORT = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}


def drive_search(query: str, max_results: int = 10) -> str:
    """Search Drive with Drive query syntax, e.g. "name contains 'budget'" or
    "mimeType='application/pdf'" or "fullText contains 'invoice'"."""
    svc = get_service("drive", "v3")
    resp = svc.files().list(
        q=query, pageSize=max_results,
        fields="files(id,name,mimeType,modifiedTime)").execute()
    files = resp.get("files", [])
    if not files:
        return "No files matched."
    return "\n".join(
        f'[{f["id"]}] {f["name"]} ({f["mimeType"]}) {f.get("modifiedTime", "")}'
        for f in files)


def drive_read(file_id: str) -> str:
    """Read a Drive file's text content (exports Google Docs/Sheets to text)."""
    svc = get_service("drive", "v3")
    meta = svc.files().get(fileId=file_id, fields="id,name,mimeType").execute()
    mime = meta["mimeType"]
    if mime in _DRIVE_EXPORT:
        data = svc.files().export(fileId=file_id, mimeType=_DRIVE_EXPORT[mime]).execute()
    else:
        data = svc.files().get_media(fileId=file_id).execute()
    text = data.decode("utf-8", "replace") if isinstance(data, bytes) else str(data)
    if len(text) > 5000:
        text = text[:5000] + "\n… (truncated)"
    return f"{meta['name']} ({mime})\n\n{text or '(no text content)'}"


# --- Calendar / Drive exports ----------------------------------------------

CALENDAR_DISPATCH = {
    "calendar_list": calendar_list,
    "calendar_create_event": calendar_create_event,
}
CALENDAR_GATED = {"calendar_create_event"}

CALENDAR_TOOLS = [
    {
        "name": "calendar_list",
        "description": (
            "List upcoming events on the user's primary calendar. Optionally bound "
            "by time_min/time_max (RFC3339, e.g. '2026-06-08T00:00:00Z'). Defaults "
            "to events from now onward."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "time_min": {"type": "string", "description": "RFC3339 lower bound. Default now."},
                "time_max": {"type": "string", "description": "RFC3339 upper bound."},
                "max_results": {"type": "integer", "description": "Max events. Default 10."},
            },
            "required": [],
        },
    },
    {
        "name": "calendar_create_event",
        "description": (
            "Create a timed event on the primary calendar. Times are RFC3339 "
            "datetimes. This writes to the user's real calendar — the user must "
            "confirm before it runs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Event title."},
                "start": {"type": "string", "description": "Start, RFC3339 (e.g. '2026-06-10T15:00:00')."},
                "end": {"type": "string", "description": "End, RFC3339."},
                "description": {"type": "string", "description": "Optional details."},
                "location": {"type": "string", "description": "Optional location."},
                "time_zone": {"type": "string", "description": "IANA tz, e.g. 'America/New_York'. Default UTC."},
            },
            "required": ["summary", "start", "end"],
        },
    },
]

DRIVE_DISPATCH = {
    "drive_search": drive_search,
    "drive_read": drive_read,
}
DRIVE_GATED = set()   # read-only

DRIVE_TOOLS = [
    {
        "name": "drive_search",
        "description": (
            "Search the user's Google Drive using Drive query syntax. Examples: "
            "\"name contains 'budget'\", \"mimeType='application/pdf'\", "
            "\"fullText contains 'invoice'\". Returns id, name, type, modified time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Drive query string."},
                "max_results": {"type": "integer", "description": "Max files. Default 10."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "drive_read",
        "description": (
            "Read a Drive file's text content by id (Google Docs/Sheets are "
            "exported to text/CSV). Large files are truncated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "Drive file id (from drive_search)."},
            },
            "required": ["file_id"],
        },
    },
]
