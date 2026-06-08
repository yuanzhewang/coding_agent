"""
stage5_auth_check.py — run the OAuth consent flow once and prove it works.

This is the "hello world" of the Google integration: no agent, no LLM, just the
auth path. The first run opens a browser — pick your account, approve the
read-only scope — and a token.json appears next to this script. It then reads
your Gmail *profile* (your address + message counts, no mail content) to confirm
the token is live.

Run this BEFORE stage5_agent.py:
    python stage5_auth_check.py
"""

import workspace_tools


def main() -> None:
    print("Authenticating with Google (a browser window will open the first time)…")
    service = workspace_tools.get_service("gmail", "v1")
    profile = service.users().getProfile(userId="me").execute()
    print("\n✅ Auth works.")
    print(f"   account:  {profile['emailAddress']}")
    print(f"   messages: {profile['messagesTotal']}")
    print(f"   token cached at: {workspace_tools.TOKEN_FILE}")
    print("\nNext: we'll add the Gmail/Calendar tools and wire up stage5_agent.py.")


if __name__ == "__main__":
    main()
