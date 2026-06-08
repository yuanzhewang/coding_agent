# Agents from scratch

Learning to build LLM agents at a low level — raw [Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python),
own loop, hand-written tool schemas. No high-level framework, so the underlying
mechanics stay visible.

The core idea: **an agent is a loop where the model's output chooses the next
action, and the result of that action feeds back into the model.** Each stage
adds one layer to that loop.

## Stages

| Stage | File | Adds |
|-------|------|------|
| 1 | `stage1_agent.py` | The bare loop + one `run_bash` tool |
| 2 | `stage2_agent.py` | Dedicated `read_file` / `write_file` / `list_dir` tools (a real coding agent) |
| 3 | `stage3_agent.py` | Streaming, error handling, permission gating, iteration cap, + Sherlog-style debug log (`/debug`, via `rpc_logger.py`) |
| 4a | `stage4a_agent.py` | Subagents — the loop re-run with its own context, returning a summary |
| 4b | `stage4b_agent.py` | Skills — progressive disclosure: catalog in the prompt, bodies loaded on demand (`skills/`) |
| 4c | `stage4c_agent.py` | Memory — a file read at each turn + a `remember` tool; persists across restarts |
| 4d | `stage4d_agent.py` | Compaction — summarize old turns when history grows, with a pair-safe cut |
| 4e | `stage4e_agent.py` | Heartbeat — scheduler + standing goal + memory bridge + `finish`; safe toolset |
| 5 | `stage5_agent.py` | Reaches outside the machine — Gmail / Calendar / Drive over OAuth 2.0 (`workspace_tools.py`) |

## Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

### Stage 5 — Google (Gmail) setup

Stage 5 talks to Gmail over OAuth 2.0. The agent loop is unchanged; the new
concept is auth, which lives in `workspace_tools.py`.

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

In the [Google Cloud Console](https://console.cloud.google.com): create a
project → enable the **Gmail API**, **Google Calendar API**, and **Google Drive
API** → configure the **OAuth consent screen** (External, add yourself as a
*test user*) → create an **OAuth client ID** of type **Desktop app** → download
the JSON as `credentials.json` in this directory.

Then mint a token (opens a browser once; needs a machine with a browser):

```bash
python stage5_auth_check.py   # consent in the browser → writes token.json
```

`credentials.json` (app identity) and `token.json` (key to your account) are
both gitignored. Changing `SCOPES` invalidates `token.json` — delete it and
re-run the auth check. In "Testing" mode the refresh token expires ~weekly.

## Run

```bash
python stage1_agent.py   # or: python stage2_agent.py
```

Then try: `how many Python files are under the current directory?` and watch it
call a tool, read the output, and answer.

> ⚠️ Stages 1–2 run bash/writes with no confirmation. Stage 3 adds permission
> gating (y/N before writes and bash). Until then, run somewhere safe.
