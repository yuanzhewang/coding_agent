"""
rpc_logger.py — a tiny "Sherlog" for LLM calls.

Each call to the Anthropic API is, in effect, an RPC: a request (model, system,
tools, messages, params) goes out, a response (content blocks, stop_reason,
usage) comes back. This module records every such pair and renders them to a
single self-contained HTML file with:

  - a human-friendly rendered view (chat bubbles, tool calls, token badges), and
  - the raw JSON for every call (in a collapsible <details>).

Open the HTML in a browser and refresh as the agent runs — it's rewritten after
every call. Inspired by Google's Sherlog RPC inspector.

Usage (from an agent script):
    import rpc_logger
    rpc_logger.record(request={...}, response=message)   # after each LLM call
    rpc_logger.set_enabled(False)                         # toggle off
    rpc_logger.path()                                     # where it's written

Debug mode is ON by default.
"""

import html
import json
from datetime import datetime

_records = []
_enabled = True
_path = "rpc_debug.html"


# --- public API ------------------------------------------------------------

def set_enabled(on: bool) -> None:
    global _enabled
    _enabled = bool(on)


def is_enabled() -> bool:
    return _enabled


def set_path(path: str) -> None:
    global _path
    _path = path


def path() -> str:
    return _path


def reset() -> None:
    _records.clear()


def record(request: dict, response) -> None:
    """Record one request/response pair and rewrite the HTML. No-op if disabled."""
    if not _enabled:
        return
    _records.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "request": _to_jsonable(request),
        "response": _to_jsonable(response),
    })
    flush()


def flush() -> None:
    """Write the current records to the HTML file."""
    with open(_path, "w", encoding="utf-8") as f:
        f.write(_page(_records))


# --- serialization ---------------------------------------------------------

def _to_jsonable(obj):
    """Convert SDK (pydantic) objects + nested containers to plain JSON types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "model_dump"):          # anthropic SDK blocks/messages
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return _to_jsonable(vars(obj))
    return str(obj)


# --- rendering -------------------------------------------------------------

def _render_content(content) -> str:
    if isinstance(content, str):
        return f'<div class="bubble">{html.escape(content)}</div>'
    if not isinstance(content, list):
        return f'<pre>{html.escape(json.dumps(content, indent=2, default=str))}</pre>'
    parts = []
    for b in content:
        if not isinstance(b, dict):
            parts.append(f'<pre>{html.escape(str(b))}</pre>')
            continue
        t = b.get("type")
        if t == "text":
            parts.append(f'<div class="bubble">{html.escape(b.get("text", ""))}</div>')
        elif t == "tool_use":
            inp = json.dumps(b.get("input", {}), indent=2, default=str)
            parts.append(
                f'<div class="tool tooluse">🔧 <b>{html.escape(b.get("name", ""))}</b>'
                f'<pre>{html.escape(inp)}</pre></div>')
        elif t == "tool_result":
            c = b.get("content", "")
            ctext = c if isinstance(c, str) else json.dumps(c, indent=2, default=str)
            err = b.get("is_error")
            cls = "tool toolresult err" if err else "tool toolresult"
            label = "↳ tool_result (error)" if err else "↳ tool_result"
            parts.append(f'<div class="{cls}">{label}<pre>{html.escape(ctext)}</pre></div>')
        elif t == "thinking":
            parts.append(f'<div class="thinking">💭 {html.escape(b.get("thinking", ""))}</div>')
        else:
            parts.append(f'<pre>{html.escape(json.dumps(b, indent=2, default=str))}</pre>')
    return "\n".join(parts)


def _render_messages(messages) -> str:
    out = []
    for m in messages:
        role = m.get("role", "?")
        out.append(f'<div class="msg {html.escape(role)}">'
                   f'<div class="role">{html.escape(role)}</div>'
                   f'{_render_content(m.get("content", ""))}</div>')
    return "\n".join(out)


def _render_request(req: dict) -> str:
    parts = ['<div class="section"><h4>Request</h4>']
    parts.append(f'<div class="params">model=<code>{html.escape(str(req.get("model")))}</code>'
                 f' · max_tokens={req.get("max_tokens")}</div>')
    sysv = req.get("system")
    if sysv:
        sys_text = sysv if isinstance(sysv, str) else json.dumps(sysv, indent=2, default=str)
        parts.append(f'<details><summary>system ({len(sys_text)} chars)</summary>'
                     f'<pre>{html.escape(sys_text)}</pre></details>')
    tools = req.get("tools") or []
    if tools:
        items = "".join(
            f'<li><b>{html.escape(t.get("name", ""))}</b> — '
            f'{html.escape(t.get("description", ""))}</li>'
            for t in tools if isinstance(t, dict))
        parts.append(f'<details><summary>tools ({len(tools)})</summary><ul>{items}</ul></details>')
    parts.append('<div class="messages">' + _render_messages(req.get("messages", [])) + '</div>')
    parts.append('</div>')
    return "".join(parts)


def _render_response(resp: dict) -> str:
    parts = ['<div class="section"><h4>Response</h4>']
    parts.append('<div class="messages">' + _render_content(resp.get("content", [])) + '</div>')
    usage = resp.get("usage") or {}
    badges = "".join(f'<span class="badge">{html.escape(k)}={v}</span>'
                     for k, v in usage.items() if v)
    parts.append(f'<div class="badges">stop_reason='
                 f'{html.escape(str(resp.get("stop_reason")))} {badges}</div>')
    parts.append('</div>')
    return "".join(parts)


def _raw(req: dict, resp: dict) -> str:
    return ('<details class="raw"><summary>raw JSON</summary>'
            '<h5>request</h5><pre>' + html.escape(json.dumps(req, indent=2, default=str)) + '</pre>'
            '<h5>response</h5><pre>' + html.escape(json.dumps(resp, indent=2, default=str)) + '</pre>'
            '</details>')


def _page(records: list) -> str:
    rows = []
    total_in = total_out = 0
    for i, rec in enumerate(records, 1):
        req, resp, t = rec["request"], rec["response"], rec["time"]
        usage = resp.get("usage") or {}
        tin = usage.get("input_tokens", 0) or 0
        tout = usage.get("output_tokens", 0) or 0
        cr = usage.get("cache_read_input_tokens", 0) or 0
        total_in += tin
        total_out += tout
        header = (f'#{i} · {t} · {html.escape(str(req.get("model")))} · '
                  f'stop={html.escape(str(resp.get("stop_reason")))} · ↑{tin} ↓{tout}'
                  + (f' · cache_read {cr}' if cr else ''))
        body = _render_request(req) + _render_response(resp) + _raw(req, resp)
        rows.append(f'<details class="rpc" open><summary>{header}</summary>{body}</details>')
    summary = f'{len(records)} call(s) · {total_in} input / {total_out} output tokens'
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>LLM RPC log</title><style>" + _CSS + "</style></head><body>"
        "<h1>LLM RPC log</h1>"
        f"<div class='summary'>{summary}</div>"
        f"<div class='gen'>generated {generated}</div>"
        + ("".join(rows) if rows else "<p class='gen'>No calls recorded yet.</p>")
        + "</body></html>"
    )


# Kept as a separate constant so its literal { } braces don't collide with the
# f-string in _page().
_CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:24px;background:#0f1115;color:#d7dae0}
h1{font-size:18px;margin:0 0 4px}
.summary{color:#9aa0aa}
.gen{color:#5b616b;font-size:12px;margin-bottom:16px}
details.rpc{border:1px solid #262a33;border-radius:8px;margin:10px 0;background:#151821}
details.rpc>summary{cursor:pointer;padding:10px 14px;font-family:ui-monospace,Menlo,monospace;font-size:13px;color:#cdd2da}
details.rpc[open]>summary{border-bottom:1px solid #262a33}
.section{padding:10px 14px}
.section h4{margin:6px 0;color:#8ab4f8;font-size:12px;text-transform:uppercase;letter-spacing:.5px}
.params{color:#9aa0aa;font-size:12px;margin-bottom:8px}
.params code{color:#e5c07b}
.msg{border-left:3px solid #333;padding:6px 10px;margin:6px 0;border-radius:4px;background:#1b1f2a}
.msg.user{border-color:#3b82f6}
.msg.assistant{border-color:#22c55e}
.msg.system{border-color:#a855f7}
.role{font-size:11px;text-transform:uppercase;color:#7d828c;margin-bottom:4px;letter-spacing:.5px}
.bubble{white-space:pre-wrap;word-break:break-word}
.tool{background:#10141c;border:1px solid #262a33;border-radius:6px;padding:6px 8px;margin:4px 0}
.tooluse{border-color:#2f6f4f}
.toolresult{border-color:#46506a}
.toolresult.err{border-color:#e06c75}
.thinking{color:#9aa0aa;font-style:italic;border-left:2px solid #555;padding-left:8px;margin:4px 0}
pre{white-space:pre-wrap;word-break:break-word;background:#0b0e13;border:1px solid #1f2430;border-radius:6px;padding:8px;overflow:auto;font:12px/1.45 ui-monospace,Menlo,monospace;color:#c8cdd6}
details details{margin:6px 0}
summary{outline:none}
.badges{margin-top:8px;color:#9aa0aa;font-size:12px}
.badge{display:inline-block;background:#1b1f2a;border:1px solid #2a2f3a;border-radius:10px;padding:1px 8px;margin-right:4px;font-size:11px}
ul{margin:6px 0;padding-left:18px}
code{font-family:ui-monospace,Menlo,monospace}
h5{margin:8px 0 2px;color:#7d828c;font-size:12px}
"""
