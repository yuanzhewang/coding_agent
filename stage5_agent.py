"""
Stage 5: the agent reaches outside the machine — Google Workspace.

Every tool until now was local: files, bash. The agent loop never cared, because
a tool is just "a name, a JSON schema, and a function." Stage 5 proves that by
adding tools that talk to Gmail, Calendar, and Drive over the network — same
loop, same gating, same debug log. The genuinely new part lives in
workspace_tools.py: OAuth 2.0, which is how Google decides to trust this process
(see that file's docstring).

What you get:
  - gmail_search / gmail_read         — read-only, run freely
  - gmail_create_draft                — writes an UN-sent draft (safe)
  - gmail_send                        — actually sends; GATED behind y/N
  - calendar_list                     — read-only
  - calendar_create_event             — writes a real event; GATED
  - drive_search / drive_read         — read-only
  - the Stage 3 local toolset         — read/write/list files, run bash
  - the Sherlog-style debug log       — every LLM call, incl. these tool calls

Setup (once): see README "Stage 5". You must have run the OAuth flow so that
token.json exists (python stage5_auth_check.py).

Run:
    python stage5_agent.py
Try:
    "what are my 3 most recent emails about?"
    "what's on my calendar this week?"
    "find the doc named 'notes' in my Drive and summarize it"
"""

import os
import subprocess

import anthropic

import rpc_logger        # 'Sherlog'-style inspector for LLM requests/responses
import workspace_tools   # Google Workspace (Gmail) tools + OAuth, from scratch

MODEL = "claude-opus-4-8"
MAX_TOKENS = 16000
MAX_ITERATIONS = 25
# Mutating tools -> gated. Local writes/bash, sending mail, creating events.
REQUIRES_APPROVAL = ({"write_file", "run_bash"}
                     | workspace_tools.GMAIL_GATED
                     | workspace_tools.CALENDAR_GATED
                     | workspace_tools.DRIVE_GATED)

client = anthropic.Anthropic()

SYSTEM = (
    "You are a personal assistant with access to the user's Gmail, Google "
    "Calendar, Google Drive, and their local filesystem. You can search/read "
    "mail, create drafts, and (only when explicitly asked) send mail; list and "
    "create calendar events; search and read Drive files; and read, write, and "
    "list local files and run bash. Prefer gmail_create_draft over gmail_send "
    "unless the user clearly says to send. Cite ids when you reference emails, "
    "events, or files. Today's date is available via bash if you need it. "
    "Be concise."
)


# --- 1. Local tools (from Stage 3) -----------------------------------------

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read() or "(empty file)"


def write_file(path: str, content: str) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {path}"


def list_dir(path: str = ".") -> str:
    entries = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        suffix = "/" if os.path.isdir(full) else f"  ({os.path.getsize(full)} bytes)"
        entries.append(name + suffix)
    return "\n".join(entries) or "(empty directory)"


def run_bash(command: str) -> str:
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=60
    )
    return (result.stdout + result.stderr) or "(no output)"


LOCAL_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file and return its contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the file."}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create or overwrite a UTF-8 text file with the given content. "
            "Creates parent directories as needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write."},
                "content": {"type": "string", "description": "Full file contents."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "List the entries in a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path. Defaults to '.'."}
            },
            "required": [],
        },
    },
    {
        "name": "run_bash",
        "description": (
            "Run a bash command and return its combined stdout/stderr. Use for "
            "running tests, programs, git, package installs, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command."}},
            "required": ["command"],
        },
    },
]

# Compose the full toolset: local + Gmail + Calendar + Drive. The loop treats
# them all identically — a tool is just a name, a schema, and a function.
TOOLS = (LOCAL_TOOLS
         + workspace_tools.GMAIL_TOOLS
         + workspace_tools.CALENDAR_TOOLS
         + workspace_tools.DRIVE_TOOLS)
DISPATCH = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "run_bash": run_bash,
    **workspace_tools.GMAIL_DISPATCH,
    **workspace_tools.CALENDAR_DISPATCH,
    **workspace_tools.DRIVE_DISPATCH,
}


# --- 2. Helpers ------------------------------------------------------------

def format_tool_call(name: str, tool_input: dict) -> str:
    if name == "write_file":
        return f"write_file: {tool_input['path']} ({len(tool_input.get('content', ''))} bytes)"
    if name == "read_file":
        return f"read_file: {tool_input['path']}"
    if name == "list_dir":
        return f"list_dir: {tool_input.get('path', '.')}"
    if name == "run_bash":
        return f"run_bash: {tool_input['command']}"
    if name == "gmail_search":
        return f"gmail_search: {tool_input['query']}"
    if name == "gmail_read":
        return f"gmail_read: {tool_input['message_id']}"
    if name in ("gmail_create_draft", "gmail_send"):
        verb = "draft" if name == "gmail_create_draft" else "SEND"
        return f"gmail {verb} → {tool_input.get('to', '?')}: {tool_input.get('subject', '')}"
    if name == "calendar_list":
        return f"calendar_list: {tool_input.get('time_min', 'now')}..{tool_input.get('time_max', '')}"
    if name == "calendar_create_event":
        return f"calendar create → {tool_input.get('summary', '')} @ {tool_input.get('start', '')}"
    if name == "drive_search":
        return f"drive_search: {tool_input['query']}"
    if name == "drive_read":
        return f"drive_read: {tool_input['file_id']}"
    return f"{name}: {tool_input}"


def confirm(name: str, tool_input: dict) -> bool:
    """Ask the user to approve a mutating tool call. Deny on EOF (safe default)."""
    print(f"\n⚠️  Agent wants to {format_tool_call(name, tool_input)}")
    if name == "write_file":
        content = tool_input.get("content", "")
        print("   ── content preview ──")
        for line in content[:300].splitlines():
            print(f"   | {line}")
        if len(content) > 300:
            print("   | …")
    if name == "gmail_send":
        print(f"   to:      {tool_input.get('to', '')}")
        print(f"   subject: {tool_input.get('subject', '')}")
        print("   ── body ──")
        for line in tool_input.get("body", "")[:500].splitlines():
            print(f"   | {line}")
    if name == "calendar_create_event":
        print(f"   title: {tool_input.get('summary', '')}")
        print(f"   start: {tool_input.get('start', '')}  end: {tool_input.get('end', '')}")
        if tool_input.get("location"):
            print(f"   where: {tool_input['location']}")
    try:
        answer = input("   Allow? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def execute_tool(name: str, tool_input: dict):
    """Run a tool. Returns (content, is_error)."""
    try:
        return DISPATCH[name](**tool_input), False
    except Exception as e:
        return f"Error: {e}", True


# --- 3. The loop (identical in shape to Stage 3) ---------------------------

def run_turn(messages: list) -> None:
    for _ in range(MAX_ITERATIONS):
        with client.messages.stream(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            printed = False
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    if not printed:
                        print("\n🤖 ", end="", flush=True)
                        printed = True
                    print(event.delta.text, end="", flush=True)
            if printed:
                print()
            response = stream.get_final_message()

        rpc_logger.record(
            request={"model": MODEL, "max_tokens": MAX_TOKENS, "system": SYSTEM,
                     "tools": TOOLS, "messages": messages},
            response=response,
        )

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name in REQUIRES_APPROVAL and not confirm(block.name, block.input):
                print("   ✗ denied")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "The user declined to run this tool. Consider "
                               "another approach or ask them what they'd prefer.",
                })
                continue

            print(f"\n🔧 {format_tool_call(block.name, block.input)}")
            content, is_error = execute_tool(block.name, block.input)
            preview = content.strip()
            print(f"   ↳ {preview[:500]}{' …' if len(preview) > 500 else ''}")

            result = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            }
            if is_error:
                result["is_error"] = True
            tool_results.append(result)

        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"\n⚠️  Hit the {MAX_ITERATIONS}-iteration cap; stopping this turn.")


# --- 4. REPL ---------------------------------------------------------------

def main() -> None:
    print("Stage 5 agent — Gmail + local files (streaming + gated).")
    print(f"Debug mode is {'on' if rpc_logger.is_enabled() else 'off'} → "
          f"logging RPCs to {rpc_logger.path()}")
    print("Commands: /debug on | /debug off | quit")
    if rpc_logger.is_enabled():
        rpc_logger.flush()
    messages = []
    while True:
        try:
            user_input = input("\n💬 ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in {"quit", "exit"}:
            break
        if user_input.lower().startswith("/debug"):
            arg = user_input[len("/debug"):].strip().lower()
            if arg in {"", "on"}:
                rpc_logger.set_enabled(True)
                rpc_logger.flush()
                print(f"debug ON → {rpc_logger.path()}")
            elif arg == "off":
                rpc_logger.set_enabled(False)
                print("debug OFF")
            else:
                print("usage: /debug [on|off]")
            continue
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        run_turn(messages)


if __name__ == "__main__":
    main()
