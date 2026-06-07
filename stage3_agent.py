"""
Stage 3: making the coding agent robust.

Same loop, same tools as Stage 2 — but now production-shaped:

  1. STREAMING. We print tokens as they arrive instead of waiting for the whole
     response. Because streaming sidesteps the request timeout, we can also
     raise max_tokens for large file writes.

  2. FIRST-CLASS ERRORS. A failing tool returns a tool_result with
     is_error=True. The model sees the failure explicitly and recovers (e.g.
     read a missing file -> get an error -> create it instead). This replaces
     Stage 2's silent "Error: ..." seatbelt.

  3. PERMISSION GATING. Before a *mutating* tool (write_file, run_bash) runs, we
     pause and ask the user y/n — showing exactly what's about to happen. This
     is the payoff of Stage 2's typed tools: we can gate a write because the
     harness can SEE it's a write, on which path, with what content. Read-only
     tools (read_file, list_dir) run without prompting.

  4. ITERATION CAP. A guard so a confused agent can't loop forever.

Run:
    python stage3_agent.py            # prompts before writes / bash
Try:
    "read /tmp/notes.txt; if it doesn't exist, create it saying hello"
and watch: read_file -> (is_error) -> write_file -> [Allow? y/N].
"""

import os
import subprocess

import anthropic

MODEL = "claude-opus-4-8"
MAX_ITERATIONS = 25                       # safety guard on the loop
REQUIRES_APPROVAL = {"write_file", "run_bash"}   # mutating tools -> gated

client = anthropic.Anthropic()

SYSTEM = (
    "You are a coding agent working in the user's current directory. You can "
    "read, write, and list files, and run bash commands. Prefer the dedicated "
    "file tools (read_file / write_file / list_dir) over bash for file work. "
    "After writing code, run it to verify it works. Be concise."
)


# --- 1. Tools (unchanged from Stage 2) -------------------------------------

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


TOOLS = [
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

DISPATCH = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "run_bash": run_bash,
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


# --- 3. The loop (now streaming + gated + capped) --------------------------

def run_turn(messages: list) -> None:
    for _ in range(MAX_ITERATIONS):
        # Stream the model's response, printing text deltas live.
        with client.messages.stream(
            model=MODEL,
            max_tokens=16000,   # safe to raise now that we stream
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

        # Same loop rules as before: append the full response, stop if no tools.
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            # Gate mutating tools behind user confirmation.
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
                result["is_error"] = True   # model sees the failure explicitly
            tool_results.append(result)

        messages.append({"role": "user", "content": tool_results})
    else:
        # Loop ran MAX_ITERATIONS times without finishing.
        print(f"\n⚠️  Hit the {MAX_ITERATIONS}-iteration cap; stopping this turn.")


# --- 4. REPL ---------------------------------------------------------------

def main() -> None:
    print("Stage 3 coding agent (streaming + gated). Type a request, or 'quit'.")
    messages = []
    while True:
        try:
            user_input = input("\n💬 ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        run_turn(messages)


if __name__ == "__main__":
    main()
