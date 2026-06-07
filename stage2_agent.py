"""
Stage 2: a real coding agent — dedicated file tools alongside bash.

Stage 1 had one opaque tool (run_bash). You could already `cat` a file or
`echo > file` through it, so why add read_file / write_file / list_dir?

Because of what the HARNESS gets to see. With run_bash, your code receives one
string: "echo ... > foo.py". It can't tell a read from a write, can't see which
path is being touched, can't validate the content. With a typed tool like

    write_file(path="foo.py", content="...")

your code receives structured arguments it can inspect: which file, what bytes.
That's the hook Stage 3 builds on — permission prompts before a write, staleness
checks ("did this file change since the model last read it?"), diff rendering,
path allow-lists. None of that is possible when everything is an opaque shell
string.

So the lesson of this stage isn't "more tools" — it's: promote an action to a
typed tool when you want the harness to reason about it. The loop itself is
unchanged from Stage 1; only the tool surface grew.

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python stage2_agent.py

Try:  "write a fizzbuzz to fizz.py, run it, then make it go to 30 instead"
and watch it chain write_file -> run_bash -> read_file -> write_file.

Still missing (Stage 3): streaming, first-class error signalling (is_error),
permission gating, an iteration cap. The try/except below is just a seatbelt so
a bad path doesn't kill the REPL.
"""

import os
import subprocess

import anthropic

MODEL = "claude-opus-4-8"
client = anthropic.Anthropic()

SYSTEM = (
    "You are a coding agent working in the user's current directory. You can "
    "read, write, and list files, and run bash commands. Prefer the dedicated "
    "file tools (read_file / write_file / list_dir) over bash for file work. "
    "After writing code, run it to verify it works. Be concise."
)


# --- 1. The tools ----------------------------------------------------------
#
# Four small functions. Note the typed signatures — `path`, `content`,
# `command` — these become the structured arguments the harness can inspect.

def read_file(path: str) -> str:
    """Return the contents of a UTF-8 text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read() or "(empty file)"


def write_file(path: str, content: str) -> str:
    """Create or overwrite a file, making parent directories as needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {path}"


def list_dir(path: str = ".") -> str:
    """List directory entries; directories get a trailing slash."""
    entries = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isdir(full):
            entries.append(f"{name}/")
        else:
            entries.append(f"{name}  ({os.path.getsize(full)} bytes)")
    return "\n".join(entries) or "(empty directory)"


def run_bash(command: str) -> str:
    """Run a shell command, returning combined stdout + stderr."""
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=60
    )
    return (result.stdout + result.stderr) or "(no output)"


# --- 2. Tool schemas: how the model learns what it can call ----------------

TOOLS = [
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file and return its contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."}
            },
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
                "path": {
                    "type": "string",
                    "description": "Directory path. Defaults to the current directory.",
                }
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
            "properties": {
                "command": {"type": "string", "description": "The bash command."}
            },
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


# --- 3. The loop (identical shape to Stage 1; just more tools) -------------

def format_tool_call(name: str, tool_input: dict) -> str:
    """One-line summary of a tool call for display (don't dump huge content)."""
    if name == "write_file":
        return f"write_file: {tool_input['path']} ({len(tool_input.get('content', ''))} bytes)"
    if name == "read_file":
        return f"read_file: {tool_input['path']}"
    if name == "list_dir":
        return f"list_dir: {tool_input.get('path', '.')}"
    if name == "run_bash":
        return f"run_bash: {tool_input['command']}"
    return f"{name}: {tool_input}"


def run_turn(messages: list) -> None:
    """Drive the agent until it stops asking for tools (mutates `messages`)."""
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,  # Stage 3 switches to streaming, letting us raise
            system=SYSTEM,    # this safely for large file writes.
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text":
                print(f"\n🤖 {block.text}")

        if response.stop_reason != "tool_use":
            return

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n🔧 {format_tool_call(block.name, block.input)}")
                try:
                    output = DISPATCH[block.name](**block.input)
                except Exception as e:
                    # Seatbelt only. Stage 3 makes this a proper is_error result
                    # so the model clearly sees the failure and can recover.
                    output = f"Error: {e}"
                preview = output.strip()
                print(f"   ↳ {preview[:500]}{' …' if len(preview) > 500 else ''}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": tool_results})


# --- 4. REPL ---------------------------------------------------------------

def main() -> None:
    print("Stage 2 coding agent. Type a request, or 'quit' to exit.")
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
