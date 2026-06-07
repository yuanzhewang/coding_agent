"""
Stage 4a: subagents.

A subagent is not a new mechanism — it's the SAME loop, run again, with its own
fresh `messages` list, its own (usually smaller) system prompt, and its own
toolset. The crucial property: the child's work happens in a separate context,
and only its final summary returns to the parent. The hundred lines of files it
read never touch the parent's context window.

Why that matters:
  - Context hygiene. "Go figure out how auth works across these 30 files and
    tell me the gist" would bloat the main thread with 30 file dumps. A subagent
    reads them in its own context and hands back three sentences.
  - Focus. The child has a narrow task and a narrow toolset.
  - (Real systems also run several children in parallel — same idea.)

To make this obvious, Stage 3's `run_turn` is refactored into `agent_loop(...)`,
a plain function. The parent runs it; `spawn_agent` runs it again for the child.
Identical code path — that's the whole point.

Design choices (deliberate, and matching real systems like Claude Code):
  - The subagent gets READ-ONLY tools (read_file, list_dir, search_files), so it
    can explore safely and runs autonomously — no per-action approval prompts.
  - One level of delegation: the subagent has no spawn_agent tool, so it can't
    recurse. (Prevents runaway trees.)
  - The parent still gates its own mutating tools (write_file, run_bash).

Run:
    python stage4a_agent.py
Try:
    "spawn a subagent to read stage1_agent.py and stage2_agent.py and report
     the key difference"
Watch the nested (indented) subagent loop run, then the parent relay the summary.
"""

import os
import subprocess

import anthropic

MODEL = "claude-opus-4-8"
MAX_ITERATIONS = 25
REQUIRES_APPROVAL = {"write_file", "run_bash"}   # parent's mutating tools

client = anthropic.Anthropic()

SYSTEM = (
    "You are a coding agent working in the user's current directory. You can "
    "read, write, and list files, run bash, and delegate. For focused, "
    "read-heavy investigation (exploring code, 'where/how is X', summarizing "
    "many files), call spawn_agent — it runs a separate agent that investigates "
    "and returns a summary, keeping your context clean. Do the actual work "
    "(edits, running things) yourself. Be concise."
)

SUBAGENT_SYSTEM = (
    "You are a read-only exploration subagent. A parent agent has given you a "
    "focused task. Investigate with read_file, list_dir, and search_files, then "
    "return a concise, self-contained summary of your findings. You cannot "
    "modify anything. Your final message is the ONLY thing the parent receives, "
    "so make it complete on its own."
)


# --- 1. Tools --------------------------------------------------------------

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


def search_files(pattern: str, path: str = ".") -> str:
    """Recursive grep — read-only, safe for subagents (no shell injection)."""
    result = subprocess.run(
        ["grep", "-rn", "--", pattern, path],
        capture_output=True, text=True, timeout=30,
    )
    out = result.stdout.strip()
    if not out:
        return "(no matches)"
    return out[:3000] + (" …" if len(out) > 3000 else "")


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
    if name == "search_files":
        return f"search_files: {tool_input['pattern']!r} in {tool_input.get('path', '.')}"
    if name == "spawn_agent":
        t = tool_input.get("task", "")
        return f"spawn_agent: {t[:80]}{'…' if len(t) > 80 else ''}"
    return f"{name}: {tool_input}"


def confirm(name: str, tool_input: dict, indent: str = "") -> bool:
    print(f"\n{indent}⚠️  Agent wants to {format_tool_call(name, tool_input)}")
    if name == "write_file":
        content = tool_input.get("content", "")
        print(f"{indent}   ── content preview ──")
        for line in content[:300].splitlines():
            print(f"{indent}   | {line}")
        if len(content) > 300:
            print(f"{indent}   | …")
    try:
        answer = input(f"{indent}   Allow? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def execute_tool(dispatch: dict, name: str, tool_input: dict):
    """Run a tool from the given dispatch table. Returns (content, is_error)."""
    try:
        return dispatch[name](**tool_input), False
    except Exception as e:
        return f"Error: {e}", True


# --- 3. THE loop — one function, used by both parent and subagents ----------

def agent_loop(messages: list, *, system: str, tools: list, dispatch: dict,
               gated: set, indent: str = "") -> str:
    """Drive an agent to completion. Returns its final assistant text.

    `indent` only affects display, so nested subagents read clearly. Everything
    else is exactly the Stage 3 loop.
    """
    final_text = ""
    for _ in range(MAX_ITERATIONS):
        with client.messages.stream(
            model=MODEL,
            max_tokens=16000,
            system=system,
            tools=tools,
            messages=messages,
        ) as stream:
            printed = False
            for event in stream:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    if not printed:
                        print(f"\n{indent}🤖 ", end="", flush=True)
                        printed = True
                    print(event.delta.text, end="", flush=True)
            if printed:
                print()
            response = stream.get_final_message()

        messages.append({"role": "assistant", "content": response.content})

        text = "".join(b.text for b in response.content if b.type == "text")
        if text:
            final_text = text  # remember the latest text as the return value

        if response.stop_reason != "tool_use":
            return final_text

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name in gated and not confirm(block.name, block.input, indent):
                print(f"{indent}   ✗ denied")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "The user declined to run this tool. Consider "
                               "another approach or ask what they'd prefer.",
                })
                continue

            print(f"\n{indent}🔧 {format_tool_call(block.name, block.input)}")
            content, is_error = execute_tool(dispatch, block.name, block.input)
            preview = content.strip()
            print(f"{indent}   ↳ {preview[:400]}{' …' if len(preview) > 400 else ''}")

            result = {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            }
            if is_error:
                result["is_error"] = True
            tool_results.append(result)

        messages.append({"role": "user", "content": tool_results})

    print(f"{indent}⚠️  Hit the {MAX_ITERATIONS}-iteration cap; stopping.")
    return final_text


# --- 4. spawn_agent: a tool whose body is just agent_loop again -------------

def spawn_agent(task: str) -> str:
    """Run a fresh read-only subagent on `task`; return only its summary."""
    print(f"\n   🔱 spawning subagent: {task[:80]}{'…' if len(task) > 80 else ''}")
    sub_messages = [{"role": "user", "content": task}]
    summary = agent_loop(
        sub_messages,
        system=SUBAGENT_SYSTEM,
        tools=SUBAGENT_TOOLS,
        dispatch=SUBAGENT_DISPATCH,
        gated=set(),          # read-only tools -> runs autonomously
        indent="      ",      # nest the child's output under the parent's
    )
    print("   🔱 subagent done — handing summary back to parent")
    return summary or "(subagent returned no text)"


# --- 5. Schemas + dispatch tables ------------------------------------------

SCHEMAS = {
    "read_file": {
        "name": "read_file",
        "description": "Read a UTF-8 text file and return its contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the file."}},
            "required": ["path"],
        },
    },
    "write_file": {
        "name": "write_file",
        "description": "Create or overwrite a UTF-8 text file. Creates parent dirs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write."},
                "content": {"type": "string", "description": "Full file contents."},
            },
            "required": ["path", "content"],
        },
    },
    "list_dir": {
        "name": "list_dir",
        "description": "List the entries in a directory.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path. Defaults to '.'."}},
            "required": [],
        },
    },
    "run_bash": {
        "name": "run_bash",
        "description": "Run a bash command, returning combined stdout/stderr.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command."}},
            "required": ["command"],
        },
    },
    "search_files": {
        "name": "search_files",
        "description": "Recursively search files for a pattern (like grep -rn). "
                       "Returns matching lines as file:line:text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Pattern to search for."},
                "path": {"type": "string", "description": "Dir or file to search. Defaults to '.'."},
            },
            "required": ["pattern"],
        },
    },
    "spawn_agent": {
        "name": "spawn_agent",
        "description": (
            "Delegate a focused, read-only investigation to a fresh subagent "
            "with its own context. It can read files, list dirs, and search, "
            "then returns a concise summary. Use for exploring code or "
            "answering 'where/how is X' without cluttering your own context. "
            "The subagent cannot modify anything."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Self-contained description of what to "
                                   "investigate and what to report back.",
                }
            },
            "required": ["task"],
        },
    },
}

# Parent can do everything, including delegate. Subagent is read-only and
# cannot spawn (one level of delegation).
PARENT_TOOLS = [SCHEMAS[n] for n in ("read_file", "write_file", "list_dir", "run_bash", "spawn_agent")]
SUBAGENT_TOOLS = [SCHEMAS[n] for n in ("read_file", "list_dir", "search_files")]

PARENT_DISPATCH = {
    "read_file": read_file, "write_file": write_file, "list_dir": list_dir,
    "run_bash": run_bash, "spawn_agent": spawn_agent,
}
SUBAGENT_DISPATCH = {
    "read_file": read_file, "list_dir": list_dir, "search_files": search_files,
}


# --- 6. REPL ---------------------------------------------------------------

def main() -> None:
    print("Stage 4a agent (subagents). Type a request, or 'quit'.")
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
        agent_loop(
            messages,
            system=SYSTEM,
            tools=PARENT_TOOLS,
            dispatch=PARENT_DISPATCH,
            gated=REQUIRES_APPROVAL,
        )


if __name__ == "__main__":
    main()
