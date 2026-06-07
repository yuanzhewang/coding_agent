"""
Stage 4d: compaction.

Every prior stage has a silent leak: `messages` only grows. A long agentic
session — especially one with many large tool results — eventually approaches
the context window. Compaction keeps it bounded: when the history gets big,
summarize the OLD turns into one note and keep the RECENT turns verbatim, so the
conversation can run indefinitely.

The whole loop lives in maybe_compact(), called at the top of every iteration:

  1. Estimate the history size (rough chars/4; use count_tokens for precision).
  2. If over threshold, pick a cut point near the end (keep the last K msgs).
  3. CRUCIAL: never start the kept tail on an orphaned tool_result — a
     tool_result must follow its matching tool_use, so we advance the cut to a
     safe boundary (an assistant msg, or a real user msg). Splitting a pair here
     would make the next request 400.
  4. Summarize the head with a separate model call, splice it back as one user
     message, and replace the history in place (messages[:] = ...).

This is the manual version. The Claude API also offers server-side compaction
(beta `compact-2026-01-12`) that does this for you — but building it by hand is
how you learn the pair-safety subtlety it's handling under the hood.

Built on 4c — skills, subagents, memory all stay.

Demo thresholds below are tiny so compaction visibly fires after a couple of
file reads; set them near your model's real context limit in production.
"""

import glob
import os
import subprocess

import anthropic

MODEL = "claude-opus-4-8"
MAX_ITERATIONS = 25
REQUIRES_APPROVAL = {"write_file", "run_bash"}
SKILLS_DIR = "skills"
MEMORY_FILE = "agent_memory.md"

# Compaction tuning (deliberately small for the demo).
COMPACT_THRESHOLD_TOKENS = 1500   # in practice: near the context-window limit
KEEP_RECENT_MESSAGES = 2          # in practice: larger (keep more verbatim)

client = anthropic.Anthropic()


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
    result = subprocess.run(
        ["grep", "-rn", "--", pattern, path],
        capture_output=True, text=True, timeout=30,
    )
    out = result.stdout.strip()
    if not out:
        return "(no matches)"
    return out[:3000] + (" …" if len(out) > 3000 else "")


# --- 2. Memory (from 4c) ---------------------------------------------------

def load_memory() -> str:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def remember(note: str) -> str:
    with open(MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"- {note}\n")
    return f"Saved to memory: {note}"


# --- 3. Skills (from 4b) ---------------------------------------------------

def _parse_frontmatter(text: str):
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            header = text[3:end].strip()
            body = text[end + 4:].lstrip("\n")
            meta = {}
            for line in header.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            return meta, body
    return {}, text


def load_skills(skills_dir: str = SKILLS_DIR) -> dict:
    skills = {}
    for path in sorted(glob.glob(os.path.join(skills_dir, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            meta, body = _parse_frontmatter(f.read())
        name = meta.get("name") or os.path.splitext(os.path.basename(path))[0]
        skills[name] = {"description": meta.get("description", ""), "body": body}
    return skills


SKILLS = load_skills()


def skills_catalog(skills: dict) -> str:
    if not skills:
        return "(none)"
    return "\n".join(f"- {n}: {s['description']}" for n, s in skills.items())


def read_skill(name: str) -> str:
    skill = SKILLS.get(name)
    if not skill:
        return f"No such skill '{name}'. Available: {', '.join(SKILLS) or '(none)'}"
    return skill["body"]


# --- 4. System prompt ------------------------------------------------------

BASE_SYSTEM = (
    "You are a coding agent working in the user's current directory. You can "
    "read, write, and list files, run bash, delegate to subagents, load skills, "
    "and keep persistent memory. For focused read-heavy investigation, use "
    "spawn_agent. When a task matches a skill, call read_skill(name) first. When "
    "you learn a durable fact, call remember(note). Be concise."
)

SUBAGENT_SYSTEM = (
    "You are a read-only exploration subagent. A parent agent has given you a "
    "focused task. Investigate with read_file, list_dir, and search_files, then "
    "return a concise, self-contained summary. You cannot modify anything."
)


def build_system() -> str:
    memory = load_memory() or "(empty — nothing remembered yet)"
    return (
        BASE_SYSTEM
        + "\n\nAvailable skills (load full text with read_skill):\n"
        + skills_catalog(SKILLS)
        + f"\n\n## Your persistent memory (reloaded from {MEMORY_FILE} each turn)\n"
        + memory
    )


# --- 5. Compaction ---------------------------------------------------------

def render_transcript(messages: list, cap=None) -> str:
    """Flatten messages (SDK blocks or dicts) to plain text. `cap` clips blocks."""
    def clip(s):
        s = str(s)
        return s if cap is None or len(s) <= cap else s[:cap] + "…"

    lines = []
    for m in messages:
        role, content = m["role"], m["content"]
        if isinstance(content, str):
            lines.append(f"{role}: {clip(content)}")
            continue
        for b in content:
            if isinstance(b, dict):          # user-built blocks (tool_result, text)
                bt = b.get("type")
                if bt == "tool_result":
                    lines.append(f"{role}: [tool_result {clip(b.get('content', ''))}]")
                elif bt == "text":
                    lines.append(f"{role}: {clip(b.get('text', ''))}")
                else:
                    lines.append(f"{role}: [{bt}]")
            else:                            # SDK response blocks (objects)
                bt = getattr(b, "type", None)
                if bt == "text":
                    lines.append(f"{role}: {clip(b.text)}")
                elif bt == "tool_use":
                    lines.append(f"{role}: [tool_use {b.name} {clip(b.input)}]")
                elif bt == "thinking":
                    pass
                else:
                    lines.append(f"{role}: [{bt}]")
    return "\n".join(lines)


def estimate_tokens(messages: list) -> int:
    return len(render_transcript(messages)) // 4   # rough; count_tokens for exact


def is_safe_tail_start(m: dict) -> bool:
    """A kept tail must NOT begin with an orphaned tool_result."""
    if m["role"] == "assistant":
        return True
    content = m["content"]
    if isinstance(content, str):
        return True
    return not any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def summarize(head: list) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="You compress conversation history. Produce a terse, factual "
               "summary preserving decisions, file paths, findings, and any "
               "state needed to continue. Use bullet points; omit pleasantries.",
        messages=[{
            "role": "user",
            "content": "Summarize this earlier conversation so it can be dropped "
                       "in as context:\n\n" + render_transcript(head, cap=3000),
        }],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def maybe_compact(messages: list, indent: str = "") -> None:
    """If history is too big, summarize the head in place (pair-safe cut)."""
    if estimate_tokens(messages) < COMPACT_THRESHOLD_TOKENS:
        return
    if len(messages) <= KEEP_RECENT_MESSAGES + 1:
        return

    cut = len(messages) - KEEP_RECENT_MESSAGES
    while cut < len(messages) and not is_safe_tail_start(messages[cut]):
        cut += 1                     # don't split a tool_use/tool_result pair
    if cut <= 0 or cut >= len(messages):
        return

    head, tail = messages[:cut], messages[cut:]
    before = estimate_tokens(messages)
    print(f"\n{indent}🗜  compacting {len(head)} old msgs (~{before} tok est) into a summary…")
    summary = summarize(head)
    messages[:] = [{
        "role": "user",
        "content": "[Summary of earlier conversation]\n" + summary,
    }] + tail
    print(f"{indent}   ↳ history now {len(messages)} msgs (~{estimate_tokens(messages)} tok est)")


# --- 6. Helpers ------------------------------------------------------------

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
    if name == "read_skill":
        return f"read_skill: {tool_input['name']}"
    if name == "remember":
        return f"remember: {tool_input['note']}"
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
    try:
        return dispatch[name](**tool_input), False
    except Exception as e:
        return f"Error: {e}", True


# --- 7. The loop (now compacts at the top of each iteration) ----------------

def agent_loop(messages: list, *, system: str, tools: list, dispatch: dict,
               gated: set, indent: str = "") -> str:
    final_text = ""
    for _ in range(MAX_ITERATIONS):
        maybe_compact(messages, indent)   # keep the history bounded

        with client.messages.stream(
            model=MODEL, max_tokens=16000, system=system, tools=tools,
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
            final_text = text

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
            result = {"type": "tool_result", "tool_use_id": block.id, "content": content}
            if is_error:
                result["is_error"] = True
            tool_results.append(result)

        messages.append({"role": "user", "content": tool_results})

    print(f"{indent}⚠️  Hit the {MAX_ITERATIONS}-iteration cap; stopping.")
    return final_text


def spawn_agent(task: str) -> str:
    print(f"\n   🔱 spawning subagent: {task[:80]}{'…' if len(task) > 80 else ''}")
    summary = agent_loop(
        [{"role": "user", "content": task}],
        system=SUBAGENT_SYSTEM, tools=SUBAGENT_TOOLS, dispatch=SUBAGENT_DISPATCH,
        gated=set(), indent="      ",
    )
    print("   🔱 subagent done — handing summary back to parent")
    return summary or "(subagent returned no text)"


# --- 8. Schemas + dispatch -------------------------------------------------

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
        "description": "Recursively search files for a pattern (like grep -rn).",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Pattern to search for."},
                "path": {"type": "string", "description": "Dir or file. Defaults to '.'."},
            },
            "required": ["pattern"],
        },
    },
    "read_skill": {
        "name": "read_skill",
        "description": "Load the full instructions for a named skill before "
                       "doing matching work.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "The skill name."}},
            "required": ["name"],
        },
    },
    "remember": {
        "name": "remember",
        "description": "Save a short, durable fact to persistent memory.",
        "input_schema": {
            "type": "object",
            "properties": {"note": {"type": "string", "description": "The fact, concise."}},
            "required": ["note"],
        },
    },
    "spawn_agent": {
        "name": "spawn_agent",
        "description": "Delegate a focused, read-only investigation to a fresh "
                       "subagent with its own context; it returns a summary.",
        "input_schema": {
            "type": "object",
            "properties": {"task": {"type": "string", "description": "What to investigate and report."}},
            "required": ["task"],
        },
    },
}

PARENT_TOOLS = [SCHEMAS[n] for n in
                ("read_file", "write_file", "list_dir", "run_bash",
                 "read_skill", "remember", "spawn_agent")]
SUBAGENT_TOOLS = [SCHEMAS[n] for n in ("read_file", "list_dir", "search_files")]

PARENT_DISPATCH = {
    "read_file": read_file, "write_file": write_file, "list_dir": list_dir,
    "run_bash": run_bash, "read_skill": read_skill, "remember": remember,
    "spawn_agent": spawn_agent,
}
SUBAGENT_DISPATCH = {
    "read_file": read_file, "list_dir": list_dir, "search_files": search_files,
}


# --- 9. REPL ---------------------------------------------------------------

def main() -> None:
    print(f"Stage 4d agent (compaction). {len(SKILLS)} skill(s); "
          f"compacts above ~{COMPACT_THRESHOLD_TOKENS} tok. Type a request, or 'quit'.")
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
        agent_loop(messages, system=build_system(), tools=PARENT_TOOLS,
                   dispatch=PARENT_DISPATCH, gated=REQUIRES_APPROVAL)


if __name__ == "__main__":
    main()
