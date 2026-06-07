"""
Stage 4e: heartbeat (autonomy).

Every stage so far is reactive — it waits for you to type. A heartbeat flips
that: the loop runs on a SCHEDULE against a STANDING GOAL, with no human in the
loop. The agent wakes on a tick, observes the world via tools, acts, records
what it did, and sleeps until the next tick.

The new pieces are small but distinct:
  - a scheduler (the for/sleep loop in heartbeat()) instead of the REPL,
  - a standing goal instead of a typed message each tick,
  - MEMORY AS THE BRIDGE BETWEEN TICKS: each tick starts with a fresh context,
    so the only way tick N+1 knows what tick N did is the memory file (loaded
    into the system prompt via build_system). This is exactly why Stage 4c
    exists — autonomy needs durable state.
  - stop conditions: a finish() tool the agent calls when the goal is met, plus
    a hard MAX_TICKS cap so it can't run forever.

Safety note (important for autonomy): there is no human to answer a permission
prompt, so an unattended agent must run with a SAFE, constrained toolset. The
heartbeat here gets read-only tools + remember + finish — no write_file, no
run_bash. The full mutating toolset from earlier stages still exists in this
file; the heartbeat just deliberately doesn't hand it over.

Built on 4d — skills, subagents, memory, compaction all still present.

Run:
    python stage4e_agent.py
It ticks a few times, inventorying *.py files and using memory to detect changes
across ticks, then finishes (or stops at MAX_TICKS).
"""

import glob
import os
import subprocess
import time

import anthropic

MODEL = "claude-opus-4-8"
MAX_ITERATIONS = 25
REQUIRES_APPROVAL = {"write_file", "run_bash"}
SKILLS_DIR = "skills"
MEMORY_FILE = "agent_memory.md"
COMPACT_THRESHOLD_TOKENS = 1500
KEEP_RECENT_MESSAGES = 2

# Heartbeat config (small for the demo).
INTERVAL_SECONDS = 3
MAX_TICKS = 3
GOAL = (
    "Maintain an inventory of the Python (*.py) files in the current directory.\n"
    "Each tick:\n"
    "  1. List the current *.py files.\n"
    "  2. Compare them against the inventory you recorded in memory on previous "
    "ticks (if any).\n"
    "  3. State briefly whether anything was added or removed since last tick.\n"
    "  4. Record the current inventory and this observation with remember().\n"
    "If you observe no changes on two consecutive ticks, the inventory is stable "
    "— call finish() with a one-line summary."
)

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


# --- 3. Heartbeat stop signal ----------------------------------------------

_hb_state = {"finished": False, "reason": ""}


def finish(summary: str) -> str:
    """The agent calls this to end the heartbeat early (goal satisfied)."""
    _hb_state["finished"] = True
    _hb_state["reason"] = summary
    return "Acknowledged — the heartbeat will stop after this tick."


# --- 4. Skills (from 4b) ---------------------------------------------------

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


# --- 5. System prompt ------------------------------------------------------

BASE_SYSTEM = (
    "You are an autonomous agent. Be concise. When you learn a durable fact, "
    "call remember(note). When a task matches a skill, call read_skill(name) "
    "first."
)

SUBAGENT_SYSTEM = (
    "You are a read-only exploration subagent. Investigate with read_file, "
    "list_dir, and search_files, then return a concise summary."
)


def build_system() -> str:
    memory = load_memory() or "(empty — nothing remembered yet)"
    return (
        BASE_SYSTEM
        + "\n\nAvailable skills (load full text with read_skill):\n"
        + skills_catalog(SKILLS)
        + f"\n\n## Your persistent memory (reloaded from {MEMORY_FILE} each tick)\n"
        + memory
    )


# --- 6. Compaction (from 4d) -----------------------------------------------

def render_transcript(messages: list, cap=None) -> str:
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
            if isinstance(b, dict):
                bt = b.get("type")
                if bt == "tool_result":
                    lines.append(f"{role}: [tool_result {clip(b.get('content', ''))}]")
                elif bt == "text":
                    lines.append(f"{role}: {clip(b.get('text', ''))}")
                else:
                    lines.append(f"{role}: [{bt}]")
            else:
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
    return len(render_transcript(messages)) // 4


def is_safe_tail_start(m: dict) -> bool:
    if m["role"] == "assistant":
        return True
    content = m["content"]
    if isinstance(content, str):
        return True
    return not any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def summarize(head: list) -> str:
    resp = client.messages.create(
        model=MODEL, max_tokens=1024,
        system="You compress conversation history. Produce a terse, factual "
               "summary preserving decisions, file paths, findings, and state "
               "needed to continue. Bullet points; omit pleasantries.",
        messages=[{"role": "user",
                   "content": "Summarize so it can be dropped in as context:\n\n"
                   + render_transcript(head, cap=3000)}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def maybe_compact(messages: list, indent: str = "") -> None:
    if estimate_tokens(messages) < COMPACT_THRESHOLD_TOKENS:
        return
    if len(messages) <= KEEP_RECENT_MESSAGES + 1:
        return
    cut = len(messages) - KEEP_RECENT_MESSAGES
    while cut < len(messages) and not is_safe_tail_start(messages[cut]):
        cut += 1
    if cut <= 0 or cut >= len(messages):
        return
    head, tail = messages[:cut], messages[cut:]
    print(f"\n{indent}🗜  compacting {len(head)} old msgs (~{estimate_tokens(messages)} tok)…")
    summary = summarize(head)
    messages[:] = [{"role": "user",
                    "content": "[Summary of earlier conversation]\n" + summary}] + tail
    print(f"{indent}   ↳ now {len(messages)} msgs (~{estimate_tokens(messages)} tok)")


# --- 7. Helpers ------------------------------------------------------------

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
    if name == "finish":
        return f"finish: {tool_input.get('summary', '')}"
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


# --- 8. The loop (unchanged) -----------------------------------------------

def agent_loop(messages: list, *, system: str, tools: list, dispatch: dict,
               gated: set, indent: str = "") -> str:
    final_text = ""
    for _ in range(MAX_ITERATIONS):
        maybe_compact(messages, indent)

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
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": "The user declined to run this tool.",
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


# --- 9. Schemas + dispatch -------------------------------------------------

SCHEMAS = {
    "read_file": {
        "name": "read_file", "description": "Read a UTF-8 text file.",
        "input_schema": {"type": "object",
                         "properties": {"path": {"type": "string", "description": "Path."}},
                         "required": ["path"]},
    },
    "write_file": {
        "name": "write_file",
        "description": "Create or overwrite a UTF-8 text file. Creates parent dirs.",
        "input_schema": {"type": "object",
                         "properties": {"path": {"type": "string", "description": "Path."},
                                        "content": {"type": "string", "description": "Contents."}},
                         "required": ["path", "content"]},
    },
    "list_dir": {
        "name": "list_dir", "description": "List the entries in a directory.",
        "input_schema": {"type": "object",
                         "properties": {"path": {"type": "string", "description": "Dir. Default '.'."}},
                         "required": []},
    },
    "run_bash": {
        "name": "run_bash", "description": "Run a bash command (stdout+stderr).",
        "input_schema": {"type": "object",
                         "properties": {"command": {"type": "string", "description": "Command."}},
                         "required": ["command"]},
    },
    "search_files": {
        "name": "search_files",
        "description": "Recursively search files for a pattern (grep -rn).",
        "input_schema": {"type": "object",
                         "properties": {"pattern": {"type": "string", "description": "Pattern."},
                                        "path": {"type": "string", "description": "Dir/file. Default '.'."}},
                         "required": ["pattern"]},
    },
    "read_skill": {
        "name": "read_skill",
        "description": "Load a named skill's full instructions before matching work.",
        "input_schema": {"type": "object",
                         "properties": {"name": {"type": "string", "description": "Skill name."}},
                         "required": ["name"]},
    },
    "remember": {
        "name": "remember", "description": "Save a short durable fact to memory.",
        "input_schema": {"type": "object",
                         "properties": {"note": {"type": "string", "description": "The fact."}},
                         "required": ["note"]},
    },
    "finish": {
        "name": "finish",
        "description": "Call when the standing goal is satisfied and no further "
                       "ticks are needed. Ends the heartbeat.",
        "input_schema": {"type": "object",
                         "properties": {"summary": {"type": "string", "description": "One-line wrap-up."}},
                         "required": ["summary"]},
    },
    "spawn_agent": {
        "name": "spawn_agent",
        "description": "Delegate a read-only investigation to a subagent; returns a summary.",
        "input_schema": {"type": "object",
                         "properties": {"task": {"type": "string", "description": "What to investigate."}},
                         "required": ["task"]},
    },
}

# Full interactive agent (not used by the heartbeat, kept for completeness).
PARENT_TOOLS = [SCHEMAS[n] for n in
                ("read_file", "write_file", "list_dir", "run_bash",
                 "read_skill", "remember", "spawn_agent")]
PARENT_DISPATCH = {
    "read_file": read_file, "write_file": write_file, "list_dir": list_dir,
    "run_bash": run_bash, "read_skill": read_skill, "remember": remember,
    "spawn_agent": spawn_agent,
}

SUBAGENT_TOOLS = [SCHEMAS[n] for n in ("read_file", "list_dir", "search_files")]
SUBAGENT_DISPATCH = {
    "read_file": read_file, "list_dir": list_dir, "search_files": search_files,
}

# Autonomous heartbeat: SAFE, read-only subset + remember + finish. No write_file
# or run_bash, because no human is present to approve a mutating action.
HEARTBEAT_TOOLS = [SCHEMAS[n] for n in
                   ("list_dir", "read_file", "search_files", "remember", "finish")]
HEARTBEAT_DISPATCH = {
    "list_dir": list_dir, "read_file": read_file, "search_files": search_files,
    "remember": remember, "finish": finish,
}


# --- 10. The heartbeat -----------------------------------------------------

def tick_prompt(tick: int, max_ticks: int, goal: str) -> str:
    return (
        f"Heartbeat tick {tick} of up to {max_ticks}. There is no human to talk "
        f"to — act autonomously and keep it brief.\n\nStanding goal:\n{goal}\n\n"
        "Observe with your tools; recall previous ticks from your memory above."
    )


def heartbeat(goal: str = GOAL, interval: int = INTERVAL_SECONDS,
              max_ticks: int = MAX_TICKS) -> None:
    print(f"💓 heartbeat: up to {max_ticks} ticks, {interval}s apart, "
          f"safe tools only ({', '.join(t['name'] for t in HEARTBEAT_TOOLS)}).")
    for tick in range(1, max_ticks + 1):
        print(f"\n===== TICK {tick}/{max_ticks} =====")
        messages = [{"role": "user", "content": tick_prompt(tick, max_ticks, goal)}]
        # Fresh context each tick — memory (in build_system) is the only bridge.
        agent_loop(messages, system=build_system(), tools=HEARTBEAT_TOOLS,
                   dispatch=HEARTBEAT_DISPATCH, gated=set())
        if _hb_state["finished"]:
            print(f"\n✅ agent called finish: {_hb_state['reason']}")
            return
        if tick < max_ticks:
            print(f"\n💤 sleeping {interval}s until the next tick…")
            time.sleep(interval)
    print("\n⏹  reached MAX_TICKS; heartbeat stopping.")


def main() -> None:
    heartbeat()


if __name__ == "__main__":
    main()
