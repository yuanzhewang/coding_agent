"""
Stage 4b: skills (progressive disclosure).

A "skill" is a task-specific playbook stored as a file in skills/. The trick is
how it reaches the model:

  - At startup we scan skills/ and inject only each skill's one-line DESCRIPTION
    into the system prompt (a small "catalog").
  - The full instructions stay on disk. The agent pulls them on demand with a
    read_skill(name) tool, only when a task matches.

That's progressive disclosure: you can give an agent dozens of playbooks without
paying for all of them in every request's context. Only the catalog (cheap) is
always present; the bodies (expensive) load when needed. It's exactly how the
skills system that powers this very session works.

This builds on Stage 4a — subagents are still here. New this stage:
skills/, load_skills(), the read_skill tool, and a system prompt that ends with
the auto-generated catalog.

Skill file format (skills/<name>.md):
    ---
    name: commit-message
    description: How to write a commit message for this project
    ---
    <full instructions...>

Run (from the repo dir, so skills/ is found):
    python stage4b_agent.py
Try:
    "using the commit-message skill, write a commit message for adding skills"
Watch: read_skill(commit-message) loads, then the output follows its rules.
"""

import glob
import os
import subprocess

import anthropic

MODEL = "claude-opus-4-8"
MAX_ITERATIONS = 25
REQUIRES_APPROVAL = {"write_file", "run_bash"}
SKILLS_DIR = "skills"

client = anthropic.Anthropic()


# --- 1. Tools (file/bash/search as before) ---------------------------------

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


# --- 2. Skills: load the catalog at startup, bodies on demand --------------

def _parse_frontmatter(text: str):
    """Return (meta_dict, body). Supports a simple `---\\nkey: val\\n---` header."""
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
    """name -> {description, body}. The bodies are NOT put in the prompt."""
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
    """Return a skill's full instructions — the on-demand half of disclosure."""
    skill = SKILLS.get(name)
    if not skill:
        return f"No such skill '{name}'. Available: {', '.join(SKILLS) or '(none)'}"
    return skill["body"]


# --- 3. System prompts (parent prompt ends with the auto-built catalog) -----

SYSTEM = (
    "You are a coding agent working in the user's current directory. You can "
    "read, write, and list files, run bash, delegate to subagents, and load "
    "skills. For focused read-heavy investigation, use spawn_agent. When a task "
    "matches a skill below, call read_skill(name) to load its full instructions "
    "BEFORE doing the work, then follow them. Be concise.\n\n"
    "Available skills (descriptions only — load full text with read_skill):\n"
    + skills_catalog(SKILLS)
)

SUBAGENT_SYSTEM = (
    "You are a read-only exploration subagent. A parent agent has given you a "
    "focused task. Investigate with read_file, list_dir, and search_files, then "
    "return a concise, self-contained summary. You cannot modify anything. Your "
    "final message is the ONLY thing the parent receives."
)


# --- 4. Helpers ------------------------------------------------------------

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


# --- 5. The loop (unchanged from 4a) ---------------------------------------

def agent_loop(messages: list, *, system: str, tools: list, dispatch: dict,
               gated: set, indent: str = "") -> str:
    final_text = ""
    for _ in range(MAX_ITERATIONS):
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


# --- 6. Schemas + dispatch -------------------------------------------------

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
        "description": "Load the full instructions for a named skill (a task-"
                       "specific playbook). Call this when a task matches one of "
                       "the available skills, before doing the work.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "The skill name."}},
            "required": ["name"],
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
                ("read_file", "write_file", "list_dir", "run_bash", "read_skill", "spawn_agent")]
SUBAGENT_TOOLS = [SCHEMAS[n] for n in ("read_file", "list_dir", "search_files")]

PARENT_DISPATCH = {
    "read_file": read_file, "write_file": write_file, "list_dir": list_dir,
    "run_bash": run_bash, "read_skill": read_skill, "spawn_agent": spawn_agent,
}
SUBAGENT_DISPATCH = {
    "read_file": read_file, "list_dir": list_dir, "search_files": search_files,
}


# --- 7. REPL ---------------------------------------------------------------

def main() -> None:
    print(f"Stage 4b agent (skills). Loaded {len(SKILLS)} skill(s): "
          f"{', '.join(SKILLS) or '(none)'}. Type a request, or 'quit'.")
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
        agent_loop(messages, system=SYSTEM, tools=PARENT_TOOLS,
                   dispatch=PARENT_DISPATCH, gated=REQUIRES_APPROVAL)


if __name__ == "__main__":
    main()
