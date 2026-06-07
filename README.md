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
| 3 | _next_ | Streaming, error handling, permission gating, iteration cap |
| 4 | _planned_ | The fancy layer: subagents, skills, memory, compaction, heartbeat |
| 5 | _planned_ | Specialize: coding / research / assistant |

## Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
python stage1_agent.py   # or: python stage2_agent.py
```

Then try: `how many Python files are under the current directory?` and watch it
call a tool, read the output, and answer.

> ⚠️ The bash tool runs whatever the model asks, with no confirmation, until
> Stage 3 adds permission gating. Run it somewhere you don't mind it poking at.
