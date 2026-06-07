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
| 3 | `stage3_agent.py` | Streaming, error handling, permission gating, iteration cap |
| 4a | `stage4a_agent.py` | Subagents — the loop re-run with its own context, returning a summary |
| 4b | `stage4b_agent.py` | Skills — progressive disclosure: catalog in the prompt, bodies loaded on demand (`skills/`) |
| 4c | `stage4c_agent.py` | Memory — a file read at each turn + a `remember` tool; persists across restarts |
| 4d | `stage4d_agent.py` | Compaction — summarize old turns when history grows, with a pair-safe cut |
| 4e | _next_ | Heartbeat / autonomy |
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

> ⚠️ Stages 1–2 run bash/writes with no confirmation. Stage 3 adds permission
> gating (y/N before writes and bash). Until then, run somewhere safe.
