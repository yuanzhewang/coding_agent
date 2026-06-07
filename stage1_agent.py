"""
Stage 1: the bare agent loop.

The whole idea of an "agent" lives in one place: a loop where the MODEL'S
output chooses the next action, and the result of that action feeds back into
the model. This file is the smallest honest version of that:

    one tool (run_bash)  +  the SDK  +  the loop.

Everything fancy later (streaming, permissions, subagents, skills, memory)
is just decoration on this loop. Read it top to bottom once and the word
"agent" stops being mysterious.

Run:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python stage1_agent.py

Then try:  "how many Python files are under the current directory?"
and watch it call `ls`/`find`, read the output, and answer.

NOTE: run_bash executes whatever the model asks, with no confirmation.
That's fine for Stage 1 on your own machine — we add permission gating in
Stage 3. Don't point this at anything you care about yet.
"""

import subprocess

import anthropic

MODEL = "claude-opus-4-8"
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


# --- 1. The tool: a Python function + a JSON schema that describes it -------
#
# The model never runs code. It emits a *request* to call a tool, by name,
# with arguments matching this schema. WE run the function and hand back the
# result. The schema is how the model knows the tool exists and how to call it.

def run_bash(command: str) -> str:
    """Execute a shell command and return its combined stdout + stderr."""
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=60
    )
    return (result.stdout + result.stderr) or "(no output)"


TOOLS = [
    {
        "name": "run_bash",
        "description": (
            "Run a bash command on the user's machine and return its output. "
            "Use this to inspect files, search the filesystem, run programs, "
            "check versions, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                }
            },
            "required": ["command"],
        },
    }
]

# Map a tool name to the Python function that implements it.
DISPATCH = {"run_bash": run_bash}


# --- 2. The loop -----------------------------------------------------------
#
# This is the agent. Call the model; if it asked for tools, run them, append
# the results, and call again. Stop when it stops asking for tools.

def run_turn(messages: list) -> None:
    """Drive the agent until it produces a final answer (no more tool calls).

    `messages` is mutated in place so the conversation persists across turns.
    """
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=(
                "You are a helpful assistant with access to a bash tool. "
                "Use it whenever you need information from the machine, then "
                "answer the user's question directly."
            ),
            tools=TOOLS,
            messages=messages,
        )

        # RULE 1: append the model's FULL response — text AND tool_use blocks.
        # If you append only the text, the tool_use blocks are lost and the
        # next request is malformed.
        messages.append({"role": "assistant", "content": response.content})

        # Show whatever text the model produced this step.
        for block in response.content:
            if block.type == "text":
                print(f"\n🤖 {block.text}")

        # If the model didn't request a tool, this turn is finished.
        if response.stop_reason != "tool_use":
            return

        # Otherwise: run every tool the model asked for and collect results.
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n🔧 run_bash: {block.input['command']}")
                output = DISPATCH[block.name](**block.input)
                preview = output.strip()
                print(f"   ↳ {preview[:500]}{' …' if len(preview) > 500 else ''}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,   # RULE 2: must match the request
                    "content": output,
                })

        # Feed results back as a user turn, then loop. The model decides what
        # to do next — maybe another tool call, maybe the final answer.
        messages.append({"role": "user", "content": tool_results})


# --- 3. A tiny REPL so you can actually talk to it --------------------------

def main() -> None:
    print("Stage 1 agent (bash). Type a request, or 'quit' to exit.")
    messages = []  # the full conversation; grows with every turn
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
