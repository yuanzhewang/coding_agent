---
name: commit-message
description: How to write a commit message for this project
---

Write commit messages in this exact style:

- Subject line: a gitmoji emoji + a conventional-commit type, e.g.
  `✨ feat(agent): add skill loading`. Imperative mood, under ~60 chars,
  no trailing period.
- Then a blank line, then a body wrapped at ~72 columns explaining WHAT
  changed and WHY (not how).
- End the body with this trailer line, verbatim:

      Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>

Common types: feat, fix, docs, refactor, test, chore.
