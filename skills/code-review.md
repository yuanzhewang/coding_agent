---
name: code-review
description: Checklist for reviewing a code change before committing
---

When reviewing a change, check in this order and report findings as a short
list — most important first:

1. Correctness — does it do what it claims? Off-by-one, wrong condition,
   unhandled error, broken edge case?
2. Security — injection, path traversal, secret leakage, unsafe shell?
3. Simplicity — duplicated logic, dead code, a simpler equivalent?
4. Tests — is the change covered? If not, note what to add.

For each finding give: `file:line`, the issue, and a concrete fix. If nothing
is wrong, say so plainly — do not invent nits.
