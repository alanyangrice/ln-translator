You are distilling a single past translation critique into a crisp,
checkable regression test case.

Given the editor's terse comment, the offending English phrasing, and the
assistant's follow-up discussion, produce:
- category: one of [$categories]
- severity: "major" or "minor"
- resolution_guidance: 1-2 sentences stating concretely what a future
  translation must do for this issue to count as RESOLVED. Derive it from
  the assistant's follow-up reaction and any agreement the editor gave.
  Do NOT include the bad phrasing as something to imitate.
- preferred_fix: the editor's or assistant's agreed better phrasing if one
  was clearly stated, else "".

Return ONLY JSON.

# Editor comment
$comment

# Offending phrasing (earlier version)
$excerpt

# Assistant follow-up discussion
$followup
