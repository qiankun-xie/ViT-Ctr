---
phase: quick
plan: 260408-mdn
type: execute
wave: 1
depends_on: []
files_modified:
  - .planning/phases/06-paper-and-supporting-information/06-CONTEXT.md
  - .planning/phases/06-paper-and-supporting-information/06-02-PLAN.md
  - .planning/phases/06-paper-and-supporting-information/06-03-PLAN.md
autonomous: true
requirements: []

must_haves:
  truths:
    - "All Phase 6 planning files reference 77 literature points, not 14"
    - "All Phase 6 planning files use new ML metrics: R²=0.968, RMSE=0.181, median_fold_error=1.10, 92.2% within 2x, 100% within 10x"
    - "All Phase 6 planning files use new Mayo metrics: R²=0.502, RMSE=0.715"
    - "n_ensemble=50 is reflected where bootstrap ensemble size is mentioned"
  artifacts:
    - path: ".planning/phases/06-paper-and-supporting-information/06-CONTEXT.md"
      provides: "Updated canonical refs with 77-point dataset"
    - path: ".planning/phases/06-paper-and-supporting-information/06-02-PLAN.md"
      provides: "Updated manuscript plan with correct validation metrics"
    - path: ".planning/phases/06-paper-and-supporting-information/06-03-PLAN.md"
      provides: "Updated SI plan with correct validation references"
  key_links:
    - from: "06-02-PLAN.md read_first block"
      to: "figures/validation/validation_summary.json"
      via: "inline metric values must match JSON"
      pattern: "R²=0.968"
---

<objective>
Update all Phase 6 planning files to replace stale 14-point validation numbers with the 77-point revalidation results.

Purpose: Phase 4 was revalidated with 77 literature points. Phase 6 plans still embed the old 14-point metrics inline, which will cause the manuscript executor to write incorrect numbers into the paper.
Output: Three updated planning files with consistent 77-point metrics throughout.
</objective>

<execution_context>
@C:/Users/xqk/.claude/get-shit-done/workflows/execute-plan.md
@C:/Users/xqk/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@figures/validation/validation_summary.json
</context>

<tasks>

<task type="auto">
  <name>Task 1: Update 06-CONTEXT.md — replace 14-point references</name>
  <files>.planning/phases/06-paper-and-supporting-information/06-CONTEXT.md</files>
  <action>
Make two targeted edits in 06-CONTEXT.md:

1. In `## Implementation Decisions` → D-07, change:
   - "ML vs Mayo验证图（14个文献点）" → "ML vs Mayo验证图（77个文献点）"

2. In `## Canonical References` → `data/literature/literature_ctr.csv` description, change:
   - "14个文献Ctr验证点" → "77个文献Ctr验证点"

3. In `## Canonical References` → `figures/validation/validation_results.csv` description, change:
   - "14点验证详细数据" → "77点验证详细数据"

No other changes. Do not alter decisions, narrative framing, or any other content.
  </action>
  <verify>
    <automated>grep -c "77" .planning/phases/06-paper-and-supporting-information/06-CONTEXT.md</automated>
  </verify>
  <done>06-CONTEXT.md contains "77" at least 3 times; "14个文献" no longer appears</done>
</task>

<task type="auto">
  <name>Task 2: Update 06-02-PLAN.md — replace all stale metrics</name>
  <files>.planning/phases/06-paper-and-supporting-information/06-02-PLAN.md</files>
  <action>
Make targeted replacements in 06-02-PLAN.md. New validated numbers from validation_summary.json (n_points=77, n_ensemble=50):

  ML:   R²=0.968, RMSE=0.181, median_fold_error=1.10, 92.2% within 2x, 100% within 10x
  Mayo: R²=0.502, RMSE=0.715

Specific replacements (exact strings):

In the `<read_first>` block (Task 1):
- "figures/validation/validation_summary.json (ML: R²=0.991, RMSE=0.126, median fold-error=1.17, 93% within 2×; Mayo: R²=0.825, RMSE=0.558)"
  → "figures/validation/validation_summary.json (ML: R²=0.968, RMSE=0.181, median fold-error=1.10, 92.2% within 2×, 100% within 10×; Mayo: R²=0.502, RMSE=0.715)"

- "data/literature/literature_ctr.csv (14 published Ctr values, 4 RAFT types, 3 methods, 8 references)"
  → "data/literature/literature_ctr.csv (77 published Ctr values, 4 RAFT types, multiple methods, expanded reference set)"

In the `<action>` block (Abstract section):
- "Key result: R²=0.991 on literature validation (14 points), 93% within 2-fold"
  → "Key result: R²=0.968 on literature validation (77 points), 92.2% within 2-fold"

In the `<action>` block (Methods §2.7):
- "2.7 Literature Validation: 14 published Ctr values, Mayo ODE-fitting baseline with fixed mean kinetic params"
  → "2.7 Literature Validation: 77 published Ctr values, Mayo ODE-fitting baseline with fixed mean kinetic params"

In the `<action>` block (Results §3.2):
- "3.2 Literature Validation: ML R²=0.991 vs Mayo R²=0.825; ML RMSE=0.126 vs Mayo RMSE=0.558; reference Figure 4"
  → "3.2 Literature Validation: ML R²=0.968 vs Mayo R²=0.502; ML RMSE=0.181 vs Mayo RMSE=0.715; reference Figure 4"

In the `<acceptance_criteria>` block:
- "paper/manuscript.tex contains `R.*0.991` or `R\$\^2\$ = 0.99` (ML validation R²)"
  → "paper/manuscript.tex contains `R.*0.968` or `R\$\^2\$ = 0.97` (ML validation R²)"

In the `<how-to-verify>` block (Task 2, checkpoint):
- "verify three-parameter claim, key metrics (R²=0.991), and web tool mention"
  → "verify three-parameter claim, key metrics (R²=0.968), and web tool mention"

- "Read Results — verify all numbers match validation_summary.json and calibration.json"
  → unchanged (already references the JSON file, which has correct numbers)

No other changes. Do not alter structure, narrative framing, LaTeX template choices, or any other content.
  </action>
  <verify>
    <automated>grep -c "0.968\|0.181\|0.715\|0.502\|77 published\|77 points" .planning/phases/06-paper-and-supporting-information/06-02-PLAN.md</automated>
  </verify>
  <done>06-02-PLAN.md contains new metrics (0.968, 0.181, 77 points); old metrics (0.991, 0.126, 14 points) no longer appear in metric contexts</done>
</task>

<task type="auto">
  <name>Task 3: Update 06-03-PLAN.md — remove stale 14-point reference if present</name>
  <files>.planning/phases/06-paper-and-supporting-information/06-03-PLAN.md</files>
  <action>
Scan 06-03-PLAN.md for any references to "14" in the context of validation points or literature data.

The SI plan does not embed inline metric numbers (it references calibration.json and bootstrap_summary.json directly), so changes here are minimal. Make only these replacements if found:

- Any occurrence of "14 published" or "14 literature" or "14个文献" → replace with "77 published" / "77 literature" / "77个文献" respectively
- Any occurrence of "14 points" in a validation context → "77 points"

If none of these patterns exist in the file, make no changes and note "no changes needed" in the summary.

Do not alter ODE derivation instructions, dataset construction details, bootstrap protocol, or any other content.
  </action>
  <verify>
    <automated>grep -n "14 published\|14 literature\|14个文献\|14 points" .planning/phases/06-paper-and-supporting-information/06-03-PLAN.md || echo "no stale references found"</automated>
  </verify>
  <done>06-03-PLAN.md contains no stale 14-point validation references</done>
</task>

</tasks>

<verification>
grep -rn "R²=0.991\|RMSE=0.126\|14个文献点\|14 published Ctr\|14 points.*valid\|93% within" .planning/phases/06-paper-and-supporting-information/
# Should return no matches
</verification>

<success_criteria>
All three Phase 6 planning files consistently reference 77 literature points and the new ML metrics (R²=0.968, RMSE=0.181). No stale 14-point or old-metric strings remain in validation contexts.
</success_criteria>

<output>
After completion, create `.planning/quick/260408-mdn-phase6-14-77/260408-mdn-SUMMARY.md`
</output>
