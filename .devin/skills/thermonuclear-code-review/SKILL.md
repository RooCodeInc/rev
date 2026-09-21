---
name: thermonuclear-code-review
description: Unusually strict code-quality review of a branch or PR (structure, abstractions, duplication, spaghetti, boundaries). Use when asked to review a PR, audit a diff, or run "thermonuclear" / "code judo" review on the Kev repo.
---

# Thermo-Nuclear Code Quality Review

Use this for a strict review focused on implementation quality, maintainability, abstraction quality and codebase
health. Above all, be **ambitious** about structure: look for "code judo" moves that preserve behaviour while making
the implementation dramatically simpler, smaller and more direct. Do not stop at local cleanup.

## How to run it on this repo

1. Get the diff: `git diff main...<branch>` (or `gh pr diff <n>`). Read every changed file in full, not just hunks.
2. Read the callers of anything the diff touches (`grep` the symbol across `kev/`, `scripts/`, `space/`, `tests/`, `modal_app.py`).
3. Check the repo's canonical homes before accepting a new helper (see "Canonical helpers" below).
4. Verify with the `kev-verify` skill (unit suites, weight-backed parity tests, and the worktree parity harness for
   numerically sensitive changes).
5. Write findings ordered by the priority list below. Prefer few high-conviction comments over many nits.

## Canonical helpers (reuse, do not re-derive)

| fact | home |
|---|---|
| load/resolve a checkpoint, read/write `head.pt` (`Meta`), warm-start LoRA+head, `LoadOptions` (+ `from_env` at CLI entry points only) | `kev/checkpoint.py` |
| option keys for a question (choice/noul/score) | `kev.api.question_keys` |
| does a record fit the training context (`MAX_STATE/MAX_BRANCH/MAX_PACKED`) | `kev.model.fits(rec, *tokenizers)`; manifests write `kev.suite.CONTEXT` |
| default device / sync / empty_cache / allocated_bytes | `kev/device.py` |
| read a manifest, sha256 a file, load a split, trainable/eval-only policy (`validate_training`), `semantic_hash`, `SYNTHETIC_SOURCES` | `kev/suite.py` |
| labelled request -> API request / internal record | `kev.data.api_request`, `kev.data.materialize` |
| selective-prediction metrics, temperature fit, paired bootstrap | `kev/metrics.py` |
| predictors (local checkpoint, remote System One endpoint, Jev) | `kev/predictors.py` |
| rows from predictions, `summarize`, `evaluate_records` | `kev/benchmark.py` |
| research gates and thresholds | `kev.experiment.GATES`, `gate_report` |

`tests/test_conventions.py` enforces the "no second copy" rules for several of these; add a row there when a new
helper becomes canonical.

## Baseline prompt

> Perform a deep code quality audit of the branch's changes. Rethink how to structure / implement the changes to
> meaningfully improve code quality without impacting behaviour. Improve abstractions and modularity, reduce spaghetti,
> improve succinctness and legibility. Be ambitious: if there is a clear path that involves restructuring part of the
> codebase, go for it. Be extremely thorough and rigorous. Measure twice, cut once.

## Non-negotiable standards

0. **Be ambitious about structural simplification.** Look for reframings where whole branches, helpers, modes or layers
   disappear. Prefer the solution that feels inevitable in hindsight. If you can delete complexity instead of moving it, push for that.
1. **No file goes from under 1k lines to over 1k lines without a very strong reason.** Decompose first.
2. **No spaghetti growth.** New ad-hoc conditionals, scattered special cases or one-off branches in unrelated flows are
   design problems, not nits. Push logic behind a dedicated abstraction or module.
3. **Clean the design, do not just accept working code.** If behaviour can stay the same and structure gets meaningfully
   cleaner, push for it. Prefer removing moving pieces over spreading the same complexity around.
4. **Direct, boring, maintainable over hacky or magical.** Flag thin wrappers, identity abstractions, pass-through helpers,
   generic mechanisms hiding simple data shapes, env-var side channels between library functions.
5. **Type and boundary cleanliness.** Question unnecessary optionality, tri-state flags, `None` modes, casts, silent
   fallbacks papering over unclear invariants. Prefer explicit typed models or shared contracts.
6. **Logic in the canonical layer; reuse existing helpers.** Flag feature logic leaking into shared paths, bespoke
   near-duplicates of canonical utilities, code in the wrong package/module.
7. **Unnecessary sequential orchestration and non-atomic updates** are smells when the cleaner structure is obvious.

## Questions to ask of every meaningful change

- Is there a code-judo move that makes this dramatically simpler?
- Can it be reframed so fewer concepts, branches or helper layers are needed?
- Does it improve or worsen the local architecture? Did a cohesive module become more coupled or stateful?
- Is this logic in the right file and layer? Is it a near-duplicate of a canonical helper?
- Did the diff add casts, optionality or ad-hoc object shapes that obscure the real invariant?
- Did the diff push a file past a healthy size, or leave an obvious decomposition undone?

## Output

Prioritise: (1) structural regressions, (2) missed dramatic simplifications, (3) spaghetti/branching growth,
(4) boundary/abstraction/type-contract problems, (5) file-size/decomposition, (6) modularity, (7) legibility.
Give file:line references. Say clearly whether the branch meets the approval bar:

- no clear structural regression
- no obvious missed opportunity for a dramatically simpler implementation
- no unjustified file-size explosion
- no spaghetti growth from special-case branching
- no hacky/magical abstraction, wrapper, cast or optionality churn
- no architecture-boundary leak or avoidable duplication of a canonical helper

Treat violations as presumptive blockers unless clearly justified. Be direct and demanding, not rude. Do not
soften major maintainability issues into mild suggestions.
