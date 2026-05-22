# Composer 2.5 — Independent Analysis Request: chunked vs whole-sequence conformance bias

You are being asked for an **independent second opinion on one self-contained algorithmic question**.
Reason from the mechanism described here only. Do **not** assume a desired answer — there is a
decision riding on whether two measurements are comparable, and we specifically want an
independent derivation, not a confirmation.

## Minimal background (all you need)

We align a finite sequence of activity labels (a "trace", length N, N up to a few thousand)
against a Petri net using a least-cost alignment (Dijkstra-style). Each step of the alignment
is one of:

- **sync move**: trace label matches an enabled net transition (cost 0).
- **log move**: consume a trace label with no matching net move (cost > 0).
- **model move**: fire a net transition not in the trace (cost > 0); a **tau** (silent)
  transition is a model move with ~0 cost.

Two derived metrics:

- **prefix-exact**: the alignment of the trace has **0 log moves and 0 model moves**
  (the net can replay the exact activity order).
- **tau-completed**: prefix-exact **and**, from the marking where the trace alignment ended,
  the net's final marking is reachable using only silent (tau) transitions.

## The mechanism in question

The alignment routine `conformance_chunked(sequence, initial_marking, chunk_size, ...)` has two
operating modes we use:

- **`full`**: `chunk_size = N`. The entire sequence is solved as **one global least-cost
  alignment**. This yields the globally minimal number of log/model moves.
- **`run`**: `chunk_size = 11`. The sequence is processed in consecutive 11-element windows.
  Each window is solved as a **local least-cost alignment**; the resulting Petri-net marking is
  **carried forward** as the initial marking of the next window. There is a boundary-merge
  heuristic (`merge_mismatched_boundaries`) that re-solves across a window boundary when the
  stitched alignment looks inconsistent, but the optimization horizon within a window is still
  only that window — there is no global re-optimization over the whole sequence.

So `full` = one global optimum; `run` = a chain of locally-optimal 11-windows with carried
marking + a local boundary-repair heuristic.

## The question

We measured `prefix-exact` and `tau-completed` for one dataset under **`full`** and for another
dataset under **`run`**, and want to compare the two datasets.

1. Does `run` (local 11-window, carried marking, boundary-merge) systematically change the
   counted log/model moves relative to `full` (global optimum)? **In which direction**
   (more deviations / fewer / either)? Explain the mechanism precisely.
2. Under what trace/net structures is the discrepancy **largest** vs **negligible**?
3. Can `run` ever report a sequence as prefix-exact / tau-completed when `full` would **not**
   (i.e., a false *positive* fit), or only the reverse? Argue both directions.
4. Given your analysis, are a `full`-measured metric and a `run`-measured metric **comparable**
   across datasets? If not, what is the minimal empirical test to quantify the bias on a small
   number of cases, and what would convince you the bias is negligible?

Please give a reasoned, mechanism-level answer to 1–4, and a concrete minimal test design for 4.
Keep it self-contained; we will cross-check your reasoning against an independent empirical
spot-check.
